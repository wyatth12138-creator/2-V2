from collections import defaultdict
import math
import random

import numpy as np
import torch

from training.strategy import set_stage, step_lr, unwrap_model
from utils.evaluation import get_eval
from utils.generate_prompts import get_click_prompt
from utils.task_registry import get_primary_task, get_target_tensor_name, get_task_spec


def active_loader_cycles(loaders):
    ordered = [(task_name, loader) for task_name, loader in loaders.items() if loader is not None and len(loader) > 0]
    cycles = [(task_name, repeat_loader(loader)) for task_name, loader in ordered]
    return ordered, cycles


def repeat_loader(loader):
    """Repeat a loader without itertools.cycle caching every batch in RAM."""
    while True:
        yield from loader


def forward_and_loss(
    datapack,
    model,
    criterion,
    opt,
    phase: str,
):
    imgs = datapack["image"].to(dtype=torch.float32, device=opt.device)
    masks = datapack["mask"].to(dtype=torch.float32, device=opt.device)
    low_masks = datapack["low_mask"].to(dtype=torch.float32, device=opt.device)
    labels = datapack["label"].to(dtype=torch.float32, device=opt.device)
    task_name = get_primary_task(datapack["task"])
    task_spec = get_task_spec(task_name)
    if phase == "segmentation" and not task_spec.is_segmentation:
        raise ValueError(f"Classification task {task_name!r} reached the segmentation phase")
    if phase == "classification" and not task_spec.is_classification:
        raise ValueError(f"Segmentation task {task_name!r} reached the classification phase")
    dataset_names = datapack.get("dataset", [])
    use_roi = torch.tensor([name in ["BUS-BRA", "TN3K"] for name in dataset_names], device=opt.device, dtype=torch.bool)
    prompts = {
        "modality_prompt": datapack["modality_prompt"].to(device=opt.device),
        "mode_prompt": datapack["mode_prompt"].to(device=opt.device),
        "type_prompt": datapack["type_prompt"].to(device=opt.device),
        "position_prompt": datapack["position_prompt"].to(device=opt.device),
        "task_prompt": datapack["task_prompt"].to(device=opt.device),
    }
    model_kwargs = {
        "pt": get_click_prompt(datapack, opt) if "pt" in datapack else None,
        "prompts": prompts,
        "use_roi": use_roi,
    }
    if task_spec.is_video:
        time_steps = imgs.shape[1] if imgs.ndim == 5 else imgs.shape[0]
        model_kwargs["mask_valid"] = torch.ones((imgs.shape[0], time_steps), dtype=torch.bool, device=opt.device)
        if "frame_valid" in datapack:
            model_kwargs["frame_valid"] = datapack["frame_valid"].to(device=opt.device)
            model_kwargs["frame_time"] = datapack["frame_time"].to(device=opt.device)
            # Only explicit clicks can reserve a conditioned memory slot.
            model_kwargs["mask_valid"] = torch.zeros_like(model_kwargs["frame_valid"], dtype=torch.bool)
            model_kwargs["use_roi"] = torch.ones(imgs.shape[0], device=opt.device, dtype=torch.bool)

    pred = model(imgs, **model_kwargs)
    pred["task"] = [task_name] * imgs.shape[0]
    pred["cls_target"] = labels
    pred["enable_prototype_loss"] = unwrap_model(model).enable_prototype_loss
    # Each stage optimizes its primary task loss while the other branch is frozen.
    pred["enable_auxiliary_losses"] = False
    pred["training_phase"] = phase

    target_name = get_target_tensor_name(task_name)
    target_tensor = low_masks if target_name == "low_mask" else masks
    loss = criterion(pred, target_tensor)
    return loss, pred, task_name


def train_round_robin(
    loaders,
    model,
    criterion,
    optimizer,
    args,
    opt,
    iter_num,
    max_iterations,
    phase,
):
    model.train()
    set_stage(model, phase)
    ordered_loaders, cycles = active_loader_cycles(loaders)
    if not cycles:
        return 0.0, iter_num, {}

    task_losses = defaultdict(list)
    steps_per_epoch = max(len(loader) for _, loader in ordered_loaders)

    for step_index in range(steps_per_epoch):
        for task_name, cycle in cycles:
            total_loss, _, _ = forward_and_loss(next(cycle), model, criterion, opt, phase=phase)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            iter_num = step_lr(args, optimizer, iter_num, max_iterations)
            task_losses[task_name].append(float(total_loss.item()))

    mean_loss = float(np.mean([loss for losses in task_losses.values() for loss in losses])) if task_losses else 0.0
    per_task_mean = {task_name: float(np.mean(losses)) for task_name, losses in task_losses.items()}
    return mean_loss, iter_num, per_task_mean
def evaluate_loader_bundle(loaders, model, criterion, opt, args, stage_name: str, domain_name: str):
    records = []
    for task_name, loader in loaders.items():
        if loader is None or len(loader) == 0:
            continue
        summary = get_eval(loader, model, criterion=criterion, opt=opt, args=args)
        summary["stage"] = stage_name
        summary["domain"] = domain_name
        summary["task"] = task_name
        records.append(summary)
    return records


def _validated_domain_loaders(loaders, phase=None):
    ordered = []
    for key, loader in loaders.items():
        if not isinstance(key, tuple) or len(key) != 2 or not all(isinstance(part, str) for part in key):
            raise ValueError("Domain loaders must use (task_name,dataset_name) tuple keys")
        task_name, _ = key
        spec = get_task_spec(task_name)
        if task_name != spec.name:
            raise ValueError("Domain loader task names must be canonical")
        if phase == "segmentation" and not spec.is_segmentation:
            raise ValueError(f"Classification domain {key} reached the segmentation stage")
        if phase == "classification" and not spec.is_classification:
            raise ValueError(f"Segmentation domain {key} reached the classification stage")
        if loader is not None and len(loader) > 0:
            ordered.append((key, loader))
    return ordered


def train_domain_updates(
    loaders, model, criterion, optimizer, args, opt, iter_num, max_iterations, phase,
    *, mode="weighted_mean", updates=None, domains_per_update=2, seed=None,
    domain_weight_power=0.0,
):
    """Opt-in balanced same-task source updates, retaining strict stage freezing.

    Each update selects ONE canonical task by global update-index rotation,
    then uniformly samples up to K distinct source datasets for that task.
    Its domain-mean losses are differentiated at the same parameter values;
    one weighted-mean or PCGrad merge precedes one optimizer step. Segmentation
    versus classification, or image versus video task losses, are never
    combined into one gradient-surgery objective. Per-source loaders repeat
    without caching batches. A singleton task reduces to normal training.

    Domain sampling and PCGrad order use a local seed plus global update
    index, so the schedule is reproducible across epoch boundaries. Domain
    gradients initially have weights N_domain**domain_weight_power, normalized
    within the selected subset. In weighted_mean mode, uniform selection of K
    out of D domains then uses the Horvitz-Thompson factor
    (D/K)*sum(selected weights)/sum(all task-domain weights). This yields an
    unbiased pre-optimizer estimator of the full task-weighted sum of the
    same-state candidate domain gradients. It is not an unbiased Adam step;
    each domain's minibatch gradient must itself estimate its domain loss
    gradient for an empirical-risk interpretation. The weighted loss log uses
    this same factor and is not renormalized back into a subset mean.

    Power 0 gives equal domain weights without requiring loader.dataset;
    power 1 targets sample-count weighting. All-domain, equal-weight and
    power-0 updates explicitly bypass scaling, preserving their prior values.
    PCGrad retains its prior subset-normalized projected update: projection is
    subset-dependent, so no full-domain unbiasedness is claimed for PCGrad.
    Task rotation supplies task balance. Use explicit updates and identical input/optimizer budgets for
    controlled comparisons; the default approximates one loader-pass budget.

    Returns (mean_loss,new_iter_num,per_domain_mean,diagnostics). The means use
    (task,dataset) keys; diagnostics have JSON-safe string keys and values.
    Dense prompt-bank tensors are protected from surgery to preserve their
    private rows. BatchNorm uses fixed running statistics in this opt-in path;
    affine parameters retain the selected stage's requires_grad setting.
    iter_num counts scheduled updates, including all-None-gradient attempts,
    while diagnostics separately records actual optimizer steps.
    """
    from training.domain_updates import merge_domain_gradients

    if phase not in ("segmentation", "classification"):
        raise ValueError("phase must be 'segmentation' or 'classification'")
    if mode not in ("weighted_mean", "pcgrad"):
        raise ValueError("mode must be 'weighted_mean' or 'pcgrad'")
    try:
        if isinstance(domain_weight_power, bool):
            raise ValueError("boolean power")
        domain_weight_power = float(domain_weight_power)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("domain_weight_power must be finite and in [0,1]") from exc
    if not math.isfinite(domain_weight_power) or not 0.0 <= domain_weight_power <= 1.0:
        raise ValueError("domain_weight_power must be finite and in [0,1]")
    if isinstance(domains_per_update, bool) or not isinstance(domains_per_update, int) or domains_per_update < 1:
        raise ValueError("domains_per_update must be a positive integer")
    if isinstance(iter_num, bool) or not isinstance(iter_num, int) or iter_num < 0:
        raise ValueError("iter_num must be a nonnegative integer")
    ordered = _validated_domain_loaders(loaders, phase)
    diagnostics = {
        "mode": mode, "phase": phase, "updates": 0, "optimizer_steps": 0,
        "microbatches": 0, "task_schedule": "round_robin",
        "domain_sampling": "uniform_without_replacement_within_task",
        "domains_per_update": domains_per_update,
        "domain_batches": {}, "task_updates": {}, "pairwise": [],
        "conflicting_pairs": 0, "comparable_pairs": 0, "projection_steps": 0,
        "conflict_fraction": 0.0, "mean_update_dot": {},
        "batchnorm_running_stats": "frozen",
        "domain_weight_power": domain_weight_power,
        "domain_weight_rule": "selected_domain_sample_count_power_normalized",
        "domain_sizes": {}, "mean_selected_weight": {},
        "sampling_correction": "horvitz_thompson" if mode == "weighted_mean" and domain_weight_power > 0 else "none",
        "sampling_correction_applied_updates": 0,
        "mean_effective_coefficient": {}, "domain_population": {},
    }
    if not ordered:
        return 0.0, iter_num, {}, diagnostics
    domain_sizes = {}
    if domain_weight_power > 0:
        for key, loader in ordered:
            try:
                size = len(loader.dataset)
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Domain {key[0]}/{key[1]} needs loader.dataset with a positive sample count "
                    "when domain_weight_power > 0"
                ) from exc
            if size <= 0:
                raise ValueError(f"Domain {key[0]}/{key[1]} has no samples for size-weighted training")
            domain_sizes[key] = size
        diagnostics["domain_sizes"] = {f"{key[0]}/{key[1]}": size for key, size in domain_sizes.items()}
    grouped = {}
    for key, loader in ordered:
        grouped.setdefault(key[0], []).append(key)
    tasks = list(grouped)
    source_mass = {key: domain_sizes[key] ** domain_weight_power if domain_weight_power > 0 else 1.0 for key, _ in ordered}
    population_mass = {task: math.fsum(source_mass[key] for key in grouped[task]) for task in tasks}
    uniform_mass = {task: len({source_mass[key] for key in grouped[task]}) == 1 for task in tasks}
    diagnostics["domain_population"] = {
        task: {
            "domains": len(grouped[task]),
            "selected_per_update": min(domains_per_update, len(grouped[task])),
            "inclusion_probability": min(domains_per_update, len(grouped[task])) / len(grouped[task]),
            "total_domain_weight": population_mass[task],
        }
        for task in tasks
    }
    if mode == "weighted_mean":
        diagnostics["domain_weight_rule"] = "full_task_sample_count_power_ht" if domain_weight_power > 0 else "equal_domain_mean"
    if updates is None:
        mean_domains = sum(min(domains_per_update, len(grouped[task])) for task in tasks) / len(tasks)
        updates = max(1, math.ceil(sum(len(loader) for _, loader in ordered) / mean_domains))
    if isinstance(updates, bool) or not isinstance(updates, int) or updates < 1:
        raise ValueError("updates must be a positive integer")
    if not isinstance(max_iterations, int) or max_iterations < 1 or iter_num + updates > max_iterations:
        raise ValueError("Requested domain updates exceed the positive max_iterations budget")

    model.train()
    set_stage(model, phase)
    module = unwrap_model(model)
    for child in module.modules():
        if isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
            child.eval()
    named_parameters = [(name, parameter) for name, parameter in module.named_parameters() if parameter.requires_grad]
    if not named_parameters:
        raise ValueError("The selected stage has no trainable parameters")
    parameters = tuple(parameter for _, parameter in named_parameters)
    protected = [name.startswith("prompt_bank.") for name, _ in named_parameters]
    cycles = {key: repeat_loader(loader) for key, loader in ordered}
    losses = defaultdict(list)
    objective_losses = []
    subset_ratio_losses = []
    sampling_scales = []
    merged_norms = []
    update_dots = defaultdict(list)
    selected_weights = defaultdict(list)
    effective_coefficients = defaultdict(list)
    pair_records = {}
    base_seed = int(getattr(args, "seed", 0) if seed is None else seed)
    diagnostics["seed"] = base_seed
    start_iter = iter_num

    for update_index in range(updates):
        global_index = start_iter + update_index
        task_name = tasks[global_index % len(tasks)]
        rng = random.Random(base_seed + 104729 * (global_index + 1))
        available = grouped[task_name]
        selected = rng.sample(available, min(domains_per_update, len(available)))
        optimizer.zero_grad(set_to_none=True)
        gradients, scalar_losses = [], []
        for key in selected:
            datapack = next(cycles[key])
            sample_domains = datapack.get("dataset", [])
            sample_domains = [sample_domains] if isinstance(sample_domains, str) else list(sample_domains)
            if not sample_domains or any(name != key[1] for name in sample_domains):
                raise ValueError(f"Domain loader {key} produced a different or missing source dataset")
            if phase == "classification":
                # This optional cross-dataset path has one binary diagnostic
                # head. BCE itself does not reject a source's third class
                # (e.g. BUSI normal=2), so validate before any model forward.
                label_message = (
                    f"Classification domain {key[0]}/{key[1]} requires finite binary labels {{0,1}}. "
                    "Explicitly define and verify the same negative=0/positive=1 clinical endpoint "
                    "across source datasets before training; remap extra source classes "
                    "(such as normal=2) in data preparation, rather than passing them to BCE."
                )
                try:
                    domain_labels = torch.as_tensor(datapack.get("label")).detach()
                except (TypeError, ValueError, RuntimeError) as exc:
                    raise ValueError(label_message) from exc
                if (
                    domain_labels.numel() == 0 or domain_labels.is_complex()
                    or not bool(torch.isfinite(domain_labels).all())
                    or not bool(((domain_labels == 0) | (domain_labels == 1)).all())
                ):
                    raise ValueError(label_message)
            loss, prediction, actual_task = forward_and_loss(datapack, model, criterion, opt, phase=phase)
            if actual_task != task_name:
                raise ValueError(f"Domain loader {key} produced task {actual_task}")
            if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
                raise ValueError("Each source domain must return a scalar mean loss")
            scalar_losses.append(loss.detach())
            gradients.append(
                torch.autograd.grad(loss, parameters, allow_unused=True)
                if loss.requires_grad else tuple(None for _ in parameters)
            )
            del prediction, loss

        values = torch.stack(scalar_losses).cpu().tolist()
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Source domain losses must be finite before an optimizer update")
        merged, summary = merge_domain_gradients(
            gradients,
            weights=[source_mass[key] for key in selected] if domain_weight_power > 0 else None,
            mode=mode, ordering=rng.sample(range(len(selected)), len(selected)), protected=protected,
        )
        sampling_scale = 1.0
        if (
            mode == "weighted_mean" and domain_weight_power > 0
            and len(selected) < len(available) and not uniform_mass[task_name]
        ):
            sampling_scale = (len(available) / len(selected)) * (
                math.fsum(source_mass[key] for key in selected) / population_mass[task_name]
            )
        # Do not multiply by one: full-domain/equal-weight paths preserve the
        # previous gradient tensors and floating-point values exactly.
        if sampling_scale != 1.0:
            merged = tuple(None if gradient is None else gradient * sampling_scale for gradient in merged)
            finite_by_device = {}
            for gradient in merged:
                if gradient is not None:
                    finite_by_device.setdefault(gradient.device, []).append(torch.isfinite(gradient).all())
            if not all(bool(torch.stack(checks).all()) for checks in finite_by_device.values()):
                raise ValueError("Sampling-corrected domain gradient overflowed before optimizer.step")
            # Raw pairwise gradient geometry is unchanged. Update diagnostics
            # refer to the actually assigned, corrected gradient direction.
            summary["update_dot"] = [value * sampling_scale for value in summary["update_dot"]]
            summary["update_norm"] *= sampling_scale
            diagnostics["sampling_correction_applied_updates"] += 1
        ratio_loss = math.fsum(weight * value for weight, value in zip(summary["weights"], values))
        subset_ratio_losses.append(ratio_loss)
        objective_losses.append(sampling_scale * ratio_loss)
        sampling_scales.append(sampling_scale)
        merged_norms.append(summary["update_norm"])
        if any(gradient is not None for gradient in merged):
            for parameter, gradient in zip(parameters, merged):
                parameter.grad = gradient
            optimizer.step()
            diagnostics["optimizer_steps"] += 1
        # Advance the planned schedule even for a constant, disconnected
        # objective: otherwise the next call repeats a task and RNG seed.
        iter_num = step_lr(args, optimizer, iter_num, max_iterations)
        for index, (key, value) in enumerate(zip(selected, values)):
            losses[key].append(float(value))
            name = f"{key[0]}/{key[1]}"
            diagnostics["domain_batches"][name] = diagnostics["domain_batches"].get(name, 0) + 1
            update_dots[name].append(summary["update_dot"][index])
            selected_weights[name].append(summary["weights"][index])
            effective_coefficients[name].append(sampling_scale * summary["weights"][index])
        diagnostics["updates"] += 1
        diagnostics["microbatches"] += len(selected)
        diagnostics["task_updates"][task_name] = diagnostics["task_updates"].get(task_name, 0) + 1
        for metric in ("conflicting_pairs", "comparable_pairs", "projection_steps"):
            diagnostics[metric] += summary[metric]
        for pair in summary["pairwise"]:
            left, right = selected[pair["domain_i"]], selected[pair["domain_j"]]
            # Keep canonical pair orientation when sampling changes order.
            names = [f"{left[0]}/{left[1]}", f"{right[0]}/{right[1]}"]
            reverse = names[0] > names[1]
            pair_key = tuple(sorted(names))
            record = pair_records.setdefault(pair_key, {"dots": [], "cosines": [], "norm_i": [], "norm_j": []})
            record["dots"].append(pair["dot"])
            if pair["cosine"] is not None:
                record["cosines"].append(pair["cosine"])
            record["norm_i"].append(pair["norm_j"] if reverse else pair["norm_i"])
            record["norm_j"].append(pair["norm_i"] if reverse else pair["norm_j"])

    diagnostics["conflict_fraction"] = diagnostics["conflicting_pairs"] / max(diagnostics["comparable_pairs"], 1)
    diagnostics["mean_update_dot"] = {name: float(np.mean(values)) for name, values in update_dots.items()}
    diagnostics["mean_selected_weight"] = {name: float(np.mean(values)) for name, values in selected_weights.items()}
    diagnostics["mean_effective_coefficient"] = {name: float(np.mean(values)) for name, values in effective_coefficients.items()}
    diagnostics["mean_sampling_scale"] = float(np.mean(sampling_scales))
    diagnostics["min_sampling_scale"] = min(sampling_scales)
    diagnostics["max_sampling_scale"] = max(sampling_scales)
    diagnostics["mean_merged_gradient_norm"] = float(np.mean(merged_norms))
    diagnostics["mean_subset_ratio_loss"] = float(np.mean(subset_ratio_losses))
    diagnostics["pairwise"] = [
        {
            "domain_i": names[0], "domain_j": names[1], "comparisons": len(record["dots"]),
            "mean_dot": float(np.mean(record["dots"])),
            "mean_cosine": float(np.mean(record["cosines"])) if record["cosines"] else None,
            "mean_norm_i": float(np.mean(record["norm_i"])), "mean_norm_j": float(np.mean(record["norm_j"])),
        }
        for names, record in sorted(pair_records.items())
    ]
    per_domain = {key: float(np.mean(values)) for key, values in losses.items()}
    diagnostics["mean_microbatch_loss"] = float(np.mean([value for values in losses.values() for value in values]))
    mean_loss = float(np.mean(objective_losses))
    diagnostics["mean_objective_loss"] = mean_loss
    print(
        f"Domain updates: stage={phase} mode={mode} weight_power={domain_weight_power:g} steps={diagnostics['optimizer_steps']} "
        f"microbatches={diagnostics['microbatches']} conflicts={diagnostics['conflicting_pairs']}/"
        f"{diagnostics['comparable_pairs']} projections={diagnostics['projection_steps']} loss={mean_loss:.5f}"
    )
    return mean_loss, iter_num, per_domain, diagnostics


def evaluate_domain_loader_bundle(loaders, model, criterion, opt, args, stage_name: str, domain_name: str):
    """Evaluate each source separately; never pool AUC or Dice across datasets."""
    records = []
    for (task_name, dataset_name), loader in _validated_domain_loaders(loaders):
        summary = dict(get_eval(loader, model, criterion=criterion, opt=opt, args=args))
        summary.update(
            stage=stage_name, task=task_name, domain=dataset_name,
            dataset=dataset_name, split_group=domain_name, num_samples=len(loader.dataset),
        )
        records.append(summary)
    return records
