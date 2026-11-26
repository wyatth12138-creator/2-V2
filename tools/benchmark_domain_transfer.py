"""Paired UIV-US domain training with optional source-validation checkpoints."""

import argparse
import hashlib
import json
import os
import random
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

FROZEN_SOURCE_SHA256 = "3aa579d19df791c65bf9664e8a90a6a02f41f67bf037f26f27733df80afed97b"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.uiv_us import build_uiv_us_vit_b
from train import build_parser
from training.data_splits import make_transforms
from training.domain_updates import merge_domain_gradients
from training.engine import forward_and_loss
from training.reporting import seed_everything
from training.run_selection import require_private_path
from training.strategy import build_optimizer, set_stage
from utils.config import get_config
from utils.dataset_ce_pat import SegmentationDataset2D, _auto_complete_path, _parse_seg_config_and_map_mask
from utils.loss_functions.sam_loss import get_criterion


TASK = "bmode_img_seg"


def checkpoint_path(args, label):
    """Validate the optional output before training; never create or overwrite it."""
    if getattr(args, "checkpoint_dir", None) is None:
        return None
    if args.eval_split != "val" or args.held_out or args.eval_limit is not None:
        raise ValueError("--checkpoint-dir requires full source-only VAL: eval_split=val, held_out=[], eval_limit=None")
    directory = require_private_path(args.checkpoint_dir, "Checkpoint directory")
    if directory.exists() and not directory.is_dir():
        raise ValueError("Checkpoint directory is not a directory")
    allowed = {"pooled", "balanced", "pcgrad"} | {f"single:{domain}" for domain in args.source}
    if label not in allowed:
        raise ValueError(f"Unknown checkpoint arm: {label}")
    destination = directory / f"seed_{args.seed}__{quote(label, safe='-_.')}.pth"
    if destination.parent.resolve() != directory or os.path.lexists(destination):
        raise FileExistsError(f"Refusing existing or escaping checkpoint path: {destination}")
    return destination


def save_checkpoint_atomic(model, destination):
    """Publish a complete CPU tensor dictionary with exclusive, atomic creation."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination):
        raise FileExistsError(f"Refusing to overwrite checkpoint: {destination}")
    state = model.state_dict()
    if not all(torch.is_tensor(value) for value in state.values()):
        raise ValueError("Expected a tensor-only model state dictionary")
    cpu_state = {name: value.detach().to(device="cpu", copy=True) for name, value in state.items()}
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            torch.save(cpu_state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        # Windows rename is atomic within this directory and refuses an existing
        # destination. POSIX rename can overwrite, so use exclusive link there.
        if os.name == "nt":
            os.rename(temporary, destination)
        else:
            os.link(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    digest = hashlib.sha256()
    with destination.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(destination), "sha256": digest.hexdigest()}


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sam", type=Path, required=True)
    parser.add_argument("--resnet", type=Path, required=True)
    parser.add_argument("--source", nargs="+", required=True)
    parser.add_argument("--held-out", nargs="*", default=[])
    parser.add_argument("--eval-split", choices=("val", "test"), default="val")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--eval-limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-lr", type=float, default=1e-4)
    parser.add_argument("--arms", nargs="+", choices=("single", "pooled", "balanced", "pcgrad"),
                        default=["single", "pooled", "balanced", "pcgrad"])
    parser.add_argument("--exclude", type=Path, help="Optional local JSON split-to-row exclusion map from a split audit.")
    parser.add_argument("--split-overrides", type=Path, help="Local JSON mapping domains to verified train/val/test rows.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path,
                        help="Optional CPU state dictionaries; only full source-only VAL, outside source directories.")
    args = parser.parse_args()
    names = args.source + args.held_out
    if len(names) != len(set(names)) or len(args.source) < 2:
        parser.error("provide at least two distinct source domains and disjoint held-out domains")
    if len(set(args.arms)) != len(args.arms):
        parser.error("provide each arm at most once")
    if args.steps < 1 or (args.eval_limit is not None and args.eval_limit < 1):
        parser.error("steps and eval-limit must be positive")
    if args.checkpoint_dir is not None:
        args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
        args.data_root = args.data_root.expanduser().resolve()
        try:
            for arm in args.arms:
                labels = [f"single:{domain}" for domain in args.source] if arm == "single" else [arm]
                for label in labels:
                    checkpoint_path(args, label)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
    return args


def eligible_dataset(opt, domain, split, transform, exclusions, overrides):
    dataset = SegmentationDataset2D(opt.data_path, domain, split, joint_transform=transform, img_size=transform.img_size)
    if domain in overrides:
        dataset.ids = overrides[domain][split]
    excluded = set(exclusions.get(domain, {}).get(split, []))
    if any(type(index) is not int or not 0 <= index < len(dataset) for index in excluded):
        raise ValueError(f"Exclusion map does not match {domain} {split}")
    indices = []
    for index, entry in enumerate(dataset.ids):
        if index in excluded:
            continue
        path = _auto_complete_path(dataset.root, entry, default_subdir="masks")
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Unreadable mask in {domain} {split}, row {index + 1}")
        mask = _parse_seg_config_and_map_mask(opt.data_path, domain, mask)
        if np.any(mask == 1):
            indices.append(index)
    if not indices:
        raise ValueError(f"No lesion-positive records in {domain} {split}")
    return Subset(dataset, indices), len(dataset) - len(indices)

def draw_training_rows(size, steps, seed):
    rng = np.random.default_rng(seed)
    order = []
    while len(order) < steps:
        order.extend(rng.permutation(size).tolist())
    return order[:steps]


def evaluate(model, datasets, criterion, opt, *, retain_dice=False):
    model.eval()
    result = {}
    with torch.no_grad():
        for domain, data in datasets.items():
            dice = []
            false_empty = 0
            for batch in DataLoader(data, batch_size=1, shuffle=False):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, predictions, _ = forward_and_loss(batch, model, criterion, opt, phase="segmentation")
                positive = predictions["masks"] > 0
                mask = batch["mask"].to(device=opt.device, dtype=torch.bool)
                intersection = (positive & mask).sum()
                predicted = positive.sum()
                score = (2.0 * intersection / (predicted + mask.sum()).clamp_min(1)).item()
                dice.append(float(score))
                false_empty += int(predicted.item() == 0)
            result[domain] = {"dice": float(np.mean(dice)), "image_count": len(dice),
                              "empty_predictions": false_empty}
            if retain_dice:
                result[domain]["per_image_dice"] = dice
    return result


def train_arm(label, mode, train_sets, schedules, eval_sets, model_args, opt, args):
    destination = checkpoint_path(args, label)
    if destination is not None and set(eval_sets) != set(args.source):
        raise ValueError("Checkpoint runs must evaluate exactly the configured source VAL domains")
    seed_everything(args.seed)
    model = build_uiv_us_vit_b(model_args, checkpoint=model_args.pretrained_ckpt).to(opt.device)
    set_stage(model, "segmentation")
    # Fix shared running statistics across arms so a late domain cannot overwrite them.
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    optimizer = build_optimizer(model, args.base_lr, "segmentation", weight_decay=model_args.weight_decay)
    criterion = get_criterion(opt=opt)
    named = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    parameters = tuple(param for _, param in named)
    protected = tuple(name.startswith("prompt_bank.") for name, _ in named)
    domains = list(train_sets)
    loaders = {
        domain: iter(DataLoader(Subset(dataset, schedules[domain]), batch_size=1, shuffle=False))
        for domain, dataset in train_sets.items()
    }
    weights = [len(train_sets[domain]) for domain in domains] if label == "pooled" else None
    losses = defaultdict(list)
    records = defaultdict(list)
    update_rng = random.Random(args.seed)
    print(f"{label} start domains={domains} trainable={sum(p.numel() for p in parameters):,}", flush=True)
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        domain_gradients = []
        for domain in domains:
            domain_index = args.source.index(domain)
            batch_seed = args.seed + 1_000_003 * (domain_index + 1) + step
            random.seed(batch_seed)
            np.random.seed(batch_seed % 2**32)
            torch.manual_seed(batch_seed)
            batch = next(loaders[domain])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _, _ = forward_and_loss(batch, model, criterion, opt, phase="segmentation")
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Non-finite loss in {label}, {domain}, step {step + 1}")
            gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
            domain_gradients.append(gradients)
            losses[domain].append(float(loss.detach().item()))
            del loss, _
        ordering = update_rng.sample(range(len(domains)), len(domains))
        merged, stats = merge_domain_gradients(
            domain_gradients, weights, mode=mode, ordering=ordering, protected=protected,
        )
        for param, gradient in zip(parameters, merged):
            param.grad = gradient
        shared = tuple(index for index, flag in enumerate(protected)
                       if not flag and all(gradient[index] is not None for gradient in domain_gradients))
        before = [parameters[index].detach().clone() for index in shared]
        optimizer.step()
        displacement = [previous - parameters[index].detach()
                        for index, previous in zip(shared, before)]
        stats["shared_adam_descent_dot"] = [
            float(torch.stack([torch.sum(delta * gradient[index], dtype=torch.float64)
                               for index, delta in zip(shared, displacement)]).sum().item())
            if shared else 0.0
            for gradient in domain_gradients
        ]
        del merged, domain_gradients, gradients, before, displacement
        for name, value in stats.items():
            records[name].append(value)
        if step == 0 or (step + 1) % 8 == 0 or step + 1 == args.steps:
            current = {domain: round(values[-1], 5) for domain, values in losses.items()}
            print(f"{label} update {step + 1}/{args.steps}: {current}", flush=True)

    print(f"{label} evaluating {args.eval_split}: {list(eval_sets)}", flush=True)
    metric = evaluate(model, eval_sets, criterion, opt)
    dice = [metric[domain]["dice"] for domain in args.source]
    result = {
        "domain_metrics": metric,
        "source_macro_dice": float(np.mean(dice)),
        "source_worst_dice": min(dice),
        "mean_training_loss": {key: float(np.mean(values)) for key, values in losses.items()},
        "observed_train_images_per_domain": {domain: len(set(schedules[domain])) for domain in domains},
        "images_drawn_per_training_domain": {domain: args.steps for domain in domains},
        "gradient_statistics": dict(records),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    if destination is not None:
        result["checkpoint"] = save_checkpoint_atomic(model, destination)
    print(f"{label} metrics {json.dumps(metric)}", flush=True)
    return result


def write_progress(path, report):
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def main():
    args = options()
    output = require_private_path(args.output, "Experiment output")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    exclusions = json.loads(args.exclude.read_text(encoding="utf-8")) if args.exclude else {}
    overrides = json.loads(args.split_overrides.read_text(encoding="utf-8")) if args.split_overrides else {}
    for domain, splits in overrides.items():
        if set(splits) != {"train", "val", "test"}:
            raise ValueError(f"Split override requires train, val and test entries for {domain}")
        paths = [entry for entries in splits.values() for entry in entries]
        if any(not isinstance(entry, str) or not entry.strip() for entry in paths) or len(paths) != len(set(paths)):
            raise ValueError(f"Split override entries must be nonempty path strings and mutually disjoint for {domain}")
    torch.set_num_threads(4)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This experiment requires a CUDA device with bfloat16 support")
    model_args = build_parser().parse_args(["--temporal_mode", "baseline", "--no-warmup"])
    model_args.pretrained_ckpt = str(args.sam)
    model_args.resnet_ckpt = str(args.resnet)
    opt = get_config("CE")
    opt.data_path = str(require_private_path(args.data_root, "Data root"))
    opt.data_roots = {TASK: opt.data_path}
    train_2d, val_2d, _, _ = make_transforms(opt, model_args)
    train_sets = {}
    eval_sets = {}
    filtered = {}
    for domain in args.source + args.held_out:
        filtered[domain] = {}
        if domain in args.source:
            train_sets[domain], filtered[domain]["train"] = eligible_dataset(
                opt, domain, "train", train_2d, exclusions, overrides,
            )
        dataset, filtered[domain][args.eval_split] = eligible_dataset(
            opt, domain, args.eval_split, val_2d, exclusions, overrides,
        )
        if args.eval_limit and args.eval_limit < len(dataset):
            rng = np.random.default_rng(args.seed + 789)
            selection = sorted(rng.choice(len(dataset), args.eval_limit, replace=False).tolist())
            dataset = Subset(dataset, selection)
        eval_sets[domain] = dataset

    schedules = {domain: draw_training_rows(len(dataset), args.steps, args.seed + 13 * index)
                 for index, (domain, dataset) in enumerate(train_sets.items())}
    report = {
        "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "protocol": ("Matched seed, draws, augmentations and update count. One microbatch per source per step "
                     "for joint arms; one for single-domain arms. Pooled uses eligible source proportions as "
                     "weights on the same stratified microbatches. Existing mask-derived foreground-click task."),
        "patient_scope": "Image-level comparisons unless source patient metadata was audited separately.",
        "train_eligible": {domain: len(data) for domain, data in train_sets.items()},
        "eval_images": {domain: len(data) for domain, data in eval_sets.items()},
        "excluded_or_mask_empty": filtered,
        "runs": {},
        "frozen_benchmark_source_sha256": FROZEN_SOURCE_SHA256,
    }
    write_progress(output, report)
    for arm in args.arms:
        if arm == "single":
            variants = [(f"single:{domain}", "weighted_mean", {domain: data})
                        for domain, data in train_sets.items()]
        else:
            variants = [(arm, "pcgrad" if arm == "pcgrad" else "weighted_mean", train_sets)]
        for label, mode, active_sets in variants:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            report["runs"][label] = train_arm(label, mode, active_sets, schedules, eval_sets, model_args, opt, args)
            write_progress(output, report)
    single_dice = {
        domain: report["runs"][f"single:{domain}"]["domain_metrics"][domain]["dice"]
        for domain in args.source if f"single:{domain}" in report["runs"]
    }
    for label, run in report["runs"].items():
        if not label.startswith("single:"):
            run["delta_dice_vs_single"] = {
                domain: run["domain_metrics"][domain]["dice"] - reference
                for domain, reference in single_dice.items()
            }
    report["complete"] = True
    write_progress(output, report)


if __name__ == "__main__":
    main()
