import argparse
import json
import os
import time
from pathlib import Path
from collections import OrderedDict


def merge_loader_bundles(*bundles):
    merged = OrderedDict()
    for bundle in bundles:
        for task_name, loader in bundle.items():
            if task_name in merged:
                raise ValueError(f"Duplicate task loader while merging bundles: {task_name}")
            merged[task_name] = loader
    return merged


def filter_task_loaders(loaders, predicate):
    from utils.task_registry import get_task_spec

    return OrderedDict(
        (task_name, loader)
        for task_name, loader in loaders.items()
        if predicate(get_task_spec(task_name[0] if isinstance(task_name, tuple) else task_name))
    )


def _steps_per_epoch(loaders):
    return max((len(loader) for loader in loaders.values()), default=1) * max(len(loaders), 1)


def _domain_steps_per_epoch(loaders, domains_per_update):
    tasks = {}
    for (task, _), loader in loaders.items():
        tasks.setdefault(task, []).append(len(loader))
    if not tasks:
        return 1
    old_rounds = max(sum(lengths) for lengths in tasks.values())
    domains_in_one_round = sum(min(len(lengths), domains_per_update) for lengths in tasks.values())
    desired_microbatches = old_rounds * len(tasks)
    rounds = max(1, (desired_microbatches + domains_in_one_round - 1) // domains_in_one_round)
    return rounds * len(tasks)


def _trainable_parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def run_training(args):
    from models.uiv_us import build_uiv_us_vit_b
    from training.data_splits import DATA_SPLITS, build_domain_loader_bundle, build_loader_bundle, make_transforms
    from training.engine import evaluate_domain_loader_bundle, evaluate_loader_bundle, train_domain_updates, train_round_robin
    from training.reporting import log_scalars, save_model, seed_everything, write_records
    from training.strategy import build_optimizer, set_stage
    from utils.config import get_config
    from utils.loss_functions.sam_loss import get_criterion
    from training.run_selection import configure_roots, selected_splits

    opt = get_config(args.task)
    configure_roots(opt, args)
    splits = selected_splits(args, DATA_SPLITS)
    seed_everything(args.seed)

    if args.init_checkpoint is None:
        model = build_uiv_us_vit_b(args, checkpoint=args.pretrained_ckpt).to(opt.device)
    else:
        from training.checkpoint import load_checkpoint

        build_args = argparse.Namespace(**vars(args))
        build_args.pretrained_ckpt = None
        build_args.resnet_ckpt = None
        model = build_uiv_us_vit_b(build_args).to(opt.device)
        load_checkpoint(model, args.init_checkpoint, allow_new_observer=True)
    criterion = get_criterion(opt=opt)
    tf_train_2d, tf_val_2d, tf_train_3d, tf_val_3d = make_transforms(opt, args)

    domain_mode = getattr(args, "domain_update", "round_robin")
    domain_aware = domain_mode != "round_robin"
    domains_per_update = getattr(args, "domains_per_update", 2)
    domain_steps = getattr(args, "domain_steps_per_epoch", None)
    if domain_aware and (domains_per_update < 1 or (domain_steps is not None and domain_steps < 1)):
        raise ValueError("domains_per_update and domain_steps_per_epoch must be positive")
    build_loaders = build_domain_loader_bundle if domain_aware else build_loader_bundle
    evaluate = evaluate_domain_loader_bundle if domain_aware else evaluate_loader_bundle

    def epoch_steps(loaders):
        if domain_aware:
            return domain_steps or _domain_steps_per_epoch(loaders, domains_per_update)
        return _steps_per_epoch(loaders)

    domain_records = []

    def train_epoch(loaders, model, criterion, optimizer, iter_num, max_iterations, phase):
        if not domain_aware:
            return train_round_robin(
                loaders, model, criterion, optimizer, args, opt, iter_num, max_iterations, phase,
            )
        loss, iteration, domain_loss, diagnostics = train_domain_updates(
            loaders, model, criterion, optimizer, args, opt, iter_num, max_iterations, phase,
            mode="pcgrad" if domain_mode == "pcgrad" else "weighted_mean",
            updates=epoch_steps(loaders), domains_per_update=domains_per_update, seed=args.seed,
            domain_weight_power=1.0 if domain_mode == "pooled" else 0.0,
        )
        domain_records.append({"phase": phase, **diagnostics})
        return loss, iteration, {f"{task}_{dataset}": value for (task, dataset), value in domain_loss.items()}

    bmode_train = build_loaders(
        opt, splits["bmode_train"], "train", args, tf_train_2d, tf_train_3d,
        fewshot_mode="all", shuffle=True,
    )
    ceus_train = build_loaders(
        opt, splits["ceus_train"], "train", args, tf_train_2d, tf_train_3d,
        fewshot_mode=args.fewshot, shuffle=True,
    )
    bmode_val = build_loaders(
        opt, splits["bmode_val"], "val", args, tf_val_2d, tf_val_3d,
        fewshot_mode="all", shuffle=False,
    )
    ceus_val = build_loaders(
        opt, splits["ceus_val"], "val", args, tf_val_2d, tf_val_3d,
        fewshot_mode="all", shuffle=False,
    )

    all_train = merge_loader_bundles(bmode_train, ceus_train)
    all_val = merge_loader_bundles(bmode_val, ceus_val)
    segmentation_train = filter_task_loaders(all_train, lambda spec: spec.is_segmentation)
    classification_train = filter_task_loaders(all_train, lambda spec: spec.is_classification)
    segmentation_val = filter_task_loaders(all_val, lambda spec: spec.is_segmentation)

    if not segmentation_train and not args.classification_only:
        raise RuntimeError("No segmentation loaders are configured for stage 1")
    if not classification_train:
        raise RuntimeError("No classification loaders are configured for stage 2")

    total_epochs = max(2, int(opt.epochs))
    seg_epochs = args.seg_epochs if args.seg_epochs is not None else max(1, total_epochs // 2)
    cls_epochs = args.cls_epochs if args.cls_epochs is not None else max(1, total_epochs - seg_epochs)
    if args.classification_only:
        seg_epochs = 0
    if (seg_epochs <= 0 and not args.classification_only) or cls_epochs <= 0:
        raise ValueError("seg_epochs and cls_epochs must both be positive")

    seg_max_iterations = epoch_steps(segmentation_train) * seg_epochs
    cls_max_iterations = epoch_steps(classification_train) * cls_epochs
    seg_iter_num = 0
    cls_iter_num = 0

    log_root = os.path.join(opt.save_path, "two_stage", "run_" + time.strftime("%Y%m%d_%H%M%S"))
    Path(log_root).mkdir(parents=True, exist_ok=False)
    with open(os.path.join(log_root, "run_config.json"), "w", encoding="utf-8") as stream:
        json.dump(vars(args), stream, default=str, indent=2)
    writer = None
    if args.keep_log:
        from torch.utils.tensorboard import SummaryWriter

        os.makedirs(log_root, exist_ok=True)
        writer = SummaryWriter(os.path.join(log_root, "tensorboard"))

    records = []

    # Stage 1: train all segmentation datasets across B-mode/video/CEUS.
    if seg_epochs:
        set_stage(model, "segmentation")
        print(
            f"Stage 1/2 segmentation: tasks={list(segmentation_train)}, epochs={seg_epochs}, "
            f"trainable_params={_trainable_parameter_count(model):,}"
        )
        optimizer = build_optimizer(
            model, args.base_lr, stage="segmentation", weight_decay=args.weight_decay,
        )
        for epoch in range(seg_epochs):
            mean_loss, seg_iter_num, per_task = train_epoch(
                segmentation_train, model, criterion, optimizer,
                seg_iter_num, seg_max_iterations, phase="segmentation",
            )
            scalars = {"train_loss": mean_loss, **{f"loss_{key}": value for key, value in per_task.items()}}
            log_scalars(writer, "stage1_segmentation", scalars, epoch)

        stage1_records = evaluate(
            segmentation_val, model, criterion, opt, args,
            stage_name="stage1_segmentation", domain_name="all_modalities",
        )
        records.extend(stage1_records)
        save_model(model, os.path.join(log_root, "stage1_segmentation.pth"))

    # Stage 2: freeze the encoder and segmentation path; train classification/FLA.
    set_stage(model, "classification")
    print(
        f"Stage 2/2 classification: tasks={list(classification_train)}, epochs={cls_epochs}, "
        f"trainable_params={_trainable_parameter_count(model):,}"
    )
    optimizer = build_optimizer(
        model, args.base_lr, stage="classification", weight_decay=args.weight_decay,
    )
    for epoch in range(cls_epochs):
        mean_loss, cls_iter_num, per_task = train_epoch(
            classification_train, model, criterion, optimizer,
            cls_iter_num, cls_max_iterations, phase="classification",
        )
        scalars = {"train_loss": mean_loss, **{f"loss_{key}": value for key, value in per_task.items()}}
        log_scalars(writer, "stage2_classification", scalars, epoch)

    final_records = evaluate(
        all_val, model, criterion, opt, args,
        stage_name="stage2_final", domain_name="all_modalities",
    )
    records.extend(final_records)
    save_model(model, os.path.join(log_root, "stage2_classification.pth"))
    write_records(log_root, "metrics", records)
    if domain_records:
        with open(os.path.join(log_root, "domain_optimization.json"), "w", encoding="utf-8") as stream:
            json.dump(domain_records, stream, indent=2, allow_nan=False)

    if writer is not None:
        writer.close()
    return log_root
