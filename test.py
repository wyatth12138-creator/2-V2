"""Independent UIV-US evaluation entry point."""

import json
from pathlib import Path

import argparse

from training.run_selection import add_data_arguments


def build_parser():
    parser = argparse.ArgumentParser(description="Two-stage UIV-US segmentation/classification training")

    add_data_arguments(parser)
    parser.add_argument("--modelname", default="UIV-US", choices=("UIV-US",), help=argparse.SUPPRESS)
    parser.add_argument("--task", default="CE", type=str)
    parser.add_argument("--fewshot", default="all", choices=["all", "1shot", "5shot", "10shot", "10pct"])
    parser.add_argument("--encoder_input_size", type=int, default=256)
    parser.add_argument("--low_image_size", type=int, default=128)
    parser.add_argument("--pretrained_ckpt", type=str, default=None)
    parser.add_argument("--resnet_ckpt", type=str, default="checkpoints/model.safetensors")
    parser.add_argument("--vit_name", default="vit_b", choices=("vit_b",), help=argparse.SUPPRESS)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--bs_img", type=int, default=4)
    parser.add_argument("--bs_vid", type=int, default=1)
    parser.add_argument("--base_lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--seg_epochs", type=int, default=None)
    parser.add_argument("--cls_epochs", type=int, default=None)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup_period", type=int, default=250)
    parser.add_argument("--keep_log", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--memory_topk", type=int, default=10)
    parser.add_argument(
        "--memory_retrieval_mode",
        type=str,
        default="topk",
        choices=["topk", "all", "recent_only", "cond_only"],
    )
    parser.add_argument("--interaction_gate_upper", type=float, default=0.5)
    parser.add_argument("--disable_dynamic_prompt", action="store_true")
    parser.add_argument("--disable_mutual_prompt", action="store_true")
    parser.add_argument("--disable_prototype_loss", action="store_true")
    parser.add_argument("--disable_memory", action="store_true")
    parser.add_argument("--disable_recursive_fusion", action="store_true")
    return parser



def main():
    para = build_parser()
    para.description = "Independent UIV-US test from a saved two-stage checkpoint"
    para.add_argument("--checkpoint", type=Path, required=True, help="Final stage2_classification.pth file")
    para.add_argument("--case_predictions", action="store_true", help="Store pseudonymous per-case scores for paired analysis.")
    args = para.parse_args()
    if args.data_root is None:
        para.error("--data_root is required for independent testing")
    if args.output_root is None:
        para.error("--output_root is required for test reports")
    if not args.checkpoint.is_file():
        para.error(f"Checkpoint not found: {args.checkpoint}")

    from models.uiv_us import build_uiv_us_vit_b
    from training.checkpoint import load_checkpoint
    from training.data_splits import DATA_SPLITS, build_loader_bundle, make_transforms
    from training.reporting import write_records
    from training.run_selection import TASK_ARGUMENTS, configure_roots, selected_splits
    from utils.config import get_config
    from utils.evaluation import get_eval
    from utils.loss_functions.sam_loss import get_criterion

    opt = get_config(args.task)
    configure_roots(opt, args)
    splits = selected_splits(args, DATA_SPLITS)
    # Match the model variant recorded by training. Raw V1 checkpoints have no observer.
    config = args.checkpoint.parent / "run_config.json"
    if args.temporal_mode is None:
        args.temporal_mode = json.loads(config.read_text(encoding="utf-8")).get("temporal_mode") if config.is_file() else "baseline"
    # The final checkpoint already contains both pretrained backbones.
    args.pretrained_ckpt = None
    args.resnet_ckpt = None
    model = build_uiv_us_vit_b(args).to(opt.device)
    load_checkpoint(model, args.checkpoint)
    criterion = get_criterion(opt=opt)
    _, tf_val_2d, _, tf_val_3d = make_transforms(opt, args)

    records = []
    predictions = [] if args.case_predictions else None
    for task in TASK_ARGUMENTS:
        group = "bmode_val" if task.startswith("bmode") else "ceus_val"
        for dataset in splits[group][task]:
            loaders = build_loader_bundle(
                opt, {task: [dataset]}, "test", args, tf_val_2d, tf_val_3d, shuffle=False
            )
            if task not in loaders:
                raise RuntimeError(f"No test samples for {task}/{dataset}")
            summary = get_eval(loaders[task], model, criterion, opt, args, predictions=predictions)
            summary.update(split="test", dataset=dataset, num_samples=len(loaders[task].dataset))
            records.append(summary)
            print(f"TEST {task}/{dataset}: {summary}", flush=True)

    report_dir = args.output_root.expanduser().resolve()
    write_records(str(report_dir), "independent_test", records)
    if predictions:
        write_records(str(report_dir), "case_predictions", predictions)
    print(f"Test reports: {report_dir}")


if __name__ == "__main__":
    main()
