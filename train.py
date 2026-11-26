import argparse
from pathlib import Path

from training.run_selection import add_data_arguments
from training.journal_runner import run_training


def build_parser():
    parser = argparse.ArgumentParser(description="Two-stage UIV-US segmentation/classification training")

    add_data_arguments(parser)
    parser.set_defaults(temporal_mode="registered")
    parser.add_argument("--init_checkpoint", type=Path, help="Start from a trained UIV-US or V2 state dictionary.")
    parser.add_argument("--classification_only", action="store_true", help="Adapt a trained model on classification only.")
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
    parser.add_argument("--domain_update", choices=("round_robin", "pooled", "balanced", "pcgrad"), default="round_robin",
                        help="Choose original rotation or dataset-homogeneous updates within the same task.")
    parser.add_argument("--domains_per_update", type=int, default=2,
                        help="Maximum domains with gradients measured before the same optimizer step.")
    parser.add_argument("--domain_steps_per_epoch", type=int,
                        help="Override the number of opt-in domain-aware optimizer steps in each epoch.")
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




if __name__ == "__main__":
    para = build_parser()
    args = para.parse_args()
    if args.data_root is None:
        para.error("--data_root is required")
    if args.classification_only and args.init_checkpoint is None:
        para.error("--classification_only needs a trained --init_checkpoint")
    if args.init_checkpoint is None:
        for name in ("pretrained_ckpt", "resnet_ckpt"):
            if getattr(args, name) is None:
                para.error(f"--{name} is required without --init_checkpoint")
    output = run_training(args)
    print(f"Training outputs: {output}")
