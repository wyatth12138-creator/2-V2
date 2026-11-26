"""Task selection and data roots shared by the train and test entry points."""

from __future__ import annotations

import os
from pathlib import Path

TASK_ARGUMENTS = {
    "bmode_img_seg": "bmode_img_seg_datasets",
    "bmode_img_cls": "bmode_img_cls_datasets",
    "bmode_vid_seg": "bmode_vid_seg_datasets",
    "bmode_vid_cls": "bmode_vid_cls_datasets",
    "ceus_vid_seg": "ceus_vid_seg_datasets",
    "ceus_vid_cls": "ceus_vid_cls_datasets",
}
PUBLIC_ROOT = Path(__file__).resolve().parents[1]


def default_process_root() -> Path:
    configured = os.environ.get("UIV_US_PROCESS_ROOT")
    if configured:
        return Path(configured).expanduser()
    adjacent = PUBLIC_ROOT.with_name("UIV-US-V2-Precess")
    if PUBLIC_ROOT.name.lower() == "uiv-us-v2" and adjacent.is_dir():
        return adjacent
    return PUBLIC_ROOT.parent / "process"


def require_private_path(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved == PUBLIC_ROOT or resolved.is_relative_to(PUBLIC_ROOT):
        raise ValueError(f"{label} must be outside the public source directory: {resolved}")
    return resolved


def add_data_arguments(parser):
    parser.add_argument("--data_root", type=Path, help="Prepared root containing segmentation/, classification/, video_seg/, ceus_seg/, ceus_cls/.")
    parser.add_argument("--output_root", type=Path, default=default_process_root() / "UIV-US-v2" / "runs",
                        help="Directory for checkpoints and reports.")
    parser.add_argument("--ordered_video_cls", action="store_true",
                        help="Use patient-split NPZ manifests for video classification without GT prompts.")
    parser.add_argument("--video_frames", type=int, default=10, help="Frame budget for ordered video classification.")
    parser.add_argument("--temporal_mode", choices=("baseline", "fixed", "wavelet", "stft", "adaptive", "projected", "anchored", "registered"), default=None)
    parser.add_argument("--only_selected", action="store_true",
                        help="Disable defaults for tasks with no explicit dataset list.")
    for task, argument in TASK_ARGUMENTS.items():
        parser.add_argument(f"--{argument}", nargs="+", metavar="DATASET", help=f"Datasets for {task}; defaults to the repository task list.")


def configure_roots(opt, args):
    if getattr(args, "data_root", None) is not None:
        root = str(require_private_path(args.data_root, "Data root"))
        if not Path(root).is_dir():
            raise FileNotFoundError(f"Data root does not exist: {root}")
        opt.data_path = root
        opt.data_roots = {task: root for task in TASK_ARGUMENTS}
    if getattr(args, "output_root", None) is not None:
        output = require_private_path(args.output_root, "Output root")
        output.mkdir(parents=True, exist_ok=True)
        opt.save_path = str(output)


def selected_splits(args, defaults):
    splits = {group: {task: list(names) for task, names in tasks.items()} for group, tasks in defaults.items()}
    for task, argument in TASK_ARGUMENTS.items():
        names = getattr(args, argument, None)
        if names is None:
            if not getattr(args, "only_selected", False):
                continue
            names = []
        for group in splits:
            if group.startswith("bmode") == task.startswith("bmode") and task in splits[group]:
                splits[group][task] = list(names)
    return splits
