from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class TaskSpec:
    name: str
    aliases: tuple[str, ...]
    prompt_name: str
    is_video: bool
    is_segmentation: bool
    is_classification: bool
    uses_temporal_memory: bool
    target_key: str
    metric_group: str


TASK_SPECS: Dict[str, TaskSpec] = {
    "bmode_img_seg": TaskSpec(
        name="bmode_img_seg",
        aliases=("seg", "segmentation", "b-mode image seg"),
        prompt_name="segmentation",
        is_video=False,
        is_segmentation=True,
        is_classification=False,
        uses_temporal_memory=False,
        target_key="low_mask",
        metric_group="segmentation",
    ),
    "bmode_img_cls": TaskSpec(
        name="bmode_img_cls",
        aliases=("cls", "classification", "b-mode image cls"),
        prompt_name="classification",
        is_video=False,
        is_segmentation=False,
        is_classification=True,
        uses_temporal_memory=False,
        target_key="label",
        metric_group="classification",
    ),
    "bmode_vid_seg": TaskSpec(
        name="bmode_vid_seg",
        aliases=("b-mode video se", "b-mode video seg", "videos segmentation"),
        prompt_name="videos segmentation",
        is_video=True,
        is_segmentation=True,
        is_classification=False,
        uses_temporal_memory=True,
        target_key="mask",
        metric_group="video_segmentation",
    ),
    "bmode_vid_cls": TaskSpec(
        name="bmode_vid_cls",
        aliases=("b-mode video classification",),
        prompt_name="videos classification",
        is_video=True,
        is_segmentation=False,
        is_classification=True,
        uses_temporal_memory=True,
        target_key="label",
        metric_group="video_classification",
    ),
    "ceus_vid_seg": TaskSpec(
        name="ceus_vid_seg",
        aliases=("ceus video se", "ceus video seg"),
        prompt_name="videos segmentation",
        is_video=True,
        is_segmentation=True,
        is_classification=False,
        uses_temporal_memory=True,
        target_key="mask",
        metric_group="video_segmentation",
    ),
    "ceus_vid_cls": TaskSpec(
        name="ceus_vid_cls",
        aliases=("ceus video cls", "ceus video classification", "videos classification"),
        prompt_name="videos classification",
        is_video=True,
        is_segmentation=False,
        is_classification=True,
        uses_temporal_memory=True,
        target_key="label",
        metric_group="video_classification",
    ),
}


_TASK_ALIAS_TO_NAME: Dict[str, str] = {}
for canonical_name, spec in TASK_SPECS.items():
    _TASK_ALIAS_TO_NAME[canonical_name] = canonical_name
    for alias in spec.aliases:
        _TASK_ALIAS_TO_NAME[alias] = canonical_name


def canonicalize_task(task_name: str) -> str:
    if task_name not in _TASK_ALIAS_TO_NAME:
        raise KeyError(f"Unknown task name: {task_name}")
    return _TASK_ALIAS_TO_NAME[task_name]


def get_task_spec(task_name: str) -> TaskSpec:
    return TASK_SPECS[canonicalize_task(task_name)]


def canonicalize_task_batch(task_batch) -> List[str]:
    if isinstance(task_batch, str):
        return [canonicalize_task(task_batch)]
    if isinstance(task_batch, Sequence):
        return [canonicalize_task(str(task_name)) for task_name in task_batch]
    raise TypeError(f"Unsupported task batch type: {type(task_batch)}")


def get_primary_task(task_batch) -> str:
    canonical = canonicalize_task_batch(task_batch)
    unique = list(dict.fromkeys(canonical))
    if len(unique) != 1:
        raise ValueError(f"Expected a homogeneous task batch, got {unique}")
    return unique[0]


def get_target_tensor_name(task_name: str) -> str:
    return get_task_spec(task_name).target_key


def task_uses_video_path(task_name: str) -> bool:
    return get_task_spec(task_name).is_video


def task_uses_segmentation(task_name: str) -> bool:
    return get_task_spec(task_name).is_segmentation


def task_uses_classification(task_name: str) -> bool:
    return get_task_spec(task_name).is_classification


def build_anchor_keys(task_batch, domain_name: str) -> List[str]:
    canonical_tasks = canonicalize_task_batch(task_batch)
    return [f"{domain_name}|{task_name}" for task_name in canonical_tasks]


def metric_name_for_forgetting(metric_group: str) -> str:
    if metric_group in ("segmentation", "video_segmentation"):
        return "dice"
    if metric_group in ("classification", "video_classification"):
        return "auc"
    return "loss"


def iter_task_specs(task_names: Iterable[str]) -> List[TaskSpec]:
    return [get_task_spec(task_name) for task_name in task_names]
