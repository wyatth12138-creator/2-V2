"""Unprompted binary diagnosis on archived, ordered ultrasound clips.

Layout: ``base_dir/ordered_video_cls/{dataset_name}/manifest.csv``. Required CSV
columns are ``patient_id,case_id,split,path,label``; split is train/val/test,
label is 0/1, and path is a dataset-relative NPZ path. Cases and file paths must
be unique, and all cases from a patient must belong to one split. The entire
manifest is validated before any array is loaded.

NPZ arrays (loaded with allow_pickle=False):
* frames: uint8 [T,H,W], [T,H,W,1], or RGB [T,H,W,3].
* timestamps: optional finite, strictly increasing [T], in seconds by default.
* timestamp_unit: optional scalar string 's' or 'frame'. Without timestamps,
  only 'frame' is valid and timestamps are generated as frame indices.
* frame_valid: optional boolean / binary [T]; defaults to all valid.

Long clips are deterministically sampled across the *whole archived clip*.
This is offline classification, not an online/prefix acquisition protocol.
Short clips receive zero images, invalid flags and increasing synthetic times.
Grayscale is replicated to RGB for both tasks. For CEUS this is a deliberately
new single-stream protocol, not the legacy B-mode/CEUS/baseline-difference
three-channel representation. No masks, spatial prompts, or GT hints are read.

JointTransform3D receives float32 images [T,3,H,W] in 0..255 and zero uint8
masks [T,H,W]. Returned images stay in 0..255 for UIV-US's pixel normalization;
the transform must preserve temporal order and frame count. Output mask and
low_mask are zero placeholders, never segmentation supervision.
"""

from __future__ import annotations

import csv
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.prompt_list import (
    modality_prompt_one_hot_dict,
    mode_prompt_one_hot_dict,
    position_prompt_dict,
    position_prompt_one_hot_dict,
    task_prompt_one_hot_dict,
    type_prompt_dict,
    type_prompt_one_hot_dict,
)


_SPLITS = frozenset(("train", "val", "test"))
_TASKS = frozenset(("bmode_vid_cls", "ceus_vid_cls"))
_COLUMNS = frozenset(("patient_id", "case_id", "split", "path", "label"))


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _relative_npz(root: Path, value: str) -> Path:
    """Reject absolute/drive paths, traversal and symlink escapes on any OS."""
    normalized = value.replace("\\", "/")
    relative = PurePosixPath(normalized)
    if (
        not normalized
        or "\x00" in normalized
        or relative.is_absolute()
        or PureWindowsPath(value).drive
        or ".." in relative.parts
        or ":" in normalized
    ):
        raise ValueError(f"NPZ path must be dataset-relative without traversal: {value!r}")
    resolved = (root / normalized).resolve()
    if not _inside(resolved, root) or resolved.suffix.lower() != ".npz":
        raise ValueError(f"NPZ path must stay inside dataset and end in .npz: {value!r}")
    return resolved


class OrderedVideoClassificationDataset(Dataset):
    """Patient-isolated binary video classification with no spatial prompts."""

    def __init__(
        self,
        base_dir: str,
        dataset_name: str,
        task_name: str,
        split: str = "train",
        transform=None,
        frame_length: int = 10,
    ) -> None:
        super().__init__()
        if task_name not in _TASKS:
            raise ValueError(f"task_name must be one of {sorted(_TASKS)}, got {task_name!r}")
        if split not in _SPLITS:
            raise ValueError(f"split must be one of {sorted(_SPLITS)}, got {split!r}")
        if isinstance(frame_length, bool) or not isinstance(frame_length, (int, np.integer)) or frame_length < 1:
            raise ValueError("frame_length must be a positive integer")
        if (
            not isinstance(dataset_name, str)
            or not dataset_name.strip()
            or dataset_name in (".", "..")
            or any(char in dataset_name for char in ("/", "\\", ":", "\x00"))
        ):
            raise ValueError("dataset_name must be a single nonempty directory name")
        self.base_dir = str(base_dir)
        self.dataset_name = dataset_name
        self.task_name = task_name
        self.split = split
        self.transform = transform
        self.frame_length = int(frame_length)
        collection = (Path(base_dir).expanduser().resolve() / "ordered_video_cls").resolve()
        self.root = (collection / dataset_name).resolve()
        if not _inside(self.root, collection):
            raise ValueError("dataset directory escapes ordered_video_cls")
        self.manifest_path = self.root / "manifest.csv"
        self.records = self._read_manifest()
        self.ids = [record["case_id"] for record in self.records]
        self.labels = [record["label"] for record in self.records]

    def _read_manifest(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        patients: dict[str, str] = {}
        cases: set[str] = set()
        paths: set[Path] = set()
        with self.manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            headers = reader.fieldnames or []
            if len(headers) != len(set(headers)) or not _COLUMNS.issubset(headers):
                raise ValueError(f"Manifest requires unique columns {sorted(_COLUMNS)}: {self.manifest_path}")
            for line, row in enumerate(reader, start=2):
                if None in row or any(row.get(key) is None for key in _COLUMNS):
                    raise ValueError(f"Malformed CSV row {line} in {self.manifest_path}")
                row = {key: str(row[key]).strip() for key in _COLUMNS}
                if not row["patient_id"] or not row["case_id"]:
                    raise ValueError(f"Empty patient_id/case_id on manifest row {line}")
                if row["split"] not in _SPLITS or row["label"] not in ("0", "1"):
                    raise ValueError(f"Invalid split or binary label on manifest row {line}")
                patient, case = row["patient_id"], row["case_id"]
                if patient in patients and patients[patient] != row["split"]:
                    raise ValueError(f"Patient {patient!r} appears in multiple splits")
                if case in cases:
                    raise ValueError(f"Duplicate case_id {case!r} in manifest")
                path = _relative_npz(self.root, row["path"])
                if path in paths:
                    raise ValueError(f"Duplicate NPZ path in manifest: {row['path']!r}")
                patients[patient] = row["split"]
                cases.add(case)
                paths.add(path)
                records.append({**row, "label": int(row["label"]), "resolved_path": path})
        # Check all metadata for isolation before touching an NPZ, including
        # nonrequested splits. Existence checks do not read the array payload.
        for record in records:
            if not record["resolved_path"].is_file():
                raise FileNotFoundError(f"Missing NPZ for case {record['case_id']!r}: {record['resolved_path']}")
        selected = [record for record in records if record["split"] == self.split]
        if not selected:
            raise ValueError(f"No cases for split {self.split!r} in {self.manifest_path}")
        return selected

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _load_clip(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        with np.load(path, allow_pickle=False) as archive:
            if "frames" not in archive:
                raise ValueError(f"Missing frames array: {path}")
            frames = archive["frames"]
            if frames.dtype != np.uint8 or frames.ndim not in (3, 4) or any(size < 1 for size in frames.shape):
                raise ValueError(f"frames must be nonempty uint8 [T,H,W] or [T,H,W,C]: {path}")
            if frames.ndim == 3:
                frames = frames[..., None]
            if frames.shape[-1] not in (1, 3):
                raise ValueError(f"frames channel count must be 1 or RGB 3: {path}")
            if frames.shape[-1] == 1:
                frames = np.repeat(frames, 3, axis=-1)
            count = len(frames)
            has_times = "timestamps" in archive
            unit = "s" if has_times else "frame"
            if "timestamp_unit" in archive:
                raw_unit = archive["timestamp_unit"]
                if raw_unit.shape != () or raw_unit.dtype.kind not in ("U", "S"):
                    raise ValueError(f"timestamp_unit must be a scalar string: {path}")
                scalar = raw_unit.item()
                unit = scalar.decode("utf-8") if isinstance(scalar, bytes) else str(scalar)
            if unit not in ("s", "frame") or (not has_times and unit != "frame"):
                raise ValueError(f"Invalid timestamp_unit {unit!r} (absent timestamps require 'frame'): {path}")
            times = archive["timestamps"] if has_times else np.arange(count, dtype=np.float64)
            if times.shape != (count,) or times.dtype.kind not in ("i", "u", "f"):
                raise ValueError(f"timestamps must be numeric [T]: {path}")
            times = times.astype(np.float64)
            if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                raise ValueError(f"timestamps must be finite and strictly increasing: {path}")
            valid = archive["frame_valid"] if "frame_valid" in archive else np.ones(count, dtype=bool)
            if valid.shape != (count,) or valid.dtype.kind not in ("b", "i", "u", "f") or not np.isin(valid, (0, 1)).all():
                raise ValueError(f"frame_valid must be boolean/binary [T]: {path}")
            valid = valid.astype(bool)
            if not valid.any():
                raise ValueError(f"Clip contains no valid frames: {path}")
        return frames, times, valid, unit

    def _prompts(self) -> dict[str, torch.Tensor]:
        # Semantic metadata only. These never depend on diagnosis or GT masks;
        # direct lookup also supports bmode_vid_cls before registry integration.
        organ = position_prompt_dict.get(self.dataset_name, "indis")
        target = type_prompt_dict.get(self.dataset_name, "tumor")
        vectors = {
            "modality_prompt": modality_prompt_one_hot_dict["ceus" if self.task_name == "ceus_vid_cls" else "bmode"],
            "mode_prompt": mode_prompt_one_hot_dict["whole"],
            "type_prompt": type_prompt_one_hot_dict[target],
            "position_prompt": position_prompt_one_hot_dict[organ],
            "task_prompt": task_prompt_one_hot_dict["videos classification"],
        }
        return {key: torch.tensor(value, dtype=torch.float32) for key, value in vectors.items()}

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        path = record["resolved_path"]
        # Re-resolve immediately before opening to also reject replaced symlinks.
        if not _inside(path.resolve(), self.root):
            raise ValueError(f"NPZ path no longer stays inside dataset: {path}")
        frames, times, valid, unit = self._load_clip(path)
        if len(frames) > self.frame_length:
            indices = np.rint(np.linspace(0, len(frames) - 1, self.frame_length)).astype(np.int64)
            frames, times, valid = frames[indices], times[indices], valid[indices]
        if not valid.any():
            raise ValueError(f"Uniform sampling selected no valid frames; increase frame_length: {path}")
        if len(frames) < self.frame_length:
            extra = self.frame_length - len(frames)
            step = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
            padding_times = times[-1] + step * np.arange(1, extra + 1, dtype=np.float64)
            times = np.concatenate((times, padding_times))
            if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                raise ValueError(f"Cannot create finite increasing padding timestamps: {path}")
            frames = np.concatenate((frames, np.zeros((extra, *frames.shape[1:]), dtype=np.uint8)))
            valid = np.concatenate((valid, np.zeros(extra, dtype=bool)))
        image = frames.transpose(0, 3, 1, 2).astype(np.float32)
        mask = np.zeros((self.frame_length, *frames.shape[1:3]), dtype=np.uint8)
        if self.transform is not None:
            result = self.transform(image, mask)
            if not isinstance(result, (tuple, list)) or len(result) not in (2, 3):
                raise ValueError("Video transform must return (image, mask) or (image, mask, low_mask)")
            image = result[0]
        image = torch.as_tensor(np.ascontiguousarray(image) if isinstance(image, np.ndarray) else image, dtype=torch.float32)
        if image.ndim != 4 or image.shape[:2] != (self.frame_length, 3) or min(image.shape[-2:]) < 1 or not torch.isfinite(image).all():
            raise ValueError("Video transform must preserve [T,3,H,W] and produce finite values")
        image = image.contiguous()
        # Keep invalid frames zero even if a spatial/intensity transform changed
        # black padding. No segmentation annotations enter these placeholders.
        image[~torch.from_numpy(valid)] = 0
        dummy = torch.zeros((self.frame_length, *image.shape[-2:]), dtype=torch.long)
        return {
            "task": self.task_name,
            "dataset": self.dataset_name,
            "image": image,
            "mask": dummy,
            "low_mask": dummy.clone(),
            "label": record["label"],
            "case_name": record["case_id"],
            "patient_id": record["patient_id"],
            "frame_time": torch.from_numpy(times.copy()),
            "frame_valid": torch.from_numpy(valid.copy()),
            "timestamp_unit": unit,
            **self._prompts(),
        }
