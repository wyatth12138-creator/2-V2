from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence


VALID_SPLITS = ("train", "val", "test")


def find_dataset_root(
    base_dir: str | Path,
    dataset_name: str,
    task_directories: Sequence[str],
) -> Path:
    """Return the first existing task/dataset root in the specified order."""

    base = Path(base_dir)
    candidates = [base / task_dir / dataset_name for task_dir in task_directories]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    rendered = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Dataset root not found; checked: {rendered}")


def read_split_entries(dataset_root: str | Path, split: str) -> list[str]:
    """Read a canonical split list and reject unsafe or ambiguous entries."""

    if split not in VALID_SPLITS:
        raise ValueError(f"Unsupported split {split!r}; expected one of {VALID_SPLITS}")

    root = Path(dataset_root)
    split_path = root / f"{split}.txt"
    if not split_path.is_file():
        raise FileNotFoundError(
            f"Canonical split list not found: {split_path}. "
            "Datasets without a split list are treated as pending and cannot be loaded."
        )

    entries: list[str] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(split_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        entry = raw_line.strip().replace("\\", "/")
        if not entry:
            continue
        path = PurePosixPath(entry)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe split entry at {split_path}:{line_number}: {entry!r}")
        normalized = path.as_posix()
        key = normalized.casefold()
        if key in seen:
            raise ValueError(f"Duplicate split entry at {split_path}:{line_number}: {normalized!r}")
        seen.add(key)
        entries.append(normalized)

    if not entries:
        raise RuntimeError(f"Empty split list (pending or invalid): {split_path}")
    return entries


def assert_disjoint_splits(dataset_root: str | Path) -> dict[str, list[str]]:
    """Load all canonical splits and fail if any sample appears in two splits."""

    split_entries = {split: read_split_entries(dataset_root, split) for split in VALID_SPLITS}
    folded = {split: {entry.casefold() for entry in entries} for split, entries in split_entries.items()}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = folded[left] & folded[right]
        if overlap:
            examples = ", ".join(sorted(overlap)[:5])
            raise ValueError(
                f"Split leakage in {dataset_root}: {left}/{right} share {len(overlap)} entries; "
                f"examples: {examples}"
            )
    return split_entries


def resolve_partitioned_file(
    dataset_root: str | Path,
    collection: str,
    split: str,
    entry: str,
    suffix: str | None = None,
) -> Path:
    """Resolve a manifest entry in flat or split-specific subdirectories."""

    root = Path(dataset_root)
    relative = PurePosixPath(entry)
    if suffix is not None:
        relative = relative.with_suffix(suffix)

    candidates = (
        root / collection / Path(*relative.parts),
        root / collection / split / Path(*relative.parts),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    rendered = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Manifest entry {entry!r} was not found; checked: {rendered}")


def split_is_available(dataset_root: str | Path, split: str) -> bool:
    """Return True only for a present, non-empty canonical split list."""

    try:
        read_split_entries(dataset_root, split)
    except (FileNotFoundError, RuntimeError, ValueError):
        return False
    return True

