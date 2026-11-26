"""Audit prepared public segmentation domains without saving images or IDs.

The explicit root must contain DOMAIN/{imgs,masks,train.txt,val.txt,test.txt}.
Only aggregate statistics are written. Exact duplicates mean equal decoded
8-bit grayscale pixels, shape, and dtype; they do not establish patient identity.
Filename category tokens are reported as metadata hints, not pathology labels.
Patient grouping is never inferred from a filename. An optional explicit CSV
with columns sample,group needs a separately declared provenance string.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import itertools
import json
from pathlib import Path, PurePosixPath
import re
import struct
import time

import cv2
import numpy as np


SPLITS = ("train", "val", "test")
EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_entry(entry):
    return PurePosixPath(entry.strip().replace("\\", "/")).as_posix().casefold()


def resolve_file(root, collection, entry):
    relative = PurePosixPath(entry.strip().replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        return None
    parts = relative.parts
    if parts[0].casefold() in {"imgs", "images", "masks", "labels"}:
        parts = parts[1:]
    if not parts:
        return None
    base = root / collection / Path(*parts)
    candidates = [base]
    candidates.extend(base.with_suffix(extension) for extension in EXTENSIONS)
    for path in candidates:
        if path.is_file():
            return path
    return None


def category_hint(entry):
    for category in ("normal", "benign", "malignant"):
        if re.search(r"(?:^|[\\/_ .(\-])" + category + r"(?:$|[\\/_ .)\-])", entry.casefold()):
            return category
    return "unresolved"


def gray_digest(image):
    digest = hashlib.sha256()
    digest.update(struct.pack("<II", *image.shape))
    digest.update(image.dtype.str.encode())
    digest.update(np.ascontiguousarray(image).tobytes())
    return digest.hexdigest()


def parse_assignments(values, option):
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} expects DOMAIN=VALUE")
        name, assigned = value.split("=", 1)
        if not name or not assigned or name in result:
            raise ValueError(f"{option} needs one nonempty assignment per domain")
        result[name] = assigned
    return result


def grouping_summary(root, domain, split_entries, mapping_path, provenance):
    if mapping_path is None:
        return {"status": "not_verified_no_explicit_group_manifest", "patient_level_disjointness_established": False}
    mapping = {}
    with mapping_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"sample", "group"}.issubset(reader.fieldnames or []):
            raise ValueError("Group CSV must have sample and group columns")
        for row in reader:
            sample, group = normalized_entry(row["sample"]), row["group"].strip()
            if not sample or not group:
                continue
            if sample in mapping and mapping[sample] != group:
                raise ValueError("Group CSV contains conflicting mappings")
            mapping[sample] = group
    groups, missing = {}, {}
    for split, entries in split_entries.items():
        found = [mapping.get(normalized_entry(entry)) for entry in entries]
        groups[split] = {group for group in found if group is not None}
        missing[split] = sum(group is None for group in found)
    overlap = {a + "__" + b: len(groups[a] & groups[b]) for a, b in itertools.combinations(SPLITS, 2)}
    return {
        "status": "explicit_mapping_with_declared_provenance_not_independently_verified",
        "provenance": provenance,
        "mapping_sha256": digest_file(mapping_path),
        "groups_per_split": {split: len(group_set) for split, group_set in groups.items()},
        "unmapped_entries_per_split": missing,
        "cross_split_group_overlap": overlap,
        "provided_group_manifest_complete_and_disjoint": not any(missing.values()) and not any(overlap.values()),
        "patient_level_disjointness_established": False,
    }


def audit_domain(base, domain, pixel_index, group_map=None, group_source=None):
    root = base / domain
    if not root.is_dir():
        raise ValueError(f"Requested domain directory is missing: {domain}")
    counters = {split: collections.Counter() for split in SPLITS}
    category_counts = {split: collections.defaultdict(collections.Counter) for split in SPLITS}
    entries, manifest_hashes = {}, {}
    for split in SPLITS:
        path = root / (split + ".txt")
        if not path.is_file():
            raise ValueError(f"Requested domain lacks a canonical {split} list: {domain}")
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        entries[split] = [line.strip() for line in lines if line.strip()]
        manifest_hashes[split] = digest_file(path)
        counters[split]["blank_manifest_lines"] = sum(not line.strip() for line in lines)
        counters[split]["manifest_entries"] = len(entries[split])
        counters[split]["duplicate_manifest_entries"] = len(entries[split]) - len({normalized_entry(entry) for entry in entries[split]})
    folded = {split: {normalized_entry(entry) for entry in values} for split, values in entries.items()}
    name_overlap = {a + "__" + b: len(folded[a] & folded[b]) for a, b in itertools.combinations(SPLITS, 2)}
    cache, mask_values = {}, collections.Counter()
    for split in SPLITS:
        for row_index, entry in enumerate(entries[split]):
            counter = counters[split]
            hint = category_hint(entry)
            category_counts[split][hint]["entries"] += 1
            embedded = PurePosixPath(entry.replace("\\", "/")).parts
            if embedded and embedded[0].casefold() in {"imgs", "images", "masks", "labels"}:
                counter["embedded_collection_prefix_requires_loader_review"] += 1
            image_path = resolve_file(root, "imgs", entry)
            mask_path = resolve_file(root, "masks", entry)
            if image_path is None:
                counter["missing_image"] += 1
            if mask_path is None:
                counter["missing_mask"] += 1
            if image_path is None or mask_path is None:
                continue
            if image_path.resolve() == mask_path.resolve():
                counter["image_mask_same_resolved_file"] += 1
                continue
            key = (str(image_path.resolve()).casefold(), str(mask_path.resolve()).casefold())
            if key not in cache:
                result = {"errors": []}
                if image_path.stat().st_size == 0:
                    result["errors"].append("zero_byte_image")
                if mask_path.stat().st_size == 0:
                    result["errors"].append("zero_byte_mask")
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                if image is None:
                    result["errors"].append("image_decode_failure")
                if mask is None:
                    result["errors"].append("mask_decode_failure")
                if image is not None:
                    result["pixel_hash"] = gray_digest(image)
                    result["shape"] = tuple(image.shape)
                    result["constant_image"] = bool(image.min() == image.max())
                if mask is not None:
                    if mask.ndim > 2:
                        rgb = mask[..., :3]
                        replicated = np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 0], rgb[..., 2])
                        opaque = mask.shape[2] == 3 or bool((mask[..., 3] == np.iinfo(mask.dtype).max).all())
                        if replicated and opaque:
                            mask = mask[..., 0]
                            result["replicated_grayscale_mask"] = True
                        else:
                            result["errors"].append("non_grayscale_mask_requires_label_review")
                    foreground = np.any(mask != 0, axis=2) if mask.ndim > 2 else mask != 0
                    result["empty_mask"] = not bool(foreground.any())
                    result["mask_hash"] = hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()
                    result["mask_values"] = tuple(int(value) for value in np.unique(mask))
                    mask_values.update({str(value): 1 for value in result["mask_values"]})
                    if image is not None and image.shape != mask.shape[:2]:
                        result["errors"].append("image_mask_shape_mismatch")
                result["splits"] = set()
                result["rows"] = collections.defaultdict(list)
                cache[key] = result
            result = cache[key]
            result["splits"].add(split)
            result["rows"][split].append(row_index)
            counter.update(result["errors"])
            if "pixel_hash" in result:
                counter["decoded_images"] += 1
                counter["constant_grayscale_images"] += int(result["constant_image"])
            if "empty_mask" in result:
                counter["replicated_grayscale_mask_files"] += int(result.get("replicated_grayscale_mask", False))
                counter["decoded_masks"] += 1
                counter["empty_foreground_masks"] += int(result["empty_mask"])
                counter["nonempty_foreground_masks"] += int(not result["empty_mask"])
                category_counts[split][hint]["empty_foreground_masks"] += int(result["empty_mask"])
            if not result["errors"]:
                counter["valid_decoded_image_mask_pairs"] += 1
    shape_counts = collections.Counter()
    for result in cache.values():
        if "pixel_hash" not in result:
            continue
        shape_counts[str(result["shape"][0]) + "x" + str(result["shape"][1])] += 1
        pixel_index[result["pixel_hash"]].append({
            "domain": domain, "splits": result["splits"],
            "mask_hash": result.get("mask_hash"), "empty_mask": result.get("empty_mask"),
            "rows": dict(result["rows"]),
        })
    config = root / "config.yaml"
    config_mapping = []
    if config.is_file():
        config_mapping = [line.strip() for line in config.read_text(encoding="utf-8", errors="replace").splitlines() if re.match(r"^\s*\d+\s*:", line)]
    return {
        "root": str(root),
        "split_counts_and_errors": {split: dict(counts) for split, counts in counters.items()},
        "split_entry_overlap_counts": name_overlap,
        "category_filename_hints": {split: {name: dict(counts) for name, counts in groups.items()} for split, groups in category_counts.items()},
        "label_mapping": config_mapping,
        "mask_pixel_values_number_of_unique_files": dict(mask_values),
        "decoded_image_shape_counts": dict(shape_counts),
        "unique_resolved_image_mask_pairs": len(cache),
        "split_manifest_sha256": manifest_hashes,
        "grouping": grouping_summary(root, domain, entries, group_map, group_source),
    }


def duplicate_summary(pixel_index, domains):
    summary = collections.Counter()
    within_domain = {domain: collections.Counter() for domain in domains}
    pairs = {a + "__" + b: collections.Counter() for a, b in itertools.combinations(domains, 2)}
    for records in pixel_index.values():
        grouped = collections.defaultdict(list)
        for record in records:
            grouped[record["domain"]].append(record)
        if len(records) > 1:
            summary["exact_duplicate_grayscale_groups"] += 1
            summary["images_in_exact_duplicate_groups"] += len(records)
        if len(grouped) > 1:
            summary["cross_domain_duplicate_pixel_groups"] += 1
        all_splits = set().union(*(record["splits"] for record in records))
        if len(all_splits) > 1:
            summary["pixel_groups_spanning_any_splits"] += 1
        if "train" in all_splits and ({"val", "test"} & all_splits):
            summary["train_vs_evaluation_duplicate_pixel_groups"] += 1
        hashes = {record["mask_hash"] for record in records if record["mask_hash"] is not None}
        if len(records) > 1 and len(hashes) > 1:
            summary["duplicate_image_groups_with_different_mask_arrays"] += 1
        for domain, subset in grouped.items():
            own_splits = set().union(*(record["splits"] for record in subset))
            if len(subset) > 1:
                within_domain[domain]["duplicate_pixel_groups"] += 1
                within_domain[domain]["images_in_duplicate_groups"] += len(subset)
            for a, b in itertools.combinations(SPLITS, 2):
                if {a, b}.issubset(own_splits):
                    within_domain[domain]["pixel_overlap_" + a + "__" + b] += 1
        for a, b in itertools.combinations(domains, 2):
            if a not in grouped or b not in grouped:
                continue
            pair = pairs[a + "__" + b]
            pair["shared_grayscale_pixel_groups"] += 1
            left = set().union(*(record["splits"] for record in grouped[a]))
            right = set().union(*(record["splits"] for record in grouped[b]))
            for first in SPLITS:
                for second in SPLITS:
                    if first in left and second in right:
                        pair[first + "__" + second] += 1
    return {
        "overall": dict(summary),
        "within_domain": {name: dict(values) for name, values in within_domain.items()},
        "cross_domain_pairs": {name: dict(values) for name, values in pairs.items()},
    }


def exclusion_plan(pixel_index, domains):
    """Return only zero-based nonblank manifest row indices, never sample IDs.

    Evaluation wins over training. Within a split, conflicting masks remove
    all ambiguous rows; equal-label duplicates keep the first domain/row.
    A shared validation/test image keeps only test if labels agree, and
    removes both evaluation copies if labels disagree.
    """
    excluded = {domain: {split: set() for split in SPLITS} for domain in domains}
    order = {domain: index for index, domain in enumerate(domains)}
    for records in pixel_index.values():
        occurrences = [
            (record["domain"], split, row, record["mask_hash"])
            for record in records for split, rows in record["rows"].items() for row in rows
        ]
        evaluation = [item for item in occurrences if item[1] in ("val", "test")]
        if evaluation:
            for domain, split, row, _ in occurrences:
                if split == "train":
                    excluded[domain][split].add(row)
        if {item[1] for item in evaluation} == {"val", "test"}:
            ambiguous = len({item[3] for item in evaluation}) > 1
            for domain, split, row, _ in evaluation:
                if ambiguous or split == "val":
                    excluded[domain][split].add(row)
        for split in SPLITS:
            same_split = [item for item in occurrences if item[1] == split]
            if len(same_split) < 2:
                continue
            same_split.sort(key=lambda item: (order[item[0]], item[2]))
            drop = same_split if len({item[3] for item in same_split}) > 1 else same_split[1:]
            for domain, _, row, _ in drop:
                excluded[domain][split].add(row)
    return {domain: {split: sorted(rows) for split, rows in splits.items()} for domain, splits in excluded.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Parent directory directly containing the requested domain folders")
    parser.add_argument("--domains", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True, help="Aggregate JSON destination")
    parser.add_argument("--exclusions-output", type=Path, help="Optional JSON containing only domain/split and zero-based manifest row indices")
    parser.add_argument("--groups-csv", action="append", default=[], metavar="DOMAIN=PATH")
    parser.add_argument("--group-source", action="append", default=[], metavar="DOMAIN=PROVENANCE")
    args = parser.parse_args()
    if len(args.domains) != len(set(args.domains)):
        parser.error("Domains must be distinct")
    try:
        group_paths = parse_assignments(args.groups_csv, "--groups-csv")
        group_sources = parse_assignments(args.group_source, "--group-source")
    except ValueError as error:
        parser.error(str(error))
    if set(group_paths) != set(group_sources) or set(group_paths) - set(args.domains):
        parser.error("Each supplied group CSV needs a provenance declaration and a requested domain")
    started = time.perf_counter()
    cv2.setNumThreads(1)
    if hasattr(cv2, "setLogLevel"):
        cv2.setLogLevel(0)  # Decoder warnings must not expose sample paths.
    pixel_index = collections.defaultdict(list)
    domain_results = {}
    for domain in args.domains:
        domain_results[domain] = audit_domain(
            args.root, domain, pixel_index,
            Path(group_paths[domain]) if domain in group_paths else None,
            group_sources.get(domain),
        )
        print(json.dumps({"completed_domain": domain, "resolved_pairs": domain_results[domain]["unique_resolved_image_mask_pairs"]}), flush=True)
    report = {
        "audit": "Prepared segmentation domain data integrity and exact grayscale duplication",
        "root": str(args.root),
        "evidence_boundary": "Decoded public segmentation images and masks; exact grayscale equality does not detect near duplicates, related views, or patient overlap. No media, paths to individual samples, patient IDs or per-sample hashes are saved.",
        "hash_definition": "SHA256 of little-endian uint32 height/width, numpy dtype string and decoded contiguous IMREAD_GRAYSCALE bytes",
        "domains": domain_results,
        "duplicates": duplicate_summary(pixel_index, args.domains),
        "seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    if args.exclusions_output is not None:
        args.exclusions_output.parent.mkdir(parents=True, exist_ok=True)
        plan = exclusion_plan(pixel_index, args.domains)
        args.exclusions_output.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "duplicates": report["duplicates"], "seconds": report["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
