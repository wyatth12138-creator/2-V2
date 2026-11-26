"""Reproduce group-disjoint segmentation exclusions from declared metadata.

This tool only reads canonical split lists and a public Sechof case manifest.
It never changes data, moves samples, or writes sample/group identifiers.
Outputs contain aggregate evidence and zero-based nonblank split-row indices.

Priority is TEST, then VAL, then TRAIN. Existing pixel exclusions stay in
force. Filename-inferred groups remain inferred leakage barriers; disjoint
groups alone do not establish verified clinical patient identity.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import itertools
import json
from pathlib import Path, PurePosixPath


SPLITS = ("train", "val", "test")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_key(value):
    return PurePosixPath(value.replace("\\", "/")).name.casefold()


def validate_plan(plan):
    if not isinstance(plan, dict):
        raise ValueError("Exclusions must be a domain-to-split JSON object")
    for domain, splits in plan.items():
        if not isinstance(domain, str) or not isinstance(splits, dict):
            raise ValueError("Invalid domain exclusions schema")
        if set(splits) - set(SPLITS):
            raise ValueError("Exclusions contain an unsupported split")
        for rows in splits.values():
            if not isinstance(rows, list) or any(isinstance(row, bool) or not isinstance(row, int) or row < 0 for row in rows):
                raise ValueError("Excluded row indices must be nonnegative integers")


def semantically_equal(left, right):
    validate_plan(left)
    validate_plan(right)
    for domain in set(left) | set(right):
        for split in SPLITS:
            if set(left.get(domain, {}).get(split, [])) != set(right.get(domain, {}).get(split, [])):
                return False
    return True


def read_case_mapping(path, domain):
    required = {"dataset_name", "task", "privacy", "case_id", "sample_paths", "case_id_source", "case_id_confidence"}
    mapping, groups = {}, set()
    sources, confidences = collections.Counter(), collections.Counter()
    records = 0
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Case manifest does not contain the required Sechof fields")
        for record in reader:
            if record["dataset_name"] != domain or record["task"] != "image_seg" or record["privacy"] != "public":
                continue
            group = record["case_id"].strip()
            samples = json.loads(record["sample_paths"])
            if not group or not isinstance(samples, list) or not samples or any(not isinstance(sample, str) for sample in samples):
                raise ValueError("Malformed public segmentation case metadata")
            records += 1
            groups.add(group)
            sources[record["case_id_source"]] += 1
            confidences[record["case_id_confidence"]] += 1
            for sample in samples:
                key = sample_key(sample)
                if not key or (key in mapping and mapping[key] != group):
                    raise ValueError("The case manifest has an ambiguous sample-to-group mapping")
                mapping[key] = group
    if not mapping:
        raise ValueError("No matching public image-segmentation cases in the supplied manifest")
    return mapping, {
        "matching_source_records": records,
        "distinct_source_groups": len(groups),
        "mapped_source_images": len(mapping),
        "group_source_counts": dict(sources),
        "source_confidence_counts": dict(confidences),
    }


def overlaps(group_sets):
    return {left + "__" + right: len(group_sets[left] & group_sets[right]) for left, right in itertools.combinations(SPLITS, 2)}


def reproduce(args):
    plan = json.loads(args.pixel_exclusions.read_text(encoding="utf-8"))
    validate_plan(plan)
    if args.domain not in plan:
        raise ValueError("Target domain is absent from the supplied pixel-exclusion plan")
    mapping, source_summary = read_case_mapping(args.case_manifest, args.domain)
    root = args.data_root / args.domain
    entries, groups, split_hashes, blank_counts = {}, {}, {}, {}
    previous = {split: set(plan[args.domain].get(split, [])) for split in SPLITS}
    for split in SPLITS:
        path = root / (split + ".txt")
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        entries[split] = [line.strip() for line in lines if line.strip()]
        blank_counts[split] = sum(not line.strip() for line in lines)
        split_hashes[split] = sha256(path)
        groups[split] = [mapping.get(sample_key(entry)) for entry in entries[split]]
        if any(group is None for group in groups[split]):
            raise ValueError("The case manifest does not cover every requested split entry")
        if any(row >= len(entries[split]) for row in previous[split]):
            raise ValueError("A supplied pixel-exclusion row is outside its canonical split")

    before = {split: set(values) for split, values in groups.items()}
    eligible = {
        split: {group for index, group in enumerate(groups[split]) if index not in previous[split]}
        for split in SPLITS
    }
    test_groups = eligible["test"]
    validation_drop = previous["val"] | {index for index, group in enumerate(groups["val"]) if group in test_groups}
    validation_groups = {group for index, group in enumerate(groups["val"]) if index not in validation_drop}
    training_drop = previous["train"] | {
        index for index, group in enumerate(groups["train"])
        if group in test_groups | validation_groups
    }
    plan[args.domain] = {
        "train": sorted(training_drop), "val": sorted(validation_drop),
        "test": sorted(previous["test"]),
    }
    retained = {
        split: {group for index, group in enumerate(groups[split]) if index not in set(plan[args.domain][split])}
        for split in SPLITS
    }
    after = overlaps(retained)
    if any(after.values()):
        raise RuntimeError("Group-disjointness verification failed")
    if set.union(*retained.values()) != set.union(*eligible.values()):
        raise RuntimeError("Group prioritization unexpectedly discarded an eligible group")
    verified_match = None
    if args.verify_against is not None:
        existing = json.loads(args.verify_against.read_text(encoding="utf-8"))
        verified_match = semantically_equal(plan, existing)
        if not verified_match:
            raise RuntimeError("Reproduced exclusions differ from the requested reference plan")

    report = {
        "scheme": "B: retain TEST, then VAL, exclude lower-priority rows sharing a supplied case group",
        "domain": args.domain,
        "data_root": str(args.data_root),
        "source_manifest": str(args.case_manifest),
        "source_manifest_sha256": sha256(args.case_manifest),
        "input_pixel_exclusions_sha256": sha256(args.pixel_exclusions),
        "split_manifest_sha256": split_hashes,
        "blank_manifest_lines": blank_counts,
        "source_summary": source_summary,
        "mapped_split_entries": sum(len(values) for values in entries.values()),
        "unmapped_split_entries": 0,
        "group_overlap_before": overlaps(before),
        "group_overlap_after": after,
        "eligible_groups_after_pixel_exclusions": len(set.union(*eligible.values())),
        "retained_distinct_groups": len(set.union(*retained.values())),
        "retained_groups_per_split": {split: len(values) for split, values in retained.items()},
        "retained_images_per_split": {split: len(entries[split]) - len(plan[args.domain][split]) for split in SPLITS},
        "all_exclusion_counts": {domain: {split: len(set(rows)) for split, rows in splits.items()} for domain, splits in plan.items()},
        "index_semantics": "Zero-based rows after discarding blank lines, matching dataset ids; no sample/group identifiers are persisted",
        "verified_clinical_patient_identity": False,
        "limitation": "Case identity inherits the source manifest provenance. Filename-inferred medium-confidence groups are leakage barriers, not independently verified clinical patient identifiers.",
        "verified_against_sha256": sha256(args.verify_against) if args.verify_against is not None else None,
        "verified_exclusion_semantic_match": verified_match,
        "output_exclusions": str(args.output_exclusions),
    }
    args.output_exclusions.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_exclusions.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    args.output_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output_exclusions": str(args.output_exclusions),
        "output_report": str(args.output_report),
        "semantic_match": verified_match,
        "retained_images_per_split": report["retained_images_per_split"],
        "group_overlap_after": after,
        "retained_groups": report["retained_distinct_groups"],
    }), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixel-exclusions", type=Path, required=True)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True, help="Parent directly containing the domain folder")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--output-exclusions", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--verify-against", type=Path, help="Require semantic equality with an existing exclusions JSON")
    args = parser.parse_args()
    if ":" in args.domain or args.domain in (".", "..") or len(PurePosixPath(args.domain.replace("\\", "/")).parts) != 1:
        parser.error("domain must be a single directory name")
    reproduce(args)


if __name__ == "__main__":
    main()
