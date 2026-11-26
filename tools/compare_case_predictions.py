"""Compare local test.py case_predictions.csv files with paired patient bootstrap.

Positive Brier improvement and net correction favor the candidate. One locked
threshold is applied to both files; this script never chooses a test threshold.
Intervals resample patients, retaining all their cases, and estimate case-weighted
means. Patient IDs must be stable across files and globally consistent across
datasets. CSV contents do not establish clinical provenance: synthetic fixtures
verify software only and must never be presented as medical evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED = {"task", "dataset", "case_id", "patient_id", "label", "probability"}
Key = tuple[str, str, str]


def load_predictions(path: Path) -> dict[Key, dict[str, Any]]:
    """Read the exact classification export schema, rejecting silent pairing loss."""
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() != ".csv" or not path.is_file():
        raise ValueError(f"Expected an existing local CSV file: {path}")
    rows: dict[Key, dict[str, Any]] = {}
    case_patients: dict[tuple[str, str], str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        if len(headers) != len(set(headers)) or not REQUIRED.issubset(headers):
            raise ValueError(f"CSV requires unique columns {sorted(REQUIRED)}: {path}")
        for line, raw in enumerate(reader, start=2):
            if None in raw or any(raw.get(name) is None for name in REQUIRED):
                raise ValueError(f"Malformed CSV row {line}: {path}")
            row = {name: raw[name].strip() for name in REQUIRED}
            if any(not row[name] for name in ("task", "dataset", "case_id", "patient_id")):
                raise ValueError(f"Empty case/task/dataset/patient identifier on row {line}: {path}")
            if row["patient_id"].lower() in {"none", "null", "nan", "n/a", "unknown"}:
                raise ValueError(f"Missing stable patient_id on row {line}: {path}")
            if row["label"] not in {"0", "1"}:
                raise ValueError(f"Nonbinary label on row {line}: {path}")
            try:
                probability = float(row["probability"])
            except ValueError as exc:
                raise ValueError(f"Invalid probability on row {line}: {path}") from exc
            if not np.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError(f"Probability outside finite [0,1] on row {line}: {path}")
            key = (row["task"], row["dataset"], row["case_id"])
            if key in rows:
                raise ValueError(f"Duplicate task/dataset/case key {key}: {path}")
            identity = (row["dataset"], row["case_id"])
            if identity in case_patients and case_patients[identity] != row["patient_id"]:
                raise ValueError(f"Case changes patient identity across tasks: {identity}")
            case_patients[identity] = row["patient_id"]
            rows[key] = {"patient_id": row["patient_id"], "label": int(row["label"]), "probability": probability}
    if not rows:
        raise ValueError(f"Empty prediction CSV: {path}")
    return rows


def _summarize(keys, baseline, candidate, threshold, bootstrap, rng):
    labels = np.asarray([baseline[key]["label"] for key in keys], dtype=np.int64)
    base_prob = np.asarray([baseline[key]["probability"] for key in keys])
    cand_prob = np.asarray([candidate[key]["probability"] for key in keys])
    patients = [baseline[key]["patient_id"] for key in keys]
    _, cluster_index = np.unique(patients, return_inverse=True)
    n_patients = int(cluster_index.max()) + 1
    base_pred, cand_pred = base_prob >= threshold, cand_prob >= threshold
    base_error, cand_error = base_pred != labels, cand_pred != labels
    positive = base_error & ~cand_error
    negative = ~base_error & cand_error
    brier_base, brier_cand = (base_prob - labels) ** 2, (cand_prob - labels) ** 2
    paired_values = np.column_stack((brier_base - brier_cand, base_error.astype(float) - cand_error))
    estimates = paired_values.mean(axis=0)
    interval = [None, None]
    warnings = []
    if n_patients >= 2:
        cluster_sizes = np.bincount(cluster_index)
        cluster_sums = np.stack([np.bincount(cluster_index, weights=paired_values[:, column]) for column in range(2)], axis=1)
        draws = np.empty((bootstrap, 2), dtype=float)
        for iteration in range(bootstrap):
            sampled = rng.integers(0, n_patients, size=n_patients)
            draws[iteration] = cluster_sums[sampled].sum(axis=0) / cluster_sizes[sampled].sum()
        interval = np.quantile(draws, [0.025, 0.975], axis=0).T.tolist()
    else:
        warnings.append("Only one patient cluster: no inferential bootstrap interval is reported.")
    if n_patients < 20:
        warnings.append("Few patient clusters: percentile intervals can be unstable; fixture results are not clinical evidence.")
    malignant = labels == 1
    return {
        "num_cases": len(keys),
        "num_patients": n_patients,
        "num_malignant_cases": int(malignant.sum()),
        "num_benign_cases": int((~malignant).sum()),
        "baseline_brier": float(brier_base.mean()),
        "candidate_brier": float(brier_cand.mean()),
        "brier_improvement": {"estimate": float(estimates[0]), "ci95": interval[0]},
        "net_correction": {"rate": float(estimates[1]), "ci95": interval[1],
                           "positive_events": int(positive.sum()), "negative_events": int(negative.sum()),
                           "net_events": int(positive.sum() - negative.sum())},
        "baseline_error_count": int(base_error.sum()),
        "candidate_error_count": int(cand_error.sum()),
        "baseline_malignant_missed": int((malignant & ~base_pred).sum()),
        "candidate_malignant_missed": int((malignant & ~cand_pred).sum()),
        "malignant_misses_corrected": int((malignant & positive).sum()),
        "malignant_misses_introduced": int((malignant & negative).sum()),
        "warnings": warnings,
    }


def compare_predictions(baseline, candidate, threshold=0.5, bootstrap=10000, seed=42, data_kind="unverified"):
    """Validate exact paired cohorts, then summarize overall and by task/dataset."""
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and within [0,1]")
    if isinstance(bootstrap, bool) or not isinstance(bootstrap, (int, np.integer)) or bootstrap < 100:
        raise ValueError("bootstrap must be an integer >= 100")
    if data_kind not in {"unverified", "real", "synthetic"}:
        raise ValueError("data_kind must be unverified, real, or synthetic")
    if set(baseline) != set(candidate):
        missing = len(set(baseline) - set(candidate))
        extra = len(set(candidate) - set(baseline))
        raise ValueError(f"Paired task/dataset/case keys differ: missing={missing}, extra={extra}")
    if not baseline:
        raise ValueError("No paired cases")
    keys = sorted(baseline)
    for key in keys:
        if baseline[key]["label"] != candidate[key]["label"]:
            raise ValueError(f"Paired labels differ for {key}")
        if not baseline[key]["patient_id"] or baseline[key]["patient_id"] != candidate[key]["patient_id"]:
            raise ValueError(f"Paired patient IDs are missing or inconsistent for {key}")
    rng = np.random.default_rng(seed)
    strata = sorted({key[:2] for key in keys})
    return {
        "schema_version": 1,
        "threshold": float(threshold),
        "threshold_policy": "Same threshold for both models; must be locked on development data before testing.",
        "positive_direction": "baseline minus candidate; positive favors candidate",
        "estimand": "case-weighted paired means; overall weights strata by their case counts",
        "bootstrap": {"unit": "patient_id", "replicates": int(bootstrap), "seed": int(seed),
                      "interval": "two-sided 95% percentile; paired whole-patient cluster resampling"},
        "data_kind": data_kind,
        "evidence_status": "software_verification_only" if data_kind == "synthetic" else "provenance_not_independently_verified",
        "interpretation": "Synthetic examples verify computation only and cannot support medical claims. Local CSV scores do not verify real patient provenance, independent splitting, or external validation.",
        "overall": _summarize(keys, baseline, candidate, threshold, bootstrap, rng),
        "by_task_dataset": [
            {"task": task, "dataset": dataset, **_summarize([key for key in keys if key[:2] == (task, dataset)], baseline, candidate, threshold, bootstrap, rng)}
            for task, dataset in strata
        ],
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON; explicitly choose the research/process directory.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-kind", choices=("unverified", "real", "synthetic"), default="unverified", help="Provenance declaration only; synthetic is software verification, never medical evidence.")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        output = args.output.expanduser().resolve()
        inputs = [path.expanduser().resolve() for path in (args.baseline, args.candidate)]
        if output in inputs or output.suffix.lower() != ".json":
            raise ValueError("output must be a .json file distinct from the input CSVs")
        report = compare_predictions(load_predictions(inputs[0]), load_predictions(inputs[1]), args.threshold, args.bootstrap, args.seed, args.data_kind)
        report["sources"] = {"baseline": str(inputs[0]), "candidate": str(inputs[1])}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Paired comparison written to {output}")
    print(report["interpretation"])


if __name__ == "__main__":
    main()
