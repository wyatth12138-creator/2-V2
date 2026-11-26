"""Summarize complete, protocol-matched benchmark_domain_transfer JSON reports.

Example: python -B tools/summarize_domain_transfer.py --reports seed42.json
seed2026.json seed3407.json --output summary.json

Uses only the standard library, reads local JSON, and never imports the model.
Seed summaries are descriptive: no patient confidence intervals or significance
tests can be recovered from aggregate Dice. Legacy reports contain paths rather
than immutable run-time split/checkpoint hashes; matching those paths and counts
does not establish historical data identity. Seed-dependent evaluation subsets
cannot be paired without an explicit, identical eval_manifest_sha256 mapping.
Those digests must describe the actual selected evaluation rows, not just the
larger split from which a seed-dependent subset was sampled.
Synthetic reports verify software only and must not be used as research evidence.
--select-on-val applies an empirical zero-margin gate to complete source-only
VAL reports; it is not a statistical non-inferiority test or a test-set selector.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import fmean


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _number(value, name, *, low=None, high=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{name} outside allowed range")
    return value


def _integer(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _names(value, name, *, minimum=0):
    if (not isinstance(value, list) or len(value) < minimum
            or any(not isinstance(x, str) or not x.strip() for x in value)
            or len(value) != len(set(value))):
        raise ValueError(f"{name} requires unique nonempty names")
    return value


def _summary(values):
    values = list(values)
    return {"n": len(values), "mean": fmean(values), "min": min(values), "max": max(values)} if values else None


def _gradient_summary(run, domains, steps, arm, expected_weights):
    stats = run.get("gradient_statistics", {})
    required = ("mode", "weights", "pairwise_scope", "pairwise", "comparable_pairs",
                "conflicting_pairs", "projection_steps", "update_dot", "shared_adam_descent_dot")
    if not isinstance(stats, dict) or any(not isinstance(stats.get(k), list) or len(stats[k]) != steps for k in required):
        raise ValueError(f"{arm}: missing or truncated gradient statistics")
    pair_values = {f"{a} | {b}": [] for a, b in itertools.combinations(domains, 2)}
    raw_update = {d: [] for d in domains}
    adam_update = {d: [] for d in domains}
    comparisons = conflicts = projections = undefined = 0
    for step in range(steps):
        mode = "pcgrad" if arm == "pcgrad" else "weighted_mean"
        if stats["mode"][step] != mode or stats["pairwise_scope"][step] != "unprotected_common_parameter_blocks":
            raise ValueError(f"{arm}: inconsistent gradient mode or support")
        weights = stats["weights"][step]
        if (not isinstance(weights, list) or len(weights) != len(domains)
                or any(not math.isclose(_number(w, "domain weight", low=0), e, abs_tol=1e-10)
                       for w, e in zip(weights, expected_weights))):
            raise ValueError(f"{arm}: incorrect domain weights")
        seen = set()
        comparable = conflicting = 0
        pairs = stats["pairwise"][step]
        if not isinstance(pairs, list):
            raise ValueError(f"{arm}: malformed pairwise statistics")
        for pair in pairs:
            i = _integer(pair["domain_i"], "domain_i")
            j = _integer(pair["domain_j"], "domain_j")
            if not 0 <= i < j < len(domains) or (i, j) in seen:
                raise ValueError(f"{arm}: duplicate or invalid domain pair")
            seen.add((i, j))
            dot = _number(pair["dot"], "gradient dot")
            cosine = pair["cosine"]
            if cosine is None:
                undefined += 1
            else:
                _number(cosine, "gradient cosine", low=-1, high=1)
                if (dot < 0) != (cosine < 0):
                    raise ValueError(f"{arm}: dot/cosine signs disagree")
                pair_values[f"{domains[i]} | {domains[j]}"].append(cosine)
                comparable += 1
                conflicting += int(dot < 0)
        if len(seen) != len(domains) * (len(domains) - 1) // 2:
            raise ValueError(f"{arm}: incomplete pairwise domain coverage")
        if (_integer(stats["comparable_pairs"][step], "comparable_pairs") != comparable
                or _integer(stats["conflicting_pairs"][step], "conflicting_pairs") != conflicting):
            raise ValueError(f"{arm}: conflict counts disagree with original gradients")
        comparisons += comparable
        conflicts += conflicting
        projections += _integer(stats["projection_steps"][step], "projection_steps")
        for field, destination in (("update_dot", raw_update), ("shared_adam_descent_dot", adam_update)):
            row = stats[field][step]
            if not isinstance(row, list) or len(row) != len(domains):
                raise ValueError(f"{arm}: inconsistent {field} domain ordering")
            for domain, value in zip(domains, row):
                destination[domain].append(_number(value, field))
    describe = lambda values: {**_summary(values), "negative_steps": sum(x < 0 for x in values)}
    return {
        "steps": steps, "original_comparable_pairs": comparisons,
        "original_conflicting_pairs": conflicts, "original_undefined_cosines": undefined,
        "original_conflict_fraction": conflicts / comparisons if comparisons else None,
        "original_pairwise_cosine": {pair: _summary(values) for pair, values in pair_values.items()},
        "projection_steps": projections,
        "merged_gradient_dot_all_support": {d: describe(v) for d, v in raw_update.items()},
        "actual_adam_descent_dot_common_unprotected_support": {d: describe(v) for d, v in adam_update.items()},
    }


def load_report(path):
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() != ".json" or not path.is_file():
        raise ValueError(f"Expected a local JSON report: {path}")
    raw = path.read_bytes()
    report = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_object,
                        parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON: {x}")))
    if not isinstance(report, dict) or report.get("complete") is not True:
        raise ValueError(f"Report is not complete: {path}")
    cfg = report["configuration"]
    if not isinstance(cfg, dict):
        raise ValueError("configuration must be a JSON object")
    source = _names(cfg["source"], "source", minimum=2)
    held_out = _names(cfg["held_out"], "held_out")
    if set(source) & set(held_out) or cfg["eval_split"] not in {"val", "test"}:
        raise ValueError("Source/held-out overlap or invalid evaluation split")
    seed = _integer(cfg["seed"], "seed")
    steps = _integer(cfg["steps"], "steps", minimum=1)
    _number(cfg["base_lr"], "base_lr", low=0)
    if cfg.get("eval_limit") is not None:
        _integer(cfg["eval_limit"], "eval_limit", minimum=1)
    arms = _names(cfg["arms"], "arms", minimum=2)
    if not {"single", "pooled"}.issubset(arms) or not set(arms) <= {"single", "pooled", "balanced", "pcgrad"}:
        raise ValueError("Require single and pooled, with optional balanced/pcgrad")
    for key in ("data_root", "sam", "resnet"):
        if not isinstance(cfg.get(key), str) or not cfg[key].strip():
            raise ValueError(f"Missing configured {key}")
    domains = source + held_out
    if (set(report["train_eligible"]) != set(source) or set(report["eval_images"]) != set(domains)
            or set(report["excluded_or_mask_empty"]) != set(domains)):
        raise ValueError("Training/evaluation domain counts do not match configuration")
    for values in (report["train_eligible"], report["eval_images"]):
        for domain, count in values.items():
            _integer(count, f"{domain} sample count", minimum=1)
    for domain, counts in report["excluded_or_mask_empty"].items():
        if not isinstance(counts, dict) or cfg["eval_split"] not in counts or (domain in source and "train" not in counts):
            raise ValueError(f"{domain}: missing split exclusion counts")
        for count in counts.values():
            _integer(count, "excluded_or_mask_empty")
    expected = {f"single:{d}" for d in source} | (set(arms) - {"single"})
    if set(report["runs"]) != expected:
        raise ValueError("Declared arms and completed runs disagree")
    if not isinstance(report.get("protocol"), str) or not report["protocol"].strip():
        raise ValueError("Missing protocol description")
    if not isinstance(report.get("patient_scope"), str) or not report["patient_scope"].strip():
        raise ValueError("Missing evaluation-unit scope")
    metrics, gradient = {}, {}
    for arm, run in report["runs"].items():
        if not isinstance(run, dict):
            raise ValueError(f"{arm}: run must be a JSON object")
        if set(run["domain_metrics"]) != set(domains):
            raise ValueError(f"{arm}: evaluation domains differ")
        metrics[arm] = {}
        for domain, value in run["domain_metrics"].items():
            metrics[arm][domain] = _number(value["dice"], "Dice", low=0, high=1)
            count = _integer(value["image_count"], "image_count", minimum=1)
            if count != report["eval_images"][domain]:
                raise ValueError(f"{arm}: evaluation image_count changed")
            if "empty_predictions" in value:
                if _integer(value["empty_predictions"], "empty_predictions") > count:
                    raise ValueError(f"{arm}: too many empty predictions")
        for key, derived in (("source_macro_dice", fmean(metrics[arm][d] for d in source)),
                             ("source_worst_dice", min(metrics[arm][d] for d in source))):
            if not math.isclose(_number(run[key], key), derived, abs_tol=1e-10):
                raise ValueError(f"{arm}: inconsistent stored {key}")
        active = [arm.removeprefix("single:")] if arm.startswith("single:") else source
        if set(run["images_drawn_per_training_domain"]) != set(active):
            raise ValueError(f"{arm}: training domains differ")
        if any(_integer(run["images_drawn_per_training_domain"][d], "exposure") != steps for d in active):
            raise ValueError(f"{arm}: domain exposure does not match update count")
        if set(run["observed_train_images_per_domain"]) != set(active):
            raise ValueError(f"{arm}: observed training domains differ")
        for domain in active:
            observed = _integer(run["observed_train_images_per_domain"][domain], "unique training images", minimum=1)
            if observed > min(steps, report["train_eligible"][domain]):
                raise ValueError(f"{arm}: impossible unique training exposure")
        total = sum(report["train_eligible"][d] for d in active)
        weights = [report["train_eligible"][d] / total if arm == "pooled" else 1 / len(active) for d in active]
        gradient[arm] = _gradient_summary(run, active, steps, arm, weights)
    for arm, run in report["runs"].items():
        for domain, observed in run["observed_train_images_per_domain"].items():
            reference = report["runs"][f"single:{domain}"]["observed_train_images_per_domain"][domain]
            if observed != reference:
                raise ValueError(f"{arm}: unique training exposure differs from single:{domain}")
        if "delta_dice_vs_single" in run:
            if set(run["delta_dice_vs_single"]) != set(source):
                raise ValueError(f"{arm}: incomplete stored joint-single deltas")
            for d in source:
                actual = metrics[arm][d] - metrics[f"single:{d}"][d]
                if not math.isclose(_number(run["delta_dice_vs_single"][d], "joint-single"), actual, abs_tol=1e-10):
                    raise ValueError(f"{arm}: inconsistent stored joint-single delta")
    signature = {"configuration": {k: v for k, v in cfg.items() if k not in {"seed", "output"}},
                 **{k: report[k] for k in ("protocol", "patient_scope", "train_eligible", "eval_images", "excluded_or_mask_empty")},
                 "eval_manifest_sha256": report.get("eval_manifest_sha256"),
                 "provenance": report.get("provenance")}
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "seed": seed,
            "signature": signature, "metrics": metrics, "gradient": gradient}


def _select_on_val(rows, candidates, *, data_kind):
    """Select once by worst seed/domain gain, macro Dice, then fixed arm order."""
    order = ("pooled", "balanced", "pcgrad")
    scores = []
    for arm in order:
        if arm not in candidates:
            continue
        gains = [(row["seed"], domain, gain) for row in rows
                 for domain, gain in row["arms"][arm]["joint_minus_same_domain_single"].items()]
        minimum = min(gain for _, _, gain in gains)
        scores.append({
            "arm": arm, "minimum_gain_all_seed_source_pairs": minimum,
            "mean_source_macro_dice": fmean(row["arms"][arm]["source_macro_dice"] for row in rows),
            "eligible": minimum >= 0.0,
            "limiting_seed_domains": [{"seed": seed, "domain": domain} for seed, domain, gain in gains if gain == minimum],
        })
    eligible = [score for score in scores if score["eligible"]]
    winner = max(eligible, key=lambda score: (score["minimum_gain_all_seed_source_pairs"],
                 score["mean_source_macro_dice"], -order.index(score["arm"]))) if eligible else None
    return {
        "status": "selected_joint" if winner else "abstain_no_joint",
        "selected_joint": winner["arm"] if winner else None,
        "empirical_noninferiority_margin": 0.0,
        "selection_basis": "min joint-minus-same-domain-single over ALL seed x source pairs >= 0; maximize minimum gain, then mean source macro Dice, then fixed arm order",
        "predefined_arm_order": list(order), "candidate_scores": scores,
        "single_references": [{"seed": row["seed"], "dice_by_source": row["same_domain_single_reference"]} for row in rows],
        "fallback": "retain the corresponding single-domain reference for each source",
        "evidence_limits": [
            "Full source VAL only; no TEST or held-out-domain score participates in this decision.",
            "The zero-margin gate is descriptive empirical screening, not statistical non-inferiority, a confidence bound or a clinical guarantee.",
            f"Only {len(rows)} training seed(s); a single-seed selection is provisional and does not establish seed robustness.",
            "Selection does not make previously viewed TEST data independent; freeze the decision before new confirmatory evaluation.",
            "Legacy report paths/counts do not certify historical split, checkpoint, code or patient identity.",
            "Synthetic selection is software verification only." if data_kind == "synthetic" else "Declared data kind does not establish clinical provenance.",
        ],
    }


def summarize(reports, *, data_kind="unverified", select_on_val=False):
    if not reports:
        raise ValueError("At least one report is required")
    if len({r["seed"] for r in reports}) != len(reports):
        raise ValueError("Duplicate seeds cannot be treated as independent paired runs")
    signature = reports[0]["signature"]
    for report in reports[1:]:
        if report["signature"] != signature:
            raise ValueError("Reports differ in training/evaluation configuration, splits, counts, protocol or provenance")
    cfg = signature["configuration"]
    if select_on_val and (cfg["eval_split"] != "val" or cfg["held_out"]
                          or "eval_limit" not in cfg or cfg["eval_limit"] is not None):
        raise ValueError("--select-on-val requires complete source-only VAL: eval_split=val, held_out=[], eval_limit=null (no evaluation subsampling)")
    fingerprint = signature["eval_manifest_sha256"]
    if fingerprint is not None:
        if (not isinstance(fingerprint, dict) or set(fingerprint) != set(cfg["source"] + cfg["held_out"])
                or any(not isinstance(x, str) or len(x) != 64
                       or any(c not in "0123456789abcdefABCDEF" for c in x) for x in fingerprint.values())):
            raise ValueError("eval_manifest_sha256 must map each evaluation domain to a SHA256 digest")
    if len(reports) > 1 and cfg.get("eval_limit") is not None and fingerprint is None:
        raise ValueError("eval_limit samples by seed; multiple seeds require identical eval_manifest_sha256 cohort fingerprints")
    source, held_out = cfg["source"], cfg["held_out"]
    domains = source + held_out
    joint = [arm for arm in cfg["arms"] if arm != "single"]
    rows = []
    for report in sorted(reports, key=lambda r: r["seed"]):
        metric = report["metrics"]
        row = {"seed": report["seed"], "report": report["path"], "report_sha256": report["sha256"],
               "same_domain_single_reference": {d: metric[f"single:{d}"][d] for d in source}, "arms": {}}
        strongest = {d: max(metric[f"single:{s}"][d] for s in source) for d in held_out}
        row["held_out_observed_max_single_source_per_seed"] = {
            d: {"dice": strongest[d], "winner_arms": [f"single:{s}" for s in source
                                                        if metric[f"single:{s}"][d] == strongest[d]]} for d in held_out}
        for arm, values in metric.items():
            info = {"dice_by_domain": values, "optimization": report["gradient"][arm]}
            if arm in joint:
                delta = {d: values[d] - metric[f"single:{d}"][d] for d in source}
                info.update(source_macro_dice=fmean(values[d] for d in source),
                            source_worst_dice=min(values[d] for d in source),
                            joint_minus_same_domain_single=delta, minimum_source_gain=min(delta.values()),
                            negative_transfer_domain_count=sum(v < 0 for v in delta.values()),
                            minus_pooled_by_domain={d: values[d] - metric["pooled"][d] for d in domains},
                            source_macro_minus_pooled=fmean(values[d] - metric["pooled"][d] for d in source),
                            source_worst_minus_pooled=min(values[d] for d in source) - min(metric["pooled"][d] for d in source),
                            held_out_minus_observed_max_single_source={d: values[d] - strongest[d] for d in held_out})
            row["arms"][arm] = info
        rows.append(row)
    aggregate = {}
    for arm in joint:
        runs = [row["arms"][arm] for row in rows]
        item = {key: _summary(run[key] for run in runs)
                for key in ("source_macro_dice", "source_worst_dice", "minimum_source_gain", "negative_transfer_domain_count",
                            "source_macro_minus_pooled", "source_worst_minus_pooled")}
        for key, names in (("dice_by_domain", domains), ("joint_minus_same_domain_single", source),
                           ("minus_pooled_by_domain", domains), ("held_out_minus_observed_max_single_source", held_out)):
            item[key] = {d: _summary(run[key][d] for run in runs) for d in names}
        item["negative_transfer_seed_domain_events"] = sum(run["negative_transfer_domain_count"] for run in runs)
        item["negative_transfer_seeds_by_domain"] = {d: sum(run["joint_minus_same_domain_single"][d] < 0 for run in runs) for d in source}
        item["optimization"] = {
            "original_conflict_fraction_across_seeds": _summary(
                run["optimization"]["original_conflict_fraction"] for run in runs
                if run["optimization"]["original_conflict_fraction"] is not None),
            "actual_adam_negative_steps_by_domain": {
                d: _summary(run["optimization"]["actual_adam_descent_dot_common_unprotected_support"][d]["negative_steps"]
                            for run in runs) for d in source},
        }
        aggregate[arm] = item
    result = {
        "schema": "uiv_us_domain_transfer_summary_v1", "data_kind": data_kind,
        "synthetic_is_software_verification_only": data_kind == "synthetic",
        "seeds": [row["seed"] for row in rows], "protocol": signature,
        "validation": {"complete_protocol_counts_exposure_and_seed_checks": True,
                       "additional_provenance_record_present": signature["provenance"] is not None,
                       "explicit_evaluation_cohort_fingerprints_present": fingerprint is not None},
        "interpretation": [
            "Positive Dice differences favor the named joint arm. Values are on the 0-to-1 scale.",
            "Held-out max(single) is an observed evaluation-set oracle comparator per seed, not a deployable or validation-selected model.",
            "Negative-transfer counts use raw delta < 0, without significance or clinical-effect claims.",
            "Mean/min/max summarize paired seed differences; no patient intervals or hypothesis tests are estimated.",
            "A positive <gradient, theta_before-theta_after> is first-order predicted descent on the logged support, not a generalization guarantee.",
            "Legacy configuration paths and counts do not certify historical split, checkpoint, code or patient identity; input provenance remains unverified.",
            "Synthetic reports are software fixtures, never clinical or algorithm-effectiveness evidence.",
        ],
        "per_seed": rows, "across_seed_paired_summaries": aggregate,
    }
    if select_on_val:
        result["val_selection"] = _select_on_val(rows, joint, data_kind=data_kind)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--select-on-val", action="store_true",
                        help="Gate joint candidates using full source-only VAL; abstain if every candidate has any negative seed/domain gain.")
    parser.add_argument("--data-kind", choices=("unverified", "real", "synthetic"), default="unverified",
                        help="Operator declaration only; JSON cannot establish clinical provenance.")
    args = parser.parse_args()
    try:
        output = args.output.expanduser().resolve()
        if output in {p.expanduser().resolve() for p in args.reports}:
            raise ValueError("Output must not overwrite an input report")
        result = summarize([load_report(p) for p in args.reports], data_kind=args.data_kind,
                           select_on_val=args.select_on_val)
        payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(f"Summarized {len(result['seeds'])} seed(s): {output}")


if __name__ == "__main__":
    main()
