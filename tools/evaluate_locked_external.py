"""Evaluate the source-VAL-selected checkpoint on explicitly locked external cohorts."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.uiv_us import build_uiv_us_vit_b
from tools.benchmark_domain_transfer import TASK, eligible_dataset, evaluate
from tools.summarize_domain_transfer import load_report, summarize
from train import build_parser
from training.checkpoint import load_checkpoint
from training.data_splits import make_transforms
from training.run_selection import require_private_path
from utils.config import get_config
from utils.loss_functions.sam_loss import get_criterion


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def paired_interval(candidate, reference):
    left, right = np.asarray(candidate), np.asarray(reference)
    if left.shape != right.shape or left.ndim != 1 or not left.size:
        raise ValueError("Models must be paired over the same ordered external images")
    differences = left - right
    rng = np.random.default_rng(17819)
    sampled = rng.integers(0, len(differences), size=(2000, len(differences)))
    lower, upper = np.quantile(differences[sampled].mean(axis=1), (0.025, 0.975))
    return {"mean_delta_dice": float(differences.mean()), "image_bootstrap_95": [float(lower), float(upper)]}


def validated_checkpoints(loaded, chosen, sources):
    frozen = {}
    required = [f"single:{domain}" for domain in sources] + [chosen]
    for report in loaded:
        report_path = Path(report["path"])
        source_bytes = report_path.read_bytes()
        if hashlib.sha256(source_bytes).hexdigest() != report["sha256"]:
            raise ValueError(f"Source VAL report changed after selection: {report_path}")
        data = json.loads(source_bytes)
        location = data["configuration"].get("checkpoint_dir")
        if not location:
            raise ValueError(f"Source VAL report has no frozen checkpoint directory: {report_path}")
        checkpoint_root = require_private_path(Path(location), "Source checkpoint directory")
        locked = {}
        for arm in required:
            evidence = data["runs"][arm].get("checkpoint")
            if not isinstance(evidence, dict) or not {"path", "sha256"} <= evidence.keys():
                raise ValueError(f"Missing validated state for seed {report['seed']}, arm {arm}")
            path = require_private_path(Path(evidence["path"]), "External checkpoint")
            if not path.is_relative_to(checkpoint_root) or sha256(path) != evidence["sha256"]:
                raise ValueError(f"Frozen state path or SHA-256 mismatch for seed {report['seed']}, arm {arm}")
            locked[arm] = {**evidence, "path": str(path)}
        frozen[str(report["seed"])] = locked
    return frozen


def evaluation_reports(loaded, requested_seeds=None):
    """Filter inference seeds AFTER every report has participated in selection.

    Default inference preserves the original report order. An explicit subset
    must be a nonempty list of unique integers present in the complete source
    VAL inputs. This changes checkpoint validation/inference only, never the
    population of reports used to select the joint arm.
    """
    if requested_seeds is None:
        return loaded
    if (
        not requested_seeds or any(type(seed) is not int for seed in requested_seeds)
        or len(set(requested_seeds)) != len(requested_seeds)
    ):
        raise ValueError("--evaluate-seeds requires a nonempty list of unique integer seeds")
    available = {report["seed"] for report in loaded}
    unknown = set(requested_seeds) - available
    if unknown:
        raise ValueError(f"--evaluate-seeds contains seeds absent from complete source VAL reports: {sorted(unknown)}")
    requested = set(requested_seeds)
    return [report for report in loaded if report["seed"] in requested]


def open_locked_cohorts(override_files, exclusions_path, held_out, selection):
    dataset_root = Path(selection["protocol"]["configuration"]["data_root"])
    opt = get_config("CE")
    opt.data_path = str(require_private_path(dataset_root, "Data root"))
    opt.data_roots = {TASK: opt.data_path}
    model_args = build_parser().parse_args(["--temporal_mode", "baseline", "--no-warmup"])
    model_args.pretrained_ckpt = None
    model_args.resnet_ckpt = None
    _, transform, _, _ = make_transforms(opt, model_args)
    overrides = {}
    for file in override_files:
        provided = json.loads(file.read_text(encoding="utf-8"))
        if not isinstance(provided, dict) or set(overrides) & set(provided):
            raise ValueError("Each external cohort may have exactly one override")
        overrides.update(provided)
    if set(overrides) != set(held_out):
        raise ValueError("Exactly the requested external datasets must have explicit overrides")
    for domain, splits in overrides.items():
        if set(splits) != {"train", "val", "test"} or splits["train"] or splits["val"]:
            raise ValueError(f"External override {domain} must have only a TEST cohort")
        rows = splits["test"]
        if not rows or any(not isinstance(entry, str) or not entry.strip() for entry in rows) or len(set(rows)) != len(rows):
            raise ValueError(f"External TEST override {domain} must contain unique sample entries")
    exclusions = json.loads(exclusions_path.read_text(encoding="utf-8")) if exclusions_path else {}
    datasets = {}
    for domain in held_out:
        datasets[domain], removed = eligible_dataset(opt, domain, "test", transform, exclusions, overrides)
        if removed:
            raise ValueError(f"Locked cohort {domain} changed during eligibility filtering")
    return datasets, model_args, opt


def evaluate_frozen(frozen, chosen, datasets, model_args, opt):
    criterion = get_criterion(opt=opt)
    reports = {}
    for seed, arms in frozen.items():
        evaluated = {}
        for arm, state in arms.items():
            print(f"Evaluating seed {seed}, {arm}: {', '.join(datasets)}", flush=True)
            model = build_uiv_us_vit_b(model_args)
            load_checkpoint(model, state["path"])
            model = model.requires_grad_(False).eval().to(opt.device)
            evaluated[arm] = evaluate(model, datasets, criterion, opt, retain_dice=True)
            del model
            torch.cuda.empty_cache()
        output = {"arms": {}}
        for arm, metrics in evaluated.items():
            output["arms"][arm] = {
                domain: {k: v for k, v in values.items() if k != "per_image_dice"}
                for domain, values in metrics.items()
            }
        output["paired_comparisons"] = {}
        for domain in datasets:
            targets = {arm: values[domain] for arm, values in evaluated.items()}
            references = [arm for arm in targets if arm != chosen]
            best = max(references, key=lambda arm: targets[arm]["dice"])
            output["paired_comparisons"][domain] = {
                "selected_joint": chosen,
                "delta_dice_vs_each_single": {
                    arm: paired_interval(targets[chosen]["per_image_dice"], targets[arm]["per_image_dice"])
                    for arm in references
                },
                "observed_best_single_comparator": best,
                "delta_dice_vs_observed_best_single": targets[chosen]["dice"] - targets[best]["dice"],
            }
        reports[seed] = output
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-reports", type=Path, nargs="+", required=True)
    parser.add_argument("--evaluate-seeds", type=int, nargs="+",
                        help="Infer only these unique seeds from the supplied full source VAL reports; selection always uses all reports.")
    parser.add_argument("--held-out", nargs="+", required=True)
    parser.add_argument("--split-overrides", type=Path, nargs="+", required=True)
    parser.add_argument("--exclude", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = require_private_path(args.output, "External report output")
    if output.exists():
        raise FileExistsError(output)
    loaded = [load_report(path) for path in args.source_reports]
    selection = summarize(loaded, data_kind="real", select_on_val=True)
    inference_reports = evaluation_reports(loaded, args.evaluate_seeds)
    config = selection["protocol"]["configuration"]
    if len(set(args.held_out)) != len(args.held_out) or set(args.held_out) & set(config["source"]):
        raise ValueError("External datasets must be unique and disjoint from source domains")
    decision = selection["val_selection"]
    chosen = decision["selected_joint"]
    result = {
        "complete": True, "source_decision": decision,
        "source_reports": [{"path": r["path"], "sha256": r["sha256"]} for r in loaded],
        "selection_seeds": selection["seeds"], "evaluated_seeds": [],
        "held_out": args.held_out, "external_evaluation_opened": False,
        "interpretation": [
            "The joint arm is fixed using the source-VAL rule before external inference.",
            "Selection uses every complete source VAL report; the optional seed subset controls checkpoint validation and inference only.",
            "Intervals resample paired images; they do not establish independent patients or capture seed variation.",
            "The observed best single is a descriptive external comparator only.",
        ],
    }
    if chosen is not None:
        frozen = validated_checkpoints(inference_reports, chosen, config["source"])
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("External evaluation requires CUDA with bfloat16 support")
        torch.set_num_threads(4)
        datasets, model_args, opt = open_locked_cohorts(args.split_overrides, args.exclude, args.held_out, selection)
        result["cohort_overrides"] = [{"path": str(p.resolve()), "sha256": sha256(p)} for p in args.split_overrides]
        result["cohort_exclusions"] = {"path": str(args.exclude.resolve()), "sha256": sha256(args.exclude)} if args.exclude else None
        result["frozen_checkpoints"] = frozen
        result["external_evaluation_opened"] = True
        result["by_seed"] = evaluate_frozen(frozen, chosen, datasets, model_args, opt)
        result["evaluated_seeds"] = [report["seed"] for report in inference_reports]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"External report: {output}", flush=True)


if __name__ == "__main__":
    main()
