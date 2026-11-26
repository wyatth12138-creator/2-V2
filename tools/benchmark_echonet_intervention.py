"""Aggregate-only, real-image perturbation check of mask intervention.

This is not a diagnostic benchmark or a genuine temporal tracking experiment.
It perturbs reference-mask-derived observations on traced EchoNet images.
No frames, masks, identifiers, or per-clip results are written to disk.

Trace construction follows the official vertex recipe at
https://github.com/echonet/dynamic/blob/master/echonet/datasets/echo.py
(LargeTrace/SmallTrace): omit the first long-axis segment, concatenate x1/y1
with reversed x2/y2, and round coordinates to integer pixel centers. A NumPy
even-odd polygon rasterizer including boundary pixels implements that geometry
without a scikit-image dependency.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch


SHIFTS = ((0, -2), (0, 2), (-2, 0), (2, 0))
METHODS = ("raw", "projected", "anchored", "registered", "transported")
PAIRED_ARMS = (
    ("image_and_mask_shift", "mask_only_shift"),
    ("local_shape_image_and_mask", "local_shape_mask_only"),
)
LOCAL_EXPANSION_STRENGTH = 0.08


def canonical_name(value):
    return value if Path(value).suffix else value + ".avi"


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_set_digest(names):
    """A reproducibility checksum, with no identifiers persisted individually."""
    return hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest()


def hash_order(names, seed):
    return sorted(set(names), key=lambda name: hashlib.sha256((str(seed) + ":" + name).encode()).digest())


def exclude_previous_test_sample(candidates, traces, report_path, file_list_hash, traces_hash, data_root):
    """Reconstruct the exact previous 80-clip sample from verified metadata.

    The original report records one missing-trace exclusion, no other failure,
    and two available frames for every accepted clip. Thus that selection can
    be reconstructed without decoding any new holdout images. If those facts
    cannot be verified, refuse to label a new TEST run independent.
    """
    prior = json.loads(report_path.read_text(encoding="utf-8"))
    if prior.get("file_list_sha256") != file_list_hash or prior.get("traces_sha256") != traces_hash:
        raise ValueError("Prior-sample exclusion requires unchanged metadata CSV hashes")
    if prior.get("split", "TEST") != "TEST" or "TEST" not in prior.get("selection", ""):
        raise ValueError("The exclusion report must describe the original TEST selection")
    count = prior.get("evaluated_clips")
    if (
        count != 80 or prior.get("evaluated_reference_frames") != 160
        or prior.get("excluded_clipping_conditions") != 0
        or prior.get("malformed_trace_rows") != 0
        or set(prior.get("skipped_counts", {})) - {"missing_traces"}
        or prior.get("test_candidate_clips", prior.get("candidate_clips")) != len(candidates)
    ):
        raise ValueError("Prior sample is not exactly reconstructable by the supported metadata-only recipe")
    selected, skipped = [], collections.Counter()
    for name in hash_order(candidates, prior["seed"]):
        if len(selected) == count:
            break
        if not traces.get(name):
            skipped["missing_traces"] += 1
            continue
        if len(traces[name]) < 2 or not (data_root / "Videos" / name).is_file():
            raise ValueError("A previously accepted clip no longer satisfies the recorded selection facts")
        selected.append(name)
    if len(selected) != count or dict(skipped) != prior.get("skipped_counts", {}):
        raise ValueError("Prior sample reconstruction did not reproduce its exact count and skip ledger")
    excluded = set(selected)
    return excluded, {
        "excluded_clip_count": len(excluded),
        "prior_report_sha256": file_digest(report_path),
        "prior_seed": prior["seed"],
        "excluded_set_sha256": selected_set_digest(selected),
        "reconstruction": "Exact original hash order, matching CSV hashes and the recorded missing-trace-only skip ledger; no new holdout images decoded.",
    }


def load_observer(project_root):
    path = project_root / "models/uiv_us/modeling/mask_intervention.py"
    spec = importlib.util.spec_from_file_location("_echonet_mask_intervention", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the mask intervention implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.correct_mask_observations, path


def traced_polygon_mask(trace, height, width):
    trace = np.asarray(trace, dtype=np.float64)
    if trace.ndim != 2 or trace.shape[1] != 4 or trace.shape[0] < 4 or not np.isfinite(trace).all():
        raise ValueError("invalid_trace_geometry")
    x1, y1, x2, y2 = trace.T
    vx = np.rint(np.concatenate((x1[1:], x2[1:][::-1]))).astype(np.int64)
    vy = np.rint(np.concatenate((y1[1:], y2[1:][::-1]))).astype(np.int64)
    y, x = np.mgrid[:height, :width]
    inside = np.zeros((height, width), dtype=bool)
    boundary = np.zeros_like(inside)
    for index in range(len(vx)):
        other = (index + 1) % len(vx)
        xa, ya, xb, yb = vx[index], vy[index], vx[other], vy[other]
        cross = (x - xa) * (yb - ya) - (y - ya) * (xb - xa)
        on_segment = (
            (cross == 0) & (x >= min(xa, xb)) & (x <= max(xa, xb))
            & (y >= min(ya, yb)) & (y <= max(ya, yb))
        )
        boundary |= on_segment
        if ya != yb:
            crossing_x = (xb - xa) * (y - ya) / float(yb - ya) + xa
            inside ^= ((ya > y) != (yb > y)) & (x < crossing_x)
    mask = inside | boundary
    if mask.sum() < 16:
        raise ValueError("empty_or_tiny_trace")
    return mask


def translate(array, dy, dx, fill):
    result = np.full_like(array, fill)
    height, width = array.shape
    source_y = slice(max(0, -dy), min(height, height - dy))
    source_x = slice(max(0, -dx), min(width, width - dx))
    target_y = slice(max(0, dy), min(height, height + dy))
    target_x = slice(max(0, dx), min(width, width + dx))
    result[target_y, target_x] = array[source_y, source_x]
    return result


def centroid(mask):
    y, x = np.nonzero(mask)
    return np.asarray((x.mean(), y.mean()), dtype=np.float64)


def mask_quality(probability):
    support = probability * (probability > 0.5)
    mass = support.sum(dim=(-2, -1)).clamp_min(1e-6)
    confidence = (support * (2 * probability - 1)).sum(dim=(-2, -1)) / mass
    return (confidence * (support.mean(dim=(-2, -1)) / 0.02).clamp(max=1)).squeeze(-1)


def summarize(values, bootstrap_samples, rng):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "n_clips": 0}
    draws = rng.integers(0, len(values), size=(bootstrap_samples, len(values)))
    means = values[draws].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "n_clips": int(len(values)),
    }


def local_expansion(gray, mask):
    """Fixed smooth image/mask warp, used only as a nonrigid perturbation.

    This inverse radial map has a small, fixed displacement and decays away
    from the reference target. It supplies a known transformed reference,
    without asserting that it reproduces real myocardial deformation.
    """
    height, width = mask.shape
    cx, cy = centroid(mask)
    y, x = np.mgrid[:height, :width].astype(np.float32)
    dy, dx = y - cy, x - cx
    inside_y, inside_x = np.nonzero(mask)
    radius = max(4.0, float(np.sqrt((inside_x - cx) ** 2 + (inside_y - cy) ** 2).max()))
    envelope = np.exp(-(dx * dx + dy * dy) / (2 * (0.8 * radius) ** 2))
    factor = 1 - LOCAL_EXPANSION_STRENGTH * envelope
    map_x = (cx + dx * factor).astype(np.float32)
    map_y = (cy + dy * factor).astype(np.float32)
    warped_image = cv2.remap(gray, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped_mask = cv2.remap(mask.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    return warped_image, warped_mask


def evaluate_frame(frame, mask, correct):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    probability = np.where(mask, 0.97, 0.03).astype(np.float32)
    original_center = centroid(mask)
    cases, display_cases, probability_cases = [], [], []
    paired_indices = []
    clipped = 0

    def add_case(arm, current_image, current_mask, target_mask, shift=None):
        current_probability = np.where(current_mask, 0.97, 0.03).astype(np.float32)
        display_cases.append(np.stack((gray, current_image), axis=0)[:, None])
        probability_cases.append(np.stack((probability, current_probability), axis=0)[:, None])
        cases.append({
            "arm": arm, "target": centroid(target_mask),
            "target_area": float(target_mask.mean() * 0.97), "shift": shift,
        })

    for dy, dx in SHIFTS:
        shifted_mask = translate(mask, dy, dx, False)
        if shifted_mask.sum() != mask.sum():
            clipped += 1
            continue
        shifted_image = translate(gray, dy, dx, 0.0)
        pair_start = len(cases)
        add_case("image_and_mask_shift", shifted_image, shifted_mask, shifted_mask, (dx, dy))
        add_case("mask_only_shift", gray, shifted_mask, mask, (dx, dy))
        paired_indices.append((pair_start, pair_start + 1))
    rigid_conditions = len(paired_indices)
    expanded_image, expanded_mask = local_expansion(gray, mask)
    pair_start = len(cases)
    add_case("local_shape_image_and_mask", expanded_image, expanded_mask, expanded_mask)
    add_case("local_shape_mask_only", gray, expanded_mask, mask)
    paired_indices.append((pair_start, pair_start + 1))
    add_case("no_change_control", gray, mask, mask)
    add_case("brightness_only_control", gray * 0.85 + 0.05, mask, mask)
    display = torch.from_numpy(np.stack(display_cases))
    probabilities = torch.from_numpy(np.stack(probability_cases))
    # Preserve the original comparison paths. Only the new variant receives
    # the registration selector.
    projected = correct(display, probabilities, use_image=False)
    anchored = correct(display, probabilities, use_image=True)
    registered = correct(display, probabilities, use_image=True, anchor_method="registration")
    outputs = {
        "raw": anchored["raw"], "projected": projected["corrected"],
        "anchored": anchored["corrected"], "registered": registered["corrected"],
        "transported": registered["transported"],
    }
    height, width = gray.shape
    pixel_scale = np.asarray(((width - 1) / 2, (height - 1) / 2))
    quality = mask_quality(probabilities).numpy()
    max_quality_difference = max(
        float(np.abs(quality[first] - quality[second]).max())
        for first, second in paired_indices
    )
    max_geometry_difference = max(
        float((anchored["raw"][first, :, :3] - anchored["raw"][second, :, :3]).abs().max())
        for first, second in paired_indices
    )
    metrics = []
    for index, case in enumerate(cases):
        for method, descriptors in outputs.items():
            current = descriptors[index, 1, 1:3].numpy().astype(np.float64)
            position = (current + 1) * pixel_scale
            delta = position - original_center
            error = position - case["target"]
            if method == "raw":
                active = False
            elif method in ("registered", "transported"):
                active = bool(registered.get("registration_active", registered["active"])[index, 1])
            else:
                active = bool((anchored if method == "anchored" else projected)["active"][index, 1])
            values = {
                "centroid_l2_error_px": float(np.linalg.norm(error)),
                "centroid_coordinate_mae_px": float(np.abs(error).mean()),
                "centroid_within_0_05px_fraction": float(np.linalg.norm(error) <= 0.05),
                "roi_area_relative_error": float(abs(float(descriptors[index, 1, 0]) - case["target_area"]) / case["target_area"]),
                "active_fraction": float(active),
            }
            if case["shift"] is not None:
                shift_vector = np.asarray(case["shift"], dtype=np.float64)
                unit_direction = shift_vector / np.linalg.norm(shift_vector)
                values["signed_shift_retention"] = float(delta.dot(unit_direction) / 2.0)
                values["absolute_shift_retention"] = float(abs(delta.dot(unit_direction) / 2.0))
            metrics.append((case["arm"], method, values))
    return metrics, {
        "clipped_rigid_conditions": clipped,
        "paired_rigid_conditions": rigid_conditions,
        "paired_nonrigid_conditions": 1,
        "equivalence_controls": 2,
        "max_quality_difference": max_quality_difference,
        "max_geometry_difference": max_geometry_difference,
    }


def run(args):
    started = time.perf_counter()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    correct, module_path = load_observer(args.project_root)
    module_hash = file_digest(module_path)
    file_list = args.data_root / "FileList.csv"
    trace_file = args.data_root / "VolumeTracings.csv"
    file_list_hash, traces_hash = file_digest(file_list), file_digest(trace_file)
    with file_list.open(newline="", encoding="utf-8-sig") as handle:
        candidates = [canonical_name(row["FileName"]) for row in csv.DictReader(handle) if row["Split"].upper() == args.split]
    candidates = hash_order(candidates, args.seed)
    original_candidate_count = len(candidates)
    candidate_set = set(candidates)
    traces = collections.defaultdict(lambda: collections.defaultdict(list))
    malformed_trace_rows = 0
    with trace_file.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            name = canonical_name(row["FileName"])
            if name not in candidate_set:
                continue
            try:
                index = int(row["Frame"])
                coordinates = tuple(float(row[key]) for key in ("X1", "Y1", "X2", "Y2"))
                traces[name][index].append(coordinates)
            except (KeyError, ValueError):
                malformed_trace_rows += 1
    excluded, exclusion_audit = set(), None
    if args.split == "TEST":
        excluded, exclusion_audit = exclude_previous_test_sample(
            candidates, traces, args.exclude_prior_report, file_list_hash, traces_hash, args.data_root
        )
        candidates = [name for name in candidates if name not in excluded]
    skipped = collections.Counter()
    clip_summaries = collections.defaultdict(list)
    accepted = traced_frames = evaluated_pairs = clipped_pairs = 0
    nonrigid_pairs = control_conditions = 0
    max_quality_difference = max_geometry_difference = 0.0
    accepted_names = []
    decoding_seconds = evaluation_seconds = 0.0
    for name in candidates:
        if accepted >= args.max_clips:
            break
        frame_traces = traces.get(name)
        if not frame_traces:
            skipped["missing_traces"] += 1
            continue
        video_path = args.data_root / "Videos" / name
        if not video_path.is_file():
            skipped["missing_video"] += 1
            continue
        capture = cv2.VideoCapture(str(video_path))
        local = collections.defaultdict(list)
        indices = list(frame_traces)
        indices = indices[:1] if len(indices) == 1 else (indices[0], indices[-1])
        for frame_index in indices:
            decoded_at = time.perf_counter()
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            successful, frame = capture.read()
            decoding_seconds += time.perf_counter() - decoded_at
            if not successful or frame is None:
                skipped["failed_traced_frame_decode"] += 1
                continue
            try:
                mask = traced_polygon_mask(frame_traces[frame_index], *frame.shape[:2])
            except ValueError as error:
                skipped[str(error)] += 1
                continue
            evaluated_at = time.perf_counter()
            measurements, frame_audit = evaluate_frame(frame, mask, correct)
            evaluation_seconds += time.perf_counter() - evaluated_at
            clipped_pairs += frame_audit["clipped_rigid_conditions"]
            max_quality_difference = max(max_quality_difference, frame_audit["max_quality_difference"])
            max_geometry_difference = max(max_geometry_difference, frame_audit["max_geometry_difference"])
            if measurements:
                traced_frames += 1
                evaluated_pairs += frame_audit["paired_rigid_conditions"]
                nonrigid_pairs += frame_audit["paired_nonrigid_conditions"]
                control_conditions += frame_audit["equivalence_controls"]
            for arm, method, values in measurements:
                for metric, value in values.items():
                    local[(arm, method, metric)].append(value)
        capture.release()
        if not local:
            skipped["no_evaluable_traced_frames"] += 1
            continue
        accepted += 1
        accepted_names.append(name)
        for key, values in local.items():
            clip_summaries[key].append(float(np.mean(values)))
    rng = np.random.default_rng(args.seed)
    report = {
        "experiment": "EchoNet reference-mask-derived paired image/mask perturbation",
        "evidence_boundary": "Real ultrasound pixels, artificial two-frame translations, reference-derived masks; no pretrained model, no diagnostic outcomes, no genuine time-trajectory claim.",
        "public_source": "https://echonet.github.io/dynamic/",
        "trace_recipe_source": "https://github.com/echonet/dynamic/blob/master/echonet/datasets/echo.py",
        "polygon_rasterization": "Official vertex ordering and integer rounding; NumPy even-odd pixel-center filling including polygon boundary.",
        "data_root": str(args.data_root),
        "split": args.split,
        "purpose": "development_edge_audit" if args.split == "VAL" else ("training_set_mechanism_audit" if args.split == "TRAIN" else "previously_unused_TEST_mechanism_holdout"),
        "selection": f"SHA256(seed:filename) ascending {args.split} clips, backfilled only for malformed/missing inputs; no outcome or correction-quality selection.",
        "seed": args.seed,
        "requested_max_clips": args.max_clips,
        "candidate_clips_before_prior_exclusion": original_candidate_count,
        "candidate_clips": len(candidates),
        "prior_test_exclusion": exclusion_audit,
        "selected_clip_set_sha256": selected_set_digest(accepted_names),
        "selected_overlap_with_prior_test": len(set(accepted_names) & excluded) if args.split == "TEST" else None,
        "evaluated_clips": accepted,
        "evaluated_reference_frames": traced_frames,
        "evaluated_paired_shift_conditions": evaluated_pairs,
        "evaluated_paired_nonrigid_conditions": nonrigid_pairs,
        "evaluated_equivalence_controls": control_conditions,
        "excluded_clipping_conditions": clipped_pairs,
        "malformed_trace_rows": malformed_trace_rows,
        "skipped_counts": dict(skipped),
        "probability_inside": 0.97,
        "probability_outside": 0.03,
        "translations": "x/y plus or minus 2 pixels; no wraparound",
        "nonrigid_control": f"Fixed inverse radial expansion, strength {LOCAL_EXPANSION_STRENGTH}, Gaussian envelope sigma=0.8*reference maximum radius; image/mask coordinated versus mask-only, never a claimed physiological deformation.",
        "equivalence_controls": "Identical image/mask and global brightness 0.85*image+0.05 with unchanged mask. The 0.05px fraction is an engineering sanity readout, not a formal statistical equivalence claim.",
        "scope_limitations": "Exact rigid shifts favor image registration. Local artificial shape changes add an area-preservation check, but do not validate real cardiac deformation, perfusion, out-of-plane motion or natural model segmentation error. Reference-derived masks are not model predictions.",
        "matched_inputs": "Mask probabilities, raw mask geometry and mask-only quality proxy are identical between arms; appearance and full descriptor residuals need not match.",
        "max_matched_mask_quality_difference": max_quality_difference,
        "max_matched_raw_geometry_difference": max_geometry_difference,
        "estimand": "Average each metric within clip over available traced frames and shifts, then average clips with equal weight; bootstrap resamples clips, not frames.",
        "bootstrap_resamples": args.bootstrap_samples,
        "module_sha256": module_hash,
        "file_list_sha256": file_list_hash,
        "traces_sha256": traces_hash,
        "reproduction_flags": {
            "split": args.split, "max_clips": args.max_clips, "seed": args.seed,
            "bootstrap_samples": args.bootstrap_samples,
            "data_root": str(args.data_root), "project_root": str(args.project_root),
            "output": str(args.output),
            "exclude_prior_report": str(args.exclude_prior_report) if args.exclude_prior_report else None,
        },
        "metrics": {},
        "paired_improvement_over_raw": {},
        "paired_registered_improvement_over_anchored": {},
        "paired_registered_improvement_over_transported": {},
        "balanced_paired_arm_errors": {},
    }
    for (arm, method, metric), values in sorted(clip_summaries.items()):
        report["metrics"].setdefault(arm, {}).setdefault(method, {})[metric] = summarize(values, args.bootstrap_samples, rng)
    all_arms = sorted({key[0] for key in clip_summaries})
    for arm in all_arms:
        for method in METHODS[1:]:
            raw = np.asarray(clip_summaries.get((arm, "raw", "centroid_l2_error_px"), []))
            corrected = np.asarray(clip_summaries.get((arm, method, "centroid_l2_error_px"), []))
            report["paired_improvement_over_raw"].setdefault(arm, {})[method] = summarize(raw - corrected, args.bootstrap_samples, rng)
        for metric in ("centroid_l2_error_px", "roi_area_relative_error"):
            registered = np.asarray(clip_summaries.get((arm, "registered", metric), []))
            anchored = np.asarray(clip_summaries.get((arm, "anchored", metric), []))
            transported = np.asarray(clip_summaries.get((arm, "transported", metric), []))
            report["paired_registered_improvement_over_anchored"].setdefault(arm, {})[metric] = summarize(anchored - registered, args.bootstrap_samples, rng)
            report["paired_registered_improvement_over_transported"].setdefault(arm, {})[metric] = summarize(transported - registered, args.bootstrap_samples, rng)
    for positive_arm, nuisance_arm in PAIRED_ARMS:
        pair_name = positive_arm + "__and__" + nuisance_arm
        report["balanced_paired_arm_errors"][pair_name] = {}
        for method in METHODS:
            for metric in ("centroid_l2_error_px", "roi_area_relative_error"):
                positive = np.asarray(clip_summaries.get((positive_arm, method, metric), []))
                nuisance = np.asarray(clip_summaries.get((nuisance_arm, method, metric), []))
                report["balanced_paired_arm_errors"][pair_name].setdefault(method, {})[metric] = summarize((positive + nuisance) / 2, args.bootstrap_samples, rng)
    report["timings_seconds"] = {
        "video_decode": decoding_seconds, "descriptor_and_correction": evaluation_seconds,
        "total": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "evaluated_clips": accepted, "evaluated_reference_frames": traced_frames, "paired_shift_conditions": evaluated_pairs, "seconds": report["timings_seconds"]["total"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Read-only EchoNet-Dynamic root containing the official CSVs and Videos")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True, help="Aggregate JSON destination; no media or per-clip identifiers are saved")
    parser.add_argument("--split", choices=("TRAIN", "VAL", "TEST"), default="VAL")
    parser.add_argument("--max-clips", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1246)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--exclude-prior-report", type=Path, help="Required for TEST: original 80-clip aggregate report used to reconstruct and exclude that exact sample")
    args = parser.parse_args()
    if args.max_clips < 1 or args.max_clips > 80 or args.bootstrap_samples < 1:
        parser.error("max-clips must be 1..80 and bootstrap-samples must be positive")
    if args.split == "TEST" and args.exclude_prior_report is None:
        parser.error("TEST requires --exclude-prior-report; use VAL for development")
    if args.split != "TEST" and args.exclude_prior_report is not None:
        parser.error("--exclude-prior-report is only used with TEST")
    run(args)


if __name__ == "__main__":
    main()
