"""Synthetic image/mask factorial experiment for the local observation rule.

The two arms share the exact same predicted masks, time grid, contrast and
noise draws. Their raw centroid changes and mask confidence are identical.
Only one arm contains the corresponding change in the image.
"""

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.uiv_us.modeling.mask_intervention import correct_mask_observations


def evaluate(trials, pixels, *, seed, noise_std):
    generator = torch.Generator().manual_seed(seed)
    height = width = 56
    frames = 6
    grid = torch.arange(width).reshape(1, 1, 1, 1, width)
    y = torch.arange(height).reshape(1, 1, 1, height, 1)
    cx = torch.randint(22, 34, (trials, 1, 1, 1, 1), generator=generator)
    cy = torch.randint(22, 34, (trials, 1, 1, 1, 1), generator=generator)
    sx = torch.randint(7, 12, (trials, 1, 1, 1, 1), generator=generator)
    sy = torch.randint(7, 12, (trials, 1, 1, 1, 1), generator=generator)
    direction = torch.randint(0, 2, (trials, 1, 1, 1, 1), generator=generator) * 2 - 1
    displacement = torch.tensor([0, pixels, 0, pixels, 0, pixels]).view(1, frames, 1, 1, 1)
    moved = ((grid - cx - direction * displacement) / sx).square() + ((y - cy) / sy).square() < 1
    still = moved[:, :1].expand_as(moved)
    polarity = torch.randint(0, 2, (trials, 1, 1, 1, 1), generator=generator) * 2 - 1
    contrast = .25 + .35 * torch.rand((trials, 1, 1, 1, 1), generator=generator)
    noise = noise_std * torch.randn(moved.shape, generator=generator)

    def render(mask):
        return (.5 + polarity * contrast * (mask.float() - .5) + noise).clamp(0, 1)

    predicted = torch.where(moved, .99, .01)
    display = torch.cat((render(moved), render(still)), dim=0)
    inputs = torch.cat((predicted, predicted), dim=0)
    assert torch.equal(inputs[:trials], inputs[trials:])
    anchored = correct_mask_observations(display, inputs)
    projected = correct_mask_observations(display, inputs, use_image=False)
    origin = anchored["raw"][:, :1, 1]
    true_change = anchored["raw"][:trials, :, 1] - origin[:trials]
    pixels_per_unit = (width - 1) / 2
    records = {}
    for name, estimates in (
        ("uncorrected", anchored["raw"]), ("projected", projected["corrected"]),
        ("anchored", anchored["corrected"]),
    ):
        relative = estimates[:, :, 1] - origin
        true_error = (relative[:trials] - true_change).abs() * pixels_per_unit
        false_error = relative[trials:].abs() * pixels_per_unit
        records[name] = {
            "true_motion_mae_px": true_error[:, 1:].mean().item(),
            "mask_only_motion_mae_px": false_error[:, 1:].mean().item(),
            "true_motion_retained_on_displaced_frames": (
                relative[:trials, 1::2] / true_change[:, 1::2]
            ).mean().item(),
            "false_motion_remaining_on_displaced_frames": (
                relative[trials:, 1::2].abs() / true_change[:, 1::2].abs()
            ).mean().item(),
        }
    return {
        "pairs": trials, "shift_px": pixels, "noise_std": noise_std,
        "anchored_correction_coverage_true": anchored["active"][:trials, 1:].float().mean().item(),
        "anchored_correction_coverage_mask_only": anchored["active"][trials:, 1:].float().mean().item(),
        "modes": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=48)
    args = parser.parse_args()
    if args.pairs <= 0:
        parser.error("--pairs must be positive")
    torch.set_num_threads(2)
    groups = [
        evaluate(args.pairs, pixels, seed=942, noise_std=noise)
        for pixels in (1, 2, 6) for noise in (0, .05)
    ]
    result = {"data_kind": "synthetic", "scope": "ROI mechanism verification", "groups": groups}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for item in groups:
        a = item["modes"]["anchored"]
        print(f"shift={item['shift_px']} noise={item['noise_std']}: "
              f"true MAE {a['true_motion_mae_px']:.3f} px; artificial MAE {a['mask_only_motion_mae_px']:.3f} px")


if __name__ == "__main__":
    main()
