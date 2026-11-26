"""Separate image-supported change from sensitivity to the predicted ROI mask."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def project_observation(
    observed: Tensor, reference: Tensor, anchor: Tensor, directions: Tensor, ridge: float = .05,
) -> Tensor:
    """Shrink the unsupported component along fixed, local nuisance directions.

    Each direction is normalized. A negligible direction contributes zero.
    The ridge keeps the solve defined for overlapping or absent directions.
    """
    if not ridge > 0:
        raise ValueError("ridge must be positive")
    if observed.ndim != 2 or reference.shape != observed.shape or anchor.shape != observed.shape:
        raise ValueError("observed, reference and anchor must have the same [B,D] shape")
    if directions.ndim != 3 or directions.shape[:2] != observed.shape:
        raise ValueError("directions must have shape [B,D,K]")
    scale = torch.linalg.vector_norm(directions, dim=1, keepdim=True)
    basis = torch.where(scale > 1e-4, directions / scale.clamp_min(1e-4), 0)
    adjoint = basis.transpose(1, 2)
    gram = adjoint @ basis
    eye = torch.eye(basis.shape[-1], dtype=basis.dtype, device=basis.device)
    residual = (observed - reference - anchor).unsqueeze(-1)
    coefficient = torch.linalg.solve(gram + ridge * eye, adjoint @ residual)
    return observed - (basis @ coefficient).squeeze(-1)


def _mask_descriptors(display: Tensor, probability: Tensor) -> Tensor:
    height, width = probability.shape[-2:]
    x = torch.linspace(-1, 1, width, device=probability.device, dtype=probability.dtype)
    y = torch.linspace(-1, 1, height, device=probability.device, dtype=probability.dtype)
    support = probability * (probability > .5)
    mass = support.sum(dim=(2, 3)).clamp_min(1e-6)
    outside = 1 - support
    inside_mean = (support * display).sum(dim=(2, 3)) / mass
    outside_mean = (outside * display).sum(dim=(2, 3)) / outside.sum(dim=(2, 3)).clamp_min(1e-6)
    return torch.cat((
        support.mean(dim=(2, 3)),
        (support * x).sum(dim=(2, 3)) / mass,
        (support * y[:, None]).sum(dim=(2, 3)) / mass,
        inside_mean, inside_mean - outside_mean,
    ), dim=1)


def _shift(mask: Tensor, dy: int, dx: int) -> Tensor:
    pad = max(abs(dy), abs(dx))
    padded = F.pad(mask, (pad, pad, pad, pad))
    h, w = mask.shape[-2:]
    return padded[..., pad-dy:pad-dy+h, pad-dx:pad-dx+w]


def _probes(mask: Tensor) -> Tensor:
    dilated = F.max_pool2d(mask, 5, stride=1, padding=2)
    eroded = -F.max_pool2d(-F.pad(mask, (2, 2, 2, 2)), 5, stride=1)
    # Identity followed by three signed pairs.
    return torch.stack((
        mask, _shift(mask, 0, 2), _shift(mask, 0, -2),
        _shift(mask, 2, 0), _shift(mask, -2, 0), dilated, eroded,
    ), dim=1)


def _boundary_scores(display: Tensor, masks: Tensor) -> Tensor:
    """Two-sided local contour contrast, independent of the current mask."""
    obj = (masks > .5).to(display.dtype)
    inner = obj + F.max_pool2d(-F.pad(obj, (1, 1, 1, 1)), 3, stride=1)
    outer = F.max_pool2d(obj, 3, stride=1, padding=1) - obj

    def stats(region: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        n = region.sum((1, 2, 3))
        mean = (display * region).sum((1, 2, 3)) / n.clamp_min(1)
        var = ((display - mean[:, None, None, None]).square() * region).sum((1, 2, 3)) / n.clamp_min(1)
        return mean, var, n

    a, va, na = stats(inner)
    b, vb, nb = stats(outer)
    score = (a - b).abs() / (va + vb + .01).sqrt()
    return torch.where((na >= 4) & (nb >= 4), score, 0)


_REGISTRATION_RADIUS = 3
_MAX_TRACKED_SHIFT = 2


def _registered_prior(previous: Tensor, current: Tensor, mask: Tensor):
    """Match local image texture on common support, without the current mask."""
    r = _REGISTRATION_RADIUS
    b, _, h, w = mask.shape
    empty_shift = torch.zeros((b, 2), dtype=torch.long, device=mask.device)
    if min(h, w) <= 2 * r:
        return mask, empty_shift, torch.zeros(b, dtype=torch.bool, device=mask.device), mask.new_zeros(b)
    support = F.max_pool2d((mask > .5).to(mask.dtype), 5, stride=1, padding=2)
    # Use the same support for every candidate so cropping cannot improve a score.
    support = F.pad(support[..., r:-r, r:-r], (r, r, r, r)).flatten(2)
    count = support.sum(2, keepdim=True)
    norm = count.clamp_min(1)
    reference = previous.flatten(2)
    reference = (reference - (reference * support).sum(2, keepdim=True) / norm) * support
    window = 2 * r + 1
    candidates = F.unfold(current, window, padding=r)
    candidates = (candidates - (candidates * support).sum(2, keepdim=True) / norm) * support
    ref_energy = reference.square().sum(2)
    candidate_energy = candidates.square().sum(2)
    scores = (reference * candidates).sum(2) / (ref_energy * candidate_energy).clamp_min(1e-12).sqrt()
    scores = scores.clamp(-1, 1)
    peaks, indices = scores.topk(2, dim=1)
    choice = indices[:, 0]
    delta = torch.stack((choice // window - r, choice % window - r), dim=-1)
    reliable = (
        (count[:, 0, 0] >= 32)
        & (ref_energy[:, 0] / norm[:, 0, 0] >= 1e-4)
        & (peaks[:, 0] >= .75)
        & (peaks[:, 0] - peaks[:, 1] >= .01)
        & (delta.abs().amax(dim=1) <= _MAX_TRACKED_SHIFT)
    )
    delta = torch.where(reliable[:, None], delta, empty_shift)
    row = torch.arange(h, device=mask.device)[None, :, None] - delta[:, 0, None, None]
    col = torch.arange(w, device=mask.device)[None, None, :] - delta[:, 1, None, None]
    on_image = (row >= 0) & (row < h) & (col >= 0) & (col < w)
    flattened = row.clamp(0, h - 1) * w + col.clamp(0, w - 1)
    transport = mask.flatten(1).gather(1, flattened.flatten(1)).view(b, 1, h, w)
    transport = transport * on_image[:, None]
    return transport, delta, reliable, peaks[:, 0]


@torch.no_grad()
def correct_mask_observations(
    display: Tensor, mask_probability: Tensor, valid: Tensor | None = None, *,
    timestamps: Tensor | None = None, use_image: bool = True, anchor_method: str = "contour",
) -> dict[str, Tensor]:
    """Build causal ROI descriptors from [B,T,1,H,W] display and mask tensors.

    The last valid frame supplies an image-checked reference mask and image. The previous
    mask is re-read on the CURRENT image to retain intensity changes within a
    fixed support. A new contour-supported displacement is estimated on a
    probe bank of the previous mask, independently of the current prediction.

    Correction requires local support and adjacent valid observations. A strong
    current contour outside the bank resets the reference without projection.
    The reference is updated with the better-supported current mask or probe.
    Registration mode uses two-frame texture matching on historical support
    and corrects only translational moments. Its transported output is an ablation.
    """
    if anchor_method not in ("contour", "registration"):
        raise ValueError("anchor_method must be contour or registration")
    registration = use_image and anchor_method == "registration"
    if display.ndim != 5 or display.shape[2] != 1 or display.shape != mask_probability.shape:
        raise ValueError("display and mask_probability must share [B,T,1,H,W] shape")
    batch, steps, _, h, w = display.shape
    if min(batch, steps, h, w) < 1:
        raise ValueError("display must have nonempty dimensions")
    if valid is None:
        valid = torch.ones((batch, steps), dtype=torch.bool, device=display.device)
    elif valid.shape != (batch, steps) or valid.dtype != torch.bool:
        raise ValueError("valid must be boolean [B,T]")
    else:
        valid = valid.to(display.device)

    if timestamps is not None:
        if timestamps.shape != (batch, steps):
            raise ValueError("timestamps must be [B,T]")
        timestamps = timestamps.to(device=display.device, dtype=torch.float64)

    image = display.detach().float()
    probability = mask_probability.detach().float()
    valid_pixels = valid[:, :, None, None, None]
    image = torch.where(valid_pixels, image, 0)
    probability = torch.where(valid_pixels, probability, 0)
    raw = _mask_descriptors(image.flatten(0, 1), probability.flatten(0, 1)).view(batch, steps, 5)
    previous_img = torch.zeros_like(image[:, 0])
    previous_mask = torch.zeros_like(probability[:, 0])
    previous_valid = torch.zeros(batch, dtype=torch.bool, device=display.device)
    cadence = torch.zeros(batch, dtype=torch.float64, device=display.device)

    outputs, anchors, activity, retention, ref_from_probe = [], [], [], [], []
    shifts, transported = [], []
    for t in range(steps):
        masks = _probes(previous_mask)
        probe_count = masks.shape[1]
        flat_masks = masks.flatten(0, 1)
        repeated_img = image[:, t].repeat_interleave(probe_count, dim=0)
        repeated_previous = previous_img.repeat_interleave(probe_count, dim=0)
        before = _boundary_scores(repeated_previous, flat_masks).view(batch, probe_count)
        descriptors = _mask_descriptors(repeated_img, flat_masks).view(batch, probe_count, 5)
        ref = descriptors[:, 0]
        directions = (descriptors[:, 1::2] - descriptors[:, 2::2]).transpose(1, 2)
        now = _boundary_scores(repeated_img, flat_masks).view(batch, probe_count)
        probe_score, best_probe = now.max(dim=1)
        raw_score = _boundary_scores(image[:, t], probability[:, t])
        adjacent = valid[:, t] & previous_valid
        if timestamps is not None and t > 0:
            gap = timestamps[:, t] - timestamps[:, t - 1]
            adjacent = adjacent & ((cadence <= 0) | (gap <= 8 * cadence))
            cadence = torch.where(adjacent, gap, cadence)
        contour_present = (before[:, 0] > .25) & (probe_score > .25)
        margin = .25
        outside_bank = raw_score > probe_score + margin
        active = adjacent & contour_present & ~outside_bank
        if registration:
            image_mask, delta, match, _ = _registered_prior(previous_img, image[:, t], previous_mask)
            active = adjacent & match
            registered_readout = _mask_descriptors(image[:, t], image_mask)
            anchor = registered_readout - ref
            # Translation matching provides no evidence about dilation or
            # enhancement. Correct only the two identified geometry channels.
            translation_channels = torch.tensor(
                [False, True, True, False, False], device=display.device,
            )
            directions = directions[:, :, :2] * translation_channels[None, :, None]
            shifts.append(torch.where(active[:, None], delta, 0))
            transported.append(torch.where(active[:, None], registered_readout, raw[:, t]))
        elif use_image:
            difference = (2 * now).softmax(dim=1) - (2 * before).softmax(dim=1)
            anchor = (difference.unsqueeze(-1) * (descriptors - ref[:, None])).sum(dim=1)
        else:
            anchor = torch.zeros_like(ref)
        anchor = torch.where(active[:, None], anchor, 0)
        zero = torch.zeros_like(ref)
        projected = project_observation(raw[:, t], ref, zero, directions)
        component = raw[:, t] - projected
        anchor_component = anchor - project_observation(anchor, zero, zero, directions)
        alignment = (anchor_component * component).sum(dim=1)
        kept = (alignment / component.square().sum(dim=1).clamp_min(1e-9)).clamp(0, 1)
        # A signed anchor can preserve part of the measured innovation. It
        # cannot reverse it or create motion when the observation did not move.
        corrected = raw[:, t] - (1 - kept[:, None]) * component
        outputs.append(torch.where(active[:, None], corrected, raw[:, t]))
        anchors.append(anchor)
        activity.append(active)
        retention.append(torch.where(active, kept, 0))
        choice = masks[torch.arange(batch, device=display.device), best_probe]
        use_probe = active & (probe_score > raw_score + margin)
        if registration:
            scale = image.new_tensor([(w - 1) / 2, (h - 1) / 2])
            discrepancy = (raw[:, t, 1:3] - registered_readout[:, 1:3]) * scale
            use_probe = active & (torch.linalg.vector_norm(discrepancy, dim=-1) > .75)
            choice = image_mask
        next_mask = torch.where(use_probe[:, None, None, None], choice, probability[:, t])
        ref_from_probe.append(use_probe)
        present = valid_pixels[:, t]
        previous_img = torch.where(present, image[:, t], previous_img)
        previous_mask = torch.where(present, next_mask, previous_mask)
        previous_valid = valid[:, t]
    result = {
        "raw": raw, "corrected": torch.stack(outputs, dim=1),
        "anchor": torch.stack(anchors, dim=1), "active": torch.stack(activity, dim=1),
        "retention": torch.stack(retention, dim=1), "reference_from_probe": torch.stack(ref_from_probe, dim=1),
    }
    if registration:
        result["registration_shift"] = torch.stack(shifts, dim=1)
        result["transported"] = torch.stack(transported, dim=1)
    return result
