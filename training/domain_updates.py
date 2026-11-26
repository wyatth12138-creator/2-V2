"""Merge same-parameter-state domain gradients without touching private paths."""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import torch
from torch import Tensor


DomainGradient = Sequence[Optional[Tensor]]


def _orders(ordering, count: int) -> list[list[int]]:
    if isinstance(ordering, Tensor):
        ordering = ordering.detach().cpu().tolist()
    if ordering is None:
        return [[j for j in range(count) if j != i] for i in range(count)]
    try:
        supplied = list(ordering)
    except TypeError as exc:
        raise ValueError("ordering must be a permutation or one permutation per domain") from exc
    if not supplied and count == 0:
        return []
    is_flat = all(isinstance(index, int) and not isinstance(index, bool) for index in supplied)
    if is_flat:
        if sorted(supplied) != list(range(count)):
            raise ValueError("flat ordering must contain every domain index exactly once")
        return [[index for index in supplied if index != i] for i in range(count)]
    if len(supplied) != count:
        raise ValueError("nested ordering must have one row per domain")
    result = []
    for i, row in enumerate(supplied):
        if isinstance(row, Tensor):
            row = row.detach().cpu().tolist()
        try:
            row = list(row)
        except TypeError as exc:
            raise ValueError("each ordering row must be a permutation of the other domains") from exc
        if any(not isinstance(index, int) or isinstance(index, bool) for index in row):
            raise ValueError("ordering indices must be integers")
        if sorted(row) == list(range(count)):
            row = [index for index in row if index != i]
        if sorted(row) != [j for j in range(count) if j != i]:
            raise ValueError("each ordering row must contain every other domain exactly once")
        result.append(row)
    return result


def _dot(left: DomainGradient, right: DomainGradient, indices: Sequence[int]) -> float:
    """One host transfer per device, not one synchronization per parameter.

    Normal training uses FP32 multiplication with FP64 scalar reduction. If
    extreme finite gradients overflow the FP32 product, retry with FP64
    multiplication; the uncommon fallback preserves input validation without
    making every large parameter tensor a double-precision temporary.
    """
    grouped: dict[torch.device, list[Tensor]] = {}
    for index in indices:
        if left[index] is not None and right[index] is not None:
            first, second = left[index].detach(), right[index].detach()
            dtype = torch.float64 if first.dtype == torch.float64 else torch.float32
            value = (first.to(dtype) * second.to(dtype)).sum(dtype=torch.float64)
            grouped.setdefault(first.device, []).append(value)
    total = math.fsum(float(torch.stack(values).sum().item()) for values in grouped.values())
    if not math.isfinite(total):
        grouped = {}
        for index in indices:
            if left[index] is not None and right[index] is not None:
                first, second = left[index].detach(), right[index].detach()
                value = (first.double() * second.double()).sum()
                grouped.setdefault(first.device, []).append(value)
        total = math.fsum(float(torch.stack(values).sum().item()) for values in grouped.values())
        if not math.isfinite(total):
            raise ValueError("gradient dot product overflowed; rescale the domain losses")
    return total


def _norm(gradient: DomainGradient, indices: Sequence[int]) -> float:
    return math.sqrt(max(_dot(gradient, gradient, indices), 0.0))


def _all_finite(tensors: Sequence[Tensor]) -> bool:
    grouped: dict[torch.device, list[Tensor]] = {}
    for tensor in tensors:
        grouped.setdefault(tensor.device, []).append(torch.isfinite(tensor).all())
    return all(bool(torch.stack(values).all().item()) for values in grouped.values())


def merge_domain_gradients(
    gradients: Sequence[DomainGradient],
    weights: Optional[Sequence[float]] = None,
    *,
    mode: str = "weighted_mean",
    ordering=None,
    protected: Optional[Sequence[bool]] = None,
) -> tuple[tuple[Optional[Tensor], ...], dict[str, Any]]:
    """Return a weighted gradient tuple and JSON-safe conflict diagnostics.

    ``gradients[d][p]`` is the gradient of domain d's MEAN loss for parameter
    p, evaluated at the SAME parameter state for all domains. ``None`` means
    that parameter does not participate in that domain. Tensor shapes, dtypes
    and devices must agree per parameter. Inputs are never mutated; outputs
    are detached and may be assigned directly to ``parameter.grad``.

    Nonnegative finite weights are normalized to sum to one. An omitted
    weight vector means uniform domains. A positive-weight domain whose loss
    has no parameter gradients still contributes its original objective
    weight; other domains are not silently reweighted. All-zero weights give
    an all-None no-op. Zero-weight domains do not constrain PCGrad.

    ``weighted_mean`` is the matched-batch reference. In ``pcgrad``, each
    domain's mutable gradient is projected against the other ORIGINAL domain
    gradients, in the supplied order, only when their dot product is negative.
    Each pair uses only parameter blocks present in BOTH domains. Private
    blocks remain untouched. ``protected[p]=True`` additionally excludes a
    parameter from surgery but retains its ordinary weighted gradient. Use
    this for dense prompt/embedding banks with private rows: None support
    alone cannot identify private rows inside an otherwise shared tensor.

    ``ordering`` may be a common full permutation, or one permutation per
    domain (with or without the domain itself). None uses deterministic index
    order; pass seeded shuffled permutations to reproduce randomized PCGrad.

    ``pairwise`` contains ORIGINAL dot/cosine/norms on each pair's unprotected
    common support. Undefined zero-norm cosines are None. ``update_dot[d]``
    is <original domain gradient, merged gradient> on its entire support;
    positive values describe local descent for an update -lr*g under plain
    gradient descent, NOT a performance guarantee or an Adam-aware bound.
    """
    if mode not in ("weighted_mean", "pcgrad"):
        raise ValueError("mode must be 'weighted_mean' or 'pcgrad'")
    try:
        domains = [tuple(domain) for domain in gradients]
    except TypeError as exc:
        raise ValueError("gradients must be a sequence of per-parameter sequences") from exc
    count = len(domains)
    parameter_count = len(domains[0]) if count else 0
    if any(len(domain) != parameter_count for domain in domains):
        raise ValueError("all domains must have the same number of parameter entries")
    orders = _orders(ordering, count)

    if protected is None:
        protected_flags = [False] * parameter_count
    else:
        if isinstance(protected, Tensor):
            protected = protected.detach().cpu().tolist()
        try:
            protected_flags = list(protected)
        except TypeError as exc:
            raise ValueError("protected must be a boolean sequence with one entry per parameter") from exc
        if len(protected_flags) != parameter_count or any(type(flag) is not bool for flag in protected_flags):
            raise ValueError("protected must be a boolean sequence with one entry per parameter")

    if weights is None:
        normalized = [1.0 / count] * count if count else []
    else:
        if isinstance(weights, Tensor):
            if weights.ndim != 1:
                raise ValueError("weights must be a one-dimensional domain weight vector")
            weights = weights.detach().cpu().tolist()
        try:
            normalized = [float(weight) for weight in weights]
        except (TypeError, ValueError) as exc:
            raise ValueError("weights must be a finite nonnegative domain weight vector") from exc
        if len(normalized) != count or any(not math.isfinite(weight) or weight < 0 for weight in normalized):
            raise ValueError("weights must be finite, nonnegative, and match the domain count")
        # Scaling by the largest entry avoids overflow for large finite weights.
        maximum = max(normalized, default=0.0)
        if maximum > 0:
            normalized = [weight / maximum for weight in normalized]
            total = math.fsum(normalized)
            normalized = [weight / total for weight in normalized]

    templates: list[Optional[Tensor]] = [None] * parameter_count
    checked_tensors = []
    for domain in domains:
        for p, gradient in enumerate(domain):
            if gradient is None:
                continue
            if not isinstance(gradient, Tensor) or not gradient.is_floating_point() or gradient.layout != torch.strided:
                raise ValueError("gradient entries must be dense real floating tensors or None")
            checked_tensors.append(gradient)
            template = templates[p]
            if template is None:
                templates[p] = gradient
            elif gradient.shape != template.shape or gradient.dtype != template.dtype or gradient.device != template.device:
                raise ValueError("gradient shape, dtype and device must agree for each parameter")
    if not _all_finite(checked_tensors):
        raise ValueError("gradient entries must be finite")

    all_indices = list(range(parameter_count))
    positive = [i for i, weight in enumerate(normalized) if weight > 0]
    norms = [_norm(domain, all_indices) for domain in domains]
    pairs, comparable, conflicts = [], 0, 0
    common_support, reference_norm_squared = {}, {}
    for offset, i in enumerate(positive):
        for j in positive[offset + 1:]:
            shared = [
                p for p in all_indices
                if not protected_flags[p] and domains[i][p] is not None and domains[j][p] is not None
            ]
            dot = _dot(domains[i], domains[j], shared)
            squared_i = max(_dot(domains[i], domains[i], shared), 0.0)
            squared_j = max(_dot(domains[j], domains[j], shared), 0.0)
            norm_i, norm_j = math.sqrt(squared_i), math.sqrt(squared_j)
            common_support[i, j] = common_support[j, i] = shared
            reference_norm_squared[i, j] = squared_j
            reference_norm_squared[j, i] = squared_i
            cosine = dot / (norm_i * norm_j) if norm_i > 0 and norm_j > 0 else None
            if cosine is not None:
                cosine = max(-1.0, min(1.0, cosine))
                comparable += 1
                conflicts += int(dot < 0)
            pairs.append({
                "domain_i": i, "domain_j": j, "dot": dot, "cosine": cosine,
                "norm_i": norm_i, "norm_j": norm_j, "shared_parameters": len(shared),
            })

    edited = [tuple(None if gradient is None else gradient.detach().clone() for gradient in domain) for domain in domains]
    projection_steps = 0
    if mode == "pcgrad":
        for i in positive:
            working = list(edited[i])
            for j in orders[i]:
                if normalized[j] == 0:
                    continue
                shared = common_support[i, j]
                dot = _dot(working, domains[j], shared)
                norm_squared = reference_norm_squared[i, j]
                if dot < 0 and norm_squared > 0:
                    coefficient = dot / norm_squared
                    for p in shared:
                        working[p] = working[p] - coefficient * domains[j][p].detach()
                    projection_steps += 1
            edited[i] = tuple(working)

    merged = []
    for p in all_indices:
        terms = [
            normalized[i] * edited[i][p]
            for i in positive if edited[i][p] is not None
        ]
        if not terms:
            merged.append(None)
            continue
        value = terms[0]
        for term in terms[1:]:
            value = value + term
        merged.append(value)
    result = tuple(merged)
    if not _all_finite([value for value in result if value is not None]):
        raise ValueError("merged gradient overflowed; rescale the domain losses")
    update_norm = _norm(result, all_indices)
    update_dot = [_dot(domain, result, all_indices) for domain in domains]
    update_cosine = [
        max(-1.0, min(1.0, dot / (norm * update_norm))) if norm > 0 and update_norm > 0 else None
        for dot, norm in zip(update_dot, norms)
    ]
    stats = {
        "mode": mode, "num_domains": count, "num_active_domains": len(positive),
        "weights": normalized, "domain_norms": norms,
        "pairwise_scope": "unprotected_common_parameter_blocks",
        "pairwise": pairs, "comparable_pairs": comparable,
        "conflicting_pairs": conflicts, "conflict_fraction": conflicts / comparable if comparable else 0.0,
        "projection_steps": projection_steps, "protected_parameters": sum(protected_flags),
        "update_norm": update_norm, "update_dot": update_dot, "update_cosine": update_cosine,
        "negative_update_domains": sum(update_dot[i] < 0 for i in positive),
    }
    return result, stats
