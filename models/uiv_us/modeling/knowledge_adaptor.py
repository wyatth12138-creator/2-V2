from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class KnowledgeAnchorBank(nn.Module):
    def __init__(self, feat_dim: int = 128, momentum: float = 0.95) -> None:
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.momentum = float(momentum)
        self._key_to_slot: Dict[str, int] = {}

    def _slot_names(self, slot: int) -> Tuple[str, str, str, str]:
        return (
            f"anchor_{slot}",
            f"anchor_init_{slot}",
            f"source_anchor_{slot}",
            f"source_anchor_init_{slot}",
        )

    def _ensure_slot(self, key: str, device: torch.device) -> int:
        if key in self._key_to_slot:
            return self._key_to_slot[key]
        slot = len(self._key_to_slot)
        self._key_to_slot[key] = slot
        anchor_name, init_name, source_name, source_init_name = self._slot_names(slot)
        self.register_buffer(anchor_name, torch.zeros(1, self.feat_dim, device=device))
        self.register_buffer(init_name, torch.zeros(1, dtype=torch.bool, device=device))
        self.register_buffer(source_name, torch.zeros(1, self.feat_dim, device=device))
        self.register_buffer(source_init_name, torch.zeros(1, dtype=torch.bool, device=device))
        return slot

    def snapshot_source_anchors(self) -> None:
        for key, slot in self._key_to_slot.items():
            anchor_name, init_name, source_name, source_init_name = self._slot_names(slot)
            anchor = getattr(self, anchor_name)
            init = getattr(self, init_name)
            if bool(init.item()):
                getattr(self, source_name).copy_(anchor.detach())
                getattr(self, source_init_name).fill_(True)

    def forward(
        self,
        feat: Optional[torch.Tensor],
        anchor_keys: Optional[List[str]] = None,
        mode: str = "disabled",
    ) -> Dict[str, torch.Tensor]:
        if feat is None or feat.numel() == 0 or anchor_keys is None or mode == "disabled":
            device = feat.device if torch.is_tensor(feat) else torch.device("cpu")
            zero = torch.zeros((), device=device, dtype=torch.float32)
            return {"anchor_loss": zero, "anchor_drift": zero}

        if isinstance(anchor_keys, str):
            anchor_keys = [anchor_keys] * feat.shape[0]
        if len(anchor_keys) != feat.shape[0]:
            raise ValueError(f"anchor_keys length {len(anchor_keys)} != batch size {feat.shape[0]}")

        total_loss = feat.new_zeros(())
        total_drift = feat.new_zeros(())
        counted_groups = 0

        ordered_groups: "OrderedDict[str, List[int]]" = OrderedDict()
        for index, key in enumerate(anchor_keys):
            ordered_groups.setdefault(str(key), []).append(index)

        for key, indices in ordered_groups.items():
            slot = self._ensure_slot(key, feat.device)
            anchor_name, init_name, source_name, source_init_name = self._slot_names(slot)
            anchor = getattr(self, anchor_name)
            anchor_init = getattr(self, init_name)
            source_anchor = getattr(self, source_name)
            source_anchor_init = getattr(self, source_init_name)

            current = feat[indices].mean(dim=0, keepdim=True)

            if mode in ("update", "target_update"):
                if not bool(anchor_init.item()):
                    anchor.copy_(current.detach())
                    anchor_init.fill_(True)
                    continue
                total_loss = total_loss + F.smooth_l1_loss(current, anchor.detach())
                total_drift = total_drift + (1.0 - F.cosine_similarity(current, anchor.detach(), dim=-1).mean())
                counted_groups += 1
                updated = self.momentum * anchor + (1.0 - self.momentum) * current.detach()
                anchor.copy_(updated)
                continue

            if mode in ("preserve", "source_preserve"):
                reference = source_anchor if bool(source_anchor_init.item()) else anchor
                reference_init = bool(source_anchor_init.item()) or bool(anchor_init.item())
                if not reference_init:
                    continue
                total_loss = total_loss + F.smooth_l1_loss(current, reference.detach())
                total_drift = total_drift + (1.0 - F.cosine_similarity(current, reference.detach(), dim=-1).mean())
                counted_groups += 1
                continue

            raise ValueError(f"Unsupported anchor mode: {mode}")

        if counted_groups == 0:
            zero = feat.new_zeros(())
            return {"anchor_loss": zero, "anchor_drift": zero}
        return {
            "anchor_loss": total_loss / counted_groups,
            "anchor_drift": total_drift / counted_groups,
        }


class SparseUpdateController:
    KEYWORDS = (
        "cls_head",
        "task_interaction",
        "dynamic_prompt_generator",
        "prototype_extractor",
        "memory_attention",
        "memory_encoder",
        "knowledge_anchor_bank",
        "prompt_bank",
    )

    def __init__(
        self,
        model: nn.Module,
        sparsity: float = 0.05,
        importance_score: str = "grad_x_param",
        ema_decay: float = 0.9,
        random_sparse: bool = False,
    ) -> None:
        self.model = model
        self.sparsity = float(sparsity)
        self.importance_score = importance_score
        self.ema_decay = float(ema_decay)
        self.random_sparse = bool(random_sparse)
        self.named_params = list(self.eligible_named_parameters(model))
        self.importance: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param, memory_format=torch.preserve_format)
            for name, param in self.named_params
        }
        self.masks: Dict[str, torch.Tensor] = {
            name: torch.ones_like(param, memory_format=torch.preserve_format)
            for name, param in self.named_params
        }

    @classmethod
    def eligible_named_parameters(cls, model: nn.Module) -> Iterator[Tuple[str, nn.Parameter]]:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(keyword in name for keyword in cls.KEYWORDS):
                yield name, param

    @classmethod
    def trainable_parameters(cls, model: nn.Module) -> Iterator[nn.Parameter]:
        for _, param in cls.eligible_named_parameters(model):
            yield param

    def reset_importance(self) -> None:
        for tensor in self.importance.values():
            tensor.zero_()

    def _score_from_param(self, param: nn.Parameter) -> Optional[torch.Tensor]:
        if param.grad is None:
            return None
        if self.importance_score == "grad_sq":
            return param.grad.detach().pow(2)
        return (param.grad.detach() * param.detach()).abs()

    @torch.no_grad()
    def accumulate(self) -> None:
        for name, param in self.named_params:
            score = self._score_from_param(param)
            if score is None:
                continue
            self.importance[name].mul_(self.ema_decay).add_(score, alpha=1.0 - self.ema_decay)

    @torch.no_grad()
    def refresh_masks(self) -> None:
        if not self.named_params:
            return
        if self.sparsity <= 0.0:
            for name in self.masks:
                self.masks[name].fill_(1.0)
            return
        if self.sparsity >= 1.0:
            for name in self.masks:
                self.masks[name].zero_()
            return

        flattened_scores = []
        for name, _ in self.named_params:
            score = self.importance[name]
            flattened_scores.append(score.reshape(-1))
        all_scores = torch.cat(flattened_scores, dim=0)
        total_params = all_scores.numel()
        keep_count = max(1, int(round(total_params * (1.0 - self.sparsity))))

        if self.random_sparse:
            perm = torch.randperm(total_params, device=all_scores.device)
            selected = perm[:keep_count]
            keep_flat = torch.zeros(total_params, device=all_scores.device, dtype=torch.bool)
            keep_flat[selected] = True
        else:
            if keep_count >= total_params:
                keep_flat = torch.ones(total_params, device=all_scores.device, dtype=torch.bool)
            else:
                topk_values = torch.topk(all_scores, k=keep_count, largest=True).values
                threshold = topk_values[-1]
                keep_flat = all_scores >= threshold

        offset = 0
        for name, param in self.named_params:
            numel = param.numel()
            self.masks[name] = keep_flat[offset: offset + numel].view_as(param).to(dtype=param.dtype)
            offset += numel

    @torch.no_grad()
    def apply_masks(self) -> None:
        for name, param in self.named_params:
            if param.grad is None:
                continue
            param.grad.mul_(self.masks[name])

