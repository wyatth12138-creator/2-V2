from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.task_registry import canonicalize_task_batch, get_task_spec


class MaskDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        target = target.float()
        intersection = torch.sum(probs * target, dim=(1, 2, 3))
        union = torch.sum(probs * probs, dim=(1, 2, 3)) + torch.sum(target * target, dim=(1, 2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class UnifiedMultiTaskLoss(nn.Module):
    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        dice_weight: float = 0.8,
        proto_weight: float = 0.1,
        prompt_weight: float = 0.05,
        anchor_weight: float = 0.05,
        mutual_weight: float = 0.05,
        reliability_weight: float = 0.02,
        gate_sparsity_weight: float = 0.005,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.proto_weight = proto_weight
        self.prompt_weight = prompt_weight
        self.anchor_weight = anchor_weight
        self.mutual_weight = mutual_weight
        self.reliability_weight = reliability_weight
        self.gate_sparsity_weight = gate_sparsity_weight
        self.pos_weight = pos_weight
        self.seg_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.cls_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.seg_dice = MaskDiceLoss()

    def _prepare_seg_target(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        if logits.ndim == 5 and target.ndim == 4:
            target = target.reshape(-1, *target.shape[2:])
        elif target.ndim == 5:
            target = target.reshape(-1, *target.shape[2:])
        if logits.ndim == 4 and target.ndim == 3:
            target = target.unsqueeze(1)
        if logits.ndim == 3:
            logits = logits.unsqueeze(1)
        if target.ndim == 2:
            target = target.unsqueeze(0).unsqueeze(0)
        return target

    def _seg_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = self._prepare_seg_target(logits, target)
        if logits.ndim == 5:
            logits = logits.reshape(-1, *logits.shape[2:])
        if logits.ndim == 3:
            logits = logits.unsqueeze(1)
        if target.ndim == 3:
            target = target.unsqueeze(1)
        if target.shape[-2:] != logits.shape[-2:]:
            target = F.interpolate(target, size=logits.shape[-2:], mode="nearest")
        if target.shape != logits.shape:
            raise ValueError(f"Segmentation target shape {target.shape} does not match logits {logits.shape}")
        loss_bce = self.seg_bce(logits, target)
        loss_dice = self.seg_dice(logits, target)
        return (1.0 - self.dice_weight) * loss_bce + self.dice_weight * loss_dice

    def _cls_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        logits = logits.reshape(-1)
        target = target.float().reshape(-1)
        if logits.shape[0] != target.shape[0]:
            raise ValueError(f"Classification target shape {target.shape} does not match logits {logits.shape}")
        return self.cls_bce(logits, target)

    def forward(self, net_output, target):
        device = net_output["low_res_logits"].device
        tasks = canonicalize_task_batch(net_output.get("task"))
        cls_target = net_output.get("cls_target", None)
        low_res_logits = net_output["low_res_logits"]
        cls_logits = net_output.get("cls_logits", None)

        total_loss = torch.zeros((), device=device)

        for index, task_name in enumerate(tasks):
            spec = get_task_spec(task_name)
            if spec.is_segmentation:
                if low_res_logits.ndim == 5:
                    logits_item = low_res_logits[index:index + 1]
                else:
                    logits_item = low_res_logits[index:index + 1]
                target_item = target[index:index + 1]
                total_loss = total_loss + self._seg_loss(logits_item, target_item)

            if spec.is_classification and cls_logits is not None and cls_target is not None:
                if cls_logits.ndim == 1:
                    logit_item = cls_logits[index:index + 1]
                else:
                    logit_item = cls_logits[index:index + 1]
                cls_item = cls_target[index:index + 1]
                total_loss = total_loss + self._cls_loss(logit_item, cls_item)

        if not bool(net_output.get("enable_auxiliary_losses", True)):
            return total_loss / max(len(tasks), 1)

        seg_proto = net_output.get("seg_proto", None)
        cls_proto = net_output.get("cls_proto", None)
        if seg_proto is not None and cls_proto is not None and bool(net_output.get("enable_prototype_loss", True)):
            proto_loss = 1.0 - F.cosine_similarity(seg_proto, cls_proto, dim=-1).mean()
            total_loss = total_loss + self.proto_weight * proto_loss

        prompt_reg_loss = net_output.get("prompt_reg_loss", None)
        if prompt_reg_loss is not None:
            total_loss = total_loss + self.prompt_weight * prompt_reg_loss

        anchor_loss = net_output.get("anchor_loss", None)
        if anchor_loss is not None:
            total_loss = total_loss + self.anchor_weight * anchor_loss

        mutual_prompt_loss = net_output.get("mutual_prompt_loss", None)
        if mutual_prompt_loss is not None:
            total_loss = total_loss + self.mutual_weight * mutual_prompt_loss

        prototype_agreement = net_output.get("prototype_agreement", None)
        gate_tensors = [
            gate for gate in (
                net_output.get("seg_to_cls_gate", None),
                net_output.get("cls_to_seg_gate", None),
                net_output.get("memory_gate", None),
            )
            if gate is not None
        ]
        if prototype_agreement is not None and gate_tensors:
            reliability_gate = torch.stack(gate_tensors, dim=0).mean(dim=0).clamp(1e-4, 1.0 - 1e-4)
            reliability_target = ((prototype_agreement.detach() + 1.0) * 0.5).clamp(0.0, 1.0)
            prototype_confidence = net_output.get("prototype_confidence", None)
            if prototype_confidence is not None:
                reliability_target = reliability_target * prototype_confidence.detach().clamp(0.0, 1.0)
            total_loss = total_loss + self.reliability_weight * F.binary_cross_entropy(
                reliability_gate,
                reliability_target,
            )
            total_loss = total_loss + self.gate_sparsity_weight * reliability_gate.mean()

        return total_loss / max(len(tasks), 1)


def get_criterion(modelname: str = "UIV-US", opt=None):
    device = torch.device(opt.device)
    pos_weight = torch.ones([1], device=device) * 2.0
    return UnifiedMultiTaskLoss(pos_weight=pos_weight)
