import torch
from torch import nn
from torch.nn import functional as F
from typing import Dict, Tuple


class GeMPool2d(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, 1)
        return x.pow(1.0 / self.p).flatten(1)


class PrototypeExtractor(nn.Module):
    def __init__(self, in_dim: int = 256, proto_dim: int = 128) -> None:
        super().__init__()
        self.pool = GeMPool2d()
        self.dense_proj = nn.Sequential(
            nn.Conv2d(in_dim, proto_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, proto_dim),
            nn.GELU(),
        )
        self.global_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, proto_dim),
            nn.GELU(),
            nn.Linear(proto_dim, proto_dim),
        )
        self.object_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, proto_dim),
            nn.GELU(),
            nn.Linear(proto_dim, proto_dim),
        )
        self.background_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, proto_dim),
            nn.GELU(),
            nn.Linear(proto_dim, proto_dim),
        )
        self.task_proj = nn.Sequential(
            nn.LayerNorm(proto_dim * 3),
            nn.Linear(proto_dim * 3, proto_dim),
            nn.GELU(),
            nn.Linear(proto_dim, proto_dim),
        )

    def _default_object_mask(self, feat: torch.Tensor) -> torch.Tensor:
        energy = feat.pow(2).mean(dim=1, keepdim=True).sqrt()
        mean = energy.mean(dim=(2, 3), keepdim=True)
        std = energy.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return torch.sigmoid((energy - mean) / std)

    def _resolve_object_mask(self, feat: torch.Tensor, mask_logits: torch.Tensor = None) -> torch.Tensor:
        if mask_logits is None:
            return self._default_object_mask(feat)
        mask = torch.sigmoid(mask_logits)
        return F.interpolate(mask, size=feat.shape[-2:], mode="bilinear", align_corners=False)

    def _weighted_pool(self, feat: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        weights = weights / (weights.sum(dim=(2, 3), keepdim=True) + 1e-6)
        return (feat * weights).sum(dim=(2, 3))

    def _similarity_map(self, dense_feat: torch.Tensor, proto: torch.Tensor) -> torch.Tensor:
        dense_feat = F.normalize(dense_feat, dim=1, eps=1e-6)
        proto = F.normalize(proto, dim=-1, eps=1e-6)
        return (dense_feat * proto[:, :, None, None]).sum(dim=1, keepdim=True)

    def forward(self, feat: torch.Tensor, mask_logits: torch.Tensor = None) -> dict:
        global_feat = self.pool(feat)
        object_mask = self._resolve_object_mask(feat, mask_logits=mask_logits).clamp(0.0, 1.0)
        background_mask = (1.0 - object_mask).clamp(0.0, 1.0)
        object_feat = self._weighted_pool(feat, object_mask)
        background_feat = self._weighted_pool(feat, background_mask)
        global_proto = self.global_proj(global_feat)
        object_proto = self.object_proj(object_feat)
        background_proto = self.background_proj(background_feat)
        task_proto = self.task_proj(torch.cat([global_proto, object_proto, background_proto], dim=-1))

        dense_proto_feat = self.dense_proj(feat)
        object_similarity = self._similarity_map(dense_proto_feat, object_proto)
        background_similarity = self._similarity_map(dense_proto_feat, background_proto)
        contrast_map = object_similarity - background_similarity
        prototype_region = torch.sigmoid(contrast_map)
        prototype_confidence = contrast_map.abs().mean(dim=(2, 3))
        return {
            "global_feat": global_feat,
            "mask_feat": object_feat,
            "object_feat": object_feat,
            "background_feat": background_feat,
            "semantic_proto": global_proto,
            "structure_proto": object_proto,
            "global_proto": global_proto,
            "mask_proto": object_proto,
            "object_proto": object_proto,
            "background_proto": background_proto,
            "task_proto": task_proto,
            "object_similarity": object_similarity,
            "background_similarity": background_similarity,
            "contrast_map": contrast_map,
            "prototype_region": prototype_region,
            "prototype_confidence": prototype_confidence,
        }


class CrossPrototypeAttention(nn.Module):
    def __init__(self, proto_dim: int = 128, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=proto_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(proto_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(proto_dim),
            nn.Linear(proto_dim, proto_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proto_dim * 2, proto_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm(tokens + attn_out)
        tokens = tokens + self.ffn(tokens)
        return tokens


class TaskInteractionBlock(nn.Module):
    def __init__(
        self,
        feat_dim: int = 256,
        proto_dim: int = 128,
        dropout: float = 0.1,
        enable_interaction: bool = True,
        max_gate: float = 0.5,
    ) -> None:
        super().__init__()
        self.enable_interaction = enable_interaction
        self.max_gate = max_gate
        self.prototype_attention = CrossPrototypeAttention(proto_dim=proto_dim, dropout=dropout)
        self.to_gamma_beta = nn.Sequential(
            nn.LayerNorm(proto_dim),
            nn.Linear(proto_dim, feat_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim * 2, feat_dim * 2),
        )
        self.to_gate = nn.Sequential(
            nn.LayerNorm(proto_dim),
            nn.Linear(proto_dim, feat_dim),
            nn.Sigmoid(),
        )

    def _pack(self, tensors, reference: torch.Tensor) -> torch.Tensor:
        packed = []
        for tensor in tensors:
            if tensor is None:
                packed.append(torch.zeros_like(reference))
            else:
                packed.append(tensor)
        return torch.stack(packed, dim=1)

    def _reliability_weight(self, reliability: Dict[str, torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
        if not reliability:
            return reference.new_ones((reference.shape[0], 1))
        gates = []
        for key in ("seg_to_cls_gate", "cls_to_seg_gate", "memory_gate"):
            gate = reliability.get(key)
            if gate is not None:
                gates.append(gate)
        if not gates:
            return reference.new_ones((reference.shape[0], 1))
        return torch.stack(gates, dim=0).mean(dim=0).clamp(0.0, 1.0)

    def forward(
        self,
        feat: torch.Tensor,
        seg_proto: torch.Tensor = None,
        cls_proto: torch.Tensor = None,
        vid_proto: torch.Tensor = None,
        reliability: Dict[str, torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        reference = seg_proto
        if reference is None:
            reference = cls_proto if cls_proto is not None else vid_proto
        if reference is None:
            pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
            reference = pooled[:, :128]

        tokens = self._pack((seg_proto, cls_proto, vid_proto), reference)
        tokens = self.prototype_attention(tokens)
        fused_proto = tokens.mean(dim=1)

        if not self.enable_interaction:
            return feat, fused_proto

        gamma_beta = self.to_gamma_beta(fused_proto)
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=-1)
        reliability_weight = self._reliability_weight(reliability or {}, fused_proto)
        gate = self.to_gate(fused_proto) * reliability_weight * self.max_gate
        gate = gate.unsqueeze(-1).unsqueeze(-1)

        gamma = torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1)
        beta = torch.tanh(beta).unsqueeze(-1).unsqueeze(-1)
        interacted = feat * (1.0 + gamma) + beta
        feat = gate * interacted + (1.0 - gate) * feat
        return feat, fused_proto
