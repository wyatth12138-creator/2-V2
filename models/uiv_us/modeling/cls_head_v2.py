import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34
from torchvision.models.feature_extraction import create_feature_extractor

from utils.prompt_list import FiLM2d, SpatialBias2d

try:
    import timm
except ImportError:  # pragma: no cover - runtime fallback for environments without timm
    timm = None


def proj1x1(in_ch, out_ch=256, p_drop=0.1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.GELU(),
        nn.Dropout(p_drop),
    )


class GeMPool2d(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, 1)
        return x.pow(1.0 / self.p).flatten(1)


def _load_checkpoint(path: str) -> Dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "Loading a .safetensors ResNet checkpoint requires the safetensors package. "
                "Install it from an offline/local wheel, or provide a .pth checkpoint."
            ) from exc
        return load_file(path)
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def _strip_known_prefixes(key: str) -> str:
    prefixes = (
        "module.",
        "model.",
        "backbone.",
        "raw_backbone.backbone.",
        "cls_head.raw_backbone.backbone.",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def load_resnet_backbone(backbone: nn.Module, checkpoint_path: Optional[str]) -> None:
    if not checkpoint_path:
        return
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"ResNet checkpoint not found: {checkpoint_path}")

    checkpoint = _load_checkpoint(checkpoint_path)
    model_state = backbone.state_dict()
    matched = {}
    skipped = []
    for key, value in checkpoint.items():
        stripped_key = _strip_known_prefixes(key)
        if stripped_key in model_state and tuple(model_state[stripped_key].shape) == tuple(value.shape):
            matched[stripped_key] = value
        else:
            skipped.append(key)

    if not matched:
        raise RuntimeError(f"No compatible ResNet weights found in checkpoint: {checkpoint_path}")
    missing, unexpected = backbone.load_state_dict(matched, strict=False)
    print(
        f"Loaded ResNet weights from {checkpoint_path}: "
        f"{len(matched)} tensors matched, {len(missing)} missing, {len(unexpected)} unexpected, {len(skipped)} skipped."
    )


class TimmResNet34Feat(nn.Module):
    def __init__(self, out_dim=256, pretrained=False, pretrained_ckpt=None, in_chans=3, p_drop=0.1):
        super().__init__()
        self.use_timm = timm is not None
        if self.use_timm:
            self.backbone = timm.create_model(
                "resnet34",
                pretrained=pretrained,
                in_chans=in_chans,
                features_only=True,
                out_indices=(2, 3, 4),
            )
            ch3, ch4, ch5 = self.backbone.feature_info.channels()
        else:
            base_model = resnet34(weights=None)
            if in_chans != 3:
                base_model.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.backbone = create_feature_extractor(
                base_model,
                return_nodes={"layer2": "f3", "layer3": "f4", "layer4": "f5"},
            )
            ch3, ch4, ch5 = 128, 256, 512
        load_resnet_backbone(self.backbone, pretrained_ckpt)
        self.p3 = proj1x1(ch3, 256, p_drop=p_drop)
        self.p4 = proj1x1(ch4, 256, p_drop=p_drop)
        self.p5 = proj1x1(ch5, 256, p_drop=p_drop)
        self.raw_fuse = nn.Sequential(
            nn.Conv2d(256 * 3, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_timm:
            f3, f4, f5 = self.backbone(x)
        else:
            feats = self.backbone(x)
            f3, f4, f5 = feats["f3"], feats["f4"], feats["f5"]
        p3 = self.p3(f3)
        p4 = F.interpolate(self.p4(f4), size=p3.shape[-2:], mode="bilinear", align_corners=False)
        p5 = F.interpolate(self.p5(f5), size=p3.shape[-2:], mode="bilinear", align_corners=False)
        return self.raw_fuse(torch.cat([p3, p4, p5], dim=1))


class PrototypeAwareClsHead(nn.Module):
    def __init__(
        self,
        num_classes: int,
        emb: int = 128,
        in_chans: int = 1,
        pretrained_resnet: bool = True,
        pretrained_ckpt: Optional[str] = None,
        p_drop: float = 0.1,
        proto_dim: int = 128,
    ) -> None:
        super().__init__()
        self.raw_backbone = TimmResNet34Feat(
            pretrained=pretrained_resnet,
            pretrained_ckpt=pretrained_ckpt,
            in_chans=3,
            out_dim=256,
            p_drop=p_drop,
        )
        self.sam_adapter = nn.Sequential(
            nn.Conv2d(256, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        self.raw_mode_film = FiLM2d(256, emb, p_drop=p_drop)
        self.raw_modality_film = FiLM2d(256, emb, p_drop=p_drop)
        self.raw_type_film = FiLM2d(256, emb, p_drop=p_drop)
        self.raw_task_film = FiLM2d(256, emb, p_drop=p_drop)
        self.raw_pos_bias = SpatialBias2d(256, emb, p_drop=p_drop)
        self.sam_mode_film = FiLM2d(256, emb, p_drop=p_drop)
        self.sam_modality_film = FiLM2d(256, emb, p_drop=p_drop)
        self.sam_type_film = FiLM2d(256, emb, p_drop=p_drop)
        self.sam_task_film = FiLM2d(256, emb, p_drop=p_drop)
        self.sam_pos_bias = SpatialBias2d(256, emb, p_drop=p_drop)
        self.fuse = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Dropout(p_drop),
        )
        self.pool = GeMPool2d()
        self.seg_proto_proj = nn.Sequential(
            nn.LayerNorm(proto_dim),
            nn.Linear(proto_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(256 * 3),
            nn.Linear(256 * 3, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        self.proto_head = nn.Sequential(
            nn.LayerNorm(256 * 2),
            nn.Linear(256 * 2, 256),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(256, proto_dim),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(proto_dim),
            nn.Linear(proto_dim, 256),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(256, num_classes),
        )

    def _inject_prompts(self, feat: torch.Tensor, prompts: Optional[Dict[str, torch.Tensor]], branch: str) -> torch.Tensor:
        if prompts is None:
            return feat
        if branch == "raw":
            if prompts.get("modality") is not None:
                feat = self.raw_modality_film(feat, prompts["modality"])
            if prompts.get("mode") is not None:
                feat = self.raw_mode_film(feat, prompts["mode"])
            if prompts.get("pos") is not None:
                feat = self.raw_pos_bias(feat, prompts["pos"])
            if prompts.get("type") is not None:
                feat = self.raw_type_film(feat, prompts["type"])
            if prompts.get("task") is not None:
                feat = self.raw_task_film(feat, prompts["task"])
        else:
            if prompts.get("modality") is not None:
                feat = self.sam_modality_film(feat, prompts["modality"])
            if prompts.get("mode") is not None:
                feat = self.sam_mode_film(feat, prompts["mode"])
            if prompts.get("pos") is not None:
                feat = self.sam_pos_bias(feat, prompts["pos"])
            if prompts.get("type") is not None:
                feat = self.sam_type_film(feat, prompts["type"])
            if prompts.get("task") is not None:
                feat = self.sam_task_film(feat, prompts["task"])
        return feat

    def _mask_guided_pool(self, feat_map: torch.Tensor, mask_logits: torch.Tensor, detach_mask: bool = True) -> torch.Tensor:
        mask = torch.sigmoid(mask_logits)
        mask = F.interpolate(mask, size=feat_map.shape[-2:], mode="bilinear", align_corners=False)
        if detach_mask:
            mask = mask.detach()
        weights = mask / (mask.sum(dim=(2, 3), keepdim=True) + 1e-6)
        return (feat_map * weights).sum(dim=(2, 3))

    def forward(
        self,
        imgs: torch.Tensor,
        imge: torch.Tensor,
        mask_logits: Optional[torch.Tensor] = None,
        prompts: Optional[Dict[str, torch.Tensor]] = None,
        use_roi: Optional[torch.Tensor] = None,
        seg_proto: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if imgs.dim() == 4 and imgs.size(1) == 1:
            imgs = imgs.repeat(1, 3, 1, 1)

        raw = self._inject_prompts(self.raw_backbone(imgs), prompts, branch="raw")
        sam = self._inject_prompts(self.sam_adapter(imge), prompts, branch="sam")
        if raw.shape[-2:] != sam.shape[-2:]:
            sam = F.interpolate(sam, size=raw.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.fuse(torch.cat([raw, sam], dim=1))
        global_feat = self.pool(fused)

        if mask_logits is not None and use_roi is not None and bool(use_roi.any()):
            roi_feat = self._mask_guided_pool(fused, mask_logits, detach_mask=True)
        else:
            roi_feat = global_feat

        if seg_proto is None:
            seg_proto_feat = torch.zeros_like(global_feat)
        else:
            seg_proto_feat = self.seg_proto_proj(seg_proto)

        gate = self.gate(torch.cat([global_feat, roi_feat, seg_proto_feat], dim=-1))
        fused_feat = gate * roi_feat + (1.0 - gate) * global_feat + 0.1 * seg_proto_feat
        cls_proto = self.proto_head(torch.cat([fused_feat, seg_proto_feat], dim=-1))
        cls_logits = self.classifier(cls_proto)
        return {
            "cls_logits": cls_logits,
            "cls_proto": cls_proto,
            "cls_feature": fused_feat,
            "global_feature": global_feat,
            "roi_feature": roi_feat,
        }
