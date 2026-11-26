import torch
from torch import nn
from torch.nn import functional as F


class PromptDeltaBlock(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        in_dim = emb_dim * 4
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(
        self,
        static_prompt: torch.Tensor,
        seg_proto: torch.Tensor,
        cls_proto: torch.Tensor,
        vid_proto: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(torch.cat([static_prompt, seg_proto, cls_proto, vid_proto], dim=-1))


class PrototypeReliabilityGate(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        in_dim = emb_dim * 3 + 1
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, source_proto: torch.Tensor, target_proto: torch.Tensor) -> torch.Tensor:
        agreement = F.cosine_similarity(source_proto, target_proto, dim=-1, eps=1e-6).unsqueeze(-1)
        evidence = torch.cat([source_proto, target_proto, torch.abs(source_proto - target_proto), agreement], dim=-1)
        return self.net(evidence)


class DynamicPromptGenerator(nn.Module):
    def __init__(
        self,
        emb_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        enable_dynamic_prompt: bool = True,
        enable_mutual_prompt: bool = True,
    ) -> None:
        super().__init__()
        self.keys = ("mode", "type", "pos", "task")
        self.internal_keys = {
            "mode": "mode",
            "type": "type_token",
            "pos": "pos",
            "task": "task",
        }
        self.enable_dynamic_prompt = enable_dynamic_prompt
        self.enable_mutual_prompt = enable_mutual_prompt
        self.blocks = nn.ModuleDict(
            {
                self.internal_keys[key]: PromptDeltaBlock(emb_dim, hidden_dim=hidden_dim, dropout=dropout)
                for key in self.keys
            }
        )
        self.seg_from_cls = nn.Sequential(
            nn.LayerNorm(emb_dim * 2),
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
        )
        self.cls_from_seg = nn.Sequential(
            nn.LayerNorm(emb_dim * 2),
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
        )
        self.vid_from_pair = nn.Sequential(
            nn.LayerNorm(emb_dim * 2),
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
        )
        self.seg_to_cls_reliability = PrototypeReliabilityGate(emb_dim, hidden_dim=hidden_dim // 2, dropout=dropout)
        self.cls_to_seg_reliability = PrototypeReliabilityGate(emb_dim, hidden_dim=hidden_dim // 2, dropout=dropout)
        self.memory_reliability = PrototypeReliabilityGate(emb_dim, hidden_dim=hidden_dim // 2, dropout=dropout)

    def _fill_proto(self, maybe_proto: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if maybe_proto is None:
            return torch.zeros_like(reference)
        return maybe_proto

    def forward(
        self,
        static_prompts: dict,
        seg_proto: torch.Tensor = None,
        cls_proto: torch.Tensor = None,
        vid_proto: torch.Tensor = None,
    ) -> dict:
        ref = static_prompts["task"]
        seg_proto = self._fill_proto(seg_proto, ref)
        cls_proto = self._fill_proto(cls_proto, ref)
        vid_proto = self._fill_proto(vid_proto, ref)

        seg_to_cls_gate = self.seg_to_cls_reliability(seg_proto, cls_proto)
        cls_to_seg_gate = self.cls_to_seg_reliability(cls_proto, seg_proto)
        memory_gate = self.memory_reliability(vid_proto, seg_proto)
        prototype_agreement = F.cosine_similarity(seg_proto, cls_proto, dim=-1, eps=1e-6).unsqueeze(-1)

        if self.enable_mutual_prompt:
            seg_delta = self.seg_from_cls(torch.cat([seg_proto, cls_proto], dim=-1))
            cls_delta = self.cls_from_seg(torch.cat([cls_proto, seg_proto], dim=-1))
            vid_delta = self.vid_from_pair(torch.cat([vid_proto, seg_proto + cls_to_seg_gate * seg_delta], dim=-1))
            seg_prompt_proto = seg_proto + cls_to_seg_gate * seg_delta
            cls_prompt_proto = cls_proto + seg_to_cls_gate * cls_delta
            vid_prompt_proto = vid_proto + memory_gate * vid_delta
        else:
            seg_prompt_proto = seg_proto
            cls_prompt_proto = cls_proto
            vid_prompt_proto = vid_proto
            seg_to_cls_gate = torch.zeros_like(seg_to_cls_gate)
            cls_to_seg_gate = torch.zeros_like(cls_to_seg_gate)
            memory_gate = torch.zeros_like(memory_gate)

        outputs = {}
        delta_reg = ref.new_zeros(())
        reliability_mean = (seg_to_cls_gate + cls_to_seg_gate).mul(0.5).squeeze(-1)
        mutual_loss = ((1.0 - F.cosine_similarity(seg_prompt_proto, cls_prompt_proto, dim=-1)) * reliability_mean).mean()
        if not self.enable_dynamic_prompt:
            for key in self.keys:
                outputs[key] = static_prompts[key]
            outputs["delta_reg"] = delta_reg
            outputs["mutual_loss"] = mutual_loss if self.enable_mutual_prompt else ref.new_zeros(())
            outputs["seg_prompt_proto"] = seg_prompt_proto
            outputs["cls_prompt_proto"] = cls_prompt_proto
            outputs["vid_prompt_proto"] = vid_prompt_proto
            outputs["seg_to_cls_gate"] = seg_to_cls_gate
            outputs["cls_to_seg_gate"] = cls_to_seg_gate
            outputs["memory_gate"] = memory_gate
            outputs["prototype_agreement"] = prototype_agreement
            outputs["modality"] = static_prompts["modality"]
            return outputs

        for key in self.keys:
            delta = self.blocks[self.internal_keys[key]](
                static_prompts[key],
                seg_prompt_proto,
                cls_prompt_proto,
                vid_prompt_proto,
            )
            outputs[key] = static_prompts[key] + delta
            delta_reg = delta_reg + delta.pow(2).mean()

        outputs["delta_reg"] = delta_reg / len(self.keys)
        outputs["mutual_loss"] = mutual_loss if self.enable_mutual_prompt else ref.new_zeros(())
        outputs["seg_prompt_proto"] = seg_prompt_proto
        outputs["cls_prompt_proto"] = cls_prompt_proto
        outputs["vid_prompt_proto"] = vid_prompt_proto
        outputs["seg_to_cls_gate"] = seg_to_cls_gate
        outputs["cls_to_seg_gate"] = cls_to_seg_gate
        outputs["memory_gate"] = memory_gate
        outputs["prototype_agreement"] = prototype_agreement
        # Modality is predefined acquisition metadata and remains deterministic.
        outputs["modality"] = static_prompts["modality"]
        return outputs
