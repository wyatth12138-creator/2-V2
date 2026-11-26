import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple


# prompt info dict
# task prompt
position_prompt_list = [
    "segmentation",
    "classification",
    "videos segmentation",
    "videos classification",
]
# position prompt
position_prompt_dict = {
    "MMOTU": "ovary",
    "BUSI": "breast",
    "TN3K": "thyroid",
    "CUBS": "carotid",
    "BUS-BRA": "breast",
    "Appendix": "appendix",
    "Fatty-Liver": "liver",
    "UDIAT": "breast",
    "DDTI": "thyroid",
    "HMC-QU": "cardiac",
    "Fetal_HC": "head",
    "BUSIS": "breast",
    "CCAU": "carotid",
    "kidneyUS_capsule": "kidney",
    "TG3K": "thyroid",
    "EchoNet-Dynamic": "cardiac",
    "EchoNet": "cardiac",
    "CAMUS": "cardiac",
    "Seven_Breast_Full": "breast",
    "Seven_Fetal_head_Full": "head",
    "Seven_Heart_Full": "cardiac",
    "Seven_Kidney_Full": "kidney",
    "Seven_Thyroid_Full": "thyroid",
    "VSegCA" : "carotid",
    "VSegIJV" : "carotid",
    "ZhangCA": "carotid",
    "JH_HC": "head",
    "Macao1": "head",
    "Breast": "breast",
    "Ty": "thyroid",
    "Thyroid": "thyroid",
    "CEUS_Breast": "breast",
    "CEUS_Ty": "thyroid",
    "Liver": "liver",
    "Prostate": "prostate",
}

# type prompt
type_prompt_dict = {
    "DDTI": "tumor",
    "MMOTU": "tumor",
    "BUSI": "tumor",
    "HMC-QU": "organ",
    "TN3K": "tumor",
    "Fetal_HC": "organ",
    "BUSIS": "tumor",
    "CCAU": "organ",
    "CUBS": "organ",
    "BUS-BRA": "tumor",
    "kidneyUS_capsule": "organ",
    "TG3K": "organ",
    "EchoNet-Dynamic": "organ",
    "UDIAT": "tumor",
    "Appendix": "organ",
    "Fatty-Liver": "organ",
    "EchoNet": "organ",
    "CAMUS": "organ",
    "Seven_Breast_Full": "tumor",
    "Seven_Fetal_head_Full": "organ",
    "Seven_Heart_Full": "organ",
    "Seven_Kidney_Full": "organ",
    "Seven_Thyroid_Full": "tumor",
    "VSegCA": "organ",
    "VSegIJV": "organ",
    "ZhangCA": "organ",
    "JH_HC": "organ",
    "Macao1": "organ",
    "Breast": "tumor",
    "Ty": "tumor",
    "Thyroid": "tumor",
    "CEUS_Breast": "tumor",
    "CEUS_Ty": "tumor",
    "Liver": "organ",
    "Prostate": "organ",
}

# mode prompt
available_mode_prompt_list = [
    "DDTI",
    "MMOTU",
    "BUSI",
    "HMC-QU",
    "TN3K",
    "Fetal_HC",
    "BUSIS",
    "CCAU",
    "CUBS",
    "BUS-BRA",
    "kidneyUS_capsule",
    "TG3K",
    "EchoNet-Dynamic",
    "UDIAT",
    "EchoNet",
    "CAMUS",
    "Seven_Breast_Full",
    "Seven_Fetal_head_Full",
    "Seven_Heart_Full",
    "Seven_Kidney_Full",
    "Seven_Thyroid_Full",
    "VSegCA",
    "VSegIJV",
    "ZhangCA",
]


# prompt one-hot
# organ prompt
position_prompt_one_hot_dict = {
    "ovary":    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "breast":   [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "thyroid":  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "carotid":  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "appendix": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    "liver":    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    "cardiac":  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    "head":     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    "kidney":   [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    "leg":      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    "prostate": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    "indis":    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
}


# task prompt
# task_prompt_one_hot_dict = {"segmentation": [1, 0], "classification": [0, 1]}

task_prompt_one_hot_dict = {
    "segmentation": [1, 0, 0, 0],
    "classification": [0, 1, 0, 0],
    "videos segmentation": [0, 0, 1, 0],
    "videos classification": [0, 0, 0, 1],
}

# type prompt
type_prompt_one_hot_dict = {
    "tumor": [1, 0],
    "organ": [0, 1],
}
# mode prompt
mode_prompt_one_hot_dict = {
    "whole": [1, 0, 0, 0],
    "local": [0, 1, 0, 0],
    "location": [0, 0, 1, 0],
    "full": [0, 0, 0, 1],
}

modality_prompt_one_hot_dict = {
    "bmode": [1, 0],
    "ceus": [0, 1],
}


class PromptBank(nn.Module):
    def __init__(self, d_modality, d_mode, d_type, d_pos, d_task, emb=128):
        super().__init__()
        self.d_modality = int(d_modality)
        self.W_modality = nn.Parameter(torch.randn(d_modality, emb) * 0.02)
        self.W_mode = nn.Parameter(torch.randn(d_mode, emb) * 0.02)
        self.W_type = nn.Parameter(torch.randn(d_type, emb) * 0.02)
        self.W_pos  = nn.Parameter(torch.randn(d_pos,  emb) * 0.02)
        self.W_task = nn.Parameter(torch.randn(d_task, emb) * 0.02)

    def forward(self, prompts):
        # prompts[k]: (B, d_k) onehot/multihot
        modality_prompt = prompts.get("modality_prompt")
        if modality_prompt is None:
            reference = prompts["mode_prompt"]
            modality_prompt = reference.new_zeros((reference.shape[0], self.d_modality))
        e_modality = modality_prompt @ self.W_modality
        e_mode = prompts["mode_prompt"] @ self.W_mode  # (B, emb)
        e_type = prompts["type_prompt"] @ self.W_type
        e_pos  = prompts["position_prompt"] @ self.W_pos
        e_task = prompts["task_prompt"] @ self.W_task
        return {"modality": e_modality, "mode": e_mode, "type": e_type, "pos": e_pos, "task": e_task}


class ModeFiLM(nn.Module):
    def __init__(self, emb_dim, channels):
        super().__init__()
        self.to_gb = nn.Sequential(
            nn.Linear(emb_dim, 2*channels),
            nn.GELU(),
            nn.Linear(2*channels, 2*channels),
        )

    def forward(self, h, e_mode):
        # h: (B,C,H,W), e_mode: (B,emb)
        B,C,_,_ = h.shape
        gb = self.to_gb(e_mode)          # (B,2C)
        gamma, beta = gb[:, :C], gb[:, C:]
        gamma = torch.tanh(gamma).view(B,C,1,1)
        beta  = beta.view(B,C,1,1)
        return h * (1.0 + gamma) + beta


class PositionSpatialBias(nn.Module):
    def __init__(self, emb_dim, channels):
        super().__init__()
        self.to_c = nn.Sequential(nn.Linear(emb_dim, channels), nn.GELU(), nn.Linear(channels, channels))

    def forward(self, h, e_pos):
        # h: (B,C,H,W)
        B,C,H,W = h.shape
        c = self.to_c(e_pos).view(B,C,1,1)  # (B,C,1,1)

        # Fixed two-dimensional coordinate basis.
        yy = torch.linspace(-1, 1, H, device=h.device).view(1,1,H,1)
        xx = torch.linspace(-1, 1, W, device=h.device).view(1,1,1,W)
        basis = (yy + xx) / 2.0  # (1,1,H,W)

        bias = c * basis  # (B,C,H,W)
        return h + bias


class FiLM2d(nn.Module):
    """x: (B,C,H,W), e: (B,E) -> channel-wise gamma/beta"""
    def __init__(self, channels: int, emb: int, hidden: int = 256, p_drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb, hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, 2 * channels),
        )

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        film = self.net(e)                 # (B, 2C)
        gamma, beta = film[:, :C], film[:, C:]
        gamma = torch.tanh(gamma)          # stabilize
        gamma = gamma.view(B, C, 1, 1)
        beta  = beta.view(B, C, 1, 1)
        return x * (1.0 + gamma) + beta


class SpatialBias2d(nn.Module):
    """
    pos embedding -> spatial bias map, add to feature map
    x: (B,C,H,W), e_pos: (B,E)
    """
    def __init__(self, channels: int, emb: int, hidden: int = 256, p_drop: float = 0.1):
        super().__init__()
        self.to_map = nn.Sequential(
            nn.Linear(emb, hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: torch.Tensor, e_pos: torch.Tensor) -> torch.Tensor:
        # produce per-channel bias, broadcast spatially
        b = self.to_map(e_pos)  # (B,C)
        return x + b.unsqueeze(-1).unsqueeze(-1)
