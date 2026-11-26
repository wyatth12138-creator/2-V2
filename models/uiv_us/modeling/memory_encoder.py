import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryEncoder(nn.Module):
    """
    Encode (image feature, mask) -> spatial memory feature map.

    Inputs:
      feat: [B, 256, 32, 32]  (your imge feature per frame)
      mask: [B, 1, 128, 128]  (GT or predicted low-res mask; can be prob or 0/1)
    Output:
      mem:  [B, mem_dim, 32, 32]
    """
    def __init__(self, feat_dim=256, mem_dim=64):
        super().__init__()
        self.mem_dim = mem_dim

        # Project feature to mem_dim
        self.feat_proj = nn.Sequential(
            nn.Conv2d(feat_dim, mem_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, mem_dim),
            nn.GELU(),
        )

        # Process mask (downsample then conv)
        self.mask_proj = nn.Sequential(
            nn.Conv2d(1, mem_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, mem_dim),
            nn.GELU(),
            nn.Conv2d(mem_dim, mem_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, mem_dim),
            nn.GELU(),
        )

        # Fuse
        self.fuse = nn.Sequential(
            nn.Conv2d(mem_dim, mem_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, mem_dim),
            nn.GELU(),
        )

    def forward(self, feat, mask):
        # feat: [B,256,32,32]
        # mask: [B,1,128,128] (float, prob or 0/1)
        B, _, Hf, Wf = feat.shape
        if mask.shape[-2:] != (Hf, Wf):
            # Downsample mask to feature resolution
            mask = F.interpolate(mask, size=(Hf, Wf), mode="bilinear", align_corners=False)

        feat_m = self.feat_proj(feat)     # [B,mem_dim,32,32]
        mask_m = self.mask_proj(mask)     # [B,mem_dim,32,32]

        mem = self.fuse(feat_m + mask_m)  # [B,mem_dim,32,32]
        return mem
