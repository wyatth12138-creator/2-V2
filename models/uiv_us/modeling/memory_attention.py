import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryAttention(nn.Module):
    """
    Query: current frame feature  feat: [B, 256, 32, 32]
    Memory tokens: mem: [B, Lm, 64]  (Lm can be padded)
    Output: fused feature [B, 256, 32, 32]
    """
    def __init__(self, feat_dim=256, mem_dim=64, num_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.feat_dim = feat_dim
        self.mem_dim = mem_dim

        self.q_ln = nn.LayerNorm(feat_dim)
        self.kv_ln = nn.LayerNorm(feat_dim)

        # project memory token dim (64) -> feat_dim (256) for attention K/V
        self.mem_to_kv = nn.Linear(mem_dim, feat_dim, bias=False)

        self.attn = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.proj_drop = nn.Dropout(dropout)

        hidden = int(feat_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, feat_dim),
            nn.Dropout(dropout),
        )

    def forward(self, feat, mem_tokens, mem_key_padding_mask=None):
        """
        feat: [B, C, H, W]
        mem_tokens: [B, Lm, mem_dim]
        mem_key_padding_mask: [B, Lm] bool (True means padding, should be ignored)
        """
        B, C, H, W = feat.shape
        assert C == self.feat_dim

        # flatten query tokens
        q = feat.flatten(2).transpose(1, 2)   # [B, Lq, C], Lq=H*W
        q = self.q_ln(q)

        # memory -> kv dim
        kv = self.mem_to_kv(mem_tokens)       # [B, Lm, C]
        kv = self.kv_ln(kv)

        # cross-attention: query attends to memory
        attn_out, _ = self.attn(
            query=q,
            key=kv,
            value=kv,
            key_padding_mask=mem_key_padding_mask,  # True means ignore
            need_weights=False,
        )
        q = q + self.proj_drop(attn_out)
        q = q + self.mlp(q)

        # reshape back
        out = q.transpose(1, 2).reshape(B, C, H, W)
        return out
