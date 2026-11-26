from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import torch

@dataclass
class MemoryItem:
    spatial: torch.Tensor                 # [1,Cm,Hm,Wm]
    pointers: Optional[torch.Tensor]      # [1,P,Cp] e.g. [1,4,64]
    frame_idx: int
    prompted: bool
    extra: Optional[Dict[str, Any]] = None


class MemoryBank:
    def __init__(self, max_cond: int = 10, max_recent: int = 10, detach: bool = True):
        self.max_cond = int(max_cond)
        self.max_recent = int(max_recent)
        self.detach = bool(detach)
        self._bank: Dict[int, Dict[str, List[MemoryItem]]] = {}

    def reset(self, video_ids: torch.Tensor):
        for vid in video_ids.detach().cpu().tolist():
            self._bank[int(vid)] = {"cond": [], "recent": []}

    def _ensure(self, vid: int):
        if vid not in self._bank:
            self._bank[vid] = {"cond": [], "recent": []}

    @torch.no_grad()
    def push_spatial(
        self,
        video_ids: torch.Tensor,         # [B]
        spatial_bt: torch.Tensor,        # [B,C,H,W]
        frame_idx: int,
        prompted_mask: torch.Tensor,     # [B] bool
        pointers_bt: Optional[torch.Tensor] = None,
        extra: Optional[Dict[str, Any]] = None,
        write_mask: Optional[torch.Tensor] = None,
    ):
        assert video_ids.ndim == 1
        B = video_ids.shape[0]
        assert spatial_bt.shape[0] == B
        assert prompted_mask.shape[0] == B

        spatial_store = spatial_bt.detach() if self.detach else spatial_bt
        vids = video_ids.detach().cpu().tolist()
        prompted_list = prompted_mask.detach().cpu().tolist()
        ptr_store = None
        if pointers_bt is not None:
            ptr_store = pointers_bt.detach() if self.detach else pointers_bt

        writes = [True] * B if write_mask is None else write_mask.detach().cpu().tolist()
        for i in range(B):
            if not writes[i]:
                continue
            extra_i = None if extra is None else {
                key: value[i:i+1] if torch.is_tensor(value) and value.ndim and value.shape[0] == B else value
                for key, value in extra.items()
            }
            vid = int(vids[i])
            self._ensure(vid)
            item = MemoryItem(
                spatial=spatial_store[i].unsqueeze(0),  # [1,C,H,W]
                pointers=None if ptr_store is None else ptr_store[i].unsqueeze(0),
                frame_idx=int(frame_idx),
                prompted=bool(prompted_list[i]),
                extra=extra_i,
            )
            qname = "cond" if item.prompted else "recent"
            q = self._bank[vid][qname]
            q.append(item)
            if qname == "cond" and len(q) > self.max_cond:
                q.pop(0)
            if qname == "recent" and len(q) > self.max_recent:
                q.pop(0)

    def get(self, vid: int):
        self._ensure(int(vid))
        return self._bank[int(vid)]

    def summary(self, vid: int):
        b = self.get(int(vid))
        return {
            "cond_len": len(b["cond"]),
            "recent_len": len(b["recent"]),
            "cond_frames": [it.frame_idx for it in b["cond"]],
            "recent_frames": [it.frame_idx for it in b["recent"]],
        }
