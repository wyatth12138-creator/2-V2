"""Strictly load compatible UIV-US checkpoints and report newly initialized weights."""

import torch


def load_checkpoint(model, path, *, allow_new_observer=False):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(torch.is_tensor(value) for value in state.values()):
        raise ValueError(f"Expected a UIV-US model state dictionary in {path}")
    if allow_new_observer and not any(key.startswith("temporal_observer.") for key in state):
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected or any(not key.startswith("temporal_observer.") for key in missing):
            raise RuntimeError(f"Incompatible UIV-US checkpoint. Missing={missing}, unexpected={unexpected}")
        if missing:
            print(f"Initialized {len(missing)} observer tensors from the V2 defaults.", flush=True)
    else:
        model.load_state_dict(state)
