"""Torch device selection for MAGICMATCH GPU pipeline."""

from __future__ import annotations

import torch


def get_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def gpu_pipeline_available() -> bool:
    return torch.cuda.is_available()


def hwc_numpy_to_torch(hwc, device: torch.device | None = None) -> torch.Tensor:
    import numpy as np

    device = device or get_torch_device()
    t = torch.from_numpy(np.asarray(hwc, dtype=np.float32)).to(device)
    if t.ndim == 3:
        return t
    raise ValueError(f"expected H×W×3, got {tuple(t.shape)}")


def hwc_torch_to_numpy(t: torch.Tensor):
    return t.detach().float().clamp(0.0, 1.0).cpu().numpy()
