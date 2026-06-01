"""GPU image resize helpers (approximate probe LANCZOS + TF bilinear)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

IMAGE_SIZE = 256


def _bchw(hwc: torch.Tensor) -> torch.Tensor:
    return hwc.permute(2, 0, 1).unsqueeze(0)


def _hwc(bchw: torch.Tensor) -> torch.Tensor:
    return bchw.squeeze(0).permute(1, 2, 0)


def fit_long_edge_torch(hwc: torch.Tensor, long_edge: int) -> torch.Tensor:
    h, w, _ = hwc.shape
    nw = max(1, math.ceil(w * min(long_edge / w, long_edge / h)))
    nh = max(1, math.ceil(h * min(long_edge / w, long_edge / h)))
    if nw == w and nh == h:
        return hwc
    out = F.interpolate(_bchw(hwc), size=(nh, nw), mode="bicubic", align_corners=False)
    return _hwc(out).clamp(0.0, 1.0)


def resize_bilinear_torch(hwc: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    out = F.interpolate(_bchw(hwc), size=(out_h, out_w), mode="bilinear", align_corners=False)
    return _hwc(out).clamp(0.0, 1.0)


def hwc_to_nhwc_numpy(hwc: torch.Tensor, size: int = IMAGE_SIZE):
    """createTensor path: high-quality downscale then bilinear 256×256 → ORT feed."""
    h, w, _ = hwc.shape
    if h != size or w != size:
        hwc = resize_bilinear_torch(hwc, size, size)
    return hwc.unsqueeze(0).detach().cpu().numpy().astype("float32", copy=False)
