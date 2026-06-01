"""GPU net reference prep — JPEG/WebP on minimal CPU buffers only."""

from __future__ import annotations

import torch

from ..probe_parity.reference import (
    NET_INPUT_SIZE,
    REF_BITMAP_JPEG_QUALITY,
    REF_WEBP_QUALITY,
    jpeg_roundtrip,
    webp_roundtrip,
)
from .device import get_torch_device, hwc_numpy_to_torch, hwc_torch_to_numpy
from .resize_torch import resize_bilinear_torch


@torch.inference_mode()
def prepare_net_reference_torch(reference_hwc: torch.Tensor | "np.ndarray") -> torch.Tensor:
    """
    Probe reference path: full-res JPEG q92 → 256 high-quality resize → WebP q92.
    JPEG + WebP round-trips stay on CPU for parity; resize runs on GPU.
    """
    import numpy as np

    device = get_torch_device()
    if isinstance(reference_hwc, torch.Tensor):
        ref_np = hwc_torch_to_numpy(reference_hwc)
    else:
        ref_np = np.asarray(reference_hwc, dtype=np.float32)

    decoded = jpeg_roundtrip(ref_np, quality=REF_BITMAP_JPEG_QUALITY)
    ref_t = hwc_numpy_to_torch(decoded, device)
    ref256_t = resize_bilinear_torch(ref_t, NET_INPUT_SIZE, NET_INPUT_SIZE)
    ref256_np = hwc_torch_to_numpy(ref256_t)
    webp_np = webp_roundtrip(ref256_np, quality=REF_WEBP_QUALITY)
    return hwc_numpy_to_torch(webp_np, device)
