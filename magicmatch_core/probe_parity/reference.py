"""Image sizing aligned with Polarr exportImageForColorMatch / resizePixelDataToBitmap."""

from __future__ import annotations

import numpy as np

NET_LONG_EDGE = 1600
NET_INPUT_SIZE = 256
REF_LONG_EDGE = 1600


def fit_to_size(width: int, height: int, box: tuple[int, int]) -> tuple[int, int]:
    """Contain fit — same geometry as Polarr fitToSize."""
    max_w, max_h = box
    scale = min(max_w / width, max_h / height)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def resize_hwc(hwc: np.ndarray, width: int, height: int, *, high_quality: bool = True) -> np.ndarray:
    """Resize H×W×3 float RGB [0,1]. high_quality ≈ probe resizePixelDataToBitmap('high')."""
    from PIL import Image

    hwc = np.asarray(hwc, dtype=np.float32)
    arr = (np.clip(hwc, 0.0, 1.0) * 255.0).astype(np.uint8)
    resample = Image.Resampling.LANCZOS if high_quality else Image.Resampling.BILINEAR
    pil = Image.fromarray(arr, "RGB").resize((width, height), resample)
    return np.asarray(pil, dtype=np.float32) / 255.0


def fit_long_edge(hwc: np.ndarray, long_edge: int) -> np.ndarray:
    h, w, _ = hwc.shape
    if max(h, w) <= long_edge:
        return np.asarray(hwc, dtype=np.float32).copy()
    nw, nh = fit_to_size(w, h, (long_edge, long_edge))
    return resize_hwc(hwc, nw, nh, high_quality=True)


def prepare_net_reference(reference_hwc: np.ndarray) -> np.ndarray:
    """Reference → longest edge 1600 → 256 (probe ref cache path)."""
    ref = fit_long_edge(reference_hwc, REF_LONG_EDGE)
    return resize_hwc(ref, NET_INPUT_SIZE, NET_INPUT_SIZE, high_quality=True)
