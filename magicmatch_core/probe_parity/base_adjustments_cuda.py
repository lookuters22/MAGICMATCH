"""Base adjustments using CUDA face ONNX (optional GPU path)."""

from __future__ import annotations

import numpy as np

from .scene_extract_cuda import extract_scene_info_bitmap_cuda
from .scene_extract import get_original_color_match_base_adjustments


def estimate_base_adjustments_cuda(
    source_hwc: np.ndarray,
    *,
    worker_feed_prepared: bool = False,
) -> dict[str, float]:
    scene = extract_scene_info_bitmap_cuda(source_hwc, worker_feed_prepared=worker_feed_prepared)
    return get_original_color_match_base_adjustments(scene)
