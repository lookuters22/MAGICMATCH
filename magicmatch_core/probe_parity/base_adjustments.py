"""Base adjustments for ONNX net feed + export (probe bitmap path)."""

from __future__ import annotations

import numpy as np

from .scene_extract import extract_scene_info_bitmap, get_original_color_match_base_adjustments


def estimate_base_adjustments(
    source_hwc: np.ndarray,
    *,
    worker_feed_prepared: bool = False,
) -> dict[str, float]:
    """
    Exact port of getOriginalColorMatchBaseAdjustments(extractSceneInfo())
    for Comfy bitmap/JPEG sources (asShot 5000K / tint 0).
    """
    scene = extract_scene_info_bitmap(source_hwc, worker_feed_prepared=worker_feed_prepared)
    return get_original_color_match_base_adjustments(scene)
