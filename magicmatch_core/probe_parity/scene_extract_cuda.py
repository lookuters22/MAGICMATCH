"""Probe-style scene extraction with CUDA face ONNX sessions."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .calibration_utils import get_masked_pixels
from .calibration_wb import get_auto_wb_params_for_color_match, get_robust_skin_illuminant
from .color_match_features import extract_color_match_features
from .face_detection_cuda import detect_faces_cuda
from .luminance import (
    LuminanceStatistics,
    get_adjusted_gamma,
    get_auto_light_params,
    get_face_area_luminance_statistics,
    get_luminance_statistics,
)
from .scene_extract import (
    BITMAP_AS_SHOT_TEMP,
    BITMAP_AS_SHOT_TINT,
    SceneInfo,
    render_detection_inputs,
)


def extract_scene_info_bitmap_cuda(
    source_hwc: np.ndarray,
    *,
    as_shot_temperature: float = BITMAP_AS_SHOT_TEMP,
    as_shot_tint: float = BITMAP_AS_SHOT_TINT,
    source_path: Path | str | None = None,
    use_probe_browser: bool | None = None,
    worker_feed_prepared: bool = False,
    use_gpu_detection_buffers: bool = False,
) -> SceneInfo:
    """Bitmap scene extract; face detect/parse ONNX runs on CUDA when available."""
    if use_probe_browser is None:
        use_probe_browser = os.environ.get("MAGICMATCH_PROBE_BROWSER", "0") == "1"
    if use_probe_browser:
        from .probe_browser_scene import extract_scene_via_probe_browser

        path = Path(source_path) if source_path is not None else None
        return extract_scene_via_probe_browser(path or source_hwc)

    if use_gpu_detection_buffers:
        from ..gpu.device import gpu_pipeline_available
        from ..gpu.reference_torch import render_detection_inputs_numpy

        if gpu_pipeline_available():
            feed = source_hwc
            if not worker_feed_prepared:
                from .reference import prepare_worker_bitmap_source

                feed = prepare_worker_bitmap_source(source_hwc)
            small, large = render_detection_inputs_numpy(feed)
        else:
            small, large = render_detection_inputs(source_hwc, worker_feed_prepared=worker_feed_prepared)
    else:
        small, large = render_detection_inputs(source_hwc, worker_feed_prepared=worker_feed_prepared)
    stats_noface = get_luminance_statistics(small, large, [])
    adjusted_gamma = get_adjusted_gamma(stats_noface)

    face_results = detect_faces_cuda(large, adjusted_gamma=adjusted_gamma)
    face_percent_and_pixels = None
    if face_results:
        face_percent_and_pixels = get_masked_pixels(large, face_results, True)

    color_features, face_data = extract_color_match_features(
        small,
        large,
        face_results,
        is_reference=False,
        face_percent_and_pixels=face_percent_and_pixels,
    )
    filtered_faces = face_data["filteredFaceDetectionResults"]

    if filtered_faces and face_percent_and_pixels is None:
        face_percent_and_pixels = get_masked_pixels(large, filtered_faces, True)

    face_area = get_face_area_luminance_statistics(
        large,
        filtered_faces,
        face_percent_and_pixels,
    )
    luminance_stats = LuminanceStatistics(
        avg_lum=stats_noface.avg_lum,
        shadows_mean=stats_noface.shadows_mean,
        clipping_percent=stats_noface.clipping_percent,
        percentiles=stats_noface.percentiles,
        face_percentiles=face_area["facePercentiles"],
        face_percent=float(face_area["facePercent"]),
        face_lum=float(face_area["faceLum"]),
    )
    avg_face = color_features.get("avgFaceHsvl")
    auto_light = get_auto_light_params(luminance_stats, avg_face)

    illuminant = get_robust_skin_illuminant(
        small,
        large,
        filtered_faces,
        face_percent=float(color_features.get("facePercent") or 0.0),
        face_colors=face_data.get("faceColors"),
        face_weights=face_data.get("faceWeights"),
    )
    auto_wb = get_auto_wb_params_for_color_match(
        illuminant,
        as_shot_temperature=as_shot_temperature,
        as_shot_tint=as_shot_tint,
    )

    base = {**auto_light, "temperature": auto_wb["temperature"], "tint": auto_wb["tint"]}
    return SceneInfo(
        auto_light_params=auto_light,
        auto_wb_params=auto_wb,
        base_adjustments=base,
        face_results=filtered_faces,
        avg_face_hsvl=avg_face,
    )
