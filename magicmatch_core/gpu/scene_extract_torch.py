"""Full GPU scene extract — detection buffers stay on device until scalar/ORT sync."""

from __future__ import annotations

import torch

from ..probe_parity.calibration_utils import get_masked_pixels
from ..probe_parity.calibration_wb import get_auto_wb_params_for_color_match, get_robust_skin_illuminant
from ..probe_parity.color_match_features import extract_color_match_features
from ..probe_parity.luminance import get_adjusted_gamma, get_auto_light_params, get_face_area_luminance_statistics
from ..probe_parity.scene_extract import (
    BITMAP_AS_SHOT_TEMP,
    BITMAP_AS_SHOT_TINT,
    SceneInfo,
    get_original_color_match_base_adjustments,
)
from .device import gpu_pipeline_available, hwc_torch_to_numpy
from .luminance_torch import get_luminance_statistics_torch
from .reference_torch import render_detection_inputs_torch


@torch.inference_mode()
def extract_scene_info_bitmap_full_gpu(
    feed_tensor: torch.Tensor,
    *,
    as_shot_temperature: float = BITMAP_AS_SHOT_TEMP,
    as_shot_tint: float = BITMAP_AS_SHOT_TINT,
) -> tuple[SceneInfo, torch.Tensor, torch.Tensor]:
    """
    GPU scene extract from worker feed tensor on device.
    Downloads detection-sized buffers once for face/color/WB parity helpers.
    """
    if not gpu_pipeline_available():
        from ..probe_parity.scene_extract import extract_scene_info_bitmap

        feed_np = hwc_torch_to_numpy(feed_tensor)
        return extract_scene_info_bitmap(feed_np, worker_feed_prepared=True), feed_tensor, feed_tensor

    small_t, large_t = render_detection_inputs_torch(feed_tensor)
    stats_noface = get_luminance_statistics_torch(small_t, large_t, [])
    adjusted_gamma = get_adjusted_gamma(stats_noface)

    from ..probe_parity.face_detection_cuda import detect_faces_cuda_from_tensor

    face_results = detect_faces_cuda_from_tensor(large_t, adjusted_gamma=adjusted_gamma)

    small_np = hwc_torch_to_numpy(small_t)
    large_np = hwc_torch_to_numpy(large_t)

    face_percent_and_pixels = None
    if face_results:
        face_percent_and_pixels = get_masked_pixels(large_np, face_results, True)

    color_features, face_data = extract_color_match_features(
        small_np,
        large_np,
        face_results,
        is_reference=False,
        face_percent_and_pixels=face_percent_and_pixels,
    )
    filtered_faces = face_data["filteredFaceDetectionResults"]

    if filtered_faces and face_percent_and_pixels is None:
        face_percent_and_pixels = get_masked_pixels(large_np, filtered_faces, True)

    face_area = get_face_area_luminance_statistics(
        large_np,
        filtered_faces,
        face_percent_and_pixels,
    )
    from ..probe_parity.luminance import LuminanceStatistics

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
        small_np,
        large_np,
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
    scene = SceneInfo(
        auto_light_params=auto_light,
        auto_wb_params=auto_wb,
        base_adjustments=base,
        face_results=filtered_faces,
        avg_face_hsvl=avg_face,
    )
    return scene, small_t, large_t


def estimate_base_adjustments_full_gpu(feed_tensor: torch.Tensor) -> dict[str, float]:
    scene, _, _ = extract_scene_info_bitmap_full_gpu(feed_tensor)
    return get_original_color_match_base_adjustments(scene)
