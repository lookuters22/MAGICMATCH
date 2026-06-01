"""
Probe-style scene extraction for Comfy bitmap sources.

Mirrors extractSceneInfo / extractImageInfoFromRefJpeg bitmap path:
  renderDetectionInputs → face detect → colorMatch features → auto light/WB.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibration_utils import get_masked_pixels
from .calibration_wb import get_auto_wb_params_for_color_match, get_robust_skin_illuminant
from .color_match_features import extract_color_match_features
from .face_detection import detect_faces
from .luminance import get_adjusted_gamma, get_auto_light_params, get_luminance_statistics
from .reference import fit_to_size, fit_long_edge, resize_hwc

BITMAP_AS_SHOT_TEMP = 5000.0
BITMAP_AS_SHOT_TINT = 0.0
DETECTION_LARGE_EDGE = 2000
DETECTION_SMALL_EDGE = 300


@dataclass
class SceneInfo:
    auto_light_params: dict[str, float]
    auto_wb_params: dict[str, float]
    base_adjustments: dict[str, float]
    face_results: list[dict]
    avg_face_hsvl: np.ndarray | None


def render_detection_inputs(hwc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bitmap path: fit small/large like probe renderDetectionInputs / ref JPEG."""
    hwc = np.clip(np.asarray(hwc, dtype=np.float32), 0.0, 1.0)
    large = fit_long_edge(hwc, DETECTION_LARGE_EDGE)
    sw, sh = fit_to_size(hwc.shape[1], hwc.shape[0], (DETECTION_SMALL_EDGE, DETECTION_SMALL_EDGE))
    small = resize_hwc(hwc, sw, sh, high_quality=True)
    return small, large


def get_original_color_match_base_adjustments(scene: SceneInfo) -> dict[str, float]:
    """Port of getOriginalColorMatchBaseAdjustments."""
    return dict(scene.base_adjustments)


def extract_scene_info_bitmap(
    source_hwc: np.ndarray,
    *,
    as_shot_temperature: float = BITMAP_AS_SHOT_TEMP,
    as_shot_tint: float = BITMAP_AS_SHOT_TINT,
) -> SceneInfo:
    """Full bitmap scene extract for color-match base adjustments."""
    small, large = render_detection_inputs(source_hwc)
    stats_noface = get_luminance_statistics(small, large, [])
    adjusted_gamma = get_adjusted_gamma(stats_noface)

    face_results = detect_faces(large, adjusted_gamma=adjusted_gamma)
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

    face_area = get_luminance_statistics(small, large, filtered_faces)
    avg_face = color_features.get("avgFaceHsvl")
    auto_light = get_auto_light_params(face_area, avg_face)

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
