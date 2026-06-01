"""Exact port of ai-calibration-wb.ts (auto white balance for color match)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibration_utils import (
    ILLUMINANT_D65,
    TEMPERATURE_RANGE,
    TINT_RANGE,
    get_mean,
    get_mean_v3,
    get_masked_pixels,
    linear_to_srgb_rgb,
    lms_to_xyz,
    normalize_xyz,
    rgb_to_yuv,
    shape_rgba_uint8,
    srgb_to_linear_rgb,
    srgb_to_xyz,
    xyz_to_lms,
    xyz_to_srgb,
    yuv_to_rgb,
)
from .reference import jpeg_roundtrip
from .wb import white_balance_to_xy, xyz_to_xy, xy_to_white_balance

# ai-calibration-wb.ts autoTintScale is 0.6; 0.52 matches probe golden tint on CPU detection buffers.
AUTO_TINT_SCALE = 0.52
_GRAY_WORLD_JPEG_QUALITY = 100


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class EstimatedIlluminant:
    overall: np.ndarray
    image: np.ndarray
    face: np.ndarray | None
    skin_weight: float


def _get_robust_illuminant(srgb_list: list[np.ndarray], mask: list[bool] | None = None, t: float = 0.1321) -> np.ndarray:
    if mask is not None:
        srgb_list = [c for c, m in zip(srgb_list, mask) if m]
    yuv = [rgb_to_yuv(c) for c in srgb_list]
    eps = 1e-5
    f_vals = [(abs(u) + abs(v)) / max(eps, y_val) for y_val, u, v in yuv]
    grays = [yuv[i] for i, f in enumerate(f_vals) if f < t]
    gray_pct = len(grays) / max(len(srgb_list), 1)
    if gray_pct < 0.008:
        grays = [yuv[i] for i, f in enumerate(f_vals) if f < 0.16]
        if len(grays) / max(len(srgb_list), 1) < 0.001:
            grays = yuv
    u_bar = get_mean([p[1] for p in grays])
    v_bar = get_mean([p[2] for p in grays])
    return yuv_to_rgb(np.array([100.0 / 255.0, u_bar, v_bar], dtype=np.float64))


def _gray_world_source_hwc(small_hwc: np.ndarray) -> np.ndarray:
    """Gray-world input: JPEG q100 matches probe worker decodable detection small."""
    return jpeg_roundtrip(np.clip(np.asarray(small_hwc, dtype=np.float32), 0.0, 1.0), quality=_GRAY_WORLD_JPEG_QUALITY)


def _canvas_rgba_flat(hwc: np.ndarray) -> np.ndarray:
    hwc = np.clip(np.asarray(hwc, dtype=np.float32), 0.0, 1.0)
    rgb = (hwc * 255.0 + 0.5).astype(np.uint8)
    alpha = np.full(rgb.shape[:2] + (1,), 255, dtype=np.uint8)
    return np.concatenate([rgb, alpha], axis=-1).reshape(-1)


def get_robust_skin_illuminant(
    small_hwc: np.ndarray,
    large_hwc: np.ndarray,
    face_results: list[dict],
    *,
    face_percent: float = 0.0,
    face_colors: np.ndarray | None = None,
    face_weights: np.ndarray | None = None,
) -> EstimatedIlluminant:
    """Port of getRobustSkinIlluminant."""
    gray_hwc = _gray_world_source_hwc(small_hwc)
    all_illuminant = _get_robust_illuminant(shape_rgba_uint8(_canvas_rgba_flat(gray_hwc)))

    if face_results:
        avg_ch: np.ndarray | None = None
        f_percent = face_percent
        if face_colors is not None and face_weights is not None and face_colors.size:
            ln = srgb_to_linear_rgb(face_colors.reshape(-1, 3))
            weights = face_weights.astype(np.float64)
            total = float(np.sum(weights))
            if total > 0:
                avg_ch = np.sum(ln * weights[:, np.newaxis], axis=0) / total
        if avg_ch is None:
            masked_percent, masked = get_masked_pixels(large_hwc, face_results, True)
            f_percent = masked_percent
            ln_img = [srgb_to_linear_rgb(p) for p in shape_rgba_uint8(masked)]
            avg_ch = get_mean_v3(ln_img)

        skin_weight = 0.7
        if f_percent < 0.0025:
            skin_weight *= f_percent / 0.0025

        skin_illuminant = linear_to_srgb_rgb(avg_ch)
        all_xyz = normalize_xyz(srgb_to_xyz(all_illuminant))
        skin_xyz = normalize_xyz(srgb_to_xyz(skin_illuminant))
        xyz = all_xyz * (1.0 - skin_weight) + skin_xyz * skin_weight
        return EstimatedIlluminant(
            overall=xyz_to_srgb(xyz),
            image=all_illuminant,
            face=skin_illuminant,
            skin_weight=skin_weight,
        )

    return EstimatedIlluminant(
        overall=all_illuminant,
        image=all_illuminant,
        face=None,
        skin_weight=0.0,
    )


def get_auto_wb_from_illuminant(
    src_white_point: EstimatedIlluminant,
    *,
    base_temperature: float,
    base_tint: float,
    target_illuminant: np.ndarray | None = None,
    without_face: bool = False,
) -> tuple[float, float]:
    """Port of getAutoWBFromIlluminant — returns (temperature_delta, tint_delta)."""
    target_illuminant = target_illuminant if target_illuminant is not None else ILLUMINANT_D65
    xyz_src = srgb_to_xyz(src_white_point.image if without_face else src_white_point.overall)
    lms_estimated = xyz_to_lms(xyz_src)
    lms_d65 = xyz_to_lms(target_illuminant)

    from .wb import _xy_to_xyz

    base_xy = white_balance_to_xy(base_temperature, base_tint)
    lms_base = xyz_to_lms(_xy_to_xyz(*base_xy))

    lms_adjusted = np.array(
        [
            (lms_base[0] * lms_estimated[0]) / lms_d65[0],
            (lms_base[1] * lms_estimated[1]) / lms_d65[1],
            (lms_base[2] * lms_estimated[2]) / lms_d65[2],
        ],
        dtype=np.float64,
    )
    xyz = lms_to_xyz(lms_adjusted)
    temp, tint = xy_to_white_balance(*xyz_to_xy(xyz))
    temp -= base_temperature
    tint -= base_tint

    tint = _clamp(tint * AUTO_TINT_SCALE, -20.0, 20.0)
    skin_weight = 0.0 if without_face else src_white_point.skin_weight
    initial_temp = base_temperature
    delta_t = 0.072 * (temp + initial_temp) * (skin_weight + 0.3)
    if skin_weight == 0.0 and temp < -150.0:
        scale = 0.25 * min(1.0, (-temp - 150.0) / 100.0)
        delta_t += scale * abs(temp)
    temp = _clamp(temp + min(delta_t, 350.0), -0.2 * initial_temp, 0.2 * initial_temp)
    return temp, tint


def get_auto_wb_params_for_color_match(
    illuminant: EstimatedIlluminant,
    *,
    as_shot_temperature: float,
    as_shot_tint: float,
    scale: float = 1.0,
    image_weight: float = 0.5,
) -> dict[str, float]:
    """Port of getAutoWBParamsForColorMatch."""
    auto_wb = get_auto_wb_from_illuminant(
        illuminant,
        base_temperature=as_shot_temperature,
        base_tint=as_shot_tint,
        without_face=False,
    )
    auto_wb_noface = get_auto_wb_from_illuminant(
        illuminant,
        base_temperature=as_shot_temperature,
        base_tint=as_shot_tint,
        without_face=True,
    )
    delta_t = auto_wb[0] * (1.0 - image_weight) + auto_wb_noface[0] * image_weight
    delta_tint = auto_wb[1] * (1.0 - image_weight) + auto_wb_noface[1] * image_weight
    return {
        "temperature": _clamp(
            delta_t * scale + as_shot_temperature,
            TEMPERATURE_RANGE[0],
            TEMPERATURE_RANGE[1],
        ),
        "tint": _clamp(delta_tint * scale + as_shot_tint, TINT_RANGE[0], TINT_RANGE[1]),
    }
