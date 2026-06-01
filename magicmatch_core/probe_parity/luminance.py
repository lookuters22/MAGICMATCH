"""
Luminance statistics + auto-light for color-match base adjustments.

Port of ai-calibration-brightness.ts (noface path when face stats are empty).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibration_utils import get_masked_pixels
from .reference import fit_long_edge, resize_hwc


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def get_lightness_rgba(rgba: np.ndarray, *, channel: int = 4, scale: float = 255.0) -> np.ndarray:
    """Port of getLightness from ai-calibration-brightness.ts."""
    data = np.asarray(rgba, dtype=np.float64).reshape(-1, channel)
    r = data[:, 0] / scale
    g = data[:, 1] / scale
    b = data[:, 2] / scale
    return (np.maximum(r, g, b) + np.minimum(r, g, b)) * 0.5


def get_lightness_hwc(hwc: np.ndarray) -> np.ndarray:
    r, g, b = hwc[..., 0], hwc[..., 1], hwc[..., 2]
    return (np.maximum(r, g, b) + np.minimum(r, g, b)) * 0.5


def compute_histogram(data: np.ndarray, num_bins: int = 256) -> np.ndarray:
    bins = np.zeros(num_bins, dtype=np.int64)
    idx = np.clip(np.floor(np.asarray(data, dtype=np.float64) * num_bins).astype(np.int32), 0, num_bins - 1)
    np.add.at(bins, idx, 1)
    return bins


def compute_percentiles(
    bins: np.ndarray,
    percentiles: tuple[float, ...] = (5, 20, 50, 75, 90, 95, 99),
) -> list[float]:
    cum = np.cumsum(bins)
    total = cum[-1] if len(cum) else 0
    if total <= 0:
        return [0.0] * len(percentiles)
    bin_size = 1.0 / len(bins)
    out: list[float] = []
    for pct in percentiles:
        target = (pct / 100.0) * total
        idx = int(np.searchsorted(cum, target, side="left"))
        idx = min(idx, len(bins) - 1)
        out.append(idx * bin_size)
    return out


def get_shadows_mean(bins: np.ndarray) -> float:
    cum = np.cumsum(bins)
    total = cum[-1] if len(cum) else 1
    p_index = 0.25 * total
    value_sum = 0.0
    for i, count in enumerate(bins):
        value_sum += count * (i + 0.5) / len(bins)
        if cum[i] >= p_index:
            denom = cum[i] if cum[i] > 0 else 1
            return value_sum / denom
    return value_sum / total


def diff_with_margin(val: float, ref: float, margin: float) -> float:
    if val > ref + margin:
        return val - ref - margin
    if val < ref - margin:
        return val - ref + margin
    return 0.0


def get_percentile_difference(
    percentiles: list[float],
    refs: list[float],
    margins: list[float],
    margin_delta: float = 0.02,
    excluded_indices: tuple[int, ...] = (-1,),
) -> float:
    max_diff = -1.0
    min_diff = 1.0
    i = 0
    for j, val in enumerate(percentiles):
        ref = refs[j] if j < len(refs) else 0.0
        margin = max((margins[j] if j < len(margins) else 0.0) - margin_delta, 0.0)
        if i in excluded_indices:
            i += 1
            continue
        diff = diff_with_margin(val, ref, margin)
        max_diff = max(diff, max_diff)
        min_diff = min(diff, min_diff)
        i += 1
    if max_diff * min_diff >= 0:
        return max_diff if abs(max_diff) > abs(min_diff) else min_diff
    return max_diff + min_diff


@dataclass
class LuminanceStatistics:
    avg_lum: float
    shadows_mean: float
    clipping_percent: float
    percentiles: list[float]
    face_percentiles: list[float] = None
    face_percent: float = 0.0
    face_lum: float = 0.0

    def __post_init__(self) -> None:
        if self.face_percentiles is None:
            self.face_percentiles = []


def get_face_area_luminance_statistics(
    large_hwc: np.ndarray,
    face_detection_results: list[dict],
    face_percent_and_pixels: tuple[float, np.ndarray] | None = None,
) -> dict[str, float | list[float] | int]:
    """Port of getFaceAreaLuminanceStatistics."""
    face_lum = 0.0
    face_percent = 0.0
    face_percentiles: list[float] = []
    face_clipping_highlights = 0.0
    face_clipping_shadows = 0.0
    face_count = len(face_detection_results)

    if face_detection_results:
        if face_percent_and_pixels is None:
            face_percent_and_pixels = get_masked_pixels(large_hwc, face_detection_results, True)
        masked_percent, masked_pixels = face_percent_and_pixels
        face_percent = masked_percent
        face_lum_array = get_lightness_rgba(masked_pixels)
        face_histogram = compute_histogram(face_lum_array)
        face_lum = float(np.mean(face_lum_array)) if face_lum_array.size else 0.0
        face_percentiles = compute_percentiles(face_histogram)
        if face_lum_array.size:
            face_clipping_highlights = float(np.sum(face_histogram[251:256]) / len(face_lum_array))
            face_clipping_shadows = float((face_histogram[0] if len(face_histogram) else 0) / len(face_lum_array))

    return {
        "facePercentiles": face_percentiles,
        "facePercent": face_percent,
        "faceLum": face_lum,
        "faceCount": face_count,
        "faceClippingHighlights": face_clipping_highlights,
        "faceClippingShadows": face_clipping_shadows,
    }


def get_luminance_statistics(
    small_hwc: np.ndarray,
    large_hwc: np.ndarray,
    face_detection_results: list[dict],
) -> LuminanceStatistics:
    """Port of getLuminanceStatistics(small, large, faces)."""
    from .calibration_utils import get_masked_pixels, hwc_to_rgba_uint8

    # Probe passes exportImageData RGBA uint8 into getLightness (ai-calibration-brightness.ts).
    lum = get_lightness_rgba(hwc_to_rgba_uint8(small_hwc).reshape(-1))
    histogram = compute_histogram(lum)
    avg_lum = float(np.mean(lum))
    shadows_mean = get_shadows_mean(histogram)
    clipping = float(np.sum(histogram[251:256]) / max(len(lum), 1))
    shadows_clipping = float((histogram[0] if len(histogram) else 0) / max(len(lum), 1))
    percentiles = compute_percentiles(histogram)

    face_percent_and_pixels = None
    if face_detection_results:
        face_percent_and_pixels = get_masked_pixels(large_hwc, face_detection_results, True)

    face_stats = get_face_area_luminance_statistics(
        large_hwc, face_detection_results, face_percent_and_pixels
    )
    return LuminanceStatistics(
        avg_lum=avg_lum,
        shadows_mean=shadows_mean,
        clipping_percent=clipping,
        percentiles=percentiles,
        face_percentiles=face_stats["facePercentiles"],
        face_percent=float(face_stats["facePercent"]),
        face_lum=float(face_stats["faceLum"]),
    )


def get_luminance_statistics_from_source(source_hwc: np.ndarray) -> LuminanceStatistics:
    """Legacy helper — no face detection."""
    from .reference import fit_long_edge, resize_hwc

    small = resize_hwc(fit_long_edge(source_hwc, 300), *fit_to_size(source_hwc.shape[1], source_hwc.shape[0], (300, 300)), high_quality=True)
    large = fit_long_edge(source_hwc, 2000)
    return get_luminance_statistics(small, large, [])


def fit_to_size(width: int, height: int, box: tuple[int, int]) -> tuple[int, int]:
    from .reference import fit_to_size as _fit

    return _fit(width, height, box)


def get_adjusted_gamma(lum_stats: LuminanceStatistics, exposure: float = 0.0) -> float:
    """Port of getAdjustedGamma."""
    lum_full = LuminanceStatistics(
        avg_lum=lum_stats.avg_lum,
        shadows_mean=lum_stats.shadows_mean,
        clipping_percent=lum_stats.clipping_percent,
        percentiles=lum_stats.percentiles,
    )
    auto = get_auto_exposure(lum_full, True)
    shadows = (0.2 - lum_stats.shadows_mean * 0.5 - (lum_stats.percentiles[0] if lum_stats.percentiles else 0) * 0.5) * 2.5
    lum = (auto["exposureDelta"] * 1.8) / 3.0 + auto["shadowsDelta"] / 100.0
    if lum < 0:
        lum = min(0.0, lum + 0.3)
    adjusted_lum = (max(lum, shadows) + lum) / 2.0 - (exposure / 4.0) * 1.8
    gamma = _clamp(1.0 - adjusted_lum, 0.4, 1.4)
    return gamma if abs(gamma - 1.0) > 0.25 else 1.0


def get_luminance_statistics_legacy(source_hwc: np.ndarray) -> LuminanceStatistics:
    """Downsampled stats — mirrors getLuminanceStatistics without face detection."""
    small = resize_hwc(fit_long_edge(source_hwc, 300), 300, 300, high_quality=True)
    lum = get_lightness_hwc(small).ravel()
    histogram = compute_histogram(lum)
    avg_lum = float(np.mean(lum))
    shadows_mean = get_shadows_mean(histogram)
    clipping = float(np.sum(histogram[251:256]) / max(len(lum), 1))
    percentiles = compute_percentiles(histogram)
    return LuminanceStatistics(
        avg_lum=avg_lum,
        shadows_mean=shadows_mean,
        clipping_percent=clipping,
        percentiles=percentiles,
    )


def get_saturation_delta(shadows: float, highlights: float, exposure: float, has_face: bool) -> float:
    delta_hs = max(0.0, shadows - 20 - highlights - 20)
    max_value = 24 if has_face else 10
    scale = 1.2 if has_face else 0.6
    if exposure > 0.5:
        scale *= max(0.0, 1.5 - exposure)
    return -min((delta_hs * scale) / 6.0, max_value)


def get_auto_exposure(inputs: LuminanceStatistics, relax_limits: bool = False) -> dict[str, float]:
    """Rule-based auto-light deltas (port of getAutoExposure)."""
    avg_lum = inputs.avg_lum
    face_lum = inputs.face_lum
    percentiles = inputs.percentiles
    face_percentiles = inputs.face_percentiles
    face_percent = inputs.face_percent
    shadows_mean = inputs.shadows_mean
    clipping_percent = inputs.clipping_percent

    scale = 3.0
    means = [
        0.131, 0.26, 0.495, 0.699, 0.846, 0.905, 0.964, 0.503,
        0.399, 0.543, 0.665, 0.74, 0.806, 0.842, 0.893, 0.648,
    ]
    stds = [
        0.09, 0.128, 0.156, 0.131, 0.087, 0.063, 0.035, 0.112,
        0.101, 0.093, 0.079, 0.068, 0.064, 0.064, 0.065, 0.068,
    ]
    noface_means = [0.196, 0.331, 0.518, 0.707, 0.85, 0.905, 0.953, 0.536]
    noface_stds = [0.159, 0.206, 0.201, 0.155, 0.093, 0.066, 0.039, 0.154]

    face_exists = len(face_percentiles) > 0
    percentile99 = percentiles[-1] if percentiles else 0.0
    percentile95 = percentiles[-2] if len(percentiles) > 1 else 0.0
    percentile90 = percentiles[-3] if len(percentiles) > 2 else 0.0
    percentile75 = percentiles[3] if len(percentiles) > 3 else 0.0
    percentile50 = percentiles[2] if len(percentiles) > 2 else 0.0

    face_weight = 0.0
    face_exposure = 0.0
    max_face_exposure = 0.0
    image_exposure = 0.0

    if face_exists:
        face_weight = 0.8
        if face_percent < 0.01:
            face_weight = 0.6 + (0.2 * face_percent) / 0.01
        face_p = face_percentiles + [face_lum]
        max_diff = get_percentile_difference(face_p, means[8:], stds[8:], 0.02, (-1,))
        max_diff2 = get_percentile_difference(face_p, means[8:], stds[8:], 0.02, (5, 6))
        face_exposure = -min(max_diff2, max_diff) * scale
        face_p95 = face_percentiles[-2] if len(face_percentiles) > 1 else 0.0
        face_p90 = face_percentiles[-3] if len(face_percentiles) > 2 else 0.0
        max_face_exposure = max(0.95 - face_p95, 0.9 - face_p90) * scale
        face_exposure = min(face_exposure, max_face_exposure)

    image_weight = 1.0 - face_weight
    if not face_exists:
        means = noface_means
        stds = noface_stds

    max_diff = get_percentile_difference(percentiles + [avg_lum], means[:8], stds[:8])
    max_diff_low = get_percentile_difference(
        percentiles + [avg_lum], means[:8], stds[:8], 0.15, (4, 5, 6)
    )
    max_diff_dark = get_percentile_difference(
        percentiles + [avg_lum], means[:8], stds[:8], 0.15, (2, 3, 4, 5, 6)
    )
    image_shadows = -max_diff_low * scale
    image_exposure = -max_diff * scale
    image_blacks = -max_diff_dark * scale
    if image_shadows < -0.2:
        image_shadows = max(image_shadows, image_blacks)

    max_exposure = (1.02 - percentile99) * scale
    max_exposure = min(
        max_exposure,
        (1.02 - 0.06 - percentile95) * scale,
        (1.02 - 0.12 - percentile90) * scale,
    )
    max_exposure = max(max_exposure, (0.32 - percentile50) * scale)

    if not face_exists:
        shadows_scale = 0.4
        image_exposure = image_exposure * (1 - shadows_scale) + image_shadows * shadows_scale
        d1 = (0.98 - percentile99) * scale
        d2 = (0.93 - percentile95) * scale
        max_exposure = d1 if not (d1 < 0.1 and d2 > d1) else 0.75 * d1 + 0.25 * d2
        max_exposure = max(max_exposure, (0.32 - percentile50) * scale)
        if clipping_percent > 0:
            max_exposure = min(
                max_exposure,
                max(
                    max(0.0, (0.95 - percentile99) * scale),
                    max(0.0, (0.55 - percentile75) * scale),
                    max(0.0, (0.32 - percentile50) * scale),
                ),
            )

    max_exposure_final = (max(image_exposure, -0.2) + max_exposure) * 0.6 + 0.3
    image_exposure = min(image_exposure, max_exposure)
    image_exposure_limited = image_exposure

    if face_exists:
        min_v = face_exposure - 0.4
        max_v = face_exposure + 0.4
        if face_percent > 0.007:
            if face_exposure < 0:
                min_v = max(face_exposure - 0.4, min(face_exposure * 5, -0.25))
            else:
                max_v = min(face_exposure + 0.4, max(0.25, face_exposure * 5))
        small_face_thresh = 0.003
        tiny_face_thresh = 0.001
        if face_percent < small_face_thresh:
            small_face_scale = (small_face_thresh - face_percent) / small_face_thresh
            min_v -= 0.4 * small_face_scale
            max_v += 0.4 * small_face_scale
            max_exposure_final -= 0.4 * small_face_scale
            if clipping_percent > 0:
                max_exposure_final *= min(1.0, (1.0 - small_face_scale) * 0.5 + 0.5)
            if face_percent < tiny_face_thresh:
                tiny_scale = (face_percent / tiny_face_thresh) * 0.6 + 0.4
                face_exposure *= tiny_scale
                max_exposure_final *= tiny_scale
        image_exposure_limited = min(max(image_exposure, min_v), max_v)

    exposure = image_exposure_limited * image_weight + face_exposure * face_weight
    if not face_exists and percentile99 < 0.97 and clipping_percent == 0 and exposure < -0.1:
        exposure = -0.1 + (exposure + 0.1) * 0.5
    if clipping_percent > 0:
        max_exposure_final *= 0.7
    if relax_limits and max_exposure_final < face_exposure:
        max_exposure_final = max_exposure_final * 0.2 + face_exposure * 0.8
    exposure = min(exposure, max_exposure_final)

    highlights = (
        min(0.97 - percentile99, 0)
        + min(0.93 - percentile95, 0)
        + min(0.86 - percentile90, 0)
    )
    highlights_for_low = min((means[2] + stds[2]) - percentiles[2], 0) + min(
        (means[3] + stds[3]) - percentiles[3], 0
    )
    delta1 = percentile95 - percentile99 if percentile95 > 0.82 else 0.07
    delta2 = percentile90 - percentile95 if percentile90 > 0.78 else 0.07
    highlights_for_close = max(min(delta1 - 0.07, 0) + min(delta2 - 0.07, 0), -0.1) * 1.4
    highlights = min(highlights, highlights_for_low, highlights_for_close)

    face_percentile99 = face_percentiles[-1] if face_percentiles else 0.0
    face_percentile95 = face_percentiles[-2] if len(face_percentiles) > 1 else 0.0
    face_percentile90 = face_percentiles[-3] if len(face_percentiles) > 2 else 0.0

    if face_exists:
        highlights = max(highlights, -0.12) * (0.2 + 2 * image_weight)
        face_highlights = (
            min(0.8 - face_percentile90, 0)
            + min(0.84 - face_percentile95, 0)
            + min(0.89 - face_percentile99, 0)
        )
        offset = 8
        face_highlights_for_low = min(
            (means[offset + 2] + stds[offset + 2]) - face_percentiles[2], 0
        ) + min((means[offset + 3] + stds[offset + 3]) - face_percentiles[3], 0)
        fd1 = 0.07
        fd2 = 0.07
        if face_percentile95 > 0.82:
            fd1 = face_percentile99 - face_percentile95
        if face_percentile90 > 0.78:
            fd2 = face_percentile95 - face_percentile90
        face_highlights_for_close = (
            max(min(fd1 - 0.05, 0) + min(fd2 - 0.05, 0), -0.08) * 1.5
        )
        highlights += min(face_highlights, face_highlights_for_low, face_highlights_for_close)

    max_highlights = -25 if clipping_percent > 0 else -35
    highlights = max(highlights * 200, max_highlights)
    whites = highlights * 0.3
    if clipping_percent > 0:
        clip_h = -min(10 + clipping_percent * 100 * 10, 50)
        whites = _clamp(whites + clip_h, -35, 0)
        highlights += clip_h

    shadows = max(0.1 - shadows_mean, 0) * 150
    blacks = _clamp(shadows * 1.5, 0, 20)

    if face_exists:
        blacks += max(0.1 - face_percentiles[0], 0) * 200
        d2 = max(0.2 - face_percentiles[1], 0) * 150
        blacks = _clamp(blacks + d2, 0, 40)
        shadows += d2
        whites *= 0.7
        max_diff_low_face = get_percentile_difference(
            face_percentiles + [face_lum],
            means[8:],
            stds[8:],
            0.02,
            (4, 5, 6),
        )
        face_shadows = -max_diff_low_face * scale
        face_shadows = max(face_exposure, face_shadows, image_shadows * 0.7)
        shadows_scale = min(45 + 1000 * face_percent, 55)
        if face_shadows > exposure:
            percent_delta = 1.0
            if exposure > 0.1:
                percent_delta = (face_shadows - exposure) / exposure
            elif face_shadows < -0.1:
                percent_delta = -(face_shadows - exposure) / face_shadows
            shadows += (face_shadows - exposure) * shadows_scale * min(percent_delta, 1.0)
    elif image_shadows > exposure:
        shadows_scale = 100 if relax_limits else 50
        shadows += (image_shadows - exposure) * shadows_scale

    max_shadows = 80 if relax_limits else 45
    if shadows > max_shadows and face_exists:
        exposure += ((shadows - max_shadows) / 100.0)

    limit = 1.0 if face_exists else 0.95
    max_exposure_abs = min(
        (limit - percentile99) * scale,
        (limit - 0.05 - percentile95) * scale,
        (limit - 0.1 - percentile90) * scale,
    )
    if face_exists:
        max_exposure_abs = min(max_exposure_abs, face_exposure)
    if exposure > max_exposure_abs:
        highlights_from_exp = min((exposure - max_exposure_abs) * 100, 60)
        highlights = min(
            highlights - highlights_from_exp * 0.2,
            max(-highlights_from_exp * 1.6, -70),
        )
    elif exposure < 0 and highlights != 0:
        highlights -= max(exposure, -0.5) * 30

    if face_exists and face_lum > 0.55:
        scale_down = max(1.0 - (face_lum - 0.55) * 1.6, 0.4)
        highlights = highlights * scale_down
        whites *= scale_down

    min_highlights = -80 if relax_limits else -60
    if highlights < min_highlights and exposure > max_exposure:
        d_exp = ((highlights - min_highlights) / 200) * (0 if face_exists else 1)
        exposure += _clamp(d_exp, -0.4, 0)

    if (
        not face_exists
        and highlights < -30
        and exposure > highlights / 100 + 0.3
        and clipping_percent == 0
    ):
        delta_highlights = (highlights + 30) * 0.75
        exposure += delta_highlights / 100
        highlights -= delta_highlights

    highlights = max(min_highlights, min(highlights, 0))
    shadows = max(0, min(shadows, max_shadows))

    if face_lum < 0.65 and exposure < 0 and clipping_percent > 0.008:
        delta_dark_area = max(
            0.7 - percentiles[2],
            0.53 - percentiles[1],
            0.36 - percentiles[0],
            -0.25 * exposure,
        )
        delta_bright_area = max(
            percentiles[3] * 0.3 + percentiles[4] * 0.7 - 0.83,
            -0.2 * exposure,
        )
        delta = min(delta_dark_area, delta_bright_area * 0.7 + delta_dark_area * 0.3)
        d_scale = 4 * _clamp((0.85 - percentiles[0] - percentiles[1]) / 0.2, 0, 1)
        d_scale = max(d_scale, min(4.0, (0.78 - avg_lum) * 16))
        if face_exists:
            delta = 0.6 - face_percentiles[1]
            d_scale = 2.5
        delta_exp = (max(delta, 0) * d_scale * (-highlights - whites - 25)) / 70
        delta1_lim = 0.0
        delta2_lim = min(
            max((0.78 - avg_lum) * 2, -exposure),
            max(0.25, -1.2 * exposure),
        )
        exposure += _clamp(delta_exp, delta1_lim, delta2_lim)

    min_exposure = -1.0
    if face_percent >= 0.02 and (face_lum < 0.4 or face_percentile99 < 0.65):
        min_exposure = max(0.4 - face_lum, 0.65 - face_percentile99) * 2

    min_exposure_limit = -1.5 if relax_limits else -1.0
    max_exposure_limit = 1.9 if relax_limits else 1.3
    exposure = max(exposure, min_exposure)
    exposure = _clamp(exposure, min_exposure_limit, max_exposure_limit)

    saturation = get_saturation_delta(shadows, highlights, exposure, face_lum > 0)
    return {
        "exposureDelta": round(exposure, 2),
        "highlightsDelta": round(highlights),
        "shadowsDelta": round(shadows),
        "whitesDelta": round(whites),
        "blacksDelta": round(blacks),
        "contrastDelta": max(0, round(shadows * 0.8)),
        "saturationDelta": round(saturation),
    }


def get_auto_light_params_from_deltas(auto_deltas: dict[str, float], scale: float = 1.0) -> dict[str, float]:
    return {
        "exposure": auto_deltas["exposureDelta"] * scale,
        "highlights": (auto_deltas["highlightsDelta"] / 100.0) * scale,
        "shadows": (auto_deltas["shadowsDelta"] / 100.0) * scale,
        "whites": (auto_deltas["whitesDelta"] / 100.0) * scale,
        "blacks": (auto_deltas["blacksDelta"] / 100.0) * scale,
        "contrast": (auto_deltas["contrastDelta"] / 100.0) * scale,
        "saturation": (auto_deltas["saturationDelta"] / 100.0) * scale,
    }


def get_auto_light_params(
    stats: LuminanceStatistics,
    face_hsvl: np.ndarray | None = None,
) -> dict[str, float]:
    """Port of getAutoLightParams(luminanceStats, avgFaceHsvl)."""
    from .face_tone import is_face_dark_skin, is_face_problematic

    auto_deltas = get_auto_exposure(stats, relax_limits=True)
    if (
        auto_deltas["exposureDelta"] > 1
        and auto_deltas["highlightsDelta"] < -50
        and auto_deltas["shadowsDelta"] < 50
    ):
        delta_exp = _clamp(auto_deltas["exposureDelta"] - 1, 0, 0.4)
        auto_deltas["exposureDelta"] -= delta_exp
        auto_deltas["shadowsDelta"] = _clamp(auto_deltas["shadowsDelta"] + delta_exp * 80, 0, 95)
        auto_deltas["contrastDelta"] = _clamp(auto_deltas["contrastDelta"] + delta_exp * 40, 0, 70)
    scale = (
        0.5
        if not is_face_problematic(face_hsvl)
        and is_face_dark_skin(face_hsvl)
        and auto_deltas["exposureDelta"] > 0.0
        else 0.8
    )
    return get_auto_light_params_from_deltas(auto_deltas, scale)
