"""Port of extractColorMatchFeatures (source / bitmap path)."""

from __future__ import annotations

import numpy as np

from .calibration_utils import get_masked_pixels, hwc_to_rgba_uint8
from .color_space import rgb_to_hsv
from .face_tone import (
    is_face_problematic,
    is_face_too_white,
    is_good_face_color,
    is_reasonable_face_color,
)


def _round5(x: float) -> float:
    return round(float(x), 5)


def get_weighted_mean_rgb(
    colors: np.ndarray, weights: np.ndarray, min_weight: float = 0.0005
) -> np.ndarray | None:
    if colors.size == 0 or weights.size == 0:
        return None
    total = float(np.sum(weights))
    if total < min_weight:
        return None
    rgb = np.sum(colors.reshape(-1, 3) * weights[:, np.newaxis], axis=0) / total
    return rgb.astype(np.float64)


def get_weighted_median_luminance(colors: np.ndarray, weights: np.ndarray) -> float:
    lums: list[tuple[float, float]] = []
    cols = colors.reshape(-1, 3)
    for i in range(min(len(weights), len(cols))):
        r, g, b = cols[i]
        lums.append(((max(r, g, b) + min(r, g, b)) * 0.5, float(weights[i])))
    if not lums:
        return 0.5
    lums.sort(key=lambda x: x[0])
    total = sum(w for _, w in lums)
    half = total * 0.5
    acc = 0.0
    for lum, w in lums:
        acc += w
        if acc >= half:
            return lum
    return lums[-1][0]


def get_face_hsvl(face_colors: np.ndarray, face_weights: np.ndarray) -> np.ndarray:
    avg_rgb = get_weighted_mean_rgb(face_colors, face_weights)
    if avg_rgb is None:
        return np.zeros(4, dtype=np.float64)
    hsv = rgb_to_hsv(avg_rgb)
    avg_lum = get_weighted_median_luminance(face_colors, face_weights)
    return np.array([hsv[0], hsv[1], hsv[2], avg_lum], dtype=np.float64)


def get_rgb_data_with_step(
    masked_rgba: np.ndarray,
    target_size: int = 2000,
    dark_threshold: float = 0.1,
    bright_threshold: float = 0.1,
) -> tuple[np.ndarray, int, int]:
    """Port of getRGBDataWithStep."""
    data = np.asarray(masked_rgba, dtype=np.uint8).reshape(-1)
    original_size = len(data) // 4
    step = max(1, int(round(original_size / target_size)))
    new_size = original_size // step
    result = np.zeros(new_size * 3, dtype=np.uint8)
    luminances = np.zeros(new_size, dtype=np.float64)
    for i in range(new_size):
        k = i * step
        base = k * 4
        result[i * 3 : i * 3 + 3] = data[base : base + 3]
        luminances[i] = float(np.mean(result[i * 3 : i * 3 + 3])) / 255.0

    sorted_indices = np.argsort(luminances)
    start_index = int(np.floor(len(sorted_indices) * dark_threshold))
    end_index = int(np.floor(len(sorted_indices) * (1.0 - bright_threshold)))
    normal_lum_index = int(np.floor(len(sorted_indices) * 0.65))

    def get_luminance(idx: int) -> float:
        return float(luminances[sorted_indices[idx]])

    normal_lum = get_luminance(normal_lum_index)
    lum_threshold = max(min(normal_lum - 70.0 / 255.0, 40.0 / 255.0), min(normal_lum - 25.0 / 255.0, 20.0 / 255.0))
    max_index = min(int(np.floor(len(sorted_indices) * 0.4)), end_index - 1000)
    while start_index < max_index and get_luminance(start_index) < lum_threshold:
        start_index += int(np.floor(len(sorted_indices) * 0.01))

    filtered_indices = sorted_indices[start_index:end_index]
    filtered = result.reshape(-1, 3)[filtered_indices].reshape(-1)
    return filtered, start_index, end_index


def extract_colors(
    rgb_data: np.ndarray,
    *,
    bin_size: int = 8,
    pixel_count_threshold: int = 5,
    max_count: int = 70,
) -> tuple[np.ndarray, np.ndarray]:
    """Port of extractColors (face path uses max_count=70)."""
    rgb = np.asarray(rgb_data, dtype=np.float64)
    if rgb.size == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float64)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    pixels = rgb.reshape(-1, 3)
    num_bins = 256 // bin_size
    total_bins = num_bins**3
    sums = np.zeros(total_bins * 3, dtype=np.float64)
    counts = np.zeros(total_bins, dtype=np.int64)
    bin_inv = 1.0 / bin_size
    for px in pixels:
        r, g, b = px * 255.0
        r_i = int(r * bin_inv)
        g_i = int(g * bin_inv)
        b_i = int(b * bin_inv)
        idx = r_i * num_bins * num_bins + g_i * num_bins + b_i
        si = idx * 3
        sums[si : si + 3] += px * 255.0
        counts[idx] += 1

    entries: list[tuple[np.ndarray, int]] = []
    for i in range(total_bins):
        count = counts[i]
        if count > pixel_count_threshold:
            si = i * 3
            avg = sums[si : si + 3] / count
            entries.append((avg, count))
    entries.sort(key=lambda x: x[1], reverse=True)
    total_pixels = len(pixels)
    selected = min(max(len(entries), 10), min(max_count, len(entries)))
    selected_entries = entries[:selected]
    colors_out = np.zeros(selected * 3, dtype=np.float32)
    weights_out = np.zeros(selected, dtype=np.float64)
    for i, (color, count) in enumerate(selected_entries):
        colors_out[i * 3 : i * 3 + 3] = [_round5(c / 255.0) for c in color]
        weights_out[i] = _round5(count / total_pixels)
    return colors_out, weights_out


def extract_color_match_features(
    small_hwc: np.ndarray,
    large_hwc: np.ndarray,
    face_results: list[dict],
    *,
    is_reference: bool = False,
    face_percent_and_pixels: tuple[float, np.ndarray] | None = None,
) -> tuple[dict, dict]:
    """Source-path port of ai.worker extractColorMatchFeatures."""
    max_confidence = max((f.get("confidence") or 0.0 for f in face_results), default=0.0)
    face_percent = 0.0
    avg_face_hsvl: np.ndarray | None = None
    face_colors: np.ndarray | None = None
    face_weights: np.ndarray | None = None
    start_index: int | None = None
    end_index: int | None = None
    filtered = list(face_results)

    if face_results:
        if face_percent_and_pixels is None:
            face_percent_and_pixels = get_masked_pixels(large_hwc, face_results, True)
        masked_percent, masked_pixels = face_percent_and_pixels
        face_percent = masked_percent
        face_rgb, start_index, end_index = get_rgb_data_with_step(masked_pixels, 2000)
        colors, weights = extract_colors(face_rgb, pixel_count_threshold=4, max_count=70)
        if colors.size and weights.size:
            face_colors = colors.astype(np.float32)
            face_weights = weights.astype(np.float64)
            avg = get_face_hsvl(face_colors, face_weights)
            reject = (
                np.any(np.isnan(avg))
                or ((avg[1] < 0.02 or avg[3] > 0.95) and max_confidence < 0.99)
                or ((avg[1] < 0.1 or is_face_too_white(avg)) and max_confidence < 0.93)
                or (not is_reasonable_face_color(avg) and max_confidence < 0.9)
                or (is_reference and not is_good_face_color(avg) and max_confidence < 0.95)
            )
            if reject:
                avg_face_hsvl = None
                face_percent = 0.0
                filtered = []
                face_colors = None
                face_weights = None
            else:
                avg_face_hsvl = avg

    features = {
        "facePercent": face_percent,
        "avgFaceHsvl": avg_face_hsvl,
        "colorPatches": [],
    }
    face_data = {
        "filteredFaceDetectionResults": filtered,
        "faceColors": face_colors,
        "faceWeights": face_weights,
        "startIndex": start_index,
        "endIndex": end_index,
    }
    return features, face_data
