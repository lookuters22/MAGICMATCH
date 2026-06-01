"""Port of extractColorMatchFeatures (source / bitmap path)."""

from __future__ import annotations

import numpy as np

from .color_space import rgb_to_hsv
from .face_tone import (
    is_face_too_white,
    is_good_face_color,
    is_reasonable_face_color,
)

# Face path: ai.worker uses pixelCountThreshold 4 / maxCount 70. Canvas-style uint8
# (ImageData Math.round) shifts bin counts; threshold 3 lands ~66-70 colors on CPU buffers.
_FACE_PIXEL_COUNT_THRESHOLD = 3
_FACE_RGB_DARK_THRESHOLD = 0.105
_FACE_RGB_BRIGHT_THRESHOLD = 0.08


def _round5(x: float) -> float:
    return round(float(x), 5)


def _canvas_uint8_from_hwc(hwc: np.ndarray) -> np.ndarray:
    """Match browser ImageData uint8 quantization (Math.round on 0–255)."""
    hwc = np.clip(np.asarray(hwc, dtype=np.float32), 0.0, 1.0)
    return (hwc * 255.0 + 0.5).astype(np.uint8)


def _canvas_rgba_flat(hwc: np.ndarray) -> np.ndarray:
    rgb = _canvas_uint8_from_hwc(hwc)
    alpha = np.full(rgb.shape[:2] + (1,), 255, dtype=np.uint8)
    return np.concatenate([rgb, alpha], axis=-1).reshape(-1)


def _get_face_masked_pixels(
    hwc: np.ndarray,
    face_results: list[dict],
    *,
    use_step: bool = True,
) -> tuple[float, np.ndarray]:
    """getMaskedPixels with canvas uint8 sampling (probe ImageData parity)."""
    h, w, _ = hwc.shape
    data = _canvas_rgba_flat(hwc)

    filtered = [f for f in face_results if f.get("skin_condition") is not None]
    has_mask = bool(filtered)
    if not filtered:
        filtered = face_results
        has_mask = False

    step = 1
    if use_step:
        estimated = 0.0
        for face in filtered:
            box = face["face_box"]
            x1, y1, x2, y2 = box
            skin_pct = face.get("skin_percent", 0.25) if has_mask else 1.0
            estimated += (x2 - x1) * (y2 - y1) * skin_pct
        step = int(round((estimated / 10000.0) ** 0.5))
        step = max(1, min(step, 4))

    result: list[int] = []
    for face in filtered:
        box = face["face_box"]
        mask = face.get("skin_condition")
        x1, y1, x2, y2 = box
        crop_h = y2 - y1
        crop_w = x2 - x1
        for i in range(0, crop_h, step):
            mask_row = mask[i] if mask is not None else None
            for j in range(0, crop_w, step):
                if mask is None or (mask_row is not None and mask_row[j]):
                    ii = ((j + x1) + (i + y1) * w) * 4
                    result.extend(
                        [
                            int(data[ii]),
                            int(data[ii + 1]),
                            int(data[ii + 2]),
                            int(data[ii + 3]),
                        ]
                    )

    estimated_total = (len(result) / 4) * step * step
    masked_percent = (estimated_total * 4) / len(data) if len(data) else 0.0
    return masked_percent, np.asarray(result, dtype=np.uint8)


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


def get_luminance_percentiles(
    colors: np.ndarray,
    weights: np.ndarray,
    percentiles: tuple[float, ...] = (0.5,),
    *,
    luma: bool = True,
) -> list[float]:
    """Port of getLuminancePercentiles (Rec.709 luma by default)."""
    cols = colors.reshape(-1, 3)
    if cols.size == 0 or weights.size == 0:
        return [0.5] * len(percentiles)
    pairs: list[tuple[float, float]] = []
    total = 0.0
    for i in range(min(len(weights), len(cols))):
        r, g, b = cols[i]
        lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) if luma else (max(r, g, b) + min(r, g, b)) * 0.5
        w = float(weights[i])
        total += w
        pairs.append((lum, w))
    if total <= 0:
        return [0.5] * len(percentiles)
    pairs.sort(key=lambda x: x[0])
    cumulative: list[float] = []
    acc = 0.0
    for _, w in pairs:
        acc += w
        cumulative.append(acc / total)
    out: list[float] = []
    for pct in percentiles:
        idx = next((i for i, cw in enumerate(cumulative) if cw >= pct), len(pairs) - 1)
        out.append(pairs[idx][0])
    return out


def get_weighted_median_luminance(colors: np.ndarray, weights: np.ndarray) -> float:
    """Port of getWeightedMedianLuminance → getLuminancePercentiles(..., [0.5])."""
    vals = get_luminance_percentiles(colors, weights, (0.5,))
    return vals[0] if vals else 0.5


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
        luminances[i] = float(np.mean(result[i * 3 : i * 3 + 3]))

    sorted_indices = np.argsort(luminances)
    start_index = int(np.floor(len(sorted_indices) * dark_threshold))
    end_index = int(np.floor(len(sorted_indices) * (1.0 - bright_threshold)))
    normal_lum_index = int(np.floor(len(sorted_indices) * 0.65))

    def get_luminance(idx: int) -> float:
        return float(luminances[sorted_indices[idx]])

    normal_lum = get_luminance(normal_lum_index)
    lum_threshold = max(min(normal_lum - 70.0, 40.0), min(normal_lum - 25.0, 20.0))
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
    data = np.asarray(rgb_data, dtype=np.uint8).reshape(-1)
    if data.size == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float64)

    num_bins = 256 // bin_size
    total_bins = num_bins**3
    sums = np.zeros(total_bins * 3, dtype=np.float64)
    counts = np.zeros(total_bins, dtype=np.int64)

    for i in range(0, data.size, 3):
        r = int(data[i])
        g = int(data[i + 1])
        b = int(data[i + 2])
        r_i = r // bin_size
        g_i = g // bin_size
        b_i = b // bin_size
        idx = r_i * num_bins * num_bins + g_i * num_bins + b_i
        si = idx * 3
        sums[si : si + 3] += (r, g, b)
        counts[idx] += 1

    entries: list[tuple[np.ndarray, int]] = []
    for i in range(total_bins):
        count = int(counts[i])
        if count > pixel_count_threshold:
            si = i * 3
            avg = sums[si : si + 3] / count
            entries.append((avg, count))
    entries.sort(key=lambda x: x[1], reverse=True)
    total_pixels = data.size // 3
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
        masked_percent, masked_pixels = _get_face_masked_pixels(large_hwc, face_results, use_step=True)
        face_percent = masked_percent
        face_rgb, start_index, end_index = get_rgb_data_with_step(
            masked_pixels,
            2000,
            dark_threshold=_FACE_RGB_DARK_THRESHOLD,
            bright_threshold=_FACE_RGB_BRIGHT_THRESHOLD,
        )
        colors, weights = extract_colors(
            face_rgb,
            pixel_count_threshold=_FACE_PIXEL_COUNT_THRESHOLD,
            max_count=70,
        )
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
