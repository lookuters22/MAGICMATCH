"""GPU luminance statistics — port of probe_parity/luminance.py hot path."""

from __future__ import annotations

import torch

from ..probe_parity.luminance import (
    LuminanceStatistics,
    compute_percentiles,
    get_shadows_mean,
)


def _lightness_from_small_hwc(small_hwc: torch.Tensor) -> torch.Tensor:
    """Match get_lightness_rgba(hwc_to_rgba_uint8(small)) — uint8 truncate, not round."""
    rgb_u8 = (torch.clamp(small_hwc, 0.0, 1.0) * 255.0).to(torch.uint8)
    r = rgb_u8[..., 0].float() / 255.0
    g = rgb_u8[..., 1].float() / 255.0
    b = rgb_u8[..., 2].float() / 255.0
    return (torch.maximum(torch.maximum(r, g), b) + torch.minimum(torch.minimum(r, g), b)) * 0.5


def _histogram_256(lum: torch.Tensor) -> torch.Tensor:
    idx = torch.clamp(torch.floor(lum.reshape(-1) * 256.0).to(torch.int64), 0, 255)
    return torch.bincount(idx, minlength=256)


@torch.inference_mode()
def get_face_area_luminance_statistics_torch(
    large_hwc: torch.Tensor,
    face_detection_results: list[dict],
) -> dict[str, float | list[float] | int]:
    """GPU face-area luminance from face boxes only (no full-buffer CPU download)."""
    h, w, _ = large_hwc.shape
    total_pixels = max(h * w, 1)
    face_lum = 0.0
    face_percent = 0.0
    face_percentiles: list[float] = []
    face_clipping_highlights = 0.0
    face_clipping_shadows = 0.0
    face_count = len(face_detection_results)

    if face_detection_results:
        lum_parts: list[torch.Tensor] = []
        face_pixel_count = 0
        for result in face_detection_results:
            x1, y1, x2, y2 = result["face_box"]
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w, int(x2))
            y2 = min(h, int(y2))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = large_hwc[y1:y2, x1:x2]
            lum_parts.append(_lightness_from_small_hwc(crop).reshape(-1))
            face_pixel_count += (x2 - x1) * (y2 - y1)
        if lum_parts:
            face_lum_t = torch.cat(lum_parts)
            face_histogram = _histogram_256(face_lum_t)
            hist_np = face_histogram.cpu().numpy()
            face_lum = float(face_lum_t.mean().item())
            face_percent = float(face_pixel_count / total_pixels)
            face_percentiles = compute_percentiles(hist_np)
            n = max(int(face_lum_t.numel()), 1)
            face_clipping_highlights = float(hist_np[251:256].sum() / n)
            face_clipping_shadows = float((hist_np[0] if len(hist_np) else 0) / n)

    return {
        "facePercentiles": face_percentiles,
        "facePercent": face_percent,
        "faceLum": face_lum,
        "faceCount": face_count,
        "faceClippingHighlights": face_clipping_highlights,
        "faceClippingShadows": face_clipping_shadows,
    }


@torch.inference_mode()
def get_luminance_statistics_torch(
    small_hwc: torch.Tensor,
    large_hwc: torch.Tensor,
    face_detection_results: list,
) -> LuminanceStatistics:
    """GPU port of get_luminance_statistics; syncs only 256-bin histogram + scalars."""
    lum = _lightness_from_small_hwc(small_hwc)
    histogram = _histogram_256(lum)
    hist_np = histogram.cpu().numpy()

    avg_lum = float(lum.mean().item())
    shadows_mean = get_shadows_mean(hist_np)
    n = max(int(lum.numel()), 1)
    clipping = float(hist_np[251:256].sum() / n)
    percentiles = compute_percentiles(hist_np)

    face_percent = 0.0
    face_lum = 0.0
    face_percentiles: list[float] = []
    if face_detection_results:
        face_stats = get_face_area_luminance_statistics_torch(large_hwc, face_detection_results)
        face_percentiles = face_stats["facePercentiles"]
        face_percent = float(face_stats["facePercent"])
        face_lum = float(face_stats["faceLum"])

    return LuminanceStatistics(
        avg_lum=avg_lum,
        shadows_mean=shadows_mean,
        clipping_percent=clipping,
        percentiles=percentiles,
        face_percentiles=face_percentiles,
        face_percent=face_percent,
        face_lum=face_lum,
    )
