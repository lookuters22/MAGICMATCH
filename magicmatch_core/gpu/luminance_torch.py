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
        from ..probe_parity.luminance import get_face_area_luminance_statistics
        from .device import hwc_torch_to_numpy

        large_np = hwc_torch_to_numpy(large_hwc)
        face_stats = get_face_area_luminance_statistics(large_np, face_detection_results, None)
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
