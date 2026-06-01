"""Apply merged LUT (Polarr probe style or legacy numpy)."""

from __future__ import annotations

import numpy as np

from .lut import apply_merged_lut_preview
from .polarr_lut_rgb import apply_polarr_color_match_probe_style

RENDER_PROBE_EXPORT = "probe_export"
RENDER_POLARR_PROBE = "polarr_probe"
RENDER_NUMPY = "numpy_legacy"


def apply_merged_lut_output(
    srgb_hwc: np.ndarray,
    merged_lut: np.ndarray,
    strength: float,
    *,
    render_mode: str = RENDER_PROBE_EXPORT,
    lut_encoding: str = "srgb_srgb",
    profile_stage: str = "current_profile_stages",
    base_adjustments: dict | None = None,
) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.0))
    if render_mode == RENDER_NUMPY:
        return apply_merged_lut_preview(srgb_hwc, merged_lut, strength)
    if render_mode == RENDER_PROBE_EXPORT:
        from .probe_parity.pipeline import apply_probe_export

        return apply_probe_export(
            srgb_hwc,
            merged_lut,
            strength,
            base_adjustments=base_adjustments,
            profile_stage=profile_stage,
            lut_encoding=lut_encoding,
        )
    return apply_polarr_color_match_probe_style(
        srgb_hwc,
        merged_lut,
        strength,
        encoding=lut_encoding,
        base_adjustments=base_adjustments,
        profile_stage=profile_stage,
    )
