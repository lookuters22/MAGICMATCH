"""End-to-end probe-parity build + apply for ComfyUI."""

from __future__ import annotations

import numpy as np

from ..inference import run_inference_from_images
from ..lut import get_merged_lut, nn_outputs_to_flat
from .base_adjustments import estimate_base_adjustments
from .develop import render_srgb_develop
from .profile_stage import ProfileStageFlags, normalize_profile_stage, profile_stage_flags
from .reference import NET_INPUT_SIZE, NET_LONG_EDGE, fit_long_edge, prepare_net_reference, resize_hwc
from .wb import DEFAULT_AS_SHOT_TEMP, DEFAULT_AS_SHOT_TINT


def _render_for_net(source_hwc: np.ndarray, adjustments: dict) -> np.ndarray:
    """1600-edge develop render → 256 ONNX input (inputWithPreset, bitmap Standard)."""
    long = fit_long_edge(source_hwc, NET_LONG_EDGE)
    rendered = render_srgb_develop(
        long,
        adjustments,
        force_color_look=False,
    )
    return resize_hwc(rendered, NET_INPUT_SIZE, NET_INPUT_SIZE, high_quality=True)


def build_merged_lut_probe_style(
    source_hwc: np.ndarray,
    reference_hwc: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """
    Probe-style ONNX build:
    - estimate baseAdjustments from source luminance
    - net source = develop(source@1600, baseAdjustments) → 256
    - net ref = reference@1600 → 256
    """
    base = estimate_base_adjustments(source_hwc)
    net_source = _render_for_net(source_hwc, base)
    net_ref = prepare_net_reference(reference_hwc)
    lut3d, lut1d = run_inference_from_images(net_source, net_ref)
    merged = get_merged_lut(*nn_outputs_to_flat(lut1d, lut3d))
    return merged, base


def apply_probe_export(
    source_hwc: np.ndarray,
    merged_lut: np.ndarray,
    strength: float,
    *,
    base_adjustments: dict | None = None,
    profile_stage: str = "current_profile_stages",
    lut_encoding: str = "srgb_srgb",
) -> np.ndarray:
    """Queued output path: develop stack + user RGB LUT at full resolution."""
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return np.asarray(source_hwc, dtype=np.float32).copy()

    base = base_adjustments or estimate_base_adjustments(source_hwc)
    stage_name = normalize_profile_stage(profile_stage)
    stage: ProfileStageFlags = profile_stage_flags(stage_name)
    lut_adjustments = {
        **base,
        "userLutStrength": strength,
    }
    return render_srgb_develop(
        source_hwc,
        lut_adjustments,
        merged_lut=merged_lut,
        lut_strength=strength,
        lut_encoding=lut_encoding,
        force_color_look=stage["forceColorLookTableWithUserLut"],
        as_shot_temp=DEFAULT_AS_SHOT_TEMP,
        as_shot_tint=DEFAULT_AS_SHOT_TINT,
    )
