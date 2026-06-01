"""Full GPU-accelerated probe build + apply (ONNX CUDA + Torch develop/buffers)."""

from __future__ import annotations

import numpy as np

from ..inference_cuda import run_inference_from_images_cuda
from ..lut import get_merged_lut, nn_outputs_to_flat
from ..probe_parity.base_adjustments_cuda import estimate_base_adjustments_cuda
from ..probe_parity.profile_stage import ProfileStageFlags, normalize_profile_stage, profile_stage_flags
from ..probe_parity.reference import NET_LONG_EDGE, fit_long_edge, prepare_net_reference, prepare_worker_bitmap_source
from ..probe_parity.wb import DEFAULT_AS_SHOT_TEMP, DEFAULT_AS_SHOT_TINT
from .develop_torch import render_srgb_develop_torch
from .device import get_torch_device, gpu_pipeline_available, hwc_numpy_to_torch, hwc_torch_to_numpy


def _render_for_net_gpu(source_hwc: np.ndarray, adjustments: dict, device) -> np.ndarray:
    long = fit_long_edge(source_hwc, NET_LONG_EDGE)
    t = hwc_numpy_to_torch(long, device)
    out = render_srgb_develop_torch(t, adjustments)
    return hwc_torch_to_numpy(out)


def build_merged_lut_probe_style_gpu_v1(
    source_hwc: np.ndarray,
    reference_hwc: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """
    Probe-parity build with GPU develop + CUDA ONNX.
    Scene extract still uses CPU stats on GPU-generated detection buffers.
    """
    device = get_torch_device()
    source_hwc = prepare_worker_bitmap_source(source_hwc)
    base = estimate_base_adjustments_cuda(
        source_hwc,
        worker_feed_prepared=True,
        use_gpu_detection_buffers=gpu_pipeline_available(),
    )
    net_source = _render_for_net_gpu(source_hwc, base, device)
    net_ref = prepare_net_reference(reference_hwc)
    lut3d, lut1d = run_inference_from_images_cuda(net_source, net_ref)
    merged = get_merged_lut(*nn_outputs_to_flat(lut1d, lut3d))
    return merged, base


def apply_probe_export_gpu_v1(
    source_hwc: np.ndarray,
    merged_lut: np.ndarray,
    strength: float,
    *,
    base_adjustments: dict | None = None,
    profile_stage: str = "current_profile_stages",
    lut_encoding: str = "srgb_srgb",
) -> np.ndarray:
    """Full-res develop + LUT on GPU when CUDA is available."""
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return np.asarray(source_hwc, dtype=np.float32).copy()

    if not gpu_pipeline_available():
        from ..probe_parity.pipeline import apply_probe_export

        return apply_probe_export(
            source_hwc,
            merged_lut,
            strength,
            base_adjustments=base_adjustments,
            profile_stage=profile_stage,
            lut_encoding=lut_encoding,
        )

    device = get_torch_device()
    source_hwc = prepare_worker_bitmap_source(source_hwc)
    base = base_adjustments
    if base is None:
        base = estimate_base_adjustments_cuda(source_hwc, use_gpu_detection_buffers=True)
    stage_name = normalize_profile_stage(profile_stage)
    stage: ProfileStageFlags = profile_stage_flags(stage_name)
    lut_adjustments = {**base, "userLutStrength": strength}
    t = hwc_numpy_to_torch(source_hwc, device)
    out = render_srgb_develop_torch(
        t,
        lut_adjustments,
        merged_lut=merged_lut,
        lut_strength=strength,
        lut_encoding=lut_encoding,
        force_color_look=stage["forceColorLookTableWithUserLut"],
        as_shot_temp=DEFAULT_AS_SHOT_TEMP,
        as_shot_tint=DEFAULT_AS_SHOT_TINT,
    )
    return hwc_torch_to_numpy(out)
