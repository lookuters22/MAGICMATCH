"""Fast fused GPU color-match pipeline — speed over strict CPU parity."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from ..inference_cuda import get_session_cuda, run_inference_from_images_cuda_tensors
from ..lut import get_merged_lut, nn_outputs_to_flat
from ..probe_parity.profile_stage import ProfileStageFlags, normalize_profile_stage, profile_stage_flags
from ..probe_parity.reference import NET_LONG_EDGE
from ..probe_parity.wb import DEFAULT_AS_SHOT_TEMP, DEFAULT_AS_SHOT_TINT
from .develop_torch import render_srgb_develop_torch
from .device import get_torch_device, gpu_pipeline_available, hwc_numpy_to_torch, hwc_torch_to_numpy
from .pipeline_full_gpu import GpuPipelineState
from .reference_prep_torch import prepare_net_reference_fast_torch
from .reference_torch import render_detection_inputs_torch
from .resize_torch import fit_long_edge_torch
from .scene_extract_fast import estimate_base_adjustments_fast_gpu


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _clamp_feed_tensor(feed_tensor: torch.Tensor) -> torch.Tensor:
    t = feed_tensor.to(device=get_torch_device(), dtype=torch.float32)
    if t.ndim == 4:
        t = t[0]
    return torch.clamp(t, 0.0, 1.0)


@torch.inference_mode()
def build_merged_lut_fast_gpu(
    source_hwc: np.ndarray,
    reference_hwc: np.ndarray,
    *,
    feed_tensor: torch.Tensor | None = None,
    reference_tensor: torch.Tensor | None = None,
) -> GpuPipelineState:
    """Fast GPU build: no JPEG normalize, bilinear ref, fast scene extract."""
    if not gpu_pipeline_available():
        from ..probe_parity.pipeline_cuda import build_merged_lut_probe_style_cuda

        merged, base = build_merged_lut_probe_style_cuda(source_hwc, reference_hwc)
        feed = np.clip(np.asarray(source_hwc, dtype=np.float32), 0.0, 1.0)
        return GpuPipelineState(merged, base, hwc_numpy_to_torch(feed, get_torch_device()))

    device = get_torch_device()
    if feed_tensor is None:
        feed_tensor = hwc_numpy_to_torch(np.clip(source_hwc, 0.0, 1.0), device)
    else:
        feed_tensor = _clamp_feed_tensor(feed_tensor)

    base = estimate_base_adjustments_fast_gpu(feed_tensor)
    long_t = fit_long_edge_torch(feed_tensor, NET_LONG_EDGE)
    net_t = render_srgb_develop_torch(long_t, base)

    if reference_tensor is None:
        ref_t = hwc_numpy_to_torch(reference_hwc, device)
    else:
        ref_t = _clamp_feed_tensor(reference_tensor)
    net_ref_t = prepare_net_reference_fast_torch(ref_t)

    lut3d, lut1d = run_inference_from_images_cuda_tensors(net_t, net_ref_t)
    merged = get_merged_lut(*nn_outputs_to_flat(lut1d, lut3d))
    return GpuPipelineState(merged, base, feed_tensor)


@torch.inference_mode()
def apply_probe_export_fast_gpu_tensor(
    feed_tensor: torch.Tensor,
    merged_lut: np.ndarray,
    strength: float,
    *,
    base_adjustments: dict,
    profile_stage: str = "current_profile_stages",
    lut_encoding: str = "srgb_srgb",
) -> torch.Tensor:
    strength = float(np.clip(strength, 0.0, 1.0))
    stage: ProfileStageFlags = profile_stage_flags(normalize_profile_stage(profile_stage))
    lut_adjustments = {**base_adjustments, "userLutStrength": strength}
    return render_srgb_develop_torch(
        _clamp_feed_tensor(feed_tensor),
        lut_adjustments,
        merged_lut=merged_lut,
        lut_strength=strength,
        lut_encoding=lut_encoding,
        force_color_look=stage["forceColorLookTableWithUserLut"],
        as_shot_temp=DEFAULT_AS_SHOT_TEMP,
        as_shot_tint=DEFAULT_AS_SHOT_TINT,
    )


def apply_probe_export_fast_gpu(
    source_hwc: np.ndarray,
    merged_lut: np.ndarray,
    strength: float,
    *,
    base_adjustments: dict | None = None,
    profile_stage: str = "current_profile_stages",
    lut_encoding: str = "srgb_srgb",
    feed_tensor: torch.Tensor | None = None,
) -> np.ndarray:
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
    if feed_tensor is None:
        feed_tensor = hwc_numpy_to_torch(np.clip(source_hwc, 0.0, 1.0), device)
    else:
        feed_tensor = _clamp_feed_tensor(feed_tensor)
    base = base_adjustments
    if base is None:
        base = estimate_base_adjustments_fast_gpu(feed_tensor)
    out_t = apply_probe_export_fast_gpu_tensor(
        feed_tensor,
        merged_lut,
        strength,
        base_adjustments=base,
        profile_stage=profile_stage,
        lut_encoding=lut_encoding,
    )
    return hwc_torch_to_numpy(out_t)


@torch.inference_mode()
def color_match_one_shot_fast_gpu(
    source_hwc: np.ndarray,
    reference_hwc: np.ndarray,
    strength: float,
    *,
    profile_stage: str = "current_profile_stages",
    lut_encoding: str = "srgb_srgb",
    source_tensor: torch.Tensor | None = None,
    reference_tensor: torch.Tensor | None = None,
) -> np.ndarray:
    """Fused fast GPU build + apply; no JPEG normalize or ref codec round-trips."""
    if not gpu_pipeline_available():
        from ..probe_parity.pipeline import apply_probe_export
        from ..inference_cuda import build_merged_lut_with_base_cuda

        merged, base = build_merged_lut_with_base_cuda(source_hwc, reference_hwc)
        return apply_probe_export(
            source_hwc,
            merged,
            strength,
            base_adjustments=base,
            profile_stage=profile_stage,
            lut_encoding=lut_encoding,
        )

    feed_tensor = source_tensor
    if feed_tensor is not None:
        feed_tensor = _clamp_feed_tensor(feed_tensor)
    else:
        feed_tensor = hwc_numpy_to_torch(np.clip(source_hwc, 0.0, 1.0), get_torch_device())

    ref_tensor = reference_tensor
    if ref_tensor is not None:
        ref_tensor = _clamp_feed_tensor(ref_tensor)

    state = build_merged_lut_fast_gpu(
        source_hwc,
        reference_hwc,
        feed_tensor=feed_tensor,
        reference_tensor=ref_tensor,
    )
    out_t = apply_probe_export_fast_gpu_tensor(
        state.feed_tensor,
        state.merged_lut,
        strength,
        base_adjustments=state.base_adjustments,
        profile_stage=profile_stage,
        lut_encoding=lut_encoding,
    )
    return hwc_torch_to_numpy(out_t)


def profile_fast_gpu_pipeline(
    source_hwc: np.ndarray,
    reference_hwc: np.ndarray,
    *,
    strength: float = 1.0,
) -> dict[str, float | str | bool | list[str]]:
    """Phase timings in ms for the fast GPU path."""
    if not gpu_pipeline_available():
        return {"error": "cuda unavailable", "cuda_available": False}

    device = get_torch_device()
    out: dict[str, float | str | bool | list[str]] = {"cuda_available": True}

    t0 = time.perf_counter()
    feed_t = hwc_numpy_to_torch(np.clip(source_hwc, 0.0, 1.0), device)
    _sync_cuda()
    out["upload_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    render_detection_inputs_torch(feed_t)
    _sync_cuda()
    out["detection_buffers_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    estimate_base_adjustments_fast_gpu(feed_t)
    _sync_cuda()
    out["scene_extract_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    base = estimate_base_adjustments_fast_gpu(feed_t)
    long_t = fit_long_edge_torch(feed_t, NET_LONG_EDGE)
    net_t = render_srgb_develop_torch(long_t, base)
    _sync_cuda()
    out["develop_net_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    prepare_net_reference_fast_torch(hwc_numpy_to_torch(reference_hwc, device))
    _sync_cuda()
    out["ref_prep_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    get_session_cuda()
    ref_t = prepare_net_reference_fast_torch(hwc_numpy_to_torch(reference_hwc, device))
    run_inference_from_images_cuda_tensors(net_t, ref_t)
    _sync_cuda()
    out["onnx_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    state = build_merged_lut_fast_gpu(source_hwc, reference_hwc, feed_tensor=feed_t)
    _sync_cuda()
    out["full_build_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    apply_probe_export_fast_gpu(
        source_hwc,
        state.merged_lut,
        strength,
        base_adjustments=state.base_adjustments,
        feed_tensor=feed_t,
    )
    _sync_cuda()
    out["apply_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    color_match_one_shot_fast_gpu(source_hwc, reference_hwc, strength)
    _sync_cuda()
    one_shot_ms = (time.perf_counter() - t0) * 1000
    out["one_shot_ms"] = one_shot_ms
    out["one_shot_fused_ms"] = one_shot_ms

    still_cpu = [
        "face ONNX postprocess (NMS/box filter)",
        "face/skin ONNX feed .cpu().numpy()",
        "color_match ONNX feed .cpu().numpy()",
        "gray-world WB scalar sync (3-vector)",
        "face luminance histogram scalar sync",
        "final IMAGE output .cpu().numpy() for ComfyUI",
    ]
    out["still_on_cpu"] = still_cpu
    out["parity_mode"] = "fast_approximate"
    return out
