"""Fused GPU color-match: single JPEG normalize, minimal CPU sync."""

from __future__ import annotations

import numpy as np
import torch

from ..inference_cuda import get_session_cuda, run_inference_from_images_cuda_tensors
from ..lut import get_merged_lut, nn_outputs_to_flat
from ..probe_parity.base_adjustments_cuda import estimate_base_adjustments_cuda
from ..probe_parity.profile_stage import ProfileStageFlags, normalize_profile_stage, profile_stage_flags
from ..probe_parity.reference import NET_LONG_EDGE, prepare_net_reference, prepare_worker_bitmap_source
from ..probe_parity.wb import DEFAULT_AS_SHOT_TEMP, DEFAULT_AS_SHOT_TINT
from .develop_torch import render_srgb_develop_torch
from .device import get_torch_device, gpu_pipeline_available, hwc_numpy_to_torch, hwc_torch_to_numpy
from .resize_torch import fit_long_edge_torch


@torch.inference_mode()
def build_merged_lut_probe_style_gpu(
    source_hwc: np.ndarray,
    reference_hwc: np.ndarray,
    *,
    worker_feed: np.ndarray | None = None,
    feed_tensor: torch.Tensor | None = None,
) -> tuple[np.ndarray, dict, torch.Tensor]:
    """
    GPU build. Returns (merged_lut, base_adjustments, feed_tensor_on_gpu) for reuse in apply.
    """
    device = get_torch_device()
    if feed_tensor is None:
        feed = worker_feed if worker_feed is not None else prepare_worker_bitmap_source(source_hwc)
        feed_tensor = hwc_numpy_to_torch(feed, device)
    base = estimate_base_adjustments_cuda(
        hwc_torch_to_numpy(feed_tensor),
        worker_feed_prepared=True,
        use_gpu_detection_buffers=gpu_pipeline_available(),
    )
    long_t = fit_long_edge_torch(feed_tensor, NET_LONG_EDGE)
    net_t = render_srgb_develop_torch(long_t, base)
    net_ref = prepare_net_reference(reference_hwc)
    lut3d, lut1d = run_inference_from_images_cuda_tensors(net_t, net_ref)
    merged = get_merged_lut(*nn_outputs_to_flat(lut1d, lut3d))
    return merged, base, feed_tensor


@torch.inference_mode()
def apply_probe_export_gpu_tensor(
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
        feed_tensor,
        lut_adjustments,
        merged_lut=merged_lut,
        lut_strength=strength,
        lut_encoding=lut_encoding,
        force_color_look=stage["forceColorLookTableWithUserLut"],
        as_shot_temp=DEFAULT_AS_SHOT_TEMP,
        as_shot_tint=DEFAULT_AS_SHOT_TINT,
    )


def apply_probe_export_gpu(
    source_hwc: np.ndarray,
    merged_lut: np.ndarray,
    strength: float,
    *,
    base_adjustments: dict | None = None,
    profile_stage: str = "current_profile_stages",
    lut_encoding: str = "srgb_srgb",
    worker_feed: np.ndarray | None = None,
    feed_tensor: torch.Tensor | None = None,
) -> np.ndarray:
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
    if feed_tensor is None:
        feed = worker_feed if worker_feed is not None else prepare_worker_bitmap_source(source_hwc)
        feed_tensor = hwc_numpy_to_torch(feed, device)
    base = base_adjustments
    if base is None:
        base = estimate_base_adjustments_cuda(
            hwc_torch_to_numpy(feed_tensor),
            worker_feed_prepared=True,
            use_gpu_detection_buffers=True,
        )
    out_t = apply_probe_export_gpu_tensor(
        feed_tensor,
        merged_lut,
        strength,
        base_adjustments=base,
        profile_stage=profile_stage,
        lut_encoding=lut_encoding,
    )
    return hwc_torch_to_numpy(out_t)


@torch.inference_mode()
def color_match_one_shot_gpu(
    source_hwc: np.ndarray,
    reference_hwc: np.ndarray,
    strength: float,
    *,
    profile_stage: str = "current_profile_stages",
    lut_encoding: str = "srgb_srgb",
) -> np.ndarray:
    """Fused build + apply: one JPEG normalize, one GPU feed tensor, no duplicate scene/resize."""
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

    feed = prepare_worker_bitmap_source(source_hwc)
    merged, base, feed_t = build_merged_lut_probe_style_gpu(
        source_hwc,
        reference_hwc,
        worker_feed=feed,
    )
    out_t = apply_probe_export_gpu_tensor(
        feed_t,
        merged,
        strength,
        base_adjustments=base,
        profile_stage=profile_stage,
        lut_encoding=lut_encoding,
    )
    return hwc_torch_to_numpy(out_t)


def profile_gpu_pipeline(
    source_hwc: np.ndarray,
    reference_hwc: np.ndarray,
) -> dict[str, float]:
    """Phase timings in milliseconds (CUDA build + apply)."""
    import time

    if not gpu_pipeline_available():
        return {"error": "cuda unavailable"}

    device = get_torch_device()
    out: dict[str, float] = {}

    t0 = time.perf_counter()
    feed = prepare_worker_bitmap_source(source_hwc)
    out["jpeg_normalize_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    feed_t = hwc_numpy_to_torch(feed, device)
    base = estimate_base_adjustments_cuda(
        feed, worker_feed_prepared=True, use_gpu_detection_buffers=True
    )
    torch.cuda.synchronize()
    out["scene_extract_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    long_t = fit_long_edge_torch(feed_t, NET_LONG_EDGE)
    net_t = render_srgb_develop_torch(long_t, base)
    torch.cuda.synchronize()
    out["develop_net_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    net_ref = prepare_net_reference(reference_hwc)
    out["ref_prep_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    get_session_cuda()
    run_inference_from_images_cuda_tensors(net_t, net_ref)
    torch.cuda.synchronize()
    out["onnx_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    merged, _, _ = build_merged_lut_probe_style_gpu(source_hwc, reference_hwc, worker_feed=feed)
    out["full_build_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    apply_probe_export_gpu(
        source_hwc,
        merged,
        1.0,
        base_adjustments=base,
        feed_tensor=feed_t,
    )
    torch.cuda.synchronize()
    out["apply_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    color_match_one_shot_gpu(source_hwc, reference_hwc, 1.0)
    torch.cuda.synchronize()
    out["one_shot_fused_ms"] = (time.perf_counter() - t0) * 1000

    return out
