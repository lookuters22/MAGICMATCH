"""Complete fused GPU color-match pipeline with phase profiling."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np
import torch

from ..inference_cuda import get_session_cuda, run_inference_from_images_cuda_tensors
from ..lut import get_merged_lut, nn_outputs_to_flat
from ..probe_parity.profile_stage import ProfileStageFlags, normalize_profile_stage, profile_stage_flags
from ..probe_parity.reference import NET_LONG_EDGE, prepare_worker_bitmap_source
from ..probe_parity.wb import DEFAULT_AS_SHOT_TEMP, DEFAULT_AS_SHOT_TINT
from .develop_torch import render_srgb_develop_torch
from .device import get_torch_device, gpu_pipeline_available, hwc_numpy_to_torch, hwc_torch_to_numpy
from .reference_prep_torch import prepare_net_reference_torch
from .reference_torch import render_detection_inputs_torch
from .resize_torch import fit_long_edge_torch
from .scene_extract_torch import estimate_base_adjustments_full_gpu, extract_scene_info_bitmap_full_gpu


@dataclass
class GpuPipelineState:
    merged_lut: np.ndarray
    base_adjustments: dict
    feed_tensor: torch.Tensor


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.inference_mode()
def build_merged_lut_full_gpu(
    source_hwc: np.ndarray,
    reference_hwc: np.ndarray,
    *,
    worker_feed: np.ndarray | None = None,
    feed_tensor: torch.Tensor | None = None,
    reference_tensor: torch.Tensor | None = None,
) -> GpuPipelineState:
    """Full GPU build: scene extract on device, GPU develop@1600, CUDA ONNX."""
    if not gpu_pipeline_available():
        from ..probe_parity.pipeline_cuda import build_merged_lut_probe_style_cuda

        feed = worker_feed if worker_feed is not None else prepare_worker_bitmap_source(source_hwc)
        merged, base = build_merged_lut_probe_style_cuda(source_hwc, reference_hwc)
        return GpuPipelineState(merged, base, hwc_numpy_to_torch(feed, get_torch_device()))

    device = get_torch_device()
    if feed_tensor is None:
        feed = worker_feed if worker_feed is not None else prepare_worker_bitmap_source(source_hwc)
        feed_tensor = hwc_numpy_to_torch(feed, device)
    base = estimate_base_adjustments_full_gpu(feed_tensor)
    long_t = fit_long_edge_torch(feed_tensor, NET_LONG_EDGE)
    net_t = render_srgb_develop_torch(long_t, base)
    if reference_tensor is None:
        ref_t = hwc_numpy_to_torch(reference_hwc, device)
    else:
        ref_t = reference_tensor.to(device=device, dtype=torch.float32)
    net_ref_t = prepare_net_reference_torch(ref_t)
    lut3d, lut1d = run_inference_from_images_cuda_tensors(net_t, net_ref_t)
    merged = get_merged_lut(*nn_outputs_to_flat(lut1d, lut3d))
    return GpuPipelineState(merged, base, feed_tensor)


@torch.inference_mode()
def apply_probe_export_full_gpu_tensor(
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


def apply_probe_export_full_gpu(
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
        feed = prepare_worker_bitmap_source(source_hwc)
        feed_tensor = hwc_numpy_to_torch(feed, device)
    base = base_adjustments
    if base is None:
        base = estimate_base_adjustments_full_gpu(feed_tensor)
    out_t = apply_probe_export_full_gpu_tensor(
        feed_tensor,
        merged_lut,
        strength,
        base_adjustments=base,
        profile_stage=profile_stage,
        lut_encoding=lut_encoding,
    )
    return hwc_torch_to_numpy(out_t)


@torch.inference_mode()
def color_match_one_shot_full_gpu(
    source_hwc: np.ndarray,
    reference_hwc: np.ndarray,
    strength: float,
    *,
    profile_stage: str = "current_profile_stages",
    lut_encoding: str = "srgb_srgb",
    source_tensor: torch.Tensor | None = None,
    reference_tensor: torch.Tensor | None = None,
) -> np.ndarray:
    """Fused full GPU build + apply; single JPEG normalize, no duplicate scene extract."""
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

    device = get_torch_device()
    feed_tensor = source_tensor
    if feed_tensor is not None:
        feed_tensor = feed_tensor.to(device=device, dtype=torch.float32)
        if feed_tensor.ndim == 4:
            feed_tensor = feed_tensor[0]
        feed = hwc_torch_to_numpy(feed_tensor)
        feed = prepare_worker_bitmap_source(feed)
        feed_tensor = hwc_numpy_to_torch(feed, device)
    else:
        feed = prepare_worker_bitmap_source(source_hwc)
        feed_tensor = hwc_numpy_to_torch(feed, device)

    ref_tensor = reference_tensor
    if ref_tensor is not None:
        ref_tensor = ref_tensor.to(device=device, dtype=torch.float32)
        if ref_tensor.ndim == 4:
            ref_tensor = ref_tensor[0]

    state = build_merged_lut_full_gpu(
        source_hwc,
        reference_hwc,
        worker_feed=feed,
        feed_tensor=feed_tensor,
        reference_tensor=ref_tensor,
    )
    out_t = apply_probe_export_full_gpu_tensor(
        state.feed_tensor,
        state.merged_lut,
        strength,
        base_adjustments=state.base_adjustments,
        profile_stage=profile_stage,
        lut_encoding=lut_encoding,
    )
    return hwc_torch_to_numpy(out_t)


def lut_hash(lut: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(lut, dtype=np.float32).tobytes()).hexdigest()[:16]


def profile_full_gpu_pipeline(
    source_hwc: np.ndarray,
    reference_hwc: np.ndarray,
    *,
    strength: float = 1.0,
) -> dict[str, float | str | bool]:
    """Phase timings in ms + lut_hash parity vs CPU build."""
    if not gpu_pipeline_available():
        return {"error": "cuda unavailable", "cuda_available": False}

    device = get_torch_device()
    out: dict[str, float | str | bool] = {"cuda_available": True}

    t0 = time.perf_counter()
    feed = prepare_worker_bitmap_source(source_hwc)
    out["jpeg_normalize_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    feed_t = hwc_numpy_to_torch(feed, device)
    _sync_cuda()
    out["upload_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    render_detection_inputs_torch(feed_t)
    _sync_cuda()
    out["detection_buffers_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    base = estimate_base_adjustments_full_gpu(feed_t)
    _sync_cuda()
    out["scene_extract_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    long_t = fit_long_edge_torch(feed_t, NET_LONG_EDGE)
    net_t = render_srgb_develop_torch(long_t, base)
    _sync_cuda()
    out["develop_net_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    ref_t = prepare_net_reference_torch(hwc_numpy_to_torch(reference_hwc, device))
    _sync_cuda()
    out["ref_prep_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    get_session_cuda()
    run_inference_from_images_cuda_tensors(net_t, ref_t)
    _sync_cuda()
    out["onnx_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    state = build_merged_lut_full_gpu(source_hwc, reference_hwc, worker_feed=feed, feed_tensor=feed_t)
    _sync_cuda()
    out["full_build_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    apply_probe_export_full_gpu(
        source_hwc,
        state.merged_lut,
        strength,
        base_adjustments=state.base_adjustments,
        feed_tensor=feed_t,
    )
    _sync_cuda()
    out["apply_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    color_match_one_shot_full_gpu(source_hwc, reference_hwc, strength)
    _sync_cuda()
    out["one_shot_fused_ms"] = (time.perf_counter() - t0) * 1000

    from ..inference import build_merged_lut_with_base

    merged_cpu, _ = build_merged_lut_with_base(source_hwc, reference_hwc)
    hash_cpu = lut_hash(merged_cpu)
    hash_gpu = lut_hash(state.merged_lut)
    max_abs = float(np.max(np.abs(np.asarray(merged_cpu) - np.asarray(state.merged_lut))))
    out["lut_hash_cpu"] = hash_cpu
    out["lut_hash_gpu"] = hash_gpu
    out["lut_hash_match"] = hash_cpu == hash_gpu
    out["lut_max_abs_delta"] = max_abs
    out["golden_lut_hash"] = "a48758ca22a2e389"
    out["cpu_matches_golden"] = hash_cpu == "a48758ca22a2e389"
    out["gpu_matches_golden"] = hash_gpu == "a48758ca22a2e389"

    out_np = color_match_one_shot_full_gpu(source_hwc, reference_hwc, strength)
    from ..probe_parity.pipeline import apply_probe_export

    _, base_cpu = build_merged_lut_with_base(source_hwc, reference_hwc)
    cpu_out = apply_probe_export(source_hwc, merged_cpu, strength, base_adjustments=base_cpu)
    out["output_mean_delta"] = float(np.mean(np.abs(out_np - cpu_out)))

    still_cpu = [
        "jpeg_normalize (worker q98 parity)",
        "detection buffer download for face/color/WB helpers (~300/2000 edge)",
        "face ONNX postprocess (NMS/box filter)",
        "reference JPEG q92 + WebP q92 on 256×256",
        "color_match ONNX feed .cpu().numpy()",
        "face detect/parse ONNX feed .cpu().numpy()",
        "final IMAGE output .cpu().numpy() for ComfyUI",
    ]
    out["still_on_cpu"] = still_cpu
    return out
