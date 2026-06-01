"""Fused GPU color-match: single JPEG normalize, minimal CPU sync."""

from __future__ import annotations

import numpy as np
import torch

from .pipeline_full_gpu import (
    apply_probe_export_full_gpu,
    apply_probe_export_full_gpu_tensor,
    build_merged_lut_full_gpu,
    color_match_one_shot_full_gpu,
    profile_full_gpu_pipeline,
)


@torch.inference_mode()
def build_merged_lut_probe_style_gpu(
    source_hwc: np.ndarray,
    reference_hwc: np.ndarray,
    *,
    worker_feed: np.ndarray | None = None,
    feed_tensor: torch.Tensor | None = None,
) -> tuple[np.ndarray, dict, torch.Tensor]:
    """GPU build. Returns (merged_lut, base_adjustments, feed_tensor_on_gpu) for reuse in apply."""
    state = build_merged_lut_full_gpu(
        source_hwc,
        reference_hwc,
        worker_feed=worker_feed,
        feed_tensor=feed_tensor,
    )
    return state.merged_lut, state.base_adjustments, state.feed_tensor


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
    return apply_probe_export_full_gpu_tensor(
        feed_tensor,
        merged_lut,
        strength,
        base_adjustments=base_adjustments,
        profile_stage=profile_stage,
        lut_encoding=lut_encoding,
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
    return apply_probe_export_full_gpu(
        source_hwc,
        merged_lut,
        strength,
        base_adjustments=base_adjustments,
        profile_stage=profile_stage,
        lut_encoding=lut_encoding,
        feed_tensor=feed_tensor,
    )


@torch.inference_mode()
def color_match_one_shot_gpu(
    source_hwc: np.ndarray,
    reference_hwc: np.ndarray,
    strength: float,
    *,
    profile_stage: str = "current_profile_stages",
    lut_encoding: str = "srgb_srgb",
) -> np.ndarray:
    return color_match_one_shot_full_gpu(
        source_hwc,
        reference_hwc,
        strength,
        profile_stage=profile_stage,
        lut_encoding=lut_encoding,
    )


def profile_gpu_pipeline(
    source_hwc: np.ndarray,
    reference_hwc: np.ndarray,
) -> dict[str, float]:
    """Phase timings in milliseconds (CUDA build + apply)."""
    report = profile_full_gpu_pipeline(source_hwc, reference_hwc)
    if "error" in report:
        return report
    return {k: v for k, v in report.items() if isinstance(v, (int, float)) and k.endswith("_ms")}
