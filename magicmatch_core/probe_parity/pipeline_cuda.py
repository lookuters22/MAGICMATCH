"""Probe-parity build pipeline with CUDA ONNX runners (optional GPU path)."""

from __future__ import annotations

import numpy as np

from ..inference_cuda import run_inference_from_images_cuda
from ..lut import get_merged_lut, nn_outputs_to_flat
from .base_adjustments import estimate_base_adjustments
from .develop import render_srgb_develop
from .pipeline import _render_for_net
from .reference import prepare_net_reference, prepare_worker_bitmap_source


def build_merged_lut_probe_style_cuda(
    source_hwc: np.ndarray,
    reference_hwc: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """
    Same probe-style build as pipeline.build_merged_lut_probe_style (CPU scene extract
    + CPU develop@1600), but color_match.onnx runs on CUDAExecutionProvider when available.
    """
    source_hwc = prepare_worker_bitmap_source(source_hwc)
    base = estimate_base_adjustments(source_hwc, worker_feed_prepared=True)
    net_source = _render_for_net(source_hwc, base)
    net_ref = prepare_net_reference(reference_hwc)
    lut3d, lut1d = run_inference_from_images_cuda(net_source, net_ref)
    merged = get_merged_lut(*nn_outputs_to_flat(lut1d, lut3d))
    return merged, base
