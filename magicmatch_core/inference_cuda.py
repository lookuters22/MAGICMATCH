"""
CUDA ONNX inference for MAGICMATCH color-match graph.

Separate from inference.py (CPU parity default). Scene extract / develop / reference
stay on the NumPy CPU stack — only color_match.onnx sess.run uses the GPU when available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
except ImportError as e:
    raise ImportError("MAGICMATCH CUDA path: pip install onnxruntime-gpu") from e

from .inference import (
    DEFAULT_ONNX,
    INPUT_NAMES,
    OUTPUT_LUT1D,
    OUTPUT_LUT3D,
    _resize_hwc_to_nhwc,
)
from .onnx_providers import create_onnx_session, cuda_available, session_active_provider

try:
    import torch

    _TORCH = True
except ImportError:
    _TORCH = False

_SESSION_CUDA: ort.InferenceSession | None = None


def get_session_cuda(onnx_path: str | Path | None = None) -> ort.InferenceSession:
    global _SESSION_CUDA
    path = Path(onnx_path or DEFAULT_ONNX)
    if not path.is_file():
        raise FileNotFoundError(
            f"MAGICMATCH ONNX not found: {path}\n"
            "Expected: MAGICMATCH/models/color_match.onnx in this custom node folder."
        )
    if _SESSION_CUDA is None:
        _SESSION_CUDA = create_onnx_session(path, prefer_cuda=True)
    return _SESSION_CUDA


def run_inference_from_images_cuda(
    source_hwc: np.ndarray,
    reference_hwc: np.ndarray,
    *,
    session: ort.InferenceSession | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run color_match.onnx on GPU when CUDA EP is available; CPU fallback otherwise."""
    if _TORCH and torch.cuda.is_available():
        from .gpu.device import hwc_numpy_to_torch

        src_t = hwc_numpy_to_torch(source_hwc, torch.device("cuda"))
        return run_inference_from_images_cuda_tensors(src_t, reference_hwc, session=session)
    sess = session or get_session_cuda()
    feeds = {
        INPUT_NAMES[0]: _resize_hwc_to_nhwc(source_hwc),
        INPUT_NAMES[1]: _resize_hwc_to_nhwc(reference_hwc),
    }
    out_map = {o.name: o for o in sess.get_outputs()}
    names = [out_map[OUTPUT_LUT3D].name, out_map[OUTPUT_LUT1D].name]
    lut3d, lut1d = sess.run(names, feeds)
    return np.asarray(lut3d, dtype=np.float32), np.asarray(lut1d, dtype=np.float32)


def run_inference_from_images_cuda_tensors(
    source_hwc: "torch.Tensor | np.ndarray",
    reference_hwc: "torch.Tensor | np.ndarray",
    *,
    session: ort.InferenceSession | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run color_match.onnx; source/ref may be CUDA tensors (GPU resize, single ORT sync)."""
    from .gpu.resize_torch import hwc_to_nhwc_numpy

    sess = session or get_session_cuda()
    if _TORCH and isinstance(source_hwc, torch.Tensor):
        src_feed = hwc_to_nhwc_numpy(source_hwc)
    else:
        src_feed = _resize_hwc_to_nhwc(np.asarray(source_hwc, dtype=np.float32))
    if _TORCH and isinstance(reference_hwc, torch.Tensor):
        ref_feed = hwc_to_nhwc_numpy(reference_hwc)
    else:
        ref_feed = _resize_hwc_to_nhwc(np.asarray(reference_hwc, dtype=np.float32))
    feeds = {
        INPUT_NAMES[0]: src_feed,
        INPUT_NAMES[1]: ref_feed,
    }
    out_map = {o.name: o for o in sess.get_outputs()}
    names = [out_map[OUTPUT_LUT3D].name, out_map[OUTPUT_LUT1D].name]
    lut3d, lut1d = sess.run(names, feeds)
    return np.asarray(lut3d, dtype=np.float32), np.asarray(lut1d, dtype=np.float32)


def build_merged_lut_with_base_cuda(
    source_hwc: np.ndarray,
    reference_hwc: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """CPU parity scene extract + develop; CUDA ONNX only (golden LUT path)."""
    from .probe_parity.pipeline_cuda import build_merged_lut_probe_style_cuda

    return build_merged_lut_probe_style_cuda(source_hwc, reference_hwc)


def color_match_session_info() -> dict[str, str | bool]:
    """Report active provider for color_match.onnx (creates session if needed)."""
    sess = get_session_cuda()
    return {
        "model": "color_match.onnx",
        "cuda_available": cuda_available(),
        "active_provider": session_active_provider(sess),
        "providers": list(sess.get_providers()),
    }
