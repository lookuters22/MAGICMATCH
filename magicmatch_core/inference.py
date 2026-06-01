"""
ONNX inference for MAGICMATCH color-match graph (Comfy IMAGE tensors).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import onnxruntime as ort
except ImportError as e:
    raise ImportError("MAGICMATCH: pip install onnxruntime") from e

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONNX = PACKAGE_ROOT / "models" / "color_match.onnx"
IMAGE_SIZE = 256

INPUT_NAMES = ("input_img:0", "ref_img:0")
OUTPUT_LUT3D = "3D_LUT"
OUTPUT_LUT1D = "1D_LUT"

_SESSION: ort.InferenceSession | None = None


def _bilinear_resize_hwc(image_hwc: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinear resize (align_corners=False) matching tf.image.resizeBilinear."""
    in_h, in_w, _ = image_hwc.shape
    if in_h == out_h and in_w == out_w:
        return image_hwc
    ys = (np.arange(out_h, dtype=np.float32) + 0.5) / out_h * in_h - 0.5
    xs = (np.arange(out_w, dtype=np.float32) + 0.5) / out_w * in_w - 0.5
    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, in_h - 1)
    x1 = np.clip(x0 + 1, 0, in_w - 1)
    y0 = np.clip(y0, 0, in_h - 1)
    x0 = np.clip(x0, 0, in_w - 1)
    fy = (ys - y0).astype(np.float32)
    fx = (xs - x0).astype(np.float32)
    out = np.empty((out_h, out_w, 3), dtype=np.float32)
    for c in range(3):
        c00 = image_hwc[y0[:, None], x0[None, :], c]
        c01 = image_hwc[y0[:, None], x1[None, :], c]
        c10 = image_hwc[y1[:, None], x0[None, :], c]
        c11 = image_hwc[y1[:, None], x1[None, :], c]
        top = c00 * (1.0 - fx)[None, :] + c01 * fx[None, :]
        bot = c10 * (1.0 - fx)[None, :] + c11 * fx[None, :]
        out[..., c] = top * (1.0 - fy)[:, None] + bot * fy[:, None]
    return np.clip(out, 0.0, 1.0)


def _resize_hwc_to_nhwc(image_hwc: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """
    H×W×3 float [0,1] → [1, size, size, 3].

    Matches neural-color-match.ts createTensor: high-quality downscale when needed,
    then tf.image.resizeBilinear (always, even at 256×256).
    """
    from PIL import Image

    image_hwc = np.clip(np.asarray(image_hwc, dtype=np.float32), 0.0, 1.0)
    arr = (image_hwc * 255.0).astype(np.uint8)
    pil = Image.fromarray(arr, "RGB")
    if pil.size != (size, size):
        pil = pil.resize((size, size), Image.Resampling.LANCZOS)
        image_hwc = np.asarray(pil, dtype=np.float32) / 255.0
    out = _bilinear_resize_hwc(image_hwc, size, size)
    return np.clip(out[np.newaxis, ...], 0.0, 1.0)


def get_session(onnx_path: str | Path | None = None) -> ort.InferenceSession:
    global _SESSION
    path = Path(onnx_path or DEFAULT_ONNX)
    if not path.is_file():
        raise FileNotFoundError(
            f"MAGICMATCH ONNX not found: {path}\n"
            "Expected: MAGICMATCH/models/color_match.onnx in this custom node folder."
        )
    if _SESSION is None:
        _SESSION = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return _SESSION


def run_inference_from_images(
    source_hwc: np.ndarray,
    reference_hwc: np.ndarray,
    *,
    session: ort.InferenceSession | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run graph on H×W×3 RGB (float 0–1). Returns lut3d, lut1d."""
    sess = session or get_session()
    feeds = {
        INPUT_NAMES[0]: _resize_hwc_to_nhwc(source_hwc),
        INPUT_NAMES[1]: _resize_hwc_to_nhwc(reference_hwc),
    }
    out_map = {o.name: o for o in sess.get_outputs()}
    names = [out_map[OUTPUT_LUT3D].name, out_map[OUTPUT_LUT1D].name]
    lut3d, lut1d = sess.run(names, feeds)
    return np.asarray(lut3d, dtype=np.float32), np.asarray(lut1d, dtype=np.float32)


def build_merged_lut(source_hwc: np.ndarray, reference_hwc: np.ndarray) -> np.ndarray:
    merged, _base = build_merged_lut_with_base(source_hwc, reference_hwc)
    return merged


def build_merged_lut_with_base(
    source_hwc: np.ndarray,
    reference_hwc: np.ndarray,
) -> tuple[np.ndarray, dict]:
    from .probe_parity.pipeline import build_merged_lut_probe_style

    return build_merged_lut_probe_style(source_hwc, reference_hwc)


def build_merged_lut_legacy(source_hwc: np.ndarray, reference_hwc: np.ndarray) -> np.ndarray:
    """Bare PIL 256×256 ONNX feed (pre-probe-parity)."""
    from .lut import get_merged_lut, nn_outputs_to_flat

    lut3d, lut1d = run_inference_from_images(source_hwc, reference_hwc)
    return get_merged_lut(*nn_outputs_to_flat(lut1d, lut3d))
