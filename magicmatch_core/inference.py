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


def _resize_hwc_to_nhwc(image_hwc: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """H×W×3 float [0,1] → [1, size, size, 3] bilinear."""
    image_hwc = np.asarray(image_hwc, dtype=np.float32)
    try:
        import torch
        import torch.nn.functional as F

        t = torch.from_numpy(image_hwc).permute(2, 0, 1).unsqueeze(0)
        t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
        out = t.permute(0, 2, 3, 1).numpy().astype(np.float32)
        return np.clip(out, 0.0, 1.0)
    except ImportError:
        from PIL import Image

        arr = (np.clip(image_hwc, 0, 1) * 255.0).astype(np.uint8)
        pil = Image.fromarray(arr, "RGB")
        pil = pil.resize((size, size), Image.Resampling.BILINEAR)
        out = np.asarray(pil, dtype=np.float32) / 255.0
        return out[np.newaxis, ...]


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
    from .lut import get_merged_lut, nn_outputs_to_flat

    lut3d, lut1d = run_inference_from_images(source_hwc, reference_hwc)
    return get_merged_lut(*nn_outputs_to_flat(lut1d, lut3d))
