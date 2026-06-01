"""Pack source + merged LUT for in-node live preview (Comfy UI payload)."""

from __future__ import annotations

import base64
import io

import numpy as np

from .image_ops import PROBE_MAX_EDGE, downscale_hwc_max_edge


def pack_live_cache(source_hwc: np.ndarray, merged_lut: np.ndarray) -> dict:
    """JSON-serializable bundle for the Comfy front-end (PNG + float32 LUT)."""
    from PIL import Image

    small = downscale_hwc_max_edge(source_hwc, PROBE_MAX_EDGE)
    merged = np.asarray(merged_lut, dtype=np.float32).reshape(-1)

    arr_u8 = (np.clip(small, 0, 1) * 255.0).astype(np.uint8)
    pil = Image.fromarray(arr_u8, "RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)

    return {
        "src_png": base64.b64encode(buf.getvalue()).decode("ascii"),
        "w": int(pil.width),
        "h": int(pil.height),
        "lut": base64.b64encode(merged.tobytes()).decode("ascii"),
        "lut_len": int(merged.size),
        "lut_size": 25,
    }
