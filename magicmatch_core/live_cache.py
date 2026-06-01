"""Pack source + merged LUT for in-node live preview (Comfy UI payload)."""

from __future__ import annotations

import base64
import io

import numpy as np

LIVE_PREVIEW_MAX_EDGE = 1024


def _downscale_hwc(hwc: np.ndarray, max_edge: int) -> np.ndarray:
    hwc = np.asarray(hwc, dtype=np.float32)
    h, w, _ = hwc.shape
    scale = min(1.0, max_edge / max(h, w))
    if scale >= 1.0:
        return hwc.copy()
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    try:
        from PIL import Image

        arr = (np.clip(hwc, 0, 1) * 255.0).astype(np.uint8)
        pil = Image.fromarray(arr, "RGB").resize((nw, nh), Image.Resampling.BILINEAR)
        return np.asarray(pil, dtype=np.float32) / 255.0
    except ImportError:
        # Nearest stride fallback
        ys = (np.linspace(0, h - 1, nh)).astype(np.int32)
        xs = (np.linspace(0, w - 1, nw)).astype(np.int32)
        return hwc[np.ix_(ys, xs)]


def pack_live_cache(source_hwc: np.ndarray, merged_lut: np.ndarray) -> dict:
    """JSON-serializable bundle for the Comfy front-end (PNG + float32 LUT)."""
    from PIL import Image

    small = _downscale_hwc(source_hwc, LIVE_PREVIEW_MAX_EDGE)
    sh, sw, _ = small.shape
    merged = np.asarray(merged_lut, dtype=np.float32).reshape(-1)

    buf = io.BytesIO()
    Image.fromarray((np.clip(small, 0, 1) * 255.0).astype(np.uint8), "RGB").save(
        buf, format="PNG", optimize=True
    )

    return {
        "src_png": base64.b64encode(buf.getvalue()).decode("ascii"),
        "w": int(sw),
        "h": int(sh),
        "lut": base64.b64encode(merged.tobytes()).decode("ascii"),
        "lut_len": int(merged.size),
        "lut_size": 25,
    }
