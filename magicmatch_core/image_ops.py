"""Image sizing aligned with scripts/color_match_probe.py (standalone parity)."""

from __future__ import annotations

import numpy as np

# Same as MAX_DISPLAY_EDGE in color_match_probe.py
PROBE_MAX_EDGE = 960


def downscale_hwc_max_edge(hwc: np.ndarray, max_edge: int = PROBE_MAX_EDGE) -> np.ndarray:
    """Bilinear downscale H×W×3 float RGB so max(w,h) <= max_edge (probe fit_for_preview)."""
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
        ys = (np.linspace(0, h - 1, nh)).astype(np.int32)
        xs = (np.linspace(0, w - 1, nw)).astype(np.int32)
        return hwc[np.ix_(ys, xs)]


def prepare_apply_source(source_hwc: np.ndarray) -> np.ndarray:
    """Source pixels passed to apply_merged_lut_preview (probe source_rgb)."""
    return downscale_hwc_max_edge(source_hwc, PROBE_MAX_EDGE)
