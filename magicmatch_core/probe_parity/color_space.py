"""RGB/HSV helpers from color-space.ts."""

from __future__ import annotations

import numpy as np


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    r, g, b = [float(np.clip(c, 0.0, 1.0)) for c in rgb[:3]]
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    delta = max_c - min_c
    h = 0.0
    s = 0.0 if max_c == 0 else delta / max_c
    v = max_c
    if delta != 0:
        if max_c == r:
            h = (g - b) / delta + (6 if g < b else 0)
        elif max_c == g:
            h = (b - r) / delta + 2
        else:
            h = (r - g) / delta + 4
        h /= 6.0
    return np.array([h, s, v], dtype=np.float64)
