"""Bundled Adobe profile look table (HSV 36×16×16)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parents[2] / "assets"
ADOBE_LUT_PATH = ASSETS / "adobe_profile_look_table.npy"
HUE_DIVISIONS = 36
SAT_DIVISIONS = 16
VAL_DIVISIONS = 16


@lru_cache(maxsize=1)
def load_adobe_profile_look_table() -> np.ndarray:
    if not ADOBE_LUT_PATH.is_file():
        raise FileNotFoundError(
            f"MAGICMATCH Adobe profile look table missing: {ADOBE_LUT_PATH}\n"
            "Re-run asset extraction from polarr-data/adobe-color.ts."
        )
    data = np.load(ADOBE_LUT_PATH)
    expected = HUE_DIVISIONS * SAT_DIVISIONS * VAL_DIVISIONS * 3
    if data.size != expected:
        raise ValueError(f"Adobe look table expected {expected} floats, got {data.size}")
    return np.asarray(data, dtype=np.float32)


def profile_look_dims() -> tuple[int, int, int]:
    return (HUE_DIVISIONS, SAT_DIVISIONS, VAL_DIVISIONS)
