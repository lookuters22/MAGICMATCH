"""
Polarr RGB user-LUT apply (port of applyLookTableRGB in renderer-cpu.ts).

Expects merged_lut flat RGBc interleaved (get_merged_lut output), 25³ divisions,
texture layout width = sat², height = hue (same as createUserLutTexture).
"""

from __future__ import annotations

import numpy as np

from .polarr_color_space import (
    ENCODING_PRESETS,
    lut_gamma_decode,
    lut_gamma_encode,
    lut_primaries_decode,
    lut_primaries_encode,
    prophoto_to_srgb,
    srgb_to_prophoto,
)

LUT_SIZE = 25


def _sample_bilinear(tex: np.ndarray, width: int, height: int, u: float, v: float) -> np.ndarray:
    x = u * (width - 1)
    y = v * (height - 1)
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    fx = x - x0
    fy = y - y0

    def pix(ix: int, iy: int) -> np.ndarray:
        idx = (iy * width + ix) * 3
        return tex[idx : idx + 3].astype(np.float32)

    c00, c01 = pix(x0, y0), pix(x1, y0)
    c10, c11 = pix(x0, y1), pix(x1, y1)
    c0 = c00 * (1.0 - fx) + c01 * fx
    c1 = c10 * (1.0 - fx) + c11 * fx
    return c0 * (1.0 - fy) + c1 * fy


def _apply_look_table_rgb_pixel(
    color: np.ndarray,
    tex: np.ndarray,
    lut_size: int,
    strength: float,
    gamma_type: int,
    primaries_type: int,
) -> np.ndarray:
    orig = color.astype(np.float32, copy=True)
    tmp = lut_primaries_encode(orig, primaries_type)
    tmp = np.clip(tmp, 0.0, 1.0)
    tmp = lut_gamma_encode(tmp, gamma_type)

    size_index = float(lut_size - 1)
    b, r, g = float(tmp[2]), float(tmp[0]), float(tmp[1])
    tex_x_base = (b * size_index + 0.5) / lut_size
    tex_y = (r * size_index + 0.5) / lut_size
    z = g * size_index

    width = lut_size * lut_size
    height = lut_size
    z_floor = int(np.floor(z))
    zf = (z_floor + tex_x_base) / lut_size
    zc = (min(z_floor + 1, int(size_index)) + tex_x_base) / lut_size

    col1 = _sample_bilinear(tex, width, height, zf, tex_y)
    col2 = _sample_bilinear(tex, width, height, zc, tex_y)
    fract_z = z - z_floor
    mapped = col1 * (1.0 - fract_z) + col2 * fract_z

    mapped = lut_gamma_decode(mapped, gamma_type)
    mapped = lut_primaries_decode(mapped, primaries_type)
    s = float(np.clip(strength, 0.0, 1.0))
    return orig * (1.0 - s) + mapped * s


def apply_polarr_rgb_lut_prophoto(
    prophoto_hwc: np.ndarray,
    merged_lut_rgb: np.ndarray,
    strength: float = 1.0,
    *,
    rgb_gamma: int = 1,
    rgb_primaries: int = 0,
) -> np.ndarray:
    """Apply user RGB LUT on ProPhoto-linear H×W×3 (matches GPU adjustments.frag path)."""
    hw = prophoto_hwc.reshape(-1, 3).astype(np.float32)
    tex = np.asarray(merged_lut_rgb, dtype=np.float32).reshape(-1)
    expected = LUT_SIZE * LUT_SIZE * LUT_SIZE * 3
    if tex.size != expected:
        raise ValueError(f"merged LUT must be {expected} floats, got {tex.size}")

    out = np.empty_like(hw)
    for i in range(hw.shape[0]):
        out[i] = _apply_look_table_rgb_pixel(
            hw[i], tex, LUT_SIZE, strength, rgb_gamma, rgb_primaries
        )
    return np.clip(out.reshape(prophoto_hwc.shape), 0.0, 1.0)


def apply_polarr_color_match_probe_style(
    srgb_hwc: np.ndarray,
    merged_lut_rgb: np.ndarray,
    strength: float = 1.0,
    *,
    encoding: str = "srgb_srgb",
) -> np.ndarray:
    """
    Comfy IMAGE (sRGB 0–1) → ProPhoto → Polarr RGB LUT → sRGB.

    Matches Polarr Next Probe LUT branch for bitmap/JPEG when develop stack
    is dominated by the user LUT (no extra sliders).
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return np.asarray(srgb_hwc, dtype=np.float32).copy()
    if strength >= 1.0 - 1e-6:
        pro = srgb_to_prophoto(srgb_hwc)
        gamma, primaries = ENCODING_PRESETS.get(encoding, ENCODING_PRESETS["srgb_srgb"])
        pro = apply_polarr_rgb_lut_prophoto(
            pro, merged_lut_rgb, 1.0, rgb_gamma=gamma, rgb_primaries=primaries
        )
        return prophoto_to_srgb(pro)

    pro = srgb_to_prophoto(srgb_hwc)
    gamma, primaries = ENCODING_PRESETS.get(encoding, ENCODING_PRESETS["srgb_srgb"])
    pro = apply_polarr_rgb_lut_prophoto(
        pro, merged_lut_rgb, strength, rgb_gamma=gamma, rgb_primaries=primaries
    )
    return prophoto_to_srgb(pro)
