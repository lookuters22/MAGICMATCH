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


def _merged_lut_to_texture(merged_lut_rgb: np.ndarray) -> np.ndarray:
    """Flat RGB merged LUT → (hue=25, sat²=625, 3) — createUserLutTexture layout."""
    tex = np.asarray(merged_lut_rgb, dtype=np.float32).reshape(-1)
    expected = LUT_SIZE * LUT_SIZE * LUT_SIZE * 3
    if tex.size != expected:
        raise ValueError(f"merged LUT must be {expected} floats, got {tex.size}")
    return tex.reshape(LUT_SIZE, LUT_SIZE * LUT_SIZE, 3)


def _sample_bilinear_hw3(tex: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorized bilinear sample on H×W×3 texture; u,v are (N,) in [0, 1]."""
    height, width = tex.shape[:2]
    x = np.clip(u, 0.0, 1.0) * (width - 1)
    y = np.clip(v, 0.0, 1.0) * (height - 1)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    fx = (x - x0)[:, np.newaxis]
    fy = (y - y0)[:, np.newaxis]

    c00 = tex[y0, x0]
    c01 = tex[y0, x1]
    c10 = tex[y1, x0]
    c11 = tex[y1, x1]
    c0 = c00 * (1.0 - fx) + c01 * fx
    c1 = c10 * (1.0 - fx) + c11 * fx
    return c0 * (1.0 - fy) + c1 * fy


def apply_polarr_rgb_lut_prophoto(
    prophoto_hwc: np.ndarray,
    merged_lut_rgb: np.ndarray,
    strength: float = 1.0,
    *,
    rgb_gamma: int = 1,
    rgb_primaries: int = 0,
) -> np.ndarray:
    """Apply user RGB LUT on ProPhoto-linear H×W×3 (matches GPU adjustments.frag path)."""
    shape = prophoto_hwc.shape
    orig = np.asarray(prophoto_hwc, dtype=np.float32).reshape(-1, 3)
    tex = _merged_lut_to_texture(merged_lut_rgb)

    tmp = lut_primaries_encode(prophoto_hwc, rgb_primaries)
    tmp = np.clip(tmp, 0.0, 1.0)
    tmp = lut_gamma_encode(tmp, rgb_gamma).reshape(-1, 3)

    size_index = float(LUT_SIZE - 1)
    r, g, b = tmp[:, 0], tmp[:, 1], tmp[:, 2]
    tex_x_base = (b * size_index + 0.5) / LUT_SIZE
    tex_y = (r * size_index + 0.5) / LUT_SIZE
    z = g * size_index
    z_floor = np.floor(z).astype(np.int32)
    z_next = np.minimum(z_floor + 1, int(size_index))
    zf = (z_floor.astype(np.float32) + tex_x_base) / LUT_SIZE
    zc = (z_next.astype(np.float32) + tex_x_base) / LUT_SIZE

    col1 = _sample_bilinear_hw3(tex, zf, tex_y)
    col2 = _sample_bilinear_hw3(tex, zc, tex_y)
    fract_z = (z - z_floor)[:, np.newaxis]
    mapped = col1 * (1.0 - fract_z) + col2 * fract_z

    mapped = lut_gamma_decode(mapped.reshape(shape), rgb_gamma).reshape(-1, 3)
    mapped = lut_primaries_decode(mapped.reshape(shape), rgb_primaries).reshape(-1, 3)
    s = float(np.clip(strength, 0.0, 1.0))
    out = orig * (1.0 - s) + mapped * s
    return np.clip(out.reshape(shape), 0.0, 1.0)


def apply_polarr_color_match_probe_style(
    srgb_hwc: np.ndarray,
    merged_lut_rgb: np.ndarray,
    strength: float = 1.0,
    *,
    encoding: str = "srgb_srgb",
    base_adjustments: dict | None = None,
    profile_stage: str = "current_profile_stages",
) -> np.ndarray:
    """
    Comfy IMAGE (sRGB 0–1) → develop + Polarr RGB LUT → sRGB when base_adjustments
    are provided (probe color-match export). Without base_adjustments, applies LUT
    on ProPhoto-linear sRGB only (legacy slice).
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return np.asarray(srgb_hwc, dtype=np.float32).copy()
    if base_adjustments is not None:
        from .probe_parity.pipeline import apply_probe_export

        return apply_probe_export(
            srgb_hwc,
            merged_lut_rgb,
            strength,
            base_adjustments=base_adjustments,
            profile_stage=profile_stage,
            lut_encoding=encoding,
        )
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
