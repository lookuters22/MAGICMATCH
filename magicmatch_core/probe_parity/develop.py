"""
Vectorized Polarr develop stack for color-match export (renderer-cpu.ts renderPixel).

Bitmap/JPEG path: no profile tone curve, no color look unless forced.
"""

from __future__ import annotations

import numpy as np

from ..polarr_color_space import (
    ENCODING_PRESETS,
    linear_to_srgb,
    prophoto_to_srgb,
    srgb_to_linear,
    srgb_to_prophoto,
)
from ..polarr_lut_rgb import apply_polarr_rgb_lut_prophoto
from .adobe_assets import load_adobe_profile_look_table, profile_look_dims
from .wb import DEFAULT_AS_SHOT_TEMP, DEFAULT_AS_SHOT_TINT, build_wb_matrix

PROPHOTO_LUMA = np.array([0.242655, 0.755158, 0.002187], dtype=np.float32)
T_A, T_B, T_C, T_D, T_E = 1.2, 0.0, 0.96, 0.22, 0.02


def _smoothstep(edge0: np.ndarray | float, edge1: np.ndarray | float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _get_luma(rgb: np.ndarray) -> np.ndarray:
    return rgb @ PROPHOTO_LUMA


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    d = mx - mn
    h = np.zeros_like(mx)
    safe_d = np.where(d > 1e-8, d, 1.0)
    h = np.where(
        (mx == r) & (d > 1e-8),
        ((g - b) / safe_d + np.where(g < b, 6.0, 0.0)) / 6.0,
        h,
    )
    h = np.where(
        (mx == g) & (d > 1e-8),
        ((b - r) / safe_d + 2.0) / 6.0,
        h,
    )
    h = np.where(
        (mx == b) & (d > 1e-8),
        ((r - g) / safe_d + 4.0) / 6.0,
        h,
    )
    h = h % 1.0
    s = np.where(mx <= 0, 0.0, d / np.maximum(mx, 1e-8))
    return np.stack([h, s, mx], axis=-1)


def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = np.floor(h * 6.0).astype(np.int32) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    out = np.empty_like(hsv)
    for idx, (c0, c1, c2) in enumerate(
        [
            (v, t, p),
            (q, v, p),
            (p, v, t),
            (p, q, v),
            (t, p, v),
            (v, p, q),
        ]
    ):
        m = i == idx
        out[m, 0], out[m, 1], out[m, 2] = c0[m], c1[m], c2[m]
    return out


def _set_hue(rgb: np.ndarray, hue: np.ndarray) -> np.ndarray:
    hsv = _rgb_to_hsv(rgb)
    hsv[..., 0] = hue
    return _hsv_to_rgb(hsv)


def _rgb_to_hue(rgb: np.ndarray) -> np.ndarray:
    return _rgb_to_hsv(rgb)[..., 0]


def _tonemap_inv(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    out = np.empty_like(rgb)
    for ch, x in enumerate([r, g, b]):
        a = x * T_C - T_A
        b_coef = x * T_D - T_B
        c = x * T_E
        disc = np.maximum(b_coef * b_coef - 4.0 * a * c, 0.0)
        out[..., ch] = (-b_coef - np.sqrt(disc)) / (2.0 * a + 1e-12)
    return out


def _tonemap(rgb: np.ndarray) -> np.ndarray:
    x = np.clip(rgb, 0.0, 32.0)
    return (x * (T_A * x + T_B)) / (x * (T_C * x + T_D) + T_E)


def _linear_to_gamma(rgb: np.ndarray) -> np.ndarray:
    return linear_to_srgb(np.maximum(rgb, 0.0))


def _gamma_to_linear(rgb: np.ndarray) -> np.ndarray:
    return srgb_to_linear(np.maximum(rgb, 0.0))


def _apply_wb(rgb: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.maximum(rgb @ matrix.T, 0.0)


def _apply_exposure_factor(rgb: np.ndarray, factor: float) -> np.ndarray:
    return rgb * factor


def _apply_shadows_highlights(
    rgb: np.ndarray, shadows: float, highlights: float, luma_map: np.ndarray
) -> np.ndarray:
    color_luma = _get_luma(rgb)
    mixed = np.maximum(color_luma + luma_map, 0.0) * 0.5
    luma = _smoothstep(0.0, 1.6, np.sqrt(mixed))
    specular = np.maximum(0.0, 1.0 - luma * luma)
    hi_mask = _smoothstep(0.05, 1.0, luma_map) * 1.2
    if highlights > 0.0:
        hi_mask *= np.maximum(0.0, 1.0 - color_luma) * 1.1
    sh_mask = 1.0 - _smoothstep(0.0, 0.5, luma_map)
    exposure_amount = shadows * sh_mask + highlights * hi_mask * specular
    return rgb * np.power(2.0, exposure_amount * 2.0)[..., np.newaxis]


def _apply_saturation(rgb: np.ndarray, saturation: float) -> np.ndarray:
    luma = _get_luma(rgb)[..., np.newaxis]
    mix_amount = min(-saturation, 1.0)
    inv = 1.0 - mix_amount
    return np.maximum(rgb * inv + luma * mix_amount, 0.0)


def _contract_blacks_whites(rgb: np.ndarray, whites: float, blacks: float) -> np.ndarray:
    blacks = max(blacks, 0.0)
    whites = min(whites, 0.0)
    luma = _get_luma(rgb)
    x = np.clip(np.sqrt(luma), 0.0, 1.0)
    whites_mask = _smoothstep(0.05, 1.0, x)
    specular = np.maximum(0.0, 1.0 - luma * luma * luma * luma)
    whites_mask *= specular
    blacks_mask = _smoothstep(0.8, -0.4, x)
    mult = np.power(2.0, whites * whites_mask + blacks * blacks_mask)[..., np.newaxis]
    return rgb * mult


def _falloff(x: np.ndarray, m: float) -> np.ndarray:
    xm = x * m
    xm = 1.0 / ((xm + 1.0) ** 2)
    mm = 1.0 / ((m + 1.0) ** 2)
    return (xm - mm) / (1.0 - mm)


def _expand_whites_blacks(rgb: np.ndarray, whites: float, blacks: float) -> np.ndarray:
    blacks = min(blacks, 0.0)
    whites = max(whites, 0.0)
    x = np.clip(_get_luma(rgb), 0.0, 1.0)
    whites_mask = _falloff(x, -0.7)
    whites_mask_low = _smoothstep(0.0, 1.0, np.sqrt(x))
    blacks_mask = _falloff(1.0 - x, -0.9)
    mask = whites_mask * blacks_mask
    blacks = blacks * 0.33 * mask
    whites = 1.0 + whites * 0.33 * mask * whites_mask_low
    out = (rgb * (1.0 - blacks[..., np.newaxis]) + blacks[..., np.newaxis]) * whites[..., np.newaxis]
    return np.clip(out, 0.0, 1.0)


def _sample_bilinear_table(data: np.ndarray, width: int, height: int, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    x = np.clip(u, 0.0, 1.0) * (width - 1)
    y = np.clip(v, 0.0, 1.0) * (height - 1)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    fx = (x - x0)[..., np.newaxis]
    fy = (y - y0)[..., np.newaxis]
    flat = data.reshape(height, width, 3)

    def gather(yi, xi):
        return flat[yi, xi]

    c00, c01 = gather(y0, x0), gather(y0, x1)
    c10, c11 = gather(y1, x0), gather(y1, x1)
    c0 = c00 * (1.0 - fx) + c01 * fx
    c1 = c10 * (1.0 - fx) + c11 * fx
    return c0 * (1.0 - fy) + c1 * fy


def _apply_hsv_look_table(rgb: np.ndarray, dims: tuple[int, int, int], table: np.ndarray) -> np.ndarray:
    hue_divs, sat_divs, val_divs = dims
    inv_sat = 1.0 / sat_divs
    inv_val = 1.0 / val_divs
    hsv = _rgb_to_hsv(rgb)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    x = (s * (sat_divs - 1.0) + 0.5) * inv_sat
    y = (h * hue_divs + 0.5) / hue_divs
    z = v * (val_divs - 1.0)
    z_floor = np.floor(z)
    zf = (z_floor + y) * inv_val
    zc = (np.ceil(z) + y) * inv_val
    width = sat_divs
    height = hue_divs * val_divs
    tex = table.reshape(height, width, 3)
    col1 = _sample_bilinear_table(tex, width, height, x, zf)
    col2 = _sample_bilinear_table(tex, width, height, x, zc)
    fract_z = (z - z_floor)[..., np.newaxis]
    factors = col1 * (1.0 - fract_z) + col2 * fract_z
    hsv[..., 0] = (h + factors[..., 0]) % 1.0
    hsv[..., 1] = s * factors[..., 1]
    hsv[..., 2] = v * factors[..., 2]
    return _hsv_to_rgb(hsv)


def _get_contrast_curve(contrast: float, width: int = 256) -> np.ndarray:
    if contrast < 0:
        pts = [
            (0, 0),
            (8, 21),
            (16, 34),
            (32, 54),
            (64, 88),
            (96, 115),
            (128, 135),
            (160, 152),
            (192, 174),
            (224, 206),
            (240, 228),
            (255, 255),
        ]
    else:
        pts = [
            (0, 0),
            (32, 8),
            (64, 35),
            (96, 71),
            (128, 117),
            (160, 171),
            (192, 212),
            (224, 241),
            (255, 255),
        ]
    intensity = abs(contrast)
    xs = np.array([p[0] / 255.0 for p in pts], dtype=np.float64)
    ys = np.array(
        [p[1] / 255.0 + intensity * (p[1] / 255.0 - p[0] / 255.0) * (1.0 - p[0] / 255.0) for p in pts],
        dtype=np.float64,
    )
    xs = np.clip(xs, 0, 1)
    ys = np.clip(ys, 0, 1)
    samples = np.linspace(0, 1, width, dtype=np.float32)
    return np.interp(samples, xs, ys).astype(np.float32)


def _apply_curve(rgb: np.ndarray, curve: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb, 0.0, 1.0)
    size = len(curve)
    out = np.empty_like(rgb)
    for ch in range(3):
        values = rgb[..., ch]
        idx = np.clip(np.floor(values * (size - 1)).astype(np.int32), 0, size - 2)
        p0x = idx / (size - 1)
        p1x = (idx + 1) / (size - 1)
        t = (values - p0x) / np.maximum(p1x - p0x, 1e-12)
        out[..., ch] = np.clip(curve[idx] + t * (curve[idx + 1] - curve[idx]), 0.0, 1.0)
    return out


def render_srgb_develop(
    srgb_hwc: np.ndarray,
    adjustments: dict,
    *,
    merged_lut: np.ndarray | None = None,
    lut_strength: float = 1.0,
    lut_encoding: str = "srgb_srgb",
    disable_profile_look: bool = False,
    force_color_look: bool = False,
    as_shot_temp: float = DEFAULT_AS_SHOT_TEMP,
    as_shot_tint: float = DEFAULT_AS_SHOT_TINT,
) -> np.ndarray:
    """Full-res develop + optional user RGB LUT → sRGB (probe export path)."""
    shape = srgb_hwc.shape
    rgb = srgb_to_prophoto(srgb_hwc).reshape(-1, 3)
    hue0 = _rgb_to_hue(rgb)
    rgb = _tonemap_inv(rgb)
    rgb = _set_hue(rgb, hue0)
    rgb = np.maximum(rgb, 0.0)

    gamma = _linear_to_gamma(rgb)
    luma_map = _get_luma(gamma)

    temp = adjustments.get("temperature", as_shot_temp)
    tint = adjustments.get("tint", as_shot_tint)
    wb = build_wb_matrix(temp, tint, as_shot_temp=as_shot_temp, as_shot_tint=as_shot_tint)
    rgb = _apply_wb(rgb, wb)

    exposure_factor = float(2.0 ** adjustments.get("exposure", 0.0))
    shadows = float(adjustments.get("shadows", 0.0))
    highlights = float(adjustments.get("highlights", 0.0))
    whites = float(adjustments.get("whites", 0.0))
    blacks = float(adjustments.get("blacks", 0.0))
    saturation = float(adjustments.get("saturation", 0.0))
    contrast = float(adjustments.get("contrast", 0.0))

    if exposure_factor != 1.0:
        rgb = _apply_exposure_factor(rgb, exposure_factor)
    if shadows != 0.0 or highlights != 0.0:
        rgb = _apply_shadows_highlights(rgb, shadows, highlights, luma_map)
    if whites < 0.0 or blacks > 0.0:
        rgb = _contract_blacks_whites(rgb, whites, blacks)

    hue = _rgb_to_hue(rgb)
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = _set_hue(rgb, hue)
    if saturation != 0.0:
        rgb = _apply_saturation(rgb, saturation)

    profile_dims = profile_look_dims()
    profile_table = load_adobe_profile_look_table()
    if force_color_look:
        rgb = _apply_hsv_look_table(rgb, profile_dims, profile_table)
    if not disable_profile_look:
        rgb = _apply_hsv_look_table(rgb, profile_dims, profile_table)

    hue = _rgb_to_hue(rgb)
    rgb = _tonemap(rgb)
    rgb = _set_hue(rgb, hue)
    rgb = _linear_to_gamma(rgb)
    hue = _rgb_to_hue(_gamma_to_linear(rgb))
    if whites > 0.0 or blacks < 0.0:
        rgb = _expand_whites_blacks(rgb, whites, blacks)
    if contrast != 0.0:
        rgb = _apply_curve(rgb, _get_contrast_curve(contrast, 256))
    rgb = _gamma_to_linear(rgb)
    rgb = _set_hue(rgb, hue)

    if merged_lut is not None and lut_strength > 0.0:
        gamma_id, primaries_id = ENCODING_PRESETS.get(lut_encoding, ENCODING_PRESETS["srgb_srgb"])
        rgb = apply_polarr_rgb_lut_prophoto(
            rgb.reshape(shape),
            merged_lut,
            lut_strength,
            rgb_gamma=gamma_id,
            rgb_primaries=primaries_id,
        ).reshape(-1, 3)

    return prophoto_to_srgb(rgb.reshape(shape))
