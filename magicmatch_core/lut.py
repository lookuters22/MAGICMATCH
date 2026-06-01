"""
1D/3D LUT apply + merged 25³ cube (17³ NN outputs → 25³ merged LUT).
"""

from __future__ import annotations

import numpy as np

MERGED_LUT_SIZE = 25
NN_LUT_SIZE = 17


def _clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def apply_1d_lut(image: np.ndarray, lut1d: np.ndarray) -> np.ndarray:
    """
    image: (N, 3) or flat length N*3 in [0,1]
    lut1d: length 3*M (channel-major: R slice, G slice, B slice)
    """
    flat = np.asarray(image, dtype=np.float32).reshape(-1)
    result = np.empty_like(flat)
    m = lut1d.size // 3
    scale = m - 1
    m_minus_2 = m - 2
    r_base, g_base, b_base = 0, m, m * 2

    for idx in range(0, flat.size, 3):
        for ch, base in enumerate((r_base, g_base, b_base)):
            scaled = float(flat[idx + ch]) * scale
            i_low = min(int(np.floor(scaled)), m_minus_2)
            frac = scaled - i_low
            low = float(lut1d[base + i_low])
            high = float(lut1d[base + i_low + 1])
            result[idx + ch] = np.clip(low * (1.0 - frac) + high * frac, 0.0, 1.0)
    return result


def apply_3d_lut(image: np.ndarray, lut3d: np.ndarray, lut_size: int) -> np.ndarray:
    """
    image: flat RGB
    lut3d: length 3*M^3, layout c * M^3 + x * M^2 + y * M + z
           where x=B, y=G, z=R.
    """
    flat = np.asarray(image, dtype=np.float32).reshape(-1)
    result = np.empty_like(flat)
    m = lut_size
    scale = m - 1
    m_minus_2 = m - 2
    m2 = m * m
    m3 = m * m * m
    r_base, g_base, b_base = 0, m3, m3 * 2

    for idx in range(0, flat.size, 3):
        x_scaled = float(flat[idx + 2]) * scale  # B -> X
        y_scaled = float(flat[idx + 1]) * scale  # G -> Y
        z_scaled = float(flat[idx + 0]) * scale  # R -> Z

        x0 = min(int(np.floor(x_scaled)), m_minus_2)
        y0 = min(int(np.floor(y_scaled)), m_minus_2)
        z0 = min(int(np.floor(z_scaled)), m_minus_2)
        x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
        fx, fy, fz = x_scaled - x0, y_scaled - y0, z_scaled - z0

        i000 = x0 * m2 + y0 * m + z0
        i001 = x0 * m2 + y0 * m + z1
        i010 = x0 * m2 + y1 * m + z0
        i011 = x0 * m2 + y1 * m + z1
        i100 = x1 * m2 + y0 * m + z0
        i101 = x1 * m2 + y0 * m + z1
        i110 = x1 * m2 + y1 * m + z0
        i111 = x1 * m2 + y1 * m + z1

        for ch, base in enumerate((r_base, g_base, b_base)):
            c000 = lut3d[base + i000]
            c001 = lut3d[base + i001]
            c010 = lut3d[base + i010]
            c011 = lut3d[base + i011]
            c100 = lut3d[base + i100]
            c101 = lut3d[base + i101]
            c110 = lut3d[base + i110]
            c111 = lut3d[base + i111]

            c00 = c000 * (1 - fx) + c100 * fx
            c01 = c001 * (1 - fx) + c101 * fx
            c10 = c010 * (1 - fx) + c110 * fx
            c11 = c011 * (1 - fx) + c111 * fx
            c0 = c00 * (1 - fy) + c10 * fy
            c1 = c01 * (1 - fy) + c11 * fy
            result[idx + ch] = np.clip(c0 * (1 - fz) + c1 * fz, 0.0, 1.0)

    return result


def transform_image(
    image: np.ndarray,
    lut1d: np.ndarray,
    lut3d: np.ndarray,
) -> np.ndarray:
    """1D LUT then 3D LUT (colorStrength=1 path in original)."""
    intermediate = apply_1d_lut(image, lut1d)
    m = lut1d.size // 3
    return apply_3d_lut(intermediate, lut3d, m)


def get_merged_lut(
    lut1d: np.ndarray,
    lut3d: np.ndarray,
    merged_size: int = MERGED_LUT_SIZE,
) -> np.ndarray:
    """
    Build 25³ merged LUT (RGBc interleaved: index (r,g,b)*3+c).
    Builds RGBc interleaved 25³ merged LUT from 1D + 3D NN outputs.
    """
    lut1d = np.asarray(lut1d, dtype=np.float32).reshape(-1)
    lut3d = np.asarray(lut3d, dtype=np.float32).reshape(-1)

    hald = np.empty(merged_size * merged_size * merged_size * 3, dtype=np.float32)
    interval = 1.0 / (merged_size - 1)
    for r in range(merged_size):
        for g in range(merged_size):
            for b in range(merged_size):
                index = (r * merged_size + g) * merged_size + b
                hald[index * 3] = r * interval
                hald[index * 3 + 1] = g * interval
                hald[index * 3 + 2] = b * interval

    return transform_image(hald, lut1d, lut3d)


def reshape_merged_lut_for_apply(merged_lut3d: np.ndarray, size: int = MERGED_LUT_SIZE) -> np.ndarray:
    """RGBc merged cube → cBGR flat layout for apply_3d_lut (transformImageWithMergedLut)."""
    merged = np.asarray(merged_lut3d, dtype=np.float32).reshape(-1)
    r = np.arange(size, dtype=np.int32)
    g = np.arange(size, dtype=np.int32)
    b = np.arange(size, dtype=np.int32)
    c = np.arange(3, dtype=np.int32)
    rr, gg, bb, cc = np.meshgrid(r, g, b, c, indexing="ij")
    src = (rr * size * size + gg * size + bb) * 3 + cc
    dst = cc * size**3 + bb * size**2 + gg * size + rr
    reshaped = np.empty_like(merged)
    reshaped[dst.ravel()] = merged[src.ravel()]
    return reshaped


def _lut_cube_from_flat(lut_flat: np.ndarray, lut_size: int) -> np.ndarray:
    """cBGR flat → (3, B, G, R) float32."""
    return np.asarray(lut_flat, dtype=np.float32).reshape(3, lut_size, lut_size, lut_size)


def apply_3d_lut_vectorized(image_hw3: np.ndarray, lut_flat: np.ndarray, lut_size: int) -> np.ndarray:
    """Vectorized trilinear 3D LUT (same indexing as apply_3d_lut)."""
    h, w, _ = image_hw3.shape
    lut = _lut_cube_from_flat(lut_flat, lut_size)
    scale = float(lut_size - 1)
    rgb = np.asarray(image_hw3, dtype=np.float32).reshape(-1, 3)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    x = np.clip(b * scale, 0.0, lut_size - 1 - 1e-5)
    y = np.clip(g * scale, 0.0, lut_size - 1 - 1e-5)
    z = np.clip(r * scale, 0.0, lut_size - 1 - 1e-5)

    x0 = np.minimum(np.floor(x).astype(np.int32), lut_size - 2)
    y0 = np.minimum(np.floor(y).astype(np.int32), lut_size - 2)
    z0 = np.minimum(np.floor(z).astype(np.int32), lut_size - 2)
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
    fx, fy, fz = (x - x0), (y - y0), (z - z0)

    out = np.empty_like(rgb)
    for c in range(3):
        c000 = lut[c, x0, y0, z0]
        c001 = lut[c, x0, y0, z1]
        c010 = lut[c, x0, y1, z0]
        c011 = lut[c, x0, y1, z1]
        c100 = lut[c, x1, y0, z0]
        c101 = lut[c, x1, y0, z1]
        c110 = lut[c, x1, y1, z0]
        c111 = lut[c, x1, y1, z1]

        c00 = c000 * (1.0 - fx) + c100 * fx
        c01 = c001 * (1.0 - fx) + c101 * fx
        c10 = c010 * (1.0 - fx) + c110 * fx
        c11 = c011 * (1.0 - fx) + c111 * fx
        c0 = c00 * (1.0 - fy) + c10 * fy
        c1 = c01 * (1.0 - fy) + c11 * fy
        out[:, c] = np.clip(c0 * (1.0 - fz) + c1 * fz, 0.0, 1.0)

    return out.reshape(h, w, 3)


def _rgb_to_hue(r: float, g: float, b: float) -> float:
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx == mn:
        return 0.0
    d = mx - mn
    if mx == r:
        h = (g - b) / d + (6.0 if g < b else 0.0)
    elif mx == g:
        h = (b - r) / d + 2.0
    else:
        h = (r - g) / d + 4.0
    return (h / 6.0) % 1.0


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    h = float(np.clip(h, 0.0, 1.0))
    s = float(np.clip(s, 0.0, 1.0))
    v = float(np.clip(v, 0.0, 1.0))
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return r, g, b


def _rgb_to_hsv(r: float, g: float, b: float) -> tuple[float, float, float]:
    mx = max(r, g, b)
    mn = min(r, g, b)
    d = mx - mn
    if mx == 0.0:
        return 0.0, 0.0, 0.0
    s = d / mx
    if d == 0.0:
        return 0.0, 0.0, mx
    if mx == r:
        h = (g - b) / d + (6.0 if g < b else 0.0)
    elif mx == g:
        h = (b - r) / d + 2.0
    else:
        h = (r - g) / d + 4.0
    return (h / 6.0) % 1.0, s, mx


def _set_hue(r: float, g: float, b: float, hue: float) -> tuple[float, float, float]:
    h, s, v = _rgb_to_hsv(r, g, b)
    return _hsv_to_rgb(hue, s, v)


def _luma_rec709(r: float, g: float, b: float) -> float:
    return 0.299 * r + 0.587 * g + 0.114 * b


def apply_color_light_strength(
    image: np.ndarray,
    result: np.ndarray,
    color_strength: float,
    light_strength: float = 1.0,
    original_input: np.ndarray | None = None,
) -> np.ndarray:
    """
    Color + luminance strength blend after LUT (lightStrength=1, no originalInput).
    """
    src = np.asarray(image, dtype=np.float32).reshape(-1)
    res = np.asarray(result, dtype=np.float32).reshape(-1)
    out = np.empty_like(src)
    inv_color = 1.0 - color_strength
    inv_light = 1.0 - light_strength

    for idx in range(0, src.size, 3):
        ir, ig, ib = float(src[idx]), float(src[idx + 1]), float(src[idx + 2])
        rr, rg, rb = float(res[idx]), float(res[idx + 1]), float(res[idx + 2])

        mr = ir * inv_color + rr * color_strength
        mg = ig * inv_color + rg * color_strength
        mb = ib * inv_color + rb * color_strength

        result_luma = _luma_rec709(rr, rg, rb)
        target_luma = result_luma
        if original_input is not None and light_strength != 1.0:
            orig = np.asarray(original_input, dtype=np.float32).reshape(-1)
            or_, og, ob = float(orig[idx]), float(orig[idx + 1]), float(orig[idx + 2])
            original_luma = _luma_rec709(or_, og, ob)
            target_luma = result_luma * light_strength + original_luma * inv_light

        mixed_luma = _luma_rec709(mr, mg, mb)
        delta = target_luma - mixed_luma
        cr, cg, cb = mr + delta, mg + delta, mb + delta
        result_hue = _rgb_to_hue(mr, mg, mb)
        cr, cg, cb = _set_hue(cr, cg, cb, result_hue)
        out[idx] = np.clip(cr, 0.0, 1.0)
        out[idx + 1] = np.clip(cg, 0.0, 1.0)
        out[idx + 2] = np.clip(cb, 0.0, 1.0)

    return out


def transform_image_with_merged_lut(image: np.ndarray, merged_lut3d: np.ndarray) -> np.ndarray:
    """Port of transformImageWithMergedLut() — apply 25³ merged cube to H×W×3 RGB."""
    hw3 = np.asarray(image, dtype=np.float32)
    if hw3.ndim == 1:
        side = int(round(hw3.size ** (1.0 / 3.0)))
        hw3 = hw3.reshape(side, side, 3) if side * side * 3 == hw3.size else hw3.reshape(-1, 3)[:, None, :]
    if hw3.ndim == 2:
        hw3 = hw3.reshape(-1, 1, 3)
    reshaped = reshape_merged_lut_for_apply(merged_lut3d, MERGED_LUT_SIZE)
    return apply_3d_lut_vectorized(hw3, reshaped, MERGED_LUT_SIZE)


def apply_color_light_strength_vectorized(
    image_hw3: np.ndarray,
    result_hw3: np.ndarray,
    color_strength: float,
) -> np.ndarray:
    """Vectorized applyColorLightStrength (lightStrength=1, no originalInput)."""
    src = np.asarray(image_hw3, dtype=np.float32).reshape(-1, 3)
    res = np.asarray(result_hw3, dtype=np.float32).reshape(-1, 3)
    inv = 1.0 - color_strength
    mixed = src * inv + res * color_strength

    result_luma = 0.299 * res[:, 0] + 0.587 * res[:, 1] + 0.114 * res[:, 2]
    mixed_luma = 0.299 * mixed[:, 0] + 0.587 * mixed[:, 1] + 0.114 * mixed[:, 2]
    delta = (result_luma - mixed_luma)[:, np.newaxis]
    corrected = mixed + delta

    mx = np.max(corrected, axis=1)
    mn = np.min(corrected, axis=1)
    d = mx - mn
    hue = np.zeros(corrected.shape[0], dtype=np.float32)
    nz = d > 1e-8
    r, g, b = corrected[:, 0], corrected[:, 1], corrected[:, 2]
    rm = (mx == r) & nz
    gm = (mx == g) & nz & ~rm
    bm = nz & ~rm & ~gm
    hue[rm] = ((g[rm] - b[rm]) / d[rm] + np.where(g[rm] < b[rm], 6.0, 0.0)) / 6.0
    hue[gm] = ((b[gm] - r[gm]) / d[gm] + 2.0) / 6.0
    hue[bm] = ((r[bm] - g[bm]) / d[bm] + 4.0) / 6.0
    hue = hue % 1.0

    v = mx
    s = np.where(mx > 1e-8, d / mx, 0.0)
    hi = np.floor(hue * 6.0).astype(np.int32) % 6
    f = hue * 6.0 - np.floor(hue * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    out = np.empty_like(corrected)
    for i in range(6):
        m = hi == i
        if not np.any(m):
            continue
        if i == 0:
            out[m] = np.stack([v[m], t[m], p[m]], axis=1)
        elif i == 1:
            out[m] = np.stack([q[m], v[m], p[m]], axis=1)
        elif i == 2:
            out[m] = np.stack([p[m], v[m], t[m]], axis=1)
        elif i == 3:
            out[m] = np.stack([p[m], q[m], v[m]], axis=1)
        elif i == 4:
            out[m] = np.stack([t[m], p[m], v[m]], axis=1)
        else:
            out[m] = np.stack([v[m], p[m], q[m]], axis=1)
    return np.clip(out, 0.0, 1.0).reshape(image_hw3.shape)


def apply_merged_lut_preview(
    rgb: np.ndarray,
    merged_lut3d: np.ndarray,
    color_strength: float = 1.0,
) -> np.ndarray:
    """
    Apply merged LUT to H×W×3 float RGB [0,1], then blend with colorStrength.

    - 0% → unchanged source
    - 100% → full merged LUT
    - between → color/light strength blend
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    if color_strength <= 0.0:
        return rgb.copy()
    lutted = transform_image_with_merged_lut(rgb, merged_lut3d)
    if color_strength >= 1.0 - 1e-6:
        return lutted
    return apply_color_light_strength_vectorized(rgb, lutted, color_strength)


def nn_outputs_to_flat(lut1d: np.ndarray, lut3d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """TF.js outputs [1,3,17] and [1,3,17,17,17] → flat arrays for get_merged_lut."""
    lut1d_flat = np.asarray(lut1d, dtype=np.float32).reshape(-1)
    lut3d_arr = np.asarray(lut3d, dtype=np.float32)
    if lut3d_arr.ndim == 5:
        # NCHW-style from graph: [1, 3, 17, 17, 17] → channel-major flat like TF dataSync
        lut3d_flat = lut3d_arr.reshape(-1)
    else:
        lut3d_flat = lut3d_arr.reshape(-1)
    return lut1d_flat, lut3d_flat
