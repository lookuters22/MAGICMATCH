"""GPU develop stack — port of probe_parity/develop.py render_srgb_develop."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..polarr_color_space import ENCODING_PRESETS
from ..probe_parity.wb import DEFAULT_AS_SHOT_TEMP, DEFAULT_AS_SHOT_TINT, build_wb_matrix
from .color_torch import linear_to_srgb, prophoto_to_srgb, srgb_to_linear, srgb_to_prophoto
from .device import get_torch_device, hwc_numpy_to_torch, hwc_torch_to_numpy
from .lut_torch import apply_polarr_rgb_lut_prophoto_torch

PROPHOTO_LUMA = (0.242655, 0.755158, 0.002187)
T_A, T_B, T_C, T_D, T_E = 1.2, 0.0, 0.96, 0.22, 0.02


def _luma(rgb: torch.Tensor) -> torch.Tensor:
    w = rgb.new_tensor(PROPHOTO_LUMA)
    return (rgb * w).sum(dim=-1)


def _smoothstep(edge0, edge1, x: torch.Tensor) -> torch.Tensor:
    t = torch.clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = torch.maximum(torch.maximum(r, g), b)
    mn = torch.minimum(torch.minimum(r, g), b)
    d = mx - mn
    h = torch.zeros_like(mx)
    safe_d = torch.where(d > 1e-8, d, torch.ones_like(d))
    h = torch.where((mx == r) & (d > 1e-8), ((g - b) / safe_d + torch.where(g < b, 6.0, 0.0)) / 6.0, h)
    h = torch.where((mx == g) & (d > 1e-8), ((b - r) / safe_d + 2.0) / 6.0, h)
    h = torch.where((mx == b) & (d > 1e-8), ((r - g) / safe_d + 4.0) / 6.0, h)
    h = h % 1.0
    s = torch.where(mx <= 0, torch.zeros_like(mx), d / torch.maximum(mx, torch.full_like(mx, 1e-8)))
    return torch.stack([h, s, mx], dim=-1)


def _hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (torch.floor(h * 6.0).to(torch.int64) % 6).to(torch.int64)
    f = h * 6.0 - torch.floor(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    out = torch.zeros_like(hsv)
    for idx, (c0, c1, c2) in enumerate([(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]):
        m = i == idx
        out[..., 0] = torch.where(m, c0, out[..., 0])
        out[..., 1] = torch.where(m, c1, out[..., 1])
        out[..., 2] = torch.where(m, c2, out[..., 2])
    return out


def _set_hue(rgb: torch.Tensor, hue: torch.Tensor) -> torch.Tensor:
    hsv = _rgb_to_hsv(rgb)
    hsv[..., 0] = hue
    return _hsv_to_rgb(hsv)


def _rgb_to_hue(rgb: torch.Tensor) -> torch.Tensor:
    return _rgb_to_hsv(rgb)[..., 0]


def _tonemap_inv(rgb: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(rgb)
    for ch in range(3):
        x = rgb[..., ch]
        a = x * T_C - T_A
        b_coef = x * T_D - T_B
        c = x * T_E
        disc = torch.clamp(b_coef * b_coef - 4.0 * a * c, min=0.0)
        out[..., ch] = (-b_coef - torch.sqrt(disc)) / (2.0 * a + 1e-12)
    return out


def _tonemap(rgb: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(rgb, 0.0, 32.0)
    return (x * (T_A * x + T_B)) / (x * (T_C * x + T_D) + T_E)


def _apply_wb(rgb: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    return torch.clamp(rgb @ matrix.T, min=0.0)


def _apply_shadows_highlights(
    rgb: torch.Tensor, shadows: float, highlights: float, luma_map: torch.Tensor
) -> torch.Tensor:
    color_luma = _luma(rgb)
    mixed = torch.clamp(color_luma + luma_map, min=0.0) * 0.5
    luma = _smoothstep(0.0, 1.6, torch.sqrt(mixed))
    specular = torch.clamp(1.0 - luma * luma, min=0.0)
    hi_mask = _smoothstep(0.05, 1.0, luma_map) * 1.2
    if highlights > 0.0:
        hi_mask = hi_mask * torch.clamp(1.0 - color_luma, min=0.0) * 1.1
    sh_mask = 1.0 - _smoothstep(0.0, 0.5, luma_map)
    exposure_amount = shadows * sh_mask + highlights * hi_mask * specular
    return rgb * (2.0 ** (exposure_amount * 2.0)).unsqueeze(-1)


def _apply_saturation(rgb: torch.Tensor, saturation: float) -> torch.Tensor:
    luma = _luma(rgb).unsqueeze(-1)
    mix_amount = min(-saturation, 1.0)
    inv = 1.0 - mix_amount
    return torch.clamp(rgb * inv + luma * mix_amount, min=0.0)


def _contract_blacks_whites(rgb: torch.Tensor, whites: float, blacks: float) -> torch.Tensor:
    blacks = max(blacks, 0.0)
    whites = min(whites, 0.0)
    luma = _luma(rgb)
    x = torch.clamp(torch.sqrt(luma), 0.0, 1.0)
    whites_mask = _smoothstep(0.05, 1.0, x)
    specular = torch.clamp(1.0 - luma * luma * luma * luma, min=0.0)
    whites_mask = whites_mask * specular
    blacks_mask = _smoothstep(0.8, -0.4, x)
    mult = (2.0 ** (whites * whites_mask + blacks * blacks_mask)).unsqueeze(-1)
    return rgb * mult


def _falloff(x: torch.Tensor, m: float) -> torch.Tensor:
    xm = x * m
    xm = 1.0 / ((xm + 1.0) ** 2)
    mm = 1.0 / ((m + 1.0) ** 2)
    return (xm - mm) / (1.0 - mm)


def _expand_whites_blacks(rgb: torch.Tensor, whites: float, blacks: float) -> torch.Tensor:
    blacks = min(blacks, 0.0)
    whites = max(whites, 0.0)
    x = torch.clamp(_luma(rgb), 0.0, 1.0)
    whites_mask = _falloff(x, -0.7)
    whites_mask_low = _smoothstep(0.0, 1.0, torch.sqrt(x))
    blacks_mask = _falloff(1.0 - x, -0.9)
    mask = whites_mask * blacks_mask
    blacks_v = blacks * 0.33 * mask
    whites_v = 1.0 + whites * 0.33 * mask * whites_mask_low
    out = (rgb * (1.0 - blacks_v.unsqueeze(-1)) + blacks_v.unsqueeze(-1)) * whites_v.unsqueeze(-1)
    return torch.clamp(out, 0.0, 1.0)


def _contrast_curve(contrast: float, width: int, device: torch.device) -> torch.Tensor:
    if contrast < 0:
        pts = [(0, 0), (8, 21), (16, 34), (32, 54), (64, 88), (96, 115), (128, 135), (160, 152), (192, 174), (224, 206), (240, 228), (255, 255)]
    else:
        pts = [(0, 0), (32, 8), (64, 35), (96, 71), (128, 117), (160, 171), (192, 212), (224, 241), (255, 255)]
    intensity = abs(contrast)
    xs = torch.tensor([p[0] / 255.0 for p in pts], dtype=torch.float64, device=device)
    ys = torch.tensor(
        [p[1] / 255.0 + intensity * (p[1] / 255.0 - p[0] / 255.0) * (1.0 - p[0] / 255.0) for p in pts],
        dtype=torch.float64,
        device=device,
    )
    xs = torch.clamp(xs, 0, 1)
    ys = torch.clamp(ys, 0, 1)
    samples = torch.linspace(0, 1, width, device=device, dtype=torch.float32)
    return torch.from_numpy(
        __import__("numpy").interp(samples.cpu().numpy(), xs.cpu().numpy(), ys.cpu().numpy()).astype("float32")
    ).to(device)


def _apply_curve(rgb: torch.Tensor, curve: torch.Tensor) -> torch.Tensor:
    rgb = torch.clamp(rgb, 0.0, 1.0)
    size = curve.shape[0]
    out = torch.empty_like(rgb)
    for ch in range(3):
        values = rgb[..., ch]
        idx = torch.clamp(torch.floor(values * (size - 1)).to(torch.int64), 0, size - 2)
        p0x = idx.float() / (size - 1)
        p1x = (idx + 1).float() / (size - 1)
        t = (values - p0x) / torch.clamp(p1x - p0x, min=1e-12)
        out[..., ch] = torch.clamp(curve[idx] + t * (curve[idx + 1] - curve[idx]), 0.0, 1.0)
    return out


@torch.inference_mode()
def render_srgb_develop_torch(
    srgb_hwc: torch.Tensor,
    adjustments: dict,
    *,
    merged_lut=None,
    lut_strength: float = 1.0,
    lut_encoding: str = "srgb_srgb",
    input_tone_map_inversed: bool = False,
    as_shot_temp: float = DEFAULT_AS_SHOT_TEMP,
    as_shot_tint: float = DEFAULT_AS_SHOT_TINT,
) -> torch.Tensor:
    """Develop + optional LUT on GPU; input/output H×W×3 float [0,1] on device."""
    shape = srgb_hwc.shape
    device = srgb_hwc.device
    if input_tone_map_inversed:
        rgb = torch.clamp(srgb_hwc, 0.0, 1.0).reshape(-1, 3)
    else:
        rgb = srgb_to_prophoto(srgb_hwc).reshape(-1, 3)
        hue0 = _rgb_to_hue(rgb)
        rgb = _tonemap_inv(rgb)
        rgb = _set_hue(rgb, hue0)
        rgb = torch.clamp(rgb, min=0.0)

    gamma = linear_to_srgb(rgb)
    luma_map = _luma(gamma)

    wb_np = build_wb_matrix(
        adjustments.get("temperature", as_shot_temp),
        adjustments.get("tint", as_shot_tint),
        as_shot_temp=as_shot_temp,
        as_shot_tint=as_shot_tint,
    )
    wb = torch.tensor(wb_np, dtype=torch.float32, device=device)
    rgb = _apply_wb(rgb, wb)

    exposure_factor = float(2.0 ** adjustments.get("exposure", 0.0))
    shadows = float(adjustments.get("shadows", 0.0))
    highlights = float(adjustments.get("highlights", 0.0))
    whites = float(adjustments.get("whites", 0.0))
    blacks = float(adjustments.get("blacks", 0.0))
    saturation = float(adjustments.get("saturation", 0.0))
    contrast = float(adjustments.get("contrast", 0.0))

    if exposure_factor != 1.0:
        rgb = rgb * exposure_factor
    if shadows != 0.0 or highlights != 0.0:
        rgb = _apply_shadows_highlights(rgb, shadows, highlights, luma_map)
    if whites < 0.0 or blacks > 0.0:
        rgb = _contract_blacks_whites(rgb, whites, blacks)

    hue = _rgb_to_hue(rgb)
    rgb = torch.clamp(rgb, 0.0, 1.0)
    rgb = _set_hue(rgb, hue)
    if saturation != 0.0:
        rgb = _apply_saturation(rgb, saturation)

    hue = _rgb_to_hue(rgb)
    rgb = _tonemap(rgb)
    rgb = _set_hue(rgb, hue)
    rgb = linear_to_srgb(rgb)
    hue = _rgb_to_hue(rgb)
    if whites > 0.0 or blacks < 0.0:
        rgb = _expand_whites_blacks(rgb, whites, blacks)
    if contrast != 0.0:
        rgb = _apply_curve(rgb, _contrast_curve(contrast, 256, device))
    rgb = _set_hue(rgb, hue)
    rgb = srgb_to_linear(rgb)

    if merged_lut is not None and lut_strength > 0.0:
        gamma_id, primaries_id = ENCODING_PRESETS.get(lut_encoding, ENCODING_PRESETS["srgb_srgb"])
        rgb = apply_polarr_rgb_lut_prophoto_torch(
            rgb.reshape(shape),
            merged_lut,
            lut_strength,
            rgb_gamma=gamma_id,
            rgb_primaries=primaries_id,
        ).reshape(-1, 3)

    return prophoto_to_srgb(rgb.reshape(shape))


def render_srgb_develop_numpy(
    srgb_hwc,
    adjustments: dict,
    *,
    device: torch.device | None = None,
    **kwargs,
):
    """NumPy in/out wrapper for pipeline integration."""
    device = device or get_torch_device()
    t = hwc_numpy_to_torch(srgb_hwc, device)
    out = render_srgb_develop_torch(t, adjustments, **kwargs)
    return hwc_torch_to_numpy(out)
