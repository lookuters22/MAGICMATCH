"""GPU Polarr 25³ RGB LUT apply."""

from __future__ import annotations

import torch

from .color_torch import lut_gamma_decode, lut_gamma_encode, lut_primaries_decode, lut_primaries_encode

LUT_SIZE = 25


def _merged_lut_to_texture(merged_lut_rgb, device: torch.device) -> torch.Tensor:
    import numpy as np

    tex = np.asarray(merged_lut_rgb, dtype=np.float32).reshape(-1)
    expected = LUT_SIZE * LUT_SIZE * LUT_SIZE * 3
    if tex.size != expected:
        raise ValueError(f"merged LUT must be {expected} floats, got {tex.size}")
    return torch.from_numpy(tex.reshape(LUT_SIZE, LUT_SIZE * LUT_SIZE, 3)).to(device)


def _sample_bilinear_hw3(tex: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    height, width = tex.shape[:2]
    x = torch.clamp(u, 0.0, 1.0) * (width - 1)
    y = torch.clamp(v, 0.0, 1.0) * (height - 1)
    x0 = torch.floor(x).to(torch.int64)
    y0 = torch.floor(y).to(torch.int64)
    x1 = torch.minimum(x0 + 1, torch.tensor(width - 1, device=tex.device))
    y1 = torch.minimum(y0 + 1, torch.tensor(height - 1, device=tex.device))
    fx = (x - x0.float()).unsqueeze(-1)
    fy = (y - y0.float()).unsqueeze(-1)
    c00 = tex[y0, x0]
    c01 = tex[y0, x1]
    c10 = tex[y1, x0]
    c11 = tex[y1, x1]
    c0 = c00 * (1.0 - fx) + c01 * fx
    c1 = c10 * (1.0 - fx) + c11 * fx
    return c0 * (1.0 - fy) + c1 * fy


@torch.inference_mode()
def apply_polarr_rgb_lut_prophoto_torch(
    prophoto_hwc: torch.Tensor,
    merged_lut_rgb,
    strength: float = 1.0,
    *,
    rgb_gamma: int = 1,
    rgb_primaries: int = 0,
) -> torch.Tensor:
    device = prophoto_hwc.device
    shape = prophoto_hwc.shape
    orig = prophoto_hwc.reshape(-1, 3)
    tex = _merged_lut_to_texture(merged_lut_rgb, device)

    tmp = lut_primaries_encode(prophoto_hwc, rgb_primaries)
    tmp = torch.clamp(tmp, 0.0, 1.0)
    tmp = lut_gamma_encode(tmp, rgb_gamma).reshape(-1, 3)

    size_index = float(LUT_SIZE - 1)
    r, g, b = tmp[:, 0], tmp[:, 1], tmp[:, 2]
    tex_x_base = (b * size_index + 0.5) / LUT_SIZE
    tex_y = (r * size_index + 0.5) / LUT_SIZE
    z = g * size_index
    z_floor = torch.floor(z).to(torch.int64)
    z_next = torch.minimum(z_floor + 1, torch.tensor(int(size_index), device=device))
    zf = (z_floor.float() + tex_x_base) / LUT_SIZE
    zc = (z_next.float() + tex_x_base) / LUT_SIZE

    col1 = _sample_bilinear_hw3(tex, zf, tex_y)
    col2 = _sample_bilinear_hw3(tex, zc, tex_y)
    fract_z = (z - z_floor.float()).unsqueeze(-1)
    mapped = col1 * (1.0 - fract_z) + col2 * fract_z

    mapped = lut_gamma_decode(mapped.reshape(shape), rgb_gamma).reshape(-1, 3)
    mapped = lut_primaries_decode(mapped.reshape(shape), rgb_primaries).reshape(-1, 3)
    s = float(torch.clamp(torch.tensor(strength), 0.0, 1.0))
    return torch.clamp(orig * (1.0 - s) + mapped * s, 0.0, 1.0).reshape(shape)
