"""Torch color-space helpers matching polarr_color_space.py."""

from __future__ import annotations

import torch

from ..polarr_color_space import (
    GAMMA_SRGB,
    MAT_LUT_DEC_SRGB,
    MAT_LUT_ENC_SRGB,
    MAT_PROPHOTO_TO_SRGB,
    MAT_SRGB_TO_PROPHOTO,
    MAT_SRGB_TO_XYZ,
    MAT_XYZ_TO_PROPHOTO,
)


def _const(device: torch.device, arr) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.float32, device=device)


def _const64(device: torch.device, arr) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.float64, device=device)


def srgb_to_linear(t: torch.Tensor) -> torch.Tensor:
    t = torch.clamp(t, 0.0, 1.0)
    return torch.where(t <= 0.04045, t / 12.92, ((t + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(t: torch.Tensor) -> torch.Tensor:
    t = torch.clamp(t, 0.0, None)
    return torch.clamp(
        torch.where(t <= 0.0031308, t * 12.92, 1.055 * torch.pow(t, 1.0 / 2.4) - 0.055),
        0.0,
        1.0,
    )


def srgb_to_prophoto(t: torch.Tensor) -> torch.Tensor:
    shape = t.shape
    flat = t.reshape(-1, 3)
    lin = srgb_to_linear(flat).double()
    m = _const64(flat.device, MAT_SRGB_TO_PROPHOTO)
    return (lin @ m.T).float().reshape(shape)


def prophoto_to_srgb(t: torch.Tensor) -> torch.Tensor:
    shape = t.shape
    flat = t.reshape(-1, 3).double()
    m = _const64(flat.device, MAT_PROPHOTO_TO_SRGB)
    xyz = (flat @ m.T).float()
    return linear_to_srgb(xyz).reshape(shape)


def lut_primaries_encode(t: torch.Tensor, primaries: int) -> torch.Tensor:
    if primaries != 0:
        raise NotImplementedError("GPU LUT path supports srgb primaries only")
    shape = t.shape
    flat = t.reshape(-1, 3).double()
    m = _const64(flat.device, MAT_LUT_ENC_SRGB)
    return (flat @ m.T).float().reshape(shape)


def lut_primaries_decode(t: torch.Tensor, primaries: int) -> torch.Tensor:
    if primaries != 0:
        raise NotImplementedError("GPU LUT path supports srgb primaries only")
    shape = t.shape
    flat = t.reshape(-1, 3).double()
    m = _const64(flat.device, MAT_LUT_DEC_SRGB)
    return (flat @ m.T).float().reshape(shape)


def lut_gamma_encode(t: torch.Tensor, gamma: int) -> torch.Tensor:
    if gamma == GAMMA_SRGB:
        return linear_to_srgb(t)
    return t


def lut_gamma_decode(t: torch.Tensor, gamma: int) -> torch.Tensor:
    if gamma == GAMMA_SRGB:
        return srgb_to_linear(t)
    return t


def srgb_to_prophoto_via_xyz(t: torch.Tensor) -> torch.Tensor:
    """bitmap.frag path: sRGB -> linear -> XYZ -> ProPhoto."""
    flat = t.reshape(-1, 3)
    lin = srgb_to_linear(flat).double()
    m1 = _const64(flat.device, MAT_SRGB_TO_XYZ)
    m2 = _const64(flat.device, MAT_XYZ_TO_PROPHOTO)
    xyz = (lin @ m1.T).float().double()
    return (xyz @ m2.T).float().reshape(t.shape)
