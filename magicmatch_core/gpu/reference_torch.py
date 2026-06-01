"""GPU detection buffer chain — port of probe_parity/reference.py hot paths."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..probe_parity.reference import fit_to_size
from .color_torch import linear_to_srgb, srgb_to_prophoto_via_xyz
from .develop_torch import _rgb_to_hue, _set_hue, _tonemap, _tonemap_inv
from .device import get_torch_device, hwc_torch_to_numpy

DETECTION_LARGE_EDGE = 2000
DETECTION_SMALL_EDGE = 300


def _to_bchw(t: torch.Tensor) -> torch.Tensor:
    return t.permute(2, 0, 1).unsqueeze(0)


def _to_hwc(t: torch.Tensor) -> torch.Tensor:
    return t.squeeze(0).permute(1, 2, 0)


def _build_mipmap_chain_bchw(img: torch.Tensor) -> list[torch.Tensor]:
    mips = [img]
    while mips[-1].shape[-2] > 1 or mips[-1].shape[-1] > 1:
        h, w = mips[-1].shape[-2:]
        if h == 1 and w == 1:
            break
        pad_h = h % 2
        pad_w = w % 2
        cur = mips[-1]
        if pad_h or pad_w:
            cur = F.pad(cur, (0, pad_w, 0, pad_h), mode="replicate")
        mips.append(F.avg_pool2d(cur, kernel_size=2, stride=2))
    return mips


def _bilinear_at_bchw(img: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    """Sample B×1×H×W or B×3×H×W; xs/ys shape (out_h, out_w) in source pixel coords."""
    _, _, in_h, in_w = img.shape
    gx = 2.0 * xs / max(in_w - 1, 1) - 1.0
    gy = 2.0 * ys / max(in_h - 1, 1) - 1.0
    grid = torch.stack((gx, gy), dim=-1).unsqueeze(0)
    return F.grid_sample(img, grid, mode="bilinear", align_corners=True)


def bilinear_sample_grid_torch(hwc: torch.Tensor, out_w: int, out_h: int) -> torch.Tensor:
    in_h, in_w, _ = hwc.shape
    device = hwc.device
    xs = (torch.arange(out_w, device=device, dtype=torch.float32) + 0.5) / out_w * in_w - 0.5
    ys = (torch.arange(out_h, device=device, dtype=torch.float32) + 0.5) / out_h * in_h - 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return _to_hwc(_bilinear_at_bchw(_to_bchw(hwc), xx, yy))


def mipmap_linear_sample_grid_torch(hwc: torch.Tensor, out_w: int, out_h: int) -> torch.Tensor:
    in_h, in_w, _ = hwc.shape
    if out_w >= in_w and out_h >= in_h:
        return bilinear_sample_grid_torch(hwc, out_w, out_h)

    mips_bchw = _build_mipmap_chain_bchw(_to_bchw(hwc))
    device = hwc.device
    xs = (torch.arange(out_w, device=device, dtype=torch.float32) + 0.5) / out_w * in_w - 0.5
    ys = (torch.arange(out_h, device=device, dtype=torch.float32) + 0.5) / out_h * in_h - 0.5
    xx, yy = torch.meshgrid(xs, ys, indexing="xy")

    level = float(math.log2(max(in_w / out_w, in_h / out_h)))
    l0 = int(math.floor(level))
    l1 = min(l0 + 1, len(mips_bchw) - 1)
    frac = level - l0
    scale0 = 2.0**l0
    scale1 = 2.0**l1
    s0 = _to_hwc(_bilinear_at_bchw(mips_bchw[l0], xx / scale0, yy / scale0))
    s1 = _to_hwc(_bilinear_at_bchw(mips_bchw[l1], xx / scale1, yy / scale1))
    return torch.clamp(s0 * (1.0 - frac) + s1 * frac, 0.0, 1.0)


@torch.inference_mode()
def bitmap_shader_import_torch(hwc: torch.Tensor, scale: int = 2) -> torch.Tensor:
    hwc = torch.clamp(hwc, 0.0, 1.0)
    h, w, _ = hwc.shape
    out_h, out_w = max(1, h // scale), max(1, w // scale)
    src_y = torch.arange(out_h, device=hwc.device, dtype=torch.int64) * scale
    src_x = torch.arange(out_w, device=hwc.device, dtype=torch.int64) * scale
    yy, xx = torch.meshgrid(src_y, src_x, indexing="ij")
    rgb = hwc[yy, xx]
    pro = srgb_to_prophoto_via_xyz(rgb.reshape(-1, 3))
    hue = _rgb_to_hue(pro)
    pro = _tonemap_inv(pro)
    pro = _set_hue(pro, hue)
    return linear_to_srgb(torch.clamp(pro, min=0.0)).reshape(out_h, out_w, 3)


@torch.inference_mode()
def transform_export_srgb_torch(tone_map_inversed_hwc: torch.Tensor, *, linear_gain: float = 1.0) -> torch.Tensor:
    from ..polarr_color_space import MAT_PROPHOTO_TO_SRGB
    from .color_torch import srgb_to_linear as stl

    hwc = torch.clamp(tone_map_inversed_hwc, 0.0, 1.0)
    rgb = stl(hwc.reshape(-1, 3).double()).float()
    rgb = _tonemap(rgb).double()
    if linear_gain != 1.0:
        rgb = rgb * linear_gain
    m = torch.tensor(MAT_PROPHOTO_TO_SRGB, device=hwc.device, dtype=torch.float64)
    out = linear_to_srgb(torch.clamp((rgb @ m.T).float(), 0.0, 1.0))
    return torch.clamp(out.reshape(hwc.shape), 0.0, 1.0)


@torch.inference_mode()
def render_detection_export_torch(
    worker_feed_hwc: torch.Tensor,
    edge: int,
    *,
    half_res_hwc: torch.Tensor | None = None,
):
    feed = torch.clamp(worker_feed_hwc, 0.0, 1.0)
    logical_h, logical_w, _ = feed.shape
    half = half_res_hwc if half_res_hwc is not None else bitmap_shader_import_torch(feed, scale=2)
    nw, nh = fit_to_size(logical_w, logical_h, (edge, edge))
    upsampled = mipmap_linear_sample_grid_torch(half, nw, nh)
    gain = 1.008 if edge >= 2000 else 1.0
    return transform_export_srgb_torch(upsampled, linear_gain=gain)


def render_detection_inputs_torch(
    hwc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU renderDetectionInputs; returns small/large H×W×3 on device."""
    half = bitmap_shader_import_torch(hwc, scale=2)
    small = render_detection_export_torch(hwc, DETECTION_SMALL_EDGE, half_res_hwc=half)
    large = render_detection_export_torch(hwc, DETECTION_LARGE_EDGE, half_res_hwc=half)
    return small, large


def render_detection_inputs_numpy(hwc, device=None):
    from .device import hwc_numpy_to_torch

    device = device or get_torch_device()
    t = hwc_numpy_to_torch(hwc, device)
    small, large = render_detection_inputs_torch(t)
    return hwc_torch_to_numpy(small), hwc_torch_to_numpy(large)
