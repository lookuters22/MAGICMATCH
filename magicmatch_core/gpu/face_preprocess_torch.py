"""GPU preprocess for face detect / skin parse ONNX feeds."""

from __future__ import annotations

import torch

from ..probe_parity.face_detection import FACE_PARSE_SIZE, MEAN_BGR
from .resize_torch import resize_bilinear_torch


def gamma_correct_torch(rgb: torch.Tensor, gamma: float) -> torch.Tensor:
    if abs(gamma - 1.0) < 1e-6:
        return rgb
    return torch.pow(torch.clamp(rgb, 0.0, 1.0), gamma)


@torch.inference_mode()
def preprocess_detect_torch(
    hwc: torch.Tensor,
    size: tuple[int, int],
    adjusted_gamma: float,
) -> torch.Tensor:
    """N×H×W×3 BGR mean-subtracted feed for face detect ONNX."""
    h, w = size
    resized = resize_bilinear_torch(hwc, h, w)
    rgb = resized * 255.0
    rgb = gamma_correct_torch(rgb / 255.0, adjusted_gamma) * 255.0
    bgr = rgb[..., [2, 1, 0]]
    mean = torch.tensor(MEAN_BGR, device=hwc.device, dtype=bgr.dtype)
    return (bgr - mean).unsqueeze(0)


@torch.inference_mode()
def preprocess_skin_torch(hwc: torch.Tensor, adjusted_gamma: float) -> torch.Tensor:
    """NCHW skin parse feed matching createTensorForSkin."""
    resized = resize_bilinear_torch(hwc, FACE_PARSE_SIZE, FACE_PARSE_SIZE)
    rgb255 = gamma_correct_torch(resized, adjusted_gamma) * 255.0
    chw = rgb255.permute(2, 0, 1)
    return ((chw / 127.5) - 1.0).unsqueeze(0)
