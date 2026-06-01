"""
ComfyUI nodes for MAGICMATCH — neural color match via ONNX.
"""

from __future__ import annotations

import numpy as np
import torch

from .magicmatch_core.apply import (
    RENDER_NUMPY,
    RENDER_POLARR_PROBE,
    apply_merged_lut_output,
)
from .magicmatch_core.inference import build_merged_lut
from .magicmatch_core.live_cache import pack_live_cache

LUT_ENCODING_OPTIONS = [
    "srgb_srgb",
    "srgb_none",
    "linear_srgb",
    "linear_none",
    "none_none",
    "srgb_adobe",
    "linear_adobe",
]

RENDER_MODE_OPTIONS = [RENDER_POLARR_PROBE, RENDER_NUMPY]


class MagicMatchLUT:
    """Cached 25³ merged LUT between Build and Preview nodes."""

    __slots__ = ("merged_lut",)

    def __init__(self, merged_lut: np.ndarray) -> None:
        self.merged_lut = np.asarray(merged_lut, dtype=np.float32).reshape(-1)


def _image_batch_to_hwc(image: torch.Tensor) -> np.ndarray:
    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError(f"MAGICMATCH: expected IMAGE [B,H,W,3], got {tuple(image.shape)}")
    if image.shape[0] != 1:
        raise ValueError(
            f"MAGICMATCH: batch size {image.shape[0]} not supported — use batch size 1."
        )
    return np.clip(image[0].detach().cpu().numpy(), 0.0, 1.0).astype(np.float32)


def _hwc_to_image(hwc: np.ndarray) -> torch.Tensor:
    hwc = np.clip(np.asarray(hwc, dtype=np.float32), 0.0, 1.0)
    return torch.from_numpy(hwc[np.newaxis, ...])


def _render_options() -> dict:
    return {
        "lut_encoding": (
            LUT_ENCODING_OPTIONS,
            {"default": "srgb_srgb"},
        ),
        "render_mode": (
            RENDER_MODE_OPTIONS,
            {"default": RENDER_POLARR_PROBE},
        ),
    }


class MagicMatchBuild:
    """Analyze source + reference (ONNX). Wire output into MagicMatch Preview."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("IMAGE",),
                "reference": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("MAGICMATCH_LUT",)
    RETURN_NAMES = ("lut",)
    FUNCTION = "build"
    CATEGORY = "MAGICMATCH"
    DESCRIPTION = "Build merged LUT (ONNX, PIL 256×256 — Polarr Next Probe net path)."

    @classmethod
    def IS_CHANGED(cls, source, reference):
        import hashlib

        s = hashlib.sha1(source.detach().cpu().numpy().tobytes()).hexdigest()[:16]
        r = hashlib.sha1(reference.detach().cpu().numpy().tobytes()).hexdigest()[:16]
        return (s, r)

    def build(self, source: torch.Tensor, reference: torch.Tensor) -> tuple[MagicMatchLUT]:
        src = _image_batch_to_hwc(source)
        ref = _image_batch_to_hwc(reference)
        merged = build_merged_lut(src, ref)
        return (MagicMatchLUT(merged),)


class MagicMatchPreview:
    """
    Apply cached LUT at full resolution. Default render matches Polarr Next Probe
    (sRGB → ProPhoto, RGB LUT + sRGB/sRGB encoding, per-channel strength).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("IMAGE",),
                "lut": ("MAGICMATCH_LUT",),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                    },
                ),
            },
            "optional": _render_options(),
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "preview"
    CATEGORY = "MAGICMATCH"
    DESCRIPTION = (
        "Full-res output. polarr_probe = Next Probe LUT path; numpy_legacy = old CPU cube."
    )

    @classmethod
    def IS_CHANGED(cls, source, lut, strength, lut_encoding="srgb_srgb", render_mode=RENDER_POLARR_PROBE):
        import hashlib

        src = source.detach().cpu().numpy()
        src_key = hashlib.sha1(src.tobytes()).hexdigest()[:16]
        lut_key = hashlib.sha1(lut.merged_lut.tobytes()).hexdigest()[:16]
        return (src_key, lut_key, float(strength), lut_encoding, render_mode)

    def preview(
        self,
        source: torch.Tensor,
        lut: MagicMatchLUT,
        strength: float,
        lut_encoding: str = "srgb_srgb",
        render_mode: str = RENDER_POLARR_PROBE,
    ) -> dict:
        src = _image_batch_to_hwc(source)
        out = apply_merged_lut_output(
            src,
            lut.merged_lut,
            strength,
            render_mode=render_mode,
            lut_encoding=lut_encoding,
        )
        cache = pack_live_cache(src, lut.merged_lut)
        return {
            "ui": {"magicmatch_live": [cache]},
            "result": (_hwc_to_image(out),),
        }


class MagicMatch:
    """Source + reference + strength in one node (re-runs ONNX when strength changes)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("IMAGE",),
                "reference": ("IMAGE",),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                    },
                ),
            },
            "optional": _render_options(),
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "match"
    CATEGORY = "MAGICMATCH"
    DESCRIPTION = "One-shot color match at full resolution (Polarr probe LUT apply by default)."

    @classmethod
    def IS_CHANGED(cls, source, reference, strength, lut_encoding="srgb_srgb", render_mode=RENDER_POLARR_PROBE):
        import hashlib

        s = hashlib.sha1(source.detach().cpu().numpy().tobytes()).hexdigest()[:16]
        r = hashlib.sha1(reference.detach().cpu().numpy().tobytes()).hexdigest()[:16]
        return (s, r, float(strength), lut_encoding, render_mode)

    def match(
        self,
        source: torch.Tensor,
        reference: torch.Tensor,
        strength: float,
        lut_encoding: str = "srgb_srgb",
        render_mode: str = RENDER_POLARR_PROBE,
    ) -> tuple[torch.Tensor]:
        src = _image_batch_to_hwc(source)
        ref = _image_batch_to_hwc(reference)
        merged = build_merged_lut(src, ref)
        out = apply_merged_lut_output(
            src,
            merged,
            strength,
            render_mode=render_mode,
            lut_encoding=lut_encoding,
        )
        return (_hwc_to_image(out),)


NODE_CLASS_MAPPINGS = {
    "MagicMatchBuild": MagicMatchBuild,
    "MagicMatchPreview": MagicMatchPreview,
    "MagicMatch": MagicMatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MagicMatchBuild": "MagicMatch Build LUT",
    "MagicMatchPreview": "MagicMatch Preview (strength)",
    "MagicMatch": "MagicMatch (one-shot)",
}
