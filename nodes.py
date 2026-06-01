"""
ComfyUI nodes for MAGICMATCH — neural color match via ONNX.
"""

from __future__ import annotations

import numpy as np
import torch

from .magicmatch_core.apply import (
    RENDER_NUMPY,
    RENDER_POLARR_PROBE,
    RENDER_PROBE_EXPORT,
    apply_merged_lut_output,
)
from .magicmatch_core.inference import build_merged_lut_with_base
from .magicmatch_core.live_cache import pack_live_cache
from .magicmatch_core.probe_parity.profile_stage import (
    PROFILE_STAGE_OPTIONS,
    normalize_profile_stage,
)

LUT_ENCODING_OPTIONS = [
    "srgb_srgb",
    "srgb_none",
    "linear_srgb",
    "linear_none",
    "none_none",
    "srgb_adobe",
    "linear_adobe",
]

# Include legacy aliases so pre-update workflows pass ComfyUI combo validation.
RENDER_MODE_COMBO = [
    RENDER_PROBE_EXPORT,
    RENDER_POLARR_PROBE,
    RENDER_NUMPY,
    "polarr",
    "numpy",
    "probe_export",
]

_RENDER_MODE_ALIASES = {
    "polarr": RENDER_POLARR_PROBE,
    "numpy": RENDER_NUMPY,
    "probe_export": RENDER_PROBE_EXPORT,
}


def _normalize_lut_encoding(value) -> str:
    if isinstance(value, str) and value in LUT_ENCODING_OPTIONS:
        return value
    return "srgb_srgb"


def _normalize_render_mode(value) -> str:
    if isinstance(value, str):
        if value in _RENDER_MODE_ALIASES:
            return _RENDER_MODE_ALIASES[value]
        if value in (RENDER_PROBE_EXPORT, RENDER_POLARR_PROBE, RENDER_NUMPY):
            return value
    return RENDER_PROBE_EXPORT


def _render_options() -> dict:
    """Required combos — optional COMBO inputs break validation on saved workflows."""
    return {
        "lut_encoding": (
            LUT_ENCODING_OPTIONS,
            {"default": "srgb_srgb"},
        ),
        "render_mode": (
            RENDER_MODE_COMBO,
            {"default": RENDER_PROBE_EXPORT},
        ),
        "profile_stage": (
            PROFILE_STAGE_OPTIONS,
            {"default": "current_profile_stages"},
        ),
    }


class MagicMatchLUT:
    """Cached 25³ merged LUT + scene base adjustments between Build and Preview nodes."""

    __slots__ = ("merged_lut", "base_adjustments", "feed_tensor")

    def __init__(
        self,
        merged_lut: np.ndarray,
        base_adjustments: dict | None = None,
        feed_tensor: torch.Tensor | None = None,
    ) -> None:
        self.merged_lut = np.asarray(merged_lut, dtype=np.float32).reshape(-1)
        self.base_adjustments = dict(base_adjustments) if base_adjustments else None
        self.feed_tensor = feed_tensor


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
    DESCRIPTION = "Build merged LUT (ONNX, probe-parity net feed: auto-light + develop@1600→256)."

    @classmethod
    def IS_CHANGED(cls, source, reference):
        import hashlib

        if source is None or reference is None:
            return float("nan")
        s = hashlib.sha1(source.detach().cpu().numpy().tobytes()).hexdigest()[:16]
        r = hashlib.sha1(reference.detach().cpu().numpy().tobytes()).hexdigest()[:16]
        return (s, r)

    def build(self, source: torch.Tensor, reference: torch.Tensor) -> tuple[MagicMatchLUT]:
        src = _image_batch_to_hwc(source)
        ref = _image_batch_to_hwc(reference)
        merged, base = build_merged_lut_with_base(src, ref)
        return (MagicMatchLUT(merged, base),)


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
                **_render_options(),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "preview"
    CATEGORY = "MAGICMATCH"
    DESCRIPTION = (
        "Full-res output. probe_export = in-Comfy develop stack + LUT; "
        "polarr_probe = LUT-only slice; numpy_legacy = old CPU cube."
    )

    @classmethod
    def IS_CHANGED(
        cls,
        source,
        lut,
        strength,
        lut_encoding="srgb_srgb",
        render_mode=RENDER_PROBE_EXPORT,
        profile_stage="current_profile_stages",
    ):
        import hashlib

        if source is None or lut is None:
            return float("nan")
        src = source.detach().cpu().numpy()
        src_key = hashlib.sha1(src.tobytes()).hexdigest()[:16]
        lut_key = hashlib.sha1(lut.merged_lut.tobytes()).hexdigest()[:16]
        return (src_key, lut_key, float(strength), lut_encoding, render_mode, profile_stage)

    def preview(
        self,
        source: torch.Tensor,
        lut: MagicMatchLUT,
        strength: float,
        lut_encoding: str = "srgb_srgb",
        render_mode: str = RENDER_PROBE_EXPORT,
        profile_stage: str = "current_profile_stages",
    ) -> dict:
        src = _image_batch_to_hwc(source)
        lut_encoding = _normalize_lut_encoding(lut_encoding)
        render_mode = _normalize_render_mode(render_mode)
        profile_stage = normalize_profile_stage(profile_stage)
        out = apply_merged_lut_output(
            src,
            lut.merged_lut,
            strength,
            render_mode=render_mode,
            lut_encoding=lut_encoding,
            profile_stage=profile_stage,
            base_adjustments=lut.base_adjustments,
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
                **_render_options(),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "match"
    CATEGORY = "MAGICMATCH"
    DESCRIPTION = "One-shot color match at full resolution (Polarr probe LUT apply by default)."

    @classmethod
    def IS_CHANGED(
        cls,
        source,
        reference,
        strength,
        lut_encoding="srgb_srgb",
        render_mode=RENDER_PROBE_EXPORT,
        profile_stage="current_profile_stages",
    ):
        import hashlib

        if source is None or reference is None:
            return float("nan")
        s = hashlib.sha1(source.detach().cpu().numpy().tobytes()).hexdigest()[:16]
        r = hashlib.sha1(reference.detach().cpu().numpy().tobytes()).hexdigest()[:16]
        return (s, r, float(strength), lut_encoding, render_mode, profile_stage)

    def match(
        self,
        source: torch.Tensor,
        reference: torch.Tensor,
        strength: float,
        lut_encoding: str = "srgb_srgb",
        render_mode: str = RENDER_PROBE_EXPORT,
        profile_stage: str = "current_profile_stages",
    ) -> tuple[torch.Tensor]:
        src = _image_batch_to_hwc(source)
        ref = _image_batch_to_hwc(reference)
        lut_encoding = _normalize_lut_encoding(lut_encoding)
        render_mode = _normalize_render_mode(render_mode)
        profile_stage = normalize_profile_stage(profile_stage)
        merged, base = build_merged_lut_with_base(src, ref)
        out = apply_merged_lut_output(
            src,
            merged,
            strength,
            render_mode=render_mode,
            lut_encoding=lut_encoding,
            profile_stage=profile_stage,
            base_adjustments=base,
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
