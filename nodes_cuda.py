"""
Experimental ComfyUI nodes — CUDA ONNX build with CPU parity develop/apply.

Output should match MagicMatch (one-shot) under MAGICMATCH; only color_match.onnx
uses CUDA when available. Set MAGICMATCH_CUDA_GPU_APPLY=1 for legacy GPU apply (brighter).
"""

from __future__ import annotations

import os

import torch

from .magicmatch_core.gpu.pipeline_v1 import apply_probe_export_gpu_v1
from .magicmatch_core.inference_cuda import build_merged_lut_with_base_cuda
from .nodes import (
    MagicMatchLUT,
    MagicMatchPreview,
    _hwc_to_image,
    _image_batch_to_hwc,
    _normalize_lut_encoding,
    _normalize_render_mode,
    _render_options,
)
from .magicmatch_core.apply import RENDER_PROBE_EXPORT, apply_merged_lut_output


def _cuda_gpu_apply_enabled() -> bool:
    return os.environ.get("MAGICMATCH_CUDA_GPU_APPLY", "0") == "1"


class MagicMatchBuildCUDA:
    """Analyze source + reference (CPU parity build + CUDA ONNX)."""

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
    CATEGORY = "MAGICMATCH/CUDA (experimental)"
    DESCRIPTION = (
        "Build merged LUT: CPU parity scene extract + develop@1600, CUDA color_match ONNX."
    )

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
        merged, base = build_merged_lut_with_base_cuda(src, ref)
        return (MagicMatchLUT(merged, base),)


class MagicMatchCUDA:
    """One-shot color match — CPU parity apply by default; CUDA ONNX build."""

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
    CATEGORY = "MAGICMATCH/CUDA (experimental)"
    DESCRIPTION = (
        "Matches MagicMatch (one-shot) look; CUDA accelerates color_match.onnx only. "
        "MAGICMATCH_CUDA_GPU_APPLY=1 enables legacy GPU apply (may look brighter)."
    )

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
        return (s, r, float(strength), lut_encoding, render_mode, profile_stage, _cuda_gpu_apply_enabled())

    def match(
        self,
        source: torch.Tensor,
        reference: torch.Tensor,
        strength: float,
        lut_encoding: str = "srgb_srgb",
        render_mode: str = RENDER_PROBE_EXPORT,
        profile_stage: str = "current_profile_stages",
    ) -> tuple[torch.Tensor]:
        from .magicmatch_core.probe_parity.profile_stage import normalize_profile_stage

        src = _image_batch_to_hwc(source)
        ref = _image_batch_to_hwc(reference)
        lut_encoding = _normalize_lut_encoding(lut_encoding)
        render_mode = _normalize_render_mode(render_mode)
        profile_stage = normalize_profile_stage(profile_stage)
        merged, base = build_merged_lut_with_base_cuda(src, ref)
        if render_mode == RENDER_PROBE_EXPORT and _cuda_gpu_apply_enabled():
            out = apply_probe_export_gpu_v1(
                src,
                merged,
                strength,
                base_adjustments=base,
                profile_stage=profile_stage,
                lut_encoding=lut_encoding,
            )
        else:
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


class MagicMatchPreviewCUDA(MagicMatchPreview):
    """CPU parity apply by default; same LUT as Build CUDA."""

    CATEGORY = "MAGICMATCH/CUDA (experimental)"
    DESCRIPTION = "Full-res CPU parity apply for LUTs from Build CUDA."

    def preview(
        self,
        source: torch.Tensor,
        lut: MagicMatchLUT,
        strength: float,
        lut_encoding: str = "srgb_srgb",
        render_mode: str = RENDER_PROBE_EXPORT,
        profile_stage: str = "current_profile_stages",
    ) -> dict:
        from .magicmatch_core.live_cache import pack_live_cache
        from .magicmatch_core.probe_parity.profile_stage import normalize_profile_stage

        src = _image_batch_to_hwc(source)
        lut_encoding = _normalize_lut_encoding(lut_encoding)
        render_mode = _normalize_render_mode(render_mode)
        profile_stage = normalize_profile_stage(profile_stage)
        if render_mode == RENDER_PROBE_EXPORT and _cuda_gpu_apply_enabled():
            out = apply_probe_export_gpu_v1(
                src,
                lut.merged_lut,
                strength,
                base_adjustments=lut.base_adjustments,
                profile_stage=profile_stage,
                lut_encoding=lut_encoding,
            )
        else:
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


CUDA_NODE_CLASS_MAPPINGS = {
    "MagicMatchBuildCUDA": MagicMatchBuildCUDA,
    "MagicMatchCUDA": MagicMatchCUDA,
    "MagicMatchPreviewCUDA": MagicMatchPreviewCUDA,
}

CUDA_NODE_DISPLAY_NAME_MAPPINGS = {
    "MagicMatchBuildCUDA": "MagicMatch Build LUT (CUDA)",
    "MagicMatchCUDA": "MagicMatch one-shot (CUDA)",
    "MagicMatchPreviewCUDA": "MagicMatch Preview (CUDA LUT)",
}
