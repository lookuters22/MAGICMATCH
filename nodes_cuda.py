"""
Experimental ComfyUI nodes — CUDA ONNX inference path.

CPU nodes in nodes.py are unchanged. Use these for GPU testing on H100/Linux;
Preview/apply still uses the CPU NumPy develop stack.
"""

from __future__ import annotations

import torch

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
from .magicmatch_core.apply import apply_merged_lut_output


class MagicMatchBuildCUDA:
    """Analyze source + reference (CUDA ONNX when available). Wire into MagicMatch Preview."""

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
        "Build merged LUT via CUDA ONNX (color_match + face models). "
        "Falls back to CPU EP if CUDA unavailable. CPU Build node unchanged."
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
    """One-shot CUDA color match (re-runs ONNX when strength changes)."""

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
    DESCRIPTION = "One-shot color match with CUDA ONNX build; apply path is CPU NumPy."

    @classmethod
    def IS_CHANGED(
        cls,
        source,
        reference,
        strength,
        lut_encoding="srgb_srgb",
        render_mode="probe_export",
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
        render_mode: str = "probe_export",
        profile_stage: str = "current_profile_stages",
    ) -> tuple[torch.Tensor]:
        from .magicmatch_core.probe_parity.profile_stage import normalize_profile_stage

        src = _image_batch_to_hwc(source)
        ref = _image_batch_to_hwc(reference)
        lut_encoding = _normalize_lut_encoding(lut_encoding)
        render_mode = _normalize_render_mode(render_mode)
        profile_stage = normalize_profile_stage(profile_stage)
        merged, base = build_merged_lut_with_base_cuda(src, ref)
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
    """Same as MagicMatch Preview — apply is CPU NumPy; included for workflow clarity."""

    CATEGORY = "MAGICMATCH/CUDA (experimental)"
    DESCRIPTION = (
        "Full-res apply for LUTs built with MagicMatch Build CUDA. "
        "Identical to MagicMatch Preview (CPU develop stack)."
    )


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
