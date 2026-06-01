"""
ComfyUI nodes — fast GPU pipeline (speed over strict parity).

CPU nodes in nodes.py are unchanged. Category: MAGICMATCH/GPU Fast (experimental).
"""

from __future__ import annotations

import torch

from .magicmatch_core.gpu.device import gpu_pipeline_available, hwc_numpy_to_torch
from .magicmatch_core.gpu.pipeline_fast import (
    apply_probe_export_fast_gpu,
    build_merged_lut_fast_gpu,
    color_match_one_shot_fast_gpu,
)
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


def _image_batch_to_feed_tensor(image: torch.Tensor) -> torch.Tensor:
    """Direct GPU upload from Comfy IMAGE tensor — no JPEG q98 normalize."""
    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError(f"MAGICMATCH: expected IMAGE [B,H,W,3], got {tuple(image.shape)}")
    if image.shape[0] != 1:
        raise ValueError(f"MAGICMATCH: batch size {image.shape[0]} not supported — use batch size 1.")
    t = image[0].detach().float().clamp(0.0, 1.0)
    if gpu_pipeline_available() and t.is_cuda:
        return t
    return hwc_numpy_to_torch(_image_batch_to_hwc(image))


class MagicMatchBuildGPUFast:
    """Analyze source + reference — fast GPU scene extract + develop + CUDA ONNX."""

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
    CATEGORY = "MAGICMATCH/GPU Fast (experimental)"
    DESCRIPTION = (
        "Fast GPU build: no JPEG normalize, bilinear ref, GPU gray-world WB. "
        "Caches GPU feed tensor for Preview."
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
        src_np = _image_batch_to_hwc(source)
        ref_np = _image_batch_to_hwc(reference)
        feed_t = _image_batch_to_feed_tensor(source)
        ref_t = reference[0].detach().float().clamp(0.0, 1.0)
        if not (gpu_pipeline_available() and ref_t.is_cuda):
            ref_t = hwc_numpy_to_torch(ref_np)
        state = build_merged_lut_fast_gpu(
            src_np,
            ref_np,
            feed_tensor=feed_t,
            reference_tensor=ref_t,
        )
        return (MagicMatchLUT(state.merged_lut, state.base_adjustments, feed_tensor=state.feed_tensor),)


class MagicMatchGPUFast:
    """One-shot fast GPU color match — fused build + apply."""

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
    CATEGORY = "MAGICMATCH/GPU Fast (experimental)"
    DESCRIPTION = "Fast fused GPU pipeline: approximate scene extract + ONNX + develop/LUT apply."

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
        from .magicmatch_core.probe_parity.profile_stage import normalize_profile_stage

        src_np = _image_batch_to_hwc(source)
        ref_np = _image_batch_to_hwc(reference)
        lut_encoding = _normalize_lut_encoding(lut_encoding)
        render_mode = _normalize_render_mode(render_mode)
        profile_stage = normalize_profile_stage(profile_stage)
        src_t = source[0].detach().float().clamp(0.0, 1.0)
        ref_t = reference[0].detach().float().clamp(0.0, 1.0)
        if render_mode == RENDER_PROBE_EXPORT:
            out = color_match_one_shot_fast_gpu(
                src_np,
                ref_np,
                strength,
                profile_stage=profile_stage,
                lut_encoding=lut_encoding,
                source_tensor=src_t,
                reference_tensor=ref_t,
            )
        else:
            feed_t = _image_batch_to_feed_tensor(source)
            state = build_merged_lut_fast_gpu(
                src_np,
                ref_np,
                feed_tensor=feed_t,
                reference_tensor=ref_t,
            )
            out = apply_merged_lut_output(
                src_np,
                state.merged_lut,
                strength,
                render_mode=render_mode,
                lut_encoding=lut_encoding,
                profile_stage=profile_stage,
                base_adjustments=state.base_adjustments,
            )
        return (_hwc_to_image(out),)


class MagicMatchPreviewGPUFast(MagicMatchPreview):
    """Fast GPU probe_export apply; reuses cached GPU feed tensor from Build GPU Fast."""

    CATEGORY = "MAGICMATCH/GPU Fast (experimental)"
    DESCRIPTION = "Fast GPU develop + LUT apply using cached feed tensor from Build GPU Fast."

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

        src_np = _image_batch_to_hwc(source)
        lut_encoding = _normalize_lut_encoding(lut_encoding)
        render_mode = _normalize_render_mode(render_mode)
        profile_stage = normalize_profile_stage(profile_stage)
        feed_t = getattr(lut, "feed_tensor", None)
        if feed_t is None:
            feed_t = _image_batch_to_feed_tensor(source)
        if render_mode == RENDER_PROBE_EXPORT:
            out = apply_probe_export_fast_gpu(
                src_np,
                lut.merged_lut,
                strength,
                base_adjustments=lut.base_adjustments,
                profile_stage=profile_stage,
                lut_encoding=lut_encoding,
                feed_tensor=feed_t,
            )
        else:
            out = apply_merged_lut_output(
                src_np,
                lut.merged_lut,
                strength,
                render_mode=render_mode,
                lut_encoding=lut_encoding,
                profile_stage=profile_stage,
                base_adjustments=lut.base_adjustments,
            )
        cache = pack_live_cache(src_np, lut.merged_lut)
        return {
            "ui": {"magicmatch_live": [cache]},
            "result": (_hwc_to_image(out),),
        }


GPU_FAST_NODE_CLASS_MAPPINGS = {
    "MagicMatchBuildGPUFast": MagicMatchBuildGPUFast,
    "MagicMatchGPUFast": MagicMatchGPUFast,
    "MagicMatchPreviewGPUFast": MagicMatchPreviewGPUFast,
}

GPU_FAST_NODE_DISPLAY_NAME_MAPPINGS = {
    "MagicMatchBuildGPUFast": "MagicMatch Build LUT (GPU Fast)",
    "MagicMatchGPUFast": "MagicMatch one-shot (GPU Fast)",
    "MagicMatchPreviewGPUFast": "MagicMatch Preview (GPU Fast LUT)",
}
