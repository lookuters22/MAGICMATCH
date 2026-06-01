"""MAGICMATCH core — ONNX inference and merged LUT apply."""

from .inference import build_merged_lut, run_inference_from_images
from .lut import apply_merged_lut_preview, get_merged_lut, nn_outputs_to_flat

__all__ = [
    "build_merged_lut",
    "run_inference_from_images",
    "get_merged_lut",
    "nn_outputs_to_flat",
    "apply_merged_lut_preview",
]
