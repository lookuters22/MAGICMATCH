"""
MAGICMATCH — ComfyUI custom nodes for neural color match (ONNX).
"""

import os

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Experimental CUDA nodes (distinct class names; safe alongside CPU nodes).
# Set MAGICMATCH_CUDA_NODES=0 to hide them from ComfyUI.
if os.environ.get("MAGICMATCH_CUDA_NODES", "1") != "0":
    from .nodes_cuda import CUDA_NODE_CLASS_MAPPINGS, CUDA_NODE_DISPLAY_NAME_MAPPINGS

    NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **CUDA_NODE_CLASS_MAPPINGS}
    NODE_DISPLAY_NAME_MAPPINGS = {
        **NODE_DISPLAY_NAME_MAPPINGS,
        **CUDA_NODE_DISPLAY_NAME_MAPPINGS,
    }

# Full GPU pipeline nodes (separate category; set MAGICMATCH_GPU_FULL_NODES=0 to hide).
if os.environ.get("MAGICMATCH_GPU_FULL_NODES", "1") != "0":
    from .nodes_gpu_full import GPU_FULL_NODE_CLASS_MAPPINGS, GPU_FULL_NODE_DISPLAY_NAME_MAPPINGS

    NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **GPU_FULL_NODE_CLASS_MAPPINGS}
    NODE_DISPLAY_NAME_MAPPINGS = {
        **NODE_DISPLAY_NAME_MAPPINGS,
        **GPU_FULL_NODE_DISPLAY_NAME_MAPPINGS,
    }

WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "js")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
