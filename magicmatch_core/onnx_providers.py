"""ONNX Runtime execution provider helpers for the optional CUDA inference path."""

from __future__ import annotations

import logging
from pathlib import Path

try:
    import onnxruntime as ort
except ImportError as e:
    raise ImportError("MAGICMATCH: pip install onnxruntime or onnxruntime-gpu") from e

logger = logging.getLogger("magicmatch.cuda")

CUDA_PROVIDER = "CUDAExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"


def get_available_providers() -> list[str]:
    return list(ort.get_available_providers())


def cuda_available() -> bool:
    return CUDA_PROVIDER in get_available_providers()


def provider_priority(*, prefer_cuda: bool = True) -> list[str | tuple[str, dict]]:
    if prefer_cuda and cuda_available():
        return [CUDA_PROVIDER, CPU_PROVIDER]
    return [CPU_PROVIDER]


def create_onnx_session(
    onnx_path: str | Path,
    *,
    prefer_cuda: bool = True,
    session_options: ort.SessionOptions | None = None,
) -> ort.InferenceSession:
    path = Path(onnx_path)
    if not path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {path}")
    providers = provider_priority(prefer_cuda=prefer_cuda)
    opts = session_options or ort.SessionOptions()
    sess = ort.InferenceSession(str(path), opts, providers=providers)
    active = session_active_provider(sess)
    if prefer_cuda and cuda_available():
        if active == CUDA_PROVIDER:
            logger.info("ONNX %s → CUDAExecutionProvider", path.name)
        else:
            logger.warning(
                "ONNX %s → requested CUDA but active provider is %s",
                path.name,
                active,
            )
    else:
        logger.info("ONNX %s → %s (CUDA unavailable or disabled)", path.name, active)
    return sess


def session_active_provider(sess: ort.InferenceSession) -> str:
    providers = sess.get_providers()
    return providers[0] if providers else "unknown"
