#!/usr/bin/env python3
"""Benchmark CPU vs CUDA ONNX paths on a source/reference pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from magicmatch_core.inference import (  # noqa: E402
    build_merged_lut_with_base,
    get_session,
    run_inference_from_images,
)
from magicmatch_core.inference_cuda import (  # noqa: E402
    build_merged_lut_with_base_cuda,
    color_match_session_info,
    get_session_cuda,
    run_inference_from_images_cuda,
)
from magicmatch_core.onnx_providers import cuda_available, get_available_providers  # noqa: E402
from magicmatch_core.gpu.device import gpu_pipeline_available  # noqa: E402
from magicmatch_core.probe_parity.face_detection import detect_faces  # noqa: E402
from magicmatch_core.probe_parity.face_detection_cuda import (  # noqa: E402
    detect_faces_cuda,
    face_session_info,
)
from magicmatch_core.probe_parity.pipeline import (  # noqa: E402
    _render_for_net,
    build_merged_lut_probe_style,
)
from magicmatch_core.probe_parity.base_adjustments import estimate_base_adjustments  # noqa: E402
from magicmatch_core.probe_parity.base_adjustments_cuda import estimate_base_adjustments_cuda  # noqa: E402
from magicmatch_core.probe_parity.reference import prepare_net_reference, prepare_worker_bitmap_source  # noqa: E402
from magicmatch_core.probe_parity.scene_extract import render_detection_inputs  # noqa: E402
from magicmatch_core.probe_parity.luminance import get_adjusted_gamma, get_luminance_statistics  # noqa: E402


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def lut_hash(lut: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(lut, dtype=np.float32).tobytes()).hexdigest()[:16]


def _timed(label: str, fn, *, repeats: int = 1) -> tuple[object, float]:
    # Warm-up once
    result = fn()
    start = time.perf_counter()
    for _ in range(repeats):
        result = fn()
    elapsed = (time.perf_counter() - start) / max(repeats, 1)
    print(f"  {label}: {elapsed * 1000:.1f} ms")
    return result, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?", help="Source image (PNG/JPG)")
    parser.add_argument("reference", type=Path, nargs="?", help="Reference image (PNG/JPG)")
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Timed repeats per benchmark (after warm-up)",
    )
    parser.add_argument(
        "--profile-gpu",
        action="store_true",
        help="Print per-phase GPU pipeline timings and exit",
    )
    args = parser.parse_args()

    default_pair = (
        ROOT.parent / "polarrnext" / "standalone_probe" / "public" / "pair" / "source.png",
        ROOT.parent / "polarrnext" / "standalone_probe" / "public" / "pair" / "reference.jpg",
    )
    source_path = args.source or default_pair[0]
    reference_path = args.reference or default_pair[1]
    if not source_path.is_file() or not reference_path.is_file():
        print(f"Missing test pair: {source_path} / {reference_path}", file=sys.stderr)
        return 2

    src = load_rgb(source_path)
    ref = load_rgb(reference_path)

    if args.profile_gpu:
        from magicmatch_core.gpu.pipeline_full_gpu import profile_full_gpu_pipeline

        print(json.dumps(profile_full_gpu_pipeline(src, ref), indent=2, default=str))
        return 0

    report: dict[str, object] = {
        "source": str(source_path),
        "reference": str(reference_path),
        "ort_providers_available": get_available_providers(),
        "cuda_available": cuda_available(),
    }

    print("=== Provider availability ===")
    print(f"  ORT providers: {get_available_providers()}")
    print(f"  cuda_available(): {cuda_available()}")

    worker_src = prepare_worker_bitmap_source(src)
    base_cpu = estimate_base_adjustments(worker_src, worker_feed_prepared=True)
    net_source = _render_for_net(worker_src, base_cpu)
    net_ref = prepare_net_reference(ref)

    print("\n=== color_match.onnx (256 feed) ===")
    get_session()
    get_session_cuda()
    _, t_cpu_infer = _timed(
        "CPU run_inference_from_images",
        lambda: run_inference_from_images(net_source, net_ref),
        repeats=args.repeats,
    )
    _, t_cuda_infer = _timed(
        "CUDA run_inference_from_images_cuda",
        lambda: run_inference_from_images_cuda(net_source, net_ref),
        repeats=args.repeats,
    )
    report["color_match"] = {
        "cpu_ms": round(t_cpu_infer * 1000, 2),
        "cuda_ms": round(t_cuda_infer * 1000, 2),
        "cpu_session": get_session().get_providers(),
        "cuda_session": color_match_session_info(),
    }

    small, large = render_detection_inputs(worker_src, worker_feed_prepared=True)
    stats = get_luminance_statistics(small, large, [])
    gamma = get_adjusted_gamma(stats)

    print("\n=== face ONNX (detect on large) ===")
    _, t_cpu_face = _timed(
        "CPU detect_faces",
        lambda: detect_faces(large, adjusted_gamma=gamma),
        repeats=max(1, args.repeats - 1),
    )
    _, t_cuda_face = _timed(
        "CUDA detect_faces_cuda",
        lambda: detect_faces_cuda(large, adjusted_gamma=gamma),
        repeats=max(1, args.repeats - 1),
    )
    report["face_detect"] = {
        "cpu_ms": round(t_cpu_face * 1000, 2),
        "cuda_ms": round(t_cuda_face * 1000, 2),
        "cuda_sessions": face_session_info(),
    }

    print("\n=== full build (probe-style) ===")
    build_cpu, t_cpu_build = _timed(
        "CPU build_merged_lut_with_base",
        lambda: build_merged_lut_with_base(src, ref),
        repeats=args.repeats,
    )
    build_cuda, t_cuda_build = _timed(
        "CUDA build_merged_lut_with_base_cuda",
        lambda: build_merged_lut_with_base_cuda(src, ref),
        repeats=args.repeats,
    )
    if gpu_pipeline_available():
        from magicmatch_core.gpu.pipeline import build_merged_lut_probe_style_gpu

        _, t_gpu_build = _timed(
            "GPU full pipeline build_merged_lut_probe_style_gpu",
            lambda: build_merged_lut_probe_style_gpu(src, ref),
            repeats=args.repeats,
        )
        report["gpu_full_build_ms"] = round(t_gpu_build * 1000, 2)
    merged_cpu, _ = build_cpu
    merged_cuda, _ = build_cuda
    hash_cpu = lut_hash(merged_cpu)
    hash_cuda = lut_hash(merged_cuda)
    max_abs = float(np.max(np.abs(np.asarray(merged_cpu) - np.asarray(merged_cuda))))
    report["full_build"] = {
        "cpu_ms": round(t_cpu_build * 1000, 2),
        "cuda_ms": round(t_cuda_build * 1000, 2),
        "lut_hash_cpu": hash_cpu,
        "lut_hash_cuda": hash_cuda,
        "lut_hash_match": hash_cpu == hash_cuda,
        "lut_max_abs_delta": max_abs,
        "golden_lut_hash": "a48758ca22a2e389",
        "cpu_matches_golden": hash_cpu == "a48758ca22a2e389",
        "cuda_matches_golden": hash_cuda == "a48758ca22a2e389",
    }
    print(f"  lut_hash CPU:  {hash_cpu}")
    print(f"  lut_hash CUDA: {hash_cuda}")
    print(f"  max |CPU-CUDA| LUT delta: {max_abs:.6g}")

    if not cuda_available():
        print("\nNote: CUDAExecutionProvider not available — CUDA timings used CPU fallback.")

    out_path = ROOT / "parity_out" / "bench_cuda_vs_cpu.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
