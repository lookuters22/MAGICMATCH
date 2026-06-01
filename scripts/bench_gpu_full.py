#!/usr/bin/env python3
"""Benchmark full GPU pipeline with phase breakdown and CPU parity comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from magicmatch_core.gpu.device import gpu_pipeline_available  # noqa: E402
from magicmatch_core.gpu.pipeline_full_gpu import profile_full_gpu_pipeline  # noqa: E402
from magicmatch_core.onnx_providers import cuda_available, get_available_providers  # noqa: E402


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?", help="Source image (PNG/JPG)")
    parser.add_argument("reference", type=Path, nargs="?", help="Reference image (PNG/JPG)")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=ROOT / "parity_out" / "bench_gpu_full.json",
        help="Write JSON report path",
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

    print("=== Full GPU pipeline profile ===")
    print(f"  torch.cuda: {gpu_pipeline_available()}")
    print(f"  ORT CUDA EP: {cuda_available()}")
    print(f"  providers: {get_available_providers()}")

    report = profile_full_gpu_pipeline(src, ref, strength=args.strength)
    report["source"] = str(source_path)
    report["reference"] = str(reference_path)
    report["strength"] = args.strength

    if "error" in report:
        print(json.dumps(report, indent=2))
        return 1

    print("\n=== Phase timings (ms) ===")
    for key, val in report.items():
        if isinstance(val, (int, float)) and key.endswith("_ms"):
            print(f"  {key}: {val:.1f}")

    print("\n=== Parity vs CPU ===")
    print(f"  lut_hash CPU:  {report.get('lut_hash_cpu')}")
    print(f"  lut_hash GPU:  {report.get('lut_hash_gpu')}")
    print(f"  golden:        {report.get('golden_lut_hash')}")
    print(f"  match:         {report.get('lut_hash_match')}")
    print(f"  max LUT delta: {report.get('lut_max_abs_delta')}")
    print(f"  output mean Δ: {report.get('output_mean_delta')}")

    print("\n=== Still on CPU ===")
    for item in report.get("still_on_cpu", []):
        print(f"  - {item}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
