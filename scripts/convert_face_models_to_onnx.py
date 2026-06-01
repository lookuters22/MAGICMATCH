#!/usr/bin/env python3
"""Convert Polarr probe face TF.js graphs to ONNX (same toolchain as color_match.onnx)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE_MODELS = (
    ROOT.parent / "polarrnext" / "standalone_probe" / "public" / "models"
)
OUT_DIR = ROOT / "models" / "face"

FACE_MODELS = (
    ("b_640_f16", "face_detect_landscape.onnx"),
    ("b_480_f16", "face_detect_portrait.onnx"),
    ("f_f16", "face_parse.onnx"),
)


def convert_one(tfjs_dir: Path, onnx_path: Path) -> bool:
    tfjs = tfjs_dir / "model.json"
    if not tfjs.is_file():
        print(f"Missing {tfjs}", file=sys.stderr)
        return False
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "tf2onnx.convert",
        "--tfjs",
        str(tfjs),
        "--output",
        str(onnx_path),
    ]
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, cwd=ROOT, check=True)
    except subprocess.CalledProcessError:
        return False
    if onnx_path.is_file():
        print(f"Wrote {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")
        return True
    return False


def main() -> int:
    if not PROBE_MODELS.is_dir():
        print(f"Missing probe models dir: {PROBE_MODELS}", file=sys.stderr)
        return 1

    ok = 0
    for subdir, out_name in FACE_MODELS:
        if convert_one(PROBE_MODELS / subdir, OUT_DIR / out_name):
            ok += 1

    if ok != len(FACE_MODELS):
        print(
            "\nSome conversions failed. Use Python 3.12 venv with requirements-convert.txt:\n"
            "  cd polarrnext/color_match_extract\n"
            "  py -3.12 -m venv .venv312\n"
            "  .venv312\\Scripts\\pip install -r requirements-convert.txt\n"
            "  .venv312\\Scripts\\python ..\\..\\MAGICMATCH\\scripts\\convert_face_models_to_onnx.py\n",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
