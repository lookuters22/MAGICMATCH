#!/usr/bin/env python3
"""Compare MAGICMATCH probe-parity pipeline against a source/reference pair on disk."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from magicmatch_core.apply import (  # noqa: E402
    RENDER_POLARR_PROBE,
    RENDER_PROBE_EXPORT,
    apply_merged_lut_output,
)
from magicmatch_core.probe_parity.base_adjustments import estimate_base_adjustments  # noqa: E402
from magicmatch_core.probe_parity.pipeline import (  # noqa: E402
    apply_probe_export,
    build_merged_lut_probe_style,
)
from magicmatch_core.probe_parity.scene_extract import extract_scene_info_bitmap  # noqa: E402


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def save_rgb(path: Path, hwc: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (np.clip(hwc, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(arr, "RGB").save(path)


def lut_hash(lut: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(lut, dtype=np.float32).tobytes()).hexdigest()[:16]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Source image path")
    parser.add_argument("reference", type=Path, help="Reference image path")
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=ROOT / "parity_out",
        help="Directory for diagnostic PNGs + JSON",
    )
    parser.add_argument("--strength", type=float, default=1.0)
    args = parser.parse_args()

    src = load_rgb(args.source)
    ref = load_rgb(args.reference)
    scene = extract_scene_info_bitmap(src)
    base = estimate_base_adjustments(src)
    merged, base_from_build = build_merged_lut_probe_style(src, ref)

    out_probe = apply_probe_export(
        src, merged, args.strength, base_adjustments=base_from_build
    )
    out_polarr = apply_merged_lut_output(
        src,
        merged,
        args.strength,
        render_mode=RENDER_POLARR_PROBE,
        base_adjustments=base_from_build,
    )

    report = {
        "source_shape": list(src.shape),
        "reference_shape": list(ref.shape),
        "face_count": len(scene.face_results),
        "base_adjustments": base,
        "base_from_build": base_from_build,
        "lut_hash": lut_hash(merged),
        "lut_mean": float(merged.mean()),
        "output_mean_probe_export": float(out_probe.mean()),
        "output_mean_polarr_probe": float(out_polarr.mean()),
        "output_diff_mean": float(np.abs(out_probe - out_polarr).mean()),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_rgb(args.out_dir / "magicmatch_probe_export.png", out_probe)
    save_rgb(args.out_dir / "magicmatch_polarr_probe.png", out_polarr)
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
