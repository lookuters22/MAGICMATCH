#!/usr/bin/env python3
"""Compare MAGICMATCH probe-parity pipeline against probe golden on a source/reference pair."""

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
from magicmatch_core.probe_parity.color_match_features import extract_color_match_features  # noqa: E402
from magicmatch_core.probe_parity.pipeline import (  # noqa: E402
    _render_for_net,
    apply_probe_export,
    build_merged_lut_probe_style,
)
from magicmatch_core.probe_parity.reference import prepare_net_reference, prepare_worker_bitmap_source  # noqa: E402
from magicmatch_core.inference import _resize_hwc_to_nhwc  # noqa: E402
from magicmatch_core.probe_parity.scene_extract import (  # noqa: E402
    extract_scene_info_bitmap,
    render_detection_inputs,
)
from magicmatch_core.probe_parity.calibration_utils import get_masked_pixels  # noqa: E402
from magicmatch_core.probe_parity.luminance import get_adjusted_gamma, get_luminance_statistics  # noqa: E402
from magicmatch_core.probe_parity.face_detection import detect_faces  # noqa: E402

# Captured from standalone probe (5180) on polarrnext/pair after Run Match.
# lut_hash verified from build_merged_lut_probe_style on this pair (2026-06-01);
# stale agent target 921fe3a509af375e never appeared in any report.json run.
PROBE_GOLDEN = {
    "base_adjustments": {
        "exposure": 0.736,
        "highlights": -0.56,
        "shadows": 0.008,
        "whites": -0.024,
        "blacks": 0.0,
        "contrast": 0.008,
        "saturation": -0.032,
        "temperature": 4553.594244980481,
        "tint": -1.9324997873548748,
    },
    "face_box_coords": 2,
    "face_percent": 0.005841372657111356,
    "avg_face_hsvl": [0.048701119494688215, 0.5028466751397473, 0.44210769285508195, 0.3052847828567028],
    "face_colors_len": 70,
    "face_colors_mean": 0.310904,
    "lum_percentiles": [
        0.171875,
        0.32421875,
        0.5390625,
        0.640625,
        0.74609375,
        0.953125,
        0.9609375,
    ],
    "lut_hash": "a48758ca22a2e389",
    "output_diff_mean": 0.0,
}


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def save_rgb(path: Path, hwc: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (np.clip(hwc, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(arr, "RGB").save(path)


def lut_hash(lut: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(lut, dtype=np.float32).tobytes()).hexdigest()[:16]


def _diff_metrics(actual: dict, golden: dict) -> dict:
    out: dict[str, float | int | bool] = {}
    for key, target in golden.items():
        if key == "base_adjustments":
            for sub_key, sub_target in target.items():
                val = actual.get(sub_key)
                if isinstance(sub_target, (int, float)) and isinstance(val, (int, float)):
                    out[f"{sub_key}_delta"] = float(val) - float(sub_target)
            continue
        if key == "face_box_coords":
            out["face_count_delta"] = int(actual.get("face_count", 0)) - int(target)
            continue
        if key == "face_colors_len":
            out["face_colors_len_delta"] = int(actual.get("face_colors_len", 0)) - int(target)
            continue
        if key == "lut_hash":
            out["lut_hash_match"] = actual.get("lut_hash") == target
            continue
        if key == "output_diff_mean":
            out["output_diff_mean_delta"] = float(actual.get("output_diff_mean", 0.0)) - float(target)
            continue
        val = actual.get(key)
        if isinstance(target, (int, float)) and isinstance(val, (int, float)):
            out[f"{key}_delta"] = float(val) - float(target)
        elif isinstance(target, list) and isinstance(val, list) and len(target) == len(val):
            out[f"{key}_max_abs_delta"] = float(
                max(abs(float(a) - float(b)) for a, b in zip(val, target))
            )
    return out


def _parity_failures(actual: dict, report: dict) -> list[str]:
    """Return human-readable failures against PROBE_GOLDEN tolerances."""
    failures: list[str] = []
    base = PROBE_GOLDEN["base_adjustments"]
    checks = [
        ("exposure", actual.get("exposure"), base["exposure"], 0.002),
        ("tint", actual.get("tint"), base["tint"], 0.05),
        ("temperature", actual.get("temperature"), base["temperature"], 1.0),
        ("whites", actual.get("whites"), base["whites"], 0.01),
    ]
    for name, val, target, tol in checks:
        if val is None or abs(float(val) - float(target)) > tol:
            failures.append(f"{name}: {val} vs {target} (±{tol})")
    if int(actual.get("face_count", 0)) != int(PROBE_GOLDEN["face_box_coords"]):
        failures.append(f"face_count: {actual.get('face_count')} vs {PROBE_GOLDEN['face_box_coords']}")
    fc_len = int(actual.get("face_colors_len", 0))
    if not (66 <= fc_len <= 70):
        failures.append(f"face_colors_len: {fc_len} not in 66–70")
    if float(report.get("output_diff_mean", 1.0)) != 0.0:
        failures.append(f"output_diff_mean: {report.get('output_diff_mean')}")
    if report.get("lut_hash") != PROBE_GOLDEN["lut_hash"]:
        failures.append(f"lut_hash: {report.get('lut_hash')} vs {PROBE_GOLDEN['lut_hash']}")
    return failures


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
    parser.add_argument(
        "--probe-browser",
        action="store_true",
        help="Use standalone_probe WebGPU worker for scene extract (MAGICMATCH_PROBE_BROWSER=1)",
    )
    args = parser.parse_args()

    if args.probe_browser:
        import os

        os.environ["MAGICMATCH_PROBE_BROWSER"] = "1"

    src = load_rgb(args.source)
    ref = load_rgb(args.reference)
    scene = extract_scene_info_bitmap(src, source_path=args.source)
    base = estimate_base_adjustments(src)
    merged, base_from_build = build_merged_lut_probe_style(src, ref)

    small, large = render_detection_inputs(src)
    stats = get_luminance_statistics(small, large, [])
    gamma = get_adjusted_gamma(stats)
    faces = scene.face_results or detect_faces(large, adjusted_gamma=gamma)
    fp = get_masked_pixels(large, faces, True) if faces else None
    features, face_data = extract_color_match_features(
        small, large, faces, is_reference=False, face_percent_and_pixels=fp
    )
    fc = face_data.get("faceColors")
    fc_mean = float(fc.reshape(-1, 3).mean()) if fc is not None and fc.size else None
    avg_face_hsvl = scene.avg_face_hsvl
    if avg_face_hsvl is None and features.get("avgFaceHsvl") is not None:
        avg_face_hsvl = features.get("avgFaceHsvl")

    out_probe = apply_probe_export(src, merged, args.strength, base_adjustments=base_from_build)
    out_polarr = apply_merged_lut_output(
        src,
        merged,
        args.strength,
        render_mode=RENDER_POLARR_PROBE,
        base_adjustments=base_from_build,
    )

    actual = {
        "face_count": len(scene.face_results),
        "face_percent": float(features.get("facePercent") or 0.0),
        "avg_face_hsvl": avg_face_hsvl.tolist() if avg_face_hsvl is not None else None,
        "face_colors_len": int(len(fc) // 3) if fc is not None else 0,
        "face_colors_mean": fc_mean,
        "lum_percentiles": stats.percentiles,
        **base,
    }

    report = {
        "source_shape": list(src.shape),
        "reference_shape": list(ref.shape),
        "detection_large_shape": list(large.shape),
        "detection_small_shape": list(small.shape),
        "actual": actual,
        "probe_golden": PROBE_GOLDEN,
        "delta": _diff_metrics({**actual, "lut_hash": lut_hash(merged)}, PROBE_GOLDEN),
        "base_from_build": base_from_build,
        "lut_hash": lut_hash(merged),
        "lut_mean": float(merged.mean()),
        "output_mean_probe_export": float(out_probe.mean()),
        "output_mean_polarr_probe": float(out_polarr.mean()),
        "output_diff_mean": float(np.abs(out_probe - out_polarr).mean()),
    }
    report["parity_failures"] = _parity_failures(actual, report)
    report["parity_ok"] = not report["parity_failures"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    worker_src = prepare_worker_bitmap_source(src)
    net_source = _render_for_net(worker_src, base_from_build)
    net_ref = prepare_net_reference(ref)
    np.save(args.out_dir / "net_source.nhwc.npy", _resize_hwc_to_nhwc(net_source))
    np.save(args.out_dir / "net_ref.nhwc.npy", _resize_hwc_to_nhwc(net_ref))
    save_rgb(args.out_dir / "magicmatch_probe_export.png", out_probe)
    save_rgb(args.out_dir / "magicmatch_polarr_probe.png", out_polarr)
    save_rgb(args.out_dir / "detection_large.png", (large * 255).astype(np.uint8))
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["parity_failures"]:
        print("PARITY FAILURES:", file=sys.stderr)
        for item in report["parity_failures"]:
            print(f"  - {item}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
