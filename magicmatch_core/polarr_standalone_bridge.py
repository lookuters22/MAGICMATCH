"""
Optional bridge to Polarr Next Probe full WebGL export (RAW + LR Profile).

The probe applies color match through renderPolarrFullRcdRawPreview + adjustments.frag
(profile look tables, tone curves, display-p3). That path is not replicated in Python.

For pixel-perfect parity with Save Image from standalone_probe, export from the probe
with the same source/reference, then compare to Comfy polarr_probe mode on JPEG/PNG.

Future: headless CLI calling standalone_probe render stack.
"""

from __future__ import annotations

from pathlib import Path

STANDALONE_PROBE_DIR = (
    Path(__file__).resolve().parents[2].parent / "polarrnext" / "standalone_probe"
)


def standalone_probe_available() -> bool:
    return (STANDALONE_PROBE_DIR / "package.json").is_file()


def standalone_probe_dev_url() -> str:
    return "http://127.0.0.1:5179/"
