"""
Legacy notes on standalone_probe WebGL export.

Full RAW + Lightroom-profile export is still richer than the in-Comfy Python stack,
but MAGICMATCH now implements probe-parity build/apply in magicmatch_core/probe_parity/
(auto-light, develop@1600→256 net feed, profile look table, probe_export render mode).
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
