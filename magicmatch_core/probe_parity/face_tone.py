"""Face heuristics from face-tone.ts (color-match base light scaling + rejection)."""

from __future__ import annotations

import numpy as np


def is_reasonable_face_color(face_hsvl: np.ndarray) -> bool:
    hue, sat, _, lum = face_hsvl
    if lum > 0.86:
        return False
    if 44 / 360 < hue < 250 / 360 and 0.15 < sat < 0.7:
        return False
    return True


def is_good_face_color(face_hsvl: np.ndarray) -> bool:
    hue, sat, _, lum = face_hsvl
    return bool(hue > 0 and hue < 30 / 360 and 0.15 < sat < 0.76 and 0.3 < lum < 0.76)


def is_face_too_white(face_hsvl: np.ndarray) -> bool:
    sat, lum = face_hsvl[1], face_hsvl[3]
    return bool(lum - sat > 0.6 and sat < 0.15)


def is_face_problematic(face_hsvl: np.ndarray | None) -> bool:
    if face_hsvl is None:
        return False
    hue, sat = face_hsvl[0], face_hsvl[1]
    if ((hue < 12 / 360 or hue > 40 / 360) and sat > 0.8) or (
        (hue < 15 / 360 or hue > 40 / 360) and sat > 0.9
    ):
        return True
    return not is_reasonable_face_color(face_hsvl)


def is_face_dark_skin(face_hsvl: np.ndarray | None) -> bool:
    if face_hsvl is None:
        return False
    return bool(face_hsvl[1] > 0.7 or face_hsvl[1] - face_hsvl[3] > 0.55)
