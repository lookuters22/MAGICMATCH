"""Port of ai-calibration-utils.ts color + masking helpers used by auto-WB."""

from __future__ import annotations

import numpy as np

from .wb import ADAPTATION_BRADFORD, ADAPTATION_BRADFORD_INV

RGB_TO_XYZ = np.array(
    [
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227],
    ],
    dtype=np.float64,
)

XYZ_TO_RGB = np.array(
    [
        [3.240481, -1.537151, -0.498536],
        [-0.969256, 1.875990, 0.0415560],
        [0.055647, -0.204041, 1.057311],
    ],
    dtype=np.float64,
)

RGB_TO_YUV = np.array(
    [
        [0.299, 0.587, 0.114],
        [-0.299, -0.587, 0.886],
        [0.701, -0.587, -0.114],
    ],
    dtype=np.float64,
)

YUV_TO_RGB = np.linalg.inv(RGB_TO_YUV)

ILLUMINANT_D65 = np.array([0.9504, 1.0, 1.0889], dtype=np.float64)

TEMPERATURE_RANGE = (2000.0, 25000.0)
TINT_RANGE = (-150.0, 150.0)


def get_mean(data: np.ndarray | list[float]) -> float:
    arr = np.asarray(data, dtype=np.float64)
    return float(np.mean(arr)) if arr.size else 0.0


def srgb_to_linear_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    out = np.where(
        rgb <= 0.0031308,
        rgb * 12.92,
        1.055 * np.power(np.maximum(rgb, 0.0), 1.0 / 2.4) - 0.055,
    )
    return np.clip(out, 0.0, 1.0)


def srgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    lin = srgb_to_linear_rgb(np.asarray(rgb, dtype=np.float64))
    return lin @ RGB_TO_XYZ.T


def xyz_to_srgb(xyz: np.ndarray) -> np.ndarray:
    rgb_lin = np.asarray(xyz, dtype=np.float64) @ XYZ_TO_RGB.T
    return linear_to_srgb_rgb(rgb_lin)


def rgb_to_yuv(rgb: np.ndarray) -> np.ndarray:
    return np.asarray(rgb, dtype=np.float64) @ RGB_TO_YUV.T


def yuv_to_rgb(yuv: np.ndarray) -> np.ndarray:
    return np.asarray(yuv, dtype=np.float64) @ YUV_TO_RGB.T


def normalize_xyz(vec: np.ndarray) -> np.ndarray:
    x, y, z = vec
    return np.array([x / y, 1.0, z / y], dtype=np.float64)


def xyz_to_lms(xyz: np.ndarray) -> np.ndarray:
    return normalize_xyz(xyz) @ ADAPTATION_BRADFORD.T


def lms_to_xyz(lms: np.ndarray) -> np.ndarray:
    return np.asarray(lms, dtype=np.float64) @ ADAPTATION_BRADFORD_INV.T


def get_mean_v3(data: list[np.ndarray]) -> np.ndarray:
    if not data:
        return np.zeros(3, dtype=np.float64)
    stack = np.stack(data, axis=0)
    return np.mean(stack, axis=0)


def shape_rgba_uint8(rgba: np.ndarray) -> list[np.ndarray]:
    rgba = np.asarray(rgba, dtype=np.float64)
    out: list[np.ndarray] = []
    for i in range(0, rgba.shape[0], 4):
        out.append(rgba[i : i + 3] / 255.0)
    return out


def hwc_to_rgba_uint8(hwc: np.ndarray) -> np.ndarray:
    hwc = np.clip(np.asarray(hwc, dtype=np.float32), 0.0, 1.0)
    rgb = (hwc * 255.0).astype(np.uint8)
    alpha = np.full(rgb.shape[:2] + (1,), 255, dtype=np.uint8)
    return np.concatenate([rgb, alpha], axis=-1).reshape(-1)


def get_masked_pixels(
    hwc: np.ndarray,
    face_results: list[dict],
    use_step: bool = True,
) -> tuple[float, np.ndarray]:
    """Port of getMaskedPixels — returns (masked_percent, masked_rgba_flat)."""
    h, w, _ = hwc.shape
    rgba = hwc_to_rgba_uint8(hwc)
    data = rgba.reshape(-1)

    filtered = [f for f in face_results if f.get("skin_condition") is not None]
    has_mask = bool(filtered)
    if not filtered:
        filtered = face_results
        has_mask = False

    step = 1
    if use_step:
        estimated = 0
        for face in filtered:
            box = face["face_box"]
            x1, y1, x2, y2 = box
            skin_pct = face.get("skin_percent", 0.25) if has_mask else 1.0
            estimated += (x2 - x1) * (y2 - y1) * skin_pct
        step = int(round((estimated / 10000.0) ** 0.5))
        step = max(1, min(step, 4))

    result: list[int] = []
    for face in filtered:
        box = face["face_box"]
        mask = face.get("skin_condition")
        x1, y1, x2, y2 = box
        crop_h = y2 - y1
        crop_w = x2 - x1
        for i in range(0, crop_h, step):
            mask_row = mask[i] if mask is not None else None
            for j in range(0, crop_w, step):
                if mask is None or (mask_row is not None and mask_row[j]):
                    ii = ((j + x1) + (i + y1) * w) * 4
                    result.extend(
                        [
                            int(data[ii]),
                            int(data[ii + 1]),
                            int(data[ii + 2]),
                            int(data[ii + 3]),
                        ]
                    )

    estimated_total = (len(result) / 4) * step * step
    masked_percent = (estimated_total * 4) / len(data) if len(data) else 0.0
    return masked_percent, np.asarray(result, dtype=np.uint8)


def normalize_face_box(face_box: tuple[int, int, int, int], width: int, height: int) -> dict:
    x1, y1, x2, y2 = face_box
    return {
        "x": x1 / width,
        "y": y1 / height,
        "width": (x2 - x1) / width,
        "height": (y2 - y1) / height,
    }
