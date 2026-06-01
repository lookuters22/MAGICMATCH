"""Color space + LUT encode/decode (port of recovered color-space.ts)."""

from __future__ import annotations

import numpy as np

# LookTableRGBGamma
GAMMA_LINEAR = 0
GAMMA_SRGB = 1

# LookTableRGBPrimaries
PRIMARIES_SRGB = 0
PRIMARIES_ADOBE = 1
PRIMARIES_NONE = 5

MAT_PROPHOTO_TO_XYZ = np.array(
    [
        [0.7976719, 0.1351878, 0.0313396],
        [0.2880406, 0.7118695, 0.0000899],
        [0.0, 0.0, 0.8248898],
    ],
    dtype=np.float64,
)

MAT_XYZ_TO_PROPHOTO = np.array(
    [
        [1.3459468, -0.2556025, -0.0511079],
        [-0.5446045, 1.5081752, 0.0205265],
        [0.0, 0.0, 1.2122831],
    ],
    dtype=np.float64,
)

MAT_XYZ_TO_SRGB = np.array(
    [
        [3.1344866, -1.6174707, -0.4907276],
        [-0.9787282, 1.9161073, 0.0334373],
        [0.071971, -0.2290228, 1.4057805],
    ],
    dtype=np.float64,
)

MAT_SRGB_TO_XYZ = np.array(
    [
        [0.4360087, 0.3851511, 0.1430402],
        [0.2224659, 0.7169286, 0.0606055],
        [0.0139209, 0.0970801, 0.713899],
    ],
    dtype=np.float64,
)

MAT_XYZ_TO_ADOBE = np.array(
    [
        [2.041369, -0.5649464, -0.3446944],
        [-0.969266, 1.8760108, 0.041556],
        [0.0134474, -0.1183897, 1.0154096],
    ],
    dtype=np.float64,
)

MAT_ADOBE_TO_XYZ = np.array(
    [
        [0.5767309, 0.185554, 0.1881852],
        [0.2973769, 0.6273491, 0.0752741],
        [0.0270343, 0.0706872, 0.9911085],
    ],
    dtype=np.float64,
)

MAT_D50_D65 = np.array(
    [
        [0.9555766, -0.0230393, 0.0631636],
        [-0.0282895, 1.0099416, 0.0210077],
        [0.0122982, -0.020483, 1.3299098],
    ],
    dtype=np.float64,
)

MAT_D65_D50 = np.array(
    [
        [1.0478112, 0.0228866, -0.050127],
        [0.0295424, 0.9904844, -0.0170491],
        [-0.0092345, 0.0150436, 0.7521316],
    ],
    dtype=np.float64,
)

MAT_LUT_ENC_SRGB = MAT_XYZ_TO_SRGB @ MAT_PROPHOTO_TO_XYZ
MAT_LUT_DEC_SRGB = MAT_XYZ_TO_PROPHOTO @ MAT_SRGB_TO_XYZ
MAT_LUT_ENC_ADOBE = MAT_D50_D65 @ MAT_XYZ_TO_ADOBE @ MAT_PROPHOTO_TO_XYZ
MAT_LUT_DEC_ADOBE = MAT_XYZ_TO_PROPHOTO @ MAT_ADOBE_TO_XYZ @ MAT_D65_D50

MAT_SRGB_TO_PROPHOTO = MAT_XYZ_TO_PROPHOTO @ MAT_SRGB_TO_XYZ
MAT_PROPHOTO_TO_SRGB = MAT_XYZ_TO_SRGB @ MAT_PROPHOTO_TO_XYZ

ENCODING_PRESETS: dict[str, tuple[int, int]] = {
    "srgb_srgb": (GAMMA_SRGB, PRIMARIES_SRGB),
    "srgb_none": (GAMMA_SRGB, PRIMARIES_NONE),
    "linear_srgb": (GAMMA_LINEAR, PRIMARIES_SRGB),
    "linear_none": (GAMMA_LINEAR, PRIMARIES_NONE),
    "none_none": (5, PRIMARIES_NONE),
    "srgb_adobe": (GAMMA_SRGB, PRIMARIES_ADOBE),
    "linear_adobe": (GAMMA_LINEAR, PRIMARIES_ADOBE),
}


def _mul3(m: np.ndarray, v: np.ndarray) -> np.ndarray:
    return (m @ v.astype(np.float64)).astype(np.float32)


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    out = np.empty_like(rgb, dtype=np.float32)
    for i in range(3):
        c = rgb[..., i].astype(np.float32)
        out[..., i] = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return out


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    out = np.empty_like(rgb, dtype=np.float32)
    for i in range(3):
        c = np.maximum(rgb[..., i].astype(np.float32), 0.0)
        out[..., i] = np.where(
            c <= 0.0031308,
            c * 12.92,
            1.055 * np.power(c, 1.0 / 2.4) - 0.055,
        )
    return np.clip(out, 0.0, 1.0)


def srgb_to_prophoto(rgb: np.ndarray) -> np.ndarray:
    shape = rgb.shape
    lin = srgb_to_linear(rgb.reshape(-1, 3))
    pro = (lin.astype(np.float64) @ MAT_SRGB_TO_PROPHOTO.T).astype(np.float32)
    return pro.reshape(shape)


def prophoto_to_srgb(rgb: np.ndarray) -> np.ndarray:
    shape = rgb.shape
    flat = rgb.reshape(-1, 3).astype(np.float64)
    xyz_lin = (flat @ MAT_PROPHOTO_TO_SRGB.T).astype(np.float32)
    return linear_to_srgb(xyz_lin.reshape(shape))


def lut_primaries_encode(rgb: np.ndarray, primaries: int) -> np.ndarray:
    flat = rgb.reshape(-1, 3).astype(np.float64)
    if primaries == PRIMARIES_SRGB:
        out = (flat @ MAT_LUT_ENC_SRGB.T).astype(np.float32)
    elif primaries == PRIMARIES_ADOBE:
        out = (flat @ MAT_LUT_ENC_ADOBE.T).astype(np.float32)
    else:
        out = flat.astype(np.float32)
    return out.reshape(rgb.shape)


def lut_primaries_decode(rgb: np.ndarray, primaries: int) -> np.ndarray:
    flat = rgb.reshape(-1, 3).astype(np.float64)
    if primaries == PRIMARIES_SRGB:
        out = (flat @ MAT_LUT_DEC_SRGB.T).astype(np.float32)
    elif primaries == PRIMARIES_ADOBE:
        out = (flat @ MAT_LUT_DEC_ADOBE.T).astype(np.float32)
    else:
        out = flat.astype(np.float32)
    return out.reshape(rgb.shape)


def lut_gamma_encode(rgb: np.ndarray, gamma: int) -> np.ndarray:
    if gamma == GAMMA_SRGB:
        return linear_to_srgb(rgb)
    if gamma == 1:  # gamma 1.8 in TS enum
        return np.power(np.maximum(rgb, 0.0), 1.0 / 1.8).astype(np.float32)
    if gamma == 2:  # gamma 2.2
        return np.power(np.maximum(rgb, 0.0), 1.0 / 2.2).astype(np.float32)
    if gamma == 3:  # rec2020
        return np.power(np.maximum(rgb, 0.0), 1.0 / 2.4).astype(np.float32)
    return rgb.astype(np.float32, copy=True)


def lut_gamma_decode(rgb: np.ndarray, gamma: int) -> np.ndarray:
    if gamma == GAMMA_SRGB:
        return srgb_to_linear(rgb)
    if gamma == 1:
        return np.power(np.maximum(rgb, 0.0), 1.8).astype(np.float32)
    if gamma == 2:
        return np.power(np.maximum(rgb, 0.0), 2.2).astype(np.float32)
    if gamma == 3:
        return np.power(np.maximum(rgb, 0.0), 2.4).astype(np.float32)
    return rgb.astype(np.float32, copy=True)
