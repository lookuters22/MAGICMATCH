"""White balance matrices (port of temperature.ts + color.ts)."""

from __future__ import annotations

import numpy as np

from ..polarr_color_space import MAT_PROPHOTO_TO_XYZ, MAT_XYZ_TO_PROPHOTO

TINT_SCALE = -3000.0

TEMP_TABLE = [
    (0, 0.18006, 0.26352, -0.24341),
    (10, 0.18066, 0.26589, -0.25479),
    (20, 0.18133, 0.26846, -0.26876),
    (30, 0.18208, 0.27119, -0.28539),
    (40, 0.18293, 0.27407, -0.3047),
    (50, 0.18388, 0.27709, -0.32675),
    (60, 0.18494, 0.28021, -0.35156),
    (70, 0.18611, 0.28342, -0.37915),
    (80, 0.1874, 0.28668, -0.40955),
    (90, 0.1888, 0.28997, -0.44278),
    (100, 0.19032, 0.29326, -0.47888),
    (125, 0.19462, 0.30141, -0.58204),
    (150, 0.19962, 0.30921, -0.70471),
    (175, 0.20525, 0.31647, -0.84901),
    (200, 0.21142, 0.32312, -1.0182),
    (225, 0.21807, 0.32909, -1.2168),
    (250, 0.22511, 0.33439, -1.4512),
    (275, 0.23247, 0.33904, -1.7298),
    (300, 0.2401, 0.34308, -2.0637),
    (325, 0.24702, 0.34655, -2.4681),
    (350, 0.25591, 0.34951, -2.9641),
    (375, 0.264, 0.352, -3.5814),
    (400, 0.27218, 0.35407, -4.3633),
    (425, 0.28039, 0.35577, -5.3762),
    (450, 0.28863, 0.35714, -6.7262),
    (475, 0.29685, 0.35823, -8.5955),
    (500, 0.30505, 0.35907, -11.324),
    (525, 0.3132, 0.35968, -15.628),
    (550, 0.32129, 0.36011, -23.325),
    (575, 0.32931, 0.36038, -40.77),
    (600, 0.33724, 0.36051, -116.45),
]

ADAPTATION_BRADFORD = np.array(
    [
        [0.8951, 0.2664, -0.1614],
        [-0.7502, 1.7135, 0.0367],
        [0.0389, -0.0685, 1.0296],
    ],
    dtype=np.float64,
)

ADAPTATION_BRADFORD_INV = np.array(
    [
        [0.9869929, -0.1470543, 0.1599627],
        [0.4323053, 0.5183603, 0.0492912],
        [-0.0085287, 0.0400428, 0.9684867],
    ],
    dtype=np.float64,
)

# Bitmap/JPEG: Photo.ts + probe polarrFullRawRenderer hard-code 5000K / 0 tint.
DEFAULT_AS_SHOT_TEMP = 5000.0
DEFAULT_AS_SHOT_TINT = 0.0


def white_balance_to_xy(temperature: float, tint: float) -> tuple[float, float]:
    r = 1.0e6 / temperature
    offset = tint * (1.0 / TINT_SCALE)
    x = y = 0.0
    for index in range(30):
        plus1_r = TEMP_TABLE[index + 1][0]
        if r < plus1_r or index == 29:
            f = (plus1_r - r) / (plus1_r - TEMP_TABLE[index][0])
            u = TEMP_TABLE[index][1] * f + TEMP_TABLE[index + 1][1] * (1.0 - f)
            v = TEMP_TABLE[index][2] * f + TEMP_TABLE[index + 1][2] * (1.0 - f)
            uu1, vv1 = 1.0, TEMP_TABLE[index][3]
            uu2, vv2 = 1.0, TEMP_TABLE[index + 1][3]
            len1 = (1.0 + vv1 * vv1) ** 0.5
            len2 = (1.0 + vv2 * vv2) ** 0.5
            uu1, vv1 = uu1 / len1, vv1 / len1
            uu2, vv2 = uu2 / len2, vv2 / len2
            uu3 = uu1 * f + uu2 * (1.0 - f)
            vv3 = vv1 * f + vv2 * (1.0 - f)
            len3 = (uu3 * uu3 + vv3 * vv3) ** 0.5
            uu3, vv3 = uu3 / len3, vv3 / len3
            u += uu3 * offset
            v += vv3 * offset
            x = (1.5 * u) / (u - 4.0 * v + 2.0)
            y = v / (u - 4.0 * v + 2.0)
            break
    return x, y


def _xy_to_xyz(x: float, y: float) -> np.ndarray:
    return np.array([x / y, 1.0, (1.0 - x - y) / y], dtype=np.float64)


def xyz_to_xy(xyz: np.ndarray) -> tuple[float, float]:
    """Port of XYZtoXY from color-conversion.ts."""
    x, y, z = np.asarray(xyz, dtype=np.float64).reshape(3)
    s = x + y + z
    return float(x / s), float(y / s)


def xy_to_white_balance(x: float, y: float) -> tuple[float, float]:
    """Port of XYtoWhiteBalance from temperature.ts."""
    u = (2.0 * x) / (1.5 - x + 6.0 * y)
    v = (3.0 * y) / (1.5 - x + 6.0 * y)
    temperature = 0.0
    tint = 0.0
    last_dt = 0.0
    last_du = 0.0
    last_dv = 0.0
    for index in range(1, 31):
        temp_at = TEMP_TABLE[index]
        temp_prior = TEMP_TABLE[index - 1]
        du = 1.0
        dv = temp_at[3]
        length = (1.0 + dv * dv) ** 0.5
        du, dv = du / length, dv / length
        uu = u - temp_at[1]
        vv = v - temp_at[2]
        dt = -uu * dv + vv * du
        if dt <= 0.0 or index == 30:
            if dt > 0.0:
                dt = 0.0
            dt = -dt
            if index == 1:
                f = 0.0
            else:
                f = dt / (last_dt + dt)
            temperature = 1.0e6 / (temp_prior[0] * f + temp_at[0] * (1.0 - f))
            uu = u - (temp_prior[1] * f + temp_at[1] * (1.0 - f))
            vv = v - (temp_prior[2] * f + temp_at[2] * (1.0 - f))
            du = du * (1.0 - f) + last_du * f
            dv = dv * (1.0 - f) + last_dv * f
            length = (du * du + dv * dv) ** 0.5
            du, dv = du / length, dv / length
            tint = (uu * du + vv * dv) * TINT_SCALE
            break
        last_dt = dt
        last_du = du
        last_dv = dv
    return temperature, tint


def _get_lms_gains(src_white: np.ndarray, dst_white: np.ndarray) -> np.ndarray:
    ps, ys, bs = ADAPTATION_BRADFORD @ src_white
    pd, yd, bd = ADAPTATION_BRADFORD @ dst_white
    gains = np.array([pd / ps, yd / ys, bd / bs], dtype=np.float64)
    for i, p in enumerate([ps, ys, bs]):
        gains[i] = max(0.1, min(gains[i] if p > 0 else 10.0, 10.0))
    return gains


def get_adaption_matrix(src_white: np.ndarray, dst_white: np.ndarray) -> np.ndarray:
    gains = _get_lms_gains(src_white, dst_white)
    scale = np.diag(gains)
    return ADAPTATION_BRADFORD_INV @ scale @ ADAPTATION_BRADFORD


def build_wb_matrix(
    temperature: float,
    tint: float,
    *,
    as_shot_temp: float = DEFAULT_AS_SHOT_TEMP,
    as_shot_tint: float = DEFAULT_AS_SHOT_TINT,
) -> np.ndarray:
    """ProPhoto working-space WB matrix used by renderCpu."""
    as_shot_xy = white_balance_to_xy(as_shot_temp, as_shot_tint)
    adjusted_xy = white_balance_to_xy(temperature, tint)
    as_shot_white = _xy_to_xyz(*as_shot_xy)
    adjusted_white = _xy_to_xyz(*adjusted_xy)
    wb_xyz = get_adaption_matrix(adjusted_white, as_shot_white)
    return MAT_XYZ_TO_PROPHOTO @ wb_xyz @ MAT_PROPHOTO_TO_XYZ
