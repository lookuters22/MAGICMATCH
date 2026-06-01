"""Polarr probe face detection + skin parsing via ONNX (converted from TF.js)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
except ImportError as e:
    raise ImportError("MAGICMATCH face detection requires onnxruntime") from e

from .calibration_utils import normalize_face_box
from .reference import resize_hwc

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
FACE_MODEL_DIR = PACKAGE_ROOT / "models" / "face"

DETECT_THRESHOLD = 0.55
FACE_PARSE_SIZE = 256
MAX_SKIN_FACES = 4
MEAN_BGR = np.array([104.0, 117.0, 123.0], dtype=np.float32)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _clamp_int(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(v))))


def _area(box: tuple[int, int, int, int]) -> float:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _ious(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[float, float]:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = _area(a)
    area_b = _area(b)
    union = area_a + area_b - inter
    iou = inter / union if union > 0 else 0.0
    io_small = inter / min(area_a, area_b) if min(area_a, area_b) > 0 else 0.0
    return iou, io_small


class PriorBox:
    def __init__(self, image_size: tuple[int, int]) -> None:
        self.image_size = image_size
        self.min_sizes = [(10, 20), (32, 64), (128, 256)]
        self.steps = (8, 16, 32)
        self.variances = (0.1, 0.2)
        self.priors = self._build_priors()

    def _build_priors(self) -> np.ndarray:
        h, w = self.image_size
        anchors: list[list[float]] = []
        for k, step in enumerate(self.steps):
            fh = int(np.ceil(h / step))
            fw = int(np.ceil(w / step))
            for min_size in self.min_sizes[k]:
                skx = min_size / w
                sky = min_size / h
                for i in range(fh):
                    for j in range(fw):
                        cx = ((j + 0.5) * step) / w
                        cy = ((i + 0.5) * step) / h
                        anchors.append([cx, cy, skx, sky])
        return np.asarray(anchors, dtype=np.float32)

    def decode(self, loc: np.ndarray) -> np.ndarray:
        priors_xy = self.priors[:, :2]
        priors_wh = self.priors[:, 2:]
        loc_xy = loc[:, :2]
        loc_wh = loc[:, 2:]
        centers = priors_xy + loc_xy * self.variances[0] * priors_wh
        sizes = priors_wh * np.exp(loc_wh * self.variances[1])
        x1y1 = centers - sizes * 0.5
        x2y2 = sizes + x1y1
        return np.concatenate([x1y1, x2y2], axis=1)


@lru_cache(maxsize=2)
def _prior_landscape() -> PriorBox:
    return PriorBox((480, 640))


@lru_cache(maxsize=2)
def _prior_portrait() -> PriorBox:
    return PriorBox((640, 480))


@lru_cache(maxsize=1)
def _landscape_session() -> ort.InferenceSession:
    path = FACE_MODEL_DIR / "face_detect_landscape.onnx"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path} — run scripts/convert_face_models_to_onnx.py")
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


@lru_cache(maxsize=1)
def _portrait_session() -> ort.InferenceSession:
    path = FACE_MODEL_DIR / "face_detect_portrait.onnx"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path} — run scripts/convert_face_models_to_onnx.py")
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


@lru_cache(maxsize=1)
def _parse_session() -> ort.InferenceSession:
    path = FACE_MODEL_DIR / "face_parse.onnx"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path} — run scripts/convert_face_models_to_onnx.py")
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _gamma_correct(rgb: np.ndarray, gamma: float) -> np.ndarray:
    if abs(gamma - 1.0) < 1e-6:
        return rgb
    return np.power(np.clip(rgb, 0.0, 1.0), gamma, dtype=np.float32)


def _preprocess_detect(hwc: np.ndarray, size: tuple[int, int], adjusted_gamma: float) -> np.ndarray:
    h, w = size
    resized = resize_hwc(hwc, w, h, high_quality=False)
    rgb = (resized * 255.0).astype(np.float32)
    rgb = _gamma_correct(rgb / 255.0, adjusted_gamma) * 255.0
    bgr = rgb[..., ::-1]
    bgr -= MEAN_BGR
    return bgr[np.newaxis, ...].astype(np.float32)


def _preprocess_skin(hwc: np.ndarray, adjusted_gamma: float) -> np.ndarray:
    resized = resize_hwc(hwc, FACE_PARSE_SIZE, FACE_PARSE_SIZE, high_quality=False)
    rgb = _gamma_correct(resized, adjusted_gamma)
    chw = np.transpose(rgb, (2, 0, 1))
    chw = chw / 127.5 - 1.0
    return chw[np.newaxis, ...].astype(np.float32)


def _nms(
    boxes_yxyx: np.ndarray,
    scores: np.ndarray,
    *,
    max_output_size: int = 40,
    iou_threshold: float = 0.4,
    score_threshold: float = DETECT_THRESHOLD,
) -> np.ndarray:
    keep: list[int] = []
    order = np.where(scores >= score_threshold)[0]
    order = order[np.argsort(-scores[order])]
    while order.size > 0 and len(keep) < max_output_size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = []
        bi = boxes_yxyx[i]
        for j in rest:
            bj = boxes_yxyx[j]
            inter_x1 = max(bi[1], bj[1])
            inter_y1 = max(bi[0], bj[0])
            inter_x2 = min(bi[3], bj[3])
            inter_y2 = min(bi[2], bj[2])
            inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
            area_i = max(0.0, bi[3] - bi[1]) * max(0.0, bi[2] - bi[0])
            area_j = max(0.0, bj[3] - bj[1]) * max(0.0, bj[2] - bj[0])
            union = area_i + area_j - inter
            ious.append(inter / union if union > 0 else 0.0)
        ious = np.asarray(ious, dtype=np.float32)
        order = rest[ious <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def _extend_box(box: list[int], img_w: int, img_h: int, padding_scale: float = 1.3, shift_factor: float = 0.4) -> list[int]:
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    side = max(h, w) * padding_scale
    delta_w = (side - w) / 2.0
    delta_h = (side - h) / 2.0
    x1 = int(max(0, np.floor(x1 - delta_w)))
    x2 = int(min(img_w - 1, np.floor(x2 + delta_w)))
    y1 = int(max(0, np.floor(y1 - delta_h * (1.0 + shift_factor))))
    y2 = int(min(img_h - 1, np.floor(y2 + delta_h * (1.0 - shift_factor))))
    return [x1, y1, x2, y2]


def _face_location(box: tuple[int, int, int, int], img_w: int, img_h: int) -> dict[str, bool]:
    x = (box[0] + box[2]) / 2.0
    face_to_edge_w = min(box[0], img_w - 1 - box[2])
    face_to_edge_h = img_h - 1 - box[3]
    face_to_edge = min(face_to_edge_h / img_h, face_to_edge_w / img_w)
    face_to_top = box[1] / img_h
    face_to_bottom = 1.0 - (box[1] + box[3]) / 2.0 / img_h
    area = _area(box) / (img_w * img_h)
    at_edge = (
        face_to_edge < 0.1
        or face_to_top < 0.02
        or (face_to_bottom < 0.32 and area > 0.006)
        or face_to_bottom < 0.2
    )
    at_bottom = face_to_bottom < 0.25
    non_center = abs(x / img_w - 0.5) > 0.2 or face_to_bottom < 0.3 or face_to_top < 0.05
    at_strict_center = abs(x / img_w - 0.5) < 0.1 and 0.4 < face_to_bottom < 0.85 and not at_edge
    return {
        "at_edge": at_edge,
        "non_center": non_center,
        "at_strict_center": at_strict_center,
        "at_bottom": at_bottom,
    }


def _get_major_faces(
    box_list: list[tuple[tuple[int, int, int, int], float, int]],
    img_w: int,
    img_h: int,
    max_prob: float,
) -> dict | None:
    if max_prob < 0.93:
        return None
    thresh = min(max_prob - 0.01, 0.975)
    filtered = [item for item in box_list if item[1] >= thresh]
    kept = []
    for item in filtered:
        box = item[0]
        face_to_edge_w = min(box[0], img_w - 1 - box[2]) / img_w
        face_to_bottom = 1.0 - (box[1] + box[3]) / 2.0 / img_h
        if face_to_edge_w > 0.1 and face_to_bottom > 0.25:
            kept.append(item)
    if not kept:
        return None
    center_ys = [(b[0][1] + b[0][3]) / 2.0 for b in kept]
    heights = [b[0][3] - b[0][1] for b in kept]
    y_std = float(np.std(center_ys)) if len(center_ys) > 1 else 0.0
    height_mean = float(np.mean(heights))
    delta = max(y_std, min(height_mean, img_h / 5.0))
    y_range = (min(center_ys) - delta, max(center_ys) + delta)
    areas = sorted(_area(b[0]) for b in kept)
    area_range = (0.6 * areas[0], 1.4 * areas[-1])
    major_area = float(np.mean(areas[max(0, len(areas) - 3) :]))
    centered_count = sum(1 for b in kept if not _face_location(b[0], img_w, img_h)["non_center"])
    return {
        "yRange": y_range,
        "areaRange": area_range,
        "majorArea": major_area,
        "centeredCount": centered_count,
    }


def _get_face_bounding_boxes(hwc: np.ndarray, adjusted_gamma: float) -> list[dict]:
    h, w, _ = hwc.shape
    is_landscape = w >= h
    if is_landscape:
        sess = _landscape_session()
        tensor = _preprocess_detect(hwc, (480, 640), adjusted_gamma)
        prior = _prior_landscape()
    else:
        sess = _portrait_session()
        tensor = _preprocess_detect(hwc, (640, 480), adjusted_gamma)
        prior = _prior_portrait()

    landmark, bbox, confidence = sess.run(None, {"input0": tensor})
    loc = np.squeeze(bbox, axis=0)
    boxes = prior.decode(loc)
    scores = np.squeeze(confidence[..., 1], axis=0)
    boxes_yxyx = np.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=1)
    selected = _nms(boxes_yxyx, scores)
    if selected.size == 0:
        return []

    original = boxes[selected]
    probs = scores[selected]
    initial = []
    for boundary, prob in zip(original, probs):
        x1 = _clamp_int(boundary[0] * w, 0, w - 1)
        x2 = _clamp_int(boundary[2] * w, 0, w - 1)
        y1 = _clamp_int(boundary[1] * h, 0, h - 1)
        y2 = _clamp_int(boundary[3] * h, 0, h - 1)
        initial.append(([x1, y1, x2, y2], float(prob)))

    max_prob = max(p for _, p in initial)
    unpadded = [(box, prob) for box, prob in initial if _area(tuple(box)) > 36 and not (prob < 0.96 and (box[2] - box[0]) > (box[3] - box[1]) * 1.2)]

    box_list = [(_extend_box(list(box), w, h), prob, idx) for idx, (box, prob) in enumerate(unpadded)]
    box_list.sort(key=lambda x: x[1], reverse=True)
    removed: set[int] = set()
    for i in range(len(box_list)):
        if i in removed:
            continue
        box1 = box_list[i][0]
        for j in range(i + 1, len(box_list)):
            if j in removed:
                continue
            iou, io_small = _ious(tuple(box1), tuple(box_list[j][0]))
            if iou > 0.5 or io_small > 0.65:
                removed.add(j)
    box_list = [item for idx, item in enumerate(box_list) if idx not in removed]

    major = _get_major_faces(box_list, w, h, max_prob)
    initial_count = len(box_list)

    def keep_face(item: tuple[tuple[int, int, int, int], float, int]) -> bool:
        box, prob, _ = item
        area = _area(box)
        face_percent = area / (h * w)
        if face_percent > 0.05 and prob < 0.8:
            return False
        if face_percent > 0.03 and prob < 0.7:
            return False
        loc = _face_location(box, w, h)
        similar = False
        face_area_ratio = 1.0
        if major:
            y = (box[1] + box[3]) / 2.0
            y_range = major["yRange"]
            area_range = major["areaRange"]
            similar = area >= area_range[0] and area <= area_range[1] and y >= y_range[0] and y <= y_range[1]
            if area < area_range[0]:
                face_area_ratio = area / area_range[0]
            if area > area_range[1]:
                face_area_ratio = area / area_range[1]
        if face_area_ratio > 4 or face_area_ratio < 0.2:
            return False
        if prob < 0.9 and face_percent > 0.08 and loc["at_edge"] and loc["non_center"]:
            return False
        if prob < 0.98 and face_percent < 0.001 and loc["at_edge"] and loc["non_center"]:
            return False
        if not similar:
            if prob < 0.7 and major and (not loc["at_strict_center"] or major["centeredCount"] > 0):
                return False
            if prob < 0.9 and (loc["at_bottom"] or face_area_ratio > 2 or face_area_ratio < 0.3):
                return False
            if prob < 0.85 and (face_percent > 0.04 or (loc["at_edge"] and loc["non_center"])):
                return False
            if prob < 0.8 and face_percent > 0.01 and (loc["at_edge"] or (loc["non_center"] and prob < 0.7)):
                return False
            if prob < 0.75 and max_prob < 0.8 and face_percent > 0.0015 and loc["non_center"]:
                return False
            if prob < 0.7 and face_percent < 0.0007:
                return False
            if prob < 0.7 and initial_count == 1 and not loc["at_strict_center"]:
                return False
        return True

    box_list = [item for item in box_list if keep_face(item)]
    if len(box_list) > 1 and major:
        major_area = major["majorArea"]
        box_list = [
            item
            for item in box_list
            if not (
                _area(item[0]) < major_area / 2.5 and item[1] < max_prob - 0.03
                or (_area(item[0]) < major_area / 5 and item[1] < 0.98 and item[1] < max_prob - 0.005)
            )
        ]

    box_list.sort(key=lambda x: _area(x[0]), reverse=True)
    results: list[dict] = []
    for box, prob, idx in box_list:
        tup = tuple(box)
        unpadded_box = unpadded[idx][0] if idx < len(unpadded) else box
        results.append(
            {
                "face_box": list(tup),
                "face_rect": normalize_face_box(tup, w, h),
                "confidence": prob,
            }
        )
    return results


def _filter_center_faces(face_results: list[dict]) -> list[dict]:
    if len(face_results) <= 1:
        return face_results
    max_conf = max(r.get("confidence") or 0.0 for r in face_results)
    face_results = [r for r in face_results if (r.get("confidence") or 0.0) > max_conf - 0.05]

    def at_edge(rect: dict) -> bool:
        return rect["x"] < 0.05 or rect["x"] + rect["width"] > 0.95 or rect["y"] < 0.02 or rect["y"] + rect["height"] > 0.8

    def at_center(rect: dict) -> bool:
        cx = rect["x"] + rect["width"] / 2.0
        cy = rect["y"] + rect["height"] / 2.0
        return 0.3 < cx < 0.7 and 0.1 < cy < 0.7

    if len(face_results) > 1:
        face_results = [
            r for r in face_results if (r.get("confidence") or 0.0) > max_conf - 0.02 or not at_edge(r["face_rect"])
        ]
    if len(face_results) > 1:
        center = [r for r in face_results if at_center(r["face_rect"]) or ((r.get("confidence") or 0.0) > max_conf - 0.02 and not at_edge(r["face_rect"]))]
        if center and len(center) < len(face_results):
            face_results = center
    return face_results


def _detect_skin_condition(hwc: np.ndarray, face_results: list[dict], adjusted_gamma: float) -> list[dict]:
    if not face_results:
        return []
    h, w, _ = hwc.shape
    sess = _parse_session()
    center = (w / 2.0, h * 0.4)
    sorted_faces = sorted(
        face_results,
        key=lambda r: abs((r["face_box"][0] + r["face_box"][2]) / 2.0 - center[0])
        + abs((r["face_box"][1] + r["face_box"][3]) / 2.0 - center[1]) * 0.5,
    )
    out: list[dict] = []
    parsed = 0
    for result in sorted_faces:
        if parsed >= MAX_SKIN_FACES:
            out.append(result)
            continue
        box = result["face_box"]
        x1, y1, x2, y2 = box
        crop_w = x2 - x1
        crop_h = y2 - y1
        if crop_h < 5 or crop_w < 5:
            continue
        crop = hwc[y1:y2, x1:x2]
        tensor = _preprocess_skin(crop, adjusted_gamma)
        skin_mask = sess.run(None, {"input": tensor})[0][0]
        cond: list[list[bool]] = [[False] * crop_w for _ in range(crop_h)]
        scale_x = FACE_PARSE_SIZE / crop_w
        scale_y = FACE_PARSE_SIZE / crop_h
        skin_count = 0
        for x in range(crop_w):
            for y in range(crop_h):
                yi = int(y * scale_y)
                xi = int(x * scale_x)
                if skin_mask[yi, xi] > 0.6:
                    cond[y][x] = True
                    skin_count += 1
        skin_pct = skin_count / max(1, crop_w * crop_h)
        face_to_edge_w = min(box[0], w - 1 - box[2]) / w
        face_to_bottom = (h - 1 - box[3]) / h
        face_to_edge = min(face_to_edge_w, face_to_bottom)
        face_to_top = box[1] / h
        confidence = result.get("confidence") or 1.0
        if skin_pct <= 0.02:
            continue
        if confidence < 0.97 and (
            skin_pct <= 0.08
            or (skin_pct <= 0.18 and confidence < 0.93 and (face_to_edge <= 0.15 or face_to_top < 0.02 or confidence < 0.7))
            or (skin_pct <= 0.1 and (face_to_edge <= 0.2 or face_to_top < 0.04 or confidence < 0.8))
            or (skin_pct <= 0.23 and confidence < 0.65 and (face_to_edge <= 0.3 or face_to_top < 0.06 or confidence < 0.58))
        ):
            continue
        out.append({**result, "skin_condition": cond, "skin_percent": skin_pct})
        parsed += 1
    return out


def detect_faces(hwc: np.ndarray, *, adjusted_gamma: float = 1.0) -> list[dict]:
    """Run probe face detection + skin parsing; returns FaceDetectionResult dicts."""
    if not FACE_MODEL_DIR.is_dir():
        return []
    needed = (
        FACE_MODEL_DIR / "face_detect_landscape.onnx",
        FACE_MODEL_DIR / "face_detect_portrait.onnx",
        FACE_MODEL_DIR / "face_parse.onnx",
    )
    if not all(p.is_file() for p in needed):
        return []

    hwc = np.clip(np.asarray(hwc, dtype=np.float32), 0.0, 1.0)
    results = _get_face_bounding_boxes(hwc, adjusted_gamma)
    if len(results) > 1:
        results = _filter_center_faces(results)
    return _detect_skin_condition(hwc, results, adjusted_gamma)
