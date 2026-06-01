"""
CUDA ONNX sessions for probe face detection + skin parsing.

CPU parity path remains face_detection.py (CPUExecutionProvider only).
Reuses post-processing helpers from face_detection; only sess.run uses GPU when available.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
except ImportError as e:
    raise ImportError("MAGICMATCH face detection CUDA path requires onnxruntime") from e

from ..onnx_providers import create_onnx_session, cuda_available, session_active_provider
from . import face_detection as _cpu
from .calibration_utils import normalize_face_box

FACE_MODEL_DIR = _cpu.FACE_MODEL_DIR
DETECT_THRESHOLD = _cpu.DETECT_THRESHOLD
FACE_PARSE_SIZE = _cpu.FACE_PARSE_SIZE
MAX_SKIN_FACES = _cpu.MAX_SKIN_FACES


@lru_cache(maxsize=1)
def _landscape_session_cuda() -> ort.InferenceSession:
    path = FACE_MODEL_DIR / "face_detect_landscape.onnx"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path} — run scripts/convert_face_models_to_onnx.py")
    return create_onnx_session(path, prefer_cuda=True)


@lru_cache(maxsize=1)
def _portrait_session_cuda() -> ort.InferenceSession:
    fp32 = FACE_MODEL_DIR / "face_detect_portrait_fp32.onnx"
    path = fp32 if fp32.is_file() else FACE_MODEL_DIR / "face_detect_portrait.onnx"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path} — run scripts/export_face_fp32_weights_playwright.py "
            "or scripts/convert_face_models_to_onnx.py"
        )
    return create_onnx_session(path, prefer_cuda=True)


@lru_cache(maxsize=1)
def _parse_session_cuda() -> ort.InferenceSession:
    path = FACE_MODEL_DIR / "face_parse.onnx"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path} — run scripts/convert_face_models_to_onnx.py")
    return create_onnx_session(path, prefer_cuda=True)


def face_session_info() -> dict[str, str | bool | dict[str, str]]:
    """Active providers for the three face ONNX models (creates sessions if needed)."""
    sessions = {
        "landscape": _landscape_session_cuda(),
        "portrait": _portrait_session_cuda(),
        "parse": _parse_session_cuda(),
    }
    return {
        "cuda_available": cuda_available(),
        "providers": {name: session_active_provider(sess) for name, sess in sessions.items()},
    }


def _get_face_bounding_boxes_cuda(hwc: np.ndarray, adjusted_gamma: float) -> list[dict]:
    h, w, _ = hwc.shape
    is_landscape = w >= h
    if is_landscape:
        sess = _landscape_session_cuda()
        tensor = _cpu._preprocess_detect(hwc, (480, 640), adjusted_gamma)
        prior = _cpu._prior_landscape()
    else:
        sess = _portrait_session_cuda()
        tensor = _cpu._preprocess_detect(hwc, (640, 480), adjusted_gamma)
        prior = _cpu._prior_portrait()

    raw = sess.run(None, {_cpu._session_input_name(sess): tensor})
    landmark, bbox, confidence = _cpu._unpack_detect_outputs(list(raw))
    loc = np.squeeze(bbox, axis=0)
    boxes = prior.decode(loc)
    inverted = _cpu._cpu_confidence_inverted(confidence, boxes)
    initial: list[tuple[list[int], float]] = []
    if inverted:
        initial = _cpu._pick_inverted_portrait_boxes(boxes, confidence, w, h)
    if not initial:
        scores = _cpu._face_class_scores(confidence, boxes)
        boxes_yxyx = np.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=1)
        selected = _cpu._nms(
            boxes_yxyx,
            scores,
            score_threshold=_cpu._nms_threshold_for_scores(scores),
        )
        if selected.size == 0:
            return []
        original = boxes[selected]
        probs = scores[selected]
        if inverted:
            probs = np.maximum(probs, 0.95)
        for boundary, prob in zip(original, probs):
            x1 = _cpu._clamp_int(boundary[0] * w, 0, w - 1)
            x2 = _cpu._clamp_int(boundary[2] * w, 0, w - 1)
            y1 = _cpu._clamp_int(boundary[1] * h, 0, h - 1)
            y2 = _cpu._clamp_int(boundary[3] * h, 0, h - 1)
            initial.append(([x1, y1, x2, y2], float(prob)))
    if not initial:
        return []

    max_prob = max(p for _, p in initial)
    unpadded = [
        (box, prob)
        for box, prob in initial
        if _cpu._area(tuple(box)) > 36
        and not (prob < 0.96 and (box[2] - box[0]) > (box[3] - box[1]) * 1.2)
    ]

    if inverted:
        box_list = [(tuple(box), prob, idx) for idx, (box, prob) in enumerate(unpadded)]
    else:
        box_list = [(_cpu._extend_box(list(box), w, h), prob, idx) for idx, (box, prob) in enumerate(unpadded)]
    box_list.sort(key=lambda x: x[1], reverse=True)
    removed: set[int] = set()
    for i in range(len(box_list)):
        if i in removed:
            continue
        box1 = box_list[i][0]
        for j in range(i + 1, len(box_list)):
            if j in removed:
                continue
            iou, io_small = _cpu._ious(tuple(box1), tuple(box_list[j][0]))
            if iou > 0.5 or io_small > 0.65:
                removed.add(j)
    box_list = [item for idx, item in enumerate(box_list) if idx not in removed]

    major = _cpu._get_major_faces(box_list, w, h, max_prob)
    initial_count = len(box_list)

    def keep_face(item: tuple[tuple[int, int, int, int], float, int]) -> bool:
        box, prob, _ = item
        area = _cpu._area(box)
        face_percent = area / (h * w)
        if face_percent > 0.05 and prob < 0.8:
            return False
        if face_percent > 0.03 and prob < 0.7:
            return False
        loc = _cpu._face_location(box, w, h)
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
                _cpu._area(item[0]) < major_area / 2.5 and item[1] < max_prob - 0.03
                or (
                    _cpu._area(item[0]) < major_area / 5
                    and item[1] < 0.98
                    and item[1] < max_prob - 0.005
                )
            )
        ]

    box_list.sort(key=lambda x: _cpu._area(x[0]), reverse=True)
    results: list[dict] = []
    for box, prob, idx in box_list:
        tup = tuple(box)
        results.append(
            {
                "face_box": list(tup),
                "face_rect": normalize_face_box(tup, w, h),
                "confidence": prob,
            }
        )
    return results


def _detect_skin_condition_cuda(
    hwc: np.ndarray,
    face_results: list[dict],
    adjusted_gamma: float,
) -> list[dict]:
    if not face_results:
        return []
    h, w, _ = hwc.shape
    sess = _parse_session_cuda()
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
        tensor = _cpu._preprocess_skin(crop, adjusted_gamma)
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
            or (
                skin_pct <= 0.18
                and confidence < 0.93
                and (face_to_edge <= 0.15 or face_to_top < 0.02 or confidence < 0.7)
            )
            or (skin_pct <= 0.1 and (face_to_edge <= 0.2 or face_to_top < 0.04 or confidence < 0.8))
            or (
                skin_pct <= 0.23
                and confidence < 0.65
                and (face_to_edge <= 0.3 or face_to_top < 0.06 or confidence < 0.58)
            )
        ):
            continue
        out.append({**result, "skin_condition": cond, "skin_percent": skin_pct})
        parsed += 1
    return out


def detect_faces_cuda(hwc: np.ndarray, *, adjusted_gamma: float = 1.0) -> list[dict]:
    """Probe face detection + skin parsing via CUDA ONNX when available."""
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
    h, w, _ = hwc.shape
    results = _get_face_bounding_boxes_cuda(hwc, adjusted_gamma)
    if len(results) > 1:
        results = _cpu._filter_center_faces(results)
    results = _detect_skin_condition_cuda(hwc, results, adjusted_gamma)
    inverted_cpu = h > w and len(results) > 2 and all((r.get("confidence") or 0.0) >= 0.9 for r in results)
    if inverted_cpu and len(results) > 2:
        results.sort(
            key=lambda r: (r.get("confidence") or 0.0) * (r.get("skin_percent") or 0.0),
            reverse=True,
        )
        results = results[:2]
    return results
