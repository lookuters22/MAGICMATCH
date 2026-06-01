"""Fast GPU scene extract — speed over strict CPU parity."""

from __future__ import annotations

import torch

from ..probe_parity.calibration_wb import EstimatedIlluminant, get_auto_wb_params_for_color_match
from ..probe_parity.luminance import LuminanceStatistics, get_adjusted_gamma, get_auto_light_params
from ..probe_parity.scene_extract import (
    BITMAP_AS_SHOT_TEMP,
    BITMAP_AS_SHOT_TINT,
    SceneInfo,
    get_original_color_match_base_adjustments,
)
from .device import gpu_pipeline_available, hwc_torch_to_numpy
from .luminance_torch import get_luminance_statistics_torch
from .reference_torch import render_detection_inputs_torch

_RGB_TO_YUV = torch.tensor(
    [[0.299, 0.587, 0.114], [-0.299, -0.587, 0.886], [0.701, -0.587, -0.114]],
    dtype=torch.float32,
)
_YUV_TO_RGB = torch.linalg.inv(_RGB_TO_YUV)


@torch.inference_mode()
def _gray_world_illuminant_fast_torch(small_hwc: torch.Tensor) -> EstimatedIlluminant:
    """Gray-world on detection small buffer; syncs only final 3-vector illuminant."""
    rgb = torch.clamp(small_hwc.reshape(-1, 3), 0.0, 1.0)
    yuv = rgb @ _RGB_TO_YUV.to(device=rgb.device, dtype=rgb.dtype).T
    y, u, v = yuv.unbind(-1)
    f_vals = (u.abs() + v.abs()) / torch.clamp(y, min=1e-5)
    grays = yuv[f_vals < 0.1321]
    gray_pct = float(grays.shape[0]) / max(float(rgb.shape[0]), 1.0)
    if gray_pct < 0.008:
        grays = yuv[f_vals < 0.16]
        if float(grays.shape[0]) / max(float(rgb.shape[0]), 1.0) < 0.001:
            grays = yuv
    if grays.numel():
        u_bar = grays[:, 1].mean()
        v_bar = grays[:, 2].mean()
    else:
        u_bar = torch.tensor(0.0, device=rgb.device, dtype=rgb.dtype)
        v_bar = torch.tensor(0.0, device=rgb.device, dtype=rgb.dtype)
    yuv_mean = torch.stack(
        [torch.tensor(100.0 / 255.0, device=rgb.device, dtype=rgb.dtype), u_bar, v_bar]
    )
    illuminant = (yuv_mean @ _YUV_TO_RGB.to(device=rgb.device, dtype=rgb.dtype).T).cpu().numpy()
    return EstimatedIlluminant(
        overall=illuminant,
        image=illuminant,
        face=None,
        skin_weight=0.0,
    )


@torch.inference_mode()
def extract_scene_info_bitmap_fast_gpu(
    feed_tensor: torch.Tensor,
    *,
    as_shot_temperature: float = BITMAP_AS_SHOT_TEMP,
    as_shot_tint: float = BITMAP_AS_SHOT_TINT,
) -> tuple[SceneInfo, torch.Tensor, torch.Tensor]:
    """Fast GPU scene extract: detection buffers + face ONNX stay on device."""
    if not gpu_pipeline_available():
        from ..probe_parity.scene_extract import extract_scene_info_bitmap

        feed_np = hwc_torch_to_numpy(feed_tensor)
        return extract_scene_info_bitmap(feed_np, worker_feed_prepared=True), feed_tensor, feed_tensor

    small_t, large_t = render_detection_inputs_torch(feed_tensor)
    stats_noface = get_luminance_statistics_torch(small_t, large_t, [])
    adjusted_gamma = get_adjusted_gamma(stats_noface)

    from ..probe_parity.face_detection_cuda import detect_faces_cuda_from_tensor

    face_results = detect_faces_cuda_from_tensor(large_t, adjusted_gamma=adjusted_gamma)
    stats = get_luminance_statistics_torch(small_t, large_t, face_results)
    auto_light = get_auto_light_params(stats, face_hsvl=None)

    illuminant = _gray_world_illuminant_fast_torch(small_t)
    auto_wb = get_auto_wb_params_for_color_match(
        illuminant,
        as_shot_temperature=as_shot_temperature,
        as_shot_tint=as_shot_tint,
    )

    base = {**auto_light, "temperature": auto_wb["temperature"], "tint": auto_wb["tint"]}
    scene = SceneInfo(
        auto_light_params=auto_light,
        auto_wb_params=auto_wb,
        base_adjustments=base,
        face_results=face_results,
        avg_face_hsvl=None,
    )
    return scene, small_t, large_t


def estimate_base_adjustments_fast_gpu(feed_tensor: torch.Tensor) -> dict[str, float]:
    scene, _, _ = extract_scene_info_bitmap_fast_gpu(feed_tensor)
    return get_original_color_match_base_adjustments(scene)
