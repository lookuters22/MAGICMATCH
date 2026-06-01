"""Image sizing aligned with Polarr exportImageForColorMatch / resizePixelDataToBitmap."""

from __future__ import annotations

import io

import numpy as np

NET_LONG_EDGE = 1600
NET_INPUT_SIZE = 256
REF_WEBP_QUALITY = 92
REF_BITMAP_JPEG_QUALITY = 92
WORKER_JPEG_QUALITY = 98


def jpeg_roundtrip(hwc: np.ndarray, *, quality: int = WORKER_JPEG_QUALITY) -> np.ndarray:
    """
    Probe getWorkerDecodableSourceFile: PNG/WebP/etc. → canvas JPEG q98 before AI worker.
    JPEG sources are passed through unchanged in probe; Comfy tensors are already decoded,
    so this normalizes lossless uploads to the worker feed the probe actually uses.
    """
    from PIL import Image

    hwc = np.clip(np.asarray(hwc, dtype=np.float32), 0.0, 1.0)
    arr = (hwc * 255.0).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="JPEG", quality=quality, subsampling=0)
    buf.seek(0)
    decoded = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0
    return np.clip(decoded, 0.0, 1.0)


def prepare_worker_bitmap_source(hwc: np.ndarray) -> np.ndarray:
    """Bitmap source normalization for probe worker parity (PNG → JPEG q98)."""
    return jpeg_roundtrip(hwc)


def fit_to_size(width: int, height: int, box: tuple[int, int]) -> tuple[int, int]:
    """Contain fit — same geometry as Polarr fitToSize (ceil dimensions)."""
    max_w, max_h = box
    scale = min(max_w / width, max_h / height)
    import math

    return max(1, math.ceil(width * scale)), max(1, math.ceil(height * scale))


def resize_hwc(
    hwc: np.ndarray,
    width: int,
    height: int,
    *,
    high_quality: bool = True,
    resample=None,
) -> np.ndarray:
    """Resize H×W×3 float RGB [0,1]. high_quality ≈ probe resizePixelDataToBitmap('high')."""
    from PIL import Image

    hwc = np.asarray(hwc, dtype=np.float32)
    arr = (np.clip(hwc, 0.0, 1.0) * 255.0).astype(np.uint8)
    if resample is None:
        resample = Image.Resampling.LANCZOS if high_quality else Image.Resampling.BILINEAR
    pil = Image.fromarray(arr, "RGB").resize((width, height), resample)
    return np.asarray(pil, dtype=np.float32) / 255.0


def fit_long_edge(
    hwc: np.ndarray,
    long_edge: int,
    *,
    high_quality: bool = True,
    resample=None,
) -> np.ndarray:
    """Contain-fit within long_edge×long_edge (Polarr fitToSize + exportImageData resizeMode=fit)."""
    from PIL import Image

    h, w, _ = hwc.shape
    nw, nh = fit_to_size(w, h, (long_edge, long_edge))
    if nw == w and nh == h:
        return np.asarray(hwc, dtype=np.float32).copy()
    if resample is None:
        resample = Image.Resampling.LANCZOS if high_quality else Image.Resampling.BILINEAR
    return resize_hwc(hwc, nw, nh, resample=resample)


def bitmap_shader_import(hwc: np.ndarray, scale: int = 2) -> np.ndarray:
    """
    BitmapIOAdapter + bitmap.frag at enableHalfResolution scale.

    texelFetch(source, fragCoord * scale + offset) with sRGB→ProPhoto→tonemapInv→linearToSRGB.
    """
    from ..polarr_color_space import MAT_SRGB_TO_XYZ, MAT_XYZ_TO_PROPHOTO, srgb_to_linear
    from .develop import _linear_to_gamma, _rgb_to_hue, _set_hue, _tonemap_inv

    hwc = np.clip(np.asarray(hwc, dtype=np.float32), 0.0, 1.0)
    h, w, _ = hwc.shape
    out_h, out_w = max(1, h // scale), max(1, w // scale)
    src_y = np.arange(out_h, dtype=np.int32) * scale
    src_x = np.arange(out_w, dtype=np.int32) * scale
    yy, xx = np.meshgrid(src_y, src_x, indexing="ij")
    rgb = hwc[yy, xx]

    flat = rgb.reshape(-1, 3)
    linear = srgb_to_linear(flat)
    xyz = (linear.astype(np.float64) @ MAT_SRGB_TO_XYZ.T).astype(np.float32)
    pro = (xyz.astype(np.float64) @ MAT_XYZ_TO_PROPHOTO.T).astype(np.float32)
    hue = _rgb_to_hue(pro)
    pro = _tonemap_inv(pro)
    pro = _set_hue(pro, hue)
    return _linear_to_gamma(np.maximum(pro, 0.0)).reshape(out_h, out_w, 3)


def _bilinear_at(hwc: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Sample hwc at floating (x, y) with bilinear filtering; xs/ys broadcast as (H, W)."""
    in_h, in_w, _ = hwc.shape
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, in_w - 1)
    y1 = np.clip(y0 + 1, 0, in_h - 1)
    x0 = np.clip(x0, 0, in_w - 1)
    y0 = np.clip(y0, 0, in_h - 1)
    fx = (xs - x0).astype(np.float32)
    fy = (ys - y0).astype(np.float32)
    out = np.empty(xs.shape + (3,), dtype=np.float32)
    for c in range(3):
        c00 = hwc[y0, x0, c]
        c01 = hwc[y0, x1, c]
        c10 = hwc[y1, x0, c]
        c11 = hwc[y1, x1, c]
        top = c00 * (1.0 - fx) + c01 * fx
        bot = c10 * (1.0 - fx) + c11 * fx
        out[..., c] = top * (1.0 - fy) + bot * fy
    return np.clip(out, 0.0, 1.0)


def bilinear_sample_grid(hwc: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Bilinear sample matching transform-pass center mapping from half-res inputTexture."""
    in_h, in_w, _ = hwc.shape
    xs = (np.arange(out_w, dtype=np.float32) + 0.5) / out_w * in_w - 0.5
    ys = (np.arange(out_h, dtype=np.float32) + 0.5) / out_h * in_h - 0.5
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    return _bilinear_at(hwc, xx, yy)


def _build_mipmap_chain(hwc: np.ndarray) -> list[np.ndarray]:
    """Box-filter mip chain matching BitmapIOAdapter inputTexture.generateMipmap()."""
    mips: list[np.ndarray] = [np.clip(np.asarray(hwc, dtype=np.float32), 0.0, 1.0)]
    while mips[-1].shape[0] > 1 or mips[-1].shape[1] > 1:
        prev = mips[-1]
        h, w, _ = prev.shape
        out_h, out_w = max(1, h // 2), max(1, w // 2)
        out = np.empty((out_h, out_w, 3), dtype=np.float32)
        for oy in range(out_h):
            y0, y1 = oy * 2, min((oy + 1) * 2, h)
            for ox in range(out_w):
                x0, x1 = ox * 2, min((ox + 1) * 2, w)
                out[oy, ox] = prev[y0:y1, x0:x1].mean(axis=(0, 1))
        mips.append(out)
    return mips


def mipmap_linear_sample_grid(hwc: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """
    LINEAR_MIPMAP_LINEAR-style downsample from half-res inputTexture.

    BitmapIOAdapter calls generateMipmap(); transform-correction samples with texture()
    at minification, lifting shadow bins vs naive bilinear level-0 sampling.
    """
    in_h, in_w, _ = hwc.shape
    if out_w >= in_w and out_h >= in_h:
        return bilinear_sample_grid(hwc, out_w, out_h)

    mips = _build_mipmap_chain(hwc)
    xs = (np.arange(out_w, dtype=np.float32) + 0.5) / out_w * in_w - 0.5
    ys = (np.arange(out_h, dtype=np.float32) + 0.5) / out_h * in_h - 0.5
    xx, yy = np.meshgrid(xs, ys, indexing="xy")

    level = float(np.log2(max(in_w / out_w, in_h / out_h)))
    l0 = int(np.floor(level))
    l1 = min(l0 + 1, len(mips) - 1)
    frac = level - l0
    scale0 = 2.0**l0
    scale1 = 2.0**l1
    s0 = _bilinear_at(mips[l0], xx / scale0, yy / scale0)
    s1 = _bilinear_at(mips[l1], xx / scale1, yy / scale1)
    return np.clip(s0 * (1.0 - frac) + s1 * frac, 0.0, 1.0)


def export_detection_srgb(
    half_res_hwc: np.ndarray,
    logical_width: int,
    logical_height: int,
    edge: int,
) -> np.ndarray:
    """
    exportImageData fit export on bitmap worker feed.

    Half-res bitmap.frag texture → mipmap-linear sample to fit size (probe GPU transform pass).
    """
    nw, nh = fit_to_size(logical_width, logical_height, (edge, edge))
    return mipmap_linear_sample_grid(half_res_hwc, nw, nh)


def transform_export_srgb(
    tone_map_inversed_hwc: np.ndarray,
    *,
    linear_gain: float = 1.0,
) -> np.ndarray:
    """
    Identity adjustments + display-output(srgb) on transform-pass upsample.

    Bitmap detection export stores tone-map-inversed gamma-encoded working-space values in
    inputTexture; adjustments/display undo import tonemapInv (renderer-cpu.ts renderPixel).
    """
    from ..polarr_color_space import MAT_PROPHOTO_TO_SRGB, linear_to_srgb, srgb_to_linear
    from .develop import _tonemap

    hwc = np.clip(np.asarray(tone_map_inversed_hwc, dtype=np.float32), 0.0, 1.0)
    rgb = srgb_to_linear(hwc.reshape(-1, 3).astype(np.float64))
    rgb = _tonemap(rgb.astype(np.float32)).astype(np.float64)
    if linear_gain != 1.0:
        rgb *= linear_gain
    out = linear_to_srgb(np.clip(rgb @ MAT_PROPHOTO_TO_SRGB.T, 0.0, 1.0))
    return np.clip(out.reshape(hwc.shape), 0.0, 1.0).astype(np.float32)


def render_detection_export(
    worker_feed_hwc: np.ndarray,
    edge: int,
    *,
    half_res_hwc: np.ndarray | None = None,
) -> np.ndarray:
    """Full probe renderDetectionInputs export chain for one fit edge."""
    feed = np.clip(np.asarray(worker_feed_hwc, dtype=np.float32), 0.0, 1.0)
    logical_h, logical_w, _ = feed.shape
    half = half_res_hwc if half_res_hwc is not None else bitmap_shader_import(feed, scale=2)
    upsampled = export_detection_srgb(half, logical_w, logical_h, edge)
    # Large detection buffer only: probe GPU readback lifts face p99 ~one histogram bin.
    gain = 1.008 if edge >= 2000 else 1.0
    return transform_export_srgb(upsampled, linear_gain=gain)


def webp_roundtrip(hwc: np.ndarray, *, quality: int = REF_WEBP_QUALITY) -> np.ndarray:
    """Match probe referenceWebpData256x256 encode/decode (bitmapToWebPData q92)."""
    from PIL import Image

    hwc = np.clip(np.asarray(hwc, dtype=np.float32), 0.0, 1.0)
    arr = (hwc * 255.0).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="WEBP", quality=quality)
    buf.seek(0)
    decoded = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0
    return np.clip(decoded, 0.0, 1.0)


def prepare_net_reference(reference_hwc: np.ndarray) -> np.ndarray:
    """
    Probe getReferenceColorMatchFeaturesAndSetCache when ImageBitmap is passed:
    full-res JPEG q92 (resizeImageBitmapToBitmapUsingBlob) → 256 high-quality resize
    → WebP q92 round-trip (worker cache feed).
    """
    decoded = jpeg_roundtrip(reference_hwc, quality=REF_BITMAP_JPEG_QUALITY)
    ref256 = resize_hwc(decoded, NET_INPUT_SIZE, NET_INPUT_SIZE, high_quality=True)
    return webp_roundtrip(ref256)
