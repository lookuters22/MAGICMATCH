# MAGICMATCH

ComfyUI custom nodes for **neural color match**: two images in, graded source out. Tune strength on a preview node before you wire the result to save/export.

## Install

1. Clone into ComfyUI `custom_nodes`:

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/lookuters22/MAGICMATCH.git
   ```

2. Install Python deps (ComfyUI’s environment):

   ```bash
   pip install -r MAGICMATCH/requirements.txt
   ```

3. Model files are included in the repo (pull gets everything):

   ```text
   MAGICMATCH/models/color_match.onnx
   MAGICMATCH/models/face/face_detect_landscape.onnx
   MAGICMATCH/models/face/face_detect_portrait.onnx
   MAGICMATCH/models/face/face_parse.onnx
   ```

   To regenerate locally (optional), use Python 3.12 + `scripts/convert_face_models_to_onnx.py` and the color-match convert script under `polarrnext/color_match_extract/`.

4. Restart ComfyUI. Nodes appear under **MAGICMATCH**.

## Workflow (strength before export)

Use **Build** + **Preview** so changing the slider only re-applies the LUT (fast):

```text
[Source] ──┬──► MagicMatch Build LUT ◄── [Reference]
           │              │
           │              ▼ lut
           └──────► MagicMatch Preview (strength) ──► Preview Image
                                         │
                                         └──► Save / export when ready
```

| Step | Node |
|------|------|
| 1 | **MagicMatch Build LUT** — run once per source/reference pair |
| 2 | **MagicMatch Preview (strength)** — run once, then use the **live preview inside the node** and drag **strength** (no re-queue) |
| 3 | When happy with the slider, **Queue** again and connect output to Save / export |

### In-node live preview

1. Run the workflow **once** (Build + Preview execute; LUT is cached).
2. Open the **MagicMatch Preview** node — you’ll see a **live image** under the widgets.
3. Drag **strength** — preview updates **instantly** in the node (WebGL, no workflow re-run).
4. When the look is right, **Queue Prompt** once more so the **image** output matches the slider, then save/export.

Live preview uses WebGL (merged 25³ LUT + strength mix). Mid-strength may differ slightly from the queued **image** output, which uses the full CPU path — queue once more before export.

**MagicMatch (one-shot)** combines build + apply in one node (simpler, slower when you change strength).

## Requirements

- ComfyUI with standard `IMAGE` tensors (batch size **1**)
- `onnxruntime`, `numpy` (see `requirements.txt`)
- Face auto-WB/light uses ONNX face models under `models/face/` (see install step 3)
- CPU inference by default (~5–15s for Build on first run per pair)

## CUDA inference (experimental)

A **separate** GPU path lives alongside the CPU parity stack — default nodes and
`scripts/parity_pair.py` golden checks are unchanged.

### Install (H100 / Linux)

```bash
# In ComfyUI's venv — onnxruntime and onnxruntime-gpu are mutually exclusive on many builds
pip uninstall -y onnxruntime
pip install -r MAGICMATCH/requirements-cuda.txt
```

Requires NVIDIA driver + CUDA libs compatible with your `onnxruntime-gpu` wheel.
ComfyUI already provides PyTorch with CUDA — the GPU pipeline uses Torch for develop,
detection buffers, and LUT apply (ONNX uses CUDA EP).

On machines without CUDA, CUDA nodes fall back to CPU EP + CPU Torch automatically.

### ComfyUI nodes

After restart, look under **MAGICMATCH/CUDA (experimental)**:

| Class | Display name |
|-------|----------------|
| `MagicMatchBuildCUDA` | MagicMatch Build LUT (CUDA) |
| `MagicMatchPreviewCUDA` | MagicMatch Preview (CUDA LUT) |
| `MagicMatchCUDA` | MagicMatch one-shot (CUDA) |

**GPU pipeline (CUDA nodes):** CUDA ONNX + GPU detection buffers + GPU develop@1600 + GPU full-res apply.

**CPU default nodes** under **MAGICMATCH** are unchanged.

Set `MAGICMATCH_CUDA_NODES=0` before starting ComfyUI to hide CUDA nodes.

Use `python3.12` on RunPod if `python` is not on PATH.

### Benchmark / parity

```bash
# CPU golden (must stay parity_ok)
python scripts/parity_pair.py \
  ../polarrnext/standalone_probe/public/pair/source.png \
  ../polarrnext/standalone_probe/public/pair/reference.jpg

# CPU vs CUDA timings + lut_hash compare
python scripts/bench_cuda_vs_cpu.py \
  ../polarrnext/standalone_probe/public/pair/source.png \
  ../polarrnext/standalone_probe/public/pair/reference.jpg
```

Expected golden `lut_hash` on the polarrnext/pair test set: `a48758ca22a2e389`.
CUDA may differ slightly if GPU face-detect scores diverge from CPU f16 quirks; compare
`lut_max_abs_delta` in the bench report.

## Nodes (CPU default)

| Class | Display name |
|-------|----------------|
| `MagicMatchBuild` | MagicMatch Build LUT |
| `MagicMatchPreview` | MagicMatch Preview (strength) |
| `MagicMatch` | MagicMatch (one-shot) |

## License

See [LICENSE](LICENSE). The bundled `color_match.onnx` is part of this package; use responsibly and in compliance with applicable terms for any upstream model you obtained.
