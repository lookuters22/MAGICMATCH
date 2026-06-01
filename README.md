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

3. Confirm the model file is present:

   ```text
   MAGICMATCH/models/color_match.onnx
   ```

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
- CPU inference by default (~5–15s for Build on first run per pair)

## Nodes

| Class | Display name |
|-------|----------------|
| `MagicMatchBuild` | MagicMatch Build LUT |
| `MagicMatchPreview` | MagicMatch Preview (strength) |
| `MagicMatch` | MagicMatch (one-shot) |

## License

See [LICENSE](LICENSE). The bundled `color_match.onnx` is part of this package; use responsibly and in compliance with applicable terms for any upstream model you obtained.
