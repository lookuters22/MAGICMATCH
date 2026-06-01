/**
 * In-node live preview — WebGL merged-LUT (same reshape as CPU path).
 * Backing store stays at cache resolution; CSS scales with the node.
 */
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const PREVIEW_CLASS = "MagicMatchPreview";
const LUT_SIZE = 25;
const MIN_PREVIEW_PX = 64;
const NODE_CHROME_H = 118;

function b64ToArrayBuffer(b64) {
  const bin = atob(b64);
  const buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return buf.buffer;
}

/** RGBc merged 25³ → cBGR flat (reshape_merged_lut_for_apply in lut.py) */
function reshapeMergedLutForApply(mergedRgb) {
  const size = LUT_SIZE;
  const merged = mergedRgb;
  const reshaped = new Float32Array(size * size * size * 3);
  for (let r = 0; r < size; r++) {
    for (let g = 0; g < size; g++) {
      for (let b = 0; b < size; b++) {
        for (let c = 0; c < 3; c++) {
          const src = (r * size * size + g * size + b) * 3 + c;
          const dst = c * size * size * size + b * size * size + g * size + r;
          reshaped[dst] = merged[src];
        }
      }
    }
  }
  return reshaped;
}

/** cBGR flat → 3D texels (OpenGL x=B fastest) */
function reshapedLutTo3DUint8(reshaped) {
  const size = LUT_SIZE;
  const n = size * size * size;
  const tex = new Uint8Array(n * 4);
  for (let b = 0; b < size; b++) {
    for (let g = 0; g < size; g++) {
      for (let r = 0; r < size; r++) {
        const idx = b * size * size + g * size + r;
        const dst = (b + g * size + r * size * size) * 4;
        tex[dst] = byte(reshaped[0 * n + idx]);
        tex[dst + 1] = byte(reshaped[1 * n + idx]);
        tex[dst + 2] = byte(reshaped[2 * n + idx]);
        tex[dst + 3] = 255;
      }
    }
  }
  return tex;
}

function byte(v) {
  return Math.round(Math.max(0, Math.min(1, v)) * 255);
}

class LivePreviewPanel {
  constructor(node) {
    this.node = node;
    this.cache = null;
    this._img = null;
    this._reshapedLut = null;
    this.gl = null;
    this.prog = null;
    this.lutTex = null;
    this.srcTex = null;
    this.uStrength = null;
    this.uSrc = null;
    this.uLut = null;
    this.displayW = 200;
    this.displayH = 200;
    this.imgW = 0;
    this.imgH = 0;
    this.aspect = 1;
    this._glBw = 0;
    this._glBh = 0;
    this._srcCanvas = document.createElement("canvas");

    this.wrap = document.createElement("div");
    this.wrap.style.cssText =
      "width:100%;margin-top:6px;overflow:visible;box-sizing:border-box;";

    this.hint = document.createElement("div");
    this.hint.textContent = "Run workflow once → live slider preview";
    this.hint.style.cssText = "font-size:11px;color:#999;margin-bottom:4px;";

    this.canvas = document.createElement("canvas");
    this.canvas.style.cssText =
      "display:block;max-width:100%;background:#1a1a1a;border-radius:4px;image-rendering:auto;";
  }

  resetGl() {
    this.gl = null;
    this.prog = null;
    this.lutTex = null;
    this.srcTex = null;
    this.uStrength = null;
    this.uSrc = null;
    this.uLut = null;
  }

  initGl() {
    if (this.gl) return true;
    const gl = this.canvas.getContext("webgl2", {
      premultipliedAlpha: false,
      preserveDrawingBuffer: true,
    });
    if (!gl) {
      this.hint.textContent = "WebGL2 required for live preview";
      return false;
    }
    this.gl = gl;

    const vs = `#version 300 es
    in vec2 a_pos;
    out vec2 v_uv;
    void main() {
      v_uv = vec2(a_pos.x * 0.5 + 0.5, 1.0 - (a_pos.y * 0.5 + 0.5));
      gl_Position = vec4(a_pos, 0.0, 1.0);
    }`;

    const fs = `#version 300 es
    precision highp float;
    precision highp sampler3D;
    in vec2 v_uv;
    out vec4 outColor;
    uniform sampler2D u_src;
    uniform sampler3D u_lut;
    uniform float u_strength;
    void main() {
      vec3 src = texture(u_src, v_uv).rgb;
      vec3 coord = vec3(src.b, src.g, src.r);
      vec3 mapped = texture(u_lut, coord).rgb;
      vec3 out_rgb = mix(src, mapped, u_strength);
      outColor = vec4(clamp(out_rgb, 0.0, 1.0), 1.0);
    }`;

    const compile = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
        console.error("[MAGICMATCH] shader", gl.getShaderInfoLog(s));
        return null;
      }
      return s;
    };

    const vsObj = compile(gl.VERTEX_SHADER, vs);
    const fsObj = compile(gl.FRAGMENT_SHADER, fs);
    if (!vsObj || !fsObj) return false;

    const prog = gl.createProgram();
    gl.attachShader(prog, vsObj);
    gl.attachShader(prog, fsObj);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      console.error("[MAGICMATCH] link", gl.getProgramInfoLog(prog));
      return false;
    }
    this.prog = prog;
    this.uStrength = gl.getUniformLocation(prog, "u_strength");
    this.uSrc = gl.getUniformLocation(prog, "u_src");
    this.uLut = gl.getUniformLocation(prog, "u_lut");

    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
      gl.STATIC_DRAW,
    );
    const loc = gl.getAttribLocation(prog, "a_pos");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

    this.lutTex = gl.createTexture();
    this.srcTex = gl.createTexture();
    return true;
  }

  /** CSS display size from decoded image aspect (width / height). */
  layoutPreview(nodeWidth, nodeHeight) {
    const aspect = this.imgW > 0 && this.imgH > 0 ? this.imgW / this.imgH : this.aspect;
    if (!aspect || aspect <= 0) return [nodeWidth, 200];

    const pad = 16;
    const availW = Math.max(MIN_PREVIEW_PX, Math.floor(nodeWidth - pad));
    let w = availW;
    let h = Math.max(MIN_PREVIEW_PX, Math.round(w / aspect));

    if (nodeHeight) {
      const availH = Math.max(MIN_PREVIEW_PX, Math.floor(nodeHeight - NODE_CHROME_H));
      if (availH > h) {
        h = availH;
        w = Math.max(MIN_PREVIEW_PX, Math.round(h * aspect));
        if (w > availW) {
          w = availW;
          h = Math.max(MIN_PREVIEW_PX, Math.round(w / aspect));
        }
      }
    }

    this.displayW = w;
    this.displayH = h;
    this.canvas.style.width = `${w}px`;
    this.canvas.style.height = `${h}px`;
    return [nodeWidth, h + NODE_CHROME_H];
  }

  /** Backing store = cache pixels × DPR (only changes when cache size changes). */
  sourceDimensions() {
    const w = this.imgW > 0 ? this.imgW : this.cache?.w ?? 1;
    const h = this.imgH > 0 ? this.imgH : this.cache?.h ?? 1;
    return { w, h };
  }

  syncGlBacking() {
    if (!this.cache) return false;
    const { w: srcW, h: srcH } = this.sourceDimensions();
    const dpr = window.devicePixelRatio || 1;
    const bw = Math.max(1, Math.round(srcW * dpr));
    const bh = Math.max(1, Math.round(srcH * dpr));
    if (bw === this._glBw && bh === this._glBh && this.gl) return false;
    this._glBw = bw;
    this._glBh = bh;
    if (this.canvas.width !== bw || this.canvas.height !== bh) {
      this.canvas.width = bw;
      this.canvas.height = bh;
      this.resetGl();
    }
    return true;
  }

  ensureGlReady() {
    if (!this.cache) return false;
    this.syncGlBacking();
    return this.initGl();
  }

  prepareLutData(cache) {
    const floats = new Float32Array(b64ToArrayBuffer(cache.lut));
    this._reshapedLut = reshapeMergedLutForApply(floats);
    return reshapedLutTo3DUint8(this._reshapedLut);
  }

  uploadLut(gl) {
    if (!this.cache) return;
    const texData = this._lutTexData || this.prepareLutData(this.cache);
    this._lutTexData = texData;

    gl.bindTexture(gl.TEXTURE_3D, this.lutTex);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
    gl.texImage3D(
      gl.TEXTURE_3D,
      0,
      gl.RGBA,
      LUT_SIZE,
      LUT_SIZE,
      LUT_SIZE,
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      texData,
    );
  }

  uploadSource(gl) {
    if (!this._img || !this.cache) return;
    const { w: srcW, h: srcH } = this.sourceDimensions();
    this._srcCanvas.width = srcW;
    this._srcCanvas.height = srcH;
    const ctx = this._srcCanvas.getContext("2d");
    ctx.drawImage(this._img, 0, 0, srcW, srcH);

    gl.bindTexture(gl.TEXTURE_2D, this.srcTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      srcW,
      srcH,
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      this._srcCanvas,
    );
  }

  refreshLayout() {
    if (!this.cache) return;
    const nw = this.node.size?.[0] ?? 320;
    const nh = this.node.size?.[1];
    this.layoutPreview(nw, nh);
    this.drawPreview(this.getStrength());
    if (this.node.setSize) {
      this.node.setSize(this.node.computeSize());
    }
  }

  setCache(cache) {
    this.cache = cache;
    this.aspect = cache.w / cache.h;
    this._lutTexData = null;
    this._reshapedLut = null;
    this._glBw = 0;
    this._glBh = 0;
    this.resetGl();

    const img = new Image();
    img.onload = () => {
      this._img = img;
      this.hint.textContent = "Live preview (slider — no re-queue)";
      this.refreshLayout();
    };
    img.onerror = () => {
      this.hint.textContent = "Preview source failed to load";
      console.error("[MAGICMATCH] failed to decode preview PNG");
    };
    img.src = "data:image/png;base64," + cache.src_png;
  }

  getStrength() {
    const w = this.node.widgets?.find((x) => x.name === "strength");
    const v = w != null ? w.value : 1.0;
    const n = Number(v);
    return Number.isFinite(n) ? Math.max(0, Math.min(1, n)) : 1.0;
  }

  drawPreview(strength) {
    if (!this.ensureGlReady()) return;
    const gl = this.gl;
    this.uploadLut(gl);
    if (this._img) this.uploadSource(gl);
    this.render(strength);
  }

  render(strength) {
    if (!this.cache || !this.gl || !this.prog) return;
    const gl = this.gl;
    const s = Math.max(0, Math.min(1, Number(strength)));
    if (!Number.isFinite(s)) return;

    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0.1, 0.1, 0.1, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(this.prog);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.srcTex);
    gl.uniform1i(this.uSrc, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_3D, this.lutTex);
    gl.uniform1i(this.uLut, 1);
    gl.uniform1f(this.uStrength, s);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }
}

const panels = new WeakMap();

function getPanel(node) {
  let p = panels.get(node);
  if (!p) {
    p = new LivePreviewPanel(node);
    panels.set(node, p);
  }
  return p;
}

function bindStrengthSlider(node, panel) {
  const strengthWidget = node.widgets?.find((w) => w.name === "strength");
  if (!strengthWidget || strengthWidget._magicmatchBound) return;
  strengthWidget._magicmatchBound = true;

  const prev = strengthWidget.callback;
  strengthWidget.callback = function (...args) {
    if (prev) prev.apply(this, args);
    panel.drawPreview(panel.getStrength());
  };
}

app.registerExtension({
  name: "MAGICMATCH.live_in_node",

  async nodeCreated(node) {
    if (node.comfyClass !== PREVIEW_CLASS) return;

    const panel = getPanel(node);
    const domWidget = node.addDOMWidget("live_preview", "magicmatch_preview", panel.wrap, {
      getValue() {
        return panel.cache;
      },
      setValue() {},
      serialize: false,
    });

    panel.wrap.appendChild(panel.hint);
    panel.wrap.appendChild(panel.canvas);

    domWidget.computeSize = function (width) {
      if (panel.imgW > 0 && panel.imgH > 0) {
        return panel.layoutPreview(width, panel.node.size?.[1]);
      }
      if (panel.cache?.w && panel.cache?.h) {
        panel.aspect = panel.cache.w / panel.cache.h;
        return panel.layoutPreview(width, panel.node.size?.[1]);
      }
      return [width, 200];
    };

    const origOnResize = node.onResize;
    node.onResize = function (size) {
      const res = origOnResize?.apply(this, arguments);
      if (panel.imgW > 0) {
        panel.layoutPreview(size[0], size[1]);
        panel.drawPreview(panel.getStrength());
      }
      return res;
    };

    bindStrengthSlider(node, panel);
  },

  async setup() {
    api.addEventListener("executed", ({ detail }) => {
      const out = detail?.output;
      if (!out?.magicmatch_live?.length) return;

      const nodeId = detail.display_node ?? detail.node;
      const graphNode =
        app.graph.getNodeById?.(nodeId) ??
        app.graph._nodes_by_id?.[nodeId];
      if (!graphNode || graphNode.comfyClass !== PREVIEW_CLASS) return;

      const panel = getPanel(graphNode);
      bindStrengthSlider(graphNode, panel);
      panel.setCache(out.magicmatch_live[0]);
      panel.drawPreview(panel.getStrength());
    });
  },
});
