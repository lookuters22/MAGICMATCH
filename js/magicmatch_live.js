/**
 * In-node live preview — WebGL merged-LUT (same reshape as CPU path).
 */
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const PREVIEW_CLASS = "MagicMatchPreview";
const LUT_SIZE = 25;
const MIN_PREVIEW_W = 80;

function b64ToArrayBuffer(b64) {
  const bin = atob(b64);
  const buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return buf.buffer;
}

/** RGBc merged 25³ → cBGR flat (same as reshape_merged_lut_for_apply in lut.py) */
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

/** cBGR flat → 3D texel (x=B,y=G,z=R) with RGB = output R,G,B at that voxel */
function reshapedLutTo3DUint8(reshaped) {
  const size = LUT_SIZE;
  const n = size * size * size;
  const tex = new Uint8Array(n * 4);
  for (let b = 0; b < size; b++) {
    for (let g = 0; g < size; g++) {
      for (let r = 0; r < size; r++) {
        const idx = b * size * size + g * size + r;
        const dst = idx * 4;
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
    this.gl = null;
    this.prog = null;
    this.lutTex = null;
    this.srcTex = null;
    this.uStrength = null;
    this.uSrc = null;
    this.uLut = null;
    this.displayW = 200;
    this.displayH = 200;
    this.aspect = 1;
    this._srcCanvas = document.createElement("canvas");

    this.wrap = document.createElement("div");
    this.wrap.style.cssText =
      "width:100%;margin-top:6px;overflow:hidden;box-sizing:border-box;";

    this.hint = document.createElement("div");
    this.hint.textContent = "Run workflow once → live slider preview";
    this.hint.style.cssText = "font-size:11px;color:#999;margin-bottom:4px;";

    this.canvas = document.createElement("canvas");
    this.canvas.style.cssText =
      "display:block;width:100%;height:auto;background:#1a1a1a;border-radius:4px;";

    this.wrap.appendChild(this.hint);
    this.wrap.appendChild(this.canvas);
  }

  initGl() {
    if (this.gl) return true;
    const gl = this.canvas.getContext("webgl2", { premultipliedAlpha: false });
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

  layoutFromNodeWidth(nodeWidth) {
    const w = Math.max(MIN_PREVIEW_W, Math.floor(nodeWidth - 24));
    const h = Math.max(MIN_PREVIEW_W, Math.round(w / this.aspect));
    this.displayW = w;
    this.displayH = h;
    return [w, h + 32];
  }

  uploadLut(gl, cache) {
    const floats = new Float32Array(b64ToArrayBuffer(cache.lut));
    const reshaped = reshapeMergedLutForApply(floats);
    const texData = reshapedLutTo3DUint8(reshaped);

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
    if (!this._img) return;
    const ctx = this._srcCanvas.getContext("2d");
    this._srcCanvas.width = this.displayW;
    this._srcCanvas.height = this.displayH;
    ctx.drawImage(this._img, 0, 0, this.displayW, this.displayH);

    gl.bindTexture(gl.TEXTURE_2D, this.srcTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      this._srcCanvas,
    );
  }

  refreshLayout() {
    if (!this.cache || !this.node.size) return;
    this.layoutFromNodeWidth(this.node.size[0]);
    this.canvas.width = this.displayW;
    this.canvas.height = this.displayH;
    if (this.gl) {
      this.uploadSource(this.gl);
      this.render(this.getStrength());
    }
  }

  setCache(cache) {
    this.cache = cache;
    this.aspect = cache.w / cache.h;
    if (!this.initGl()) return;

    const gl = this.gl;
    this.uploadLut(gl, cache);

    const img = new Image();
    img.onload = () => {
      this._img = img;
      this.hint.textContent = "Live preview (slider — no re-queue)";
      this.refreshLayout();
    };
    img.src = "data:image/png;base64," + cache.src_png;
  }

  getStrength() {
    const w = this.node.widgets?.find((x) => x.name === "strength");
    return w ? Number(w.value) : 1.0;
  }

  render(strength) {
    if (!this.cache || !this.gl || !this.prog) return;
    const gl = this.gl;
    gl.viewport(0, 0, this.displayW, this.displayH);
    gl.useProgram(this.prog);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.srcTex);
    gl.uniform1i(this.uSrc, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_3D, this.lutTex);
    gl.uniform1i(this.uLut, 1);
    gl.uniform1f(this.uStrength, Math.max(0, Math.min(1, strength)));
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

    domWidget.computeSize = function (width) {
      if (panel.cache) {
        return panel.layoutFromNodeWidth(width);
      }
      return [width, 200];
    };

    const origOnResize = node.onResize;
    node.onResize = function (size) {
      const res = origOnResize?.apply(this, arguments);
      panel.refreshLayout();
      return res;
    };

    const strengthWidget = node.widgets?.find((w) => w.name === "strength");
    if (strengthWidget) {
      const prev = strengthWidget.callback;
      strengthWidget.callback = function (v, ...rest) {
        if (prev) prev.call(this, v, ...rest);
        panel.render(Number(v));
      };
    }
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

      getPanel(graphNode).setCache(out.magicmatch_live[0]);
      const sw = graphNode.widgets?.find((w) => w.name === "strength");
      if (sw) getPanel(graphNode).render(Number(sw.value));
    });
  },
});
