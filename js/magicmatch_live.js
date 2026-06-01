/**
 * In-node live preview — WebGL merged-LUT apply (matches CPU path after first run).
 */
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const PREVIEW_CLASS = "MagicMatchPreview";
const LUT_SIZE = 25;
const MAX_DISPLAY_W = 280;
const MAX_DISPLAY_H = 200;

function b64ToArrayBuffer(b64) {
  const bin = atob(b64);
  const buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return buf.buffer;
}

/** RGBc merged LUT → 3D texture texels (x=B, y=G, z=R), RGBA8 for broad GPU support */
function mergedLutTo3DUint8(floats) {
  const size = LUT_SIZE;
  const tex = new Uint8Array(size * size * size * 4);
  for (let r = 0; r < size; r++) {
    for (let g = 0; g < size; g++) {
      for (let b = 0; b < size; b++) {
        const src = (r * size * size + g * size + b) * 3;
        const dst = (b * size * size + g * size + r) * 4;
        tex[dst] = Math.round(Math.max(0, Math.min(1, floats[src])) * 255);
        tex[dst + 1] = Math.round(Math.max(0, Math.min(1, floats[src + 1])) * 255);
        tex[dst + 2] = Math.round(Math.max(0, Math.min(1, floats[src + 2])) * 255);
        tex[dst + 3] = 255;
      }
    }
  }
  return tex;
}

class LivePreviewPanel {
  constructor(node) {
    this.node = node;
    this.cache = null;
    this.gl = null;
    this.prog = null;
    this.lutTex = null;
    this.srcTex = null;
    this.uStrength = null;
    this.uSrc = null;
    this.uLut = null;
    this.displayW = MAX_DISPLAY_W;
    this.displayH = MAX_DISPLAY_H;
    this._srcCanvas = document.createElement("canvas");

    this.wrap = document.createElement("div");
    this.wrap.style.cssText =
      "width:100%;margin-top:6px;overflow:hidden;box-sizing:border-box;";

    this.hint = document.createElement("div");
    this.hint.textContent = "Run workflow once → live slider preview";
    this.hint.style.cssText = "font-size:11px;color:#999;margin-bottom:4px;";

    this.canvas = document.createElement("canvas");
    this.canvas.style.cssText =
      "display:block;width:100%;max-height:" +
      MAX_DISPLAY_H +
      "px;background:#1a1a1a;border-radius:4px;object-fit:contain;";

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

  _resizeNode() {
    const h = this.displayH + 32;
    if (this.node.setSize && this.node.size) {
      this.node.setSize([this.node.size[0], h]);
    }
    this.node.onResize?.(this.node.size);
  }

  setCache(cache) {
    this.cache = cache;
    if (!this.initGl()) return;

    const gl = this.gl;
    const floats = new Float32Array(b64ToArrayBuffer(cache.lut));
    const texData = mergedLutTo3DUint8(floats);

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

    const img = new Image();
    img.onload = () => {
      const scale = Math.min(
        1,
        MAX_DISPLAY_W / img.width,
        MAX_DISPLAY_H / img.height,
      );
      this.displayW = Math.max(1, Math.round(img.width * scale));
      this.displayH = Math.max(1, Math.round(img.height * scale));

      this.canvas.width = this.displayW;
      this.canvas.height = this.displayH;
      this._resizeNode();

      const ctx = this._srcCanvas.getContext("2d");
      this._srcCanvas.width = this.displayW;
      this._srcCanvas.height = this.displayH;
      ctx.drawImage(img, 0, 0, this.displayW, this.displayH);

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

      this.hint.textContent = "Live preview (slider — no re-queue)";
      this.render(this.getStrength());
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
      return [Math.min(width, 320), panel.displayH + 32];
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
