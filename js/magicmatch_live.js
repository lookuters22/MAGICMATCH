/**
 * In-node live preview for MagicMatch Preview (WebGL LUT + strength slider).
 * Run the workflow once to cache; then drag strength without re-queueing.
 */
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const PREVIEW_CLASS = "MagicMatchPreview";
const LUT_SIZE = 25;

function b64ToArrayBuffer(b64) {
  const bin = atob(b64);
  const buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return buf.buffer;
}

function mergedLutTo3DTextureData(floats) {
  const size = LUT_SIZE;
  const tex = new Float32Array(size * size * size * 3);
  for (let r = 0; r < size; r++) {
    for (let g = 0; g < size; g++) {
      for (let b = 0; b < size; b++) {
        const src = (r * size * size + g * size + b) * 3;
        const dst = (b * size * size + g * size + r) * 3;
        tex[dst] = floats[src];
        tex[dst + 1] = floats[src + 1];
        tex[dst + 2] = floats[src + 2];
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
    this.previewH = 200;

    this.wrap = document.createElement("div");
    this.wrap.style.width = "100%";
    this.wrap.style.marginTop = "6px";

    this.hint = document.createElement("div");
    this.hint.textContent = "Run workflow once → live slider preview";
    this.hint.style.cssText = "font-size:11px;color:#999;margin-bottom:4px;";

    this.canvas = document.createElement("canvas");
    this.canvas.style.width = "100%";
    this.canvas.style.height = "auto";
    this.canvas.style.display = "block";
    this.canvas.style.background = "#1a1a1a";
    this.canvas.style.borderRadius = "4px";

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
      v_uv = a_pos * 0.5 + 0.5;
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
      outColor = vec4(out_rgb, 1.0);
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

  setCache(cache) {
    this.cache = cache;
    if (!this.initGl()) return;

    const gl = this.gl;
    const floats = new Float32Array(b64ToArrayBuffer(cache.lut));
    const texData = mergedLutTo3DTextureData(floats);

    gl.bindTexture(gl.TEXTURE_3D, this.lutTex);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
    gl.texImage3D(
      gl.TEXTURE_3D,
      0,
      gl.RGB32F,
      LUT_SIZE,
      LUT_SIZE,
      LUT_SIZE,
      0,
      gl.RGB,
      gl.FLOAT,
      texData,
    );

    const img = new Image();
    img.onload = () => {
      this.canvas.width = img.width;
      this.canvas.height = img.height;
      this.previewH = Math.min(360, Math.max(120, img.height));
      this.node.setSize?.(this.node.size);

      gl.bindTexture(gl.TEXTURE_2D, this.srcTex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);

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
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.useProgram(this.prog);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.srcTex);
    gl.uniform1i(gl.getUniformLocation(this.prog, "u_src"), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_3D, this.lutTex);
    gl.uniform1i(gl.getUniformLocation(this.prog, "u_lut"), 1);
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
      const h = panel.previewH + 28;
      return [width, h];
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

      const nodeId = detail.display_node || detail.node;
      const graphNode = app.graph.getNodeById?.(nodeId) ?? app.graph._nodes_by_id?.[nodeId];
      if (!graphNode || graphNode.comfyClass !== PREVIEW_CLASS) return;

      getPanel(graphNode).setCache(out.magicmatch_live[0]);
      const sw = graphNode.widgets?.find((w) => w.name === "strength");
      if (sw) getPanel(graphNode).render(Number(sw.value));
    });
  },
});
