/**
 * MAGICMATCH — auto re-run Preview when the strength slider moves.
 * Build LUT stays cached (Comfy only re-executes Preview, ~0.5s).
 */
import { app } from "../../../scripts/app.js";

const DEBOUNCE_MS = 150;
const PREVIEW_CLASS = "MagicMatchPreview";

function shouldAutoRefresh(node) {
  const w = node.widgets?.find((x) => x.name === "auto_refresh");
  return w === undefined || w.value !== false;
}

app.registerExtension({
  name: "MAGICMATCH.live_strength",

  nodeCreated(node) {
    if (node.comfyClass !== PREVIEW_CLASS) return;

    const strengthWidget = node.widgets?.find((w) => w.name === "strength");
    if (!strengthWidget) return;

    let timer = null;
    const prevCallback = strengthWidget.callback;

    strengthWidget.callback = function (value, ...rest) {
      if (prevCallback) {
        prevCallback.call(this, value, ...rest);
      }
      if (!shouldAutoRefresh(node)) return;

      clearTimeout(timer);
      timer = setTimeout(() => {
        try {
          if (app.runningNodeId !== undefined && app.runningNodeId !== null) return;
          app.queuePrompt(0, 1);
        } catch (e) {
          console.warn("[MAGICMATCH] auto_refresh queue failed:", e);
        }
      }, DEBOUNCE_MS);
    };
  },
});
