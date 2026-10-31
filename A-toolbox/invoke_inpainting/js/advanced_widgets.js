import { app } from "../../scripts/app.js";

// Keeps the Invoke / Scene Context nodes short: only the settings you normally touch show. Everything else sits behind one
// "Advanced settings" button on the node (the choice is saved with the workflow). Nothing is removed or renamed, so saved
// workflows and API graphs keep working; hidden widgets still send their values.
const SHOWN = {
  InvokeInpaintingMOSS: ["seed", "control_after_generate", "steps"],
  SceneContextQwenVL: ["model"],
};

function apply(node, resize) {
  const keep = SHOWN[node.comfyClass || node.type];
  if (!keep) return;
  const open = !!node.properties?.advanced;
  const before = node.computeSize()[1];
  for (const w of node.widgets || []) {
    if (w.__advBtn || keep.includes(w.name)) continue;
    if (!("__cs" in w)) w.__cs = Object.prototype.hasOwnProperty.call(w, "computeSize") ? w.computeSize : null;
    w.hidden = !open;
    if (!open) w.computeSize = () => [0, -4];
    else if (w.__cs) w.computeSize = w.__cs; else delete w.computeSize;
    const el = w.element || w.inputEl;
    if (el && el.style) el.style.display = open ? "" : "none";
  }
  const btn = (node.widgets || []).find((w) => w.__advBtn);
  if (btn) btn.name = open ? "Advanced settings (hide)" : "Advanced settings...";
  const delta = node.computeSize()[1] - before;
  if (resize && delta) node.setSize([node.size[0], Math.max(80, node.size[1] + delta)]);
  node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
  name: "midfr.advanced_widgets",
  nodeCreated(node) {
    const cls = node.comfyClass || node.type;
    const keep = SHOWN[cls];
    if (!keep || (node.widgets || []).some((w) => w.__advBtn)) return;
    const btn = node.addWidget("button", "Advanced settings...", null, () => {
      node.properties = node.properties || {};
      node.properties.advanced = !node.properties.advanced;
      apply(node, true);
    });
    btn.__advBtn = true;
    btn.serialize = false;
    // put the button right after the settings that stay visible
    const ws = node.widgets;
    ws.splice(ws.indexOf(btn), 1);
    let at = 0;
    ws.forEach((w, i) => { if (keep.includes(w.name)) at = i + 1; });
    ws.splice(at, 0, btn);
    const onConfigure = node.onConfigure;
    node.onConfigure = function (...a) { const r = onConfigure?.apply(this, a); setTimeout(() => apply(node, false), 0); return r; };
    setTimeout(() => apply(node, true), 0);
  },
});
