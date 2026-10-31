import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { openEditor } from "./bbox_modal.js";

// Image BBox Layout (MOSS): upload an image, then draw boxes and paint a mask in the full-screen editor
// (bbox_modal.js). Boxes are stored normalised (0..1, top-left origin) in the hidden `bbox_data` widget, so they
// stay correct whatever size the image is resized to; the painted mask is a small PNG named in `mask_file`.


function splitName(v) {
  let type = "input";
  const m = String(v).match(/^(.*) \[(input|output|temp)\]$/);
  if (m) { v = m[1]; type = m[2]; }
  const i = v.lastIndexOf("/");
  return { filename: i >= 0 ? v.slice(i + 1) : v, subfolder: i >= 0 ? v.slice(0, i) : "", type };
}

function viewUrl(value) {
  const p = new URLSearchParams({ ...splitName(value), rand: String(Math.random()) });
  return api.apiURL("/view?" + p.toString());
}

// Big photos make ComfyUI's mask editor crawl (it paints at the file's full size), and the node resizes to
// max_megapixels when it runs anyway. So shrink on upload to at most this many pixels, keeping transparency.
const MAX_UPLOAD_MP = 1.5;
async function shrinkFile(file) {
  try {
    const bmp = await createImageBitmap(file);
    const px = bmp.width * bmp.height;
    const cap = MAX_UPLOAD_MP * 1e6;
    if (px <= cap) { bmp.close?.(); return file; }
    const k = Math.sqrt(cap / px);
    const w = Math.max(1, Math.round(bmp.width * k)), h = Math.max(1, Math.round(bmp.height * k));
    const c = document.createElement("canvas"); c.width = w; c.height = h;
    const x = c.getContext("2d"); x.imageSmoothingQuality = "high"; x.drawImage(bmp, 0, 0, w, h); bmp.close?.();
    const blob = await new Promise((r) => c.toBlob(r, "image/png"));
    if (!blob) return file;
    return new File([blob], file.name.replace(/\.[^.]+$/, "") + ".png", { type: "image/png" });
  } catch { return file; } // can't decode here: upload as is
}

async function uploadFile(file) {
  const body = new FormData();
  body.append("image", file);
  body.append("type", "input");
  const resp = await api.fetchApi("/upload/image", { method: "POST", body });
  if (resp.status !== 200) throw new Error(`${resp.status} ${resp.statusText}`);
  const data = await resp.json();
  return data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
}

// ComfyUI only autosaves/undoes widget changes it hears about. Edits made inside our own DOM editor
// don't trigger that, so after a page reload the node came back with its old boxes. Tell it explicitly.
let notifyTimer = null;
function notifyChange(delay = 0) {
  clearTimeout(notifyTimer);
  notifyTimer = setTimeout(() => {
    try { app.extensionManager?.workflow?.activeWorkflow?.changeTracker?.checkState?.(); } catch { /* older frontends */ }
    try { app.graph?.change?.(); } catch { /* ignore */ }
  }, delay);
}

const clamp01 = (v) => Math.min(1, Math.max(0, v));

const COLORS = ["#ff5252", "#40c4ff", "#ffd740", "#69f0ae", "#e040fb", "#ff9100", "#18ffff", "#b2ff59"];
const MAX_STAGE_H = 520;
const EMPTY_H = 150;
const EXTRA_H = 112;       // buttons + summary + padding around the preview

function addStyle() {
  if (document.getElementById("bbl-style")) return;
  const s = document.createElement("style");
  s.id = "bbl-style";
  s.textContent = `
    .bbl-root{display:flex;flex-direction:column;gap:6px;width:100%;height:100%;box-sizing:border-box;
      pointer-events:auto;font-size:12px;color:#ddd;font-family:sans-serif}
    .bbl-bar{display:flex;align-items:center;gap:6px}
    .bbl-btn{white-space:nowrap;padding:3px 10px;cursor:pointer;border-radius:4px;border:1px solid #666;background:#333;color:#eee;font-size:12px}
    .bbl-btn:hover{background:#444}
    .bbl-btn.pri{background:#2d6cdf;border-color:#2d6cdf;color:#fff;font-weight:600;padding:5px 14px}
    .bbl-btn.pri:hover{background:#3a7af0}
    .bbl-name{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;opacity:.7}
    .bbl-stage{position:relative;margin:0 auto;background:#1e1e1e;border:1px solid #555;border-radius:4px;overflow:hidden;user-select:none;cursor:pointer}
    .bbl-stage img{display:block;width:100%;height:auto;pointer-events:none;-webkit-user-drag:none;
      background:repeating-conic-gradient(#555 0% 25%, #3a3a3a 0% 50%) 0 0 / 16px 16px}
    .bbl-stage canvas{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
    .bbl-empty{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;text-align:center;
      white-space:pre-line;opacity:.55}
    .bbl-hint{opacity:.6}
  `;
  document.head.appendChild(s);
}

app.registerExtension({
  name: "MidFR.ImageBBoxLayout",

  async nodeCreated(node) {
    if (node.comfyClass !== "ImageBBoxLayout") return;
    const imageW = node.widgets.find((w) => w.name === "image");
    const dataW = node.widgets.find((w) => w.name === "bbox_data");
    const maskW = node.widgets.find((w) => w.name === "mask_file");
    if (!imageW || !dataW) return;
    for (const w of [imageW, dataW, maskW]) if (w) { w.hidden = true; w.computeSize = () => [0, -4]; }
    addStyle();

    let boxes = [];       // {x,y,w,h,caption}, normalised
    let aspect = 0;       // image width / height once loaded
    let maskImg = null;   // the painted mask (alpha), for the preview
    let promptMode = false; // the plain prompt box is shown instead of the box editor: keep the picture, drop the editing UI

    // ---------- DOM: a read-only preview; editing happens in the full-screen editor ----------
    const root = document.createElement("div"); root.className = "bbl-root";
    const bar = document.createElement("div"); bar.className = "bbl-bar";
    const upBtn = document.createElement("button"); upBtn.className = "bbl-btn"; upBtn.textContent = "Upload image";
    const nameEl = document.createElement("span"); nameEl.className = "bbl-name";
    bar.append(upBtn, nameEl);

    const stage = document.createElement("div"); stage.className = "bbl-stage";
    const img = document.createElement("img"); img.draggable = false; img.decoding = "async"; img.style.display = "none";
    const prev = document.createElement("canvas");
    const empty = document.createElement("div"); empty.className = "bbl-empty";
    empty.textContent = "Drop an image here\nor click to upload";
    stage.append(img, prev, empty);

    const row = document.createElement("div"); row.className = "bbl-bar";
    const editBtn = document.createElement("button"); editBtn.className = "bbl-btn pri"; editBtn.textContent = "Edit boxes & mask";
    const summary = document.createElement("span"); summary.className = "bbl-hint";
    row.append(editBtn, summary);

    const fileIn = document.createElement("input"); fileIn.type = "file"; fileIn.accept = "image/*"; fileIn.style.display = "none";
    root.append(bar, stage, row, fileIn);

    // ---------- state ----------
    function save() {
      dataW.value = JSON.stringify(boxes);
      node.setDirtyCanvas(true, true);
      notifyChange(250);
      drawPreview();
    }
    function loadBoxes() {
      let raw = [];
      try { raw = JSON.parse(dataW.value || "[]"); } catch { raw = []; }
      boxes = [];
      if (Array.isArray(raw)) {
        for (const r of raw) {
          const x = clamp01(+r.x), y = clamp01(+r.y);
          const w = Math.min(+r.w, 1 - x), h = Math.min(+r.h, 1 - y);
          if (!(w > 0) || !(h > 0)) continue;
          boxes.push({ x, y, w, h, caption: typeof r.caption === "string" ? r.caption : "", strength: r.strength == null || r.strength === "" ? null : +r.strength });
        }
      }
    }
    function loadMaskImg() {
      maskImg = null;
      if (!maskW || !maskW.value) { drawPreview(); return; }
      const m = new Image();
      const v = maskW.value;
      m.onload = () => { if (maskW.value === v) { maskImg = m; drawPreview(); } };
      m.src = viewUrl(v);
    }

    // ---------- preview ----------
    function drawPreview() {
      const has = imageW.value && imageW.value !== "none";
      const painted = !!(maskW && maskW.value);
      summary.textContent = has ? `${boxes.length} box${boxes.length === 1 ? "" : "es"}${painted ? " · mask painted" : ""}` : "";
      if (!aspect) return;
      const W = 640, H = Math.max(1, Math.round(W / aspect));
      if (prev.width !== W || prev.height !== H) { prev.width = W; prev.height = H; }
      const c = prev.getContext("2d");
      c.clearRect(0, 0, W, H);
      if (maskImg) { c.globalAlpha = 0.55; c.drawImage(maskImg, 0, 0, W, H); c.globalAlpha = 1; }
      boxes.forEach((b, i) => {
        const col = COLORS[i % COLORS.length];
        const x = b.x * W, y = b.y * H, w = b.w * W, h = b.h * H;
        c.fillStyle = col + "26"; c.fillRect(x, y, w, h);
        c.strokeStyle = col; c.lineWidth = 3; c.strokeRect(x, y, w, h);
        c.font = "bold 18px sans-serif"; c.fillStyle = col; c.fillRect(x, y, 26, 24);
        c.fillStyle = "#000"; c.textBaseline = "middle"; c.textAlign = "center"; c.fillText(String(i + 1), x + 13, y + 13); c.textAlign = "start";
      });
    }

    // ---------- image ----------
    function stageW() { return Math.max(120, (node.size?.[0] ?? 400) - 24); }
    function stageH(w) {
      if (!aspect) return EMPTY_H;
      return Math.min(w / aspect, MAX_STAGE_H);
    }
    function layoutStage() {
      if (!aspect) { stage.style.aspectRatio = ""; stage.style.maxWidth = ""; stage.style.width = "100%"; stage.style.height = EMPTY_H + "px"; return; }
      stage.style.aspectRatio = `${aspect}`;
      stage.style.height = "auto";
      stage.style.width = "100%";
      stage.style.maxWidth = MAX_STAGE_H * aspect + "px";
    }
    function fitNode() {
      const sz = node.computeSize();
      node.setSize([Math.max(node.size?.[0] ?? 0, sz[0]), sz[1]]);
      node.setDirtyCanvas(true, true);
    }
    function refreshImage() {
      const v = imageW.value;
      const has = v && v !== "none";
      nameEl.textContent = has ? v : "no image";
      nameEl.title = has ? v : "";
      empty.style.display = has ? "none" : "flex";
      img.style.display = has ? "block" : "none";
      prev.style.display = has && !promptMode ? "block" : "none";
      editBtn.disabled = !has;
      if (!has) { delete img.dataset.v; img.removeAttribute("src"); aspect = 0; layoutStage(); drawPreview(); return; }
      if (img.dataset.v !== v) { img.dataset.v = v; img.src = viewUrl(v); }
    }
    img.addEventListener("load", () => {
      if (img.naturalWidth && img.naturalHeight) {
        aspect = img.naturalWidth / img.naturalHeight;
        layoutStage();
        fitNode();
        drawPreview();
      }
    });
    img.addEventListener("error", () => { nameEl.textContent = "can't load: " + imageW.value; });

    async function handleFile(file) {
      if (!file || !file.type.startsWith("image/")) return;
      try {
        imageW.value = await uploadFile(await shrinkFile(file));
        if (maskW && maskW.value) { maskW.value = ""; maskImg = null; } // a mask belongs to one picture
        node.setDirtyCanvas(true, true);
        notifyChange();
        refreshImage();
      } catch (e) {
        console.error("Upload failed", e);
        alert("Upload failed: " + e.message);
      }
    }
    upBtn.addEventListener("click", () => fileIn.click());
    fileIn.addEventListener("change", () => { if (fileIn.files[0]) handleFile(fileIn.files[0]); fileIn.value = ""; });
    stage.addEventListener("dragover", (e) => { e.preventDefault(); e.stopPropagation(); stage.style.borderColor = "#4a9eff"; });
    stage.addEventListener("dragleave", () => { stage.style.borderColor = ""; });
    stage.addEventListener("drop", (e) => {
      e.preventDefault(); e.stopPropagation(); stage.style.borderColor = "";
      if (e.dataTransfer?.files?.length) handleFile(e.dataTransfer.files[0]);
    });

    // ---------- editor ----------
    async function uploadMask(blob) {
      if (!maskW) return;
      if (!blob) { if (maskW.value) { maskW.value = ""; maskImg = null; notifyChange(); } return; }
      const name = maskW.value || `midfr_mask_${Math.random().toString(36).slice(2, 10)}.png`;
      const body = new FormData();
      body.append("image", new File([blob], name, { type: "image/png" }));
      body.append("type", "input"); body.append("overwrite", "true");
      const resp = await api.fetchApi("/upload/image", { method: "POST", body });
      if (resp.status !== 200) throw new Error(`${resp.status} ${resp.statusText}`);
      const data = await resp.json();
      maskW.value = data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
      notifyChange();
    }
    function edit() {
      const v = imageW.value;
      if (!v || v === "none") { fileIn.click(); return; }
      const mp = node.widgets.find((w) => w.name === "max_megapixels")?.value ?? 1;
      openEditor({
        imageUrl: viewUrl(v),
        maskUrl: maskW && maskW.value ? viewUrl(maskW.value) : null,
        maxMp: mp,
        boxes,
        onBoxes: (b) => { boxes = b; save(); },
        onMask: async (blob) => {
          try { await uploadMask(blob); } catch (e) { console.error("mask upload failed", e); alert("Couldn't save the mask: " + e.message); }
        },
        onClose: () => { loadMaskImg(); node.setDirtyCanvas(true, true); },
      });
    }
    editBtn.addEventListener("click", edit);
    stage.addEventListener("click", () => { if (imageW.value && imageW.value !== "none") { if (!promptMode) edit(); } else fileIn.click(); });

    // ---------- widget ----------
    const extraH = () => EXTRA_H - (promptMode ? 34 : 0);
    const widget = node.addDOMWidget("bbox_editor", "custom", root, {
      serialize: false,
      hideOnZoom: false,
      getMinHeight: () => stageH(stageW()) + extraH(),
    });
    widget.computeSize = (w) => [w, stageH(Math.max(120, (w ?? 400) - 24)) + extraH()];
    node.__bblSetMode = (v) => {
      promptMode = !!v;
      row.style.display = promptMode ? "none" : "";
      stage.style.cursor = promptMode ? "default" : "";
      refreshImage();
    };

    function restore() {
      loadBoxes();
      layoutStage();
      refreshImage();
      loadMaskImg();
      drawPreview();
    }
    const origConfigure = node.onConfigure;
    node.onConfigure = function () {
      const r = origConfigure ? origConfigure.apply(this, arguments) : undefined;
      restore();
      return r;
    };

    const sz = node.computeSize();
    node.setSize([Math.max(node.size?.[0] ?? 0, 420), Math.max(node.size?.[1] ?? 0, sz[1])]);
    restore();
  },
});


// Settings live in a popup opened by a gear button at the top right of the node, so the node stays small.
const SETTINGS = ["max_megapixels"];
const LEGACY = ["mask_mode", "mask_grow", "mask_feather"]; // unused by the backend now; kept so saved workflows load
const LABELS = {
  max_megapixels: ["Max megapixels", "Image is downscaled to this size (sides snap to 16)"],
  mask_mode: ["Mask from", "boxes + masks, boxes only, masks only, or the whole image"],
  mask_grow: ["Grow mask (px)", "Expand the mask this many pixels"],
  mask_feather: ["Feather (px)", "Soft mask edge; pair with Differential Diffusion"],
};
const BTN = 22; // gear hit box, px

function hideSettings(node, resize) {
  const before = node.computeSize()[1];
  for (const w of node.widgets || []) {
    if (SETTINGS.includes(w.name) || LEGACY.includes(w.name)) { w.hidden = true; w.computeSize = () => [0, -4]; }
  }
  const delta = node.computeSize()[1] - before;
  if (resize && delta) node.setSize([node.size[0], Math.max(120, node.size[1] + delta)]);
  node.setDirtyCanvas?.(true, true);
}

// One or the other: the box editor (default) or the plain prompt box. The choice lives in node.properties.show_prompt
// so it is saved with the workflow.
function applyView(node, resize) {
  const showPrompt = !!node.properties?.show_prompt;
  const before = node.computeSize()[1];
  for (const w of node.widgets || []) {
    if (w.name !== "prompt" && w.name !== "bbox_editor") continue;
    if (!("__cs" in w)) w.__cs = Object.prototype.hasOwnProperty.call(w, "computeSize") ? w.computeSize : null;
    if (w.name === "bbox_editor") { // the picture stays; only its editing controls go
      w.hidden = false;
      if (w.__cs) w.computeSize = w.__cs; else delete w.computeSize;
      const el0 = w.element || w.inputEl;
      if (el0 && el0.style) el0.style.display = "";
      node.__bblSetMode?.(showPrompt);
      continue;
    }
    const hide = !showPrompt;
    if (hide && w.value) { // hidden prompt text must not keep reaching the boxes (it is added to every box's prompt)
      w.value = "";
      const t = w.element || w.inputEl;
      if (t && "value" in t) t.value = "";
    }
    w.hidden = hide;
    if (hide) w.computeSize = () => [0, -4];
    else if (w.__cs) w.computeSize = w.__cs; else delete w.computeSize;
    const el = w.element || w.inputEl;
    if (el && el.style) el.style.display = hide ? "none" : "";
  }
  const delta = node.computeSize()[1] - before;
  if (resize && delta) node.setSize([node.size[0], Math.max(120, node.size[1] + delta)]);
  node.setDirtyCanvas?.(true, true);
}

function inGear(node, pos) {
  const titleH = (window.LiteGraph && LiteGraph.NODE_TITLE_HEIGHT) || 30;
  return pos[1] < 0 && pos[1] > -titleH && pos[0] > node.size[0] - BTN - 10;
}

let openPanel = null;
function closePanel() { openPanel?.remove(); openPanel = null; document.removeEventListener("keydown", onKey, true); }
function onKey(e) { if (e.key === "Escape") closePanel(); }

function showSettings(node) {
  closePanel();
  const ds = app.canvas?.ds;
  const rect = app.canvas?.canvas?.getBoundingClientRect?.() || { left: 0, top: 0 };
  const scale = ds?.scale || 1, off = ds?.offset || [0, 0];
  let left = rect.left + (node.pos[0] + node.size[0] + off[0]) * scale + 8;
  let top = rect.top + (node.pos[1] + off[1]) * scale - 30 * scale;

  const panel = document.createElement("div");
  panel.style.cssText = "position:fixed;z-index:10000;width:280px;padding:12px;background:#2b2b2b;color:#ddd;" +
    "border:1px solid #666;border-radius:8px;box-shadow:0 6px 24px rgba(0,0,0,.5);font:12px sans-serif";
  const head = document.createElement("div");
  head.style.cssText = "display:flex;align-items:center;margin-bottom:8px;font-weight:600;font-size:13px";
  head.innerHTML = '<span style="flex:1">BBox Layout settings</span>';
  const x = document.createElement("button");
  x.textContent = "\u2715"; x.style.cssText = "background:none;border:0;color:#ddd;cursor:pointer;font-size:14px";
  x.onclick = closePanel; head.appendChild(x); panel.appendChild(head);

  const trow = document.createElement("label");
  trow.style.cssText = "display:flex;align-items:center;gap:8px;margin:0 0 10px;padding-bottom:10px;border-bottom:1px solid #444;cursor:pointer";
  const tcb = document.createElement("input"); tcb.type = "checkbox"; tcb.checked = !!node.properties?.show_prompt;
  const ttx = document.createElement("span"); ttx.textContent = "Show prompt box instead of boxes";
  trow.title = "On: a plain prompt box, the box editor is hidden. Off: draw and prompt boxes (the prompt box is hidden).";
  tcb.onchange = () => {
    node.properties = node.properties || {};
    node.properties.show_prompt = tcb.checked;
    applyView(node, true);
    notifyChange(100);
  };
  trow.append(tcb, ttx); panel.appendChild(trow);

  for (const name of SETTINGS) {
    const w = node.widgets?.find((v) => v.name === name);
    if (!w) continue;
    const row = document.createElement("label");
    row.title = LABELS[name][1];
    row.style.cssText = "display:flex;align-items:center;gap:8px;margin:6px 0";
    const span = document.createElement("span"); span.textContent = LABELS[name][0]; span.style.flex = "1";
    let input;
    if (name === "mask_mode") {
      input = document.createElement("select");
      for (const v of w.options?.values || []) {
        const o = document.createElement("option"); o.value = o.textContent = v; input.appendChild(o);
      }
      input.value = w.value;
    } else {
      input = document.createElement("input"); input.type = "number";
      input.min = w.options?.min ?? 0; input.max = w.options?.max ?? 512;
      input.step = name === "max_megapixels" ? 0.05 : (w.options?.step2 ?? w.options?.step ?? 8);
      input.value = w.value;
    }
    input.style.cssText = "width:110px;background:#1e1e1e;color:#ddd;border:1px solid #555;border-radius:4px;padding:3px 5px";
    input.onchange = () => {
      let v = name === "mask_mode" ? input.value : Number(input.value);
      if (name !== "mask_mode") {
        if (!Number.isFinite(v)) v = w.value;
        v = Math.min(Number(input.max), Math.max(Number(input.min), v));
        if (name !== "max_megapixels") v = Math.round(v);
        input.value = v;
      }
      w.value = v;
      w.callback?.(v);
      node.setDirtyCanvas?.(true, true);
      notifyChange(250);
    };
    row.append(span, input); panel.appendChild(row);
  }
  document.body.appendChild(panel);
  const pw = panel.offsetWidth, ph = panel.offsetHeight;
  left = Math.max(8, Math.min(left, innerWidth - pw - 8));
  top = Math.max(8, Math.min(top, innerHeight - ph - 8));
  panel.style.left = left + "px"; panel.style.top = top + "px";
  openPanel = panel;
  document.addEventListener("keydown", onKey, true);
}

app.registerExtension({
  name: "MidFR.ImageBBoxLayout.Settings",
  nodeCreated(node) {
    if (node.comfyClass !== "ImageBBoxLayout") return;
    const draw = node.onDrawForeground;
    node.onDrawForeground = function (ctx) {
      draw?.apply(this, arguments);
      if (this.flags?.collapsed) return;
      const titleH = (window.LiteGraph && LiteGraph.NODE_TITLE_HEIGHT) || 30;
      const cx = this.size[0] - BTN / 2 - 8, cy = -titleH / 2;
      ctx.save();
      ctx.fillStyle = openPanel ? "#5a7bd8" : "rgba(255,255,255,0.12)";
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(cx - BTN / 2, cy - BTN / 2, BTN, BTN, 5); else ctx.rect(cx - BTN / 2, cy - BTN / 2, BTN, BTN);
      ctx.fill();
      ctx.fillStyle = "#ddd"; ctx.font = "15px sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText("\u2699", cx, cy + 1);
      ctx.restore();
    };
    const down = node.onMouseDown;
    node.onMouseDown = function (e, pos) {
      if (inGear(this, pos)) { openPanel ? closePanel() : showSettings(this); this.setDirtyCanvas?.(true, true); return true; }
      return down?.apply(this, arguments);
    };
    const removed = node.onRemoved;
    node.onRemoved = function () { closePanel(); return removed?.apply(this, arguments); };
    const cfg = node.onConfigure;
    node.onConfigure = function () {
      const r = cfg?.apply(this, arguments);
      hideSettings(this, false); // saved size already excludes them
      applyView(this, false);
      return r;
    };
    setTimeout(() => { hideSettings(node, true); applyView(node, true); }, 0);
  },
});
