import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Image BBox Layout (MOSS): upload an image, drag boxes on it, prompt each box.
// Boxes are stored normalised (0..1, top-left origin) in the hidden `bbox_data` widget, so they
// stay correct whatever size the image is resized to; the backend turns them into MOSS pixels.

const COLORS = ["#ff5252", "#40c4ff", "#ffd740", "#69f0ae", "#e040fb", "#ff9100", "#18ffff", "#b2ff59"];
const MIN = 0.01;          // smallest box edge (fraction of the image)
const PANEL_H = 150;       // chips + caption box
const MAX_STAGE_H = 640;
const EMPTY_H = 200;

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

async function uploadFile(file) {
  const body = new FormData();
  body.append("image", file);
  body.append("type", "input");
  const resp = await api.fetchApi("/upload/image", { method: "POST", body });
  if (resp.status !== 200) throw new Error(`${resp.status} ${resp.statusText}`);
  const data = await resp.json();
  return data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
}

const clamp01 = (v) => Math.min(1, Math.max(0, v));
const round4 = (v) => Math.round(v * 10000) / 10000;

function addStyle() {
  if (document.getElementById("bbl-style")) return;
  const s = document.createElement("style");
  s.id = "bbl-style";
  s.textContent = `
    .bbl-root{display:flex;flex-direction:column;gap:6px;width:100%;height:100%;box-sizing:border-box;
      pointer-events:auto;font-size:12px;color:#ddd;font-family:sans-serif}
    .bbl-bar{display:flex;align-items:center;gap:6px}
    .bbl-btn{padding:3px 10px;cursor:pointer;border-radius:4px;border:1px solid #666;background:#333;color:#eee;font-size:12px}
    .bbl-btn:hover{background:#444}
    .bbl-name{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;opacity:.7}
    .bbl-stage{position:relative;margin:0 auto;background:#1e1e1e;border:1px solid #555;border-radius:4px;overflow:hidden;user-select:none}
    .bbl-stage img{display:block;width:100%;height:auto;pointer-events:none;-webkit-user-drag:none}
    .bbl-empty{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;text-align:center;
      white-space:pre-line;opacity:.55;cursor:pointer}
    .bbl-over{position:absolute;inset:0;cursor:crosshair;touch-action:none;outline:none}
    .bbl-box{position:absolute;box-sizing:border-box;border:2px solid var(--c);background:color-mix(in srgb,var(--c) 14%,transparent);cursor:move;touch-action:none}
    .bbl-box.sel{border-width:3px;background:color-mix(in srgb,var(--c) 26%,transparent)}
    .bbl-tag{position:absolute;left:0;top:0;background:var(--c);color:#000;font-weight:bold;padding:0 5px;border-radius:0 0 3px 0;
      font-size:11px;line-height:16px;white-space:nowrap;max-width:140px;overflow:hidden;text-overflow:ellipsis;pointer-events:none}
    .bbl-h{position:absolute;width:12px;height:12px;background:var(--c);border:1px solid #000;box-sizing:border-box;display:none;touch-action:none}
    .bbl-box.sel .bbl-h{display:block}
    .bbl-h[data-h=nw]{left:-7px;top:-7px;cursor:nwse-resize}.bbl-h[data-h=ne]{right:-7px;top:-7px;cursor:nesw-resize}
    .bbl-h[data-h=sw]{left:-7px;bottom:-7px;cursor:nesw-resize}.bbl-h[data-h=se]{right:-7px;bottom:-7px;cursor:nwse-resize}
    .bbl-panel{display:flex;flex-direction:column;gap:5px;min-height:0}
    .bbl-chips{display:flex;flex-wrap:wrap;gap:4px;min-height:20px;align-items:center}
    .bbl-chip{padding:1px 8px;border-radius:10px;border:2px solid var(--c);background:#2a2a2a;color:#eee;cursor:pointer;font-size:11px}
    .bbl-chip.sel{background:var(--c);color:#000;font-weight:bold}
    .bbl-row{display:flex;gap:6px;align-items:center}
    .bbl-ta{width:100%;min-height:64px;resize:vertical;background:#222;color:#eee;border:1px solid #666;border-radius:4px;padding:4px 6px;
      font-size:12px;font-family:inherit;box-sizing:border-box}
    .bbl-hint{opacity:.55}
  `;
  document.head.appendChild(s);
}

app.registerExtension({
  name: "SimpleImageUploadResize.BBoxLayout",

  async nodeCreated(node) {
    if (node.comfyClass !== "ImageBBoxLayout") return;
    const imageW = node.widgets.find((w) => w.name === "image");
    const dataW = node.widgets.find((w) => w.name === "bbox_data");
    if (!imageW || !dataW) return;
    for (const w of [imageW, dataW]) { w.hidden = true; w.computeSize = () => [0, -4]; }
    addStyle();

    let boxes = [];       // {x,y,w,h,caption}
    let sel = -1;
    let aspect = 0;       // image width / height once loaded
    let els = [];         // box elements, parallel to boxes

    // ---------- DOM ----------
    const root = document.createElement("div"); root.className = "bbl-root";

    const bar = document.createElement("div"); bar.className = "bbl-bar";
    const upBtn = document.createElement("button"); upBtn.className = "bbl-btn"; upBtn.textContent = "Upload image";
    const nameEl = document.createElement("span"); nameEl.className = "bbl-name";
    const clearBtn = document.createElement("button"); clearBtn.className = "bbl-btn"; clearBtn.textContent = "Clear boxes";
    bar.append(upBtn, nameEl, clearBtn);

    const stage = document.createElement("div"); stage.className = "bbl-stage";
    const img = document.createElement("img"); img.draggable = false; img.decoding = "async"; img.style.display = "none";
    const empty = document.createElement("div"); empty.className = "bbl-empty";
    empty.textContent = "Drop an image here\nor click to upload";
    const over = document.createElement("div"); over.className = "bbl-over"; over.tabIndex = 0; over.style.display = "none";
    stage.append(img, empty, over);

    const panel = document.createElement("div"); panel.className = "bbl-panel";
    const chips = document.createElement("div"); chips.className = "bbl-chips";
    const ctrl = document.createElement("div"); ctrl.className = "bbl-row";
    const delBtn = document.createElement("button"); delBtn.className = "bbl-btn"; delBtn.textContent = "Delete box";
    ctrl.append(delBtn);
    const capIn = document.createElement("textarea"); capIn.className = "bbl-ta";
    capIn.placeholder = "Prompt for this box: what should appear here?";
    const hint = document.createElement("div"); hint.className = "bbl-hint";
    hint.textContent = "Drag on the image to add a box. Click a box to edit its prompt; drag its corners to resize.";
    panel.append(chips, ctrl, capIn, hint);

    const fileIn = document.createElement("input"); fileIn.type = "file"; fileIn.accept = "image/*"; fileIn.style.display = "none";
    root.append(bar, stage, panel, fileIn);

    // Keep canvas shortcuts (Delete, Ctrl+..., etc.) from firing while typing here.
    for (const el of [capIn]) el.addEventListener("keydown", (e) => e.stopPropagation());

    // ---------- state ----------
    function save() {
      dataW.value = JSON.stringify(boxes.map((b) => ({
        x: round4(b.x), y: round4(b.y), w: round4(b.w), h: round4(b.h),
        caption: b.caption,
      })));
      node.setDirtyCanvas(true, true);
      app.graph?.change?.();
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
          boxes.push({
            x, y, w, h,
            caption: typeof r.caption === "string" ? r.caption : "",
          });
        }
      }
      sel = -1;
    }

    // ---------- rendering ----------
    const color = (i) => COLORS[i % COLORS.length];

    function geom(el, b) {
      el.style.left = b.x * 100 + "%"; el.style.top = b.y * 100 + "%";
      el.style.width = b.w * 100 + "%"; el.style.height = b.h * 100 + "%";
    }

    function tagText(i) {
      const b = boxes[i];
      return `#${i + 1}` + (b.caption ? `: ${b.caption}` : "");
    }

    function renderBoxes() {
      over.replaceChildren();
      els = boxes.map((b, i) => {
        const el = document.createElement("div");
        el.className = "bbl-box" + (i === sel ? " sel" : "");
        el.style.setProperty("--c", color(i));
        el.dataset.i = i;
        const tag = document.createElement("div"); tag.className = "bbl-tag"; tag.textContent = tagText(i);
        el.append(tag);
        for (const h of ["nw", "ne", "sw", "se"]) {
          const hd = document.createElement("div"); hd.className = "bbl-h"; hd.dataset.h = h; el.append(hd);
        }
        geom(el, b);
        over.append(el);
        return el;
      });
      renderPanel();
    }

    function updateSelection() {
      els.forEach((el, i) => el.classList.toggle("sel", i === sel));
      renderPanel();
    }

    function renderPanel() {
      chips.replaceChildren();
      boxes.forEach((b, i) => {
        const c = document.createElement("button");
        c.className = "bbl-chip" + (i === sel ? " sel" : "");
        c.style.setProperty("--c", color(i));
        c.textContent = `#${i + 1}`;
        c.addEventListener("click", () => { sel = i; updateSelection(); });
        chips.append(c);
      });
      const has = sel >= 0 && sel < boxes.length;
      ctrl.style.display = capIn.style.display = has ? "" : "none";
      hint.style.display = has ? "none" : "";
      hint.textContent = boxes.length
        ? "Select a box to edit its prompt. Without boxes, the prompt is passed through as plain text."
        : "Drag on the image to add a box. Without boxes, the prompt is passed through as plain text.";
      if (has) {
        const b = boxes[sel];
        capIn.value = b.caption;
      }
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
      over.style.display = has ? "block" : "none";
      img.style.display = has ? "block" : "none";
      if (!has) { delete img.dataset.v; img.removeAttribute("src"); aspect = 0; layoutStage(); return; }
      if (img.dataset.v !== v) { img.dataset.v = v; img.src = viewUrl(v); }
    }
    img.addEventListener("load", () => {
      if (img.naturalWidth && img.naturalHeight) {
        aspect = img.naturalWidth / img.naturalHeight;
        layoutStage();
        fitNode();
      }
    });
    img.addEventListener("error", () => { nameEl.textContent = "can't load: " + imageW.value; });

    async function handleFile(file) {
      if (!file || !file.type.startsWith("image/")) return;
      try {
        imageW.value = await uploadFile(file);
        node.setDirtyCanvas(true, true);
        refreshImage();
      } catch (e) {
        console.error("Upload failed", e);
        alert("Upload failed: " + e.message);
      }
    }
    upBtn.addEventListener("click", () => fileIn.click());
    empty.addEventListener("click", () => fileIn.click());
    fileIn.addEventListener("change", () => { if (fileIn.files[0]) handleFile(fileIn.files[0]); fileIn.value = ""; });
    for (const t of [stage, empty]) {
      t.addEventListener("dragover", (e) => { e.preventDefault(); e.stopPropagation(); stage.style.borderColor = "#4a9eff"; });
      t.addEventListener("dragleave", () => { stage.style.borderColor = ""; });
      t.addEventListener("drop", (e) => {
        e.preventDefault(); e.stopPropagation(); stage.style.borderColor = "";
        if (e.dataTransfer?.files?.length) handleFile(e.dataTransfer.files[0]);
      });
    }

    // ---------- box interaction ----------
    const pt = (e) => {
      const r = over.getBoundingClientRect();
      return { x: clamp01((e.clientX - r.left) / r.width), y: clamp01((e.clientY - r.top) / r.height) };
    };

    let drag = null;
    // drag = {kind:"new"|"move"|"size", i, el, start:{x,y}, orig:{...}, fx, fy}

    over.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      const boxEl = e.target.closest?.(".bbl-box");
      const handle = e.target.closest?.(".bbl-h");
      const p = pt(e);
      e.preventDefault(); e.stopPropagation();
      over.focus({ preventScroll: true });
      if (boxEl) {
        const i = +boxEl.dataset.i;
        if (sel !== i) { sel = i; updateSelection(); }
        const b = boxes[i];
        if (handle) {
          const hx = handle.dataset.h;
          // the corner opposite the grabbed one stays fixed
          drag = { kind: "size", i, el: boxEl, fx: hx.includes("w") ? b.x + b.w : b.x, fy: hx.includes("n") ? b.y + b.h : b.y };
        } else {
          drag = { kind: "move", i, el: boxEl, start: p, orig: { x: b.x, y: b.y } };
        }
      } else {
        const el = document.createElement("div");
        el.className = "bbl-box sel"; el.style.setProperty("--c", color(boxes.length));
        el.style.pointerEvents = "none";
        over.append(el);
        drag = { kind: "new", el, start: p, cur: { x: p.x, y: p.y, w: 0, h: 0 } };
        geom(el, drag.cur);
      }
      over.setPointerCapture(e.pointerId);
    });

    over.addEventListener("pointermove", (e) => {
      if (!drag) return;
      const p = pt(e);
      if (drag.kind === "new") {
        const c = drag.cur;
        c.x = Math.min(drag.start.x, p.x); c.y = Math.min(drag.start.y, p.y);
        c.w = Math.abs(p.x - drag.start.x); c.h = Math.abs(p.y - drag.start.y);
        geom(drag.el, c);
      } else if (drag.kind === "move") {
        const b = boxes[drag.i];
        b.x = Math.min(1 - b.w, Math.max(0, drag.orig.x + p.x - drag.start.x));
        b.y = Math.min(1 - b.h, Math.max(0, drag.orig.y + p.y - drag.start.y));
        geom(drag.el, b);
      } else {
        const b = boxes[drag.i];
        b.x = Math.min(drag.fx, p.x); b.y = Math.min(drag.fy, p.y);
        b.w = Math.abs(p.x - drag.fx); b.h = Math.abs(p.y - drag.fy);
        geom(drag.el, b);
      }
    });

    function endDrag(e) {
      if (!drag) return;
      const d = drag; drag = null;
      try { over.releasePointerCapture(e.pointerId); } catch { /* already released */ }
      if (d.kind === "new") {
        d.el.remove();
        const c = d.cur;
        if (c.w >= MIN && c.h >= MIN) {
          boxes.push({ x: c.x, y: c.y, w: c.w, h: c.h, caption: "" });
          sel = boxes.length - 1;
          renderBoxes();
          save();
          capIn.focus({ preventScroll: true });
        } else if (sel !== -1) { sel = -1; updateSelection(); }
        return;
      }
      const b = boxes[d.i];
      if (b.w < MIN || b.h < MIN) {          // shrunk to nothing: remove it
        boxes.splice(d.i, 1); sel = -1; renderBoxes();
      }
      save();
    }
    over.addEventListener("pointerup", endDrag);
    over.addEventListener("pointercancel", endDrag);

    over.addEventListener("keydown", (e) => {
      e.stopPropagation();
      if ((e.key === "Delete" || e.key === "Backspace") && sel >= 0) { e.preventDefault(); deleteSelected(); }
    });

    function deleteSelected() {
      if (sel < 0) return;
      boxes.splice(sel, 1); sel = -1; renderBoxes(); save();
    }
    delBtn.addEventListener("click", deleteSelected);
    clearBtn.addEventListener("click", () => { boxes = []; sel = -1; renderBoxes(); save(); });

    capIn.addEventListener("input", () => {
      if (sel < 0) return;
      boxes[sel].caption = capIn.value;
      const t = els[sel]?.querySelector(".bbl-tag"); if (t) t.textContent = tagText(sel);
      save();
    });

    // ---------- widget ----------
    const widget = node.addDOMWidget("bbox_editor", "custom", root, {
      serialize: false,
      hideOnZoom: false,
      getMinHeight: () => stageH(stageW()) + PANEL_H + 40,
    });
    widget.computeSize = (w) => [w, stageH(Math.max(120, (w ?? 400) - 24)) + PANEL_H + 40];

    function restore() {
      loadBoxes();
      layoutStage();
      refreshImage();
      renderBoxes();
    }
    const origConfigure = node.onConfigure;
    node.onConfigure = function () {
      const r = origConfigure ? origConfigure.apply(this, arguments) : undefined;
      restore();
      return r;
    };

    const sz = node.computeSize();
    node.setSize([Math.max(sz[0], 420), Math.max(sz[1], 520)]);
    restore();
  },
});
