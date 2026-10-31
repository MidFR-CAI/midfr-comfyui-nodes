// Full-screen editor for the Image BBox Layout node: draw/move/resize boxes and paint an inpaint mask on one canvas,
// modelled on InvokeAI's canvas (without layers). It lives outside the node graph, so panning the graph, the DOM
// widget scaling and node redraws can't make it laggy: one 2D canvas, redrawn only when something changes.
//
//   Tools:  B brush · E eraser · C box · V / Space-drag / middle-drag pan · wheel zoom · [ ] brush size
//           Ctrl+Z undo · Ctrl+Shift+Z / Ctrl+Y redo · 0 fit · Delete removes the selected box · Esc / Done closes
//
// The caller owns persistence: onBoxes(boxes) fires on every box change, onMask(blobOrNull) once when the editor closes
// with a changed mask.

const COLORS = ["#ff5252", "#40c4ff", "#ffd740", "#69f0ae", "#e040fb", "#ff9100", "#18ffff", "#b2ff59"];
const MASK_MAX = 1280;      // longest side of the mask canvas
const GRID = 16;            // boxes snap to this many OUTPUT pixels (MOSS works on a 16 px grid)
const HANDLE = 6;           // handle half-size, screen px
const UNDO_MAX = 20;
const MASK_RGB = "rgb(214,64,230)";

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const snap16 = (n) => Math.max(16, Math.floor((Math.floor(n) + 8 - 1) / 16) * 16);

// Same sizing as the backend (resize_pair with always_snap): what the node will output.
export function outputSize(w, h, maxMp) {
  const target = (maxMp || 1) * 1e6;
  if (w * h > target) {
    const k = Math.sqrt(target / (w * h));
    return [snap16(Math.floor(w * k)), snap16(Math.floor(h * k))];
  }
  return [snap16(w), snap16(h)];
}

function el(tag, props = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "style") Object.assign(e.style, v);
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null) e[k] = v;
  }
  for (const c of kids) if (c != null && c !== false) e.append(c);
  return e;
}

const CSS = `
.bbm{position:fixed;inset:0;z-index:10050;background:#141414;color:#ddd;font:13px sans-serif;display:flex;flex-direction:column;user-select:none}
.bbm button{font:inherit;color:#eee;background:#333;border:1px solid #555;border-radius:4px;padding:4px 10px;cursor:pointer}
.bbm button:hover{background:#444}.bbm button.on{background:#2d6cdf;border-color:#2d6cdf;color:#fff}
.bbm button:disabled{opacity:.4;cursor:default}
.bbm-top{display:flex;align-items:center;gap:8px;padding:6px 10px;background:#1f1f1f;border-bottom:1px solid #333;flex-wrap:wrap}
.bbm-top .sp{flex:1}.bbm-title{font-weight:600}
.bbm-main{flex:1;display:flex;min-height:0}
.bbm-tools{display:flex;flex-direction:column;gap:6px;padding:8px;background:#1b1b1b;border-right:1px solid #333}
.bbm-tools button{width:42px;height:42px;padding:0;font-size:11px;line-height:1.1}
.bbm-stage{flex:1;position:relative;min-width:0;overflow:hidden}
.bbm-stage canvas{position:absolute;inset:0;width:100%;height:100%;touch-action:none;outline:none}
.bbm-side{width:290px;background:#1b1b1b;border-left:1px solid #333;padding:10px;display:flex;flex-direction:column;gap:8px;overflow:auto}
.bbm-side h4{margin:0;font-size:12px;opacity:.7;font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.bbm-chips{display:flex;flex-wrap:wrap;gap:4px}
.bbm-chip{border-radius:12px !important;border:2px solid var(--c) !important;padding:2px 10px !important}
.bbm-chip.on{background:var(--c) !important;color:#000 !important;font-weight:bold}
.bbm textarea{width:100%;min-height:90px;resize:vertical;background:#222;color:#eee;border:1px solid #555;border-radius:4px;padding:5px 6px;box-sizing:border-box;font:inherit;user-select:text}
.bbm input[type=range]{width:140px}.bbm .num{width:44px;text-align:right}
.bbm .dim{opacity:.6}
`;

export function openEditor(opts) {
  const { imageUrl, maskUrl, maxMp, boxes: initBoxes, onBoxes, onMask, onClose } = opts;
  if (document.querySelector(".bbm")) return null;
  if (!document.getElementById("bbm-style")) document.head.append(el("style", { id: "bbm-style", textContent: CSS }));

  // ---------- state ----------
  let natW = 0, natH = 0, outW = 0, outH = 0;
  let img = null;
  let boxes = (initBoxes || []).map((b) => ({ x: +b.x, y: +b.y, w: +b.w, h: +b.h, caption: b.caption || "", strength: b.strength == null || b.strength === "" ? null : +b.strength }));
  let sel = boxes.length ? 0 : -1;
  let tool = "box";
  let brush = 40;                       // image px (mask canvas px / maskK)
  let view = { s: 1, tx: 0, ty: 0 };
  let space = false, panning = null, painting = null, boxDrag = null, hover = null, mouse = null;
  let undo = [], redo = [];
  let maskCv = null, mctx = null, maskK = 1, maskDirty = false, maskLoaded = false;
  let raf = 0, closed = false;
  const dpr = () => window.devicePixelRatio || 1;

  // ---------- DOM ----------
  const cv = el("canvas", { tabIndex: 0 });
  const ctx = cv.getContext("2d");
  const stage = el("div", { className: "bbm-stage" }, cv);

  const mkTool = (label, title, t) => el("button", { textContent: label, title, onclick: () => setTool(t) });
  const bBox = mkTool("Box", "Box (C)", "box"), bBrush = mkTool("Brush", "Brush (B)", "brush"),
    bErase = mkTool("Erase", "Eraser (E)", "erase"), bView = mkTool("Pan", "Pan (V or hold Space)", "view");
  const toolBtns = { box: bBox, brush: bBrush, erase: bErase, view: bView };
  const tools = el("div", { className: "bbm-tools" }, bBox, bBrush, bErase, bView);

  const sizeIn = el("input", { type: "range", min: 2, max: 400, value: brush, title: "Brush size ([ and ])",
    oninput: () => { brush = +sizeIn.value; sizeNum.value = brush; scheduleDraw(); } });
  const sizeNum = el("input", { type: "number", className: "num", min: 2, max: 400, value: brush,
    onchange: () => { brush = clamp(+sizeNum.value || brush, 2, 400); sizeIn.value = brush; sizeNum.value = brush; scheduleDraw(); } });
  const brushOpts = el("span", { style: { display: "none", alignItems: "center", gap: "6px" } }, "Size", sizeIn, sizeNum, "px");
  const undoBtn = el("button", { textContent: "Undo", title: "Ctrl+Z", onclick: doUndo });
  const redoBtn = el("button", { textContent: "Redo", title: "Ctrl+Shift+Z", onclick: doRedo });
  const clearMaskBtn = el("button", { textContent: "Clear mask", onclick: () => { push(); mctx.clearRect(0, 0, maskCv.width, maskCv.height); maskDirty = true; scheduleDraw(); } });
  const fitBtn = el("button", { textContent: "Fit", title: "0", onclick: () => { fit(); scheduleDraw(); } });
  const zoomLbl = el("span", { className: "dim", textContent: "" });
  const doneBtn = el("button", { textContent: "Done", className: "on", onclick: () => close() });
  const top = el("div", { className: "bbm-top" },
    el("span", { className: "bbm-title", textContent: "Edit boxes & mask" }),
    undoBtn, redoBtn, clearMaskBtn, brushOpts, el("span", { className: "sp" }), zoomLbl, fitBtn, doneBtn);

  const chips = el("div", { className: "bbm-chips" });
  const capIn = el("textarea", { placeholder: "Prompt for this box: what should appear here?",
    oninput: () => { if (sel >= 0) { boxes[sel].caption = capIn.value; emitBoxes(); scheduleDraw(); } } });
  capIn.addEventListener("keydown", (e) => e.stopPropagation());
  const strIn = el("input", { type: "number", min: 0, max: 1, step: 0.05, className: "num", placeholder: "default",
    title: "Denoising strength for just this box. Blank = auto: the Scene Context (Qwen VL) node picks it from the kind of edit (the highest if the box does several), otherwise the Invoke node's strength. Type a number to set it yourself. In testing, edits that change anything need about 0.95-1.",
    oninput: () => { if (sel >= 0) { const v = strIn.value.trim(); boxes[sel].strength = v === "" ? null : clamp(+v || 0, 0, 1); emitBoxes(); } } });
  strIn.addEventListener("keydown", (e) => e.stopPropagation());
  const strRow = el("div", { style: { display: "flex", alignItems: "center", gap: "6px" } }, "Strength", strIn, el("span", { className: "dim", textContent: "(blank = auto)" }));
  const delBtn = el("button", { textContent: "Delete box", onclick: deleteSel });
  const clearBoxesBtn = el("button", { textContent: "Clear boxes", onclick: () => { push(); boxes = []; sel = -1; emitBoxes(); renderSide(); scheduleDraw(); } });
  const info = el("div", { className: "dim" });
  const side = el("div", { className: "bbm-side" },
    el("h4", { textContent: "Boxes" }), chips, capIn, strRow, el("div", { style: { display: "flex", gap: "6px" } }, delBtn, clearBoxesBtn),
    el("h4", { textContent: "Output" }), info,
    el("div", { className: "dim", style: { marginTop: "auto", lineHeight: "1.5" } },
      "Box: drag to draw, drag a box to move, drag a handle to resize (snaps to 16 px).\nBrush/Eraser: paint where the edit should happen.\nWheel zooms, Space-drag pans."));
  side.lastChild.style.whiteSpace = "pre-line";

  const root = el("div", { className: "bbm" }, top, el("div", { className: "bbm-main" }, tools, stage, side));
  document.body.append(root);

  // ---------- helpers ----------
  const color = (i) => COLORS[i % COLORS.length];
  const toScreen = (ix, iy) => [view.tx + ix * view.s, view.ty + iy * view.s];
  const toImage = (sx, sy) => [(sx - view.tx) / view.s, (sy - view.ty) / view.s];
  const evPt = (e) => { const r = cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
  // normalised box edge -> snapped to the output grid
  const snapN = (v, outDim) => clamp(Math.round(v * outDim / GRID) * GRID / outDim, 0, 1);
  const r8 = (v) => Math.round(v * 1e8) / 1e8;

  function emitBoxes() {
    onBoxes?.(boxes.map((b) => ({ x: r8(b.x), y: r8(b.y), w: r8(b.w), h: r8(b.h), caption: b.caption, strength: b.strength })));
    renderInfo();
  }
  function fit() {
    const W = cv.clientWidth, H = cv.clientHeight;
    if (!natW || !W || !H) return;
    const s = Math.min(W / natW, H / natH) * 0.94;
    view = { s, tx: (W - natW * s) / 2, ty: (H - natH * s) / 2 };
  }
  function resize() {
    const d = dpr(), W = cv.clientWidth, H = cv.clientHeight;
    cv.width = Math.max(1, Math.round(W * d)); cv.height = Math.max(1, Math.round(H * d));
    scheduleDraw();
  }

  // ---------- history (mask + boxes together) ----------
  const snapshot = () => ({ m: maskLoaded ? mctx.getImageData(0, 0, maskCv.width, maskCv.height) : null, b: JSON.stringify(boxes), sel });
  function restore(s) {
    if (s.m) mctx.putImageData(s.m, 0, 0);
    boxes = JSON.parse(s.b); sel = Math.min(s.sel, boxes.length - 1);
    maskDirty = true; emitBoxes(); renderSide(); scheduleDraw();
  }
  function push() { undo.push(snapshot()); if (undo.length > UNDO_MAX) undo.shift(); redo = []; syncButtons(); }
  function doUndo() { const s = undo.pop(); if (!s) return; redo.push(snapshot()); restore(s); syncButtons(); }
  function doRedo() { const s = redo.pop(); if (!s) return; undo.push(snapshot()); restore(s); syncButtons(); }
  function syncButtons() { undoBtn.disabled = !undo.length; redoBtn.disabled = !redo.length; }

  // ---------- side panel ----------
  function renderInfo() {
    info.textContent = natW ? `${outW} × ${outH} px (${(outW * outH / 1e6).toFixed(2)} MP)  ·  ${boxes.length} box${boxes.length === 1 ? "" : "es"}` : "";
  }
  function renderSide() {
    chips.replaceChildren(...boxes.map((b, i) => {
      const c = el("button", { className: "bbm-chip" + (i === sel ? " on" : ""), textContent: `Box ${i + 1}`, title: b.caption || "(no prompt)",
        onclick: () => { sel = i; renderSide(); scheduleDraw(); } });
      c.style.setProperty("--c", color(i));
      return c;
    }));
    capIn.disabled = delBtn.disabled = strIn.disabled = sel < 0;
    strIn.value = sel >= 0 && boxes[sel].strength != null ? boxes[sel].strength : "";
    capIn.value = sel >= 0 ? boxes[sel].caption : "";
    renderInfo();
  }
  function deleteSel() {
    if (sel < 0) return;
    push(); boxes.splice(sel, 1); sel = Math.min(sel, boxes.length - 1); emitBoxes(); renderSide(); scheduleDraw();
  }

  function setTool(t) {
    tool = t;
    for (const [k, b] of Object.entries(toolBtns)) b.classList.toggle("on", k === t);
    brushOpts.style.display = t === "brush" || t === "erase" ? "inline-flex" : "none";
    cursor();
    scheduleDraw();
  }
  function cursor() {
    const t = space || panning ? "grab" : tool;
    cv.style.cursor = t === "brush" || t === "erase" ? "none" : t === "view" || t === "grab" ? (panning ? "grabbing" : "grab") : "crosshair";
  }

  // ---------- drawing ----------
  function scheduleDraw() { if (!raf && !closed) raf = requestAnimationFrame(draw); }
  function boxRectPx(b) { return [b.x * natW, b.y * natH, (b.x + b.w) * natW, (b.y + b.h) * natH]; }
  function handles(b) {
    const [x0, y0, x1, y1] = boxRectPx(b).map((v, i) => (i % 2 ? toScreen(0, v)[1] : toScreen(v, 0)[0]));
    const mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
    return { nw: [x0, y0], n: [mx, y0], ne: [x1, y0], e: [x1, my], se: [x1, y1], s: [mx, y1], sw: [x0, y1], w: [x0, my] };
  }
  function draw() {
    raf = 0;
    if (closed) return;
    const d = dpr(), W = cv.width, H = cv.height;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = "#141414"; ctx.fillRect(0, 0, W, H);
    if (!img) return;
    ctx.setTransform(d, 0, 0, d, 0, 0);
    // image + mask in image space
    ctx.save();
    ctx.translate(view.tx, view.ty); ctx.scale(view.s, view.s);
    ctx.imageSmoothingEnabled = view.s < 3;
    ctx.drawImage(img, 0, 0, natW, natH);
    if (maskCv) {
      ctx.globalAlpha = 0.55;
      ctx.drawImage(maskCv, 0, 0, natW, natH);
      ctx.globalAlpha = 1;
    }
    ctx.restore();
    ctx.strokeStyle = "#444"; ctx.lineWidth = 1;
    const [ax, ay] = toScreen(0, 0), [bx, by] = toScreen(natW, natH);
    ctx.strokeRect(ax - 0.5, ay - 0.5, bx - ax + 1, by - ay + 1);
    // boxes (screen space, constant line width)
    boxes.forEach((b, i) => {
      const [x0, y0, x1, y1] = boxRectPx(b);
      const [sx0, sy0] = toScreen(x0, y0), [sx1, sy1] = toScreen(x1, y1);
      const c = color(i), on = i === sel;
      ctx.fillStyle = c + (on ? "33" : "1f"); ctx.fillRect(sx0, sy0, sx1 - sx0, sy1 - sy0);
      ctx.strokeStyle = c; ctx.lineWidth = on ? 3 : 2; ctx.strokeRect(sx0, sy0, sx1 - sx0, sy1 - sy0);
      const label = `Box ${i + 1}` + (b.caption ? `: ${b.caption.slice(0, 28)}${b.caption.length > 28 ? "…" : ""}` : "");
      ctx.font = "12px sans-serif";
      const tw = ctx.measureText(label).width + 10;
      ctx.fillStyle = c; ctx.fillRect(sx0, sy0, Math.min(tw, Math.max(60, sx1 - sx0)), 17);
      ctx.fillStyle = "#000"; ctx.textBaseline = "middle"; ctx.fillText(label, sx0 + 5, sy0 + 9);
      if (on) {
        const dw = Math.round(b.w * outW), dh = Math.round(b.h * outH);
        const t = `${dw} × ${dh}`, w2 = ctx.measureText(t).width + 10;
        ctx.fillStyle = "#000c"; ctx.fillRect(sx1 - w2, sy1 + 3, w2, 17);
        ctx.fillStyle = "#fff"; ctx.fillText(t, sx1 - w2 + 5, sy1 + 12);
        for (const [, [hx, hy]] of Object.entries(handles(b))) {
          ctx.fillStyle = c; ctx.fillRect(hx - HANDLE, hy - HANDLE, HANDLE * 2, HANDLE * 2);
          ctx.strokeStyle = "#000"; ctx.lineWidth = 1; ctx.strokeRect(hx - HANDLE + 0.5, hy - HANDLE + 0.5, HANDLE * 2 - 1, HANDLE * 2 - 1);
        }
      }
    });
    // new-box rubber band is just boxes[last] while dragging; brush ring:
    if ((tool === "brush" || tool === "erase") && mouse && !space && !panning) {
      const r = (brush / 2) * view.s;
      ctx.beginPath(); ctx.arc(mouse[0], mouse[1], Math.max(1, r), 0, Math.PI * 2);
      ctx.strokeStyle = "#000"; ctx.lineWidth = 3; ctx.stroke();
      ctx.strokeStyle = tool === "erase" ? "#ff8a80" : "#fff"; ctx.lineWidth = 1.5; ctx.stroke();
    }
    zoomLbl.textContent = Math.round(view.s * 100) + "%";
  }

  // ---------- hit testing ----------
  function hitHandle(sx, sy) {
    if (sel < 0) return null;
    const hs = handles(boxes[sel]);
    for (const [k, [hx, hy]] of Object.entries(hs)) if (Math.abs(sx - hx) <= HANDLE + 3 && Math.abs(sy - hy) <= HANDLE + 3) return k;
    return null;
  }
  function hitBox(sx, sy) {
    const [ix, iy] = toImage(sx, sy), nx = ix / natW, ny = iy / natH;
    for (let i = boxes.length - 1; i >= 0; i--) {
      const b = boxes[i];
      if (nx >= b.x && nx <= b.x + b.w && ny >= b.y && ny <= b.y + b.h) return i;
    }
    return -1;
  }
  const HCUR = { nw: "nwse-resize", se: "nwse-resize", ne: "nesw-resize", sw: "nesw-resize", n: "ns-resize", s: "ns-resize", e: "ew-resize", w: "ew-resize" };

  // ---------- mask painting ----------
  function strokeMask(a, b) {
    mctx.globalCompositeOperation = tool === "erase" ? "destination-out" : "source-over";
    mctx.strokeStyle = MASK_RGB; mctx.lineCap = "round"; mctx.lineJoin = "round";
    mctx.lineWidth = brush * maskK;
    mctx.beginPath(); mctx.moveTo(a[0] * maskK, a[1] * maskK); mctx.lineTo(b[0] * maskK + 0.01, b[1] * maskK); mctx.stroke();
    maskDirty = true;
  }

  // ---------- pointer ----------
  cv.addEventListener("pointerdown", (e) => {
    if (!img) return;
    cv.focus({ preventScroll: true });
    const [sx, sy] = evPt(e);
    if (e.button === 1 || e.button === 2 || space || tool === "view") {
      e.preventDefault();
      panning = { x: sx, y: sy, tx: view.tx, ty: view.ty };
      cv.setPointerCapture(e.pointerId); cursor(); return;
    }
    if (e.button !== 0) return;
    const [ix, iy] = toImage(sx, sy);
    if (tool === "brush" || tool === "erase") {
      if (!maskLoaded) return;
      push();
      painting = { last: [ix, iy] };
      strokeMask(painting.last, painting.last);
      cv.setPointerCapture(e.pointerId); scheduleDraw(); return;
    }
    // box tool
    const hnd = hitHandle(sx, sy);
    if (hnd) {
      push();
      const b = boxes[sel];
      boxDrag = { kind: "size", h: hnd, x0: b.x, y0: b.y, x1: b.x + b.w, y1: b.y + b.h };
    } else {
      const i = hitBox(sx, sy);
      if (i >= 0) {
        if (sel !== i) { sel = i; renderSide(); }
        push();
        const b = boxes[i];
        boxDrag = { kind: "move", sx, sy, ox: b.x, oy: b.y };
      } else {
        push();
        const nx = clamp(ix / natW, 0, 1), ny = clamp(iy / natH, 0, 1);
        boxes.push({ x: snapN(nx, outW), y: snapN(ny, outH), w: 0, h: 0, caption: "", strength: null });
        sel = boxes.length - 1;
        boxDrag = { kind: "new", ax: nx, ay: ny };
      }
    }
    cv.setPointerCapture(e.pointerId); scheduleDraw();
  });

  cv.addEventListener("pointermove", (e) => {
    const [sx, sy] = evPt(e);
    mouse = [sx, sy];
    if (panning) { view.tx = panning.tx + sx - panning.x; view.ty = panning.ty + sy - panning.y; scheduleDraw(); return; }
    if (painting) {
      const evs = e.getCoalescedEvents ? e.getCoalescedEvents() : [];
      for (const ev of (evs.length ? evs : [e])) {
        const [px, py] = evPt(ev), p = toImage(px, py);
        strokeMask(painting.last, p); painting.last = p;
      }
      scheduleDraw(); return;
    }
    if (boxDrag) {
      const [ix, iy] = toImage(sx, sy), nx = clamp(ix / natW, 0, 1), ny = clamp(iy / natH, 0, 1), b = boxes[sel];
      const minW = GRID / outW, minH = GRID / outH;
      if (boxDrag.kind === "new") {
        const x0 = snapN(Math.min(boxDrag.ax, nx), outW), x1 = snapN(Math.max(boxDrag.ax, nx), outW);
        const y0 = snapN(Math.min(boxDrag.ay, ny), outH), y1 = snapN(Math.max(boxDrag.ay, ny), outH);
        Object.assign(b, { x: x0, y: y0, w: x1 - x0, h: y1 - y0 });
      } else if (boxDrag.kind === "move") {
        const dx = (sx - boxDrag.sx) / view.s / natW, dy = (sy - boxDrag.sy) / view.s / natH;
        b.x = clamp(snapN(boxDrag.ox + dx, outW), 0, 1 - b.w); b.y = clamp(snapN(boxDrag.oy + dy, outH), 0, 1 - b.h);
      } else {
        let { x0, y0, x1, y1 } = boxDrag; const h = boxDrag.h;
        if (h.includes("w")) x0 = Math.min(snapN(nx, outW), x1 - minW);
        if (h.includes("e")) x1 = Math.max(snapN(nx, outW), x0 + minW);
        if (h.includes("n")) y0 = Math.min(snapN(ny, outH), y1 - minH);
        if (h.includes("s")) y1 = Math.max(snapN(ny, outH), y0 + minH);
        Object.assign(b, { x: x0, y: y0, w: x1 - x0, h: y1 - y0 });
      }
      scheduleDraw(); return;
    }
    // hover cursor
    if (tool === "box" && !space) {
      const hnd = hitHandle(sx, sy);
      cv.style.cursor = hnd ? HCUR[hnd] : hitBox(sx, sy) >= 0 ? "move" : "crosshair";
    }
    if (tool === "brush" || tool === "erase") scheduleDraw();
  });

  const endPointer = (e) => {
    try { cv.releasePointerCapture(e.pointerId); } catch { /* already released */ }
    if (panning) { panning = null; cursor(); return; }
    if (painting) { painting = null; return; }
    if (boxDrag) {
      const d = boxDrag; boxDrag = null;
      const b = boxes[sel];
      if (b && (b.w < GRID / outW - 1e-9 || b.h < GRID / outH - 1e-9)) {
        // a click, not a drag: drop the empty box (for "new"), keep the geometry for the others
        if (d.kind === "new") { boxes.splice(sel, 1); undo.pop(); sel = boxes.length ? Math.min(sel, boxes.length - 1) : -1; syncButtons(); }
      }
      emitBoxes(); renderSide(); scheduleDraw();
    }
  };
  cv.addEventListener("pointerup", endPointer);
  cv.addEventListener("pointercancel", endPointer);
  cv.addEventListener("pointerleave", () => { mouse = null; scheduleDraw(); });
  cv.addEventListener("contextmenu", (e) => e.preventDefault());
  cv.addEventListener("wheel", (e) => {
    e.preventDefault();
    const [sx, sy] = evPt(e);
    const f = Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0015));
    const s = clamp(view.s * f, 0.05, 40), k = s / view.s;
    view.tx = sx - (sx - view.tx) * k; view.ty = sy - (sy - view.ty) * k; view.s = s;
    scheduleDraw();
  }, { passive: false });

  // ---------- keys ----------
  function onKey(e) {
    if (closed) return;
    const typing = /^(TEXTAREA|INPUT)$/.test(e.target?.tagName) && e.target.type !== "range";
    if (e.key === "Escape") { e.preventDefault(); e.stopImmediatePropagation(); if (typing) e.target.blur(); else close(); return; }
    if (typing) { e.stopPropagation(); return; }
    const k = e.key.toLowerCase();
    let used = true;
    if ((e.ctrlKey || e.metaKey) && k === "z") { e.shiftKey ? doRedo() : doUndo(); }
    else if ((e.ctrlKey || e.metaKey) && k === "y") doRedo();
    else if (e.ctrlKey || e.metaKey || e.altKey) used = false;
    else if (k === "b") setTool("brush");
    else if (k === "e") setTool("erase");
    else if (k === "c") setTool("box");
    else if (k === "v") setTool("view");
    else if (k === "0" || k === "f") { fit(); scheduleDraw(); }
    else if (k === "[") { brush = clamp(Math.round(brush * 0.85 - 1), 2, 400); sizeIn.value = sizeNum.value = brush; scheduleDraw(); }
    else if (k === "]") { brush = clamp(Math.round(brush * 1.15 + 1), 2, 400); sizeIn.value = sizeNum.value = brush; scheduleDraw(); }
    else if (k === "delete" || k === "backspace") deleteSel();
    else if (k === " ") { if (!space) { space = true; cursor(); } }
    else used = false;
    if (used) { e.preventDefault(); e.stopImmediatePropagation(); } else e.stopPropagation();
  }
  function onKeyUp(e) { if (e.key === " ") { space = false; cursor(); } }
  document.addEventListener("keydown", onKey, true);
  document.addEventListener("keyup", onKeyUp, true);
  root.addEventListener("mousedown", (e) => { if (e.target.closest?.("button")) e.preventDefault(); }); // keep keyboard focus on the canvas
  const ro = new ResizeObserver(() => resize());
  ro.observe(stage);
  window.addEventListener("resize", resize);

  // ---------- load ----------
  function maskIsEmpty() {
    const d = mctx.getImageData(0, 0, maskCv.width, maskCv.height).data;
    for (let i = 3; i < d.length; i += 4) if (d[i] > 3) return false;
    return true;
  }
  const im = new Image();
  im.onload = () => {
    if (closed) return;
    img = im; natW = im.naturalWidth; natH = im.naturalHeight;
    [outW, outH] = outputSize(natW, natH, maxMp);
    const k = Math.min(1, MASK_MAX / Math.max(natW, natH));
    maskCv = document.createElement("canvas"); maskCv.width = Math.max(1, Math.round(natW * k)); maskCv.height = Math.max(1, Math.round(natH * k));
    mctx = maskCv.getContext("2d", { willReadFrequently: true });
    maskK = maskCv.width / natW;
    const done = () => { maskLoaded = true; resize(); fit(); renderSide(); scheduleDraw(); };
    if (maskUrl) {
      const m = new Image();
      m.onload = () => { mctx.drawImage(m, 0, 0, maskCv.width, maskCv.height); done(); };
      m.onerror = done;
      m.src = maskUrl;
    } else done();
  };
  im.onerror = () => { info.textContent = "Can't load the image."; };
  im.src = imageUrl;

  setTool("box"); syncButtons(); renderSide();
  requestAnimationFrame(() => { resize(); cv.focus({ preventScroll: true }); });

  // ---------- close ----------
  async function close() {
    if (closed) return;
    const changed = maskDirty && maskLoaded;
    let blob = null;
    if (changed) {
      if (maskIsEmpty()) blob = null;
      else blob = await new Promise((r) => maskCv.toBlob(r, "image/png"));
    }
    closed = true;
    cancelAnimationFrame(raf);
    document.removeEventListener("keydown", onKey, true);
    document.removeEventListener("keyup", onKeyUp, true);
    window.removeEventListener("resize", resize);
    ro.disconnect();
    root.remove();
    try { if (changed) await onMask?.(blob); } finally { onClose?.(); }
  }
  return { close };
}
