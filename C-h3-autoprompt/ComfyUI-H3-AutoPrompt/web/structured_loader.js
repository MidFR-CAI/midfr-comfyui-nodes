import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/* H3 Structured Media Loader panel -- embedded DIRECTLY on the node's
 * canvas body via addDOMWidget, matching how the original pack's own
 * loader shows its picture/video/audio grid inline rather than behind a
 * button. No popup: everything is visible and editable on the node itself,
 * and every change commits immediately to the hidden loader_state widget
 * (same convention as the original pack's own LoaderPanel.commit()) rather
 * than needing an explicit Save step.
 *
 * Categories: Subject (Complete / Visual Reference / Outfit Reference /
 * Vocal Reference), Shot (Complete / First Frame / Last Frame / Motion /
 * Music / SFX / Reference Images), Audio (Dialogue / Music, spans all
 * shots). Uploads go through ComfyUI's own built-in /upload/image endpoint
 * (accepts any file type despite the name), so there's no custom upload
 * backend to maintain.
 */

const LOADER_NAME = "H3StructuredMediaLoader";
const MAX_PICTURES = 9, MAX_VIDEOS = 3, MAX_AUDIO = 3;
const NODE_W = 640, PANEL_H = 560;
/* Node chrome above/below the panel (title bar, the single output slot, padding). */
const LOADER_CHROME = 56;

const CSS = `
.h3s-panel{width:100%;background:#20242c;color:#d7dbe2;font-family:system-ui,sans-serif;
  border-radius:0 0 8px 8px;display:flex;flex-direction:column;box-sizing:border-box;}
.h3s-headrow{display:flex;align-items:center;gap:10px;padding:6px 10px;
  background:#1b1f27;border-bottom:1px solid #2a2f3a;}
.h3s-counts{display:flex;gap:10px;font-size:10px;color:#9aa3b2;flex:1;}
.h3s-count.over{color:#f07070;font-weight:700;}
.h3s-tabs{display:flex;gap:2px;padding:6px 10px 0;background:#1b1f27;}
.h3s-tab{padding:5px 14px;font-size:11px;border-radius:6px 6px 0 0;cursor:pointer;
  background:#20242c;color:#9aa3b2;}
.h3s-tab.active{background:#262b34;color:#dde2ea;font-weight:600;}
.h3s-body{flex:1;overflow:auto;padding:10px;background:#262b34;min-height:0;}
.h3s-row{background:#1d212a;border:1px solid #2a2f3a;border-radius:8px;
  padding:10px;margin-bottom:10px;}
.h3s-rowhead{display:flex;justify-content:space-between;align-items:center;
  margin-bottom:8px;}
.h3s-rowtitle{font-size:11px;font-weight:700;color:#e0a94c;}
.h3s-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:8px;}
.h3s-zone{border:2px dashed #333a45;border-radius:7px;padding:6px;min-height:88px;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:5px;cursor:pointer;background:#12151b;position:relative;}
.h3s-zone.dragover{border-color:#3d7fbf;background:#16202c;}
.h3s-zone.filled{border-style:solid;border-color:#3a4250;padding:0 3px 3px;overflow:hidden;justify-content:flex-start;}
.h3s-zone.disabled{opacity:.35;pointer-events:none;}
.h3s-zone img,.h3s-zone video{width:100%;height:64px;object-fit:cover;}
.h3s-zonelabel{font-size:9px;color:#8890a0;text-align:center;padding:0 4px;}
.h3s-zonelabel.big{font-size:10px;color:#aab2c0;}
/* Name sits BELOW the artwork rather than overlaying it -- as an overlay it
 * covered the thumbnail it was labelling. */
.h3s-zonename{font-size:8px;color:#9aa3b2;padding:2px 3px 0;white-space:nowrap;
  overflow:hidden;text-overflow:ellipsis;text-align:center;}
.h3s-tools{position:absolute;top:2px;right:2px;display:flex;gap:2px;z-index:2;}
.h3s-tool{background:#000a;color:#fff;border:none;border-radius:4px;width:17px;
  height:17px;font-size:10px;line-height:1;cursor:pointer;padding:0;}
.h3s-tool:hover{background:#1f4f7d;}
.h3s-trimmed{position:absolute;bottom:2px;left:2px;background:#000a;color:#e0a94c;
  font-size:8px;border-radius:3px;padding:1px 4px;}
.h3s-view{position:fixed;inset:0;background:rgba(6,8,12,.86);z-index:10100;display:flex;
  align-items:center;justify-content:center;}
.h3s-viewbox{background:#191c22;border:1px solid #303642;border-radius:10px;
  max-width:92vw;max-height:90vh;display:flex;flex-direction:column;overflow:hidden;
  font-family:system-ui,sans-serif;color:#d7dbe2;}
.h3s-viewhead{display:flex;align-items:center;gap:10px;padding:8px 12px;background:#1b1f27;
  border-bottom:1px solid #2a2f3a;font-size:12px;}
.h3s-viewhead .sp{flex:1;}
.h3s-viewbody{padding:10px;overflow:auto;display:flex;flex-direction:column;
  align-items:center;gap:8px;}
.h3s-viewbody img,.h3s-viewbody video{max-width:86vw;max-height:64vh;object-fit:contain;}
.h3s-viewfoot{display:flex;align-items:center;gap:8px;padding:8px 12px;background:#1b1f27;
  border-top:1px solid #2a2f3a;font-size:11px;flex-wrap:wrap;}
.h3s-viewfoot .sp{flex:1;}
.h3s-viewfoot input[type=number]{width:62px;background:#12151b;color:#dde2ea;
  border:1px solid #2e3440;border-radius:5px;padding:2px 5px;font-size:11px;}
.h3s-viewfoot select{background:#12151b;color:#dde2ea;border:1px solid #2e3440;
  border-radius:5px;padding:2px 5px;font-size:11px;}
.h3s-zoneremove{position:absolute;top:1px;right:1px;background:#0009;color:#fff;
  border:none;border-radius:4px;width:16px;height:16px;font-size:10px;cursor:pointer;
  line-height:1;}
.h3s-btn{background:#232833;color:#c9cfda;border:1px solid #333a45;border-radius:6px;
  padding:4px 10px;font-size:10px;cursor:pointer;}
.h3s-btn:hover{background:#2c3340;}
.h3s-btn.primary{background:#1f4f7d;border-color:#3d7fbf;color:#dbeafe;}
.h3s-btn.small{padding:2px 7px;font-size:9px;}
.h3s-empty{padding:18px;text-align:center;color:#6b7484;font-size:11px;}
.h3s-outfitlist{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px;}
.h3s-outfititem{width:88px;}
.h3s-outfitcap{font-size:8px;color:#8890a0;text-align:center;margin-top:2px;}
.h3s-cbar{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:8px;}
.h3s-cbar label{font-size:10px;color:#9aa3b2;display:flex;align-items:center;gap:4px;}
.h3s-cnote{font-size:10px;color:#8890a0;margin:4px 0;}
.h3s-cwarn{font-size:10px;color:#e0a94c;margin:4px 0;}
.h3s-panelrow{display:flex;gap:10px;background:#1d212a;border:1px solid #2a2f3a;
  border-radius:8px;padding:8px;margin-bottom:8px;}
.h3s-pthumb{flex:0 0 130px;height:110px;border-radius:6px;border:1px solid #333a45;
  background-repeat:no-repeat;background-color:#12151b;}
.h3s-pbody{flex:1;min-width:0;}
.h3s-phead{display:flex;align-items:center;gap:8px;font-size:11px;font-weight:700;
  color:#e0a94c;margin-bottom:5px;}
.h3s-phead .sp{flex:1;}
.h3s-secs{font-size:9px;color:#8890a0;font-weight:400;}
.h3s-brow{display:flex;gap:5px;align-items:center;margin-bottom:4px;}
.h3s-brow input[type=text]{flex:1;min-width:0;background:#12151b;color:#dde2ea;
  border:1px solid #2e3440;border-radius:5px;padding:3px 6px;font-size:10px;}
.h3s-brow input[type=number]{width:42px;background:#12151b;color:#dde2ea;
  border:1px solid #2e3440;border-radius:5px;padding:3px 5px;font-size:10px;}
.h3s-chunkbar{display:flex;align-items:center;gap:8px;background:#1b1f27;
  border:1px solid #2a2f3a;border-radius:8px;padding:8px;margin-bottom:10px;
  font-size:11px;}
.h3s-chunkbar .sp{flex:1;}
.h3s-page{position:relative;display:inline-block;max-width:100%;user-select:none;}
.h3s-page{border:1px solid #333a45;border-radius:6px;overflow:hidden;line-height:0;
  position:relative;display:inline-block;}
/* Fit the page to the editor window. A full-resolution comic page is far taller
   than any browser viewport, and scrolling while drawing boxes is miserable.
   Coordinates are converted from the displayed size, so scaling is safe. */
.h3s-page img{display:block;max-width:100%;max-height:74vh;width:auto;height:auto;}
.h3s-page.drawing{cursor:crosshair;}
.h3s-rect{position:absolute;border:2px solid #e0a94c;background:#e0a94c18;
  border-radius:3px;cursor:move;box-sizing:border-box;}
.h3s-grip{position:absolute;right:-5px;bottom:-5px;width:12px;height:12px;
  background:#e0a94c;border:1px solid #141820;border-radius:2px;cursor:nwse-resize;}
.h3s-rect.bub .h3s-grip{background:#7ee5a8;}
.h3s-rect.sel{border-color:#7fb2e5;background:#7fb2e522;}
.h3s-rect.bub{border-color:#7ee5a8;background:#7ee5a818;border-style:dashed;}
.h3s-rect b{position:absolute;top:-9px;left:-9px;background:#e0a94c;color:#141820;
  font-size:9px;font-weight:700;border-radius:50%;width:17px;height:17px;
  display:flex;align-items:center;justify-content:center;}
.h3s-rect.sel b{background:#7fb2e5;}
.h3s-rect.bub b{background:#7ee5a8;border-radius:4px;}
.h3s-edover{position:fixed;inset:0;background:rgba(6,8,12,.86);z-index:10120;
  display:flex;align-items:center;justify-content:center;}
.h3s-edbox{width:min(1100px,95vw);height:min(88vh,900px);background:#191c22;
  border:1px solid #303642;border-radius:10px;display:flex;flex-direction:column;
  overflow:hidden;font-family:system-ui,sans-serif;color:#d7dbe2;}
.h3s-edhead{display:flex;align-items:center;gap:10px;padding:9px 13px;background:#1b1f27;
  border-bottom:1px solid #2a2f3a;font-size:12px;font-weight:600;}
.h3s-edhead .sp{flex:1;}
.h3s-edbody{flex:1;overflow:auto;padding:12px;display:flex;gap:14px;align-items:flex-start;}
.h3s-edleft{flex:0 0 auto;}
.h3s-edright{flex:1;min-width:280px;max-height:100%;overflow:auto;}
.h3s-mode{display:flex;gap:6px;align-items:center;margin:6px 0;flex-wrap:wrap;}
.h3s-mode .on{background:#1f4f7d;border-color:#3d7fbf;color:#dbeafe;}
.h3s-frame{position:relative;display:inline-block;max-width:100%;cursor:crosshair;}
.h3s-frame img{max-width:100%;display:block;border-radius:6px;border:1px solid #333a45;}
.h3s-mark{position:absolute;width:22px;height:22px;margin:-11px 0 0 -11px;border-radius:50%;
  background:#1f4f7dcc;border:2px solid #7fb2e5;color:#fff;font-size:11px;font-weight:700;
  display:flex;align-items:center;justify-content:center;pointer-events:none;}
.h3s-prow{display:flex;gap:6px;align-items:center;margin-bottom:5px;background:#1d212a;
  border:1px solid #2a2f3a;border-radius:6px;padding:5px;}
.h3s-prow .num{width:20px;height:20px;border-radius:50%;background:#1f4f7d;color:#fff;
  font-size:10px;font-weight:700;display:flex;align-items:center;justify-content:center;
  flex:0 0 auto;}
.h3s-prow input[type=text]{flex:1;min-width:0;background:#12151b;color:#dde2ea;
  border:1px solid #2e3440;border-radius:5px;padding:3px 6px;font-size:10px;}
.h3s-prow input[type=number]{width:44px;background:#12151b;color:#dde2ea;
  border:1px solid #2e3440;border-radius:5px;padding:3px 5px;font-size:10px;}
.h3s-chunkbar select{background:#12151b;color:#dde2ea;border:1px solid #2e3440;
  border-radius:5px;padding:2px 6px;font-size:11px;}
`;

let cssDone = false;
function injectCSS() {
  if (cssDone) return;
  cssDone = true;
  const s = document.createElement("style");
  s.textContent = CSS;
  document.head.append(s);
}

function el(tag, props = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "style" && typeof v === "object") Object.assign(e.style, v);
    else if (k === "class") e.className = v;
    else if (k.startsWith("on") && typeof v === "function")
      e.addEventListener(k.slice(2).toLowerCase(), v);
    else { try { e[k] = v; } catch (err) { e.setAttribute(k, v); } }
  }
  for (const c of children.flat(Infinity)) {
    if (c == null) continue;
    e.append(c.nodeType ? c : document.createTextNode(c));
  }
  return e;
}

function kindOf(file) {
  const ext = (file.name || "").split(".").pop().toLowerCase();
  if (["png", "jpg", "jpeg", "webp", "gif", "bmp"].includes(ext)) return "picture";
  if (["mp4", "webm", "mov", "mkv", "avi"].includes(ext)) return "video";
  if (["wav", "mp3", "flac", "ogg", "m4a", "aac"].includes(ext)) return "audio";
  if (file.type?.startsWith("image/")) return "picture";
  if (file.type?.startsWith("video/")) return "video";
  if (file.type?.startsWith("audio/")) return "audio";
  return null;
}

function viewURL(annotated) {
  let name = String(annotated || ""), type = "input";
  const m = name.match(/^(.*)\s\[(input|output|temp)\]$/);
  if (m) { name = m[1]; type = m[2]; }
  let sub = "";
  const slash = name.lastIndexOf("/");
  if (slash >= 0) { sub = name.slice(0, slash); name = name.slice(slash + 1); }
  return api.apiURL(`/view?filename=${encodeURIComponent(name)}` +
    `&subfolder=${encodeURIComponent(sub)}&type=${type}`);
}

async function uploadFile(file) {
  const kind = kindOf(file);
  if (!kind) throw new Error(`Unrecognized file type: ${file.name}`);
  const form = new FormData();
  form.append("image", file);
  form.append("subfolder", "h3_structured");
  const resp = await api.fetchApi("/upload/image", { method: "POST", body: form });
  if (!resp.ok) throw new Error(`Upload failed (${resp.status})`);
  const data = await resp.json();
  const name = data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
  return { file: `${name} [${data.type || "input"}]`, kind, name: file.name };
}

function countMedia(state) {
  let p = 0, v = 0, a = 0;
  const bump = (item, hint) => {
    if (!item) return;
    const k = item.kind || hint;
    if (k === "picture") p++; else if (k === "video") v++; else if (k === "audio") a++;
  };
  bump(state.storyboard, "picture");
  // In comic mode the panels of the ACTIVE CHUNK become the images, not the page
  // itself -- the page is only the source they're cropped from.
  if (state.replace?.enabled && state.replace.video) {
    v += 1;
    if (state.replace.keep_audio !== false) a += 1;
    for (const pp of state.replace.people || []) if (pp.voice) a += 1;
  }
  if (state.comic?.enabled) {
    const ch = (state.comic.chunks || [])[state.comic.chunk_index || 0];
    p += ch ? ch.length : (state.comic.panels || []).length;
  }
  for (const s of state.subjects || []) {
    bump(s.complete);
    for (const r of s.visual_refs || []) bump(r, "picture");
    for (const r of s.outfit_refs || []) bump(r, "picture");
    bump(s.vocal_ref, "audio");
  }
  for (const sh of state.shots || []) {
    bump(sh.complete, "video"); bump(sh.continue_from, "video");
    if (sh.continue_from?.continue_audio) bump({ kind: "audio" }, "audio");
    bump(sh.first_frame, "picture"); bump(sh.last_frame, "picture");
    bump(sh.motion, "video"); bump(sh.music, "audio"); bump(sh.sfx, "audio");
    for (const r of sh.reference_images || []) bump(r, "picture");
  }
  for (const r of state.audio?.dialogue || []) bump(r, "audio");
  for (const r of state.audio?.music || []) bump(r, "audio");
  return { p, v, a };
}

/* The loader's node-level default resize, mirrored here so a file's own picker
 * can show what "default" currently resolves to. Kept module-level rather than
 * threaded through every dropZone call. */
let _loaderDefaultSide = 0;

const RESIZE_CHOICES = [["", "default"], [0, "original"], [2048, "2048"],
                         [1536, "1536"], [1280, "1280"], [1024, "1024"], [768, "768"]];



/* Full-size viewer, doubling as the resize/trim editor -- the same job the
 * reference loader's crop/trim dialog does. Images get resize; video gets
 * resize plus a start/end trim; audio gets trim. Trim is stored as
 * {start, end} seconds, matching the reference loader's own shape. */
function openViewer(item, onEdit) {
  injectCSS();
  const url = viewURL(item.file);
  const isVid = item.kind === "video";
  const isAud = item.kind === "audio";

  const media = isVid
    ? el("video", { src: url, controls: true, autoplay: false })
    : isAud
      ? el("audio", { src: url, controls: true, style: { width: "420px" } })
      : el("img", { src: url });

  const foot = [];
  if (!isAud) {
    // A file with no setting of its own shows "default" and follows the node's
    // default-resize dropdown. Previously it displayed "original" while actually
    // inheriting, so the control disagreed with what the backend did.
    const curSide = item.max_side == null ? "" : String(item.max_side);
    foot.push(el("span", {}, "resize"),
      el("select", { onchange: (e) => {
          const v = e.target.value;
          item.max_side = v === "" ? null : (parseInt(v, 10) || 0);
          onEdit?.();
        } },
        RESIZE_CHOICES.map(([v, lb]) => el("option", { value: String(v),
          // Label the inherit option with what it currently resolves to.
          selected: String(v) === curSide },
          v === ""
            ? `default (${_loaderDefaultSide ? _loaderDefaultSide + "px" : "original"})`
            : lb))));
  }
  if (isVid || isAud) {
    item.trim = item.trim || {};
    const startIn = el("input", { type: "number", min: 0, step: 0.1,
      value: item.trim.start ?? "", placeholder: "0",
      onchange: (e) => {
        const v = parseFloat(e.target.value);
        item.trim.start = Number.isNaN(v) ? null : Math.max(0, v);
        onEdit?.();
      } });
    const endIn = el("input", { type: "number", min: 0, step: 0.1,
      value: item.trim.end ?? "", placeholder: "end",
      onchange: (e) => {
        const v = parseFloat(e.target.value);
        item.trim.end = Number.isNaN(v) ? null : Math.max(0, v);
        onEdit?.();
      } });
    // Fill the end box from the real duration once the browser knows it, so
    // "end" isn't a guess.
    media.addEventListener("loadedmetadata", () => {
      item.duration = media.duration || item.duration;
      if (!endIn.value && item.duration) endIn.placeholder = item.duration.toFixed(1);
    });
    foot.push(el("span", { style: { marginLeft: "8px" } }, "trim"), startIn,
      el("span", {}, "\u2013"), endIn, el("span", {}, "s"),
      el("button", { class: "h3s-btn", title: "Use the player's current position as the start",
        onclick: () => {
          item.trim.start = Math.round((media.currentTime || 0) * 10) / 10;
          startIn.value = item.trim.start; onEdit?.();
        } }, "\u2196 start here"),
      el("button", { class: "h3s-btn", title: "Use the player's current position as the end",
        onclick: () => {
          item.trim.end = Math.round((media.currentTime || 0) * 10) / 10;
          endIn.value = item.trim.end; onEdit?.();
        } }, "end here \u2197"),
      el("button", { class: "h3s-btn", onclick: () => {
        item.trim = {}; startIn.value = ""; endIn.value = ""; onEdit?.();
      } }, "clear"));
  }

  const close = () => { window.removeEventListener("keydown", onKey); overlay.remove(); };
  const onKey = (e) => { if (e.key === "Escape") close(); };
  window.addEventListener("keydown", onKey);

  const overlay = el("div", { class: "h3s-view",
    onmousedown: (e) => { if (e.target === overlay) close(); } },
    el("div", { class: "h3s-viewbox" },
      el("div", { class: "h3s-viewhead" },
        el("span", {}, item.name || item.file),
        el("span", { class: "sp" }),
        el("button", { class: "h3s-btn", onclick: close }, "\u2715")),
      el("div", { class: "h3s-viewbody" }, media),
      el("div", { class: "h3s-viewfoot" }, foot,
        el("span", { class: "sp" }),
        el("button", { class: "h3s-btn primary", onclick: close }, "Done"))));
  document.body.append(overlay);
}

function dropZone({ label, accept, value, onSet, onRemove, disabled, onEdit }) {
  const zone = el("div", { class: "h3s-zone" + (value ? " filled" : "") + (disabled ? " disabled" : "") });

  function render() {
    zone.replaceChildren();
    if (value) {
      const media = value.kind === "picture"
        ? el("img", { src: viewURL(value.file) })
        : value.kind === "video"
          ? el("video", { src: viewURL(value.file), muted: true })
          : el("div", { style: { fontSize: "24px", color: "#7ea7d8" } }, "\u266a");
      const t = value.trim || {};
      const trimmed = (t.start || t.end) ? `${(t.start || 0).toFixed(1)}\u2013${
        t.end ? t.end.toFixed(1) : "end"}s` : null;
      const kids = [media,
        // Top-right tool cluster, matching the reference loader's layout.
        el("div", { class: "h3s-tools" },
          el("button", { class: "h3s-tool", title: "View full size / resize / trim",
            onclick: (e) => { e.stopPropagation(); openViewer(value, () => { onEdit?.(); render(); }); } },
            "\u2922"),
          el("button", { class: "h3s-tool", title: "Remove",
            onclick: (e) => { e.stopPropagation(); onRemove(); value = null; render(); } },
            "\u2715")),
        trimmed ? el("div", { class: "h3s-trimmed" }, trimmed) : null,
        el("div", { class: "h3s-zonename", title: value.name || value.file },
          value.name || value.file)];
      zone.append(...kids.filter(Boolean));
    } else {
      zone.append(el("div", { class: "h3s-zonelabel big" }, label),
        el("div", { class: "h3s-zonelabel" }, "drop or click"));
    }
  }
  render();

  const input = el("input", { type: "file", style: { display: "none" },
    onchange: async (e) => {
      const f = e.target.files[0];
      if (!f) return;
      try {
        const item = await uploadFile(f);
        if (accept && item.kind !== accept) { alert(`Expected ${accept}, got ${item.kind}`); return; }
        value = item; onSet(item); render();
      } catch (err) { alert(err.message); }
    } });

  zone.append(input);
  zone.addEventListener("click", () => {
    if (disabled) return;
    // A filled tile opens its preview; an empty one opens the file picker.
    if (value) openViewer(value, () => { onEdit?.(); render(); });
    else input.click();
  });
  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("dragover"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
  zone.addEventListener("drop", async (e) => {
    e.preventDefault(); zone.classList.remove("dragover");
    if (disabled) return;
    const f = e.dataTransfer.files[0];
    if (!f) return;
    try {
      const item = await uploadFile(f);
      if (accept && item.kind !== accept) { alert(`Expected ${accept}, got ${item.kind}`); return; }
      value = item; onSet(item); render();
    } catch (err) { alert(err.message); }
  });
  return zone;
}

function subjectRow(subject, index, onChange, onRemove, rerenderRow, shotCount) {
  const completeKind = subject.complete?.kind || null;
  const showVisual = !completeKind;
  const showVocal = completeKind !== "video";
  const showOutfitFromTwo = completeKind === "video";

  const grid = el("div", { class: "h3s-grid" });
  grid.append(dropZone({
    label: "Complete", value: subject.complete,
    onSet: (v) => { subject.complete = v; onChange(true); rerenderRow(); },
    onRemove: () => { subject.complete = null; onChange(true); rerenderRow(); },
    onEdit: onChange,
  }));

  if (showVocal) {
    // accept is deliberately unset: a video is fine here, its audio track is
    // extracted on the Python side (alloc.audio() already handles kind="video").
    grid.append(dropZone({ label: "Vocal Reference (audio or video)", value: subject.vocal_ref,
      onSet: (v) => { subject.vocal_ref = v; onChange(); },
      onRemove: () => { subject.vocal_ref = null; onChange(); }, onEdit: onChange }));
  }

  // Visual References: same self-rendering flex-wrap list as Outfit References
  // below. Previously these were pushed into the grid inside a bare wrapper div,
  // which made every visual reference stack vertically inside a single grid cell
  // instead of sitting as fixed squares side by side -- and because the list had
  // no render function of its own, adding one didn't redraw until a tab switch
  // rebuilt the whole body.
  const visualWrap = el("div", { class: "h3s-outfitlist" });
  function renderVisuals() {
    visualWrap.replaceChildren();
    (subject.visual_refs || []).forEach((r, i) => {
      visualWrap.append(el("div", { class: "h3s-outfititem" },
        dropZone({ label: "Visual Reference", accept: "picture", value: r, onSet: () => {},
          onRemove: () => { subject.visual_refs.splice(i, 1); onChange(); renderVisuals(); },
          onEdit: onChange })));
    });
    visualWrap.append(el("div", { class: "h3s-outfititem" },
      dropZone({ label: "+ Visual Reference", accept: "picture", value: null,
        onSet: (v) => { (subject.visual_refs ||= []).push(v); onChange(); renderVisuals(); },
        onRemove: () => {} })));
  }
  if (showVisual) renderVisuals();

  // Each outfit carries its OWN scope, chosen from the dropdown under it.
  // Previously the scope came from list position (outfit #2 meant Shot 2), so
  // dressing Shot 2 meant first adding a Shot 1 outfit you didn't want.
  const outfitWrap = el("div", { class: "h3s-outfitlist" });

  function scopeChoices() {
    const n = Math.max(typeof shotCount === "function" ? shotCount() : 0, 1);
    const opts = [["all", "all shots"]];
    for (let i = 1; i <= n; i++) opts.push([String(i), `Shot ${i}`]);
    return opts;
  }

  function scopeSelect(ref) {
    // A Complete-video subject already wears shot 1's outfit, so default a new
    // outfit there to Shot 2 instead of overriding the entire video.
    const cur = ref.scope != null && ref.scope !== ""
      ? String(ref.scope) : (showOutfitFromTwo ? "2" : "all");
    if (ref.scope == null || ref.scope === "") ref.scope = cur;
    return el("select", {
      title: "Which shot(s) this outfit applies to.",
      style: { width: "100%", marginTop: "2px", fontSize: "9px", background: "#12151b",
               color: "#dde2ea", border: "1px solid #2e3440", borderRadius: "4px" },
      onchange: (e) => { ref.scope = e.target.value; onChange(); } },
      scopeChoices().map(([v, lb]) => el("option", { value: v, selected: v === cur }, lb)));
  }

  function renderOutfits() {
    outfitWrap.replaceChildren();
    (subject.outfit_refs || []).forEach((r, i) => {
      outfitWrap.append(el("div", { class: "h3s-outfititem" },
        dropZone({ label: "Outfit", accept: "picture", value: r, onSet: () => {},
          onRemove: () => { subject.outfit_refs.splice(i, 1); onChange(); renderOutfits(); },
          onEdit: onChange }),
        scopeSelect(r)));
    });
    outfitWrap.append(el("div", { class: "h3s-outfititem" },
      dropZone({ label: "+ Outfit", accept: "picture", value: null,
        onSet: (v) => {
          v.scope = showOutfitFromTwo ? "2" : "all";
          (subject.outfit_refs ||= []).push(v); onChange(); renderOutfits();
        }, onRemove: () => {} })));
  }
  renderOutfits();

  return el("div", { class: "h3s-row" },
    el("div", { class: "h3s-rowhead" },
      el("span", { class: "h3s-rowtitle" }, `Subject ${index + 1}`),
      el("button", { class: "h3s-btn small", onclick: onRemove }, "\u2715 remove")),
    grid,
    showVisual ? el("div", {},
      el("div", { style: { fontSize: "9px", color: "#8890a0", marginTop: "6px" } }, "Visual References"),
      visualWrap) : null,
    el("div", { style: { fontSize: "9px", color: "#8890a0", marginTop: "6px" } }, "Outfit References"),
    outfitWrap);
}

function shotRow(shot, index, onChange, onRemove, rerenderRow) {
  // The old "Complete (replacement)" slot lived here before Replace became its
  // own tab. Two routes to the same result was the source of real confusion, so
  // it's gone -- character swaps belong in the Replace tab, which also handles
  // the person mapping this slot never could. A shot from an older saved
  // workflow that still carries one is handled below rather than silently
  // ignored or silently applied.
  const isComplete = !!shot.complete;
  const grid = el("div", { class: "h3s-grid" });
  // Continuation only makes sense on the opening shot: the target video resumes
  // from the END of this clip, so a later shot has nothing to resume into.
  if (index === 0 && !isComplete) {
    grid.append(el("div", {},
      dropZone({ label: "Continue From (video)", accept: "video", value: shot.continue_from,
        onSet: (v) => { shot.continue_from = v; onChange(true); rerenderRow(); },
        onRemove: () => { shot.continue_from = null; onChange(true); rerenderRow(); },
        onEdit: onChange }),
      shot.continue_from
        ? el("label", { style: { fontSize: "9px", color: "#8890a0", display: "flex",
                                 alignItems: "center", gap: "3px", marginTop: "2px" },
            title: "Carry the clip's audible character (room tone, ambience, musical feel) "
                 + "into the new audio. Cited as an audio REFERENCE, not a copy, per the "
                 + "guide. Uses one of the 3 audio slots." },
            el("input", { type: "checkbox", checked: !!shot.continue_from.continue_audio,
              onchange: (e) => { shot.continue_from.continue_audio = e.target.checked;
                                 onChange(true); } }),
            "continue audio")
        : null));
  }

  const others = [["First Frame", "first_frame", "picture"], ["Last Frame", "last_frame", "picture"],
    ["Motion", "motion", "video"], ["Music", "music", "audio"], ["SFX", "sfx", "audio"]];
  for (const [label, key, kind] of others) {
    grid.append(dropZone({ label, accept: kind, value: shot[key], disabled: isComplete,
      onSet: (v) => { shot[key] = v; onChange(true); },
      onRemove: () => { shot[key] = null; onChange(true); }, onEdit: onChange }));
  }

  const refWrap = el("div", { class: "h3s-outfitlist" });
  function renderRefs() {
    refWrap.replaceChildren();
    (shot.reference_images || []).forEach((r, i) => {
      refWrap.append(dropZone({ label: "Reference", accept: "picture", value: r, disabled: isComplete,
        onSet: () => {}, onEdit: onChange,
        onRemove: () => { shot.reference_images.splice(i, 1); onChange(true); renderRefs(); } }));
    });
    refWrap.append(dropZone({ label: "+ Reference Image", accept: "picture", value: null, disabled: isComplete,
      onSet: (v) => { (shot.reference_images ||= []).push(v); onChange(true); renderRefs(); }, onRemove: () => {} }));
  }
  renderRefs();

  return el("div", { class: "h3s-row" },
    el("div", { class: "h3s-rowhead" },
      el("span", { class: "h3s-rowtitle" }, `Shot ${index + 1}`),
      el("button", { class: "h3s-btn small", onclick: onRemove }, "\u2715 remove")),
    grid,
    isComplete ? el("div", { style: { fontSize: "9px", color: "#e0a94c", marginTop: "6px" } },
      "Complete replaces this shot's footage entirely -- other inputs are disabled.") : null,
    isComplete ? null : el("div", {},
      el("div", { style: { fontSize: "9px", color: "#8890a0", marginTop: "6px" } }, "Reference Images"),
      refWrap));
}

function audioSection(state, onChange) {
  const dlgWrap = el("div", { class: "h3s-outfitlist" });
  const musWrap = el("div", { class: "h3s-outfitlist" });
  function renderList(wrap, key, label) {
    wrap.replaceChildren();
    (state.audio[key] || []).forEach((r, i) => {
      wrap.append(dropZone({ label, accept: "audio", value: r, onSet: () => {},
        onRemove: () => { state.audio[key].splice(i, 1); onChange(true); renderList(wrap, key, label); } }));
    });
    wrap.append(dropZone({ label: `+ ${label}`, accept: "audio", value: null,
      onSet: (v) => { (state.audio[key] ||= []).push(v); onChange(true); renderList(wrap, key, label); },
      onRemove: () => {} }));
  }
  renderList(dlgWrap, "dialogue", "Dialogue");
  renderList(musWrap, "music", "Music");
  return el("div", { class: "h3s-row" },
    el("div", { class: "h3s-rowtitle" }, "Audio (spans all shots)"),
    el("div", { style: { fontSize: "9px", color: "#8890a0", margin: "6px 0 3px" } }, "Dialogue"), dlgWrap,
    el("div", { style: { fontSize: "9px", color: "#8890a0", margin: "8px 0 3px" } }, "Music"), musWrap);
}

/** Builds the always-visible panel and mounts it on the node via
 * addDOMWidget, matching the original pack's own inline-grid convention
 * instead of a popup. Every change commits immediately to the hidden
 * loader_state widget (same as the original pack's LoaderPanel.commit()). */
/* ---- Comic tab -------------------------------------------------------- */

async function h3Post(path, body) {
  const resp = await api.fetchApi(`/h3_autoprompt/${path}`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok || data.error) throw new Error(data.error || `${path} failed (${resp.status})`);
  return data;
}

async function comicPost(route, body) {
  const resp = await api.fetchApi(`/h3_autoprompt/comic/${route}`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok || data.error) throw new Error(data.error || `${route} failed (${resp.status})`);
  return data;
}

/* Shows a panel region without cutting new files: the whole page is the
 * background, scaled and offset so the panel's rectangle fills the thumbnail. */
function panelThumb(pageFile, page, p) {
  const boxW = 130, boxH = 110;
  const scale = Math.min(boxW / Math.max(1, p.w), boxH / Math.max(1, p.h));
  return el("div", { class: "h3s-pthumb", style: {
    backgroundImage: `url(${viewURL(pageFile)})`,
    backgroundSize: `${page.w * scale}px ${page.h * scale}px`,
    backgroundPosition: `${-p.x * scale}px ${-p.y * scale}px`,
  } });
}

/* ---- Replace tab ------------------------------------------------------ */

/* A character-only edit is its own job, not a Shot field: the shot structure
 * comes entirely from the source clip, and the only real decision is who
 * replaces whom. Marking is manual because face detection finds nothing on
 * animated or stylised footage -- it's offered as a hint, never relied on. */
function replaceSection(state, commit, rerender) {
  state.replace = state.replace || {};
  const r = state.replace;
  r.people = r.people || [];
  const wrap = el("div", {});
  const status = el("div", { class: "h3s-cnote" });

  function whereFor(fx) {
    return fx < 0.38 ? "the person on the left"
      : fx > 0.62 ? "the person on the right" : "the person in the centre";
  }

  wrap.append(el("div", { class: "h3s-cbar" },
    el("label", {},
      el("input", { type: "checkbox", checked: !!r.enabled,
        onchange: (e) => {
          r.enabled = e.target.checked;
          if (r.enabled && state.comic) state.comic.enabled = false;
          commit(); rerender();
        } }),
      "replace mode"),
    el("label", { title: "Keep the source clip's audio. Only the performer's "
                       + "appearance changes, so the words and timing still fit." },
      el("input", { type: "checkbox", checked: r.keep_audio !== false,
        onchange: (e) => { r.keep_audio = e.target.checked; commit(); } }),
      "keep original audio")));

  if (!r.enabled) {
    wrap.append(el("div", { class: "h3s-cnote" },
      "Turn replace mode on to swap the performer in a clip while keeping its "
      + "camera, motion, timing and environment exactly as they are. It replaces "
      + "the Shot tab."));
    return wrap;
  }

  wrap.append(el("div", { class: "h3s-outfitlist" },
    dropZone({ label: "Source video", accept: "video", value: r.video,
      onSet: (v) => { r.video = v; r.frame = null; r.people = []; commit(); rerender(); },
      onRemove: () => { r.video = null; r.frame = null; commit(); rerender(); },
      onEdit: commit })));

  if (!r.video?.file) { wrap.append(status); return wrap; }

  wrap.append(el("button", { class: "h3s-btn primary", onclick: async () => {
    status.textContent = "Reading first frame\u2026";
    try {
      const d = await h3Post("replace/frame", { file: r.video.file });
      Object.assign(r, { frame: d.frame, frameW: d.w, frameH: d.h,
                          srcDuration: d.duration, hint: d.hint });
      // Detected faces are a starting point only -- the user confirms or edits.
      if ((d.faces || []).length && !r.people.length) {
        r.people = d.faces.map((f) => ({
          fx: (f.x + f.w / 2) / d.w, fy: (f.y + f.h / 2) / d.h,
          where: f.where ? `the person ${f.where}` : "", subject: null,
        }));
      }
      commit(); rerender();
    } catch (err) { status.textContent = err.message; }
  } }, r.frame ? "Reload frame" : "Load first frame"), status);

  // Nothing here can watch the source clip, so the person's own account of it is
  // the only reliable description -- and the guide requires a real scene
  // description, not a list of reference relationships.
  wrap.append(el("div", { style: { marginTop: "8px" } },
    el("div", { style: { fontSize: "10px", fontWeight: "700", color: "#e0a94c" } },
      "Describe the source video"),
    el("div", { style: { fontSize: "9px", color: "#8890a0", margin: "2px 0 4px" } },
      "What happens in the clip: framing, where each person is, the setting and "
      + "lighting, every action in order, and the camera. This is used almost "
      + "word-for-word \u2014 the model only formats it and inserts labels."),
    el("textarea", { value: r.description || "",
      placeholder: "e.g. A medium tracking shot follows two people along a rainy "
                 + "pavement. The man on the left gestures as he talks; the woman "
                 + "on the right laughs and looks away. The camera drifts right...",
      style: { width: "100%", minHeight: "88px", background: "#12151b",
               color: "#dde2ea", border: "1px solid #2e3440", borderRadius: "6px",
               padding: "5px 7px", fontSize: "11px", fontFamily: "inherit",
               resize: "vertical", boxSizing: "border-box" },
      onchange: (e) => { r.description = e.target.value; commit(); } }),
    !((r.description || "").trim())
      ? el("div", { class: "h3s-cwarn" },
          "\u26a0 Without a description the model has to guess the clip's contents "
          + "from a few sampled frames, which is where replacements usually go wrong.")
      : null));

  if (!r.frame) return wrap;

  if (r.hint) wrap.append(el("div", { class: "h3s-cwarn" }, "\u26a0 " + r.hint));
  wrap.append(el("div", { class: "h3s-cnote" },
    "Click the frame to mark each person being replaced, then say which Subject "
    + "takes their place."
    + (r.srcDuration ? `  Source is ${r.srcDuration}s long.` : "")));

  const frame = el("div", { class: "h3s-frame", onclick: (e) => {
    const img = e.currentTarget.querySelector("img");
    const box = img.getBoundingClientRect();
    const fx = (e.clientX - box.left) / box.width;
    const fy = (e.clientY - box.top) / box.height;
    if (fx < 0 || fx > 1 || fy < 0 || fy > 1) return;
    r.people.push({ fx, fy, where: whereFor(fx), subject: null });
    commit(); rerender();
  } }, el("img", { src: viewURL(r.frame) }));
  r.people.forEach((p, i) => {
    frame.append(el("div", { class: "h3s-mark",
      style: { left: `${(p.fx || 0) * 100}%`, top: `${(p.fy || 0) * 100}%` } },
      String(i + 1)));
  });
  wrap.append(frame);

  r.people.forEach((p, i) => {
    wrap.append(el("div", { class: "h3s-prow" },
      el("div", { class: "num" }, String(i + 1)),
      el("input", { type: "text", value: p.where || "",
        placeholder: "who this is, in words (e.g. the woman on the left in red)",
        title: "How the model should identify this person in the frame. Position "
             + "alone is enough when people are clearly separated.",
        onchange: (e) => { p.where = e.target.value; commit(); } }),
      el("span", { style: { fontSize: "9px", color: "#8890a0" } }, "\u2190 Subject"),
      el("input", { type: "number", min: 1, max: 12, value: p.subject || "",
        title: "Which Subject replaces this person. Leave blank to keep them "
             + "exactly as they are in the source.",
        onchange: (e) => {
          p.subject = e.target.value ? parseInt(e.target.value, 10) : null;
          commit();
        } }),
      // What actually transfers from that subject's references. All four by
      // default; unticking some is how you stop, say, the wardrobe coming
      // across when only the face should.
      ...(() => {
        p.keep = p.keep || { face: true, hair: true, body: true, wardrobe: true };
        return [["face", "face"], ["hair", "hair"], ["body", "build"],
                ["wardrobe", "fit"]].map(([k, lbl]) =>
          el("label", { style: { fontSize: "9px", color: "#8890a0", display: "flex",
                                 alignItems: "center", gap: "2px" },
              title: `Carry ${k} over from this subject's references.` },
            el("input", { type: "checkbox", checked: p.keep[k] !== false,
              onchange: (e) => { p.keep[k] = e.target.checked; commit(); } }), lbl));
      })(),
      el("button", { class: "h3s-btn small",
        onclick: () => { r.people.splice(i, 1); commit(); rerender(); } }, "\u2715")),
      // Voice per replaced person, so a two-hander doesn't end up with one
      // timbre on both speakers. Accepts a video too -- its track is extracted.
      el("div", { style: { display: "flex", alignItems: "center", gap: "6px",
                           marginTop: "4px", paddingLeft: "26px" } },
        el("label", { style: { fontSize: "9px", color: "#8890a0", display: "flex",
                               alignItems: "center", gap: "3px" },
            title: "Re-dub this person in the Vocal Reference set on their Subject "
                 + "in the Subject tab. Tick this rather than loading the same "
                 + "file twice." },
          el("input", { type: "checkbox", checked: !!p.use_subject_voice,
            onchange: (e) => { p.use_subject_voice = e.target.checked; commit(); rerender(); } }),
          "use subject's voice"),
        el("span", { style: { fontSize: "9px", color: "#8890a0" } }, "or"),
        el("div", { style: { width: "96px" } },
          dropZone({ label: "Voice ref", value: p.voice,
            onSet: (v) => { p.voice = v; commit(); },
            onRemove: () => { p.voice = null; commit(); },
            onEdit: commit })),
        el("span", { style: { fontSize: "9px", color: "#8890a0", flex: "1" } },
          p.voice ? "this person's lines are re-dubbed in that timbre"
                  : "optional \u2014 audio or video; leave empty to keep the "
                    + "original voice")));
  });

  if (!r.people.length) {
    wrap.append(el("div", { class: "h3s-cnote" },
      "No one marked yet \u2014 click the frame above."));
  } else if (!r.people.some((p) => p.subject)) {
    wrap.append(el("div", { class: "h3s-cwarn" },
      "\u26a0 No Subject assigned to anyone yet, so nothing will be replaced."));
  } else {
    wrap.append(el("div", { class: "h3s-cnote" },
      "Everyone marked gets their own <Subject> tag. Leave the Subject number "
      + "blank to mark someone who should be left exactly as they are \u2014 they'll "
      + "be preserved by name instead of being caught by the replacement."));
  }
  return wrap;
}

/* The panel editor lives in its own window. Mounted in the node body it kept a
 * full-page canvas plus every overlay alive inside the graph's redraw path,
 * which made the whole node sluggish. */
/* The per-panel list: durations, balloon text, speaker and ordering. Shared by
 * the editor window and the tab summary so they can't drift apart. */
function comicPanelList(state, commit, rerender, status) {
  const c = state.comic;
  const wrap = el("div", {});
  const chunks = c.chunks || [];
  const active = new Set((chunks[c.chunk_index || 0] || []).map(Number));
  const move = (arr, i, d) => {
    const j = i + d;
    if (j < 0 || j >= arr.length) return;
    [arr[i], arr[j]] = [arr[j], arr[i]];
    commit(); rerender();
  };
  async function retime() {
    try {
      const d = await comicPost("retime", { panels: c.panels,
                                             merge_aspect: !!c.merge_aspect });
      c.panels.forEach((p, i) => { p.seconds = d.seconds[i]; });
      c.chunks = d.chunks; c.chunkSeconds = d.chunk_seconds;
      c.chunkAspects = d.chunk_aspects;
      commit();
    } catch (err) { if (status) status.textContent = err.message; }
  }
  if (chunks.length > 1) {
    wrap.append(el("div", { class: "h3s-chunkbar" },
      el("span", {}, "\u{1F3AC} generating"),
      el("select", { onchange: (e) => {
        c.chunk_index = parseInt(e.target.value, 10) || 0; commit(); rerender();
      } }, chunks.map((ch, i) => el("option", {
        value: String(i), selected: i === (c.chunk_index || 0) },
        `Chunk ${i + 1} of ${chunks.length} \u2014 panels ${ch.join(", ")} \u2014 ${
          (c.chunkSeconds || [])[i] ?? "?"}s \u2014 ${
          (c.chunkAspects || [])[i] ?? "?"}`))),
      el("span", { class: "sp" }),
      el("span", { style: { fontSize: "10px", color: "#8890a0" } },
        (new Set(c.chunkAspects || [])).size > 1
          ? "Split partly by orientation: one generation is one frame size, so "
            + "landscape and portrait panels can't share a clip."
          : "Too long for one clip \u2014 run each chunk separately."),
      el("label", { style: { fontSize: "10px", color: "#8890a0", display: "flex",
                             alignItems: "center", gap: "3px", marginLeft: "8px" },
          title: "Keep panels together even when their orientation changes. They'll "
               + "share one frame size, so some will be letterboxed or cropped." },
        el("input", { type: "checkbox", checked: !!c.merge_aspect,
          onchange: (e) => { c.merge_aspect = e.target.checked; commit(); retime(); } }),
        "merge orientations")));
  }
  // The cast lives here rather than in the Subject tab: which characters appear
  // in which PANEL is part of the page's structure, and a global subject row
  // can't express that.
  c.subjects = c.subjects || [];
  const cast = el("div", { class: "h3s-row" },
    el("div", { class: "h3s-rowtitle" }, "Cast"),
    el("div", { class: "h3s-cnote" },
      "One entry per recurring character. The reference image fixes their look "
      + "across panels; the voice fixes how they sound. Tick them on each panel "
      + "below, and use their number when assigning balloons."));
  c.subjects.forEach((cs, si) => {
    cast.append(el("div", { class: "h3s-prow" },
      el("div", { class: "num" }, String(si + 1)),
      el("input", { type: "text", value: cs.name || "", placeholder: "name",
        style: { flex: "0 0 110px" },
        onchange: (e) => { cs.name = e.target.value; commit(); } }),
      el("div", { style: { width: "86px" } },
        dropZone({ label: "Look", accept: "picture", value: cs.ref,
          onSet: (v) => { cs.ref = v; commit(); },
          onRemove: () => { cs.ref = null; commit(); }, onEdit: commit })),
      el("div", { style: { width: "86px" } },
        dropZone({ label: "Voice", value: cs.voice,
          onSet: (v) => { cs.voice = v; commit(); },
          onRemove: () => { cs.voice = null; commit(); }, onEdit: commit })),
      el("button", { class: "h3s-btn small",
        onclick: () => { c.subjects.splice(si, 1); commit(); rerender(); } }, "\u2715")));
  });
  cast.append(el("button", { class: "h3s-btn", onclick: () => {
    c.subjects.push({ name: "", ref: null, voice: null }); commit(); rerender();
  } }, "+ Add character"));
  wrap.append(cast);

  if (!c.panels.length) {
    wrap.append(el("div", { class: "h3s-cnote" },
      "No panels yet \u2014 drag on the page to outline the first one."));
    return wrap;
  }
  c.panels.forEach((p, pi) => {
    const inChunk = !chunks.length || active.has(pi + 1);
    const bubbles = el("div", {});
    (p.bubbles || []).forEach((b, bi) => {
      bubbles.append(el("div", { class: "h3s-brow" },
        el("span", { style: { fontSize: "9px", color: "#8890a0" } }, `${pi + 1}.${bi + 1}`),
        el("button", { class: "h3s-btn small", title: "Speak earlier",
          onclick: () => move(p.bubbles, bi, -1) }, "\u2191"),
        el("button", { class: "h3s-btn small", title: "Speak later",
          onclick: () => move(p.bubbles, bi, 1) }, "\u2193"),
        el("input", { type: "text", value: b.text || "",
          placeholder: "what this balloon says (verbatim)",
          onchange: (e) => { b.text = e.target.value; commit(); retime(); } }),
        el("button", { class: "h3s-btn small", title: "Read this balloon with OCR",
          onclick: async (ev) => {
            const btn = ev.target; btn.textContent = "\u2026";
            try {
              const d = await comicPost("ocr", { file: c.page.file,
                box: { x: p.x + b.x, y: p.y + b.y, w: b.w, h: b.h } });
              if (d.text) { b.text = d.text; commit(); await retime(); rerender(); }
              else {
                status.textContent = d.reason
                  || "Nothing legible in that box \u2014 type it in.";
                btn.textContent = "OCR";
              }
            } catch (err) { status.textContent = err.message; btn.textContent = "OCR"; }
          } }, "OCR"),
        el("input", { type: "number", min: 1, max: 12, value: b.speaker_subject || "",
          title: "Which Subject says this.", disabled: !!b.is_sound,
          onchange: (e) => {
            b.speaker_subject = e.target.value ? parseInt(e.target.value, 10) : null;
            commit();
          } }),
        // A "cough" or "CRASH" balloon is a sound, not a line -- without this it
        // would be spoken aloud by a character.
        el("label", { style: { fontSize: "9px", color: "#8890a0", display: "flex",
                               alignItems: "center", gap: "3px" },
            title: "This balloon is a sound effect, not speech. It becomes an "
                 + "audible sound in the scene instead of a spoken line." },
          el("input", { type: "checkbox", checked: !!b.is_sound,
            onchange: (e) => { b.is_sound = e.target.checked;
                               if (b.is_sound) b.speaker_subject = null;
                               commit(); rerender(); } }), "sfx"),
        el("button", { class: "h3s-btn small",
          onclick: () => { p.bubbles.splice(bi, 1); commit(); retime(); rerender(); } },
          "\u2715")));
    });

    // Per-panel description and manual duration. A written description is used
    // verbatim and suppresses auto-tagging for that panel; a manual duration
    // overrides the speech estimate.
    const meta = el("div", { style: { marginBottom: "5px" } },
      el("textarea", { value: p.description || "",
        placeholder: `What happens in panel ${pi + 1} (used almost word-for-word)`,
        style: { width: "100%", minHeight: "44px", background: "#12151b",
                 color: "#dde2ea", border: "1px solid #2e3440", borderRadius: "5px",
                 padding: "4px 6px", fontSize: "10px", fontFamily: "inherit",
                 resize: "vertical", boxSizing: "border-box" },
        onchange: (e) => { p.description = e.target.value; commit(); rerender(); } }),
      el("div", { style: { display: "flex", alignItems: "center", gap: "5px",
                           marginTop: "3px", flexWrap: "wrap" } },
        el("span", { style: { fontSize: "9px", color: "#8890a0" },
          title: "Seconds for this panel. Leave blank and it is estimated from how "
               + "long the dialogue takes to say." }, "duration"),
        el("input", { type: "number", min: 0.3, step: 0.1, value: p.duration ?? "",
          style: { width: "52px", background: "#12151b", color: "#dde2ea",
                   border: "1px solid #2e3440", borderRadius: "4px",
                   padding: "1px 4px", fontSize: "9px" },
          onchange: (e) => {
            const v = parseFloat(e.target.value);
            p.duration = Number.isNaN(v) || v <= 0 ? null : v;
            commit(); retime();
          } }),
        el("span", { style: { fontSize: "9px", color: "#8890a0" } },
          p.duration ? "s (manual)" : `s \u2014 auto: ${p.seconds ?? "?"}s`),
        // Characters placed in this panel, with where they were clicked.
        ...((p.marks || []).map((m) => {
          const cs = (c.subjects || [])[m.n - 1];
          const side = m.fx < 0.38 ? "left" : m.fx > 0.62 ? "right" : "centre";
          return el("span", { style: { fontSize: "9px", color: "#7ee5a8",
                                       display: "flex", alignItems: "center", gap: "2px" },
              title: "Placed with the Place character tool. Click again to move." },
            `${cs?.name || "#" + m.n} (${side})`,
            el("button", { class: "h3s-btn small",
              onclick: () => { p.marks = p.marks.filter((x) => x !== m);
                               commit(); rerender(); } }, "\u2715"));
        })),
        !(p.marks || []).length && (c.subjects || []).length
          ? el("span", { style: { fontSize: "9px", color: "#8890a0" } },
              "no characters placed \u2014 use Place character")
          : null,
        (p.description || "").trim()
          ? el("span", { style: { fontSize: "9px", color: "#7ee5a8" } },
              "\u2713 description used verbatim")
          : el("span", { style: { fontSize: "9px", color: "#8890a0" } },
              "no description \u2014 falls back to auto-tagging")));

    wrap.append(el("div", { class: "h3s-panelrow", style: inChunk ? {} : { opacity: ".4" } },
      // `img` belongs to the editor window, not here -- referencing it threw a
      // ReferenceError that blanked the whole tab. The page dimensions are
      // cached on the state when the editor measures them.
      panelThumb(c.page.file, c.pageSize || { w: 1, h: 1 }, p),
      el("div", { class: "h3s-pbody" },
        el("div", { class: "h3s-phead" }, `Panel ${pi + 1}`,
          el("span", { class: "h3s-secs" }, `${p.seconds ?? "?"}s`),
          el("span", { class: "sp" }),
          inChunk ? null : el("span", { class: "h3s-secs" }, "(another chunk)"),
          el("button", { class: "h3s-btn small", title: "Move this panel earlier",
            onclick: () => move(c.panels, pi, -1) }, "\u2191"),
          el("button", { class: "h3s-btn small", title: "Move this panel later",
            onclick: () => move(c.panels, pi, 1) }, "\u2193"),
          el("button", { class: "h3s-btn small",
            onclick: () => { c.panels.splice(pi, 1); commit(); retime(); rerender(); } },
            "\u2715")),
        meta,
        meta,
        (p.bubbles || []).length ? bubbles
          : el("div", { style: { fontSize: "9px", color: "#8890a0" } },
              "No balloons \u2014 switch to Draw balloon and drag around one."))));
  });
  return wrap;
}

function openComicEditor(state, commit, onClose) {
  injectCSS();
  const c = state.comic;
  const status = el("div", { class: "h3s-cnote" });
  const left = el("div", { class: "h3s-edleft" });
  const right = el("div", { class: "h3s-edright" });

  // The hoisted canvas was written against comicSection's scope. Supply the
  // same names here rather than rewriting every call site.
  const rerender = () => rebuild();
  async function retime() {
    try {
      const d = await comicPost("retime", { panels: c.panels,
                                             merge_aspect: !!c.merge_aspect });
      c.panels.forEach((p, i) => { p.seconds = d.seconds[i]; });
      c.chunks = d.chunks; c.chunkSeconds = d.chunk_seconds;
      c.chunkAspects = d.chunk_aspects;
      commit();
    } catch (err) { status.textContent = err.message; }
  }
  let sized = false;     // guards the onload -> rebuild -> onload loop

  function rebuild() {
    left.replaceChildren();
    right.replaceChildren();
  // Panels and balloons are drawn by hand. Auto-detection stays available but
    // reads real pages badly enough that it can't be the main route -- a wrong
    // split or a mis-read line is worse than an empty box.
    const modeBtn = (m, label, title) => el("button", {
      class: "h3s-btn" + (c.mode === m ? " on" : ""), title,
      onclick: () => { c.mode = m; commit(); rerender(); } }, label);

      left.append(el("div", { class: "h3s-mode" },
      modeBtn("panel", "Draw panel", "Drag on the page to outline a panel."),
      modeBtn("bubble", "Draw balloon", "Drag inside a panel to outline a balloon."),
    modeBtn("subject", "Place character",
            "Click inside a panel to mark where a character stands in it."),
    c.mode === "subject"
      ? el("select", { style: { fontSize: "10px", background: "#12151b",
                                 color: "#dde2ea", border: "1px solid #2e3440",
                                 borderRadius: "4px" },
          onchange: (e) => { c.selSubject = parseInt(e.target.value, 10); commit(); } },
          (c.subjects || []).length
            ? (c.subjects || []).map((cs, si) => el("option", {
                value: String(si + 1), selected: (c.selSubject || 1) === si + 1 },
                `${si + 1}. ${cs.name || "unnamed"}`))
            : [el("option", { value: "0" }, "add a character first")])
      : null,
      el("span", { style: { fontSize: "10px", color: "#8890a0" } },
        c.mode === "panel" ? "Drag to outline each panel, in reading order."
                            : "Drag around a balloon; it joins the panel containing it."),
      el("span", { style: { flex: "1" } }),
      el("button", { class: "h3s-btn",
        title: "Optional. Often wrong on real pages -- treat whatever it adds as a draft.",
        onclick: async () => {
          status.textContent = "Detecting\u2026";
          try {
            const d = await comicPost("analyze", { file: c.page.file, rtl: !!c.rtl,
                                                  merge_aspect: !!c.merge_aspect });
            Object.assign(c, { pageSize: d.page, panels: d.panels, ocrDetail: d.ocr_detail });
            await retime(); rerender();
          } catch (err) { status.textContent = err.message; }
        } }, "Try auto-detect"),
      el("button", { class: "h3s-btn",
        onclick: () => { c.panels = []; commit(); rerender(); } }, "Clear all")));

    const img = el("img", { src: viewURL(c.page.file),
      onload: () => {
        // Cache the real dimensions on the state: the tab's panel list renders
        // thumbnails without the editor open and has no <img> of its own.
        c.pageSize = { w: img.naturalWidth || 1, h: img.naturalHeight || 1 };
        // Redraw once. Without the guard rebuild() recreates the <img>, which
        // fires onload again, forever.
        if (sized) return;
        sized = true;
        redrawOverlay();
      } });
    const surface = el("div", { class: "h3s-page drawing" }, img);

    /* Live geometry, read at the moment of use rather than cached.
     * pageSize used to default to 1 before onload fired, which made the scale
     * factor ~1000x wrong and put every box nowhere near the cursor. Reading
     * naturalWidth/clientWidth off the element avoids that entirely, and the
     * border now lives on the container so the two measurements agree. */
    const geom = () => {
      const nw = img.naturalWidth || 1;
      const cw = img.clientWidth || nw;
      return { k: nw / cw, nw, nh: img.naturalHeight || 1 };
    };
    const pctOf = (v, total) => `${(v / Math.max(1, total)) * 100}%`;

    function commitAndRedraw() { commit(); rerender(); }

    /* Shared move/resize behaviour for any box already on the page. Without this
     * a box drawn slightly wrong had to be deleted and redrawn. */
    function attachDrag(node, get, set) {
      const startDrag = (e, mode) => {
        e.preventDefault(); e.stopPropagation();
        const { k } = geom();
        const o = { ...get() };
        const ox = e.clientX, oy = e.clientY;
        const { nw, nh } = geom();
        const onMove = (ev) => {
          const dx = (ev.clientX - ox) * k, dy = (ev.clientY - oy) * k;
          const v = mode === "move"
            ? { ...o, x: Math.round(o.x + dx), y: Math.round(o.y + dy) }
            : { ...o, w: Math.max(8, Math.round(o.w + dx)),
                       h: Math.max(8, Math.round(o.h + dy)) };
          set(v);
          // Style the live element directly. Rebuilding the overlay here removed
          // the very node under the cursor, so the drag lost its target and the
          // box ended up somewhere else entirely.
          Object.assign(node.style, {
            left: `${(v.x / nw) * 100}%`, top: `${(v.y / nh) * 100}%`,
            width: `${(v.w / nw) * 100}%`, height: `${(v.h / nh) * 100}%` });
        };
        const onUp = () => {
          window.removeEventListener("mousemove", onMove);
          window.removeEventListener("mouseup", onUp);
          commit();               // persist; the overlay is already correct
          redrawOverlay();        // renumber/reattach without a full re-render
        };
        window.addEventListener("mousemove", onMove);
        window.addEventListener("mouseup", onUp);
      };
      node.addEventListener("mousedown", (e) => {
        // Panels blanket the page, so in balloon mode a press on a panel is the
        // start of a new balloon, not a request to move the panel. In character
        // mode the click must reach the surface handler -- startDrag calls
        // stopPropagation, which is why placing a character did nothing.
        if (c.mode === "subject") return;
        if (c.mode === "bubble" && !node.classList.contains("bub")) return;
        startDrag(e, "move");
      });
      const grip = el("div", { class: "h3s-grip" });
      grip.addEventListener("mousedown", (e) => startDrag(e, "resize"));
      node.append(grip);
    }

    function redrawOverlay() {
      surface.querySelectorAll(".h3s-rect").forEach((n) => n.remove());
      const { nw, nh } = geom();
      c.panels.forEach((p, pi) => {
        const box = el("div", { class: "h3s-rect", title: `Panel ${pi + 1}` },
          el("b", {}, String(pi + 1)));
        Object.assign(box.style, { left: pctOf(p.x, nw), top: pctOf(p.y, nh),
                                    width: pctOf(p.w, nw), height: pctOf(p.h, nh) });
        attachDrag(box, () => p, (v) => Object.assign(p, v));
        surface.append(box);
        (p.bubbles || []).forEach((b, bi) => {
          const bb = el("div", { class: "h3s-rect bub",
            title: `Balloon ${pi + 1}.${bi + 1}` }, el("b", {}, `${pi + 1}.${bi + 1}`));
          Object.assign(bb.style, { left: pctOf(p.x + b.x, nw), top: pctOf(p.y + b.y, nh),
                                     width: pctOf(b.w, nw), height: pctOf(b.h, nh) });
          // Bubble coords are panel-relative; convert in and out so dragging a
          // balloon doesn't fight with the panel it belongs to.
          attachDrag(bb,
            () => ({ x: p.x + b.x, y: p.y + b.y, w: b.w, h: b.h }),
            (v) => Object.assign(b, { x: v.x - p.x, y: v.y - p.y, w: v.w, h: v.h }));
          surface.append(bb);
        });
      });
    }
    redrawOverlay();

    // Drag on empty space to draw a new box. The move/up listeners are attached
    // only for the duration of a drag -- binding them to window on every rebuild
    // accumulated a fresh pair each time the overlay redrew.
    surface.addEventListener("mousedown", (e) => {
      const hit = e.target.closest(".h3s-rect");
      if (c.mode === "subject") {
        // A single click, not a drag: mark where the selected character stands.
        const n = c.selSubject || 1;
        if (!(c.subjects || []).length) {
          status.textContent = "Add a character to the Cast list first."; return;
        }
        const r0 = img.getBoundingClientRect();
        const k0 = (img.naturalWidth || 1) / Math.max(1, r0.width);
        const px = (e.clientX - r0.left) * k0, py = (e.clientY - r0.top) * k0;
        const pi = c.panels.findIndex((pp) => px >= pp.x && px <= pp.x + pp.w
                                            && py >= pp.y && py <= pp.y + pp.h);
        if (pi < 0) { status.textContent = "Click inside a panel."; return; }
        const pp = c.panels[pi];
        pp.marks = (pp.marks || []).filter((m) => m.n !== n);
        pp.marks.push({ n, fx: (px - pp.x) / Math.max(1, pp.w),
                            fy: (py - pp.y) / Math.max(1, pp.h) });
        commit(); redrawOverlay();
        right.replaceChildren(comicPanelList(state, commit, rebuild, status));
        e.preventDefault();
        return;
      }
      // In balloon mode a panel underneath is something to draw OVER, not move.
      if (hit && !(c.mode === "bubble" && !hit.classList.contains("bub"))) return;
      const r = img.getBoundingClientRect();
      const sx0 = e.clientX - r.left, sy0 = e.clientY - r.top;
      const ghost = el("div", { class: "h3s-rect" + (c.mode === "bubble" ? " bub" : "") });
      surface.append(ghost);
      e.preventDefault();

      const onMove = (ev) => {
        const cx = ev.clientX - r.left, cy = ev.clientY - r.top;
        Object.assign(ghost.style, {
          left: `${Math.min(sx0, cx)}px`, top: `${Math.min(sy0, cy)}px`,
          width: `${Math.abs(cx - sx0)}px`, height: `${Math.abs(cy - sy0)}px` });
      };
      const onUp = (ev) => {
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
        ghost.remove();
        const cx = ev.clientX - r.left, cy = ev.clientY - r.top;
        const sx = Math.min(sx0, cx), sy = Math.min(sy0, cy);
        const sw = Math.abs(cx - sx0), sh = Math.abs(cy - sy0);
        if (sw < 8 || sh < 8) return;            // ignore stray clicks
        // Displayed size -> page pixels, so the scaled-to-fit view is harmless.
        const k = (img.naturalWidth || 1) / Math.max(1, r.width);
        const box = { x: Math.round(sx * k), y: Math.round(sy * k),
                      w: Math.round(sw * k), h: Math.round(sh * k) };
        if (c.mode === "panel") {
          c.panels.push({ ...box, bubbles: [], seconds: 1.2 });
        } else {
          const mx = box.x + box.w / 2, my = box.y + box.h / 2;
          const idx = c.panels.findIndex((p) => mx >= p.x && mx <= p.x + p.w
                                              && my >= p.y && my <= p.y + p.h);
          if (idx < 0) {
            status.textContent = "Draw the panel first, then its balloons.";
            return;
          }
          const p = c.panels[idx];
          (p.bubbles = p.bubbles || []).push({ x: box.x - p.x, y: box.y - p.y,
                                               w: box.w, h: box.h, text: "" });
        }
        commit();
        redrawOverlay();
        right.replaceChildren(comicPanelList(state, commit, rebuild, status));
      };
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    });

      left.append(surface);
    right.append(comicPanelList(state, commit, rebuild, status));
  }

  const close = () => {
    window.removeEventListener("keydown", onKey);
    overlay.remove();
    commit();
    onClose?.();
  };
  const onKey = (e) => { if (e.key === "Escape") close(); };
  window.addEventListener("keydown", onKey);

  const overlay = el("div", { class: "h3s-edover",
    onmousedown: (e) => { if (e.target === overlay) close(); } },
    el("div", { class: "h3s-edbox" },
      el("div", { class: "h3s-edhead" },
        el("span", {}, "\u270e Panel editor"),
        status,
        el("span", { class: "sp" }),
        el("button", { class: "h3s-btn primary", onclick: close }, "Done")),
      el("div", { class: "h3s-edbody" }, left, right)));
  document.body.append(overlay);
  rebuild();
}

function comicSection(state, commit, rerender) {
  state.comic = state.comic || {};
  const c = state.comic;
  c.panels = c.panels || [];
  if (c.mode === undefined) c.mode = "panel";
  const wrap = el("div", {});
  const status = el("div", { class: "h3s-cnote" });

  wrap.append(el("div", { class: "h3s-cbar" },
    el("label", {},
      el("input", { type: "checkbox", checked: !!c.enabled,
        onchange: (e) => {
          c.enabled = e.target.checked;
          // Comic and Replace both take over the Shot tab; leaving both on
          // produced a state neither could build from.
          if (c.enabled && state.replace) state.replace.enabled = false;
          commit(); rerender();
        } }),
      "comic mode"),
    el("label", { title: "Only affects the order auto-detect suggests. Panels you "
                       + "draw keep the order you draw them in." },
      el("input", { type: "checkbox", checked: !!c.rtl,
        onchange: (e) => { c.rtl = e.target.checked; commit(); } }),
      "right-to-left (manga)")));

  if (!c.enabled) {
    wrap.append(el("div", { class: "h3s-cnote" },
      "Turn comic mode on to build shots from a comic page. It replaces the Shot tab: "
      + "each panel becomes a shot, its balloons are erased from the artwork, and the "
      + "dialogue is spoken aloud instead of shown as text."));
    return wrap;
  }

  wrap.append(el("div", { class: "h3s-outfitlist" },
    dropZone({ label: "Comic page", accept: "picture", value: c.page,
      onSet: (v) => { c.page = v; c.panels = []; commit(); rerender(); },
      onRemove: () => { c.page = null; c.panels = []; commit(); rerender(); },
      onEdit: commit })), status);

  if (!c.page?.file) return wrap;

  wrap.append(el("div", { class: "h3s-cbar" },
    el("button", { class: "h3s-btn primary",
      onclick: () => openComicEditor(state, commit, rerender) },
      "\u270e Open panel editor\u2026"),
    el("span", { style: { fontSize: "10px", color: "#8890a0" } },
      c.panels.length
        ? `${c.panels.length} panel(s), ${
            c.panels.reduce((n, p) => n + (p.bubbles || []).length, 0)} balloon(s)`
        : "no panels yet")));

  async function retime() {
    try {
      const d = await comicPost("retime", { panels: c.panels,
                                             merge_aspect: !!c.merge_aspect });
      c.panels.forEach((p, i) => { p.seconds = d.seconds[i]; });
      c.chunks = d.chunks; c.chunkSeconds = d.chunk_seconds;
      c.chunkAspects = d.chunk_aspects;
      commit();
    } catch (err) { status.textContent = err.message; }
  }



  if (!c.panels.length) {
    wrap.append(el("div", { class: "h3s-cnote" },
      "No panels yet \u2014 drag on the page above to outline the first one."));
    return wrap;
  }

  const chunks = c.chunks || [];
  if (chunks.length > 1) {
    wrap.append(el("div", { class: "h3s-chunkbar" },
      el("span", {}, "\u{1F3AC} generating"),
      el("select", { onchange: (e) => {
        c.chunk_index = parseInt(e.target.value, 10) || 0; commit(); rerender();
      } }, chunks.map((ch, i) => el("option", {
        value: String(i), selected: i === (c.chunk_index || 0) },
        `Chunk ${i + 1} of ${chunks.length} \u2014 panels ${ch.join(", ")} \u2014 ${
          (c.chunkSeconds || [])[i] ?? "?"}s \u2014 ${
          (c.chunkAspects || [])[i] ?? "?"}`))),
      el("span", { class: "sp" }),
      el("span", { style: { fontSize: "10px", color: "#8890a0" } },
        (new Set(c.chunkAspects || [])).size > 1
          ? "Split partly by orientation: one generation is one frame size, so "
            + "landscape and portrait panels can't share a clip."
          : "Too long for one clip \u2014 run each chunk separately."),
      el("label", { style: { fontSize: "10px", color: "#8890a0", display: "flex",
                             alignItems: "center", gap: "3px", marginLeft: "8px" },
          title: "Keep panels together even when their orientation changes. They'll "
               + "share one frame size, so some will be letterboxed or cropped." },
        el("input", { type: "checkbox", checked: !!c.merge_aspect,
          onchange: (e) => { c.merge_aspect = e.target.checked; commit(); retime(); } }),
        "merge orientations")));
  }

  const active = new Set((chunks[c.chunk_index || 0] || []).map(Number));
  const move = (arr, i, d) => {
    const j = i + d;
    if (j < 0 || j >= arr.length) return;
    [arr[i], arr[j]] = [arr[j], arr[i]];
    commit(); rerender();
  };

  wrap.append(comicPanelList(state, commit, rerender, status));

  return wrap;
}

function buildPanel(node) {
  injectCSS();
  const w = node.widgets?.find((ww) => ww.name === "loader_state");
  let state = {};
  state.subjects ||= []; state.shots ||= [];
  state.audio ||= { dialogue: [], music: [] };

  let tab = "subjects";
  const counts = el("div", { class: "h3s-counts" });
  const tabs = el("div", { class: "h3s-tabs" });
  const body = el("div", { class: "h3s-body" });
  // DEFAULT for files that don't have their own setting. Each loaded file has
  // its own resize dropdown, and that always wins -- so a character sheet left on
  // "original" stays full-resolution no matter what the default here is.
  const sizeSel = el("select", {
    title: "Longest side of images/video frames passed to the H3 model. "
         + "Original keeps them untouched.",
    style: { fontSize: "10px", background: "#12151b", color: "#dde2ea",
             border: "1px solid #2e3440", borderRadius: "4px", padding: "1px 4px" },
    onchange: (e) => {
      state.max_media_side = parseInt(e.target.value, 10) || 0;
      commit();
    } },
    [[0, "original"], [2048, "2048 px"], [1536, "1536 px"], [1280, "1280 px"],
     [1024, "1024 px"], [768, "768 px"]].map(([v, lb]) =>
      el("option", { value: String(v),
        selected: String(v) === String(state.max_media_side || 0) }, lb)));

  const durationInput = el("input", { type: "number", min: 5, max: 15, step: 0.5,
    style: { width: "54px", background: "#12151b", color: "#dde2ea",
             border: "1px solid #2e3440", borderRadius: "4px", padding: "1px 4px",
             fontSize: "10px" },
    value: state.duration_seconds ?? 10,
    onchange: (e) => {
      const v = parseFloat(e.target.value);
      state.duration_seconds = Number.isNaN(v) ? 10 : Math.max(5, Math.min(15, v));
      e.target.value = state.duration_seconds;
      commit();
    } });
  const mpInput = el("input", { type: "number", min: 0.1, max: 8, step: 0.1,
    style: { width: "52px", background: "#12151b", color: "#dde2ea",
             border: "1px solid #2e3440", borderRadius: "4px", padding: "1px 4px",
             fontSize: "10px" },
    value: state.megapixels ?? 2.0,
    onchange: (e) => {
      const v = parseFloat(e.target.value);
      state.megapixels = Number.isNaN(v) ? 2.0 : Math.max(0.1, Math.min(8, v));
      e.target.value = state.megapixels; commit();
    } });
  const aspectSel = el("select", { style: { fontSize: "10px", background: "#12151b", color: "#dde2ea",
                           border: "1px solid #2e3440", borderRadius: "4px" },
    onchange: (e) => { state.aspect_ratio = e.target.value; commit(); } },
    ["16:9","9:16","1:1","4:3","3:4","21:9","4:5","5:4","3:2","2:3"].map((r) =>
      el("option", { value: r, selected: r === (state.aspect_ratio || "16:9") }, r)));

  // sizeSel/durationInput/mpInput/aspectSel are built above using whatever
  // `state` holds *at panel-construction time* -- which is the empty default,
  // since readSavedState() hasn't run yet. Every other part of the panel
  // (counts/tabs/body) re-renders from `state` after the real saved state
  // loads, so it looks right; these four never got that re-render, so they
  // kept showing their built-in defaults even though the real value was
  // sitting correctly in the widget/properties the whole time. This syncs
  // their displayed value to whatever `state` currently is; call it any time
  // `state` is replaced wholesale (initial load, and on every DOM-rebuild
  // reload) rather than mutated in place by the controls' own onchange.
  function syncTopBar() {
    sizeSel.value = String(state.max_media_side || 0);
    durationInput.value = state.duration_seconds ?? 10;
    mpInput.value = state.megapixels ?? 2.0;
    aspectSel.value = state.aspect_ratio || "16:9";
  }

  const root = el("div", { class: "h3s-panel", style: { height: `${PANEL_H}px` } },
    el("div", { class: "h3s-headrow" }, counts,
      el("span", { style: { fontSize: "10px", color: "#8890a0" },
        title: "Default for newly added files. Each file has its own resize "
             + "dropdown that overrides this." }, "default resize"), sizeSel,
      el("span", { style: { fontSize: "10px", color: "#8890a0", marginLeft: "8px" },
        title: "Seconds, sent out of this node's duration output. H3 accepts 5-15s. "
             + "Comic mode computes this from the estimated speech and overrides it." },
        "duration"),
      durationInput,
      el("span", { style: { fontSize: "10px", color: "#8890a0" } }, "s"),
      el("span", { style: { fontSize: "10px", color: "#8890a0", marginLeft: "8px" },
        title: "Total output pixels. Width and height are derived from this and "
             + "the aspect ratio, snapped to multiples of 32." }, "MP"),
      mpInput,
      el("span", { style: { fontSize: "10px", color: "#8890a0", marginLeft: "6px" },
        title: "Default shape. Replace mode, comic mode and first/last-frame "
             + "anchors override this with the aspect of their own input." },
        "aspect"),
      aspectSel),
    tabs, body);

  function commit(resizeMaybe) {
    _loaderDefaultSide = state.max_media_side || 0;
    const json = JSON.stringify(state);
    if (w) { w.value = json; w.callback?.(json); }
    // Mirror into node.properties, which is always serialized with the workflow.
    // A widget flagged type="hidden" is not guaranteed to be, which is how the
    // whole loader state could vanish on a workflow tab switch.
    try { node.properties = node.properties || {}; node.properties.h3sLoaderState = json; }
    catch (e) { /* properties unavailable */ }
    node.setDirtyCanvas?.(true, true);
    renderCounts();
  }

  function readSavedState() {
    const fromWidget = node.widgets?.find((ww) => ww.name === "loader_state")?.value;
    const fromProps = node.properties?.h3sLoaderState;
    for (const raw of [fromWidget, fromProps]) {
      if (!raw || raw === "{}") continue;
      try {
        const parsed = JSON.parse(raw);
        if (parsed && typeof parsed === "object" &&
            ((parsed.subjects || []).length || (parsed.shots || []).length
             || (parsed.audio?.dialogue || []).length || (parsed.audio?.music || []).length)) {
          return parsed;
        }
      } catch (e) { /* try the next source */ }
    }
    return null;
  }

  function renderCounts() {
    const { p, v, a } = countMedia(state);
    const span = (n, max, label) => el("span", { class: "h3s-count" + (n > max ? " over" : "") },
      `${label} ${n}/${max}`);
    counts.replaceChildren(span(p, MAX_PICTURES, "images"), span(v, MAX_VIDEOS, "videos"), span(a, MAX_AUDIO, "audio"));
  }

  function renderTabs() {
    tabs.replaceChildren(
      ...["subjects", "shots", "audio", "comic", "replace"].map((t) => {
        const on = state.comic?.enabled || state.replace?.enabled;
        const label = t === "subjects" ? (state.comic?.enabled ? "Subject (off)" : "Subject")
          : t === "shots" ? (on ? "Shot (off)" : "Shot")
          : t === "audio" ? "Audio"
          : t === "comic" ? (state.comic?.enabled ? "Comic \u25cf" : "Comic")
          : (state.replace?.enabled ? "Replace \u25cf" : "Replace");
        return el("div", { class: "h3s-tab" + (t === tab ? " active" : ""),
          style: (t === "shots" && on) ? { opacity: ".45" } : {},
          title: (t === "shots" && on)
            ? "Comic or replace mode builds the shots instead; this tab is inactive."
            : "",
          onclick: () => { tab = t; renderTabs(); renderBody(); } }, label);
      }));
  }

  function renderBody() {
    body.replaceChildren();
    if (tab === "subjects") {
      if (state.comic?.enabled) {
        body.append(el("div", { class: "h3s-empty" },
          "Comic mode is on \u2014 its cast is defined in the Comic tab, where each "
          + "character also gets the panels they appear in. Turn comic mode off to "
          + "use this tab."));
        return;
      }
      if (!state.subjects.length) body.append(el("div", { class: "h3s-empty" }, "No subjects yet."));
      state.subjects.forEach((s, i) => body.append(subjectRow(s, i, commit,
        () => { state.subjects.splice(i, 1); commit(); renderBody(); }, renderBody,
        () => (state.shots || []).length)));
      body.append(el("button", { class: "h3s-btn primary", onclick: () => {
        state.subjects.push({ visual_refs: [], outfit_refs: [] }); commit(); renderBody();
      } }, "+ Add Subject"));
    } else if (tab === "comic") {
      body.append(comicSection(state, commit, () => { renderTabs(); renderBody(); }));
    } else if (tab === "replace") {
      body.append(replaceSection(state, commit, () => { renderTabs(); renderBody(); }));
    } else if (tab === "shots") {
      if (state.comic?.enabled || state.replace?.enabled) {
        const which = state.comic?.enabled ? "Comic" : "Replace";
        body.append(el("div", { class: "h3s-empty" },
          `${which} mode is on \u2014 the shots are built there instead. Turn it off `
          + `in the ${which} tab to use this tab.`));
        return;
      }
      if (!state.shots.length) body.append(el("div", { class: "h3s-empty" }, "No shots yet."));
      // Storyboard spans shots, so it sits above them rather than inside one.
      body.append(el("div", { class: "h3s-row" },
        el("div", { class: "h3s-rowtitle" }, "Storyboard (plans all shots)"),
        el("div", { style: { fontSize: "9px", color: "#8890a0", margin: "4px 0 6px" } },
          "Optional shot-planning image: viewpoint, subject placement and shot order. "
          + "Cited as a plan, not as a frame that appears in the video."),
        el("div", { class: "h3s-outfitlist" },
          dropZone({ label: "Storyboard", accept: "picture", value: state.storyboard,
            onSet: (v) => { state.storyboard = v; commit(); },
            onRemove: () => { state.storyboard = null; commit(); },
            onEdit: commit }))));
      state.shots.forEach((s, i) => body.append(shotRow(s, i, commit,
        () => { state.shots.splice(i, 1); commit(); renderBody(); }, renderBody)));
      body.append(el("button", { class: "h3s-btn primary", onclick: () => {
        state.shots.push({ reference_images: [] }); commit(); renderBody();
      } }, "+ Add Shot"));
    } else {
      body.append(audioSection(state, commit));
    }
  }

  _loaderDefaultSide = state.max_media_side || 0;
  const _initial = readSavedState();
  if (_initial) {
    state = _initial;
    state.subjects ||= []; state.shots ||= [];
    state.audio ||= { dialogue: [], music: [] };
    commit();
  }

  syncTopBar();
  renderCounts(); renderTabs(); renderBody();

  // Re-read the widget's persisted value and redraw. Needed because ComfyUI
  // tears down and rebuilds this node's DOM when you switch workflow tabs (and
  // when a saved workflow loads): the panel's in-memory `state` object built at
  // creation time is gone by then, so without this the panel comes back empty
  // even though the widget itself still holds the real value.
  node._h3sReload = () => {
    const fresh = readSavedState();
    // An empty read must not wipe state that's still in memory: a DOM rebuild can
    // fire before the widget value has been restored.
    if (fresh) {
      state = fresh;
      state.subjects ||= []; state.shots ||= [];
      state.audio ||= { dialogue: [], music: [] };
      // Push back into the widget -- that's what the backend actually receives, so
      // restoring only the in-memory copy would still execute with an empty loader.
      commit();
    }
    _loaderDefaultSide = state.max_media_side || 0;
    syncTopBar();
    renderCounts(); renderTabs(); renderBody();
  };

  return root;
}

app.registerExtension({
  name: "H3AutoPrompt.StructuredLoader",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== LOADER_NAME) return;
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      const w = this.widgets?.find((ww) => ww.name === "loader_state");
      if (w) { w.hidden = true; w.type = "hidden"; w.computeSize = () => [0, -4]; }

      const panelRoot = buildPanel(this);
      this._h3sRoot = panelRoot;
      const widget = this.addDOMWidget("h3s_panel", "div", panelRoot, { serialize: false });
      // Only set an initial size on a freshly placed node (size[1] still at its
      // default). After that the node is whatever size it's dragged to -- the
      // panel body scrolls rather than the node being forced back open.
      // Fixed height, NOT derived from this.size. Deriving it from the node and
      // then feeding it back through widget.computeSize -- which is what decides
      // the node's height -- is a loop: any chrome the estimate misses gets added
      // again on every layout pass. The reference loader sizes once and leaves it.
      this._h3sFit = () => {
        panelRoot.style.height = `${PANEL_H}px`;
        widget.computedHeight = PANEL_H;
        widget.computeSize = () => [NODE_W, PANEL_H];
        try {
          const min = this.computeSize?.();
          this.size[0] = Math.max(NODE_W, this.size[0] || 0);
          this.size[1] = Math.max(min?.[1] || 0, PANEL_H);
        } catch (e) { /* Vue may own layout */ }
      };
      this._h3sFit();
      return r;
    };

    // No minimum enforced: the node can be dragged to any size and the panel
    // follows it, scrolling internally when the content doesn't fit.
    const onResize = nodeType.prototype.onResize;
    nodeType.prototype.onResize = function (size) {
      const r = onResize?.apply(this, arguments);
      requestAnimationFrame(() => this._h3sFit?.());
      return r;
    };

    // Fires when a saved workflow loads and when ComfyUI rebuilds this node's
    // DOM (notably on a workflow tab switch) -- re-read the persisted state so
    // the panel doesn't come back blank.
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      setTimeout(() => this._h3sReload?.(), 0);
      return r;
    };
  },
});

/* Self-heal after a DOM rebuild. onConfigure reliably fires on workflow load,
 * but switching workflow tabs can tear down and recreate a node's DOM without
 * it -- leaving the panel detached from the document and its in-memory state
 * gone, which looked like "the loader forgot everything". Detecting the
 * detached element and re-reading the persisted widget value restores it. */
setInterval(() => {
  try {
    for (const node of app.graph?._nodes || []) {
      if (node.type !== LOADER_NAME) continue;
      const panel = node._h3sRoot
        || node.widgets?.find((w) => w.name === "h3s_panel")?.element;
      if (panel && !panel.isConnected) node._h3sReload?.();
    }
  } catch (e) { /* graph not ready */ }
}, 1000);
