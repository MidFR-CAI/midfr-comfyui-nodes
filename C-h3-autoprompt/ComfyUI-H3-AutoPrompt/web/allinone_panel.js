import { app } from "../../scripts/app.js";

/* H3 AutoPrompt: All-in-One -- node-body panel.
 *
 * Two jobs:
 *  1. Collapse every settings widget (model paths, whisper, device, budgets,
 *     mode/duration) behind a "Settings..." button that opens a modal, so the
 *     node itself stays compact instead of showing a tall stack of fields.
 *  2. Render per-shot prompt boxes inline, one per shot defined in a connected
 *     Structured Media Loader (falling back to a single shot when nothing is
 *     connected, so there's always somewhere to write).
 *
 * The boxes write to this node's existing `shot_briefs` widget -- the same JSON
 * shape the structured composer reads -- so no new backend plumbing is needed.
 */

const ALLINONE_NAME = "H3AutoPromptAllInOne";
const STRUCT_LOADER = "H3StructuredMediaLoader";
const NODE_W = 560, HEAD_H = 34, MIN_H = 240;
/* Hard ceiling on the panel. Without it the panel tracked the node's own height,
 * and the node was being forced very tall by its output slots -- which produced a
 * shot box so long its bottom ran off the canvas. Roughly matches the media
 * loader's own height so the two nodes sit side by side sensibly. */
/* Space the node's own chrome takes above/below the panel: title bar, the two
 * output slots and padding. Subtracted from the node height to get the panel
 * height, so the panel ends exactly at the node's bottom edge instead of being
 * clipped by it. */
const CHROME = 74;
/* Height of one shot block (label + text box + its dialogue rows) and the hard
 * ceiling past which the body scrolls instead of the panel growing. */
const SHOT_BLOCK = 190;
const MAX_H = 620;
/* Shot text boxes are a fixed height (set in CSS) and do NOT track the node --
 * only the scrolling window around them does, like the media loader's grid. */

/* Everything that should live behind the Settings button rather than on the
 * node face. Anything not listed (and not one of the hidden JSON state
 * widgets) is left alone. */
/* Offered in the dropdown; the custom box accepts anything else. Mirrors the
 * guide's style vocabulary plus the variants that keep coming up. */
const STYLE_PRESETS = [
  "auto (detect from references)",
  "Live-action, cinematic",
  "2D-animated, anime",
  "2D-animated",
  "3D CG",
  "claymation",
  "watercolor",
  "vintage film",
];

const SETTINGS_WIDGETS = [
  "vlm_backend", "vlm_model_path_or_id", "vlm_mmproj_path",
  "whisper_model_size", "device", "use_diarization",
  "max_vlm_images", "vlm_n_ctx", "vlm_detail", "use_vlm", "debug",
];
/* Model paths get a Browse button instead of paste-only text, using ComfyUI's
 * own /internal/files listing where available and falling back to a local file
 * picker (which yields a real absolute path via webkitRelativePath/name). */
/* Only used when models/gguf is empty, in which case the Python side falls back
 * to plain STRING fields and there's no dropdown to pick from. */
const FILE_WIDGETS = { vlm_model_path_or_id: ".gguf", vlm_mmproj_path: ".gguf" };

const CSS = `
.h3a-panel{width:100%;background:#20242c;color:#d7dbe2;font-family:system-ui,sans-serif;
  display:flex;flex-direction:column;box-sizing:border-box;overflow:hidden;}
.h3a-head{display:flex;align-items:center;gap:8px;padding:6px 10px;background:#1b1f27;
  border-bottom:1px solid #2a2f3a;font-size:10px;color:#9aa3b2;}
.h3a-head .sp{flex:1;}
.h3a-body{flex:1;overflow:auto;padding:8px;min-height:0;}
.h3a-shot{margin-bottom:8px;}
.h3a-label{font-size:9px;font-weight:700;color:#e0a94c;margin-bottom:3px;}
.h3a-ta{width:100%;height:130px;min-height:60px;background:#12151b;color:#dde2ea;
  border:1px solid #2e3440;border-radius:6px;padding:6px 8px;font-size:11px;
  font-family:inherit;resize:vertical;box-sizing:border-box;}
.h3a-ta:focus{outline:none;border-color:#4a5568;}
.h3a-btn{background:#232833;color:#c9cfda;border:1px solid #333a45;border-radius:6px;
  padding:4px 10px;font-size:10px;cursor:pointer;}
.h3a-btn:hover{background:#2c3340;}
.h3a-btn.primary{background:#1f4f7d;border-color:#3d7fbf;color:#dbeafe;}
.h3a-over{position:fixed;inset:0;background:rgba(8,10,14,.72);z-index:10090;
  display:flex;align-items:center;justify-content:center;}
.h3a-modal{width:min(520px,94vw);max-height:86vh;background:#191c22;
  border:1px solid #303642;border-radius:10px;display:flex;flex-direction:column;
  overflow:hidden;font-family:system-ui,sans-serif;color:#d7dbe2;
  box-shadow:0 24px 64px rgba(0,0,0,.55);}
.h3a-mhead{display:flex;align-items:center;padding:10px 14px;background:#1b1f27;
  border-bottom:1px solid #2a2f3a;font-size:13px;font-weight:600;}
.h3a-mbody{flex:1;overflow:auto;padding:12px 14px;}
.h3a-field{display:flex;align-items:center;gap:10px;margin-bottom:9px;}
.h3a-fname{flex:0 0 155px;font-size:11px;color:#9aa3b2;}
.h3a-field input[type=text],.h3a-field input[type=number],.h3a-field select{
  flex:1;background:#12151b;color:#dde2ea;border:1px solid #2e3440;border-radius:6px;
  padding:4px 7px;font-size:11px;min-width:0;}
.h3a-field input:focus,.h3a-field select:focus{outline:none;border-color:#4a5568;}
.h3a-mfoot{display:flex;padding:10px 14px;background:#1b1f27;
  border-top:1px solid #2a2f3a;}
.h3a-mfoot .sp{flex:1;}
.h3a-empty{padding:14px;text-align:center;color:#6b7484;font-size:11px;}
.h3a-dur{display:flex;align-items:center;gap:5px;font-size:10px;color:#9aa3b2;}
.h3a-dur input{width:56px;background:#12151b;color:#dde2ea;border:1px solid #2e3440;
  border-radius:5px;padding:2px 5px;font-size:10px;}
.h3a-shot{display:flex;flex-direction:column;margin-bottom:8px;min-height:0;}
.h3a-dlg{margin-top:5px;display:flex;flex-direction:column;gap:4px;}
.h3a-dlgrow{background:#181c24;border:1px solid #262b34;border-radius:6px;padding:5px;
  display:flex;flex-direction:column;gap:4px;}
.h3a-dlgctl{display:flex;gap:5px;align-items:center;flex-wrap:wrap;}
.h3a-dlgctl input[type=number]{width:44px;}
.h3a-dlgctl input[type=text]{flex:1;min-width:80px;}
.h3a-dlgctl input,.h3a-dlgctl select{background:#12151b;color:#dde2ea;
  border:1px solid #2e3440;border-radius:5px;padding:2px 5px;font-size:10px;}
.h3a-dlgctl label{font-size:9px;color:#8890a0;display:flex;align-items:center;gap:3px;}
.h3a-dlgtext{width:100%;background:#12151b;color:#dde2ea;border:1px solid #2e3440;
  border-radius:5px;padding:3px 6px;font-size:10px;box-sizing:border-box;}
.h3a-browse{background:#232833;color:#c9cfda;border:1px solid #333a45;border-radius:5px;
  padding:3px 8px;font-size:10px;cursor:pointer;white-space:nowrap;}
.h3a-browse:hover{background:#2c3340;}
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

function originNode(node, slotIndex) {
  let n = node.getInputNode?.(slotIndex);
  let guard = 0;
  while (n && /reroute/i.test(n.type || "") && guard++ < 16) {
    const nn = n.getInputNode?.(0);
    if (!nn) break;
    n = nn;
  }
  return n || null;
}

/** Which special mode the connected loader is in, if any. Comic and Replace
 * both build the shots themselves, so neither the shot boxes nor the
 * whole-video brief apply -- the brief was still appearing in Replace mode
 * because "no shots" was being read as "needs a brief". */
function connectedMode(node) {
  const idx = (node.inputs || []).findIndex((i) => i.name === "references");
  if (idx < 0 || node.inputs[idx].link == null) return null;
  const loader = originNode(node, idx);
  if (!loader || loader.type !== STRUCT_LOADER) return null;
  try {
    const st = JSON.parse(
      loader.widgets?.find((w) => w.name === "loader_state")?.value || "{}");
    if (st.replace?.enabled) return "replace";
    if (st.comic?.enabled) return "comic";
  } catch (e) { /* fall through */ }
  return null;
}

/** Shot count from a connected Structured Media Loader, read straight off its
 * widget (no execution needed). Returns 0 when none is connected. */
function connectedShotCount(node) {
  // Input is named "references" on this node (type H3_STRUCTURED_REFS). Looking up
  // the wrong name here silently pinned the shot count at 0 -> always one box.
  const idx = (node.inputs || []).findIndex((i) => i.name === "references");
  if (idx < 0 || node.inputs[idx].link == null) return 0;
  const loader = originNode(node, idx);
  if (!loader || loader.type !== STRUCT_LOADER) return 0;
  try {
    const state = JSON.parse(
      loader.widgets?.find((w) => w.name === "loader_state")?.value || "{}");
    return Array.isArray(state.shots) ? state.shots.length : 0;
  } catch (e) { return 0; }
}

/** Renders one editable control per settings widget, bound directly to the
 * widget's own value so the modal is just a different surface for the same
 * data -- no separate copy to keep in sync. */
/* Widget values are restored from a saved workflow BY INDEX, not by name. Any
 * change to a node's widget list (duration_seconds becoming a link-only socket,
 * for instance) shifts every later value by one, so an old workflow can load a
 * boolean into a text field and a number into a checkbox. Re-typing each value
 * against its widget's declared type puts it back to something sane instead of
 * silently running with garbage settings. */
/* ComfyUI restores widgets_values POSITIONALLY. duration_seconds is a
 * forceInput socket, and whether such a widget occupies a slot in that array
 * differs between frontend versions -- so every value after it can land one
 * widget off, which is what made the model dropdowns and vlm_n_ctx swap places
 * on each restart. node.properties survives verbatim and is keyed by NAME, so
 * mirroring the settings there and restoring from it is immune to any shift. */
function saveSettings(node) {
  try {
    const map = {};
    for (const name of SETTINGS_WIDGETS) {
      const w = node.widgets?.find((x) => x.name === name);
      if (w && w.value !== undefined) map[name] = w.value;
    }
    node.properties = node.properties || {};
    node.properties.h3aSettings = map;
  } catch (e) { /* properties unavailable */ }
}

function restoreSettings(node) {
  const map = node.properties?.h3aSettings;
  if (!map || typeof map !== "object") return;
  for (const name of SETTINGS_WIDGETS) {
    if (!(name in map)) continue;
    const w = node.widgets?.find((x) => x.name === name);
    if (!w) continue;
    const val = map[name];
    // A stale saved model filename that no longer exists in models/gguf would
    // fail the combo check in repairWidgetValue below and reset cleanly.
    w.value = val;
    w.callback?.(val);
  }
}

function repairWidgetValue(w) {
  const t = w._h3aType, vals = w._h3aValues;
  const def = w.options?.default;
  if (Array.isArray(vals) && vals.length) {
    // A saved value that isn't in the CURRENT combo list (model file moved,
    // folder rescanned in a different order, etc.) is not necessarily wrong --
    // it just isn't visible to the dropdown *right now*. Resetting it to
    // vals[0] here and then letting saveSettings() persist that reset is what
    // silently ate real settings across a restart: the moment the list looked
    // stale, the repair overwrote the by-name mirror with the wrong value
    // forever. So: only fall back when there's truly nothing usable (empty/
    // null/undefined). Otherwise leave the value alone -- ComfyUI still sends
    // it to Python as-is, and it reappears in the dropdown the moment the
    // file is back or the list is rescanned.
    if (w.value === undefined || w.value === null || w.value === "") {
      w.value = vals.includes(def) ? def : vals[0];
    }
  } else if (t === "toggle") {
    if (typeof w.value !== "boolean") w.value = typeof def === "boolean" ? def : true;
  } else if (t === "number") {
    if (typeof w.value !== "number" || Number.isNaN(w.value)) {
      w.value = typeof def === "number" ? def : 0;
    }
  } else if (t === "text" || t === "string" || t === "customtext") {
    if (typeof w.value !== "string") w.value = typeof def === "string" ? def : "";
  }
}

function openSettings(node) {
  injectCSS();
  const rows = [];
  for (const name of SETTINGS_WIDGETS) {
    const w = node.widgets?.find((ww) => ww.name === name);
    if (!w) continue;
    repairWidgetValue(w);
    let control;
    // Declared type/options, captured before the widget was hidden -- guessing from
    // typeof w.value was what put a checkbox next to "whisper model size".
    const values = w._h3aValues ?? w.options?.values;
    const declared = w._h3aType ?? w.type;
    if (Array.isArray(values) && values.length) {
      control = el("select", { onchange: (e) => { w.value = e.target.value; w.callback?.(w.value); } },
        values.map((v) => el("option", { value: v, selected: String(v) === String(w.value) }, String(v))));
    } else if (declared === "toggle" || typeof w.value === "boolean") {
      control = el("input", { type: "checkbox", checked: !!w.value,
        onchange: (e) => { w.value = e.target.checked; w.callback?.(w.value); } });
    } else if (declared === "number" || typeof w.value === "number") {
      control = el("input", { type: "number", value: w.value,
        step: Number.isInteger(w.value) ? 1 : 0.1,
        onchange: (e) => {
          const n = parseFloat(e.target.value);
          if (!Number.isNaN(n)) { w.value = n; w.callback?.(w.value); }
        } });
    } else if (FILE_WIDGETS[name]) {
      // Paste-free path entry. The browser can't hand back a real absolute path
      // from a file picker for security reasons, so this offers both: pick a file
      // to fill in its name, or type/paste a full path as before. Where ComfyUI
      // exposes its model folders we prefill from there.
      const text = el("input", { type: "text", value: w.value ?? "",
        onchange: (e) => { w.value = e.target.value; w.callback?.(w.value); } });
      const picker = el("input", { type: "file", accept: FILE_WIDGETS[name],
        style: { display: "none" },
        onchange: (e) => {
          const f = e.target.files?.[0];
          if (!f) return;
          // path is populated on Electron/desktop builds; elsewhere fall back to
          // the bare filename, which is enough when the file sits in models/vlm.
          const chosen = f.path || f.webkitRelativePath || f.name;
          text.value = chosen;
          w.value = chosen; w.callback?.(w.value);
        } });
      control = el("div", { style: { display: "flex", gap: "6px", flex: "1", minWidth: "0" } },
        text, picker,
        el("button", { class: "h3a-browse", onclick: () => picker.click() }, "Browse\u2026"));
    } else {
      control = el("input", { type: "text", value: w.value ?? "",
        onchange: (e) => { w.value = e.target.value; w.callback?.(w.value); } });
    }
    rows.push(el("div", { class: "h3a-field" },
      el("span", { class: "h3a-fname" }, name.replace(/_/g, " ")), control));
  }

  const close = () => {
    window.removeEventListener("keydown", onKey);
    overlay.remove();
    saveSettings(node);          // keep the by-name mirror current
    node.setDirtyCanvas?.(true, true);
  };
  const onKey = (e) => { if (e.key === "Escape") close(); };
  window.addEventListener("keydown", onKey);

  const overlay = el("div", { class: "h3a-over",
    onmousedown: (e) => { if (e.target === overlay) close(); } },
    el("div", { class: "h3a-modal" },
      el("div", { class: "h3a-mhead" }, "\u2699\uFE0F All-in-One Settings"),
      el("div", { class: "h3a-mbody" }, rows.length ? rows
        : el("div", { class: "h3a-empty" }, "No settings found on this node.")),
      el("div", { class: "h3a-mfoot" },
        el("span", { class: "sp" }),
        el("button", { class: "h3a-btn primary", onclick: close }, "Done"))));
  document.body.append(overlay);
}

function buildPanel(node) {
  injectCSS();
  const w = node.widgets?.find((ww) => ww.name === "shot_briefs");
  let prompts = [];

  const info = el("span", {});
  // duration_seconds is a forceInput socket now (link a FLOAT into it), so there
  // is deliberately no duration control here -- there's no widget to bind to.
  // Style lives on the node face, not buried in settings: auto-detection is only
  // reliable at the extremes, so this is something you'll want to correct often.
  const styleW = node.widgets?.find((ww) => ww.name === "style");
  let styleSel = null, styleCustom = null;
  if (styleW) {
    const setStyle = (v) => {
      styleW.value = v; styleW.callback?.(v);
      try { node.properties = node.properties || {}; node.properties.h3aStyle = v; }
      catch (e) { /* properties unavailable */ }
      node.setDirtyCanvas?.(true, true);
    };
    const isPreset = STYLE_PRESETS.includes(styleW.value);
    styleCustom = el("input", { type: "text", placeholder: "describe the style\u2026",
      value: isPreset ? "" : (styleW.value || ""),
      style: { display: isPreset ? "none" : "", width: "150px", fontSize: "10px",
               background: "#12151b", color: "#dde2ea", border: "1px solid #2e3440",
               borderRadius: "4px", padding: "1px 4px" },
      onchange: (e) => setStyle(e.target.value.trim()
                                 || "auto (detect from references)") });
    styleSel = el("select", {
      title: "Visual style of the target video. Auto reads it from the character "
           + "references and cross-checks against pixel statistics; pick a preset "
           + "or choose custom to type your own.",
      style: { fontSize: "10px", background: "#12151b", color: "#dde2ea",
               border: "1px solid #2e3440", borderRadius: "4px", maxWidth: "170px" },
      onchange: (e) => {
        if (e.target.value === "__custom__") {
          styleCustom.style.display = "";
          styleCustom.focus();
        } else {
          styleCustom.style.display = "none";
          setStyle(e.target.value);
        }
      } },
      [...STYLE_PRESETS.map((v) =>
          el("option", { value: v, selected: isPreset && v === styleW.value }, v)),
       el("option", { value: "__custom__", selected: !isPreset }, "custom\u2026")]);
  }
  const head = el("div", { class: "h3a-head" }, info,
    styleSel ? el("span", { style: { fontSize: "10px", color: "#8890a0" } }, "style") : null,
    styleSel, styleCustom, el("span", { class: "sp" }),
    el("button", { class: "h3a-btn", onclick: () => openSettings(node) }, "\u2699\uFE0F Settings\u2026"));
  const body = el("div", { class: "h3a-body" });
  const root = el("div", { class: "h3a-panel" }, head, body);
  let lastCount = -1;
  let lastMode = undefined;
  // Height is a pure function of the content, never of node.size. See the note
  // on syncHeight: deriving it from the node is a feedback loop.
  let panelHeight = MIN_H - CHROME;
  let updateTotal = () => {};

  function commit() {
    const json = JSON.stringify(prompts);
    if (w) { w.value = json; w.callback?.(json); }
    // Mirror into properties: node.properties is always serialized with the
    // workflow, whereas a widget flagged type="hidden" is not guaranteed to be.
    // Losing shot prompts on a tab switch traces back to exactly that gap.
    try { node.properties = node.properties || {}; node.properties.h3aShotBriefs = json; }
    catch (e) { /* properties unavailable */ }
    node.setDirtyCanvas?.(true, true);
  }

  /* Reads the saved prompts back, preferring whichever source actually has
   * content -- the widget if it survived, the properties mirror otherwise. */
  function readSaved() {
    const fromWidget = node.widgets?.find((ww) => ww.name === "shot_briefs")?.value;
    const fromProps = node.properties?.h3aShotBriefs;
    for (const raw of [fromWidget, fromProps]) {
      if (!raw || raw === "[]") continue;
      try {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed) && parsed.length) return parsed;
      } catch (e) { /* try the next source */ }
    }
    return [];
  }

  function render(force) {
    // Comic/Replace build their own shots and dialogue in the loader; this node
    // contributes nothing but settings, so show that rather than an input that
    // would be ignored.
    // Zero shots from the loader => whole-video brief instead of per-shot boxes.
    // The two are alternatives, not both: once shots exist they carry the text,
    // and a separate brief would just be a second place to say the same thing.
    // Declared before any use: these were below the mode guard, so the guard hit
    // the temporal dead zone and threw on node creation, which took the whole
    // extension down with it.
    const mode = connectedMode(node);
    const rawCount = connectedShotCount(node);
    const n = Math.max(1, rawCount);
    // Compare against the RAW count, not the clamped one. n is floored at 1, so
    // dropping from one shot to zero looked identical to "still one shot" and the
    // re-render was skipped -- which is why the brief box never came back until
    // the loader link was cycled.
    if (!force && rawCount === lastCount && mode === lastMode) return;
    lastCount = rawCount; lastMode = mode;

    if (mode) {
      info.textContent = mode === "replace" ? "replace mode" : "comic mode";
      body.replaceChildren(el("div", { class: "h3a-shot" },
        el("div", { class: "h3a-label" },
          mode === "replace" ? "Replace mode" : "Comic mode"),
        el("div", { style: { fontSize: "10px", color: "#8890a0", lineHeight: "1.5" } },
          mode === "replace"
            ? "The prompt is built entirely from the Replace tab \u2014 who replaces "
              + "whom, what carries over, and how the audio is handled. No brief or "
              + "shot text is used."
            : "Shots come from the comic page: one per panel, with the balloon text "
              + "as dialogue. No brief or shot text is used.")));
      measure(1);
      syncHeight();
      lastMode = mode;
      return;
    }

    if (prompts.length !== n) {
      prompts = Array.from({ length: n }, (_, i) => prompts[i] || { text: "", dialogue: [], duration: null });
      commit();
    }
    updateTotal = () => {
      const set = prompts.filter((p) => p && p.duration);
      const sum = set.reduce((a, p) => a + p.duration, 0);
      info.textContent = `${n} shot${n === 1 ? "" : "s"}`
        + (set.length ? `  ·  ${sum.toFixed(1)}s set` : "");
    };
    updateTotal();
    body.replaceChildren();

    // Comic/Replace build their own shots and dialogue in the loader; this node
    // contributes nothing but settings, so show that rather than an input that
    // would be ignored.
    if (mode) {
      info.textContent = mode === "replace" ? "replace mode" : "comic mode";
      body.replaceChildren(el("div", { class: "h3a-shot" },
        el("div", { class: "h3a-label" },
          mode === "replace" ? "Replace mode" : "Comic mode"),
        el("div", { style: { fontSize: "10px", color: "#8890a0", lineHeight: "1.5" } },
          mode === "replace"
            ? "The prompt is built entirely from the Replace tab \u2014 who replaces "
              + "whom, what carries over, and how the audio is handled. No brief or "
              + "shot text is used."
            : "Shots come from the comic page: one per panel, with the balloon text "
              + "as dialogue. No brief or shot text is used.")));
      measure(1);
      syncHeight();
      lastMode = mode;
      return;
    }

    // Zero shots from the loader => whole-video brief instead of per-shot boxes.
    // They're alternatives, not both: once shots exist they carry the text, and a
    // separate brief would just be a second place to say the same thing.
    if (rawCount === 0) {
      const bw = node.widgets?.find((ww) => ww.name === "brief");
      info.textContent = "no shots defined \u2014 whole-video brief";
      body.append(el("div", { class: "h3a-shot" },
        el("div", { class: "h3a-label" }, "Brief (whole video)"),
        el("div", { style: { fontSize: "9px", color: "#8890a0", marginBottom: "3px" } },
          "Describes the video as a whole. Pairs with a storyboard \u2014 the shots get "
          + "planned from this. Add shots in the loader to write them individually instead."),
        el("textarea", { class: "h3a-ta", value: bw?.value || "",
          placeholder: "What happens across the whole video...",
          onchange: (e) => {
            if (bw) { bw.value = e.target.value; bw.callback?.(bw.value); }
            try {
              node.properties = node.properties || {};
              node.properties.h3aBrief = e.target.value;
            } catch (err) { /* properties unavailable */ }
            node.setDirtyCanvas?.(true, true);
          } })));
      measure(1);
      syncHeight();
      return;
    }

    for (let i = 0; i < n; i++) {
      const shot = prompts[i];
      const dlgWrap = el("div", { class: "h3a-dlg" });

      function renderDialogue() {
        dlgWrap.replaceChildren();
        (shot.dialogue || []).forEach((d, di) => {
          const hasSubject = d.subject != null && d.subject !== "";
          dlgWrap.append(el("div", { class: "h3a-dlgrow" },
            el("div", { class: "h3a-dlgctl" },
              el("label", { title: "Which subject says this. Leave blank and use the "
                                 + "label field for anyone who isn't a tracked subject." }, "subj",
                el("input", { type: "number", min: 1, max: 12, value: d.subject || "",
                  onchange: (e) => { d.subject = e.target.value ? parseInt(e.target.value, 10) : null;
                    commit(); renderDialogue(); } })),
              el("input", { type: "text", placeholder: "or label (e.g. a passing stranger)",
                value: d.label || "", disabled: hasSubject,
                onchange: (e) => { d.label = e.target.value.trim() || null; commit(); } }),
              el("input", { type: "text", placeholder: "language", style: { flex: "0 0 74px" },
                value: d.language || "",
                onchange: (e) => { d.language = e.target.value.trim() || null; commit(); } }),
              el("label", {},
                el("input", { type: "checkbox", checked: !!d.voiceover,
                  onchange: (e) => { d.voiceover = e.target.checked; commit(); } }), "off-screen"),
              el("button", { class: "h3a-btn", onclick: () => {
                shot.dialogue.splice(di, 1); commit(); renderDialogue();
              } }, "\u2715")),
            el("input", { class: "h3a-dlgtext", type: "text", placeholder: "what they say (verbatim)",
              value: d.text || "",
              onchange: (e) => { d.text = e.target.value; commit(); } })));
        });
        dlgWrap.append(el("button", { class: "h3a-btn", onclick: () => {
          (shot.dialogue ||= []).push({ subject: null, label: null, text: "",
                                        language: null, voiceover: false });
          commit(); renderDialogue();
        } }, "+ Add dialogue line"));
      }
      renderDialogue();

      const durInput = el("input", { type: "number", min: 0.1, step: 0.5,
        value: shot.duration ?? "", placeholder: "auto",
        title: "Seconds for this shot. Leave blank to share whatever time the other "
             + "shots don't use. The total can't exceed the node's duration input -- "
             + "if it does, every shot is scaled down to fit.",
        style: { width: "58px", background: "#12151b", color: "#dde2ea",
                 border: "1px solid #2e3440", borderRadius: "5px",
                 padding: "1px 4px", fontSize: "9px" },
        onchange: (e) => {
          const v = parseFloat(e.target.value);
          shot.duration = (!Number.isNaN(v) && v > 0) ? v : null;
          commit(); updateTotal();
        } });

      body.append(el("div", { class: "h3a-shot" },
        el("div", { class: "h3a-label",
                    style: { display: "flex", alignItems: "center", gap: "6px" } },
          `Shot ${i + 1}`,
          el("span", { style: { flex: "1" } }),
          el("span", { style: { fontWeight: "400", color: "#8890a0" } }, "dur"),
          durInput),
        el("textarea", { class: "h3a-ta", value: shot.text || "",
          placeholder: `What happens in shot ${i + 1}...`,
          onchange: (e) => { shot.text = e.target.value; commit(); } }),
        dlgWrap));
    }
    // Height comes from the shot count, capped; past the cap the body scrolls.
    measure(n);
    syncHeight();
  }

  /* Fills the panel to the node's current height and divides the leftover space
   * among the shot text boxes, so the content ends exactly at the node's bottom
   * edge rather than overflowing it. */
  function syncHeight() {
    const panelW = node.widgets?.find((ww) => ww.name === "h3a_panel");
    // computeSize must NOT derive from node.size. ComfyUI computes the node's
    // height FROM the widgets' computeSize, so returning "node height minus
    // chrome" is a feedback loop: any rounding drift compounds every frame and
    // the node grows without bound. panelHeight is a value we own, changed only
    // when the person actually resizes the node.
    const h = Math.max(40, panelHeight);   // set by render() from the content
    root.style.height = `${h}px`;
    if (panelW) { panelW.computedHeight = h; panelW.computeSize = () => [NODE_W, h]; }
    try {
      // LiteGraph's own computeSize already accounts for title, slots and widget
      // chrome -- ask it rather than guessing a constant, and never override it.
      const min = node.computeSize?.();
      node.size[0] = Math.max(NODE_W, node.size[0] || 0);
      node.size[1] = Math.max(min?.[1] || 0, h);
    } catch (e) { /* Vue may own layout; the CSS height keeps the panel intact */ }

    // Text boxes keep their fixed CSS height (plus whatever the resize grip is
    // dragged to); the body scrolls when the content exceeds the cap.
  }

  /* Deterministic: derived from how many shot blocks exist, capped. Never reads
   * node.size, so it cannot feed back into ComfyUI's layout. */
  function measure(nBlocks) {
    panelHeight = Math.min(MAX_H, Math.max(MIN_H - CHROME,
                                            HEAD_H + nBlocks * SHOT_BLOCK + 16));
    // computeSize lives on the node prototype, so it reads the height from here.
    node._h3aPanelHeight = panelHeight;
  }

  node._h3aSyncHeight = syncHeight;
  node._h3aRender = render;
  const normalize = (arr) => arr.map((p) => (typeof p === "string"
    ? { text: p, dialogue: [], duration: null }
    : { text: String(p?.text || ""), dialogue: p?.dialogue || [],
        duration: (p?.duration ?? null) }));

  node._h3aReload = () => {
    // Same widget-vs-properties recovery as the shot prompts: a hidden widget's
    // value doesn't reliably survive a workflow tab switch.
    try {
      const bw = node.widgets?.find((ww) => ww.name === "brief");
      const savedBrief = node.properties?.h3aBrief;
      if (bw && savedBrief && !bw.value) { bw.value = savedBrief; bw.callback?.(bw.value); }
      const sw2 = node.widgets?.find((ww) => ww.name === "style");
      const savedStyle = node.properties?.h3aStyle;
      if (sw2 && savedStyle) { sw2.value = savedStyle; sw2.callback?.(sw2.value); }
    } catch (e) { /* ignore */ }
    const saved = normalize(readSaved());
    // Don't let an empty read wipe prompts that are still in memory -- a rebuild
    // can fire before the widget value has been restored.
    if (saved.length || !prompts.length) prompts = saved;
    // Write straight back to the widget: if the value came from the properties
    // mirror because the widget lost it, the widget is what actually gets sent to
    // the backend at execution time, so it has to be repopulated or the node would
    // run with no shot prompts at all.
    commit();
    render(true);
  };
  prompts = normalize(readSaved());
  render(true);
  return root;
}

app.registerExtension({
  name: "H3AutoPrompt.AllInOnePanel",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== ALLINONE_NAME) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      // Collapse the settings stack off the node face -- they stay real widgets
      // (so they still serialize and execute normally), just not drawn here.
      for (const name of [...SETTINGS_WIDGETS, "shot_briefs", "brief", "style"]) {
        const ww = this.widgets?.find((x) => x.name === name);
        if (!ww) continue;
        // Stash the declared type/options first: setting type="hidden" below wipes
        // them, and without them the settings modal has to guess a control from the
        // current value -- which renders the wrong control entirely whenever a value
        // is stale (e.g. a checkbox for whisper_model_size).
        if (ww._h3aType === undefined) {
          ww._h3aType = ww.type;
          ww._h3aValues = ww.options?.values;
        }
        repairWidgetValue(ww);
        ww.hidden = true; ww.type = "hidden"; ww.computeSize = () => [0, -4];
      }
      // Deliberately NOT saving here: widget values are still defaults at this
      // point, and writing them would clobber the by-name map restored from the
      // workflow. Saving happens after onConfigure, and on every settings edit.
      const panelRoot = buildPanel(this);
      this._h3aRoot = panelRoot;
      this.addDOMWidget("h3a_panel", "div", panelRoot, { serialize: false });
      return r;
    };


    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function () {
      const r = onConnectionsChange?.apply(this, arguments);
      setTimeout(() => this._h3aRender?.(true), 0);
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      setTimeout(() => {
        // By-name restore first -- it overwrites anything that positional
        // restore put in the wrong widget -- then type-sanitise what's left.
        restoreSettings(this);
        for (const name of SETTINGS_WIDGETS) {
          const ww = this.widgets?.find((x) => x.name === name);
          if (ww) repairWidgetValue(ww);
        }
        // Persist the post-restore, post-repair state so the by-name map exists
        // even for someone who never opens the settings modal -- without it there
        // would be nothing to restore from after a positional shift.
        saveSettings(this);
        this._h3aReload?.();
      }, 0);
      return r;
    };
  },
});

/* Same self-heal + loader-sync watchdog used by the other panels: a workflow
 * tab switch can rebuild the node's DOM without firing onConfigure, which
 * would otherwise leave the panel detached and showing stale contents. */
setInterval(() => {
  try {
    for (const node of app.graph?._nodes || []) {
      if (node.type !== ALLINONE_NAME) continue;
      const panel = node._h3aRoot
        || node.widgets?.find((x) => x.name === "h3a_panel")?.element;
      if (panel && !panel.isConnected) { node._h3aReload?.(); continue; }
      node._h3aRender?.();
    }
  } catch (e) { /* graph not ready */ }
}, 1000);
