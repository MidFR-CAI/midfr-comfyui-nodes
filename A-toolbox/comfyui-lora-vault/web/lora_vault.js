// Lora Vault - in-node panel for "Lora Vault Loader".
// Shows rich Civitai metadata for every <lora:...> in the node, and lets you search all of
// Civitai (filtered by the base model you pick) and add a LoRA with one click. Nothing is
// downloaded until a workflow actually runs.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE = "LoraVaultLoader";
const TAG_RE = /<lora:([^<>]+)>/gi;
const SORTS = ["Most Downloaded", "Highest Rated", "Most Liked", "Newest", "Most Collected"];

// ------------------------------------------------------------------ helpers
async function call(path, method = "GET", body) {
  const init = body === undefined ? { method } : { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const r = await api.fetchApi(`/lora_vault/api${path}`, init);
  let j = {};
  try { j = await r.json(); } catch (_) { /* ignore */ }
  if (!r.ok || j.ok === false) throw new Error(j.error || `HTTP ${r.status}`);
  return j;
}

function h(tag, props = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v === false || v == null) continue;
    if (k === "class") e.className = v;
    else if (k === "style" && typeof v === "object") Object.assign(e.style, v);
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (k === "value") e.value = v;
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) {
    if (c == null || c === false) continue;
    e.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return e;
}

const fmtNum = (n) => (n == null ? "–" : n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : n >= 1e3 ? (n / 1e3).toFixed(1) + "k" : String(n));
const fmtSize = (kb) => (!kb ? "" : kb > 1024 * 1024 ? (kb / 1024 / 1024).toFixed(1) + " GB" : kb > 1024 ? Math.round(kb / 1024) + " MB" : Math.round(kb) + " KB");
const ls = {
  get(k, d) { try { const v = localStorage.getItem("lora_vault." + k); return v == null ? d : JSON.parse(v); } catch (_) { return d; } },
  set(k, v) { try { localStorage.setItem("lora_vault." + k, JSON.stringify(v)); } catch (_) { /* ignore */ } },
};
// 'Model Name - Version' (same rule as the server) so tags read like the files do
const nameRef = (model, version, id) => {
  const r = `${model || ""} - ${version || ""}`.replace(/[<>:|]+/g, " ").replace(/\s+/g, " ").replace(/^[\s-]+|[\s-]+$/g, "");
  return r && !/^(v|id:|civitai:|version:)?\d+$/i.test(r) ? r : "v" + id;
};
const layerTok = (a, b) => (a == null && b == null ? "" : `:layers=${a ?? 0}-${b ?? ""}`);
const fmtTag = (ref, s, c, a, b) => `<lora:${ref}:${+s.toFixed(3)}${c !== s ? ":" + +c.toFixed(3) : ""}${layerTok(a, b)}>`;
const STATUS = {
  ok: null,
  possibly_removed: ["⚠ Possibly removed", "warn", "Civitai returned 404 once. Will confirm on the next check."],
  removed: ["✖ Removed from Civitai", "bad", "Confirmed missing. Your saved metadata is kept; weights are kept only if you downloaded them."],
  archived: ["✖ Archived by creator", "bad", "Civitai no longer serves the files."],
  taken_down: ["✖ Taken down", "bad", "Removed by Civitai moderation."],
  unpublished: ["⚠ Unpublished", "warn", "This version is no longer public."],
};

const CSS = `
.lv{display:flex;flex-direction:column;gap:6px;font:12px/1.35 sans-serif;color:var(--fg-color,#ddd);height:100%;min-height:260px;box-sizing:border-box}
.lv *{box-sizing:border-box}
.lv-bar{display:flex;flex-wrap:wrap;gap:4px;align-items:center}
.lv input,.lv select,.lv button,.lv textarea{background:var(--comfy-input-bg,#222);color:var(--input-text,#ddd);border:1px solid var(--border-color,#444);border-radius:4px;padding:3px 6px;font:inherit}
.lv button{cursor:pointer}.lv button:hover{filter:brightness(1.25)}
.lv button.pri{background:#2d6cdf;border-color:#2d6cdf;color:#fff}
.lv button.mini{padding:1px 5px}
.lv-body{flex:1;overflow:auto;display:flex;flex-direction:column;gap:6px;min-height:120px;padding-right:2px}
.lv-card{display:flex;gap:8px;border:1px solid var(--border-color,#444);border-radius:6px;padding:6px;background:rgba(127,127,127,.08)}
.lv-card.bad{border-color:#c0392b}.lv-card.warn{border-color:#d8a31a}
.lv-thumb{width:64px;height:84px;flex:none;border-radius:4px;background:#111 center/cover;position:relative;overflow:hidden;cursor:pointer}
.lv-thumb.blur{filter:blur(10px)}
.lv-main{flex:1;min-width:0;display:flex;flex-direction:column;gap:3px}
.lv-title{font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.lv-sub{opacity:.7;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.lv-row{display:flex;flex-wrap:wrap;gap:4px;align-items:center}
.lv-chip{border-radius:10px;padding:0 7px;border:1px solid var(--border-color,#555);background:rgba(127,127,127,.15);cursor:default;white-space:nowrap}
.lv-chip.tw{cursor:pointer}.lv-chip.tw:hover{background:#2d6cdf55}
.lv-chip.ok{border-color:#2e9d57;color:#58d68d}.lv-chip.warn{border-color:#d8a31a;color:#f0c24b}.lv-chip.bad{border-color:#c0392b;color:#ff7a6b}
.lv-detail{margin-top:4px;border-top:1px dashed var(--border-color,#444);padding-top:5px;display:flex;flex-direction:column;gap:5px}
.lv-desc{white-space:pre-wrap;max-height:120px;overflow:auto;opacity:.85}
.lv-strip{display:flex;gap:4px;overflow-x:auto}
.lv-strip .lv-thumb{width:84px;height:110px}
.lv-meta{white-space:pre-wrap;max-height:90px;overflow:auto;background:rgba(0,0,0,.25);padding:4px;border-radius:4px}
.lv-bar-prog{height:4px;background:#333;border-radius:2px;overflow:hidden}.lv-bar-prog>div{height:100%;background:#2d6cdf}
.lv-empty{opacity:.6;text-align:center;padding:14px 6px}
.lv-err{color:#ff7a6b}
.lv input.num{width:56px}
.lv button.cog{padding:1px 6px}.lv button.cog.on{background:#2d6cdf;border-color:#2d6cdf;color:#fff}
.lv a{color:#6aa9ff}
`;

function injectCss() {
  if (document.getElementById("lora-vault-css")) return;
  document.head.append(h("style", { id: "lora-vault-css" }, CSS));
}

// ------------------------------------------------------------------ panel
class VaultPanel {
  constructor(node, textWidget) {
    this.node = node;
    this.tw = textWidget;
    this.items = [];
    this.mode = "loaded"; // loaded | search | saved
    this.expanded = new Set();
    this.progress = {};
    this.search = { q: "", sort: SORTS[0], nsfw: false, items: [], local: [], cursor: null, loading: false, error: "" };
    this.saved = { q: "", items: [] };
    this.blur = ls.get("blur", true);
    this.lastText = null;
    this.fetchedModels = new Set();   // model ids whose full version list we've asked Civitai for
    this.verErr = {};
    this.peek = new Set();            // version ids whose previews are expanded in the list
    this.bases = [];
    this.base = (node.properties && node.properties.vault_base) || ls.get("base", "");

    this.baseSel = h("select", { title: "Base model you're generating with. Filters Civitai search and flags mismatched LoRAs.", onchange: () => this.setBase(this.baseSel.value) });
    this.qInput = h("input", { placeholder: "Search Civitai LoRAs…", style: { flex: "1", minWidth: "90px" }, onkeydown: (e) => { if (e.key === "Enter") this.runSearch(true); } });
    this.body = h("div", { class: "lv-body" });
    this.root = h("div", { class: "lv" },
      h("div", { class: "lv-bar" },
        this.baseSel, this.qInput,
        h("button", { class: "pri", onclick: () => this.runSearch(true), title: "Search all of Civitai" }, "Search"),
        h("button", { onclick: () => this.showSaved(), title: "Browse LoRAs you've already saved" }, "Saved"),
        h("button", { onclick: () => { this.mode = "loaded"; this.render(); }, title: "Show the LoRAs in this node" }, "In node"),
        this.blurBox = h("label", { class: "lv-row", title: "Blur mature previews (shared with the vault page)" },
          h("input", { type: "checkbox", checked: this.blur, onchange: (e) => this.setBlur(e.target.checked) }), "blur"),
        h("button", { onclick: () => window.open("/lora_vault", "_blank"), title: "Open the full vault manager" }, "Vault ↗")),
      this.body);
    this.populateBases();
    this.loadBlur();
    this.render();

    const poll = () => { if (this.tw.value !== this.lastText) this.refresh(); };
    this.timer = setInterval(poll, 700);
    if (this.tw.inputEl) this.tw.inputEl.addEventListener("input", () => this.refreshSoon());
  }

  destroy() { clearInterval(this.timer); this.closeCog(); }

  // -------- blur setting (stored on the server so the node and the vault page agree)
  async loadBlur() {
    try { const s = await call("/settings"); this.blur = s.blur_mature !== false; ls.set("blur", this.blur); this.blurBox.firstChild.checked = this.blur; this.render(); } catch (_) { /* keep local value */ }
  }
  async setBlur(v) {
    this.blur = v; ls.set("blur", v); this.render();
    try { await call("/settings", "POST", { blur_mature: v }); } catch (_) { /* ignore */ }
  }

  // -------- base model
  async populateBases() {
    let bases = [];
    try { bases = (await call("/bases")).bases; } catch (_) { /* offline: leave empty */ }
    this.bases = bases;
    this.baseSel.replaceChildren(h("option", { value: "" }, "Any base model"),
      ...bases.map((b) => h("option", { value: b }, b)));
    if (this.base && !bases.includes(this.base)) this.baseSel.append(h("option", { value: this.base }, this.base));
    this.baseSel.value = this.base || "";
  }
  setBase(b) {
    this.base = b;
    ls.set("base", b);
    if (this.node.properties) this.node.properties.vault_base = b;
    if (this.mode === "search" && (this.search.q || this.search.items.length)) this.runSearch(true);
    else if (this.mode === "saved") this.showSaved();
    else this.render();
  }

  // -------- text <-> tags
  get text() { return this.tw.value || ""; }
  setText(t) {
    this.tw.value = t;
    if (this.tw.inputEl) this.tw.inputEl.value = t;
    if (this.tw.callback) this.tw.callback(t);
    app.graph && app.graph.setDirtyCanvas(true, true);
    this.refresh();
  }
  mutate(fn) {
    let i = 0;
    return this.text.replace(TAG_RE, (m, body) => fn(i++, m, body));
  }
  addTag(versionId, ref) {
    ref = ref || "v" + versionId;
    const tag = fmtTag(ref, 1, 1);
    if (this.inNode(versionId, ref)) return;
    this.setText((this.text.trimEnd() + "\n" + tag).trim());
  }
  setStrength(idx, it, s, c) {
    this.setText(this.mutate((i, m) => (i === idx ? fmtTag(it.ref, s, c, it.layer_start, it.layer_end) : m)));
  }
  setLayers(idx, it, a, b) {
    this.setText(this.mutate((i, m) => (i === idx ? fmtTag(it.ref, it.strength, it.clip_strength, a, b) : m)));
  }
  removeAt(idx) {
    this.setText(this.mutate((i, m) => (i === idx ? "" : m)).replace(/\n{2,}/g, "\n").trim());
  }

  // -------- resolving
  refreshSoon() { clearTimeout(this._t); this._t = setTimeout(() => this.refresh(), 350); }
  async refresh() {
    const text = this.text;
    this.lastText = text;
    try {
      this.items = (await call("/resolve", "POST", { text })).items;
    } catch (e) {
      this.items = [{ ok: false, ref: "?", error: "Can't reach the vault server: " + e.message }];
    }
    this.render();
    this.fetchVersionLists();
  }
  // Older saved entries don't know their sibling versions: ask Civitai (through Firefox) once, remember the answer.
  async fetchVersionLists(force) {
    for (const it of this.items) {
      const v = it.ok && it.kind === "vault" && it.item;
      if (!v || (v.versions_known && !force) || this.fetchedModels.has(v.model_id)) continue;
      this.fetchedModels.add(v.model_id);
      try { await call("/model_versions/" + v.model_id); delete this.verErr[v.model_id]; }
      catch (e) { this.verErr[v.model_id] = e.message; }
      try { this.items = (await call("/resolve", "POST", { text: this.text })).items; } catch (_) { /* keep old */ }
      this.render();
    }
  }

  // -------- rendering
  render() {
    const top = this.body.scrollTop;
    this._render();
    this.body.scrollTop = top;
  }
  _render() {
    if (this.mode === "search") return this.renderSearch();
    if (this.mode === "saved") return this.renderSaved();
    const kids = [];
    if (!this.items.length) {
      kids.push(h("div", { class: "lv-empty" }, "No LoRAs yet. Pick a base model above and search Civitai, or open Saved. ",
        "LoRAs download only when you run the workflow."));
    }
    this.items.forEach((it, idx) => kids.push(this.card(it, idx)));
    this.body.replaceChildren(...kids);
  }

  thumb(src, nsfw, onclick) {
    const blur = this.blur && nsfw >= 4;
    const t = h("div", { class: "lv-thumb" + (blur ? " blur" : ""), onclick: (e) => { if (t.classList.contains("blur")) { t.classList.remove("blur"); e.stopPropagation(); } else if (onclick) onclick(e); } });
    if (src) t.style.backgroundImage = `url("${src}")`;
    return t;
  }

  card(it, idx) {
    if (!it.ok) {
      return h("div", { class: "lv-card bad" }, h("div", { class: "lv-main" },
        h("div", { class: "lv-title" }, it.ref),
        h("div", { class: "lv-err" }, it.error),
        h("div", { class: "lv-row" }, h("button", { class: "mini", onclick: () => this.removeAt(idx) }, "Remove"))));
    }
    if (it.kind === "file") {
      return h("div", { class: "lv-card" }, h("div", { class: "lv-main" },
        h("div", { class: "lv-title" }, it.name), h("div", { class: "lv-sub" }, "Local file (not in vault)"),
        this.strengthRow(it, idx, null),
        h("div", { class: "lv-row" }, h("button", { class: "mini", onclick: () => this.removeAt(idx) }, "Remove"))));
    }
    const v = it.item;
    const st = STATUS[v.status];
    const open = this.expanded.has(v.version_id);
    const prog = this.progress[v.version_id];
    const first = (v.previews || []).find((p) => p.type === "image");
    const mismatch = this.base && v.base_model && v.base_model !== this.base;
    const localChip = v.local === "pinned" ? ["📌 Pinned", "ok"] : v.local === "cached" ? ["✔ On disk", "ok"] : ["☁ Downloads on run", ""];
    const card = h("div", { class: "lv-card" + (st ? " " + st[1] : "") },
      this.thumb(first ? first.src : v.thumb, first ? first.nsfwLevel : (v.thumb_nsfw || 0), () => this.toggle(v.version_id)),
      h("div", { class: "lv-main" },
        h("div", { class: "lv-title", title: `${v.model_name} — ${v.version_name}` }, v.model_name),
        h("div", { class: "lv-sub" }, `${v.version_name}${v.creator ? " · by " + v.creator : ""}`),
        h("div", { class: "lv-row" },
          h("span", { class: "lv-chip" }, v.base_model || "?"),
          mismatch && h("span", { class: "lv-chip warn", title: `You selected ${this.base}` }, "⚠ different base"),
          h("span", { class: "lv-chip " + localChip[1] }, localChip[0]),
          st && h("span", { class: "lv-chip " + st[1], title: st[2] }, st[0])),
        prog && !prog.finished && h("div", { class: "lv-bar-prog" }, h("div", { style: { width: prog.total ? Math.min(100, (prog.done / prog.total) * 100) + "%" : "30%" } })),
        this.strengthRow(it, idx, v),
        (v.versions || []).length > 1 && h("div", { class: "lv-row", title: "Other versions on this model page - open Details to look at them or add several" },
          h("span", { style: { opacity: ".7" } }, `${v.versions.length} versions:`),
          ...v.versions.slice(0, 5).map((o) => h("span", { class: "lv-chip" + (o.id === v.version_id ? " ok" : this.inNode(o.id, o.tag_ref) ? " ok" : ""), style: { cursor: "pointer" },
            onclick: () => { this.expanded.add(v.version_id); this.peek.add(o.id); this.render(); } }, o.name + (this.inNode(o.id, o.tag_ref) ? " ✔" : ""))),
          v.versions.length > 5 && h("span", { class: "lv-chip", style: { cursor: "pointer" }, onclick: () => this.toggle(v.version_id) }, `+${v.versions.length - 5}`)),
        this.verErr[v.model_id] && h("div", { class: "lv-err" }, "Couldn't list other versions: " + this.verErr[v.model_id]),
        (v.trigger_words || []).length > 0 && h("div", { class: "lv-row" }, ...v.trigger_words.map((w) =>
          h("span", { class: "lv-chip tw", title: "Click to copy", onclick: () => navigator.clipboard && navigator.clipboard.writeText(w) }, w))),
        h("div", { class: "lv-row" },
          h("button", { class: "mini", onclick: () => this.toggle(v.version_id) }, open ? "Less ▲" : "Details ▼"),
          v.local === "pinned"
            ? h("button", { class: "mini", onclick: () => this.act("unpin", v) }, "Unpin")
            : h("button", { class: "mini", title: "Download now and keep forever (backup)", onclick: () => this.act("pin", v) }, "📌 Pin"),
          h("button", { class: "mini", onclick: () => this.removeAt(idx) }, "Remove")),
        open && this.detail(v)));
    return card;
  }

  strengthRow(it, idx, v) {
    const split = it.clip_strength !== it.strength;
    const ranged = it.layer_start != null || it.layer_end != null;
    const num = (x, fallback) => (isNaN(x) ? fallback : x);
    const input = h("input", { class: "num", type: "number", step: "0.05", value: +it.strength.toFixed(3),
      title: split ? "Model strength (clip strength is split: see the cog)" : "Strength for model and clip",
      onchange: (e) => { const s = num(parseFloat(e.target.value), 1); this.setStrength(idx, it, s, split ? it.clip_strength : s); } });
    const cog = h("button", { class: "cog mini" + (split || ranged ? " on" : ""), title: "Layer range and split clip/model strength", onclick: (e) => { e.stopPropagation(); this.openCog(cog, idx, it); } }, "⚙");
    return h("div", { class: "lv-row" },
      h("span", { style: { opacity: ".7" } }, split ? "model" : "strength"), input,
      split && h("span", { class: "lv-chip", title: "Clip strength (split)" }, "clip " + +it.clip_strength.toFixed(3)),
      ranged && h("span", { class: "lv-chip", title: "Only these transformer layers are patched" }, `layers ${it.layer_start ?? 0}–${it.layer_end ?? "end"}`),
      cog);
  }

  // Little cog menu: layer range + split clip/model strength (floating, so cards don't grow).
  openCog(anchor, idx, it) {
    this.closeCog();
    const num = (x) => { const n = parseInt(x, 10); return isNaN(n) || n < 0 ? null : n; };
    // The card is redrawn after every change, so keep this menu's own copy of the values and always write all of them.
    const st = { s: it.strength, c: it.clip_strength, a: it.layer_start, b: it.layer_end };
    const write = () => this.setText(this.mutate((i, m) => (i === idx ? fmtTag(it.ref, st.s, st.c, st.a, st.b) : m)));
    const split = st.c !== st.s;
    const startIn = h("input", { class: "num", type: "number", min: "0", step: "1", placeholder: "first", value: st.a ?? "" });
    const endIn = h("input", { class: "num", type: "number", min: "0", step: "1", placeholder: "last", value: st.b ?? "" });
    startIn.onchange = endIn.onchange = () => { st.a = num(startIn.value); st.b = num(endIn.value); write(); };
    const clipIn = h("input", { class: "num", type: "number", step: "0.05", value: +st.c.toFixed(3), disabled: !split,
      onchange: (e) => { const c = parseFloat(e.target.value); st.c = isNaN(c) ? st.s : c; write(); } });
    const splitBox = h("input", { type: "checkbox", checked: split, onchange: (e) => {
      clipIn.disabled = !e.target.checked;
      if (!e.target.checked) { st.c = st.s; clipIn.value = +st.s.toFixed(3); write(); }
    } });
    const panel = h("div", { class: "lv lv-cog", style: { position: "fixed", zIndex: 10000, width: "250px", padding: "10px", height: "auto", minHeight: "0",
      background: "var(--comfy-menu-bg,#2b2b2b)", border: "1px solid var(--border-color,#666)", borderRadius: "8px", boxShadow: "0 6px 24px rgba(0,0,0,.5)" } },
      h("div", { style: { fontWeight: "600", marginBottom: "6px" } }, "LoRA settings"),
      h("div", { class: "lv-row", style: { justifyContent: "space-between" }, title: "Patch only these transformer layers (inclusive). Blank = all. Early layers mostly shape structure and identity, late layers more of the style and texture." },
        h("span", {}, "Layers"), h("span", { class: "lv-row" }, startIn, " to ", endIn)),
      h("div", { class: "lv-sub", style: { whiteSpace: "normal", margin: "2px 0 8px" } }, "Blank = all layers. Try a later range for a style LoRA to keep edits accurate."),
      h("label", { class: "lv-row", style: { justifyContent: "space-between" } }, h("span", { class: "lv-row" }, splitBox, "Split clip / model"), clipIn),
      h("div", { class: "lv-row", style: { marginTop: "8px", justifyContent: "flex-end" } },
        h("button", { class: "mini", onclick: () => { st.a = st.b = null; startIn.value = endIn.value = ""; write(); } }, "Reset layers"),
        h("button", { class: "mini pri", onclick: () => this.closeCog() }, "Done")));
    document.body.append(panel);
    const r = anchor.getBoundingClientRect();
    panel.style.left = Math.max(8, Math.min(r.left, innerWidth - panel.offsetWidth - 8)) + "px";
    panel.style.top = Math.max(8, Math.min(r.bottom + 4, innerHeight - panel.offsetHeight - 8)) + "px";
    this._cog = panel;
    this._cogOutside = (e) => { if (!panel.contains(e.target) && e.target !== anchor) this.closeCog(); };
    this._cogKey = (e) => { if (e.key === "Escape") this.closeCog(); };
    setTimeout(() => { document.addEventListener("pointerdown", this._cogOutside, true); document.addEventListener("keydown", this._cogKey, true); }, 0);
  }
  closeCog() {
    if (this._cog) this._cog.remove();
    this._cog = null;
    document.removeEventListener("pointerdown", this._cogOutside, true);
    document.removeEventListener("keydown", this._cogKey, true);
  }

  detail(v) {
    const s = v.stats || {};
    const lic = v.license || {};
    return h("div", { class: "lv-detail" },
      h("div", { class: "lv-row" },
        h("span", { class: "lv-chip" }, `⬇ ${fmtNum(s.downloadCount)}`), h("span", { class: "lv-chip" }, `👍 ${fmtNum(s.thumbsUpCount)}`),
        v.file_size_kb && h("span", { class: "lv-chip" }, fmtSize(v.file_size_kb)),
        v.model_type && h("span", { class: "lv-chip" }, v.model_type),
        v.published_at && h("span", { class: "lv-chip" }, "published " + String(v.published_at).slice(0, 10)),
        lic.allowCommercialUse != null && h("span", { class: "lv-chip", title: "Commercial use" }, "license: " + JSON.stringify(lic.allowCommercialUse).replace(/[{}"[\]]/g, ""))),
      (v.tags || []).length > 0 && h("div", { class: "lv-row" }, ...v.tags.slice(0, 10).map((t) => h("span", { class: "lv-chip" }, t))),
      (v.previews || []).length > 0 && h("div", { class: "lv-strip" }, ...v.previews.map((p) => this.previewThumb(p))),
      v.version_description && h("div", { class: "lv-desc" }, v.version_description),
      v.description && h("div", { class: "lv-desc" }, v.description),
      v.status_detail && h("div", { class: STATUS[v.status] ? "lv-err" : "" }, v.status_detail),
      (v.versions || v.other_versions || []).length > 1 && this.versionList(v),
      h("div", { class: "lv-row" },
        h("a", { href: v.civitai_url, target: "_blank", rel: "noopener" }, "Open on Civitai ↗"),
        h("button", { class: "mini", onclick: () => this.act("check", v) }, "Check still exists")));
  }

  inNode(id, ref) {
    if (new RegExp(`<lora:v${id}[:>]`).test(this.text)) return true;
    if (this.items.some((it) => it.ok && it.kind === "vault" && it.item.version_id === id)) return true;
    return !!ref && this.text.toLowerCase().includes(`<lora:${ref.toLowerCase()}:`) || !!ref && this.text.toLowerCase().includes(`<lora:${ref.toLowerCase()}>`);
  }
  versionList(v) {
    const list = v.versions || v.other_versions;
    const mk = (o) => {
      const used = this.inNode(o.id, o.tag_ref), cur = o.id === v.version_id, open = this.peek.has(o.id);
      const t = this.thumb(o.thumb, o.thumb_nsfw || 0, () => { open ? this.peek.delete(o.id) : this.peek.add(o.id); this.render(); });
      t.style.width = "36px"; t.style.height = "48px";
      return h("div", { class: "lv-detail", style: { borderTop: "none", margin: 0, padding: "2px 0" } },
        h("div", { class: "lv-row" }, t,
          h("div", { style: { flex: "1", minWidth: "0" } },
            h("div", {}, h("span", { class: "lv-chip" + (cur ? " ok" : "") }, o.name + (o.saved ? " ✔" : "")), " ", h("span", { style: { opacity: ".6" } }, o.baseModel || "")),
            (o.trainedWords || []).length > 0 && h("div", { class: "lv-sub", title: o.trainedWords.join(", ") }, "🏷 " + o.trainedWords.slice(0, 4).join(", "))),
          h("button", { class: "mini", onclick: () => { open ? this.peek.delete(o.id) : this.peek.add(o.id); this.render(); } }, open ? "Hide" : "Look"),
          used ? h("span", { class: "lv-chip ok" }, "in node")
            : h("button", { class: "mini", title: "Use this version in addition to the others", onclick: () => this.addVersion(o) }, "Add also"),
          !cur && !used && h("button", { class: "mini", title: "Replace the current version with this one", onclick: () => this.swapVersion(v, o) }, "Use instead")),
        open && this.versionPeek(o));
    };
    return h("div", { class: "lv-detail", style: { borderTop: "none", margin: 0, padding: 0 } },
      h("div", { style: { opacity: ".7" } }, `Versions (${list.length}) - look at any of them without saving; add several to use them together`),
      ...list.map(mk),
      h("div", { class: "lv-row" }, h("button", { class: "mini", onclick: () => { this.fetchedModels.delete(v.model_id); this.fetchVersionLists(true); } }, "↻ Refresh list")));
  }
  versionPeek(o) {
    const imgs = (o.images || []).map((i, n) => ({ idx: n, type: i.type, nsfwLevel: i.nsfwLevel || 0, meta: i.meta, src: i.src, full: i.url, local: false }));
    return h("div", { class: "lv-detail" },
      imgs.length ? h("div", { class: "lv-strip" }, ...imgs.map((p) => this.previewThumb(p))) : h("div", { class: "lv-sub" }, "No previews listed."),
      o.description_html && h("div", { class: "lv-desc" }, new DOMParser().parseFromString(o.description_html, "text/html").body.textContent.trim()),
      (o.trainedWords || []).length > 0 && h("div", { class: "lv-row" }, ...o.trainedWords.map((w) =>
        h("span", { class: "lv-chip tw", title: "Click to copy", onclick: () => navigator.clipboard && navigator.clipboard.writeText(w) }, w))));
  }
  async addVersion(o) {
    try { if (!o.saved) await call("/save", "POST", { version_id: o.id }); o.saved = true; this.addTag(o.id, o.tag_ref); }
    catch (e) { alert("Couldn't add: " + e.message); }
  }

  previewThumb(p) {
    const el = this.thumb(p.type === "image" ? p.src : null, p.nsfwLevel || 0, () => {
      const m = p.meta;
      if (m && m.prompt && navigator.clipboard) navigator.clipboard.writeText(m.prompt);
      this._showMeta(el, p);
    });
    if (p.type !== "image") {
      if (this.blur && (p.nsfwLevel || 0) >= 4) el.append(h("span", { style: { position: "absolute", bottom: "2px", left: "4px" } }, "▶ video"));
      else { // autoplay silently; no audio in the node panel
        const v = h("video", { src: p.local ? p.src : p.full, loop: true, autoplay: true, playsinline: true, style: { width: "100%", height: "100%", objectFit: "cover", pointerEvents: "none" } });
        v.muted = true; v.play().catch(() => {}); el.append(v);
      }
    }
    return el;
  }
  _showMeta(el, p) {
    const strip = el.parentElement;
    let box = strip.nextSibling && strip.nextSibling.classList && strip.nextSibling.classList.contains("lv-meta") ? strip.nextSibling : null;
    const m = p.meta || {};
    const neg = m.negativePrompt;
    const rest = Object.entries(m).filter(([k, v]) => !["prompt", "negativePrompt", "resources", "civitaiResources"].includes(k) && typeof v !== "object").map(([k, v]) => `${k}: ${v}`);
    const parts = [];
    if (m.prompt) parts.push("PROMPT (copied):\n" + m.prompt);
    if (neg) parts.push("NEGATIVE:\n" + neg);
    if (rest.length) parts.push(rest.join(" · "));
    if (!parts.length) parts.push("No generation data on this image.");
    if (!p.local) parts.push("(preview image not stored locally - pin to keep it)");
    if (!box) { box = h("div", { class: "lv-meta" }); strip.after(box); }
    box.textContent = parts.join("\n\n");
  }

  toggle(vid) {
    this.expanded.has(vid) ? this.expanded.delete(vid) : this.expanded.add(vid);
    this.render();
  }

  async swapVersion(v, other) {
    try {
      await call("/save", "POST", { version_id: other.id });
      const ref = other.tag_ref || "v" + other.id;
      const idxs = this.items.map((it, i) => (it.ok && it.kind === "vault" && it.item.version_id === v.version_id ? i : -1)).filter((i) => i >= 0);
      this.setText(this.mutate((i, m) => { const k = idxs.indexOf(i); if (k < 0) return m; const it = this.items[i]; return fmtTag(ref, it.strength, it.clip_strength, it.layer_start, it.layer_end); }));
    } catch (e) { alert("Couldn't switch version: " + e.message); }
  }

  async act(name, v) {
    try {
      await call("/" + name, "POST", { version_id: v.version_id });
      if (name === "pin") this.pollJobs();
      else await this.refresh();
    } catch (e) { alert(e.message); }
  }

  async pollJobs() {
    for (let i = 0; i < 3600; i++) {
      let jobs = {};
      try { jobs = (await call("/jobs")).jobs; } catch (_) { break; }
      for (const [id, j] of Object.entries(jobs)) this.progress[id] = { done: j.done, total: j.total, finished: j.state !== "downloading" && j.state !== "queued" };
      await this.refresh();
      const busy = Object.values(jobs).some((j) => j.state === "downloading" || j.state === "queued");
      if (!busy) break;
      await new Promise((r) => setTimeout(r, 1000));
    }
  }

  onProgress(d) {
    this.progress[d.version_id] = { done: d.done, total: d.total, finished: !!d.finished };
    if (this.mode === "loaded") this.render();
    if (d.finished) this.refresh();
  }

  // -------- saved list
  async showSaved() {
    this.mode = "saved";
    try {
      const q = new URLSearchParams({ q: this.saved.q || "" });
      if (this.base) q.set("base", this.base);
      this.saved.items = (await call("/list?" + q)).items;
    } catch (e) { this.saved.items = []; this.saved.error = e.message; }
    this.render();
  }
  savedRow(v, leave) {
    const used = this.inNode(v.version_id, v.tag_ref);
    return h("div", { class: "lv-card" },
      this.thumb(v.thumb, v.thumb_nsfw, null),
      h("div", { class: "lv-main" },
        h("div", { class: "lv-title", title: v.model_name }, v.model_name), h("div", { class: "lv-sub" }, `${v.version_name} · ${v.base_model}`),
        h("div", { class: "lv-row" },
          h("span", { class: "lv-chip " + (v.local === "none" ? "" : "ok") }, v.local === "pinned" ? "📌 Pinned" : v.local === "cached" ? "✔ On disk" : "☁ Not downloaded"),
          STATUS[v.status] && h("span", { class: "lv-chip " + STATUS[v.status][1] }, STATUS[v.status][0]),
          used ? h("span", { class: "lv-chip ok" }, "✔ In node")
            : h("button", { class: "mini pri", onclick: () => { this.addTag(v.version_id, v.tag_ref); if (leave) this.mode = "loaded"; else this.render(); } }, "Use"))));
  }
  renderSaved() {
    const filter = h("input", { placeholder: "Filter saved…", value: this.saved.q, style: { width: "100%" },
      oninput: (e) => { this.saved.q = e.target.value; clearTimeout(this._st); this._st = setTimeout(() => this.showSaved(), 300); } });
    const rows = this.saved.items.map((v) => this.savedRow(v, true));
    this.body.replaceChildren(filter, ...(rows.length ? rows : [h("div", { class: "lv-empty" }, this.saved.error || (this.base ? `Nothing saved for ${this.base}.` : "Nothing saved yet."))]));
  }

  // -------- civitai search
  async runSearch(reset) {
    const s = this.search;
    s.q = this.qInput.value.trim();
    this.mode = "search";
    if (reset) { s.items = []; s.cursor = null; s.local = []; }
    s.loading = true; s.error = "";
    this.render();
    if (reset) {   // your own library first: works even when Civitai / Firefox is unreachable
      try {
        const lq = new URLSearchParams({ q: s.q });
        if (this.base) lq.set("base", this.base);
        const mine = (await call("/list?" + lq)).items;
        s.local = mine.sort((a, b) => (b.local === "pinned") - (a.local === "pinned") || (b.pinned ? 1 : 0) - (a.pinned ? 1 : 0));
      } catch (_) { s.local = []; }
      this.render();
    }
    try {
      const p = new URLSearchParams({ q: s.q, sort: s.sort, nsfw: s.nsfw ? "1" : "0" });
      if (this.base) p.set("base", this.base);
      if (!reset && s.cursor) p.set("cursor", s.cursor);
      const r = await call("/search?" + p);
      s.items = s.items.concat(r.items);
      s.cursor = r.next_cursor;
    } catch (e) { s.error = e.message; }
    s.loading = false;
    this.render();
  }

  renderSearch() {
    const s = this.search;
    const ctl = h("div", { class: "lv-row" },
      h("select", { onchange: (e) => { s.sort = e.target.value; this.runSearch(true); } }, ...SORTS.map((x) => h("option", { value: x, selected: x === s.sort }, x))),
      h("label", { class: "lv-row" }, h("input", { type: "checkbox", checked: s.nsfw, onchange: (e) => { s.nsfw = e.target.checked; this.runSearch(true); } }), "mature"),);
    // order: pinned, then saved, then everything else on Civitai that matches
    const local = s.local || [];
    const pinned = local.filter((v) => v.local === "pinned" || v.pinned), saved = local.filter((v) => !(v.local === "pinned" || v.pinned));
    const have = new Set(local.map((v) => v.version_id));
    const civit = s.items.filter((m) => !m.versions.every((v) => have.has(v.id)));   // skip models you already have in full
    const head = (t, n) => h("div", { class: "lv-sub", style: { fontWeight: "600", marginTop: "4px" } }, `${t} (${n})`);
    const rows = civit.map((m) => this.resultCard(m));
    const tail = s.loading ? h("div", { class: "lv-empty" }, "Searching Civitai…")
      : s.error ? h("div", { class: "lv-empty lv-err" }, "Civitai: " + s.error)
      : !rows.length && !local.length ? h("div", { class: "lv-empty" }, this.base ? `No LoRAs found for ${this.base}.` : "No results.")
      : s.cursor ? h("button", { onclick: () => this.runSearch(false) }, "Load more") : null;
    this.body.replaceChildren(...[ctl,
      pinned.length > 0 && head("📌 Pinned", pinned.length), ...pinned.map((v) => this.savedRow(v, false)),
      saved.length > 0 && head("Saved", saved.length), ...saved.map((v) => this.savedRow(v, false)),
      (rows.length > 0 || s.loading || s.error) && head("From Civitai", rows.length), ...rows, tail].filter(Boolean));
  }

  resultCard(m) {
    const thumb = this.thumb(m.versions[0].thumb, m.versions[0].thumb_nsfw || 0, null);
    const rows = h("div", { class: "lv-detail", style: { borderTop: "none", margin: 0, padding: 0 } });
    let sel = m.versions[0];
    const draw = () => {
      rows.replaceChildren(...m.versions.map((v) => {
        const ref = nameRef(m.name, v.name, v.id);
        const used = this.inNode(v.id, ref);
        const btn = h("button", { class: "mini" + (used || v.saved ? "" : " pri") }, used ? "✔ In node" : v.saved ? "Use" : "Add");
        btn.disabled = used;
        btn.addEventListener("click", async () => {
          btn.disabled = true; btn.textContent = "Saving…";
          try { if (!v.saved) await call("/save", "POST", { version_id: v.id }); v.saved = true; this.addTag(v.id, ref); draw(); }
          catch (e) { btn.disabled = false; btn.textContent = "Add"; alert("Couldn't add: " + e.message); }
        });
        return h("div", { class: "lv-row", onmouseenter: () => { sel = v; thumb.style.backgroundImage = v.thumb ? `url("${v.thumb}")` : ""; } },
          h("span", { class: "lv-chip" }, v.name), h("span", { style: { opacity: ".6" } }, `${v.baseModel || ""} ${fmtSize(v.size_kb)}`),
          v.saved && h("span", { class: "lv-chip ok", title: "Already in your vault" }, "saved"), btn,
          (v.trainedWords || []).length > 0 && h("span", { class: "lv-sub", title: v.trainedWords.join(", ") }, v.trainedWords.slice(0, 3).join(", ")));
      }));
    };
    draw();
    return h("div", { class: "lv-card" }, thumb, h("div", { class: "lv-main" },
      h("div", { class: "lv-title", title: m.name }, m.name),
      h("div", { class: "lv-sub" }, `by ${m.creator || "?"} · ⬇ ${fmtNum(m.downloads)} · 👍 ${fmtNum(m.likes)}`),
      rows,
      h("div", { class: "lv-row" }, h("span", { style: { opacity: ".6" } }, `${this.itemCount()} in node`),
        h("a", { href: `https://civitai.com/models/${m.model_id}`, target: "_blank", rel: "noopener" }, "↗"))));
  }
  itemCount() { return (this.text.match(TAG_RE) || []).length; }
}

// ------------------------------------------------------------------ extension
app.registerExtension({
  name: "LoraVault.Panel",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE) return;
    injectCss();
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated ? onCreated.apply(this, arguments) : undefined;
      const tw = this.widgets && this.widgets.find((w) => w.name === "lora_text");
      if (tw) {
        this.properties = this.properties || {};
        const panel = new VaultPanel(this, tw);
        this._vaultPanel = panel;
        this.addDOMWidget("vault_panel", "div", panel.root, { serialize: false, hideOnZoom: false, getMinHeight: () => 320 });
        this.setSize([Math.max(this.size[0], 460), Math.max(this.size[1], 560)]);
        panel.refresh();
      }
      return r;
    };
    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      if (this._vaultPanel) this._vaultPanel.destroy();
      return onRemoved ? onRemoved.apply(this, arguments) : undefined;
    };
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
      const p = this._vaultPanel;
      if (p) {
        if (this.properties && this.properties.vault_base) { p.base = this.properties.vault_base; p.baseSel.value = p.base; }
        p.refresh();
      }
      return r;
    };
  },
  setup() {
    api.addEventListener("lora_vault.progress", (e) => {
      for (const n of (app.graph && app.graph._nodes) || []) {
        if (n._vaultPanel) n._vaultPanel.onProgress(e.detail);
      }
    });
  },
});
