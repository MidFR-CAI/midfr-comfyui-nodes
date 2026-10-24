import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const POLL_MS = 3000;
let timer = null;
let state = null;
let busy = false;

/* ------------------------------------------------------------------ style */

const CSS = `
.rr { font-size: 13px; color: var(--fg-color, #ddd); padding: 10px 12px 20px;
      font-variant-numeric: tabular-nums; line-height: 1.45; }
.rr h3 { font-size: 12px; font-weight: 600; letter-spacing: .01em; margin: 18px 0 8px;
         color: var(--fg-color, #ddd); opacity: .75; }
.rr h3:first-child { margin-top: 4px; }
.rr .device { opacity: .6; font-size: 11px; margin-bottom: 12px; }

.rr-meter { margin-bottom: 12px; }
.rr-meter .row { display: flex; justify-content: space-between; align-items: baseline;
                 margin-bottom: 5px; }
.rr-meter .label { font-weight: 500; }
.rr-meter .value { opacity: .7; font-size: 12px; }
.rr-track { height: 10px; border-radius: 2px; overflow: hidden;
            background: var(--comfy-input-bg, #1a1a1a);
            box-shadow: inset 0 0 0 1px var(--border-color, #3a3a3a); }
.rr-fill { height: 100%; width: 0; transition: width .45s cubic-bezier(.22,.61,.36,1);
           background: #4f9d69; }
.rr-fill.warn { background: #c9a227; }
.rr-fill.critical { background: #c0554a; }
.rr-fill.self { background: #5b8fb9; }
.rr-split { display: flex; height: 100%; }

.rr-btn { font: inherit; padding: 6px 12px; border-radius: 3px; cursor: pointer;
          color: var(--fg-color, #ddd); background: var(--comfy-input-bg, #222);
          border: 1px solid var(--border-color, #3a3a3a); }
.rr-btn:hover:not(:disabled) { background: var(--comfy-menu-bg, #333); }
.rr-btn:disabled { opacity: .5; cursor: default; }
.rr-btn.primary { background: #3d6b8a; border-color: #3d6b8a; color: #fff; }
.rr-btn.primary:hover:not(:disabled) { background: #47799c; }
.rr-btn.small { padding: 3px 9px; font-size: 12px; }
.rr-btn:focus-visible { outline: 2px solid #7fb4d8; outline-offset: 2px; }

.rr-controls { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
.rr select, .rr input[type=number] { font: inherit; padding: 5px 6px; border-radius: 3px;
    color: var(--fg-color, #ddd); background: var(--comfy-input-bg, #222);
    border: 1px solid var(--border-color, #3a3a3a); }
.rr input[type=number] { width: 62px; }
.rr label.check { display: flex; gap: 7px; align-items: flex-start; cursor: pointer;
                  margin: 7px 0; }
.rr label.check input { margin-top: 2px; flex: none; }
.rr label.check span small { display: block; opacity: .55; font-size: 11px; }

.rr-note { margin-top: 9px; font-size: 12px; opacity: .75; min-height: 1.2em; }
.rr-note.error { color: #e08b80; opacity: 1; }

.rr-item { border-left: 3px solid var(--border-color, #3a3a3a); padding: 7px 0 7px 10px;
           margin-bottom: 9px; }
.rr-item.critical { border-left-color: #c0554a; }
.rr-item.warn { border-left-color: #c9a227; }
.rr-item.info { border-left-color: #4f9d69; }
.rr-item .title { font-weight: 600; margin-bottom: 2px; }
.rr-item .detail { opacity: .72; font-size: 12px; }
.rr-item .rr-btn { margin-top: 7px; }

.rr-proc { display: flex; justify-content: space-between; align-items: center;
           gap: 8px; padding: 5px 0; border-bottom: 1px solid var(--border-color, #333); }
.rr-proc:last-child { border-bottom: 0; }
.rr-proc .name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.rr-proc .name em { font-style: normal; opacity: .5; font-size: 11px; }
.rr-proc .size { opacity: .7; font-size: 12px; flex: none; }
.rr-empty { opacity: .55; font-size: 12px; }
@media (prefers-reduced-motion: reduce) { .rr-fill { transition: none; } }
`;

function injectStyle() {
  if (document.getElementById("rr-style")) return;
  const tag = document.createElement("style");
  tag.id = "rr-style";
  tag.textContent = CSS;
  document.head.appendChild(tag);
}

/* ------------------------------------------------------------------ utils */

const fmt = (bytes) => {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let n = bytes, i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i += 1; }
  return `${i < 2 ? Math.round(n) : n.toFixed(1)} ${units[i]}`;
};

const severity = (pct) => (pct >= 92 ? "critical" : pct >= 80 ? "warn" : "");

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}

async function call(path, options) {
  const res = await api.fetchApi(path, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || `request failed (${res.status})`);
  return body;
}

/* ------------------------------------------------------------------ pieces */

function meter(label, used, total, extra) {
  const pct = total ? (used / total) * 100 : 0;
  const box = el("div", "rr-meter");
  const row = el("div", "row");
  row.append(el("span", "label", label),
             el("span", "value", `${fmt(used)} of ${fmt(total)}${extra ? ` · ${extra}` : ""}`));
  const track = el("div", "rr-track");
  const fill = el("div", `rr-fill ${severity(pct)}`);
  fill.style.width = `${Math.min(100, pct)}%`;
  track.append(fill);
  box.append(row, track);
  return box;
}

function gpuMeter(gpu) {
  const mine = gpu.processes.find((p) => p.is_self)?.vram || 0;
  const others = Math.max(0, gpu.used - mine);
  const box = el("div", "rr-meter");
  const row = el("div", "row");
  const extra = [
    gpu.utilization != null ? `${gpu.utilization}% busy` : null,
    gpu.temperature != null ? `${gpu.temperature}°C` : null,
  ].filter(Boolean).join(" · ");
  row.append(el("span", "label", gpu.name),
             el("span", "value", `${fmt(gpu.free)} free${extra ? ` · ${extra}` : ""}`));

  const track = el("div", "rr-track");
  const split = el("div", "rr-split");
  const self = el("div", "rr-fill self");
  self.style.width = `${(mine / gpu.total) * 100}%`;
  self.title = `ComfyUI: ${fmt(mine)}`;
  const rest = el("div", `rr-fill ${severity(gpu.percent)}`);
  rest.style.width = `${(others / gpu.total) * 100}%`;
  rest.title = `Other programs: ${fmt(others)}`;
  split.append(self, rest);
  track.append(split);
  box.append(row, track);
  return box;
}

function suggestionItem(item, allowTerminate, refresh, note) {
  const box = el("div", `rr-item ${item.severity}`);
  box.append(el("div", "title", item.title), el("div", "detail", item.detail));

  if (allowTerminate && item.pid) {
    const btn = el("button", "rr-btn small", `Close ${item.process}`);
    btn.onclick = async () => {
      if (!confirm(`Close ${item.process} (pid ${item.pid})?\n\nUnsaved work in that program will be lost.`)) return;
      btn.disabled = true;
      try {
        const res = await call("/reaper/terminate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ pid: item.pid }),
        });
        note(`Closed ${res.name}.`);
        refresh();
      } catch (err) {
        note(err.message, true);
        btn.disabled = false;
      }
    };
    box.append(btn);
  }
  return box;
}

function checkbox(labelText, hint, checked, onChange) {
  const label = el("label", "check");
  const input = el("input");
  input.type = "checkbox";
  input.checked = !!checked;
  input.onchange = () => onChange(input.checked);
  const span = el("span", null, labelText);
  if (hint) span.append(el("small", null, hint));
  label.append(input, span);
  return label;
}

/* ------------------------------------------------------------------ render */

function render(root) {
  root.replaceChildren();
  const wrap = el("div", "rr");
  root.append(wrap);

  if (!state) {
    wrap.append(el("div", "rr-empty", "Reading the machine…"));
    return;
  }

  const cfg = state.config;
  const noteEl = el("div", "rr-note");
  const note = (msg, isError) => {
    noteEl.textContent = msg;
    noteEl.classList.toggle("error", !!isError);
  };
  const refresh = () => poll(root);

  wrap.append(el("div", "device", state.platform));

  /* memory ---------------------------------------------------------- */
  if (state.gpus.length) {
    state.gpus.forEach((gpu) => wrap.append(gpuMeter(gpu)));
  } else if (state.torch?.total) {
    wrap.append(meter("Graphics memory", state.torch.used, state.torch.total,
                      `${fmt(state.torch.free)} free`));
  }
  if (state.memory.available) {
    wrap.append(meter("System memory", state.memory.used, state.memory.total,
                      `${fmt(state.memory.free)} free`));
  }

  /* clean now -------------------------------------------------------- */
  wrap.append(el("h3", null, "Free memory now"));
  const controls = el("div", "rr-controls");
  const level = el("select");
  [["light", "Light — keep models loaded"],
   ["standard", "Standard — release unused models"],
   ["aggressive", "Deep — unload everything"]].forEach(([value, text]) => {
    const opt = el("option", null, text);
    opt.value = value;
    level.append(opt);
  });
  level.value = "standard";

  const run = el("button", "rr-btn primary", "Free memory");
  run.onclick = async () => {
    if (busy) return;
    busy = true;
    run.disabled = true;
    note("Working…");
    try {
      const res = await call("/reaper/clean", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ level: level.value, purge_execution_cache: cfg.purge_execution_cache }),
      });
      const gained = res.reclaimed > 0 ? `Freed ${fmt(res.reclaimed)}` : "Nothing left to free";
      note(`${gained} · ${fmt(res.after?.free)} now available · ${res.ms} ms`);
      await poll(root);
    } catch (err) {
      note(err.message, true);
    } finally {
      busy = false;
      run.disabled = false;
    }
  };
  controls.append(level, run);
  wrap.append(controls, noteEl);

  if (state.last_pass) {
    const lp = state.last_pass;
    wrap.append(el("div", "rr-empty",
      `Last automatic pass: ${lp.level} before run ${lp.run}, freed ${fmt(lp.reclaimed)}.`));
  }

  /* suggestions ------------------------------------------------------ */
  wrap.append(el("h3", null, "What to close"));
  if (!state.psutil) {
    wrap.append(el("div", "rr-empty",
      "Install psutil to see what else is using this machine: pip install psutil"));
  } else {
    state.suggestions.forEach((s) =>
      wrap.append(suggestionItem(s, cfg.allow_terminate, refresh, note)));
    if (!state.nvml) {
      wrap.append(el("div", "rr-empty",
        "Per-program graphics memory needs NVML: pip install nvidia-ml-py"));
    }
  }

  /* processes -------------------------------------------------------- */
  if (state.processes.length) {
    wrap.append(el("h3", null, "Biggest memory users"));
    state.processes.forEach((p) => {
      const row = el("div", "rr-proc");
      const name = el("div", "name", p.is_self ? `${p.name} (this ComfyUI)` : p.name);
      if (p.count > 1) name.append(el("em", null, ` ${p.count} processes`));
      row.append(name, el("div", "size", fmt(p.rss)));
      wrap.append(row);
    });
  }

  /* settings --------------------------------------------------------- */
  wrap.append(el("h3", null, "Settings"));
  const push = async (updates) => {
    try {
      state.config = await call("/reaper/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(updates),
      });
      render(root);
    } catch (err) {
      note(err.message, true);
    }
  };

  wrap.append(checkbox("Free memory before every run",
    "Runs a pass on the queue worker, before any node executes.",
    cfg.auto_clean, (v) => push({ auto_clean: v })));

  const modeRow = el("div", "rr-controls");
  modeRow.append(el("span", null, "How hard to clean"));
  const mode = el("select");
  [["adaptive", "Adaptive — only when memory is tight"],
   ["light", "Light every run"],
   ["standard", "Standard every run"],
   ["aggressive", "Deep every run"],
   ["off", "Never"]].forEach(([value, text]) => {
    const opt = el("option", null, text);
    opt.value = value;
    mode.append(opt);
  });
  mode.value = cfg.mode;
  mode.onchange = () => push({ mode: mode.value });
  modeRow.append(mode);
  wrap.append(modeRow);

  if (cfg.mode === "adaptive") {
    const row = el("div", "rr-controls");
    row.style.marginTop = "7px";
    row.append(el("span", null, "Clean deeply below"));
    const pct = el("input");
    pct.type = "number";
    pct.min = 5; pct.max = 90; pct.step = 5;
    pct.value = cfg.adaptive_free_vram_pct;
    pct.onchange = () => push({ adaptive_free_vram_pct: parseFloat(pct.value) });
    row.append(pct, el("span", null, "% free VRAM"));
    wrap.append(row);
  }

  wrap.append(checkbox("Also drop cached node results",
    "Reclaims the most memory and forces the whole graph to run again next time.",
    cfg.purge_execution_cache, (v) => push({ purge_execution_cache: v })));

  if (cfg.purge_execution_cache) {
    const row = el("div", "rr-controls");
    row.append(el("span", null, "Drop the cache every"));
    const n = el("input");
    n.type = "number";
    n.min = 1; n.max = 100;
    n.value = cfg.purge_every_n_runs;
    n.onchange = () => push({ purge_every_n_runs: parseInt(n.value, 10) });
    row.append(n, el("span", null, "runs"));
    wrap.append(row);
  }

  wrap.append(checkbox("Let this panel close other programs",
    "Adds a close button to suggestions. Sends a normal quit request, never a force kill.",
    cfg.allow_terminate, (v) => push({ allow_terminate: v })));

  wrap.append(checkbox("Log each pass to the console", null,
    cfg.verbose, (v) => push({ verbose: v })));
}

/* ------------------------------------------------------------------ polling */

async function poll(root) {
  try {
    state = await call("/reaper/status");
  } catch (err) {
    root.replaceChildren();
    const wrap = el("div", "rr");
    wrap.append(el("div", "rr-note error",
      `Cannot reach Resource Reaper on the server: ${err.message}`));
    root.append(wrap);
    return;
  }
  render(root);
}

function start(root) {
  injectStyle();
  poll(root);
  stop();
  timer = setInterval(() => { if (!busy) poll(root); }, POLL_MS);
}

function stop() {
  if (timer) clearInterval(timer);
  timer = null;
}

/* ------------------------------------------------------------------ mount */

app.registerExtension({
  name: "ResourceReaper.Panel",
  async setup() {
    injectStyle();

    if (app.extensionManager?.registerSidebarTab) {
      app.extensionManager.registerSidebarTab({
        id: "resource-reaper",
        icon: "pi pi-chart-bar",
        title: "Memory",
        tooltip: "Memory and what to close",
        type: "custom",
        render: (el) => start(el),
        destroy: () => stop(),
      });
      return;
    }

    // Older frontends: a floating panel opened from a button.
    const panel = document.createElement("div");
    Object.assign(panel.style, {
      position: "fixed", top: "60px", right: "16px", width: "330px",
      maxHeight: "78vh", overflowY: "auto", zIndex: 1200, display: "none",
      background: "var(--comfy-menu-bg, #202020)", borderRadius: "6px",
      border: "1px solid var(--border-color, #3a3a3a)",
      boxShadow: "0 8px 28px rgba(0,0,0,.45)",
    });
    document.body.append(panel);

    const button = document.createElement("button");
    button.textContent = "Memory";
    button.onclick = () => {
      const open = panel.style.display === "none";
      panel.style.display = open ? "block" : "none";
      if (open) start(panel); else stop();
    };
    (document.querySelector(".comfy-menu") || document.body).append(button);
  },
});
