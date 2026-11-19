// Lora Vault buttons on Civitai: Add / Pin / Download under the Download button on a model page, and a
// pin icon on model cards that pins a model without opening it. No accounts, licences or subscriptions -
// it only calls your own machine. If ComfyUI isn't running the helper keeps what you add and sends it later.
(() => {
  const ext = globalThis.browser ?? globalThis.chrome;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  // "Receiving end does not exist" = the helper in the background is asleep or was just reloaded.
  // Retry a few times (messaging wakes it); if it never answers this page is a leftover from before
  // the extension was (re)installed and needs a refresh.
  const send = async (path, method = "GET", body) => {
    let last;
    for (let i = 0; i < 4; i++) {
      try { return await ext.runtime.sendMessage({ type: "vault", path, method, body }); }
      catch (e) { last = e; await sleep(600 * (i + 1)); }
    }
    return { ok: false, offline: true, error: String((last && last.message) || last) };
  };

  // Save in the background on the vault server and poll, so no message is held open while previews are fetched.
  // Resolves {ok, queued, error}. {queued:true} means ComfyUI isn't running; the helper will send it when it is.
  async function saveAndWait(body) {
    const r = await send("/save", "POST", { ...body, async: true });
    if (r.queued) return { ok: true, queued: true };
    if (!r.ok) return { ok: false, error: r.error || "Failed" };
    for (let failures = 0; ; ) {
      await sleep(1000);
      const j = await send("/save_job/" + r.job);
      if (j.offline) { if (++failures > 20) return { ok: false, error: "Lost contact with the vault." }; continue; }
      failures = 0;
      if (j.ok === false || j.state === "error") return { ok: false, error: j.error || "Failed" };
      if (j.state === "done") return { ok: true };
    }
  }

  const queuedModels = async () => new Set(((await ext.storage.local.get("queue")).queue || []).map((q) => q.model_id));

  const pinIcon = (filled) => `<svg viewBox="0 0 24 24" width="16" height="16" fill="${filled ? "currentColor" : "none"}" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 17v5"/><path d="M9 10.76a2 2 0 0 1-1.11 1.79l-1.78.9A2 2 0 0 0 5 15.24V16a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-.76a2 2 0 0 0-1.11-1.79l-1.78-.9A2 2 0 0 1 15 10.76V7a1 1 0 0 1 1-1 2 2 0 0 0 0-4H8a2 2 0 0 0 0 4 1 1 0 0 1 1 1z"/></svg>`;

  const diskIcon = `<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15.2 3a2 2 0 0 1 1.4.6l3.8 3.8a2 2 0 0 1 .6 1.4V19a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z"/><path d="M17 21v-7a1 1 0 0 0-1-1H8a1 1 0 0 0-1 1v7"/><path d="M7 3v4a1 1 0 0 0 1 1h7"/></svg>`;
  const plusIcon = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5v14M5 12h14"/></svg>`;

  // ---------- buttons on the model page (shadow DOM so Civitai's CSS can't touch them)
  const host = document.createElement("div");
  host.style.cssText = "display:block;margin:8px 0 0";
  const root = host.attachShadow({ mode: "open" });
  root.innerHTML = `
    <style>
      .row{display:grid;grid-template-columns:1fr 1fr;gap:8px}
      button{font:inherit;font-weight:600;display:flex;align-items:center;justify-content:center;gap:6px;height:36px;padding:0 8px;white-space:nowrap;border-radius:4px;cursor:pointer;
        border:1px solid rgba(34,139,230,.55);background:rgba(34,139,230,.18);color:#74c0fc}
      button:hover:not(:disabled){background:rgba(34,139,230,.28)}
      button:disabled{opacity:.7;cursor:default}
      button.green{border-color:rgba(47,158,68,.7);background:rgba(47,158,68,.25);color:#8ce99a}
      .s{font:12px/1.4 system-ui,sans-serif;color:#9aa3b2;margin-top:4px}.ok{color:#3fb97a}.warn{color:#e0a82e}.bad{color:#ff7a6b}
    </style>
    <div class="row" id="btns"></div><div class="s" id="state"></div>`;
  const $ = (s) => root.querySelector(s);

  const ids = () => {
    const m = location.pathname.match(/^\/models\/(\d+)/);
    if (!m) return null;
    const v = new URLSearchParams(location.search).get("modelVersionId");
    return { modelId: +m[1], versionId: v && /^\d+$/.test(v) ? +v : null };
  };

  function setState(text, cls) { const s = $("#state"); s.className = "s " + (cls || ""); s.textContent = text || ""; }
  function setButtons(list) {
    const box = $("#btns"); box.replaceChildren();
    for (const [icon, label, fn, disabled, green] of list) {
      const b = document.createElement("button");
      b.innerHTML = icon + `<span>${label}</span>`; b.disabled = !!disabled; b.classList.toggle("green", !!green);
      b.onclick = async () => { b.disabled = true; b.classList.add("green"); try { await fn(); } finally { b.disabled = false; } };
      box.append(b);
    }
  }

  // Sit right under the big Download button (inside its panel); fall back to just under the page title.
  function place() {
    const link = [...document.querySelectorAll('a[href*="/api/download/models/"], a, button')]
      .find((el) => /^\s*download\b/i.test(el.textContent) && el.textContent.length < 40 && el.offsetParent);
    if (link) {
      const up = link.parentElement && getComputedStyle(link.parentElement);
      (up && up.display === "flex" && up.flexDirection === "row" && link.parentElement.children.length > 1 ? link.parentElement : link).after(host);
    } else {
      const h1 = document.querySelector("h1");
      if (h1) h1.after(host);
    }
  }

  let current = null;
  async function refreshBar() {
    const id = ids();
    if (!id) { host.remove(); return; }
    current = id;
    if (!host.isConnected) place();
    const q = id.versionId ? `version_ids=${id.versionId}&model_ids=${id.modelId}` : `model_ids=${id.modelId}`;
    const r = await send("/status?" + q);   // offline just means "not known": the buttons still work and queue
    const v = id.versionId && r.versions && r.versions[id.versionId];
    const m = r.models && r.models[id.modelId];
    const saved = v || (!id.versionId && m);
    const pinned = v ? v.pinned : m && m.pinned;
    const local = v && v.local;
    const waiting = (await queuedModels()).has(id.modelId);
    const bad = saved && (v ? v.status : m.status) !== "ok";
    setState(waiting ? "Queued - added when ComfyUI starts." : bad ? "In your vault, but flagged on Civitai." : saved ? `✔ In your vault${pinned ? " (pinned)" : ""}` : "",
      waiting ? "warn" : bad ? "warn" : "ok");
    const base = { model_id: id.modelId, version_id: id.versionId };
    const run = (extra) => async () => {
      setState("Saving… (fetching details and previews through Firefox)");
      const res = await saveAndWait({ ...base, ...extra });
      if (!res.ok) setState(res.error, "bad"); else await refreshBar();
    };
    setButtons([
      [plusIcon, saved ? "In Vault" : "Add to Vault", run({}), saved],
      [pinIcon(pinned && local), pinned && local ? "Pinned" : "Pin + Download", run({ pin: true, download: true }), pinned && local, pinned && local],
    ]);
  }

  // ---------- pin icon on cards in listings
  const info = new Map();   // model id -> {pinned, status}; filled in by /status
  const paint = (btn, mid, queued) => {
    const i = info.get(mid) || {};
    btn.innerHTML = diskIcon;
    btn.title = queued ? "Queued - click to cancel" : i.pinned ? "Pinned in your Lora Vault (remove it on the vault page)" : i.saved ? "In your Lora Vault - click to remove" : "Add to Lora Vault";
    btn.style.background = i.saved ? "#1f7a4d" : "#4a4f5acc";   // grey -> green once it is in the vault
    btn.style.borderColor = i.saved && i.status && i.status !== "ok" ? "#e0a82e" : "transparent";
  };

  // Civitai puts a column of round icons (menu, create...) down the card's top-right. Sit just under the lowest one.
  function belowCardActions(btn, a) {
    const box = a.getBoundingClientRect();
    const icons = [...(a.parentElement || a).querySelectorAll("button, a, [role=button]")].filter((el) => {
      const r = el.getBoundingClientRect();
      return el !== btn && el.contains(btn) === false && r.width > 0 && r.width < 48 && r.height < 48 && r.top - box.top < 120 && r.left > box.left + box.width / 2 && r.top >= box.top - 2;
    });
    if (!icons.length) return;
    const lowest = icons.reduce((x, y) => (y.getBoundingClientRect().bottom > x.getBoundingClientRect().bottom ? y : x));
    const r = lowest.getBoundingClientRect();
    btn.style.top = Math.round(r.bottom - box.top + 6) + "px";
    btn.style.right = Math.round(box.right - r.right + (r.width - 24) / 2) + "px";   // centred under that icon
  }

  const repaintCards = (mid, queued, error) => {
    for (const b of document.querySelectorAll(".lora-vault-pin")) if (b.dataset.mid === String(mid)) { paint(b, mid, queued); if (error) b.title = error; }
  };

  // Undo a misclick: drop it from the extension's waiting list, and from the vault if it got that far.
  async function removeFromVault(mid, was) {
    const q = (await ext.storage.local.get("queue")).queue || [];
    await ext.storage.local.set({ queue: q.filter((x) => x.model_id !== mid) });
    if (!was.remote) return { ok: true };
    const r = await send("/remove_many", "POST", { model_ids: [mid], delete_files: true });
    return r.ok ? { ok: true } : { ok: false, error: r.offline ? "ComfyUI isn't running - can't remove it right now." : r.error || "Couldn't remove it." };
  }

  async function decorateCards() {
    const cards = new Map();
    for (const a of document.querySelectorAll('a[href*="/models/"]')) {
      const m = a.getAttribute("href").match(/\/models\/(\d+)/);
      if (!m || a.querySelector(".lora-vault-pin") || !a.querySelector("img, picture, video")) continue;
      if (!cards.has(+m[1])) cards.set(+m[1], []);
      cards.get(+m[1]).push(a);
    }
    if (!cards.size) return;
    const unknown = [...cards.keys()].filter((mid) => !info.has(mid)).slice(0, 200);
    if (unknown.length) {
      const r = await send("/status?model_ids=" + unknown.join(","));
      for (const mid of unknown) {
        const s = r.models && r.models[mid];
        info.set(mid, s ? { saved: true, remote: true, pinned: s.pinned, status: s.status } : {});
      }
    }
    const waiting = await queuedModels();
    for (const [mid, anchors] of cards) {
      for (const a of anchors) {
        if (a.querySelector(".lora-vault-pin")) continue;
        if (getComputedStyle(a).position === "static") a.style.position = "relative";
        const btn = document.createElement("button");
        btn.className = "lora-vault-pin";
        btn.style.cssText = "position:absolute;top:54px;right:8px;z-index:5;width:24px;height:24px;padding:0;display:flex;align-items:center;justify-content:center;" +
          "color:#fff;border:2px solid transparent;border-radius:50%;cursor:pointer;backdrop-filter:blur(4px)";
        paint(btn, mid, waiting.has(mid));
        belowCardActions(btn, a);
        setTimeout(() => belowCardActions(btn, a), 1500);   // the card's own icons may still be laying out
        btn.addEventListener("click", async (e) => {
          e.preventDefault(); e.stopPropagation(); e.stopImmediatePropagation();   // don't open the model
          if (btn.dataset.busy || (info.get(mid) || {}).pinned) return;   // pinned ones are removed on the vault page, not by a stray click
          btn.dataset.busy = "1";
          const before = info.get(mid) || {}, removing = !!before.saved;
          info.set(mid, removing ? {} : { saved: true, status: "ok" });   // flip straight away; undone below if it fails
          repaintCards(mid, false);
          const res = removing ? await removeFromVault(mid, before) : await saveAndWait({ model_id: mid });
          delete btn.dataset.busy;
          if (!res.ok) info.set(mid, before);
          else if (!removing && !res.queued) info.set(mid, { saved: true, remote: true, status: "ok" });
          repaintCards(mid, !removing && res.queued, res.ok ? "" : res.error);
        }, true);
        btn.dataset.mid = String(mid);
        a.append(btn);
      }
    }
  }

  // Civitai is a single-page app: keep the buttons in place as the page changes.
  let lastHref = "", timer = null;
  const tick = () => { clearTimeout(timer); timer = setTimeout(() => { decorateCards(); if (ids() && !host.isConnected) refreshBar(); }, 800); };
  setInterval(() => { if (location.href !== lastHref) { lastHref = location.href; info.clear(); refreshBar(); } }, 600);
  new MutationObserver(tick).observe(document.body, { childList: true, subtree: true });
  document.addEventListener("visibilitychange", () => { if (!document.hidden) { info.clear(); tick(); } });
  setTimeout(decorateCards, 1500);
})();
