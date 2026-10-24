// Adds an "Add to Lora Vault" panel on Civitai model pages and a ✔ badge on model cards
// you already have. No accounts, licences or subscriptions - it only calls your own machine.
(() => {
  const ext = globalThis.browser ?? globalThis.chrome;
  // "Receiving end does not exist" = the helper in the background is asleep or was just reloaded.
  // Retry a few times (messaging wakes it); if it never answers this page is a leftover from before
  // the extension was (re)installed and needs a refresh.
  const send = async (path, method = "GET", body) => {
    let last;
    for (let i = 0; i < 4; i++) {
      try { return await ext.runtime.sendMessage({ type: "vault", path, method, body }); }
      catch (e) { last = e; await new Promise((r) => setTimeout(r, 600 * (i + 1))); }
    }
    const msg = String((last && last.message) || last);
    return { ok: false, offline: true, error: /receiving end|context invalidated/i.test(msg)
      ? "The extension was reloaded - refresh this page (Ctrl+F5). If it keeps happening, open about:debugging, find Lora Vault and check its background page for errors."
      : msg };
  };

  // ---------- floating panel (shadow DOM so Civitai's CSS can't touch it)
  const host = document.createElement("div");
  host.style.cssText = "position:fixed;right:16px;bottom:16px;z-index:2147483647;display:none";
  const root = host.attachShadow({ mode: "open" });
  root.innerHTML = `
    <style>
      .box{font:13px/1.35 system-ui,sans-serif;background:#1f232b;color:#e6e8ee;border:1px solid #3a4150;border-radius:12px;padding:10px 12px;width:230px;box-shadow:0 6px 24px #0008}
      .t{font-weight:700;margin-bottom:6px;display:flex;justify-content:space-between;align-items:center}
      .x{cursor:pointer;opacity:.5;border:0;background:none;color:inherit;font-size:16px}
      button.b{width:100%;margin-top:6px;padding:7px 8px;border-radius:8px;border:1px solid #4c8dff;background:#4c8dff;color:#fff;font:inherit;cursor:pointer}
      button.b.alt{background:transparent;color:#e6e8ee;border-color:#4a5263}
      button.b:disabled{opacity:.6;cursor:wait}
      .s{font-size:12px;opacity:.85}.ok{color:#3fb97a}.warn{color:#e0a82e}.bad{color:#ff7a6b}
    </style>
    <div class="box"><div class="t"><span>💾 Lora Vault</span><button class="x" title="Hide">×</button></div>
      <div class="s" id="state">…</div><div id="btns"></div></div>`;
  const $ = (s) => root.querySelector(s);
  $(".x").onclick = () => { host.style.display = "none"; dismissed = location.pathname; };
  let dismissed = null;

  const ids = () => {
    const m = location.pathname.match(/^\/models\/(\d+)/);
    if (!m) return null;
    const v = new URLSearchParams(location.search).get("modelVersionId");
    return { modelId: +m[1], versionId: v && /^\d+$/.test(v) ? +v : null };
  };

  let current = null;
  function setState(html, cls) { const s = $("#state"); s.className = "s " + (cls || ""); s.textContent = html; }
  function setButtons(list) {
    const box = $("#btns"); box.replaceChildren();
    for (const [label, alt, fn] of list) {
      const b = document.createElement("button");
      b.className = "b" + (alt ? " alt" : ""); b.textContent = label;
      b.onclick = async () => { b.disabled = true; try { await fn(); } finally { b.disabled = false; } };
      box.append(b);
    }
  }

  async function refreshPanel() {
    const id = ids();
    if (!id || dismissed === location.pathname) { host.style.display = "none"; return; }
    current = id;
    host.style.display = "block";
    const q = id.versionId ? `version_ids=${id.versionId}&model_ids=${id.modelId}` : `model_ids=${id.modelId}`;
    const r = await send("/status?" + q);
    if (r.offline || r.error) {
      setState("Vault server not reachable. Start ComfyUI (with Lora Vault installed).", "warn");
      setButtons([["Retry", true, refreshPanel]]);
      return;
    }
    const v = id.versionId && r.versions && r.versions[id.versionId];
    const m = r.models && r.models[id.modelId];
    const saved = v || (!id.versionId && m);
    const status = v ? v.status : m ? m.status : "ok";
    if (saved) {
      const bad = status !== "ok";
      setState(bad ? `In your vault - but flagged "${status.replace("_", " ")}" on Civitai.` :
        v ? `✔ In your vault${v.pinned ? " (pinned)" : v.local ? " (downloaded)" : " (downloads when used)"}` : `✔ ${m.versions.length} version(s) of this model in your vault`, bad ? "warn" : "ok");
      const btns = [];
      if (id.versionId && !v) btns.push(["Add this version", false, () => save(false)]);
      if (v && !v.pinned) btns.push(["📌 Pin (download & keep)", true, () => save(true)]);
      if (!id.versionId && m && !m.pinned) btns.push(["📌 Pin latest", true, () => save(true)]);
      setButtons(btns);
    } else {
      setState(id.versionId ? "Not in your vault yet." : "Not in your vault yet. Adds the newest version unless you pick one.");
      setButtons([["Add to Lora Vault", false, () => save(false)], ["Add + pin (download now)", true, () => save(true)]]);
    }
  }

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  async function save(pin) {
    const id = current;
    setState("Saving… (fetching details and previews through Firefox)");
    // The save runs on the vault server in the background; we poll, so no message is held open.
    const r = await send("/save", "POST", { model_id: id.modelId, version_id: id.versionId, pin, async: true });
    if (!r.ok) { setState(r.error || "Failed", "bad"); return; }
    let failures = 0;
    for (let n = 1; ; n++) {
      await sleep(1000);
      const j = await send("/save_job/" + r.job);
      if (j.offline) { if (++failures > 20) { setState(j.error || "Lost contact with the vault.", "bad"); return; } continue; }
      failures = 0;
      if (j.ok === false) { setState(j.error || "Failed", "bad"); return; }
      if (j.state === "error") { setState(j.error || "Failed", "bad"); return; }
      if (j.state === "done") break;
      setState(`Saving… ${n}s (fetching details and previews through Firefox)`);
    }
    await refreshPanel();
    if (pin) setState("✔ Saved. Downloading in the background (watch the vault page).", "ok");
  }

  // ---------- ✔ badges on cards in listings
  const badged = new WeakSet();
  async function badgeCards() {
    const links = [...document.querySelectorAll('a[href*="/models/"]')].filter((a) => !badged.has(a));
    const byModel = new Map();
    for (const a of links) {
      const m = a.getAttribute("href").match(/\/models\/(\d+)/);
      if (!m) continue;
      if (!byModel.has(+m[1])) byModel.set(+m[1], []);
      byModel.get(+m[1]).push(a);
      badged.add(a);
    }
    if (!byModel.size) return;
    const ids = [...byModel.keys()].slice(0, 200);
    const r = await send("/status?model_ids=" + ids.join(","));
    if (!r.models) return;
    for (const [mid, info] of Object.entries(r.models)) {
      for (const a of byModel.get(+mid) || []) {
        if (!a.querySelector("img, picture, video") || a.querySelector(".lora-vault-badge")) continue;
        if (getComputedStyle(a).position === "static") a.style.position = "relative";
        const b = document.createElement("span");
        b.className = "lora-vault-badge";
        b.textContent = info.status !== "ok" ? "⚠ vault" : info.pinned ? "📌 vault" : "✔ vault";
        b.title = "In your Lora Vault";
        b.style.cssText = "position:absolute;top:6px;left:6px;z-index:5;background:" + (info.status !== "ok" ? "#b9770e" : "#1f7a4d") +
          ";color:#fff;font:600 11px system-ui;padding:2px 6px;border-radius:8px;pointer-events:none";
        a.append(b);
      }
    }
  }

  document.documentElement.append(host);
  let lastHref = "";
  setInterval(() => { if (location.href !== lastHref) { lastHref = location.href; dismissed = dismissed === location.pathname ? dismissed : null; refreshPanel(); } }, 600);
  let t = null;
  new MutationObserver(() => { clearTimeout(t); t = setTimeout(badgeCards, 1200); }).observe(document.body, { childList: true, subtree: true });
  setTimeout(badgeCards, 1500);
})();
