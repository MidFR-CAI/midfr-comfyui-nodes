// Lora Vault helper. Two jobs:
//  1. Relay: ComfyUI's vault can't (and doesn't) talk to Civitai itself. It queues requests on
//     the local server; this script long-polls for them and runs them with *Firefox's* network
//     route and Civitai login, then posts the result back. Big LoRA files are saved with
//     Firefox's own download manager and the vault just picks them up.
//  2. Forward messages from Civitai pages / the popup to the local server.
const ext = globalThis.browser ?? globalThis.chrome;
const DEFAULT_SERVER = "http://127.0.0.1:8199";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function server() {
  const s = await ext.storage.local.get("server");
  return (s.server || DEFAULT_SERVER).replace(/\/+$/, "");
}

async function call({ path, method = "GET", body }) {
  const url = (await server()) + "/api" + path;
  const init = { method, headers: { "X-Lora-Vault": "1" } };
  if (body !== undefined) { init.headers["Content-Type"] = "application/json"; init.body = JSON.stringify(body); }
  const r = await fetch(url, init);
  let j = {};
  try { j = await r.json(); } catch (_) { /* non-JSON error page */ }
  if (!r.ok && j.ok === undefined) j = { ok: false, error: "HTTP " + r.status };
  return j;
}

// Saves made while ComfyUI isn't running are kept here and sent, oldest first, once the vault answers again.
async function queued() { return (await ext.storage.local.get("queue")).queue || []; }
async function enqueue(add) {
  const q = await queued();
  const same = q.find((x) => x.model_id === add.model_id && x.version_id === add.version_id);
  if (same) { same.pin = same.pin || add.pin; same.download = same.download || add.download; }
  else q.push(add);
  await ext.storage.local.set({ queue: q });
}
let flushing = false;
async function flushQueue() {
  if (flushing) return;
  flushing = true;
  try {
    for (let q = await queued(); q.length; q = await queued()) {
      const { model_id, version_id, pin, download } = q[0];
      const r = await call({ path: "/save", method: "POST", body: { model_id, version_id, pin, download } }).catch(() => null);
      if (!r) return; // vault went away again: keep the rest for next time
      if (!r.ok) { // Civitai refused this one (removed, restricted): drop it, but remember why
        const f = (await ext.storage.local.get("failed")).failed || [];
        f.push({ model_id, version_id, error: r.error || "failed" });
        await ext.storage.local.set({ failed: f.slice(-20) });
      }
      await ext.storage.local.set({ queue: (await queued()).slice(1) });
    }
  } finally { flushing = false; }
}

ext.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (!msg || msg.type !== "vault") return;
  startLoop();
  const offline = (e) => msg.method === "POST" && msg.path === "/save"
    ? enqueue({ model_id: msg.body.model_id, version_id: msg.body.version_id, pin: !!msg.body.pin, download: !!msg.body.download }).then(() => ({ ok: true, queued: true }))
    : { ok: false, offline: true, error: String(e) };
  call(msg).then(sendResponse, (e) => Promise.resolve(offline(e)).then(sendResponse));
  return true; // async response
});

// ------------------------------------------------------------------ relay loop
let looping = false;
async function startLoop() {
  if (looping) return;
  looping = true;
  let backoff = 1000;
  for (;;) {
    try {
      const r = await fetch((await server()) + "/api/relay/next?v=" + encodeURIComponent(ext.runtime.getManifest().version), { headers: { "X-Lora-Vault": "1" } });
      const j = await r.json();
      backoff = 1000;
      if (j.task) handle(j.task); // run concurrently; keep polling
      else flushQueue();          // the vault is up: send anything saved while it was away
    } catch (_) {
      await sleep(backoff);
      backoff = Math.min(backoff * 2, 15000);
    }
  }
}

function b64(u8) {
  let s = "";
  for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
  return btoa(s);
}

const post = (path, body) => call({ path, method: "POST", body }).catch(() => ({}));

// fetch with a deadline, so a stuck connection reports which URL hung instead of vanishing
async function timedFetch(url, init, ms = 90000) {
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), ms);
  try { return await fetch(url, { ...init, signal: ac.signal }); }
  catch (e) { if (ac.signal.aborted) throw new Error("timed out after " + ms / 1000 + "s: " + url); throw e; }
  finally { clearTimeout(timer); }
}

// resolve: ask Civitai for the download link WITHOUT following the redirect (redirect:"manual"), so no file
// body is ever streamed into Firefox. webRequest tells us where each hop pointed; we stop at the first
// address that is not Civitai itself (the signed CDN link) and hand it to the vault.
const redirects = new Map(); // requested url -> redirect target
function onRedirect(d) {
  if (redirects.size > 50) redirects.clear(); // never let this grow
  redirects.set(d.url, d.redirectUrl);
}
if (ext.webRequest && ext.webRequest.onBeforeRedirect) {
  ext.webRequest.onBeforeRedirect.addListener(onRedirect, { urls: ["https://*.civitai.com/*", "https://civitai.com/*", "https://civitai.red/*", "https://*.civitai.red/*"] });
}
const isCivitai = (u) => { try { return /(^|\.)civitai\.(com|red|green)$/i.test(new URL(u).hostname); } catch (_) { return true; } };
let resolveBusy = 0;

async function handleResolve(t) {
  if (resolveBusy >= 2) { await sleep(300); }
  resolveBusy++;
  let current = t.url, note = "Civitai gave no CDN redirect";
  try {
    for (let hop = 0; hop < 6; hop++) {
      redirects.delete(current);
      const ac = new AbortController();
      const timer = setTimeout(() => ac.abort(), 30000);
      let r;
      try { r = await fetch(current, { credentials: "include", redirect: "manual", signal: ac.signal }); }
      finally { clearTimeout(timer); }
      const to = redirects.get(current);
      redirects.delete(current);
      if (to) {
        if (!isCivitai(to)) return await post("/relay/result", { id: t.id, ok: true, resolved: to });
        current = to; continue;
      }
      // no redirect seen: a web page (login / early access) or the file itself - never read the body
      try { r.body && r.body.cancel(); } catch (_) {}
      note = /html/i.test(r.headers.get("content-type") || "") ? "Civitai answered with a web page (login / early access needed)"
                                                                  : "Civitai gave no CDN redirect (HTTP " + r.status + ")";
      break;
    }
    return await post("/relay/result", { id: t.id, ok: true, resolved: null, note });
  } finally { resolveBusy--; }
}

async function handle(t) {
  try {
    if (t.kind === "resolve") {
      await handleResolve(t);
    } else if (t.kind === "json") {
      const r = await timedFetch(t.url, { headers: t.headers || {}, credentials: "include" });
      await post("/relay/result", { id: t.id, ok: true, status: r.status, body: await r.text() });
    } else if (t.kind === "bytes") {
      const r = await timedFetch(t.url, { headers: t.headers || {}, credentials: "include" });
      const buf = new Uint8Array(await r.arrayBuffer());
      await post("/relay/result", { id: t.id, ok: true, status: r.status, body_b64: b64(buf) });
    } else if (t.kind === "download") {
      await handleDownload(t);
    } else {
      await post("/relay/result", { id: t.id, ok: false, error: "unknown task kind " + t.kind });
    }
  } catch (e) {
    await post("/relay/result", { id: t.id, ok: false, error: "Firefox couldn't fetch it: " + String(e && e.message || e) });
  }
}

async function handleDownload(t) {
  const fail = (error) => post("/relay/result", { id: t.id, ok: false, error });
  let id;
  try {
    id = await ext.downloads.download({ url: t.url, filename: t.filename, conflictAction: "uniquify", saveAs: false });
  } catch (e) { return fail("Firefox refused the download: " + String(e && e.message || e)); }
  let lastBytes = -1;
  for (;;) {
    const [item] = await ext.downloads.search({ id });
    if (!item) return fail("The download disappeared from Firefox's list.");
    const total = item.totalBytes > 0 ? item.totalBytes : item.fileSize || 0;
    if (item.bytesReceived !== lastBytes) {
      lastBytes = item.bytesReceived;
      post("/relay/progress", { id: t.id, received: item.bytesReceived, total });
    }
    if (item.state === "complete") {
      if (/html/i.test(item.mime || "")) {
        ext.downloads.erase({ id }).catch(() => {});
        return fail("Civitai sent a web page instead of the file - log in to civitai.com in Firefox (or add an API key in the vault settings).");
      }
      await post("/relay/result", { id: t.id, ok: true, path: item.filename });
      ext.downloads.erase({ id }).catch(() => {}); // history entry only; the vault moves the file
      return;
    }
    if (item.state === "interrupted") return fail("Download interrupted: " + (item.error || "unknown error"));
    await sleep(500);
  }
}

// keep the loop alive across browser-suspended background pages
if (ext.alarms) {
  ext.alarms.create("lora-vault-keepalive", { periodInMinutes: 0.5 });
  ext.alarms.onAlarm.addListener(() => startLoop());
}
ext.runtime.onStartup && ext.runtime.onStartup.addListener(startLoop);
ext.runtime.onInstalled && ext.runtime.onInstalled.addListener(startLoop);
startLoop();
