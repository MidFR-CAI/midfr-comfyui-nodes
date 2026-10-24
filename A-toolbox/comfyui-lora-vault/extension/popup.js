const ext = globalThis.browser ?? globalThis.chrome;
const st = document.getElementById("st"), srv = document.getElementById("srv");
async function test() {
  st.textContent = "Checking…"; st.className = "";
  let r = { ok: false };
  for (let i = 0; i < 4 && !r.ok; i++) {   // the first message wakes the background page
    r = await ext.runtime.sendMessage({ type: "vault", path: "/ping" }).catch(() => ({ ok: false }));
    if (!r.ok) await new Promise((x) => setTimeout(x, 500));
  }
  if (r && r.ok) { st.textContent = `Connected to the vault - ${r.count} LoRA(s). This extension is relaying all Civitai traffic for ComfyUI while Firefox is open.`; st.className = "ok"; }
  else { st.textContent = "Can't reach the vault. Is ComfyUI running?"; st.className = "bad"; }
}
ext.storage.local.get("server").then((s) => { srv.value = s.server || "http://127.0.0.1:8199"; test(); });
document.getElementById("save").onclick = async () => { await ext.storage.local.set({ server: srv.value.trim() }); test(); };
