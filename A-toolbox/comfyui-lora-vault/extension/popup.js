const ext = globalThis.browser ?? globalThis.chrome;
const st = document.getElementById("st"), srv = document.getElementById("srv");
async function test() {
  st.textContent = "Checking…"; st.className = "";
  let r = { ok: false };
  for (let i = 0; i < 4 && !r.ok; i++) {   // the first message wakes the background page
    r = await ext.runtime.sendMessage({ type: "vault", path: "/ping" }).catch(() => ({ ok: false }));
    if (!r.ok) await new Promise((x) => setTimeout(x, 500));
  }
  const { queue = [], failed = [] } = await ext.storage.local.get(["queue", "failed"]);
  const waiting = queue.length ? ` ${queue.length} save(s) waiting.` : "", skipped = failed.length ? ` ${failed.length} couldn't be saved (removed on Civitai?).` : "";
  if (r && r.ok) { st.textContent = `Connected to the vault - ${r.count} LoRA(s).${waiting}${skipped}`; st.className = "ok"; }
  else { st.textContent = `Vault offline. Anything you add on Civitai is kept and sent when ComfyUI starts.${waiting}${skipped}`; st.className = ""; }
}
ext.storage.local.get("server").then((s) => { srv.value = s.server || "http://127.0.0.1:8199"; test(); });
document.getElementById("save").onclick = async () => { await ext.storage.local.set({ server: srv.value.trim() }); test(); };
