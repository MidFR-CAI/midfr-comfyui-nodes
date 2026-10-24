"""Loads the real extension into Chromium, visits a (stubbed) civitai.com model page, clicks
'Add to Lora Vault', and checks the vault DB. Also checks web pages can't call the server."""
import os, sys, shutil, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]
os.environ["LORA_VAULT_DATA_DIR"] = "/tmp/lv_ext/data"; os.environ["LORA_VAULT_LORA_DIR"] = "/tmp/lv_ext/loras"
shutil.rmtree("/tmp/lv_ext", ignore_errors=True)
from fake_civitai import Fake
from vault import config, db, server
fake = Fake().start()
config.update({"civitai_hosts": [f"http://127.0.0.1:{fake.api_port}"], "civitai_api_key": "KEY123", "transport": "firefox", "cache_limit_gb": 0})
server.start_in_thread(8199); time.sleep(1)

from playwright.sync_api import sync_playwright
# Firefox gets a MV2 (persistent background) manifest; current Chromium only loads MV3, so test a copy
import shutil, tempfile
EXT = os.path.join(tempfile.mkdtemp(prefix="lv_ext_"), "extension")
shutil.copytree(os.path.join(ROOT, "extension"), EXT, ignore=shutil.ignore_patterns("*.xpi", "*.bat", "signed"))
if os.environ.get("LV_EXT_MV2") != "1":
    shutil.copyfile(os.path.join(HERE, "manifest_mv3_chromium.json"), os.path.join(EXT, "manifest.json"))
PAGE = """<html><body><h1>Pony Thing</h1>
<a href="/models/22/pony"><img src="https://civitai.com/x.png" width=100 height=100></a>
<a href="/models/11/other"><img src="https://civitai.com/y.png" width=100 height=100></a></body></html>"""
with sync_playwright() as pw:
    ctx = pw.chromium.launch_persistent_context("/tmp/lv_ext/profile", headless=False, executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome", args=["--no-sandbox", "--disable-background-networking", "--disable-sync", "--no-first-run", "--disable-component-update", "--disable-default-apps", "--disable-features=OptimizationHints,Translate" + (",ExtensionManifestV2Unsupported,ExtensionManifestV2Disabled" if os.environ.get("LV_EXT_MV2") == "1" else "")] +
        [f"--disable-extensions-except={EXT}", f"--load-extension={EXT}", "--headless=new"])
    ctx.route("https://civitai.com/**", lambda r: r.fulfill(status=200, content_type="text/html", body=PAGE) if r.request.resource_type == "document" else r.fulfill(status=200, body=b""))
    pg = ctx.new_page(); errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto("https://civitai.com/models/22/pony-thing")
    pg.wait_for_function("document.documentElement.lastElementChild && [...document.querySelectorAll('div')].some(d=>d.shadowRoot)", timeout=8000)
    sh = pg.evaluate_handle("[...document.querySelectorAll('div')].find(d=>d.shadowRoot).shadowRoot")
    txt = lambda: sh.evaluate("s=>s.querySelector('.box').innerText")
    pg.wait_for_function("s=>s.querySelector('#btns button')", arg=sh, timeout=8000)
    assert "Not in your vault" in txt(), txt()
    sh.evaluate("s=>s.querySelector('#btns button').click()")
    try:
        pg.wait_for_function("s=>s.querySelector('#state').textContent.toLowerCase().includes('in your vault')", arg=sh, timeout=8000)
    except Exception:
        print("PANEL STATE AFTER CLICK:", txt().replace("\n", " | ")); raise
    rec = db.get_version(222); assert rec and rec["model_name"] == "Pony Thing" and not rec["local_path"], rec
    print("saved via extension:", rec["model_name"], "| panel:", txt().replace("\n", " | "))
    print("STAGE badge", flush=True)
    # listing badge on the saved model's card only
    pg.wait_for_function("document.querySelectorAll('.lora-vault-badge').length>=1", timeout=8000)
    assert pg.evaluate("document.querySelectorAll('.lora-vault-badge').length") == 1
    print("STAGE origin-guard", flush=True)
    # a normal web page (civitai.com origin) must NOT be able to call the vault server directly
    import urllib.request, urllib.error
    def hit(origin, host="127.0.0.1:8199", method="GET", headers=None):
        h = {"Origin": origin, "Host": host, **(headers or {})}
        try:
            return urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8199/api/ping", method=method, headers=h), timeout=5).status
        except urllib.error.HTTPError as e:
            return e.code
    assert hit("https://civitai.com") == 403 and hit("https://evil.example") == 403, "web pages must be rejected"
    assert hit("chrome-extension://abc") == 200
    assert hit("http://127.0.0.1:8188") == 200
    assert hit("chrome-extension://abc", host="evil.example:8199") == 403, "DNS-rebinding guard"
    req = urllib.request.Request("http://127.0.0.1:8199/api/pin", b"{}", {"Content-Type": "text/plain"}, method="POST")
    try: urllib.request.urlopen(req, timeout=5); raise AssertionError("write without header must fail")
    except urllib.error.HTTPError as e: assert e.code == 403
    print("origin guard OK", flush=True)
    print("STAGE pin", flush=True)
    # pin from the panel -> downloads
    pg.goto("https://civitai.com/models/22?modelVersionId=222")
    pg.wait_for_function("s=>[...s.querySelectorAll('#btns button')].some(b=>b.textContent.includes('Pin'))", arg=pg.evaluate_handle("[...document.querySelectorAll('div')].find(d=>d.shadowRoot).shadowRoot"), timeout=8000)
    pg.evaluate("[...[...document.querySelectorAll('div')].find(d=>d.shadowRoot).shadowRoot.querySelectorAll('#btns button')].find(b=>b.textContent.includes('Pin')).click()")
    print("STAGE pin-wait", flush=True)
    for _ in range(40):
        if (db.get_version(222) or {}).get("pinned") and db.get_version(222)["local_path"]: break
        time.sleep(0.25)
    assert db.get_version(222)["pinned"] and os.path.isfile(db.get_version(222)["local_path"])
    print("pinned + downloaded via extension OK; page errors:", errs)
    # backend-initiated, relayed through the real extension: API + search + a file download
    from vault import civitai, relay, snapshot, storage
    assert relay.connected()
    rec = snapshot.save_version(version_id=111)
    assert rec["model_name"] == "Cool Style" and rec["snapshot"]["previews"][0]["local"]
    assert [m["name"] for m in civitai.search_models(query="cool", base_model="SDXL 1.0")["items"]] == ["Cool Style"]
    p = storage.ensure_local(111)
    from fake_civitai import PAYLOAD
    assert open(p, "rb").read() == PAYLOAD
    print("relay through real extension OK ->", p)
    ctx.close()
