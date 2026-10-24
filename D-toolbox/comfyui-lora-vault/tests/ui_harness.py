"""Headless-browser test of the node panel against the real API + fake Civitai."""
import asyncio, os, sys, threading, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]
os.environ["LORA_VAULT_DATA_DIR"] = "/tmp/lv_ui/data"; os.environ["LORA_VAULT_LORA_DIR"] = "/tmp/lv_ui/loras"
import shutil; shutil.rmtree("/tmp/lv_ui", ignore_errors=True)
from aiohttp import web
from fake_civitai import Fake
from vault import api, config, storage
fake = Fake().start()
config.update({"civitai_hosts": [f"http://127.0.0.1:{fake.api_port}"], "civitai_api_key": "KEY123", "transport": "direct"})
app = web.Application()
for m, p, h in api.table("/lora_vault/api"): app.router.add_route(m, p, h)
H = os.path.join(HERE, "harness")
app.router.add_get("/", lambda r: web.FileResponse(os.path.join(H, "index.html")))
app.router.add_get("/scripts/app.js", lambda r: web.FileResponse(os.path.join(H, "app.js")))
app.router.add_get("/scripts/api.js", lambda r: web.FileResponse(os.path.join(H, "api.js")))
app.router.add_get("/extensions/lora-vault/lora_vault.js", lambda r: web.FileResponse(os.path.join(ROOT, "web", "lora_vault.js")))
app.router.add_get("/img/{p:.*}", lambda r: web.Response(body=b"x"))
ready = threading.Event(); port = {}
def serve():
    loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
    async def go():
        r = web.AppRunner(app); await r.setup(); s = web.TCPSite(r, "127.0.0.1", 0); await s.start()
        port["p"] = s._server.sockets[0].getsockname()[1]; ready.set()
    loop.run_until_complete(go()); loop.run_forever()
threading.Thread(target=serve, daemon=True).start(); ready.wait(5)

from playwright.sync_api import sync_playwright
errors = []
with sync_playwright() as pw:
    b = pw.chromium.launch(); pg = b.new_page(viewport={"width": 560, "height": 760})
    pg.on("pageerror", lambda e: errors.append(str(e))); pg.on("console", lambda m: m.type == "error" and errors.append(m.text))
    pg.goto(f"http://127.0.0.1:{port['p']}/"); pg.wait_for_function("window.__ready === true")
    # base-model dropdown is populated from /enums
    pg.wait_for_function("document.querySelectorAll('.lv select option').length > 1")
    opts = pg.eval_on_selector_all(".lv-bar select option", "o=>o.map(x=>x.textContent)")
    assert "Pony" in opts, opts
    pg.select_option(".lv-bar select", "Pony")
    pg.fill(".lv-bar input", "pony"); pg.click("text=Search")
    pg.wait_for_selector(".lv-card"); assert "Pony Thing" in pg.inner_text(".lv-body")
    assert "Cool Style" not in pg.inner_text(".lv-body"), "base filter must exclude SDXL LoRA"
    pg.screenshot(path="/tmp/lv_search.png")
    pg.click(".lv-card button.pri")                                   # Add
    pg.wait_for_function("window.__node.widgets[0].value.includes('<lora:Pony Thing - v2:1>')")
    pg.wait_for_selector("text=In node >> nth=0")
    pg.click(".lv-bar >> text=In node")
    pg.wait_for_selector(".lv-card .lv-title:text('Pony Thing')")
    assert "Downloads on run" in pg.inner_text(".lv-body")           # saved, NOT downloaded
    assert not os.path.exists("/tmp/lv_ui/loras/_vault/cache") or not os.listdir("/tmp/lv_ui/loras/_vault/cache")
    # strength edit rewrites the tag
    pg.fill(".lv-card input.num >> nth=0", "0.7"); pg.dispatch_event(".lv-card input.num >> nth=0", "change")
    pg.wait_for_function("window.__node.widgets[0].value.includes('<lora:Pony Thing - v2:0.7:1>') || window.__node.widgets[0].value.includes('<lora:Pony Thing - v2:0.7>')")
    # details
    pg.click("text=Details"); pg.wait_for_selector(".lv-detail")
    d = pg.inner_text(".lv-detail"); assert "Model desc" in d and "pt" in pg.inner_text(".lv-card"), d
    pg.screenshot(path="/tmp/lv_detail.png")
    # collection: add two versions of one model page from the search view, stays in search
    pg.select_option(".lv-bar select", "Pony"); pg.fill(".lv-bar input", "collection"); pg.click(".lv-bar >> text=Search")
    pg.wait_for_selector(".lv-card .lv-title:text('Collection Pack')")
    pg.click(".lv-card:has-text('Collection Pack') .lv-row:has-text('Beta') button")
    pg.wait_for_function("window.__node.widgets[0].value.includes('<lora:Collection Pack - Beta:')")
    pg.click(".lv-card:has-text('Collection Pack') .lv-row:has-text('Gamma') button")
    pg.wait_for_function("window.__node.widgets[0].value.includes('<lora:Collection Pack - Gamma (mature):')")
    assert "In node" in pg.inner_text(".lv-card:has-text('Collection Pack') .lv-row:has-text('Beta')")
    # versions list inside the node card: "Add also" the third
    pg.click(".lv-bar >> text=In node"); pg.wait_for_selector(".lv-card:has-text('Beta')")
    pg.click(".lv-card:has-text('Beta') >> text=Details"); pg.wait_for_selector("text=Add also")
    pg.click(".lv-row:has-text('Alpha') >> text=Look"); pg.wait_for_selector(".lv-tw, .lv-chip.tw:has-text('alpha')")
    pg.click("text=Add also >> nth=0")
    pg.wait_for_function("window.__node.widgets[0].value.includes('<lora:Collection Pack - Alpha:')")
    # search lists your own library first (pinned, then saved) and only then Civitai
    pg.fill(".lv-bar input", "pack"); pg.click(".lv-bar >> text=Search")
    pg.wait_for_selector("text=Saved (")
    t = pg.inner_text(".lv-body")
    assert t.index("Saved (") < t.index("Collection Pack") and "From Civitai" not in t.split("Saved (")[1].split("Collection Pack")[0], t
    pg.fill(".lv-bar input", "pony"); pg.click(".lv-bar >> text=Search"); pg.wait_for_selector("text=Saved (")
    assert "Pony Thing" in pg.inner_text(".lv-body")
    # multi-word queries match by word, not as one exact phrase
    pg.fill(".lv-bar input", "thing pony"); pg.click(".lv-bar >> text=Search"); pg.wait_for_selector("text=Saved (")
    # blur setting is shared with the server
    pg.uncheck(".lv-bar input[type=checkbox]")
    pg.wait_for_function("true"); time.sleep(0.5)
    assert config.get("blur_mature") is False
    pg.check(".lv-bar input[type=checkbox]"); time.sleep(0.5)
    # base mismatch warning
    pg.select_option(".lv-bar select", "SDXL 1.0"); pg.click("text=In node")
    pg.wait_for_selector("text=different base")
    # hand-typed text with an unknown LoRA shows an error card
    pg.evaluate("(()=>{const w=window.__node.widgets[0]; w.value += '\\n<lora:v999:1>'; })()")
    pg.wait_for_selector("text=not in your vault", timeout=4000)
    # pin -> real download + progress -> pinned
    pg.click("text=Pin >> nth=0"); pg.wait_for_selector("text=Pinned", timeout=8000)
    assert os.listdir("/tmp/lv_ui/loras/_vault/pinned")
    pg.select_option(".lv-bar select", ""); pg.fill(".lv-bar input", ""); pg.click(".lv-bar >> text=Search"); pg.wait_for_selector("text=Pinned (")
    t = pg.inner_text(".lv-body"); assert t.index("Pinned (") < t.index("Saved ("), t
    pg.click("text=Saved"); pg.wait_for_selector(".lv-card"); pg.screenshot(path="/tmp/lv_saved.png")
    b.close()
print("UI OK; page errors:", errors)
assert not errors, errors
