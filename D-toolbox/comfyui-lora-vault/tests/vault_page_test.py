"""Headless-browser test of the /lora_vault manager page (detail view, versions, blur)."""
import asyncio, os, sys, threading, time, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]
os.environ["LORA_VAULT_DATA_DIR"] = "/tmp/lv_pg/data"; os.environ["LORA_VAULT_LORA_DIR"] = "/tmp/lv_pg/loras"
shutil.rmtree("/tmp/lv_pg", ignore_errors=True)
from aiohttp import web
from fake_civitai import Fake
from vault import api, config
fake = Fake().start()
config.update({"civitai_hosts": [f"http://127.0.0.1:{fake.api_port}"], "civitai_api_key": "KEY123", "transport": "direct"})
app = web.Application()
for m, p, h in api.table("/lora_vault/api"): app.router.add_route(m, p, h)
app.router.add_get("/lora_vault", lambda r: web.FileResponse(os.path.join(ROOT, "ui", "index.html")))
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
    b = pw.chromium.launch(); pg = b.new_page(viewport={"width": 1300, "height": 900})
    pg.on("pageerror", lambda e: (errors.append(str(e)), print("PAGEERR", e)))
    pg.goto(f"http://127.0.0.1:{port['p']}/lora_vault")
    pg.fill("#url", "https://civitai.com/models/33?modelVersionId=332"); pg.click("#addBtn")
    pg.wait_for_selector(".card:has-text('Collection Pack')")
    pg.click(".card:has-text('Collection Pack') .t")
    pg.wait_for_selector("#detail[open]")
    txt = pg.inner_text("#detail")
    assert "a cat" in txt and "Seed" in txt, txt                    # generation data
    assert "Model desc" in txt and "hello" in txt, txt               # both descriptions
    assert "alert(1)" not in pg.inner_html("#detail"), "script must be stripped"
    assert "Alpha" in txt and "Gamma" in txt, txt                    # unsaved sibling versions listed
    # save a sibling from the version list, then the mature one: previews must exist and be blurred
    pg.click(".vrow:has-text('Gamma') >> text=Save"); pg.wait_for_selector(".vrow:has-text('Gamma') >> text=View")
    pg.click(".vrow:has-text('Gamma') >> text=View")
    pg.wait_for_selector(".big.blur")
    assert "mature one" not in pg.inner_text(".stage")
    pg.click(".big"); pg.wait_for_selector(".big:not(.blur)")
    assert "mature one" in pg.inner_text("#detail") and "Euler a" in pg.inner_text("#detail")
    pg.screenshot(path="/tmp/lv_page_detail.png")
    # bulk save the remaining version
    pg.click("text=Save all"); pg.wait_for_selector(".vrow:has-text('Alpha') >> text=View", timeout=10000)
    pg.click("#dClose")
    # unblur toggle persists to server and un-blurs the library
    pg.uncheck("#blurMature"); time.sleep(0.5)
    assert config.get("blur_mature") is False
    pg.click(".card:has-text('Collection Pack') .t"); pg.wait_for_selector("#detail[open]")
    pg.click(".vrow:has-text('Gamma') >> text=View"); pg.wait_for_selector(".big")
    assert pg.query_selector(".big.blur") is None
    # video: autoplay, muted, with an unmute control
    pg.evaluate("""()=>{detail.previews.unshift({type:'video',src:'data:video/mp4;base64,',full:'data:video/mp4;base64,',local:false,nsfwLevel:1,meta:{}});curPrev=0;renderDetail();}""")
    v = pg.query_selector("video.big"); assert v and pg.evaluate("v=>v.muted&&v.autoplay&&v.loop", v)
    pg.click("text=Unmute"); assert pg.evaluate("v=>v.muted", v) is False
    pg.click("#dClose"); pg.screenshot(path="/tmp/lv_page_lib.png")
    b.close()
print("VAULT PAGE OK; errors:", errors); assert not errors, errors
