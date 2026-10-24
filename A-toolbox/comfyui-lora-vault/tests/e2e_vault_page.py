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
    pg.wait_for_selector(".tile[title='Collection Pack']")
    assert pg.query_selector(".tile .pv")                                # preview-only 2:3 tile
    box = pg.query_selector(".tile").bounding_box(); assert abs(box["height"] / box["width"] - 1.5) < 0.02, box
    pg.hover(".tile"); assert "Collection Pack" in pg.inner_text(".tile .cap")   # name shows on hover
    pg.click(".tile[title='Collection Pack']")
    pg.wait_for_selector("#detail[open]")
    txt = pg.inner_text("#detail")
    assert "a cat" in txt and "Seed" in txt, txt                    # generation data
    assert "Model desc" in txt and "hello" in txt, txt               # both descriptions
    assert "alert(1)" not in pg.inner_html("#detail"), "script must be stripped"
    assert "Alpha" in txt and "Gamma" in txt, txt                    # unsaved sibling versions listed
    # save a sibling from the version list, then the mature one: previews must exist and be blurred
    # the other versions are visible WITHOUT saving them: look at Alpha first
    pg.click(".vrow:has-text('Alpha') .nm"); pg.wait_for_selector("text=not saved to your vault yet")
    assert "a cat" in pg.inner_text("#detail") and "alpha" in pg.inner_text("#detail")
    from vault import db as _db
    assert _db.get_version(331) is None
    pg.click(".vrow:has-text('Gamma') >> text=Save"); pg.wait_for_selector(".vrow:has-text('Gamma') >> text=Copy tag")
    pg.click(".vrow:has-text('Gamma') >> text=View")
    pg.wait_for_selector(".big.blur")
    assert "mature one" not in pg.inner_text(".stage")
    pg.click(".big"); pg.wait_for_selector(".big:not(.blur)")
    assert "mature one" in pg.inner_text("#detail") and "Euler a" in pg.inner_text("#detail")
    pg.screenshot(path="/tmp/lv_page_detail.png")
    # bulk save the remaining version
    pg.click("text=Save all"); pg.wait_for_selector(".vrow:has-text('Alpha') >> text=Copy tag", timeout=10000)
    pg.click("#dClose")
    # unblur toggle persists to server and un-blurs the library
    pg.uncheck("#blurMature"); time.sleep(0.5)
    assert config.get("blur_mature") is False
    pg.click(".tile[title='Collection Pack']"); pg.wait_for_selector("#detail[open]")
    pg.click(".vrow:has-text('Gamma') >> text=View"); pg.wait_for_selector(".big")
    assert pg.query_selector(".big.blur") is None
    # video: autoplay, muted, with an unmute control
    # (the fake data: URL is not playable, so check the element synchronously before its error handler swaps it out)
    ok = pg.evaluate("""()=>{detail.previews.unshift({type:'video',src:'data:video/mp4;base64,',full:'data:video/mp4;base64,',local:false,nsfwLevel:1,meta:{}});curPrev=0;renderDetail();
      const v=document.querySelector('video.big'); const r=!!v&&v.muted&&v.autoplay&&v.loop; const b=[...document.querySelectorAll('button')].find(x=>x.textContent.includes('Unmute')); b&&b.click(); return [r, v&&v.muted, !!b];}""")
    assert ok == [True, False, True], ok
    pg.click("#dClose")
    # --- select mode: click, shift-range, drag box, select all, then a real batch remove
    pg.on("dialog", lambda d: d.accept())
    pg.evaluate("clearInterval(timer)")   # no background refresh while we fake the item list
    pg.evaluate("""()=>{const base=items[0]; const keep=items.slice(); items=[]; for(let i=0;i<6;i++) items.push({...base, model_id:7000+i, version_id:7000+i, model_name:'M'+i, thumb:null, thumb_video:null}); window._keep=keep; render();}""")
    pg.click("#selBtn"); assert pg.evaluate("document.body.classList.contains('selecting')")
    tiles = pg.query_selector_all(".tile")
    tiles[1].click(); assert pg.inner_text("#selCount") == "1 selected"
    tiles[3].click(modifiers=["Shift"]); assert pg.inner_text("#selCount") == "3 selected", pg.inner_text("#selCount")
    pg.click("#selNone"); assert pg.inner_text("#selCount") == "0 selected"
    a, c = tiles[0].bounding_box(), tiles[1].bounding_box()
    pg.mouse.move(a["x"] + 3, a["y"] + 3); pg.mouse.down(); pg.mouse.move(c["x"] + c["width"] - 3, c["y"] + c["height"] - 3, steps=6); pg.mouse.up()
    assert pg.inner_text("#selCount") == "2 selected", pg.inner_text("#selCount")
    assert not pg.evaluate("document.querySelector('#detail').open"), "drag must not open a tile"
    pg.click("#selAll"); assert pg.inner_text("#selCount") == "6 selected"
    pg.click("#selDone"); assert not pg.evaluate("document.body.classList.contains('selecting')")
    # real removal of the seeded model through the UI (model 33 -> every saved version goes)
    pg.evaluate("()=>{items=window._keep; render();}")
    pg.click("#selBtn"); pg.click(".tile[title='Collection Pack']"); pg.click("#selDel")
    for _ in range(40):
        if not _db.list_versions(): break
        time.sleep(0.25)
    assert not _db.list_versions(), "batch remove should have emptied the vault"
    # many video covers must not all play at once (memory/CPU), and polling must not rebuild an unchanged grid
    n = pg.evaluate("""()=>{const base=items[0]; items=[]; for(let i=0;i<40;i++) items.push({...base, model_id:9000+i, version_id:9000+i, model_name:'V'+i, thumb_video:'data:video/mp4;base64,AAAA', thumb_nsfw:1}); render(); return items.length}""")
    time.sleep(0.8)
    live = pg.evaluate("document.querySelectorAll('.tile video').length")
    assert n == 40 and live <= 8, live
    pg.screenshot(path="/tmp/lv_page_lib.png")
    b.close()
print("VAULT PAGE OK; errors:", errors); assert not errors, errors
