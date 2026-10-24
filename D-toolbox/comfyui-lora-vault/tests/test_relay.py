"""Firefox-relay mode: the vault must do ALL Civitai traffic through the helper (extension).
A Python thread plays the extension; the server's direct network path is booby-trapped."""
import base64, json, os, socket, sys, threading, time, urllib.error, urllib.request

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from fake_civitai import Fake, PAYLOAD  # noqa: E402


class Boom:
    def open(self, *a, **k):
        raise AssertionError("ComfyUI's Python must not talk to Civitai directly in firefox mode")


class Helper(threading.Thread):
    """Stand-in for the Firefox extension's relay loop."""
    def __init__(self, port, downloads):
        super().__init__(daemon=True)
        self.no_resolve = False
        self.base, self.dl, self.stop, self.log = f"http://127.0.0.1:{port}/api", downloads, False, []

    def _post(self, path, body):
        req = urllib.request.Request(self.base + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json", "X-Lora-Vault": "1"})
        urllib.request.urlopen(req, timeout=10).read()

    def run(self):
        while not self.stop:
            try:
                j = json.load(urllib.request.urlopen(self.base + "/relay/next", timeout=30))
            except Exception:
                time.sleep(0.2); continue
            if j.get("task"):
                threading.Thread(target=self.handle, args=(j["task"],), daemon=True).start()

    def handle(self, t):
        self.log.append((t["kind"], t["url"]))
        try:
            if t["kind"] in ("json", "bytes"):
                try:
                    r = urllib.request.urlopen(urllib.request.Request(t["url"], headers=t["headers"]), timeout=10)
                    status, body = r.status, r.read()
                except urllib.error.HTTPError as e:
                    status, body = e.code, b""
                out = {"id": t["id"], "ok": True, "status": status}
                out["body" if t["kind"] == "json" else "body_b64"] = body.decode() if t["kind"] == "json" else base64.b64encode(body).decode()
                self._post("/relay/result", out)
            elif t["kind"] == "resolve":
                if self.no_resolve:
                    return self._post("/relay/result", {"id": t["id"], "ok": False, "error": "unknown task kind resolve"})
                class NR(urllib.request.HTTPRedirectHandler):
                    def redirect_request(self, *a, **k): return None
                try:
                    urllib.request.build_opener(NR).open(urllib.request.Request(t["url"]), timeout=10).read()
                    out = {"id": t["id"], "ok": True, "resolved": None, "note": "no redirect"}
                except urllib.error.HTTPError as e:
                    out = {"id": t["id"], "ok": True, "resolved": e.headers.get("Location")}
                self._post("/relay/result", out)
            else:
                r = urllib.request.urlopen(t["url"], timeout=10)
                if "html" in (r.headers.get("Content-Type") or ""):
                    return self._post("/relay/result", {"id": t["id"], "ok": False, "error": "Civitai sent a web page instead of the file"})
                path = os.path.join(self.dl, t["filename"]); os.makedirs(os.path.dirname(path), exist_ok=True)
                data = r.read(); total = len(data)
                with open(path, "wb") as f:
                    for i in range(0, total, 20000):
                        f.write(data[i:i + 20000])
                        self._post("/relay/progress", {"id": t["id"], "received": min(i + 20000, total), "total": total})
                self._post("/relay/result", {"id": t["id"], "ok": True, "path": path})
        except Exception as e:  # pragma: no cover
            self._post("/relay/result", {"id": t["id"], "ok": False, "error": str(e)})


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LORA_VAULT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LORA_VAULT_LORA_DIR", str(tmp_path / "loras"))
    fake = Fake().start()
    from vault import civitai, config, relay, server
    config.update({"civitai_hosts": [f"http://127.0.0.1:{fake.api_port}"], "civitai_api_key": "KEY123",
                   "cache_limit_gb": 0, "transport": "firefox", "download_mode": "firefox"})
    monkeypatch.setattr(civitai, "_opener", Boom())
    port = free_port(); server.start_in_thread(port); time.sleep(0.6)
    relay._last_poll = 0.0
    return fake, tmp_path, port


def start_helper(port, tmp_path):
    from vault import relay
    h = Helper(port, str(tmp_path / "Downloads")); h.start()
    for _ in range(50):
        if relay.connected(): return h
        time.sleep(0.1)
    raise AssertionError("helper never connected")


def test_not_connected_fails_fast_with_clear_message(env):
    fake, tmp, port = env
    from vault import civitai, snapshot
    t = time.time()
    with pytest.raises(civitai.NetworkError, match="Firefox helper"):
        snapshot.save_version(version_id=111)
    assert time.time() - t < 3


def test_everything_goes_through_the_helper(env):
    fake, tmp, port = env
    from vault import civitai, db, snapshot, storage
    h = start_helper(port, tmp)
    rec = snapshot.save_version(version_id=111)           # API json + preview bytes via helper
    assert rec["model_name"] == "Cool Style" and rec["snapshot"]["previews"][0]["local"]
    assert {k for k, _ in h.log} >= {"json", "bytes"}
    bases = civitai.get_base_models(); assert "Pony" in bases   # enums via helper
    assert [m["name"] for m in civitai.search_models(query="pony", base_model="Pony")["items"]] == ["Pony Thing"]
    seen = []
    path = storage.ensure_local(111, progress=lambda d, t: seen.append((d, t)))   # download via helper
    assert open(path, "rb").read() == PAYLOAD
    assert not os.path.exists(os.path.join(tmp, "Downloads", "LoraVault", os.path.basename(path))), "moved out of Firefox's folder"
    assert seen and seen[-1][0] == seen[-1][1] == len(PAYLOAD)
    dl_urls = [u for k, u in h.log if k == "download"]
    assert dl_urls and "token=KEY123" in dl_urls[0]
    h.stop = True


def test_login_page_instead_of_file_is_reported(env):
    fake, tmp, port = env
    from vault import civitai, config, snapshot, storage
    h = start_helper(port, tmp)
    snapshot.save_version(version_id=111)
    config.update({"civitai_api_key": ""})   # no token -> fake serves an HTML login page
    with pytest.raises(civitai.AuthRequired, match="web page"):
        storage.ensure_local(111)
    h.stop = True


def test_tiny_wrong_file_gets_login_hint_and_is_deleted(env):
    fake, tmp, port = env
    from vault import civitai, db, snapshot, storage
    h = start_helper(port, tmp)
    snapshot.save_version(version_id=111)
    fake.payload = b"tiny"
    with pytest.raises(civitai.AuthRequired, match="tiny file"):
        storage.ensure_local(111)
    assert db.get_version(111)["local_path"] is None
    h.stop = True


def test_removal_check_via_helper_and_skips_when_helper_gone(env):
    fake, tmp, port = env
    from vault import checker, db, snapshot, relay
    h = start_helper(port, tmp)
    snapshot.save_version(version_id=111)
    assert checker.check_version(111) == "ok"
    fake.hidden.add(111); fake.dead_models.add(11)
    assert checker.check_version(111) == "possibly_removed"
    h.stop = True
    relay._last_poll = 0.0           # helper gone: a full check must do nothing, not mark anything
    before = db.get_version(111)["miss_count"]
    checker.scheduler.check_all()
    assert db.get_version(111)["miss_count"] == before


def test_side_server_accepts_large_relay_results():
    """Regression: a >1 MB Civitai response (buzz/early-access models) was rejected with HTTP 413,
    so Firefox 'never answered' and the save timed out."""
    import asyncio
    from aiohttp.test_utils import TestClient, TestServer
    from vault import server

    async def go():
        async with TestClient(TestServer(server.build_app())) as c:
            big = {"id": "nope", "ok": True, "status": 200, "body": "x" * (3 * 1024 * 1024)}
            r = await c.post("/api/relay/result", json=big, headers={"X-Lora-Vault": "1", "Origin": "moz-extension://abc"})
            assert r.status == 200, r.status
    asyncio.run(go())


def test_download_goes_direct_from_comfyui_after_firefox_resolves_the_link(env):
    fake, tmp, port = env
    from vault import civitai, snapshot, storage
    from vault import config
    config.update({"download_mode": "direct"})
    h = start_helper(port, tmp)
    snapshot.save_version(version_id=111)
    # ComfyUI may only fetch the CDN link itself; the booby-trapped opener is swapped for a real one
    import urllib.request as ur
    class NR(ur.HTTPRedirectHandler):
        def redirect_request(self, *a, **k): return None
    civitai._opener = ur.build_opener(NR)
    seen = []
    path = storage.ensure_local(111, progress=lambda d, t: seen.append((d, t)))
    assert open(path, "rb").read() == PAYLOAD
    kinds = [k for k, _ in h.log]
    assert "resolve" in kinds and "download" not in kinds
    assert fake.cdn_auth_seen and all(a is None for a in fake.cdn_auth_seen), "API key must not reach the CDN"
    assert seen and seen[-1][0] == len(PAYLOAD)
    h.stop = True


def test_old_extension_without_resolve_falls_back_to_firefox_download(env):
    fake, tmp, port = env
    from vault import snapshot, storage
    from vault import config
    config.update({"download_mode": "direct"})
    h = start_helper(port, tmp); h.no_resolve = True
    snapshot.save_version(version_id=111)
    path = storage.ensure_local(111)
    assert open(path, "rb").read() == PAYLOAD
    assert "download" in [k for k, _ in h.log]
    h.stop = True


def test_download_mode_firefox_skips_resolve(env):
    fake, tmp, port = env
    from vault import config, snapshot, storage
    h = start_helper(port, tmp)
    snapshot.save_version(version_id=111)
    storage.ensure_local(111)
    assert "resolve" not in [k for k, _ in h.log]
    h.stop = True
