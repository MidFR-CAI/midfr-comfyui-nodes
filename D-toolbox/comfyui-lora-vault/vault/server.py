"""Side server for the browser extension (127.0.0.1 only).

ComfyUI's own server rejects requests whose Origin isn't its own host (a CSRF guard),
which blocks extensions, so the extension talks to this tiny server instead. It only
accepts browser requests from extension origins (or ComfyUI itself) - a normal web
page can't call it - and requires a custom header on writes, which forces a CORS
preflight that only an allowed origin can pass.
"""
import asyncio
import threading

from aiohttp import web

from . import api, config

ALLOWED_PREFIXES = ("chrome-extension://", "moz-extension://", "http://127.0.0.1:8188", "http://localhost:8188")
LOCAL_HOSTS = ("127.0.0.1", "localhost")


@web.middleware
async def guard(request: web.Request, handler):
    host = (request.headers.get("Host") or "").split(":")[0].lower()
    if host not in LOCAL_HOSTS:  # DNS-rebinding guard
        return web.json_response({"ok": False, "error": "bad host"}, status=403)
    origin = request.headers.get("Origin")
    if origin and not origin.startswith(ALLOWED_PREFIXES):
        return web.json_response({"ok": False, "error": "origin not allowed"}, status=403)
    if request.method == "OPTIONS":
        resp = web.Response(status=204)
    else:
        if request.method != "GET" and request.headers.get("X-Lora-Vault") != "1":
            return web.json_response({"ok": False, "error": "missing X-Lora-Vault header"}, status=403)
        resp = await handler(request)
    if origin:
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Lora-Vault"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        resp.headers["Vary"] = "Origin"
    return resp


def build_app() -> web.Application:
    # aiohttp rejects request bodies over 1 MB by default (HTTP 413). Civitai's JSON for a model with
    # many example images, or a preview image, is bigger than that; the helper posts it back here.
    app = web.Application(middlewares=[guard], client_max_size=512 * 1024 * 1024)
    for method, path, handler in api.table("/api"):
        app.router.add_route(method, path, handler)
        if method == "POST":
            app.router.add_route("OPTIONS", path, handler)
    return app


def start_in_thread(port: int | None = None) -> threading.Thread:
    port = port or config.get("side_server_port")

    def run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        runner = web.AppRunner(build_app(), access_log=None)
        loop.run_until_complete(runner.setup())
        try:
            loop.run_until_complete(web.TCPSite(runner, "127.0.0.1", port).start())
            print(f"[LoraVault] extension server listening on http://127.0.0.1:{port}")
            loop.run_forever()
        except OSError as e:
            print(f"[LoraVault] could not start extension server on port {port}: {e}")

    t = threading.Thread(target=run, daemon=True, name="lora-vault-server")
    t.start()
    return t
