"""HTTP routes the sidebar panel talks to."""

from __future__ import annotations

import logging

from aiohttp import web

from . import cleanup, config, hook, sysinfo

log = logging.getLogger("ResourceReaper")


def register() -> bool:
    try:
        from server import PromptServer
        routes = PromptServer.instance.routes
    except Exception as exc:
        log.warning("[ResourceReaper] could not register routes: %s", exc)
        return False

    @routes.get("/reaper/status")
    async def status(request):
        data = sysinfo.snapshot()
        data["config"] = config.load()
        data["torch"] = cleanup.vram_snapshot()
        data["last_pass"] = hook.last_pass()
        return web.json_response(data)

    @routes.post("/reaper/clean")
    async def clean(request):
        try:
            body = await request.json()
        except Exception:
            body = {}
        level = body.get("level", "standard")
        purge = bool(body.get("purge_execution_cache", False))
        result = hook.run_pass("manual") if level == "adaptive" else \
            cleanup.run_cleanup(level=level, purge_cache=purge, reason="manual")
        return web.json_response(result)

    @routes.get("/reaper/config")
    async def get_config(request):
        return web.json_response(config.load())

    @routes.post("/reaper/config")
    async def set_config(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "expected a JSON body"}, status=400)
        return web.json_response(config.save(body))

    @routes.post("/reaper/terminate")
    async def terminate(request):
        if not config.load().get("allow_terminate", False):
            return web.json_response(
                {"ok": False, "error": "closing programs is turned off in settings"},
                status=403)
        try:
            body = await request.json()
            pid = int(body["pid"])
        except Exception:
            return web.json_response({"ok": False, "error": "a pid is required"}, status=400)
        result = sysinfo.terminate(pid)
        return web.json_response(result, status=200 if result["ok"] else 409)

    return True
