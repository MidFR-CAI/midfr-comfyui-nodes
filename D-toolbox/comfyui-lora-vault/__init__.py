import os

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

_started = False


def _start():
    global _started
    if _started:
        return
    _started = True
    try:
        from aiohttp import web
        from server import PromptServer

        from .vault import api, checker, config, server, storage

        routes = PromptServer.instance.routes
        for method, path, handler in api.table("/lora_vault/api"):
            routes.route(method, path)(handler)

        ui_file = os.path.join(os.path.dirname(__file__), "ui", "index.html")

        @routes.get("/lora_vault")
        async def vault_page(request):
            return web.FileResponse(ui_file, headers={"Cache-Control": "no-cache"})

        storage.reconcile()
        checker.scheduler.start()
        if config.get("side_server_enabled"):
            server.start_in_thread()
        print("[LoraVault] ready - open http://127.0.0.1:8188/lora_vault")
    except Exception as e:  # never break ComfyUI startup
        print(f"[LoraVault] failed to start: {e}")


_start()
