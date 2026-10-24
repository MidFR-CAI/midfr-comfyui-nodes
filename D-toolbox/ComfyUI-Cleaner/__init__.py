"""
Resource Reaper — keeps ComfyUI from getting slower run after run.

Loaded by ComfyUI at startup. Three things happen here:
  1. PromptExecutor is instrumented so cleanup can reach its caches
  2. PromptExecutor.execute is wrapped, so every prompt is preceded by a pass
  3. the /reaper/* routes and the sidebar panel are registered
"""

import logging

logging.getLogger("ResourceReaper").setLevel(logging.INFO)
log = logging.getLogger("ResourceReaper")

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]


def _startup():
    from . import api, cleanup, config, hook

    cfg = config.load()
    cleanup.patch_executor()

    hooked = hook.install()
    routed = api.register()

    log.info("[ResourceReaper] loaded — auto-clean %s, mode %s%s%s",
             "on" if cfg["auto_clean"] else "off",
             cfg["mode"],
             "" if hooked else " (pre-run hook unavailable, use the node instead)",
             "" if routed else " (panel unavailable)")

    from .sysinfo import psutil, _nvml_ready
    if psutil is None:
        log.warning("[ResourceReaper] psutil not installed — no system suggestions. "
                    "Install with: pip install psutil")
    if not _nvml_ready:
        log.info("[ResourceReaper] NVML unavailable — per-process VRAM will be hidden. "
                 "On NVIDIA, install with: pip install nvidia-ml-py")


try:
    _startup()
except Exception:  # never take ComfyUI down on import
    log.exception("[ResourceReaper] startup failed; nodes are still available")
