"""
Runs a cleanup pass immediately before ComfyUI executes a prompt.

A node in the graph cannot guarantee it goes first: ComfyUI orders execution by
data dependency, so a node only runs before whatever consumes its output. To
actually be first, we wrap PromptExecutor.execute, which is the single point
every queued prompt passes through, on the worker thread, after the queue has
handed the job over and before any node runs.
"""

from __future__ import annotations

import inspect
import logging

from . import cleanup, config

log = logging.getLogger("ResourceReaper")

_run_counter = 0
_last_pass: dict | None = None
_installed = False


def last_pass() -> dict | None:
    return _last_pass


def run_pass(reason: str = "pre-run") -> dict:
    """One cleanup pass, using current settings. Never raises."""
    global _run_counter, _last_pass

    cfg = config.load()
    _run_counter += 1

    mode = cfg.get("mode", "adaptive")
    if mode == "adaptive":
        level = cleanup.resolve_adaptive_level(float(cfg.get("adaptive_free_vram_pct", 25.0)))
    else:
        level = mode

    purge = bool(cfg.get("purge_execution_cache", False))
    if purge:
        every = max(1, int(cfg.get("purge_every_n_runs", 1)))
        purge = (_run_counter % every == 0)

    try:
        result = cleanup.run_cleanup(level=level, purge_cache=purge, reason=reason)
    except Exception as exc:  # pragma: no cover - must never break a run
        log.exception("[ResourceReaper] cleanup pass failed")
        result = {"level": level, "reason": reason, "actions": [], "errors": [str(exc)],
                  "before": {}, "after": {}, "reclaimed": 0, "ms": 0.0}

    result["run"] = _run_counter
    result["purged_cache"] = purge
    _last_pass = result

    if cfg.get("verbose", True) and level != "off":
        from .sysinfo import fmt
        free = result.get("after", {}).get("free")
        log.info("[ResourceReaper] %s pass (%s): reclaimed %s, %s free, %.0fms",
                 level, reason, fmt(result.get("reclaimed", 0)), fmt(free), result.get("ms", 0))
        for err in result.get("errors", []):
            log.warning("[ResourceReaper] %s", err)

    return result


def install() -> bool:
    """Wrap PromptExecutor.execute. Safe to call more than once."""
    global _installed
    if _installed:
        return True

    try:
        import execution
    except Exception as exc:
        log.warning("[ResourceReaper] execution module unavailable, auto-clean off: %s", exc)
        return False

    cls = getattr(execution, "PromptExecutor", None)
    if cls is None or not hasattr(cls, "execute"):
        log.warning("[ResourceReaper] PromptExecutor.execute not found, auto-clean off")
        return False

    original = cls.execute

    if inspect.iscoroutinefunction(original):
        async def wrapped(self, *args, **kwargs):
            if config.load().get("auto_clean", True):
                run_pass("pre-run")
            return await original(self, *args, **kwargs)
    else:
        def wrapped(self, *args, **kwargs):
            if config.load().get("auto_clean", True):
                run_pass("pre-run")
            return original(self, *args, **kwargs)

    wrapped.__wrapped__ = original
    cls.execute = wrapped
    cls._reaper_execute_patched = True
    _installed = True
    return True
