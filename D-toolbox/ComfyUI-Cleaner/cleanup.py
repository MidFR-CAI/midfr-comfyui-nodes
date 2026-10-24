"""
Memory reclamation primitives for ComfyUI.

Everything here is defensive: ComfyUI's internals move around between versions,
so each operation is attempted independently and failures are recorded rather
than raised. A cleanup pass must never be able to break a generation.
"""

from __future__ import annotations

import gc
import logging
import time
import weakref

log = logging.getLogger("ResourceReaper")

# Populated by patch_executor() so we can reach the running PromptExecutor
# without relying on it being exported anywhere.
_EXECUTOR_REFS: list = []

LEVELS = ("off", "light", "standard", "aggressive")


# --------------------------------------------------------------------------
# lazy imports
# --------------------------------------------------------------------------

def _torch():
    try:
        import torch
        return torch
    except Exception:
        return None


def _mm():
    try:
        import comfy.model_management as mm
        return mm
    except Exception:
        return None


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------

def vram_snapshot() -> dict:
    """Best-effort view of the compute device's memory, in bytes."""
    snap = {
        "device": "unknown",
        "total": None,
        "free": None,
        "used": None,
        "torch_allocated": None,
        "torch_reserved": None,
    }

    mm = _mm()
    if mm is not None:
        try:
            dev = mm.get_torch_device()
            snap["device"] = str(dev)
            snap["total"] = int(mm.get_total_memory(dev))
            snap["free"] = int(mm.get_free_memory(dev))
            snap["used"] = snap["total"] - snap["free"]
        except Exception as exc:
            log.debug("vram_snapshot via model_management failed: %s", exc)

    torch = _torch()
    if torch is not None:
        try:
            if torch.cuda.is_available():
                snap["torch_allocated"] = int(torch.cuda.memory_allocated())
                snap["torch_reserved"] = int(torch.cuda.memory_reserved())
                if snap["total"] is None:
                    free, total = torch.cuda.mem_get_info()
                    snap["total"] = int(total)
                    snap["free"] = int(free)
                    snap["used"] = int(total - free)
                    snap["device"] = torch.cuda.get_device_name(0)
        except Exception as exc:
            log.debug("vram_snapshot via torch failed: %s", exc)

    return snap


def free_fraction() -> float | None:
    """Free VRAM as a fraction of total, or None if it can't be read."""
    snap = vram_snapshot()
    if not snap["total"]:
        return None
    return snap["free"] / snap["total"]


# --------------------------------------------------------------------------
# reaching the executor
# --------------------------------------------------------------------------

def patch_executor() -> bool:
    """
    Wrap PromptExecutor.__init__ so we keep a weak reference to every executor
    instance. Called once at import time, before the worker thread starts.
    """
    try:
        import execution
    except Exception:
        return False

    cls = getattr(execution, "PromptExecutor", None)
    if cls is None or getattr(cls, "_reaper_tracked", False):
        return False

    original = cls.__init__

    def tracked_init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        _EXECUTOR_REFS.append(weakref.ref(self))

    cls.__init__ = tracked_init
    cls._reaper_tracked = True
    return True


def _live_executors() -> list:
    live = [ref() for ref in _EXECUTOR_REFS]
    live = [e for e in live if e is not None]
    if live:
        return live
    # Fallback for load-order surprises: sweep the heap once.
    found = []
    for obj in gc.get_objects():
        try:
            if type(obj).__name__ == "PromptExecutor":
                found.append(obj)
        except Exception:
            continue
    return found


def purge_execution_cache() -> dict:
    """
    Drop ComfyUI's per-node result cache and its object_storage (which is what
    holds loaded checkpoints, LoRAs, CLIP models and so on between runs).

    This is the operation that actually reclaims long-lived RAM, and it is also
    the expensive one: the next queued prompt re-executes and re-loads from
    scratch. Off by default for that reason.
    """
    result = {"executors": 0, "cleared": [], "errors": []}

    for ex in _live_executors():
        result["executors"] += 1

        # Newer ComfyUI: executor.caches is a CacheSet of cache objects.
        caches = getattr(ex, "caches", None)
        if caches is not None:
            for name in ("outputs", "ui", "objects"):
                cache = getattr(caches, name, None)
                if cache is None:
                    continue
                for attr in ("cache", "subcaches"):
                    store = getattr(cache, attr, None)
                    if isinstance(store, dict):
                        try:
                            store.clear()
                            result["cleared"].append(f"caches.{name}.{attr}")
                        except Exception as exc:
                            result["errors"].append(f"caches.{name}.{attr}: {exc}")

        # Older ComfyUI: plain dicts on the executor.
        for attr in ("outputs", "outputs_ui", "old_prompt", "object_storage"):
            store = getattr(ex, attr, None)
            if isinstance(store, dict):
                try:
                    store.clear()
                    result["cleared"].append(attr)
                except Exception as exc:
                    result["errors"].append(f"{attr}: {exc}")

    return result


# --------------------------------------------------------------------------
# the cleanup pass
# --------------------------------------------------------------------------

def run_cleanup(level: str = "standard",
                purge_cache: bool = False,
                reason: str = "manual") -> dict:
    """
    Reclaim memory and report what changed.

    light       gc + CUDA allocator cache. Models stay resident, so the next
                run does not pay a reload cost. Fixes allocator bloat only.
    standard    light, plus ComfyUI's own model housekeeping and a forced
                allocator flush. Frees models ComfyUI considers unneeded.
    aggressive  standard, plus unload every loaded model and collect IPC
                handles. Guarantees a clean slate, costs a full model reload.
    """
    if level not in LEVELS:
        level = "standard"

    started = time.perf_counter()
    before = vram_snapshot()
    actions: list[str] = []
    errors: list[str] = []

    if level == "off":
        return {
            "level": level, "reason": reason, "actions": [], "errors": [],
            "before": before, "after": before, "reclaimed": 0, "ms": 0.0,
        }

    mm = _mm()
    torch = _torch()

    if level == "aggressive":
        if mm is not None and hasattr(mm, "unload_all_models"):
            try:
                mm.unload_all_models()
                actions.append("unloaded all models")
            except Exception as exc:
                errors.append(f"unload_all_models: {exc}")
    elif level == "standard":
        if mm is not None and hasattr(mm, "cleanup_models"):
            try:
                try:
                    mm.cleanup_models(keep_clone_weights_loaded=True)
                except TypeError:
                    mm.cleanup_models()
                actions.append("released unneeded models")
            except Exception as exc:
                errors.append(f"cleanup_models: {exc}")

    if purge_cache:
        purge = purge_execution_cache()
        if purge["cleared"]:
            actions.append(f"purged execution cache ({len(purge['cleared'])} stores)")
        errors.extend(purge["errors"])

    try:
        collected = gc.collect()
        actions.append(f"collected {collected} python objects")
    except Exception as exc:
        errors.append(f"gc.collect: {exc}")

    if mm is not None and hasattr(mm, "soft_empty_cache"):
        try:
            mm.soft_empty_cache(force=(level != "light"))
            actions.append("flushed allocator cache")
        except Exception as exc:
            errors.append(f"soft_empty_cache: {exc}")
    elif torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                actions.append("flushed allocator cache")
        except Exception as exc:
            errors.append(f"empty_cache: {exc}")

    if level == "aggressive" and torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.ipc_collect()
                actions.append("collected IPC handles")
        except Exception as exc:
            errors.append(f"ipc_collect: {exc}")

    after = vram_snapshot()
    reclaimed = 0
    if before["free"] is not None and after["free"] is not None:
        reclaimed = after["free"] - before["free"]

    return {
        "level": level,
        "reason": reason,
        "actions": actions,
        "errors": errors,
        "before": before,
        "after": after,
        "reclaimed": reclaimed,
        "ms": round((time.perf_counter() - started) * 1000, 1),
    }


def resolve_adaptive_level(free_pct_threshold: float) -> str:
    """
    Pick a level from current pressure. Below the threshold we go aggressive,
    within 15 points above it we do standard housekeeping, otherwise light.
    """
    frac = free_fraction()
    if frac is None:
        return "light"
    pct = frac * 100
    if pct < free_pct_threshold:
        return "aggressive"
    if pct < free_pct_threshold + 15:
        return "standard"
    return "light"
