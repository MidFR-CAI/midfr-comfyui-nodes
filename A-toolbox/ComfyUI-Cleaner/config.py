"""Settings, persisted next to this file as config.json."""

from __future__ import annotations

import json
import logging
import os
import threading

log = logging.getLogger("ResourceReaper")

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

DEFAULTS = {
    # Run a pass automatically before every queued prompt.
    "auto_clean": True,

    # off | light | standard | aggressive | adaptive
    # adaptive picks a level from current VRAM pressure each run, which is the
    # only setting that does not make every run slower.
    "mode": "adaptive",

    # Below this percentage of free VRAM, adaptive mode goes aggressive.
    "adaptive_free_vram_pct": 25.0,

    # Also drop ComfyUI's node result cache and object storage. Reclaims the
    # most RAM; forces the whole graph to re-execute on the next run.
    "purge_execution_cache": False,

    # If purging is on, do it only every Nth run. 1 = every run.
    "purge_every_n_runs": 5,

    # Allow the panel to close other programs. Off by default.
    "allow_terminate": False,

    # Print a one-line summary of each pass to the console.
    "verbose": True,
}

_lock = threading.Lock()
_cache: dict | None = None


def load() -> dict:
    global _cache
    with _lock:
        if _cache is None:
            data = dict(DEFAULTS)
            if os.path.exists(PATH):
                try:
                    with open(PATH, "r", encoding="utf-8") as fh:
                        data.update(json.load(fh))
                except Exception as exc:
                    log.warning("[ResourceReaper] could not read config.json: %s", exc)
            _cache = data
        return dict(_cache)


def save(updates: dict) -> dict:
    global _cache
    with _lock:
        data = dict(_cache or DEFAULTS)
        for key, value in updates.items():
            if key in DEFAULTS:
                data[key] = value
        _cache = data
        try:
            with open(PATH, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        except Exception as exc:
            log.warning("[ResourceReaper] could not write config.json: %s", exc)
        return dict(data)
