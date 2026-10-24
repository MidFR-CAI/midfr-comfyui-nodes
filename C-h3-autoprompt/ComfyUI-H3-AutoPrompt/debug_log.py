"""
debug_log.py -- opt-in tracing of the whole prompt-building pipeline, printed
straight to the ComfyUI console window.

Turned on with the All-in-One node's "debug" toggle. Off by default and cheap
when off: every call site goes through `on()` first, so nothing is formatted
unless tracing is actually enabled.

The point is to answer "where did this file go, and what became of it" without
reading the code -- every reference is followed from the loader panel's JSON,
through tag allocation, into the plan, into what the VLM was shown, and out
into the finished prompt's sections.
"""

from __future__ import annotations

import json
from typing import Any

_ENABLED = False
_PREFIX = "[H3]"


def set_enabled(flag: bool) -> None:
    global _ENABLED
    _ENABLED = bool(flag)


def on() -> bool:
    return _ENABLED


def section(title: str) -> None:
    if not _ENABLED:
        return
    print(f"\n{_PREFIX} {'=' * 8} {title} {'=' * 8}")


def line(text: str = "") -> None:
    if not _ENABLED:
        return
    print(f"{_PREFIX} {text}")


def kv(key: str, value: Any) -> None:
    if not _ENABLED:
        return
    print(f"{_PREFIX}   {key}: {value}")


def bullet(text: str) -> None:
    if not _ENABLED:
        return
    print(f"{_PREFIX}   - {text}")


def blob(label: str, obj: Any, limit: int = 2000) -> None:
    """Pretty-prints a dict/list, truncated so a big tensor-free structure
    doesn't flood the console."""
    if not _ENABLED:
        return
    try:
        text = json.dumps(obj, indent=2, default=str)
    except Exception:
        text = repr(obj)
    if len(text) > limit:
        text = text[:limit] + f"\n... [{len(text) - limit} more chars]"
    print(f"{_PREFIX} {label}:")
    for ln in text.splitlines():
        print(f"{_PREFIX}   {ln}")


def multiline(label: str, text: str) -> None:
    if not _ENABLED:
        return
    print(f"{_PREFIX} {label}:")
    for ln in (text or "").splitlines():
        print(f"{_PREFIX}   {ln}")
