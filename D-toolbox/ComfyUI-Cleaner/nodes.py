"""
Graph nodes. These are for explicit control inside a workflow; the automatic
pre-run pass in hook.py is what guarantees a clean before every generation.
"""

from __future__ import annotations

from . import cleanup, hook
from .sysinfo import fmt


class AnyType(str):
    """Matches any socket type, so the passthrough accepts whatever you wire."""

    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False

    def __hash__(self):
        return hash("*")


ANY = AnyType("*")


def _report(result: dict) -> str:
    lines = [f"level: {result['level']}  ({result['ms']} ms)"]
    before, after = result.get("before", {}), result.get("after", {})
    if after.get("total"):
        lines.append(f"VRAM free: {fmt(before.get('free'))} -> {fmt(after.get('free'))} "
                     f"of {fmt(after.get('total'))}")
    for action in result.get("actions", []):
        lines.append(f"  {action}")
    for err in result.get("errors", []):
        lines.append(f"  failed: {err}")
    return "\n".join(lines)


class ResourceReaperClean:
    """
    Frees memory, then passes its input straight through.

    Because ComfyUI runs nodes in dependency order, this cleans at the point in
    the graph where its output is needed — not at the start of the run. To make
    it the genuine first step, right-click your checkpoint loader, convert
    ckpt_name to an input, and route it through this node's passthrough.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "level": (["light", "standard", "aggressive"], {"default": "standard"}),
                "purge_execution_cache": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "passthrough": (ANY, {}),
            },
        }

    RETURN_TYPES = (ANY, "STRING")
    RETURN_NAMES = ("passthrough", "report")
    FUNCTION = "clean"
    CATEGORY = "utils/resource reaper"
    DESCRIPTION = ("Reclaims VRAM and RAM, then forwards whatever is wired into "
                   "passthrough. Wire a loader's ckpt_name through it to run first.")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # never cached, so it runs on every queue

    def clean(self, level, purge_execution_cache, passthrough=None):
        result = cleanup.run_cleanup(level=level,
                                     purge_cache=purge_execution_cache,
                                     reason="node")
        return (passthrough, _report(result))


class ResourceReaperCleanAfter:
    """Cleans once images have been produced, so VRAM is free while you look."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "level": (["light", "standard", "aggressive"], {"default": "aggressive"}),
                "purge_execution_cache": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "clean"
    OUTPUT_NODE = True
    CATEGORY = "utils/resource reaper"
    DESCRIPTION = "Frees memory after a run completes. Chain before or after Save Image."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def clean(self, images, level, purge_execution_cache):
        result = cleanup.run_cleanup(level=level,
                                     purge_cache=purge_execution_cache,
                                     reason="node-after")
        return {"ui": {"text": [_report(result)]}, "result": (images,)}


class ResourceReaperStatus:
    """Emits the numbers from the last pre-run pass as text, for debugging."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"optional": {"passthrough": (ANY, {})}}

    RETURN_TYPES = (ANY, "STRING")
    RETURN_NAMES = ("passthrough", "status")
    FUNCTION = "status"
    CATEGORY = "utils/resource reaper"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def status(self, passthrough=None):
        last = hook.last_pass()
        snap = cleanup.vram_snapshot()
        lines = [f"device: {snap['device']}",
                 f"VRAM free: {fmt(snap['free'])} of {fmt(snap['total'])}"]
        if last:
            lines.append(f"last pass: {last['level']} on run {last.get('run')}, "
                         f"reclaimed {fmt(last.get('reclaimed', 0))}")
        else:
            lines.append("last pass: none yet")
        return (passthrough, "\n".join(lines))


NODE_CLASS_MAPPINGS = {
    "ResourceReaperClean": ResourceReaperClean,
    "ResourceReaperCleanAfter": ResourceReaperCleanAfter,
    "ResourceReaperStatus": ResourceReaperStatus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ResourceReaperClean": "Free Memory (passthrough)",
    "ResourceReaperCleanAfter": "Free Memory After Run",
    "ResourceReaperStatus": "Memory Status",
}
