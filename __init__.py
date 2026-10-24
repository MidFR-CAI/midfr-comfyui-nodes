"""MidFR ComfyUI Nodes: loads every pack in this repo when it is cloned into custom_nodes.

    cd ComfyUI/custom_nodes
    git clone https://github.com/MidFR-CAI/midfr-comfyui-nodes

Each tool lives in its own folder (A-toolbox/*, B-image-tagging/*, C-h3-autoprompt/*)
and is loaded as a sub-package. A tool that fails to load (missing dependency, etc.)
is reported in the console and skipped; it never stops the others.

To skip tools, list folder names (one per line, '#' for comments) in `disabled.txt`
next to this file, or set MIDFR_DISABLE="comfyui-lora-vault,ComfyUI-Cleaner".
"""
import importlib.util
import os
import re
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_PACKS = ("A-toolbox", "B-image-tagging", "C-h3-autoprompt")
_TAG = "[MidFR]"

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]


def _disabled():
    names = {n.strip().lower() for n in os.environ.get("MIDFR_DISABLE", "").split(",") if n.strip()}
    try:
        with open(os.path.join(_ROOT, "disabled.txt"), encoding="utf-8") as fh:
            for line in fh:
                line = line.split("#", 1)[0].strip()
                if line:
                    names.add(line.lower())
    except OSError:
        pass
    return names


def _import_tool(folder, path):
    full = __name__ + "." + re.sub(r"\W", "_", folder)
    spec = importlib.util.spec_from_file_location(
        full, os.path.join(path, "__init__.py"), submodule_search_locations=[path])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        for key in [k for k in sys.modules if k == full or k.startswith(full + ".")]:
            sys.modules.pop(key, None)
        raise
    return mod


def _register_web(folder, path, mod):
    """ComfyUI serves one web folder per package; register each tool's own."""
    rel = getattr(mod, "WEB_DIRECTORY", None)
    if not rel:
        return
    web_dir = os.path.abspath(os.path.join(path, rel))
    if not os.path.isdir(web_dir):
        return
    try:
        import nodes as comfy_nodes  # ComfyUI's own nodes module
        comfy_nodes.EXTENSION_WEB_DIRS["midfr-" + re.sub(r"\W", "_", folder)] = web_dir
    except Exception as exc:
        print(f"{_TAG} could not register web files for {folder}: {exc}")


def _load_all():
    skip = _disabled()
    loaded, failed, skipped = [], [], []
    for pack in _PACKS:
        pack_dir = os.path.join(_ROOT, pack)
        if not os.path.isdir(pack_dir):
            continue
        for folder in sorted(os.listdir(pack_dir)):
            path = os.path.join(pack_dir, folder)
            if not os.path.isfile(os.path.join(path, "__init__.py")):
                continue
            if folder.lower() in skip or pack.lower() in skip:
                skipped.append(folder)
                continue
            try:
                mod = _import_tool(folder, path)
            except BaseException as exc:  # one broken tool must not take down ComfyUI
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                failed.append(folder)
                print(f"{_TAG} {folder}: failed to load ({type(exc).__name__}: {exc})")
                continue
            for key, cls in (getattr(mod, "NODE_CLASS_MAPPINGS", None) or {}).items():
                if key in NODE_CLASS_MAPPINGS:
                    print(f"{_TAG} {folder}: node '{key}' already registered, skipping")
                    continue
                NODE_CLASS_MAPPINGS[key] = cls
            for key, name in (getattr(mod, "NODE_DISPLAY_NAME_MAPPINGS", None) or {}).items():
                NODE_DISPLAY_NAME_MAPPINGS.setdefault(key, name)
            _register_web(folder, path, mod)
            loaded.append(folder)
    print(f"{_TAG} loaded {len(loaded)} tool(s), {len(NODE_CLASS_MAPPINGS)} node(s)"
          + (f"; skipped: {', '.join(skipped)}" if skipped else "")
          + (f"; FAILED: {', '.join(failed)}" if failed else ""))


_load_all()
