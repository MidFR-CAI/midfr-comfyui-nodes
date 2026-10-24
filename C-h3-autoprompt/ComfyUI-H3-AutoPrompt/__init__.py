from .nodes import NODE_CLASS_MAPPINGS as _LEGACY_NODES, \
    NODE_DISPLAY_NAME_MAPPINGS as _LEGACY_NAMES
from .categorized_nodes import NODE_CLASS_MAPPINGS as _CAT_NODES, \
    NODE_DISPLAY_NAME_MAPPINGS as _CAT_NAMES
from .structured_nodes import NODE_CLASS_MAPPINGS as _STRUCT_NODES, \
    NODE_DISPLAY_NAME_MAPPINGS as _STRUCT_NAMES

NODE_CLASS_MAPPINGS = {**_LEGACY_NODES, **_CAT_NODES, **_STRUCT_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {**_LEGACY_NAMES, **_CAT_NAMES, **_STRUCT_NAMES}

# Comic-tab routes. A missing OpenCV/Tesseract must never stop the nodes from
# loading, so this is allowed to fail quietly -- comic mode simply won't work.
try:
    from . import web_api  # noqa: F401
except Exception as exc:  # pragma: no cover
    print(f"[H3 AutoPrompt] comic routes unavailable: {exc}")

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
