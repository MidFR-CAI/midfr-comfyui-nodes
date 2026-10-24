"""MOSS-Image layout prompt builder (no ComfyUI dependencies).

Builds the same JSON as ComfyUI's built-in "MOSS-Image Layout Prompt" node:

    {"summary": {"caption": "<whole scene>"},
     "items": [{"label": "object", "caption": "<box prompt>", "bbox": [ymin, xmin, ymax, xmax]}],
     "bbox_coord_space": "abs_centralized",
     "bbox_format": "yxyx"}

`abs_centralized` = absolute pixels with the canvas centre as the origin (x right, y down),
so on a 1024x1024 canvas coordinates run -512..512. Source: OpenMOSS/MOSS-Image USERGUIDE.md.
"""
import json
import math


def parse_boxes(data):
    """The editor's `bbox_data` (JSON list, coordinates normalised 0..1, top-left origin)
    -> clean list of {x0,y0,x1,y1,caption}. Invalid entries are dropped."""
    try:
        raw = json.loads(data) if isinstance(data, (str, bytes)) else data
    except (TypeError, ValueError):
        return []
    if not isinstance(raw, list):
        return []
    out = []
    for b in raw:
        if not isinstance(b, dict):
            continue
        try:
            x0 = float(b["x"]); y0 = float(b["y"])
            x1 = x0 + float(b["w"]); y1 = y0 + float(b["h"])
        except (KeyError, TypeError, ValueError):
            continue
        if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
            continue
        x0, x1 = sorted((min(max(x0, 0.0), 1.0), min(max(x1, 0.0), 1.0)))
        y0, y1 = sorted((min(max(y0, 0.0), 1.0), min(max(y1, 0.0), 1.0)))
        if x1 <= x0 or y1 <= y0:
            continue
        out.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1,
                    "caption": str(b.get("caption") or "").strip()})
    return out


def _floor(v):
    return math.floor(round(v, 9))


def _ceil(v):
    return math.ceil(round(v, 9))


def build_prompt(bbox_data, width, height, prompt=""):
    """-> MOSS-Image layout JSON when there are boxes, otherwise the plain prompt."""
    items = []
    for b in parse_boxes(bbox_data):
        # Pixel edges enclosing the box, origin at the canvas centre (same maths as the core node).
        ymin = _floor(b["y0"] * height) - height // 2
        xmin = _floor(b["x0"] * width) - width // 2
        ymax = _ceil(b["y1"] * height) - height // 2
        xmax = _ceil(b["x1"] * width) - width // 2
        if ymax <= ymin or xmax <= xmin:
            continue
        items.append({"label": "object", "caption": b["caption"], "bbox": [ymin, xmin, ymax, xmax]})
    if not items:
        return prompt
    doc = {"summary": {"caption": prompt.strip()}, "items": items,
           "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}
    return json.dumps(doc, ensure_ascii=False, separators=(",", ":"))
