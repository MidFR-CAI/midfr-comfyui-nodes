"""MOSS-Image layout prompt builder (no ComfyUI dependencies).

MOSS-Image takes a JSON prompt:

    {"summary": {"meta": [...], "canvas": "single", "caption": "<whole scene>"},
     "items": [{"label": "object|character|text", "caption": "...", "text": "...", "bbox": [ymin, xmin, ymax, xmax]}],
     "bbox_coord_space": "abs_centralized",
     "bbox_format": "yxyx"}

`abs_centralized` = absolute pixels with the canvas centre as the origin (x right, y down),
so on a 1024x1024 canvas coordinates run -512..512. Source: OpenMOSS/MOSS-Image USERGUIDE.md.
"""
import json
import math

LABELS = ("object", "character", "text")
CANVASES = ("single", "multi_panel", "collage", "none")


def parse_boxes(data):
    """The editor's `bbox_data` (JSON list, coordinates normalised 0..1, top-left origin)
    -> clean list of {x0,y0,x1,y1,label,caption,text}. Invalid entries are dropped."""
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
        label = b.get("label") if b.get("label") in LABELS else "object"
        text = b.get("text")
        out.append({
            "x0": x0, "y0": y0, "x1": x1, "y1": y1, "label": label,
            "caption": str(b.get("caption") or "").strip(),
            "text": text if isinstance(text, str) and text.strip() else "",
        })
    return out


def _px(v):
    return int(math.floor(v + 0.5))


def build_layout(bbox_data, width, height, scene_prompt="", meta="", canvas="single", pretty=False):
    """-> MOSS-Image layout JSON string, or None when there are no usable boxes."""
    boxes = bbox_data if isinstance(bbox_data, list) and all("x0" in b for b in bbox_data) else parse_boxes(bbox_data)
    items = []
    for b in boxes:
        ymin = _px(b["y0"] * height - height / 2)
        xmin = _px(b["x0"] * width - width / 2)
        ymax = _px(b["y1"] * height - height / 2)
        xmax = _px(b["x1"] * width - width / 2)
        if ymax <= ymin or xmax <= xmin:
            continue
        item = {"label": b["label"], "caption": b["caption"]}
        if b["label"] == "text" and b["text"]:
            item["text"] = b["text"]
        item["bbox"] = [ymin, xmin, ymax, xmax]
        items.append(item)
    if not items:
        return None

    summary = {}
    tags = [t.strip() for t in (meta or "").split(",") if t.strip()]
    if tags:
        summary["meta"] = tags
    if canvas and canvas != "none":
        summary["canvas"] = canvas
    summary["caption"] = (scene_prompt or "").strip()

    doc = {"summary": summary, "items": items,
           "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}
    if pretty:
        return json.dumps(doc, indent=2, ensure_ascii=False)
    return json.dumps(doc, separators=(",", ":"), ensure_ascii=False)
