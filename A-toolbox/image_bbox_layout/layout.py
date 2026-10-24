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


def box_rects(bbox_data, width, height):
    """Pixel rectangles (x0, y0, x1, y1) of the boxes on a width x height image, top-left origin.
    Used to turn the boxes into a mask; same enclosing floor/ceil maths as the layout JSON."""
    rects = []
    for b in parse_boxes(bbox_data):
        x0, x1 = max(0, _floor(b["x0"] * width)), min(width, _ceil(b["x1"] * width))
        y0, y1 = max(0, _floor(b["y0"] * height)), min(height, _ceil(b["y1"] * height))
        if x1 > x0 and y1 > y0:
            rects.append((x0, y0, x1, y1))
    return rects


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


MASK_MODES = ("boxes + masks", "boxes only", "masks only", "whole image")


def soften_mask(arr, grow=0, feather=0):
    """arr: 2-D float numpy array in [0,1] (1 = regenerate). Returns it grown by `grow` px then
    blurred by `feather` px (soft edge for Differential Diffusion). Works on a 1/8 size copy."""
    import numpy as np
    from PIL import Image, ImageFilter

    if (grow <= 0 and feather <= 0) or not arr.any():
        return arr
    h, w = arr.shape
    s = 8
    small = Image.fromarray((np.clip(arr, 0, 1) * 255).astype("uint8"), "L")
    small = small.resize((max(1, w // s), max(1, h // s)), Image.BILINEAR)
    g = int(round(grow / s))
    if g > 0:
        small = small.filter(ImageFilter.MaxFilter(2 * g + 1))
    big = small.resize((w, h), Image.BILINEAR)
    if feather > 0:
        big = big.filter(ImageFilter.GaussianBlur(feather / 2.5))
    out = np.asarray(big, dtype="float32") / 255.0
    return np.maximum(out, 0.0) if grow <= 0 else out


def combine_mask(mode, box_mask, alpha_mask, ext_mask):
    """Each argument is a 2-D float array (or None). Returns the pre-softening mask."""
    import numpy as np

    if mode == "whole image":
        return np.ones_like(box_mask)
    parts = []
    if mode in ("boxes + masks", "boxes only"):
        parts.append(box_mask)
    if mode in ("boxes + masks", "masks only"):
        parts += [m for m in (alpha_mask, ext_mask) if m is not None]
    return np.maximum.reduce(parts) if parts else np.zeros_like(box_mask)
