"""
tagger.py -- Danbooru-style tagging for comic panels (WD SwinV2 Tagger v3).

A general 4B VLM reads comic art poorly: it hedges, invents, and misses the
things that actually matter for a shot description (who is present, pose,
expression, setting). A booru tagger is trained on exactly this material and
returns concrete, high-precision tags instead. The plan is therefore:

    panel image -> tagger -> tags -> VLM (formats tags into prose)

so the VLM is only doing the thing it IS good at -- turning a reliable list of
facts into a sentence -- rather than being asked to perceive.

Three ways to get tags, tried in order:
  1. An already-installed ComfyUI WD14 tagger node, found via ComfyUI's node
     registry. Most people doing anime work already have one, and it brings its
     own model management.
  2. onnxruntime directly against SmilingWolf/wd-swinv2-tagger-v3, downloading
     through huggingface_hub.
  3. Nothing -- the caller falls back to the VLM alone, and is told why.

NOTE: the fallback chain and formatting are tested, but the actual model has
NOT been run here (no network access to HuggingFace in the build environment).
The ONNX path follows the model card's documented preprocessing.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

REPO_ID = "SmilingWolf/wd-swinv2-tagger-v3"
_MODEL_FILE = "model.onnx"
_LABEL_FILE = "selected_tags.csv"

# Booru tags that describe the medium rather than the content. They'd push the
# VLM toward describing a drawing instead of a scene, which is the opposite of
# what a video prompt needs.
_MEDIUM_TAGS = {
    "comic", "monochrome", "greyscale", "grayscale", "manga", "speech_bubble",
    "text", "english_text", "japanese_text", "translation_request", "borders",
    "panel", "4koma", "sketch", "lineart", "traditional_media", "watermark",
    "signature", "artist_name", "commentary", "commentary_request", "web_address",
}

_session = None
_tag_names: List[str] = []
_rating_count = 0


def _try_comfy_node():
    """A WD14 tagger node already installed in ComfyUI, if there is one."""
    try:
        import nodes as comfy_nodes
    except Exception:
        return None
    mapping = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}) or {}
    for key, cls in mapping.items():
        k = key.lower()
        if "wd14" in k or ("tagger" in k and "wd" in k):
            return cls
    return None


def _load_onnx():
    """Loads the ONNX model and its label list, downloading if needed."""
    global _session, _tag_names, _rating_count
    if _session is not None:
        return True
    try:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        import pandas as pd
    except Exception:
        return False
    try:
        model_path = hf_hub_download(REPO_ID, _MODEL_FILE)
        label_path = hf_hub_download(REPO_ID, _LABEL_FILE)
    except Exception:
        return False
    try:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
            if "CUDAExecutionProvider" in ort.get_available_providers() \
            else ["CPUExecutionProvider"]
        _session = ort.InferenceSession(model_path, providers=providers)
        df = pd.read_csv(label_path)
        _tag_names = df["name"].tolist()
        # category 9 == rating tags, which lead the list in this model's labels.
        _rating_count = int((df["category"] == 9).sum())
        return True
    except Exception:
        _session = None
        return False


def _preprocess(img, size: int):
    """Square-pads to white and resizes, per the model card's preprocessing."""
    import numpy as np
    from PIL import Image
    img = img.convert("RGBA")
    canvas = Image.new("RGBA", img.size, (255, 255, 255))
    canvas.alpha_composite(img)
    img = canvas.convert("RGB")
    w, h = img.size
    side = max(w, h)
    square = Image.new("RGB", (side, side), (255, 255, 255))
    square.paste(img, ((side - w) // 2, (side - h) // 2))
    if side != size:
        square = square.resize((size, size), Image.BICUBIC)
    arr = np.asarray(square, dtype=np.float32)
    arr = arr[:, :, ::-1]                 # RGB -> BGR, as the model expects
    return np.expand_dims(arr, 0)


def tag_image(img, threshold: float = 0.35, limit: int = 30) -> List[str]:
    """Returns content tags for one PIL image, most confident first.

    An empty list means tagging wasn't available -- check `status()` for why,
    rather than treating it as "nothing in the picture".
    """
    if not _load_onnx():
        return []
    try:
        import numpy as np
        inp = _session.get_inputs()[0]
        size = inp.shape[1] if isinstance(inp.shape[1], int) else 448
        batch = _preprocess(img, size)
        preds = _session.run(None, {inp.name: batch})[0][0]
        scored = [(_tag_names[i], float(preds[i]))
                  for i in range(_rating_count, len(_tag_names))
                  if float(preds[i]) >= threshold]
        scored.sort(key=lambda t: t[1], reverse=True)
        out = []
        for name, _conf in scored:
            clean = name.replace("_", " ").strip()
            if name in _MEDIUM_TAGS or clean in _MEDIUM_TAGS:
                continue          # medium, not content -- see _MEDIUM_TAGS
            out.append(clean)
            if len(out) >= limit:
                break
        return out
    except Exception:
        return []


def status() -> dict:
    """Whether tagging is usable, and if not, what to do about it."""
    if _try_comfy_node() is not None:
        return {"ok": True, "source": "ComfyUI WD14 tagger node"}
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        return {"ok": False,
                "reason": "onnxruntime isn't installed. Install it, or install a "
                          "ComfyUI WD14 tagger node, to tag comic panels."}
    try:
        import huggingface_hub  # noqa: F401
    except Exception:
        return {"ok": False,
                "reason": "huggingface_hub isn't installed, so "
                          f"{REPO_ID} can't be fetched. Install it, or install a "
                          "ComfyUI WD14 tagger node."}
    if _load_onnx():
        return {"ok": True, "source": REPO_ID}
    return {"ok": False,
            "reason": f"couldn't load {REPO_ID} (no network, or the download "
                      f"failed). Panels will be described by the VLM alone."}
