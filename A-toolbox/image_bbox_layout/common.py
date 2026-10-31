"""Shared load / resize helpers for the image upload nodes.

Pure PIL on purpose (torch is imported lazily in `to_tensors`) so the logic can be
unit-tested without ComfyUI.
"""
from PIL import Image, ImageOps

LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS


def snap(n, multiple):
    """Round n to the nearest multiple (ties down), never below one multiple."""
    return max(multiple, ((int(n) + multiple // 2 - 1) // multiple) * multiple)


def load_rgb_and_alpha(path):
    """-> (RGB image, alpha channel or None). Honours EXIF rotation.

    Alpha is what ComfyUI's mask editor writes when you paint a mask, and what a
    transparent PNG carries; it becomes the MASK output (same convention as the
    built-in Load Image: mask = 1 - alpha).
    """
    with Image.open(path) as im:
        img = ImageOps.exif_transpose(im)
        img.load()
    if img.mode == "I":
        img = img.point(lambda i: i * (1 / 255))
    has_alpha = "A" in img.getbands() or (img.mode == "P" and "transparency" in img.info)
    alpha = img.convert("RGBA").getchannel("A") if has_alpha else None
    return img.convert("RGB"), alpha


def resize_pair(rgb, alpha, max_megapixels, multiple=32, always_snap=False):
    """Downscale to at most `max_megapixels`, sides snapped to `multiple`.

    Images already under the cap are left alone unless `always_snap` is set, in
    which case their sides are still snapped. RGB and alpha are resized together.
    """
    w, h = rgb.size
    target = max_megapixels * 1_000_000
    if w * h > target:
        scale = (target / (w * h)) ** 0.5
        nw, nh = snap(int(w * scale), multiple), snap(int(h * scale), multiple)
    elif always_snap:
        nw, nh = snap(w, multiple), snap(h, multiple)
    else:
        return rgb, alpha
    if (nw, nh) == (w, h):
        return rgb, alpha
    rgb = rgb.resize((nw, nh), LANCZOS)
    if alpha is not None:
        alpha = alpha.resize((nw, nh), LANCZOS)
    return rgb, alpha


def to_tensors(rgb, alpha):
    """-> (IMAGE [1,H,W,3] float 0..1, MASK [1,H,W] float 0..1; zeros when there is no alpha)."""
    import numpy as np
    import torch

    image = torch.from_numpy(np.array(rgb).astype(np.float32) / 255.0)[None,]
    if alpha is None:
        mask = torch.zeros((1, rgb.height, rgb.width), dtype=torch.float32)
    else:
        mask = 1.0 - torch.from_numpy(np.array(alpha).astype(np.float32) / 255.0)[None,]
        mask = mask.clamp(0.0, 1.0)  # Lanczos can overshoot slightly
    return image, mask


def load_painted_mask(path, size):
    """Mask painted in the node's own paint tool: a PNG whose ALPHA is the mask. -> float32 HxW array 0..1 at `size`
    (width, height), or None if the file is missing / unreadable."""
    import numpy as np

    try:
        with Image.open(path) as im:
            a = im.convert("RGBA").getchannel("A")
    except Exception:
        return None
    if a.size != tuple(size):
        a = a.resize(tuple(size), Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR)
    return np.asarray(a, dtype="float32") / 255.0
