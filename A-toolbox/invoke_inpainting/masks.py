"""Mask and size maths ported from InvokeAI (create_gradient_mask, expand_mask_with_fade,
canvasProcessingSize). numpy + Pillow only, so it can be tested without ComfyUI/torch.

Conventions here: masks are float arrays in [0, 1], 1 = regenerate. Where Invoke works in 0..255 with
255 = keep, the port converts at the edges and keeps Invoke's arithmetic in between."""
import numpy as np
from PIL import Image

LATENT_SCALE = 8  # Invoke builds the denoise mask at 1/8 size (MOSS raw latent scale)
GRID = 16  # MOSS-Image works in multiples of 16
OPTIMAL_SIDE = 1024
COHERENCE_MODES = ("Gaussian Blur", "Box Blur", "Staged")


# ---------------------------------------------------------------- sizes
def snap(value, grid=GRID):
    return max(grid, int(round(value / grid)) * grid)


def processing_size(width, height, scale="auto", grid=GRID, optimal=OPTIMAL_SIDE, max_side=4096):
    """Invoke 'auto' scaling: snap to the grid, and grow undersized boxes toward ~1 MP, keeping aspect."""
    w, h = snap(width, grid), snap(height, grid)
    if scale != "auto" or w * h >= optimal * optimal:
        return w, h
    if w == h:
        return optimal, optimal
    ratio = w / h
    longest = optimal - grid
    cw, ch = w, h
    while cw * ch < optimal * optimal and longest < max_side:
        longest += grid
        if ratio > 1:
            cw, ch = longest, snap(longest / ratio, grid)
        else:
            ch, cw = longest, snap(longest * ratio, grid)
    return cw, ch


def mask_rect(mask, threshold=0.5):
    """Bounding rect (x0, y0, x1, y1) of mask > threshold, or None."""
    ys, xs = np.where(mask > threshold)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def resize_mask(mask, width, height, resample=Image.BILINEAR):
    if mask.shape == (height, width):
        return mask.astype("float32")
    im = Image.fromarray(np.ascontiguousarray(mask, dtype="float32"), "F")
    return np.asarray(im.resize((width, height), resample), dtype="float32")


# ---------------------------------------------------------------- gradient mask
def _kernel(radius, mode):
    size = radius * 2 + 1
    yy, xx = np.mgrid[0:size, 0:size].astype("float32")
    dist = np.sqrt((yy - radius) ** 2 + (xx - radius) ** 2)
    if mode == "Box Blur":
        k = np.where(dist <= radius, 1.0 - dist / radius, 0.0)
    else:  # Gaussian Blur / Staged: gaussian with sigma = radius / 2.5, centre normalised to 1
        sigma = radius / 2.5
        k = np.exp(-(dist ** 2) / (2 * sigma ** 2))
        k = np.where(dist > radius, 0.0, k)
    return k.astype("float32")


def _max_filter(img, kernel):
    """Weighted max filter: max over offsets of img shifted * weight (zero padded), as Invoke does."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode="constant")
    out = np.zeros_like(img)
    h, w = img.shape
    for i in range(kh):
        for j in range(kw):
            wt = kernel[i, j]
            if wt > 0:
                np.maximum(out, padded[i:i + h, j:j + w] * wt, out=out)
    return out


def gradient_mask(mask, edge_size=16, mode="Gaussian Blur", min_denoise=0.0):
    """Invoke 'Create Gradient Mask'. mask: HxW float (1 = regenerate) at processing size.

    Returns (weight, area): weight is the denoise mask (HxW float 0..1, 1 = fully regenerated, values in
    between are regenerated only late in the run: this is the 'gradient'), area is a bool HxW of
    everything the model may touch (mask plus the grown edge)."""
    h, w = mask.shape
    sw, sh = max(1, w // LATENT_SCALE), max(1, h // LATENT_SCALE)
    small = resize_mask(np.clip(mask, 0, 1), sw, sh) * 255.0
    keep_orig = 255.0 - small  # Invoke's mask image: 255 = keep
    radius = edge_size // LATENT_SCALE
    if radius > 0:
        regen = 255.0 - keep_orig
        keep = 255.0 - _max_filter(regen, _kernel(radius, mode))
        threshold = (1.0 - min_denoise) * 255.0
        if mode == "Staged":
            sel = (keep < keep_orig) & (keep_orig > threshold)
            keep = np.where(sel, threshold, keep_orig)
        sel = (keep > threshold) & (keep < 255.0)
        keep = np.where(sel, threshold, keep)
    else:
        keep = keep_orig.copy()
    keep = np.clip(keep, 0, 255).astype("uint8")
    area_small = keep < 255
    weight_small = 1.0 - keep.astype("float32") / 255.0
    up = lambda a: np.asarray(Image.fromarray(a.astype("float32"), "F").resize((w, h), Image.NEAREST))
    return up(weight_small).astype("float32"), up(area_small.astype("float32")) > 0.5


# ---------------------------------------------------------------- paste-back fade
def _distance_to_black(binary_white):
    """Distance (px) from every pixel to the nearest 0 pixel of a 0/1 array."""
    try:
        import cv2

        return cv2.distanceTransform((binary_white * 255).astype("uint8"), cv2.DIST_L2, 5)
    except Exception:
        pass
    try:
        from scipy.ndimage import distance_transform_edt

        return distance_transform_edt(binary_white > 0).astype("float32")
    except Exception:
        pass
    # last resort: chamfer-ish by repeated 3x3 min filter on a quarter-size copy
    from PIL import ImageFilter

    h, w = binary_white.shape
    im = Image.fromarray((binary_white * 255).astype("uint8"), "L").resize((max(1, w // 4), max(1, h // 4)), Image.NEAREST)
    dist = np.zeros(im.size[::-1], dtype="float32")
    cur = im
    for step in range(1, 130):
        cur = cur.filter(ImageFilter.MinFilter(3))
        arr = np.asarray(cur)
        if not arr.any():
            dist[:] = np.where(dist == 0, step, dist)
            break
        dist[arr > 0] = step
    return np.asarray(Image.fromarray(dist * 4, "F").resize((w, h), Image.BILINEAR))


_X = np.array([0.0, 0.0, 0.2, 0.8, 1.0])
_Y = np.array([0.0, 0.0, 0.2, 0.9, 1.0])


def fade_alpha(area, fade_px=16):
    """Invoke 'Expand Mask With Fade' + 'Apply Mask To Image': alpha of the generated image when it is
    pasted over the original. 1 inside `area`, easing to 0 over `fade_px` pixels outside it."""
    if fade_px <= 0:
        return area.astype("float32")
    if not area.any():
        return np.zeros(area.shape, dtype="float32")
    white = (~area).astype("float32")  # 1 = discard (outside the area)
    dist = _distance_to_black(white)
    x = np.array([0.0, 1.0 / fade_px, 0.2, 0.8, 1.0])
    poly = np.poly1d(np.polyfit(x, _Y, 3))
    fade = np.where(dist >= fade_px, 1.0, np.clip(poly(np.clip(dist / fade_px, 0, 1)), 0, 1))
    fade = np.where(dist <= 0, 0.0, fade)
    return (1.0 - fade).astype("float32")


# ---------------------------------------------------------------- strength -> ComfyUI denoise
def strength_to_denoise(strength, shift):
    """Invoke strength is the starting sigma; ComfyUI's denoise is a position in time that the model's
    flow shift then bends. Inverse of sigma = shift*t / (1 + (shift-1)*t)."""
    s = min(max(float(strength), 0.0), 1.0)
    return min(1.0, max(0.0, s / (shift - (shift - 1.0) * s)))


# ---------------------------------------------------------------- plan (everything except the model calls)
def plan_inpaint(height, width, box=None, inpaint=None, scale="auto", edge_size=16,
                 coherence_mode="Gaussian Blur", min_denoise=0.0, mask_blur=16, crop=True):
    """Work out the crop, processing size, denoise mask and paste-back alpha.

    box:     HxW float, the generation box(es). None / empty -> the whole image. Several boxes -> one crop
             around all of them, and the result is only pasted inside the boxes.
    inpaint: HxW float, what to repaint inside the box (1 = repaint). None / empty -> the whole box
             (plain img2img on the box: no gradient edge, hard paste at the box edge, like Invoke).
    crop:    False = process the whole image (needed when the conditioning carries the full image as a
             reference, as MOSS edit does); the box then only limits where the result is pasted and, with
             no inpaint mask, is itself the area to repaint.
    Returns dict: rect, crop_size (w, h), proc_size (w, h), weight (None or proc-size denoise mask),
    alpha (crop-size paste-back alpha), masked (bool)."""
    box_hard = None
    rect = mask_rect(box) if box is not None else None
    if rect is None:
        rect = (0, 0, width, height)
    else:
        box_hard = box > 0.5
    if not crop:
        rect = (0, 0, width, height)
    x0, y0, x1, y1 = rect
    cw, ch = x1 - x0, y1 - y0
    pw, ph = processing_size(cw, ch, scale)
    box_crop = box_hard[y0:y1, x0:x1].astype("float32") if box_hard is not None else None
    # Where the result may be pasted: the box, fading out past its edge (as Invoke fades past a mask), so the
    # box is a soft suggestion: things that straddle the edge (a collar) keep their full strength inside it
    # and blend into the original just outside it.
    paste_crop = fade_alpha(box_hard, mask_blur)[y0:y1, x0:x1] if box_hard is not None else None

    m = None
    if inpaint is not None:
        m = np.clip(inpaint[y0:y1, x0:x1], 0, 1).astype("float32")
        if box_crop is not None:
            m = m * box_crop
        if m.max() <= 0.001:
            m = None
    if m is None and not crop and box_crop is not None:
        m = box_crop.copy()  # no mask painted: repaint the box

    if m is None:
        weight = None
        alpha = paste_crop if paste_crop is not None else np.ones((ch, cw), "float32")
    else:
        weight, area = gradient_mask(resize_mask(m, pw, ph), edge_size, coherence_mode, min_denoise)
        alpha = resize_mask(fade_alpha(area, mask_blur), cw, ch)
        if paste_crop is not None:
            alpha = alpha * paste_crop
    return {"rect": rect, "crop_size": (cw, ch), "proc_size": (pw, ph), "weight": weight,
            "alpha": alpha.astype("float32"), "masked": m is not None}
