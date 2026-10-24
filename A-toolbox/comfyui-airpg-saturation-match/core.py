"""Saturation match in Oklab: scale the CHROMA of an edited picture so its overall colourfulness equals the source's, leaving
hue and lightness alone. Unlike a global colour-statistics match it does not drag the new colours (a different outfit, a new
background) back toward the old ones; it only removes the over-saturation a distilled/edit LoRA adds."""
from __future__ import annotations

import torch
import torch.nn.functional as F

_M1 = torch.tensor([[0.4122214708, 0.5363325363, 0.0514459929],
                    [0.2119034982, 0.6806995451, 0.1073969566],
                    [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = torch.tensor([[0.2104542553, 0.7936177850, -0.0040720468],
                    [1.9779984951, -2.4285922050, 0.4505937099],
                    [0.0259040371, 0.7827717662, -0.8086757660]])


def srgb_to_linear(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).clamp(min=0) ** 2.4)


def linear_to_srgb(x):
    x = x.clamp(0, 1)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def to_oklab(rgb):
    """(...,3) sRGB in [0,1] -> (...,3) L,a,b."""
    lin = srgb_to_linear(rgb)
    lms = lin @ _M1.to(rgb).T
    return torch.sign(lms) * lms.abs().pow(1 / 3) @ _M2.to(rgb).T


def from_oklab(lab):
    lms_ = lab @ torch.linalg.inv(_M2).to(lab).T
    lms = lms_ ** 3
    lin = lms @ torch.linalg.inv(_M1).to(lab).T
    return linear_to_srgb(lin)


def chroma_stat(rgb, weight=None, stat="mean", lo=0.08, hi=0.97):
    """Colourfulness of a picture: weighted mean (or a high percentile) of Oklab chroma over pixels that are neither
    near-black nor near-white (those carry no colour information and only dilute the average). rgb (B,H,W,3)."""
    lab = to_oklab(rgb)
    c = torch.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    valid = ((lab[..., 0] > lo) & (lab[..., 0] < hi)).float()
    if weight is not None:
        valid = valid * weight
    out = []
    for b in range(c.shape[0]):
        v, cb = valid[b].flatten(), c[b].flatten()
        if v.sum() < 1:
            out.append(cb.new_tensor(0.0)); continue
        if stat == "mean":
            out.append((cb * v).sum() / v.sum())
        else:                                       # p90 over valid pixels
            sel = cb[v > 0.5]
            out.append(torch.quantile(sel, 0.9) if sel.numel() else cb.new_tensor(0.0))
    return torch.stack(out)


def _fit_mask(mask, h, w, device):
    m = mask.float().to(device)
    if m.ndim == 2:
        m = m[None]
    if m.shape[-2:] != (h, w):
        m = F.interpolate(m[:, None], size=(h, w), mode="bilinear", align_corners=False)[:, 0]
    return m


def saturation_match(ref, target, strength=1.0, stat="mean", max_gain=1.5, min_gain=0.5, only_reduce=False, mask=None,
                     return_gain=False):
    """ref, target: (B,H,W,3) in [0,1] (sizes may differ). ``mask``: (B,H,W) in the TARGET's frame, 1 = measure these pixels
    (e.g. the unchanged skin/background); used for both pictures. Returns the target with its chroma scaled."""
    if ref.shape[0] != target.shape[0]:
        ref = ref[:1].expand(target.shape[0], -1, -1, -1)
    mt = None if mask is None else _fit_mask(mask, target.shape[1], target.shape[2], target.device)
    mr = None if mask is None else _fit_mask(mask, ref.shape[1], ref.shape[2], ref.device)
    s_ref = chroma_stat(ref[..., :3], mr, stat)
    s_tgt = chroma_stat(target[..., :3], mt, stat)
    gain = (s_ref / s_tgt.clamp(min=1e-6)).clamp(min_gain, 1.0 if only_reduce else max_gain)
    gain = 1.0 + (gain - 1.0) * strength
    lab = to_oklab(target[..., :3])
    lab = torch.cat([lab[..., :1], lab[..., 1:] * gain.view(-1, 1, 1, 1)], -1)
    out = from_oklab(lab)
    if target.shape[-1] > 3:
        out = torch.cat([out, target[..., 3:]], -1)
    return (out, gain) if return_gain else out


# ---------------------------------------------------------------- automatic "what stayed the same" detection
def consistent_mask(ref, target, l_tol=0.07, hue_tol_deg=28.0, neutral=0.035, pool=5):
    """Soft mask (B,H,W) of the pixels the edit left alone. ``ref`` is resized to the target's size (same framing assumed).
    A pixel is unchanged when its Oklab lightness is within ``l_tol`` and its hue within ``hue_tol_deg`` (or both are near
    neutral). Those two are what an edit does NOT touch when it merely over-saturates, and what it DOES touch when it replaces
    an outfit or redraws a mouth. Chroma is deliberately not compared: that is the thing being measured."""
    h, w = target.shape[1:3]
    r = ref[..., :3]
    if r.shape[1:3] != (h, w):
        r = F.interpolate(r.permute(0, 3, 1, 2), size=(h, w), mode="bilinear", align_corners=False, antialias=True).permute(0, 2, 3, 1)
    if r.shape[0] != target.shape[0]:
        r = r[:1].expand(target.shape[0], -1, -1, -1)
    a, b = to_oklab(r), to_oklab(target[..., :3])
    ca = torch.sqrt(a[..., 1] ** 2 + a[..., 2] ** 2)
    cb = torch.sqrt(b[..., 1] ** 2 + b[..., 2] ** 2)
    dl = (a[..., 0] - b[..., 0]).abs()
    ha, hb = torch.atan2(a[..., 2], a[..., 1]), torch.atan2(b[..., 2], b[..., 1])
    dh = torch.rad2deg(torch.atan2(torch.sin(ha - hb), torch.cos(ha - hb)).abs())
    both_neutral = (ca < neutral) & (cb < neutral)
    same = (dl < l_tol) & ((dh < hue_tol_deg) | both_neutral)
    m = same.float()[:, None]
    m = 1.0 - F.max_pool2d(1.0 - m, pool, 1, pool // 2)          # erode: drop isolated agreeing pixels at the edit's borders
    m = F.avg_pool2d(m, pool, 1, pool // 2, count_include_pad=False)  # soften
    return m[:, 0]


def auto_gain(ref, target, region, min_pixels=0.02, min_chroma=0.02):
    """Median per-pixel chroma ratio ref/target over the unchanged region. Returns (gain (B,), used_fraction (B,))."""
    h, w = target.shape[1:3]
    r = ref[..., :3]
    if r.shape[1:3] != (h, w):
        r = F.interpolate(r.permute(0, 3, 1, 2), size=(h, w), mode="bilinear", align_corners=False, antialias=True).permute(0, 2, 3, 1)
    if r.shape[0] != target.shape[0]:
        r = r[:1].expand(target.shape[0], -1, -1, -1)
    a, b = to_oklab(r), to_oklab(target[..., :3])
    ca = torch.sqrt(a[..., 1] ** 2 + a[..., 2] ** 2)
    cb = torch.sqrt(b[..., 1] ** 2 + b[..., 2] ** 2)
    gains, used = [], []
    for i in range(target.shape[0]):
        sel = (region[i] > 0.9) & (cb[i] > min_chroma) & (ca[i] > min_chroma * 0.5)
        used.append(sel.float().mean())
        if sel.float().mean() < min_pixels:
            gains.append(torch.tensor(float("nan"))); continue
        # ratio of summed chroma over the unchanged pixels (robust to a few wild per-pixel ratios, weights saturated areas)
        gains.append(ca[i][sel].sum() / cb[i][sel].sum().clamp(min=1e-6))
    return torch.stack(gains), torch.stack(used)


def auto_saturation_match(ref, target, strength=1.0, changed_strength=0.5, max_gain=1.5, min_gain=0.4, only_reduce=True,
                          l_tol=0.07, hue_tol_deg=28.0, return_info=False):
    """Measure the colourfulness change only on pixels the edit left alone (skin, background...), then scale chroma by that
    factor: fully on those pixels, ``changed_strength`` of it on the pixels the edit changed (a new dress is also usually
    over-saturated, but its true colour is unknown, so it is corrected more gently). Falls back to a global match when too few
    pixels stayed the same (a very different picture)."""
    region = consistent_mask(ref, target, l_tol, hue_tol_deg)
    gain, used = auto_gain(ref, target, region)
    fallback = torch.isnan(gain)
    if fallback.any():
        glob = chroma_stat(ref[..., :3]) / chroma_stat(target[..., :3]).clamp(min=1e-6)
        gain = torch.where(fallback, glob.expand_as(gain), gain)
    gain = gain.clamp(min_gain, 1.0 if only_reduce else max_gain)
    gain = 1.0 + (gain - 1.0) * strength
    weight = region + (1.0 - region) * changed_strength       # per-pixel share of the correction
    # Same material can sit on both sides of the mask edge (skin that was visible in the source vs skin where a blouse was), so a
    # step in strength would show as a seam. Spread the weight over a wide area so it glides instead.
    k = max(5, int(min(target.shape[1], target.shape[2]) * 0.08)) | 1
    wt = weight[:, None]
    for _ in range(3):
        wt = F.avg_pool2d(F.pad(wt, (k // 2,) * 4, mode="replicate"), k, 1)
    weight = wt[:, 0]
    g = 1.0 + (gain.view(-1, 1, 1) - 1.0) * weight
    lab = to_oklab(target[..., :3])
    lab = torch.cat([lab[..., :1], lab[..., 1:] * g[..., None]], -1)
    out = from_oklab(lab)
    if target.shape[-1] > 3:
        out = torch.cat([out, target[..., 3:]], -1)
    return (out, gain, region, used, fallback) if return_info else out


def restore_neutrals(ref, target, amount=1.0, source_neutral=0.035, edit_max=0.06, l_tol=0.12, pool=5, apply_to=None):
    """Pull back colour casts: pixels that were NEUTRAL in the source (grey wall, white/black clothing...) and are still about
    as light, but have picked up a mild tint in the edit, go back toward the source's chroma. Strongly coloured pixels
    (``edit_max``: a new dress, skin where a white blouse was) are never touched. Candidates are decided from ``target`` as the
    edit produced it; the pull-back is applied to ``apply_to`` (e.g. the same picture after a saturation correction), default
    ``target``. Returns the image."""
    h, w = target.shape[1:3]
    r = ref[..., :3]
    if r.shape[1:3] != (h, w):
        r = F.interpolate(r.permute(0, 3, 1, 2), size=(h, w), mode="bilinear", align_corners=False, antialias=True).permute(0, 2, 3, 1)
    if r.shape[0] != target.shape[0]:
        r = r[:1].expand(target.shape[0], -1, -1, -1)
    a, b = to_oklab(r), to_oklab(target[..., :3])
    ca = torch.sqrt(a[..., 1] ** 2 + a[..., 2] ** 2)
    cb = torch.sqrt(b[..., 1] ** 2 + b[..., 2] ** 2)
    ok = (ca < source_neutral) & (cb < edit_max) & (cb > ca) & ((a[..., 0] - b[..., 0]).abs() < l_tol)
    m = ok.float()[:, None]
    m = 1.0 - F.max_pool2d(1.0 - m, pool, 1, pool // 2)
    m = F.avg_pool2d(m, pool, 1, pool // 2, count_include_pad=False)[:, 0] * amount
    base = target if apply_to is None else apply_to
    bl = to_oklab(base[..., :3])
    lab = torch.cat([bl[..., :1], bl[..., 1:] * (1 - m[..., None]) + a[..., 1:] * m[..., None]], -1)
    out = from_oklab(lab)
    return torch.cat([out, base[..., 3:]], -1) if base.shape[-1] > 3 else out
