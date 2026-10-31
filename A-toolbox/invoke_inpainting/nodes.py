"""Invoke Inpainting (MOSS): crop to a box, repaint inside it (optionally only inside a mask), paste back.
Settings default to InvokeAI's canvas defaults. See README.md for what is copied and what is not."""
import numpy as np
import os
import torch
import torch.nn.functional as F

import nodes as comfy_nodes  # ComfyUI's own nodes.py (common_ksampler)

from .masks import COHERENCE_MODES, FILL_MODES, blank_reference, plan_inpaint, resize_mask, strength_to_denoise

SIGMA_EPS = 1e-4  # Invoke's epsilon in the mask-promotion test


def _mask_np(mask, height, width):
    if mask is None:
        return None
    a = mask.reshape(-1, mask.shape[-2], mask.shape[-1])[0].detach().float().cpu().numpy()
    return resize_mask(a, width, height) if a.shape != (height, width) else a


def _resize(img_bhwc, height, width):
    if tuple(img_bhwc.shape[1:3]) == (height, width):
        return img_bhwc
    x = img_bhwc.permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(height, width), mode="bicubic", antialias=True, align_corners=False)
    return x.clamp(0, 1).permute(0, 2, 3, 1)


def _invoke_mask_rule(sigma, denoise_mask, extra_options):
    """Invoke's per-step gradient promotion: a pixel is repainted once mask >= the NEXT sigma."""
    sigmas = extra_options["sigmas"].to(denoise_mask.device)
    cur = sigma.flatten()[0].to(denoise_mask.device)
    idx = int(torch.argmin((sigmas - cur).abs()))
    nxt = sigmas[idx + 1] if idx + 1 < len(sigmas) else torch.zeros((), device=sigmas.device)
    return (denoise_mask >= nxt + SIGMA_EPS).to(denoise_mask.dtype)


def _grow(a, r):
    """Dilate a 2-D float mask by r px."""
    t = torch.from_numpy(np.ascontiguousarray(a)).float()[None, None]
    return F.max_pool2d(t, 2 * r + 1, 1, r)[0, 0].numpy()


def _skin(x):
    """x: B,H,W,3 in 0..1 -> B,H,W float mask of skin-coloured pixels (orange-ish, moderate saturation)."""
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx, mn = x.max(-1).values, x.min(-1).values
    sat = (mx - mn) / (mx + 1e-4)
    return ((r >= g) & (g >= b) & ((r - b) > 0.06) & (sat > 0.12) & (sat < 0.75) & (b <= 0.9 * g)).float()  # b/g ~0.7 on skin, ~1 on lips


def _ss(x, lo, hi):
    return ((x - lo) / (hi - lo)).clamp(0, 1)


def _skin_soft(x):
    """Pointwise (no spatial spread) soft version of _skin: 1 on clearly skin-coloured pixels, fading to 0 over a small colour
    range, so anti-aliased pixels get a proportional share and nothing leaks onto neighbouring pixels."""
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx, mn = x.max(-1).values, x.min(-1).values
    sat = (mx - mn) / (mx + 1e-4)
    w = _ss(r - g, -0.01, 0.01) * _ss(g - b, -0.01, 0.01) * _ss(r - b, 0.03, 0.09)
    w = w * _ss(sat, 0.08, 0.16) * (1 - _ss(sat, 0.70, 0.80)) * _ss(0.9 * g - b, -0.02, 0.04)
    return w


def _soft_dil(m, r=7, s=9):
    return F.avg_pool2d(F.max_pool2d(m, r, 1, r // 2), s, 1, s // 2, count_include_pad=False).clamp(0, 1)


def _blur(t, k=49):
    for _ in range(2):
        t = F.avg_pool2d(t, k, 1, k // 2, count_include_pad=False)
    return t


def _skin_poly(gen, base, weight=None, lam=1e-2, max_px=200000, max_change=0.35):
    """gen, base: 1,H,W,3. The model redraws skin a little lighter/paler than it was (even skin the edit never meant to touch).
    Fit one pointwise colour map (2nd-order polynomial in RGB) from the result's skin to the original's, on pixels that are skin
    in BOTH pictures at the same place (arms, cheeks, hands: wherever the pose was kept) and did not change by more than
    `max_change` (a lip that replaced skin is a real change, not drift). Apply it to all the result's skin. A colour map has no
    spatial part, so it cannot blur, soften an edge or leave a seam."""
    dev = gen.device
    sel = (_skin(base)[0] > 0.5) & (_skin(gen)[0] > 0.5) & ((gen - base).abs().amax(-1)[0] < max_change)
    n = int(sel.sum())
    if n < 2000:
        return gen

    def feats(x):
        r, g, b = x[:, 0], x[:, 1], x[:, 2]
        return torch.stack([torch.ones_like(r), r, g, b, r * r, g * g, b * b, r * g, r * b, g * b], 1)

    xs, ys = gen[0][sel].double(), base[0][sel].double()
    step = max(1, n // max_px)
    xs, ys = xs[::step], ys[::step]
    X = feats(xs)
    A = X.T @ X + lam * torch.eye(X.shape[1], dtype=X.dtype, device=dev)
    Wm = torch.linalg.solve(A, X.T @ ys)
    flat = gen[0].reshape(-1, 3).double()
    out = (feats(flat) @ Wm).reshape(gen[0].shape).float().clamp(0, 1)
    soft = _skin_soft(gen)[0][..., None]
    return (gen[0] + (out - gen[0]) * soft).clamp(0, 1)[None]


def _seam_match(gen, crop, alpha):
    """Mild tone drift of repainted areas that should look like what was there (background, skin, an arm): measure the
    difference between the original and the result along the inside of the repaint edge, spread it inwards a little and
    take it out. Big differences (a new white dress over purple) are not drift and are left alone, as are pixels whose
    colour is unlike the edge's. gen, crop: 1,H,W,3; alpha: H,W."""
    a = (alpha > 0.5).float()[None, None]
    band = a * F.max_pool2d(1 - a, 25, 1, 12)
    d = (crop - gen).permute(0, 3, 1, 2)
    gb = band * (d.abs().amax(1, keepdim=True) < 0.18).float()
    if float(gb.sum()) < 200:
        return gen
    den = _blur(gb, 81)
    field = _blur(d * gb, 81) / (den + 1e-4)
    g = gen.permute(0, 3, 1, 2)
    lpg = _blur(g * gb, 81) / (den + 1e-4)
    sim = ((g - lpg).abs().amax(1, keepdim=True) < 0.15).float()
    reach = _blur(F.max_pool2d(gb, 97, 1, 48), 31)  # the correction fades out about 48px from the edge it was measured on
    w = reach.clamp(0, 1) * sim * a
    return (g + field * w).clamp(0, 1).permute(0, 2, 3, 1)


def _scene_text(text):
    """The bbox node's 'prompt' output is the layout JSON of ALL boxes once any box exists. Per pass only this box's own
    caption may be in the prompt (otherwise one box's edit leaks into another), so keep just the scene sentence."""
    import json

    text = (text or "").strip()
    if text.startswith("{"):
        try:
            doc = json.loads(text)
            return str((doc.get("summary") or {}).get("caption") or "").strip()
        except (TypeError, ValueError, AttributeError):
            pass
    return text


def _layout_prompt(caption, summary, x0, y0, x1, y1, width, height):
    """MOSS-Image layout JSON for one box (pixels, origin at the picture centre), same format as Image BBox Layout."""
    import json

    doc = {"summary": {"caption": summary}, "items": [{"label": "object", "caption": caption or "",
           "bbox": [y0 - height // 2, x0 - width // 2, y1 - height // 2, x1 - width // 2]}],
           "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}
    return json.dumps(doc, ensure_ascii=False, separators=(",", ":"))


class InvokeInpaintingMOSS:
    CATEGORY = "image/inpaint"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "pasted_mask")
    DESCRIPTION = (
        "InvokeAI-style inpainting for MOSS-Image. box_mask = where the edit may happen (crop + paste-back); "
        "inpaint_mask = what to repaint inside the box (roughly: edges blend). No inpaint_mask repaints the "
        "whole box. Settings default to Invoke's."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 30.0, "step": 0.1}),
                "strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01,
                                       "tooltip": "Invoke's denoising strength (converted for ComfyUI's flow shift)"}),
                "edge_size": ("INT", {"default": 16, "min": 0, "max": 512, "step": 8,
                                      "tooltip": "Invoke 'edge size': how far the repaint grows past the mask"}),
                "coherence_mode": (list(COHERENCE_MODES), {"default": "Gaussian Blur"}),
                "minimum_denoise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 512,
                                      "tooltip": "Invoke 'mask blur': fade of the paste-back edge in px"}),
                "scale_before_processing": (["auto", "off"], {"default": "auto",
                                            "tooltip": "auto: upscale small boxes to ~1 MP, as Invoke does"}),
                "crop_to_box": ("BOOLEAN", {"default": False, "tooltip":
                    "On = Invoke's behaviour (generate only the box region). Leave OFF when the text encoder carries the "
                    "image as a reference (MOSS edit), or the model redraws the whole picture inside the box"}),
                "context_padding": ("INT", {"default": -1, "min": -1, "max": 2048, "step": 1, "tooltip":
                    "-1 = auto. With crop_to_box, or when a small box is enlarged (box_working_mp): widen the crop by this many px around "
                    "the box so the model sees surrounding context. Only the box is repainted and pasted back"}),
            },
            "optional": {
                "positive": ("CONDITIONING", {"tooltip": "Not needed when clip + passes are connected (per-box mode)"}),
                "negative": ("CONDITIONING",),
                "box_mask": ("MASK", {"tooltip": "Generation box(es), e.g. the 'boxes' output of Image BBox Layout. "
                                                 "Empty = whole image"}),
                "inpaint_mask": ("MASK", {"tooltip": "What to repaint inside the box. Empty = whole box"}),
                # Per-box mode: connect clip and the bbox node's 'passes'. Each box then gets its own pass, prompt and
                # strength (set in the box editor; blank = this node's strength), one after the other on the result so far.
                "clip": ("CLIP", {"tooltip": "With passes: encodes each box's prompt itself (per-box mode)"}),
                "passes": ("STRING", {"forceInput": True, "tooltip": "The 'passes' output of Image BBox Layout"}),
                "edit_format": (list(EDIT_FORMATS), {"default": "ai-toolkit", "tooltip": "Per-box mode: the layout your edit LoRA was trained with"}),
                "hide_edit_area": (FILL_MODES, {"default": "keep", "tooltip": "Per-box mode: how the box is hidden in the reference picture"}),
                "expand": ("INT", {"default": 16, "min": 0, "max": 256, "step": 8, "tooltip": "Per-box mode: grow the hidden area by this many px"}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Per-box mode: empty = zeroed conditioning"}),
                "prompt": ("STRING", {"default": "", "multiline": True, "tooltip":
                    "Per-box mode: the description of the whole picture (connect the bbox node's 'prompt' output, or type it). "
                    "Used as the scene description for every box. With no boxes, this is the prompt for one whole-picture pass"}),
                "prompt_style": (["instruction", "layout"], {"default": "instruction", "tooltip":
                    "Per-box mode. layout: MOSS-Image layout JSON (box position + caption). instruction: just the box's text "
                    "(plus the scene text), like a normal edit instruction - try this for edit LoRAs"}),
                "edge_restore": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 0.95, "step": 0.05, "tooltip":
                    "Per-box mode: keep the original picture in cells that are only partly inside the repaint mask "
                    "(removes the lighter blocky patches along the mask edge). Higher = more of the edge kept original. 0 = off"}),
                "skin_match": ("BOOLEAN", {"default": True, "tooltip":
                    "Per-box mode: where both the original and the result are skin, bring the repainted skin's tone back to the original's (lighter patches, e.g. inside a new neckline)"}),
                "tone_fix": ("BOOLEAN", {"default": False, "tooltip":
                    "Remove the colour shift the VAE round trip adds to the parts of the picture that were not repainted (blocky lighter patches next to the edit)"}),
                "box_working_mp": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 16.0, "step": 0.1, "tooltip":
                    "-1 = auto: a small box (under ~0.06 MP: headwear, mouth, eyes) is cropped out with its surroundings (a window 3x the box) and enlarged to ~1 MP; a bigger one "
                    "(shoes, outfits) is edited inside the whole picture, which keeps pose and surroundings. "
                    "Otherwise: a box (plus context_padding) smaller than this many megapixels is cropped out, enlarged to about this "
                    "size, edited there, scaled back down and blended in. 0 = never"}),
                "seam_match": ("BOOLEAN", {"default": True, "tooltip":
                    "Per-box mode: remove mild tone drift along the inside of the repaint edge (background, skin, a hand) so the edit leaves no visible patch outline"}),
                "preserve_unchanged": ("BOOLEAN", {"default": False, "tooltip":
                    "Per-box mode: after colour correction, wherever the result is still the same as the original at a coarse scale "
                    "(a hand, the wall, skin the edit did not touch), keep the original pixels. Only what really changed is pasted. No prompting needed"}),
                "hide_mask": ("MASK", {"tooltip":
                    "Optional (Edit Mask's `hide` output): items the edit removes or replaces (a helmet, a sword) are blurred out of the reference "
                    "picture only, so the model does not copy them. The rest of the reference stays as `hide_edit_area` says"}),
                "context": ("STRING", {"forceInput": True, "tooltip":
                    "Per-box mode: a 'keep these unchanged' line (from Scene Context (Qwen VL)) added after each box's instruction, "
                    "so skin tone, lips, eye colour etc. are stated instead of left for the model to guess"}),
            },
        }

    def run(self, model, vae, image, seed, steps, cfg, strength, edge_size,
            coherence_mode, minimum_denoise, mask_blur, scale_before_processing, crop_to_box=False,
            context_padding=-1, positive=None, negative=None, box_mask=None, inpaint_mask=None,
            clip=None, passes=None, edit_format="ai-toolkit", hide_edit_area="keep", expand=16, negative_prompt="", box_working_mp=-1.0, prompt="", prompt_style="instruction", tone_fix=False, edge_restore=0.0, skin_match=True, seam_match=True, preserve_unchanged=False, context="", hide_mask=None):
        if clip is not None and (passes or (prompt or "").strip()):
            return self.run_passes(model, clip, vae, image, passes, seed, steps, cfg, strength, edge_size, coherence_mode,
                                   minimum_denoise, mask_blur, scale_before_processing, inpaint_mask, edit_format,
                                   hide_edit_area, expand, negative_prompt, box_working_mp, context_padding, prompt, prompt_style, tone_fix, edge_restore, skin_match, seam_match, preserve_unchanged, context, hide_mask)
        if positive is None or negative is None:
            raise ValueError("Connect positive and negative, or clip + passes for per-box mode.")
        return self.run_one(model, positive, negative, vae, image, seed, steps, cfg, strength, edge_size,
                            coherence_mode, minimum_denoise, mask_blur, scale_before_processing, crop_to_box,
                            max(0, context_padding), box_mask, inpaint_mask)

    def run_passes(self, model, clip, vae, image, passes, seed, steps, cfg, strength, edge_size, coherence_mode,
                   minimum_denoise, mask_blur, scale, inpaint_mask, edit_format, hide_edit_area, expand, negative_prompt,
                   box_working_mp=-1.0, context_padding=-1, scene="", prompt_style="instruction", tone_fix=False, edge_restore=0.0, skin_match=True, seam_match=True, preserve_unchanged=False, context="", hide_mask=None):
        import json

        try:
            todo = json.loads(passes) if passes else []
        except (TypeError, ValueError):
            todo = []
        enc = TextEncodeMOSSInpaint()
        cur = image[:1, :, :, :3]
        _, H, W, _ = cur.shape
        scene = _scene_text(scene)
        if not todo:
            if not scene:
                raise ValueError("per-box mode: no boxes and no prompt. Draw a box in Image BBox Layout or fill in prompt.")
            todo = [{"rect": [0, 0, W, H], "prompt": scene, "strength": None}]  # whole picture, plain prompt
        inp = _mask_np(inpaint_mask, H, W)
        hide = _mask_np(hide_mask, H, W)
        union = torch.zeros((1, H, W), dtype=torch.float32, device=cur.device)
        for p in todo:
            x0, y0, x1, y1 = (int(v) for v in p["rect"])
            st = p.get("strength")
            st = strength if st is None else float(st)
            if "caption" in p:  # layout prompt for the full picture, with this node's scene text when given
                p = dict(p, prompt=_layout_prompt(p["caption"], " ".join(t for t in (scene or p.get("summary", ""), p.get("context") or context) if t), x0, y0, x1, y1, W, H),
                         summary=scene or p.get("summary", ""))
                if prompt_style == "instruction":
                    p["prompt"] = " ".join(t for t in (scene or p.get("summary", ""), p["caption"], p.get("context") or context) if t).strip()
            # auto (-1), from the tests: a small box (headwear, mouth, eyes: under ~0.06 MP) is edited on its own enlarged crop, where
            # the model sees only that part (in the whole picture it redraws the head around it); a bigger box (shoes, outfits) is
            # edited inside the whole picture, which keeps pose and surroundings. Tiny boxes get more surrounding context.
            # The user's own numbers win
            box_area = (x1 - x0) * (y1 - y0)
            eff_mp = box_working_mp
            if box_working_mp < 0:
                eff_mp = 1.0 if box_area < 0.06e6 else 0.0
            # the window the model sees is the box plus as much context again on every side (3x the box), like cropping a ~1 MP
            # window out of an upscaled picture: the face/hair/body around the box is in view, and the box is at a size the model draws well
            eff_pad = max(32, max(x1 - x0, y1 - y0)) if context_padding < 0 else context_padding
            if eff_mp > 0:
                px0, py0 = max(0, x0 - eff_pad), max(0, y0 - eff_pad)
                px1, py1 = min(W, x1 + eff_pad), min(H, y1 + eff_pad)
                cw, ch = px1 - px0, py1 - py0
                if cw * ch < 0.9 * eff_mp * 1e6:  # small box: work on an enlarged crop
                    k = (eff_mp * 1e6 / (cw * ch)) ** 0.5
                    sw, sh = max(16, round(cw * k / 16) * 16), max(16, round(ch * k / 16) * 16)
                    fx, fy = sw / cw, sh / ch
                    crop = cur[:, py0:py1, px0:px1, :]
                    sub = _resize(crop, sh, sw)
                    bx0, bx1 = int((x0 - px0) * fx), min(sw, int(np.ceil((x1 - px0) * fx)))
                    by0, by1 = int((y0 - py0) * fy), min(sh, int(np.ceil((y1 - py0) * fy)))
                    box = np.zeros((sh, sw), dtype=np.float32)
                    box[by0:by1, bx0:bx1] = 1.0
                    box_t = torch.from_numpy(box)[None]
                    inp_t = None
                    if inp is not None:
                        ic = resize_mask(inp[py0:py1, px0:px1], sw, sh)
                        ic = (_grow(ic, 8) if edge_restore > 0 else ic) * box
                        if float(ic.max()) > 0.001:
                            inp_t = torch.from_numpy(np.ascontiguousarray(ic)).float()[None]
                    # the layout prompt's coordinates are relative to the picture the model sees: redo them for the crop
                    sub_prompt = _layout_prompt(p.get("caption"), p.get("summary", ""), bx0, by0, bx1, by1, sw, sh) \
                        if ("caption" in p and prompt_style == "layout") else (p.get("prompt") or "")
                    hide_c = resize_mask(hide[py0:py1, px0:px1], sw, sh) if hide is not None else None
                    pos, neg = self._encode_pair(enc, clip, vae, edit_format, sub, hide_edit_area, expand, box_t, inp_t,
                                                 sub_prompt, negative_prompt, hide_c)
                    res, pasted = self.run_one(model, pos, neg, vae, sub, seed, steps, cfg, st, edge_size, coherence_mode,
                                               minimum_denoise, mask_blur, "off", False, 0, box_t, inp_t, tone_fix, edge_restore, skin_match, seam_match, preserve_unchanged)
                    gen = _resize(res.to(cur.device).float(), ch, cw)
                    a = _resize(pasted.to(cur.device).float()[..., None], ch, cw)
                    cur = cur.clone()
                    cur[:, py0:py1, px0:px1, :] = gen * a + crop * (1.0 - a)
                    union[:, py0:py1, px0:px1] = torch.maximum(union[:, py0:py1, px0:px1], a[..., 0])
                    continue
            box = np.zeros((H, W), dtype=np.float32)
            box[y0:y1, x0:x1] = 1.0
            box_t = torch.from_numpy(box)[None]
            inp_t = None
            if inp is not None and float((inp * box).max()) > 0.001:  # only paint inside this box
                grown = _grow(inp, 8) if edge_restore > 0 else inp  # edge_restore trims about one cell off the edge
                inp_t = torch.from_numpy(np.ascontiguousarray(grown * box)).float()[None]
            pos, neg = self._encode_pair(enc, clip, vae, edit_format, cur, hide_edit_area, expand, box_t, inp_t,
                                         p.get("prompt") or "", negative_prompt, hide * box if hide is not None else None)
            cur, pasted = self.run_one(model, pos, neg, vae, cur, seed, steps, cfg, st, edge_size, coherence_mode,
                                       minimum_denoise, mask_blur, scale, False, 0, box_t, inp_t, tone_fix, edge_restore, skin_match, seam_match, preserve_unchanged)
            union = torch.maximum(union, pasted.to(union.device))
        return (cur, union)

    @staticmethod
    def _encode_pair(enc, clip, vae, edit_format, image, hide_edit_area, expand, box_t, inp_t, prompt, negative_prompt, hide=None):
        if hide is not None and float(np.max(hide)) > 0.001:  # items the edit removes: blurred out of the reference only
            arr = image[0].detach().float().cpu().numpy()
            image = torch.from_numpy(np.ascontiguousarray(blank_reference(arr, hide, "blurred", int(expand)))).to(image.device, image.dtype)[None]
        args = dict(clip=clip, vae=vae, edit_format=edit_format, image=image, hide_edit_area=hide_edit_area,
                    expand=expand, box_mask=box_t, inpaint_mask=inp_t)
        pos = enc.encode(prompt=prompt, **args)[0]
        neg = enc.encode(prompt=negative_prompt or "", **args)[0]
        if not (negative_prompt or "").strip():  # same as ConditioningZeroOut
            zeroed = []
            for t in neg:
                d = t[1].copy()
                if d.get("pooled_output") is not None:
                    d["pooled_output"] = torch.zeros_like(d["pooled_output"])
                zeroed.append([torch.zeros_like(t[0]), d])
            neg = zeroed
        return pos, neg

    def run_one(self, model, positive, negative, vae, image, seed, steps, cfg, strength, edge_size,
                coherence_mode, minimum_denoise, mask_blur, scale_before_processing, crop_to_box=False,
                context_padding=0, box_mask=None, inpaint_mask=None, tone_fix=False, edge_restore=0.0, skin_match=True, seam_match=True, preserve_unchanged=False):
        image = image[:1]
        _, H, W, _ = image.shape
        plan = plan_inpaint(H, W, _mask_np(box_mask, H, W), _mask_np(inpaint_mask, H, W),
                            scale_before_processing, edge_size, coherence_mode, minimum_denoise, mask_blur, crop_to_box,
                            context_padding)
        x0, y0, x1, y1 = plan["rect"]
        cw, ch = plan["crop_size"]
        pw, ph = plan["proc_size"]

        crop = image[:, y0:y1, x0:x1, :3]
        proc = _resize(crop, ph, pw)
        latent = {"samples": vae.encode(proc[:, :, :, :3])}

        m = model.clone()
        if plan["weight"] is not None:
            weight = torch.from_numpy(np.ascontiguousarray(plan["weight"])).float()
            latent["noise_mask"] = weight.reshape(1, 1, ph, pw)
            m.set_model_denoise_mask_function(_invoke_mask_rule)

        shift = getattr(m.get_model_object("model_sampling"), "shift", 1.0)
        denoise = strength_to_denoise(strength, float(shift))
        out = comfy_nodes.common_ksampler(m, seed, steps, cfg, "euler", "simple", positive, negative,
                                          latent, denoise=denoise)[0]

        gen = vae.decode(out["samples"])
        if gen.dim() == 5:
            gen = gen.reshape(-1, *gen.shape[-3:])
        gen = gen[:1, :, :, :3].to(image.device).float()
        if tone_fix:
            # cells the sampler left alone come back through the VAE (encode + decode), which shifts colour a little and
            # shows up as lighter blocks next to the edit. Measure that shift on the untouched picture and take it out.
            rt = vae.decode(latent["samples"])
            if rt.dim() == 5:
                rt = rt.reshape(-1, *rt.shape[-3:])
            rt = _resize(rt[:1, :, :, :3].to(image.device).float(), gen.shape[1], gen.shape[2])
            base = _resize(proc[:, :, :, :3].to(image.device).float(), gen.shape[1], gen.shape[2])
            d = (base - rt).permute(0, 3, 1, 2)
            for _ in range(2):
                d = F.avg_pool2d(d, 33, 1, 16, count_include_pad=False)
            gen = (gen + d.permute(0, 2, 3, 1)).clamp(0, 1)
        if skin_match:
            # skin the model repainted lighter/darker than it was (e.g. cleavage inside a new neckline): where the original
            # AND the result are both skin-coloured, bring the result's low-frequency tone back to the original's.
            base = _resize(proc[:, :, :, :3].to(image.device).float(), gen.shape[1], gen.shape[2])
            gen = _skin_poly(gen, base, plan["weight"])
        gen = _resize(gen, ch, cw)

        alpha = torch.from_numpy(np.ascontiguousarray(plan["alpha"])).float().to(image.device)
        if edge_restore > 0 and plan["weight"] is not None:
            # Cells only partly inside the repaint mask are denoised for just the last few steps; the model shifts their
            # tone (lighter, washed blocks along the mask edge). Show the original there instead: the more a cell is
            # repainted, the more of the result is used (nothing below `edge_restore`, everything at full weight).
            wt = torch.from_numpy(np.ascontiguousarray(plan["weight"])).float()[None, None]
            wt = F.interpolate(wt, size=(max(1, ph // 16), max(1, pw // 16)), mode="bilinear", align_corners=False)
            wt = F.interpolate(wt, size=(alpha.shape[0], alpha.shape[1]), mode="bilinear", align_corners=False)[0, 0]
            alpha = alpha * ((wt.to(alpha.device) - edge_restore) / (1.0 - edge_restore)).clamp(0, 1)
            # no half-transparent band: a pixel is the original or the result (a see-through band shows as a ghost of the
            # new item beside it); the edge stays about mask_blur wide
            alpha = ((alpha - 0.3) / 0.4).clamp(0, 1)
            # ...but where the result clearly differs from the original (old outfit turned into the new one) it is a real
            # change: keep it, so no sliver of the old item survives along the edge
            diff = (gen - crop[:, :, :, :3].float()).abs().amax(-1)[0]
            real = (diff > 0.22).float()[None, None]
            real = F.avg_pool2d(F.max_pool2d(real, 5, 1, 2), 5, 1, 2)[0, 0]
            reach = (torch.from_numpy(np.ascontiguousarray(plan["alpha"])).float().to(alpha.device) > 0.5).float()[None, None]
            reach = F.max_pool2d(reach, 13, 1, 6)[0, 0]  # the old item's outline sits a pixel or two outside the painted mask
            alpha = torch.maximum(alpha, real * reach)
        if preserve_unchanged:
            # what the model merely redrew (a hand, wall, plain skin: slightly blurrier, slightly different) is the original's
            # content, not an edit: judge by tone/colour at a coarse scale, so only real changes (a new dress, an open
            # mouth, a moved brow) are pasted and the rest stays the crisp original
            gl = _blur(gen.permute(0, 3, 1, 2), 9)
            cl = _blur(crop[:, :, :, :3].float().permute(0, 3, 1, 2), 9)
            dd = (gl - cl).abs().amax(1, keepdim=True)
            chg = ((dd - 0.08) / 0.10).clamp(0, 1)
            closed = -F.max_pool2d(-F.max_pool2d((dd > 0.10).float(), 31, 1, 15), 31, 1, 15)  # fill holes up to ~30px: a subtly
            # different patch (grey wall turning white) inside a changed object is part of the change
            keep = _soft_dil(torch.maximum(chg, closed), 9, 9)[0, 0]
            alpha = alpha * keep
        if seam_match:
            gen = _seam_match(gen, crop[:, :, :, :3].float(), alpha)
        a = alpha[None, :, :, None]
        result = image.clone()
        result[:, y0:y1, x0:x1, :3] = gen * a + crop * (1.0 - a)
        pasted = torch.zeros((1, H, W), dtype=torch.float32, device=image.device)
        pasted[0, y0:y1, x0:x1] = alpha
        return (result, pasted)


try:  # the edit-LoRA layouts of ComfyUI's own MOSS-Image Edit encoder
    from comfy_extras.nodes_moss_image import EDIT_FORMATS
except Exception:  # older / newer ComfyUI: same values as when this was written
    EDIT_FORMATS = {
        "ai-toolkit": ("index_timestep_zero_isolated", True),
        "ai-toolkit (kv_cache off)": ("index_timestep_zero", True),
        "Moss-Image-Edit": ("index_timestep_zero_text_start", False),
    }


class TextEncodeMOSSInpaint:
    """ComfyUI's 'Text Encode MOSS-Image Edit', plus: the part of the picture that is about to be repainted is hidden
    from the reference (in what the text encoder sees AND in the reference latents). With the stock encoder the model is
    shown the old picture as the thing to copy, so below strength 1 (which starts from that same picture) nothing
    changes. Hiding the edit area lets it be changed while the surroundings stay the reference."""
    CATEGORY = "conditioning/moss_image"
    FUNCTION = "encode"
    RETURN_TYPES = ("CONDITIONING",)
    DESCRIPTION = ("Text Encode MOSS-Image Edit with the edit area hidden from the reference picture. Use it for the "
                   "positive and the negative prompt, with the same box_mask / inpaint_mask as Invoke Inpainting (MOSS).")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamic_prompts": True}),
                "vae": ("VAE",),
                "edit_format": (list(EDIT_FORMATS), {"tooltip": "The layout your edit LoRA was trained with (same as the stock node)"}),
                "image": ("IMAGE", {"tooltip": "The picture being edited (already at the size you want to work at)"}),
                "hide_edit_area": (FILL_MODES, {"default": "grey", "tooltip":
                    "grey: replace the edit area in the reference with flat grey. blurred: a very blurry copy (keeps rough "
                    "colours/layout). keep: behave like the stock node"}),
                "expand": ("INT", {"default": 16, "min": 0, "max": 256, "step": 8, "tooltip":
                    "Grow the hidden area (and feather its edge) by this many px, so the edit has room to land"}),
            },
            "optional": {
                "box_mask": ("MASK", {"tooltip": "Same as Invoke Inpainting's box_mask. Empty = whole image"}),
                "inpaint_mask": ("MASK", {"tooltip": "Same as Invoke Inpainting's inpaint_mask. Empty = the whole box"}),
            },
        }

    def encode(self, clip, prompt, vae, edit_format, image, hide_edit_area="grey", expand=16, box_mask=None, inpaint_mask=None):
        import math

        import comfy.utils
        import node_helpers

        method, vision = EDIT_FORMATS[edit_format]
        img = image[:1, :, :, :3]
        _, H, W, _ = img.shape
        box, inp = _mask_np(box_mask, H, W), _mask_np(inpaint_mask, H, W)
        area = None
        if inp is not None and float(inp.max()) > 0.001:
            area = inp * (box > 0.5) if (box is not None and float(box.max()) > 0.5) else inp
        elif box is not None and float(box.max()) > 0.001:
            area = box
        if area is not None and hide_edit_area != "keep":
            arr = img[0].detach().float().cpu().numpy()
            img = torch.from_numpy(np.ascontiguousarray(blank_reference(arr, area, hide_edit_area, int(expand)))).to(image.device, image.dtype)[None]

        vl = []
        if vision:
            scale = min(1.0, math.sqrt(384 * 384 / (H * W)))
            vl.append(comfy.utils.common_upscale(img.movedim(-1, 1), round(W * scale), round(H * scale), "area", "disabled").movedim(1, -1))
            prompt = "".join("Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1) for i in range(len(vl))) + prompt
        conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(prompt, images=vl))
        refs = [vae.encode(img)]
        return (node_helpers.conditioning_set_values(conditioning, {"reference_latents": refs, "reference_latents_method": method}),)


NODE_CLASS_MAPPINGS = {"InvokeInpaintingMOSS": InvokeInpaintingMOSS, "TextEncodeMOSSInpaint": TextEncodeMOSSInpaint}
NODE_DISPLAY_NAME_MAPPINGS = {"InvokeInpaintingMOSS": "Invoke Inpainting (MOSS)",
                              "TextEncodeMOSSInpaint": "Text Encode MOSS-Image Edit (masked reference)"}
