"""Invoke Inpainting (MOSS): crop to a box, repaint inside it (optionally only inside a mask), paste back.
Settings default to InvokeAI's canvas defaults. See README.md for what is copied and what is not."""
import numpy as np
import torch
import torch.nn.functional as F

import nodes as comfy_nodes  # ComfyUI's own nodes.py (common_ksampler)

from .masks import COHERENCE_MODES, plan_inpaint, resize_mask, strength_to_denoise

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
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01,
                                       "tooltip": "Invoke's denoising strength (converted for ComfyUI's flow shift)"}),
                "edge_size": ("INT", {"default": 16, "min": 0, "max": 512, "step": 8,
                                      "tooltip": "Invoke 'edge size': how far the repaint grows past the mask"}),
                "coherence_mode": (list(COHERENCE_MODES), {"default": "Gaussian Blur"}),
                "minimum_denoise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mask_blur": ("INT", {"default": 16, "min": 0, "max": 512,
                                      "tooltip": "Invoke 'mask blur': fade of the paste-back edge in px"}),
                "scale_before_processing": (["auto", "off"], {"default": "auto",
                                            "tooltip": "auto: upscale small boxes to ~1 MP, as Invoke does"}),
                "crop_to_box": ("BOOLEAN", {"default": False, "tooltip":
                    "On = Invoke's behaviour (generate only the box region). Leave OFF when the text encoder carries the "
                    "image as a reference (MOSS edit), or the model redraws the whole picture inside the box"}),
                "context_padding": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 8, "tooltip":
                    "Only with crop_to_box: widen the crop by this many px around the box so the model sees "
                    "surrounding context. Only the box is repainted and pasted back"}),
            },
            "optional": {
                "box_mask": ("MASK", {"tooltip": "Generation box(es), e.g. the 'boxes' output of Image BBox Layout. "
                                                 "Empty = whole image"}),
                "inpaint_mask": ("MASK", {"tooltip": "What to repaint inside the box. Empty = whole box"}),
            },
        }

    def run(self, model, positive, negative, vae, image, seed, steps, cfg, strength, edge_size,
            coherence_mode, minimum_denoise, mask_blur, scale_before_processing, crop_to_box=False,
            context_padding=0, box_mask=None, inpaint_mask=None):
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
        gen = _resize(gen[:1, :, :, :3].to(image.device).float(), ch, cw)

        alpha = torch.from_numpy(np.ascontiguousarray(plan["alpha"])).float().to(image.device)
        a = alpha[None, :, :, None]
        result = image.clone()
        result[:, y0:y1, x0:x1, :3] = gen * a + crop * (1.0 - a)
        pasted = torch.zeros((1, H, W), dtype=torch.float32, device=image.device)
        pasted[0, y0:y1, x0:x1] = alpha
        return (result, pasted)


NODE_CLASS_MAPPINGS = {"InvokeInpaintingMOSS": InvokeInpaintingMOSS}
NODE_DISPLAY_NAME_MAPPINGS = {"InvokeInpaintingMOSS": "Invoke Inpainting (MOSS)"}
