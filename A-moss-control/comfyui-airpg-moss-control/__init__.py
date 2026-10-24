"""AIRPG MOSS control adapter for ComfyUI.

Loads an adapter trained with training/train.py (one .safetensors: LoRA keys + video_embedder.ctrl_weight), patches the MOSS
DiT's embedder so the control tensor is added to the image tokens, and applies the LoRA through ComfyUI's normal patch system.
Nothing here replaces the model: strength 0 (or no inputs connected) is the plain MOSS-Image.
"""
import os

import torch

try:
    from .core import add_control, build_control, split_adapter, tokens_of
except ImportError:                                   # imported as a plain module by the unit tests
    from core import add_control, build_control, split_adapter, tokens_of


def _to_bchw(img):                                    # ComfyUI IMAGE (B,H,W,3) in [0,1] -> (B,3,H,W) in [-1,1]
    return None if img is None else img[..., :3].permute(0, 3, 1, 2).float() * 2 - 1


class AIRPGMossControl:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {
            "required": {
                "model": ("MODEL",), "vae": ("VAE",),
                "adapter": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "image": ("IMAGE",), "mask": ("MASK",), "lineart": ("IMAGE",), "depth": ("IMAGE",),
                "colour": ("IMAGE",), "pose": ("IMAGE",), "tile": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("MODEL", "LATENT")
    RETURN_NAMES = ("model", "latent")
    FUNCTION = "apply"
    CATEGORY = "AIRPG/moss"

    def apply(self, model, vae, adapter, strength, image=None, mask=None, lineart=None, depth=None, colour=None, pose=None,
              tile=None):
        import comfy.lora
        import comfy.patcher_extension as pe
        import comfy.utils
        import folder_paths
        from safetensors import safe_open

        path = folder_paths.get_full_path_or_raise("loras", adapter)
        with safe_open(path, "pt") as f:
            meta = f.metadata() or {}
        sd = comfy.utils.load_torch_file(path)
        lora_sd, emb, kinds = split_adapter(sd, meta)

        def encode(px):                                 # pixels [-1,1] (B,3,H,W) -> packed normalised latent (B,128,H/16,W/16)
            return vae.encode(((px + 1) / 2).permute(0, 2, 3, 1).clamp(0, 1)).float().cpu()

        m = None if mask is None else mask.reshape(-1, 1, *mask.shape[-2:]).float()
        control = build_control(kinds, encode, _to_bchw(image), m, lineart=_to_bchw(lineart), depth=_to_bchw(depth),
                                colour=_to_bchw(colour), pose=_to_bchw(pose), tile=_to_bchw(tile))
        tokens, ctrl_weight = tokens_of(control), emb["ctrl_weight"]

        patched = model.clone()
        key_map = comfy.lora.model_lora_keys_unet(patched.model, {})
        loaded = comfy.lora.load_lora(lora_sd, key_map)
        patched.add_patches(loaded, strength)

        def wrapper(executor, *args, **kwargs):
            net = executor.class_obj
            hook = net.video_embedder.register_forward_hook(
                lambda mod, inp, out: add_control(out, inp[0], tokens, ctrl_weight, strength))
            try:
                return executor(*args, **kwargs)
            finally:
                hook.remove()

        patched.add_wrapper_with_key(pe.WrappersMP.DIFFUSION_MODEL, "airpg_moss_control", wrapper)
        h, w = control.shape[-2:]
        return (patched, {"samples": torch.zeros(1, 128, h, w)})


NODE_CLASS_MAPPINGS = {"AIRPGMossControl": AIRPGMossControl}
NODE_DISPLAY_NAME_MAPPINGS = {"AIRPGMossControl": "AIRPG MOSS Control (adapter)"}
