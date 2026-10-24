"""AIRPG Saturation Match: fixes the over-saturation of LoRA edits without pulling the new outfit/background colours back
to the source's (Oklab chroma scaling, hue and lightness untouched)."""
try:
    from .core import auto_saturation_match, restore_neutrals, saturation_match
except ImportError:
    from core import auto_saturation_match, restore_neutrals, saturation_match


class AIRPGSaturationMatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_ref": ("IMAGE",), "image_target": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.5, "step": 0.05}),
                "measure": (["mean", "p90"], {"default": "mean"}),
                "only_reduce": ("BOOLEAN", {"default": True}),
                "max_gain": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 3.0, "step": 0.05}),
                "min_gain": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "measure_mask": ("MASK",),
                "region": (["auto", "global", "mask"], {"default": "auto", "tooltip": "auto: measure only on pixels the edit left alone; global: whole picture; mask: use measure_mask"}),
                "changed_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "auto mode: share of the correction applied to the pixels the edit changed (new outfit...)"}),
                "light_tolerance": ("FLOAT", {"default": 0.07, "min": 0.01, "max": 0.3, "step": 0.01}),
                "hue_tolerance": ("FLOAT", {"default": 28.0, "min": 5.0, "max": 90.0, "step": 1.0}),
                "restore_neutrals": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "pull colour casts back: pixels that were grey/white/black in the source and picked up a mild tint go back toward the source (strongly coloured new items are never touched)"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "MASK")
    RETURN_NAMES = ("image", "gain", "unchanged_mask")
    FUNCTION = "run"
    CATEGORY = "AIRPG/image"

    def run(self, image_ref, image_target, strength, measure, only_reduce, max_gain, min_gain, measure_mask=None,
            region="auto", changed_strength=0.5, light_tolerance=0.07, hue_tolerance=28.0, restore_neutrals=0.0):
        import torch
        if region == "auto":
            out, gain, mask, used, fell_back = auto_saturation_match(
                image_ref, image_target, strength, changed_strength, max_gain, min_gain, only_reduce,
                light_tolerance, hue_tolerance, return_info=True)
            print(f"[airpg-saturation-match] auto: chroma gain x{float(gain.mean()):.3f} measured on {float(used.mean())*100:.1f}% of pixels"
                  + (" (too few unchanged pixels: used the whole picture)" if bool(fell_back.any()) else ""))
            if restore_neutrals > 0:
                out = globals()['restore_neutrals'](image_ref, image_target, restore_neutrals, apply_to=out)
            return (out, float(gain.mean()), mask)
        m = measure_mask if region == "mask" else None
        out, gain = saturation_match(image_ref, image_target, strength, measure, max_gain, min_gain, only_reduce, m, return_gain=True)
        print(f"[airpg-saturation-match] {region}: chroma gain x{float(gain.mean()):.3f}")
        return (out, float(gain.mean()), torch.zeros(image_target.shape[:3]))


NODE_CLASS_MAPPINGS = {"AIRPGSaturationMatch": AIRPGSaturationMatch}
NODE_DISPLAY_NAME_MAPPINGS = {"AIRPGSaturationMatch": "AIRPG Saturation Match"}
