from .math_utils import shift_sigma, strength_to_denoise


class FlowStrengthToDenoise:
    """InvokeAI-style img2img strength -> the `denoise` value KSampler needs on a flow model (MOSS-Image, Flux, ...)."""

    CATEGORY = "sampling/strength"
    RETURN_TYPES = ("FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("denoise", "start_sigma", "info")
    FUNCTION = "convert"
    DESCRIPTION = (
        "KSampler's denoise counts steps along a shifted schedule, so 0.9 keeps almost none of the source "
        "(MOSS-Image, shift 5: ~2%). Strength here is the sigma sampling starts at (InvokeAI's img2img strength): "
        "0.8 keeps 20% of the source image. Wire `denoise` into KSampler."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01,
                                       "tooltip": "Start sigma: 1.0 = redraw everything, 0.5 = half the source kept, lower = closer to the source"}),
            },
        }

    def convert(self, model, strength):
        shift = 1.0
        try:
            ms = model.get_model_object("model_sampling")
            shift = float(getattr(ms, "shift", 1.0))
        except Exception:
            pass
        denoise = strength_to_denoise(strength, shift)
        start = shift_sigma(denoise, shift)
        info = f"strength {strength:.2f} -> denoise {denoise:.3f} (start sigma {start:.3f}, model shift {shift:g})"
        return (denoise, start, info)


NODE_CLASS_MAPPINGS = {"FlowStrengthToDenoise": FlowStrengthToDenoise}
NODE_DISPLAY_NAME_MAPPINGS = {"FlowStrengthToDenoise": "Strength to Denoise (flow models)"}
