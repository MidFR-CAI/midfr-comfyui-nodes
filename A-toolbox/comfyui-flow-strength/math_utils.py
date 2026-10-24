"""Strength <-> KSampler denoise for flow-matching models (ComfyUI-free, so it can be unit-tested)."""


def shift_sigma(t, shift):
    """Sigma at schedule position t (0..1) for a flow model with the given shift (ComfyUI's time_snr_shift)."""
    return t if shift == 1.0 else shift * t / (1.0 + (shift - 1.0) * t)


def strength_to_denoise(strength, shift):
    """Denoise to give KSampler so that sampling STARTS at sigma == strength.

    InvokeAI measures img2img strength directly in sigma: strength 0.8 starts at sigma 0.8, keeping 20% of the
    source latent. KSampler's `denoise` instead counts steps along the shifted schedule, where position t has
    sigma = shift*t / (1 + (shift-1)*t). Inverting that gives t = s / (shift - (shift-1)*s).
    """
    s = min(1.0, max(0.0, float(strength)))
    if s == 0.0:
        return 0.0
    return min(1.0, s / (shift - (shift - 1.0) * s))
