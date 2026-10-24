# Strength to Denoise (flow models)

One node, **Strength to Denoise (flow models)** (`sampling/strength`). Outputs `denoise`, `start_sigma`, `info`.

KSampler's `denoise` counts steps along the model's *shifted* schedule, so on MOSS-Image (shift 5) `denoise 0.9` starts at sigma 0.98 and keeps only ~2% of your source image. InvokeAI's img2img strength is the start sigma itself (strength 0.8 keeps 20%). This node converts the Invoke-style strength into the `denoise` KSampler needs, reading the shift from the connected model:

| strength (start sigma) | KSampler denoise (shift 5) |
|---|---|
| 0.9 | 0.64 |
| 0.8 | 0.44 |
| 0.7 | 0.32 |
| 0.5 | 0.17 |

Wire the model into `model`, set `strength`, and connect `denoise` to the KSampler's `denoise` input (hover its widget to expose the socket). Works for any flow-matching model whose shift ComfyUI knows (MOSS-Image, Flux, SD3, ...). On a non-flow model the shift is 1 and `denoise` equals `strength`.

Not replicated from Invoke: its gradient-mask promotion (soft mask edges firm up as sampling proceeds). Feather the mask yourself if you see seams.
