# Invoke Inpainting (MOSS)

Category `image/inpaint`. Does the whole job in one node, the way InvokeAI's canvas does it:

1. **Box**: with `crop_to_box` on (Invoke's behaviour) crops to `box_mask` (several boxes: one crop around all of them; the result is pasted only inside the boxes). No box = whole image. **`crop_to_box` is off by default**: when the text encoder carries the image as a reference (MOSS edit) a cropped box makes the model redraw the whole picture inside the box. Off = the whole image is processed, the box only limits where the result is pasted and, with no `inpaint_mask`, is itself what gets repainted. Turn it on for prompt-only conditioning, which is how Invoke works. **`context_padding`** (px, default 0, only with `crop_to_box`) widens the crop around the box so the model sees surrounding context; only the box is repainted and pasted back, the padding is context only.
2. **Scale**: `scale_before_processing: auto` grows small boxes to about 1 MP (sides multiples of 16) and scales the result back.
3. **Mask**: `inpaint_mask` is what to repaint inside the box. The edit lands roughly in it: the mask is grown by `edge_size` (Gaussian falloff) and each pixel is only repainted once the noise level drops below its mask value, so the edges blend instead of showing a seam. No `inpaint_mask` = repaint the whole box (plain img2img on the box).
4. **Sample**: Euler / simple, with Invoke's `strength` converted for ComfyUI's flow shift (the same maths as *Strength to Denoise*).
5. **Paste back**: the result fades into the original over `mask_blur` px just outside the box and mask, so the box is a soft suggestion: anything that straddles its edge (a collar, a sleeve) keeps full strength up to the edge and blends out beyond it.

Inputs: `model`, `positive`, `negative`, `vae`, `image`, optional `box_mask`, `inpaint_mask`. Outputs: `image` (full image, edit pasted in) and `pasted_mask`.
Defaults are Invoke's canvas defaults: strength 0.75, edge size 16, Gaussian Blur, minimum denoise 0, mask blur 16, scale auto.

Typical wiring: *Image BBox Layout* `image` -> `image`, `boxes` -> `box_mask`, `prompt` -> your text encoder; a painted mask (Load Image / mask editor) -> `inpaint_mask`. Do not also use Set Latent Noise Mask or Differential Diffusion: the node handles both.

Not copied: Invoke's exact MOSS sigma schedule (it uses ComfyUI's `simple`), mask resizing at the 1/8 raw latent grid (ComfyUI resizes to its own latent grid), the Invoke-only "Staged" edge details beyond the ported code. Tests: `python -m unittest discover -s tests` (numpy, Pillow). The ComfyUI-side code (`nodes.py`) has not been run here.
