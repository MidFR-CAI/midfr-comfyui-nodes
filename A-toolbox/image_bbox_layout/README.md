# Image BBox Layout (MOSS)

Category `image/layout`. Outputs: `image`, `mask`, `prompt`, `boxes`.

1. Click the upload box (or drop an image). It is resized to `max_megapixels` with sides snapped to 16, which is what MOSS-Image needs.
2. Type the whole-image description in `prompt`.
3. Drag on the image to add a box, then type that box's prompt. Drag a box to move it, drag a corner to resize it, `Delete` removes it.

`prompt` is MOSS-Image's layout prompt, the same JSON ComfyUI's built-in *MOSS-Image Layout Prompt* node makes (pixel coordinates of the resized image, origin at the canvas centre, `[ymin, xmin, ymax, xmax]`):
```json
{"summary":{"caption":"a travel poster"},
 "items":[{"label":"object","caption":"big mountain","bbox":[-192,-288,384,288]}],
 "bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}
```
**With no boxes, `prompt` is just the text you typed**, so it can go straight into the text encoder either way. Turn MOSS-Image's prompt upsampler off when using layouts (it rewrites them). Layout prompting is marked experimental by MOSS-Image ([format reference](https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md)).

## Masks
- `boxes`: the plain boxes, no grow or feather. Feed this to **Invoke Inpainting (MOSS)** as `box_mask`.
- `mask`: built from `mask_mode`: *boxes + masks* (default: boxes, image transparency and the optional `mask` input), *boxes only*, *masks only*, or *whole image*. `mask_grow` expands it, `mask_feather` softens the edge. Use it with *Set Latent Noise Mask* + **Differential Diffusion** if you build your own inpaint graph.

Tests (Pillow, numpy): `python -m unittest discover -s tests`.

The node shows only the image editor and `prompt`; `max_megapixels` and the mask settings are behind the gear button at the top right of the node.

The node shows only the image editor and `prompt`. `max_megapixels` and the mask settings open in a popup from the gear button at the top right of the node.
