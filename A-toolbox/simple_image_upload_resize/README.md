# Image Upload + Resize

Category `image` / `image/layout`.

| Node | Does |
|---|---|
| **Simple Image Upload + Resize** | Upload one image, downscale to `max_megapixels` (sides divisible by 32; smaller images untouched). Outputs `image` and `mask`. |
| **Multi Image Upload + Resize (9)** | Drop up to nine images, resized together into one bundle. |
| **Image Set Splitter (9)** | Splits that bundle into `image_1`..`image_9`. |
| **Image BBox Layout (MOSS)** | Upload + resize, then draw boxes on the image and prompt each. Outputs a MOSS-Image layout prompt. |

## Mask output (Simple Image Upload + Resize)
`mask` follows Load Image: `1 - alpha`. Use a transparent PNG, or right-click the node, *Open in MaskEditor* and paint. An image without alpha gives an all-zero mask the size of the resized image.

## Image BBox Layout (MOSS)
Outputs: `image`, `mask`, `prompt`.

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

Tests (Pillow only): `python -m unittest discover -s tests`.
