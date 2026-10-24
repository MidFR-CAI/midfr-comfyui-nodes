# Image Upload + Resize

Category `image` / `image/layout`.

| Node | Does |
|---|---|
| **Simple Image Upload + Resize** | Upload one image, downscale to `max_megapixels` (sides divisible by 32; smaller images untouched). Outputs `image` and `mask`. |
| **Multi Image Upload + Resize (9)** | Drop up to nine images, resized together into one bundle. |
| **Image Set Splitter (9)** | Splits that bundle into `image_1`..`image_9`. |
| **Image BBox Layout (MOSS)** | Upload + resize, then draw boxes on the image and prompt each. Outputs MOSS-Image layout JSON. |

## Mask output (Simple Image Upload + Resize)
`mask` follows Load Image: `1 - alpha`. Use a transparent PNG, or right-click the node, *Open in MaskEditor* and paint. An image without alpha gives an all-zero mask the size of the resized image.

## Image BBox Layout (MOSS)
Outputs: `image`, `mask`, `width`, `height`, `bbox_json`.

1. Click the upload box (or drop an image). It is resized to `max_megapixels` with sides snapped to 16, which is what MOSS-Image needs.
2. Drag on the image to add a box. Click a box to pick its label (`object` / `character` / `text`), type its prompt, and for `text` boxes the exact text to render. Drag a box to move it, drag a corner to resize, `Delete` removes it.
3. `scene_prompt` is the whole-scene description.

`bbox_json` is MOSS-Image's layout prompt: coordinates are absolute pixels of the resized image, origin at the canvas centre, `[ymin, xmin, ymax, xmax]`:
```json
{"summary":{"canvas":"single","caption":"a travel poster"},
 "items":[{"label":"text","caption":"big title at the top","text":"EXPLORE","bbox":[-416,-384,-288,384]}],
 "bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}
```
Feed it to MOSS-Image's text prompt, with width/height from this node as the generation size, and turn the prompt upsampler off (it rewrites layouts). Layout prompting is marked experimental by MOSS-Image ([format reference](https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md)).

**No boxes = no output.** `bbox_json` is blocked, so nodes wired to it don't run. Turn `block_when_empty` off to get an empty string instead. Optional: `output_format` (compact / pretty), `meta` tags, `canvas`.

Tests (Pillow only): `python -m unittest discover -s tests`.
