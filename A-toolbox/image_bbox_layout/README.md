# Image BBox Layout (MOSS)

Category `image/layout`. Outputs: `image`, `painted`, `prompt`, `boxes`.

1. Click the upload box (or drop an image). It is resized to `max_megapixels` with sides snapped to 16, which is what MOSS-Image needs.
2. Click **Edit boxes & mask** (or the picture) to open the full-screen editor, modelled on Invoke's canvas without layers. It sits outside the node graph, so it stays smooth.
   - **Box** (`C`): drag to draw a box, drag a box to move it, drag a handle to resize. Edges snap to 16 px of the output size and the box size is shown. Type each box's prompt in the side panel.
   - **Brush** (`B`) / **Eraser** (`E`): paint where the edit should happen (shown in magenta). `[` and `]` change the brush size.
   - **Pan** (`V`, hold `Space`, or middle/right-drag), mouse wheel to zoom, `0` to fit.
   - `Ctrl+Z` undo, `Ctrl+Shift+Z` / `Ctrl+Y` redo, `Delete` removes the selected box, `Esc` or **Done** closes. Changes apply live; the mask is saved when you close.
3. The whole-image description goes in the `prompt` box, hidden by default: open the ⚙ menu and tick **Show prompt box instead of boxes** to use it.

`prompt` is MOSS-Image's layout prompt, the same JSON ComfyUI's built-in *MOSS-Image Layout Prompt* node makes (pixel coordinates of the resized image, origin at the canvas centre, `[ymin, xmin, ymax, xmax]`):
```json
{"summary":{"caption":"a travel poster"},
 "items":[{"label":"object","caption":"big mountain","bbox":[-192,-288,384,288]}],
 "bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}
```
**With no boxes, `prompt` is just the text you typed**, so it can go straight into the text encoder either way. Turn MOSS-Image's prompt upsampler off when using layouts (it rewrites them). Layout prompting is marked experimental by MOSS-Image ([format reference](https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md)).

## Masks
- `boxes`: the plain boxes, no grow or feather. Feed this to **Invoke Inpainting (MOSS)** as `box_mask`.
- `painted`: only what you painted, no grow or feather. Paint it in the editor (Brush / Eraser). It is saved as a small mask file, not in the image. Feed this to **Invoke Inpainting (MOSS)** as `inpaint_mask`.
Anything transparent in the uploaded image, and the optional `mask` input, are merged into `painted`. (Older versions also had a grown/feathered `mask` output; it was removed. Old workflows still load.)

Tests (Pillow, numpy): `python -m unittest discover -s tests`.

Uploads larger than 1.5 MP are shrunk in the browser when added (the node only works at `max_megapixels` anyway, and big files are slow to edit). Transparency is kept.

The ⚙ menu (top right of the node) also has **Show prompt box instead of boxes**: it swaps the editing controls for a plain prompt box (and back); the picture stays. The box editor is the default; the choice is saved with the workflow.
