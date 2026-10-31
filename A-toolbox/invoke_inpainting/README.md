# Invoke Inpainting (MOSS)

Category `image/inpaint`. Does the whole job in one node, the way InvokeAI's canvas does it:

1. **Box**: with `crop_to_box` on (Invoke's behaviour) crops to `box_mask` (several boxes: one crop around all of them; the result is pasted only inside the boxes). No box = whole image. **`crop_to_box` is off by default**: when the text encoder carries the image as a reference (MOSS edit) a cropped box makes the model redraw the whole picture inside the box. Off = the whole image is processed, the box only limits where the result is pasted and, with no `inpaint_mask`, is itself what gets repainted. Turn it on for prompt-only conditioning, which is how Invoke works. **`context_padding`** (px, default 0, only with `crop_to_box`) widens the crop around the box so the model sees surrounding context; only the box is repainted and pasted back, the padding is context only.
2. **Scale**: `scale_before_processing: auto` grows small boxes to about 1 MP (sides multiples of 16) and scales the result back.
3. **Mask**: `inpaint_mask` is what to repaint inside the box. The edit lands roughly in it: the mask is grown by `edge_size` (Gaussian falloff) and each pixel is only repainted once the noise level drops below its mask value, so the edges blend instead of showing a seam. No `inpaint_mask` = repaint the whole box (plain img2img on the box).
4. **Sample**: Euler / simple, with Invoke's `strength` converted for ComfyUI's flow shift (the same maths as *Strength to Denoise*).
5. **Paste back**: the result fades into the original over `mask_blur` px just outside the box and mask, so the box is a soft suggestion: anything that straddles its edge (a collar, a sleeve) keeps full strength up to the edge and blends out beyond it.

Inputs: `model`, `positive`, `negative`, `vae`, `image`, optional `box_mask`, `inpaint_mask`. Outputs: `image` (full image, edit pasted in) and `pasted_mask`.
Defaults are Invoke's canvas defaults: strength 0.75, edge size 16, Gaussian Blur, minimum denoise 0, mask blur 16, scale auto.

Typical wiring: *Image BBox Layout* `image` -> `image`, `boxes` -> `box_mask`, `prompt` -> your text encoder; `painted` -> `inpaint_mask` (paint it in the bbox node's editor). Do not also use Set Latent Noise Mask or Differential Diffusion: the node handles both.

Not copied: Invoke's exact MOSS sigma schedule (it uses ComfyUI's `simple`), mask resizing at the 1/8 raw latent grid (ComfyUI resizes to its own latent grid), the Invoke-only "Staged" edge details beyond the ported code. Tests: `python -m unittest discover -s tests` (numpy, Pillow). The ComfyUI-side code (`nodes.py`) has not been run here.

## Strength below 1 and edit encoders: Text Encode MOSS-Image Edit (masked reference)
With ComfyUI's *Text Encode MOSS-Image Edit* the whole picture is the reference the model is told to copy, both in what the text encoder sees and in the reference latents. At strength 1 the sampler starts from pure noise, so your edit instruction wins. Below 1 it starts from that same picture, so there is nothing to change and little does.

This pack has a drop-in replacement, **Text Encode MOSS-Image Edit (masked reference)** (category `conditioning/moss_image`). It takes the same inputs plus the same `box_mask` / `inpaint_mask` as the inpainting node, and hides the part that is about to be repainted from the reference (`hide_edit_area`: `grey`, `blurred` which keeps rough colours and layout, or `keep` for the stock behaviour; `expand` grows and feathers the hidden area). The surroundings stay the reference. Use it for the positive and the negative prompt, with an image at the working size, and wire the same masks into it as into the inpainting node. Untested against the real model: compare `grey` / `blurred` at strength about 0.75 with the same seed.

## Per-box passes (different strength per box)

All boxes normally share one sampling pass and one strength. For edits that need different strengths (a clothes change wants ~0.95-1, a face tweak ~0.7), connect **clip** and the bbox node's **passes** output to Invoke Inpainting. Each box is then edited in its own pass, in order, each on the result of the previous one, with its own prompt and its own strength (set in the box editor; blank = this node's `strength`). `positive` / `negative` are not needed in this mode; `edit_format`, `hide_edit_area`, `expand` and `negative_prompt` (empty = zeroed) are used instead. Each pass re-encodes the prompt, so it costs one text-encoder run per box.

**Small boxes (faces etc.):** set `box_working_mp` (e.g. 1.0) in per-box mode. A box (plus `context_padding`) smaller than that is cropped out, enlarged to about that many megapixels, encoded and edited there (the reference is the enlarged crop, so the model sees a big face), then scaled back down and blended into the picture. Bigger boxes are edited at the full-picture size as before. `0` = off.

**Prompt:** in per-box mode the node has its own `prompt` input (connect the bbox node's `prompt` output, or type it): the description of the whole picture, used as the scene text for every box. With no boxes, it is the prompt for a single whole-picture pass.

**Prompt style:** `prompt_style` `layout` sends the box prompt as a layout description ("this box contains ..."); `instruction` sends plain edit instructions ("open her mouth"), which keeps small features such as a mouth where they are. For clothes and faces, `instruction` with `hide_edit_area` = `keep`, strength 1.0 and the edit LoRA on all layers is the combination that worked best in testing.

**Settings that matter (tested bbox-only, no painted mask, on two different images and edits):**
- `cfg` around **1.5**. At 3 the edit model washes the whole picture towards grey-beige, inside the boxes and outside them (skin, sky and furniture all lose saturation). At 1.0 it drifts (warped faces, leftover bits of the old item). The default is 1.5.
- Style LoRAs (Shadman / DevilHS) from layer **14** (`layers=14->`) keep hair curl and texture better than from 24 (24 smoothed it); the edit LoRA (armV5) stays on **all layers** (`end_layer` -1): limited to 0-23 it ghosts the old outfit through the new one.
- `mask_blur` **0**. Blurring the paste-back edge is what makes dress outlines and arms soft with a pale glow; the model and the style LoRAs already draw crisp edges, so they are pasted as drawn.
- `hide_edit_area` `keep`, `prompt_style` `instruction`, the edit LoRA on all layers.
- Strength: use 1.0 for edits that must replace something (an outfit). At 0.9-0.95 the model keeps the original and the edit simply does not happen; there is no strength in between that changes the outfit and keeps the skin tone.
- Keep the expression box **tight** (just the mouth and lips). A loose box around the face lets the model redraw the head and pose; a tight one changes only the expression.
- `seam_match` (default on): a regenerated box can come back a touch lighter or darker overall than its surroundings, which shows as a faint straight tone step along the box edge. This measures that drift along the inside of the repaint edge and spreads the correction inwards a little. Colour only, no blurring of detail.

- `skin_match` (default on): at full strength the model redraws skin a little lighter and paler than the original, even skin the edit never meant to change (arms, cheeks, hands), and no setting of cfg, strength, LoRAs or the reference picture removes that. This fits ONE pointwise colour curve (2nd-order polynomial in RGB) from the result's skin to the original's, using pixels that are skin in both pictures at the same place and did not change much (so a lip that replaced skin is left alone), and applies it to the result's skin weighted by how skin-coloured each pixel is. A colour curve has no spatial part, so it cannot blur, soften an edge, leave a seam or spill a halo onto the wall.

Other pixel-space helpers, **off by default** (written for the grey cast that cfg 3 caused; with cfg 1.5 they are not needed, and each involves smoothing). A painted mask is for limiting an edit (one eye, one garment), not something the blend depends on:
- `edge_restore` (0 = off, ~0.45): cells only partly inside a painted mask come out washed; this shows the original there instead.
- `tone_fix`: takes out the colour shift the VAE adds to cells the sampler did not touch.
- `preserve_unchanged`: keeps the original pixels wherever the result is still the same as the original at a coarse scale.

## Scene Context (Qwen VL): stating what must stay the same

Edit models drift on whatever the prompt does not say. Wire the bbox node's **passes** into **Scene Context (Qwen VL)** (category `image/midfr`, uses a Qwen3-VL folder in `models/LLM`, no download) and its **passes** output into Invoke Inpainting. For each box it looks at that box's crop only and writes a line such as "Keep these unchanged: deep warm brown skin, full lips." that Invoke adds after the box's instruction. Why per box: a whole-picture note ("dark curly hair") made the model grow hair into an outfit box that had none, so each note only covers attributes visible inside its own box, and leaves out the box's own edit text and `exclude` (clothing, expression, pose, background by default). The VLM is unloaded after it runs, so it does not take VRAM from MOSS; the notes are cached per crop.

**edit_aware (default on)** makes the notes padding for each box's own edit text, without touching your prompt. Per box the VLM writes: the problem points that are visible in the redrawn area (skin tone, lip size, tattoos... as real values), a detailed description of the old item being replaced or removed including where it spreads (so a frilled hem lying on a sofa, or a headband, is not left behind), and a short sentence per surrounding object or person with its colour (so a grey jacket does not tint the beige sofa). In testing (grey suit over a frilled white dress, headwear removed, heels changed) this removed the leftover ruffle and the headband and kept the sofa beige. Make sure the box covers every part of the old item (an elbow outside the box shows through the new sleeve).

**Auto crop/context (`box_working_mp` and `context_padding` = -1, the default)**: a small box (under ~0.06 MP: headwear, mouth, eyes) is cropped out together with its surroundings (a window 3x the box), enlarged to ~1 MP, edited, and only the box is pasted back; the model sees the head around a headband instead of a headband on nothing (a crop of just the box made her bald). Bigger boxes (shoes, outfit) are edited inside the whole picture, which keeps pose and surroundings. Everything except seed and steps is behind the node's "Advanced settings..." button.

Known limit: the model redraws everything inside a box at full strength, so a feature the box overlaps but the edit does not touch (eyes under a headband box) can drift. Fixes the VLM notes and `preserve_unchanged` did not give; a trimmed box loses the hair around the band. Use a painted mask (the bbox node's Paint tool) to keep that feature out of the repaint.

**auto_strength (default on)**: each box's denoise strength is set from the kind of edit the VLM sees (replace / remove / add / recolour / detail; the highest if one box does several). In the box editor, leave a box's Strength blank for auto, or type a number to set it yourself. Current table: every kind is about 1.0, because in testing nothing below ~0.95 changed the outfit or even recoloured same-shape shoes (0.6 and 0.75 change nothing, 0.9 only greys them). Lower a kind in `TYPE_STRENGTH` (context_vlm.py) if a test shows it works lower.

In testing this helps lips, eyes and hair stay put and slightly reduces skin drift (cheek +49 -> +35), but it does not replace `skin_match`, which fixes the skin tone itself. Invoke also has an optional `context` text input used for every box that has no note of its own.

## Edit Mask (Qwen VL + SAM 2)
Wire the bbox node's `passes` (or Scene Context's) into this node and its `mask` output into Invoke Inpainting's `inpaint_mask`. With no boxes, wire the prompt into `prompt` and the whole image is treated as one box.

For each box the VLM reads the edit type, and the mask follows it:
- **change X to Y / add** (`region`): the whole box is repainted, so the new item can extend past the old one (sleeves over arms). The VLM names the visible body parts, SAM outlines each, and the ones the instruction would not change (decided from the instruction text alone, as a yes/no probability under `keep_below`) are cut out of the repaint (blue in `preview`). The old item is never cut out.
- **remove / recolour / background** (`mask`): only the outline of the item (SAM on a tight ~1.5 MP window), then an eyedropper step samples the item's colours and grows the mask into connected pixels of the same colour (catches thin parts like frills).
`mode` forces one behaviour for every box. `log` and the ComfyUI console list the decisions per box. SAM 2.1 Large (~0.9 GB, Apache-2.0) downloads to `models/sam2/sam2.1-hiera-large` on first use.

In `region` mode the box is grown by `grow_box` percent (default 6) on every side, so the new item has room to continue past it (a collar, a hem); the parts that stay are still protected, and the grown boxes come out on `passes` (wire the mask node between the bbox node and Scene Context: bbox `passes` -> Edit Mask -> its `passes` -> Scene Context / Invoke, and its `mask` -> Invoke's `inpaint_mask`). Edges of kept parts are shrunk 2 px and old-item-coloured pixels on those edges are repainted, so the old colour does not bleed round a kept hand.

Known limit: a box can only repaint inside itself; grow_box gives a margin, but a much larger overhang still needs a bigger box.

### Edit Blend (trim + seams)

Run it after the edit. Wire `original` = the source image, `edited` = Invoke's output, `mask` and `margin` = the Edit Mask node's `mask` and `margin` outputs. The grown part of a box (`margin`) gives the new item room; here any big change the edit made in that margin that does not look like the new item (its colours must match what the edit drew inside the box, and connect to it) goes back to the original. Small differences (skin tone) are left alone so no line appears. Colour differences along the edge of the edited area are smoothed inward over `seam_px` (0 = off). `strictness` lower = trims more.

### Edit Mask: `segment` mode (default)

The picture is split into named components in three layers: OneFormer (COCO panoptic) names the scenery (wall, sofa, window ...); SAM 3 paints the parts and items the VLM listed (face, hair, helmet, sword ...) plus the target of each instruction (lips, headband ...), and where it overlaps the scenery it wins; SAM 2.1 automatic segments then clean up SAM 3's edges. The VLM rules each component in or out: the instruction text alone (never the picture) decides whether it would look different, it also sees the whole list at once, anything worn goes with an outfit change and all scenery goes with a setting change. Boxes only constrain where the edit applies: replace / add repaint the (grown) box minus the parts and items that stay (face, hair, hands ...), remove / recolour / background repaint only the target components. No boxes: the whole image is the region. Weights: SAM 3 from `facebook/sam3` (needs your Hugging Face access) with an ungated mirror as fallback, OneFormer and SAM 2.1 download on first use. The older `auto` / `mask` / `region` modes are still selectable.

## Advanced edit (Qwen Image 2.1): change clothes, keep the body

Lives in the Advanced Edit subgraph of `workflows/Qwen Pipeline.json`. Models: Sapiens2 seg + pointmap, Qwen3-VL, Qwen Image 2.1 + Fun Controlnet Union. Nodes, in order:

- **Edit Planner (Qwen VL)** reads the picture and your plain instruction and writes the prompt (see below).
- **Sapiens Regions** splits the Sapiens2 segmentation into `person`, `trusted` (skin), `clothing`, `remove` (hair/glasses), `face`, plus a colour `preview`.
- **Sapiens2 Metric Depth**: metric depth (metres) of the clothed figure, turned into the control image by **Depth To Control**.

The clothed figure's depth is the Union control (strength 0.7-0.8, end 0.8). Strength 1.0 makes the new clothes cling to the old outline. Describe the whole result in the prompt, name what to remove (helmet, sword) and put armor words in the negative.

### Edit Planner (Qwen VL): only as much control as the edit needs

`Edit Planner (Qwen VL)` (uses a Qwen3-VL folder in `models/LLM`, unloaded after it runs) reads the picture and your plain instruction and decides:
- **edit type** (outfit / recolor / local / background / pose) and whether the body outline changes. Only an outfit change that alters the silhouette
  switches the **depth control** on (`use_control`, strength and end from the node). Recolours and small edits are plain reference edits.
- **what is removed**: old clothing, armor and held items the instruction implies but does not name (replacing armor removes helmet, plates, cape, sword;
  replacing a dress removes the dress and ruffles only). They go into the negative prompt and `drop_classes` (hair/glasses dropped from the silhouette).
  Anything the instruction itself names is never listed as removed.
- **what is added**, in concrete terms (garment length and fit, what a held object looks like, what the new setting contains).
- **what must stay**: face, hair, body build, pose (posture only), camera angle, framing, art style and background (unless the edit changes it),
  written from the picture into the prompt even when the instruction does not mention them.

The `Gate Image / Gate Text / Gate Model (lazy)` nodes make the control optional: when the planner switches it off, Sapiens depth is not executed at all.

## Control modes
The planner's `control_mode` picks what Qwen gets as the control image: `auto` (the planner decides), `raw` (no planner, no VLM, no control; your instruction goes to Qwen unchanged), `none` (plain reference edit) or `depth` (the clothed figure's depth: Sapiens2 + person mask). Only the chosen branch runs.

## Qwen 3D Character (re-pose / re-camera)
`workflows/Qwen 3D Character.json` fits her MHR body from the photo, optionally copies a pose from a reference image (`MHR Pose Transfer`), orbits/zooms the camera (`MHR View Render`) and projects the photo's colours onto the posed body. Qwen then redraws the render with optional outfit / character references. Reference loaders are bypassed by default; un-bypass the ones you want. Geometry only (no generator or texture model); surface the photo never saw comes out smeared.


## Qwen Character Builder (photo to rigged, textured character, saved to a library)
`workflows/Qwen Character Builder.json`, all inside ComfyUI:
1. Perception and body fit from the photo (Sapiens2 + SAM 3D Body + MHR slider fit).
2. **MHR Neutral Pose** puts her fitted body in a neutral A-pose (frame fits height and arm span); its depth is the Qwen control, Qwen redraws her
   from the photo in that pose. `ImagePick` skips this when the photo already is a clean full-body A-pose (pick `b`).
3. Pixal3D (shape only) builds the mesh from the A-pose image; **Decimate** and **Unwrap** prepare it.
4. **Mesh Orbit Cameras** makes N views (default 6) with a depth control and a view-specific prompt each (back views say the face is hidden, or Qwen draws
   the front again). Qwen draws one image per view (list execution: every node runs over all views before the next node).
5. **Mesh Project Texture** projects the views onto the mesh (visibility and facing weighted, silhouette trimmed, gutter fill).
6. **Character Library: Save** stores mesh, fitted pose data, the A-pose image and a thumbnail in `ComfyUI/models/characters/<name>/`; a GLB is also saved.

Speed/VRAM: `Qwen Image 2.1 Cache` (int8) lowers VRAM; resolution, steps and view count trade time for quality. Lower resolutions drift from the
depth control, so views misalign with the mesh; keep views near 1 MP when texture alignment matters.

**Character Library: Load** lists the saved characters (dropdown, press R after saving) and outputs mesh, pose data, thumbnail and the A-pose image.

## Qwen Pipeline (all-in-one)

`workflows/Qwen Pipeline.json`: one graph, three green subgraphs, orange custom nodes.

- **Mode Selection**: Simple edit (image + prompt straight to Qwen) / Advanced edit (VLM plan + Sapiens depth control) / Model extraction and edit.
  Only the chosen subgraph is sent to the backend (the others are muted when you press Run).
- **Input: image or saved model**: upload an image, or switch to "saved model" and pick a character from `models/characters`
  (its source image replaces the input and the mesh is reused, no extraction).
- **Model extraction and edit**: A-pose redraw -> Pixal3D shape -> multi-view (and face/bust close-up) Qwen texturing -> saved to the library
  (skipped when a saved model is used) -> the edit prompt is run with the character's front depth as control.
- **Output**: the image plus a 3D model preview/GLB when a character was built or loaded.
- **Live Preview**: one node that shows an image for every step (segmentation, depth, A-pose, each generated view...).
  Add a `Step preview` node anywhere to put a step in it.
- `MeshOrbitCameras.closeups` (1-3) adds zoomed head-and-bust views that `MeshProjectTexture` weights over the body views.
- **Pose reference (optional)**: in Model extraction and edit, switch it on and pick an image; the character (built or saved) is re-posed
  to that image's pose (SAM 3D Body + the MHR rig, texture kept) before its depth is used as the Qwen control. Off = the character's A-pose.
- The input node shows a clickable thumbnail gallery of `models/characters` (button "refresh library" after a new save).
- **Pose library**: with "re-pose" on and the pose coming from an image, the pose is saved to `models/poses/<image name>` (thumbnail included) unless
  `save_to_library` is off. Switch to "saved pose" and click a thumbnail in the pose node's gallery to reuse it without the image.

### Qwen Inputs (one node for all inputs)
`Qwen Inputs` replaces the separate input and pose nodes. It shows a reference image picker, then three checkboxes:
- **Use 3D Model**: the input becomes the model library (saved characters + imported models, with thumbnails). *Import a 3D model* takes
  `.glb .gltf .obj .stl` plus their texture / `.mtl` files (up axis auto-detected, facing selectable; FBX/USDZ: convert to GLB first).
  Imports are stored in `models/characters`. A model with no stored rig is auto-rigged (SAM 3D Body on a render) when a pose is applied;
  an untextured model is textured from the reference image with the multi-view Qwen stage.
- **Use Saved Pose**: pick a saved pose, or take the pose from an image (optionally saving it to `models/poses`).
- **Use Other Reference**: up to 3 more images, each tagged Outfit / Background / Same Character / Other Character / Face / Hairstyle /
  Object / Style / Lighting. The tags become prompt notes and fill `image_2..4` of the Qwen text encode; *Same Character* images are also
  given to the A-pose and view-texturing stages as extra angles.
The old `Input: image or saved model` and `Pose reference` nodes are still registered so older workflows load.
