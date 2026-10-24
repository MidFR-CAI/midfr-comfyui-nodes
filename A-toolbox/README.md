# Pack A: Toolbox

Independent workflow tools. Install any or all by copying their folders into `ComfyUI/custom_nodes`.

| Folder | Node(s) | Does |
|---|---|---|
| `ComfyUI-Cleaner` | Free Memory (passthrough) · Free Memory After Run · Memory Status | **Resource Reaper.** Frees VRAM/RAM automatically before every run (adaptive light/standard/deep), plus a sidebar panel listing other programs holding your GPU. Optional: `pip install psutil nvidia-ml-py`. |
| `ComfyUI-TextEditPause` | Text Edit (Pause) | Stops the workflow, opens a popup with the incoming text, continues with your edited version (Resume) or cancels (Abort). |
| `simple_image_upload_resize` | Simple Image Upload + Resize · Multi Image Upload + Resize (9) · Image Set Splitter (9) | Drag-and-drop images, downscaled to a megapixel limit. Simple Upload also outputs a `mask` (alpha channel / mask editor). |
| `comfyui-airpg-saturation-match` | AIRPG Saturation Match | Matches an edited image's colour intensity to its reference (Oklab chroma only, hue and lightness untouched). Auto mode measures only unchanged pixels. |
| `comfyui-lora-vault` | Lora Vault Loader | Stores LoRA metadata and previews, downloads weights on first use, evicts old ones, keeps a snapshot if Civitai removes a model. Uses an optional Firefox helper extension; see its README. |
| `comfyui-flow-strength` | Strength to Denoise (flow models) | InvokeAI-style img2img strength -> the `denoise` KSampler needs (MOSS-Image, Flux...). Details in its README. |
| `image_bbox_layout` | Image BBox Layout (MOSS) | Upload an image, draw boxes on it, prompt each, get a MOSS-Image layout prompt (plain text when there are no boxes). Also outputs the boxes, a grown/feathered mask and the mask you paint (right-click > Open in MaskEditor). |
| `invoke_inpainting` | Invoke Inpainting (MOSS) | One-node InvokeAI-style inpainting: crop to a box, repaint only inside an optional mask with soft edges, paste back. Settings default to Invoke's. |
