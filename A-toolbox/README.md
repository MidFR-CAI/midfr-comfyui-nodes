# Pack A: Toolbox

Five independent workflow tools. Install any or all by copying their folders into `ComfyUI/custom_nodes`.

| Folder | Node(s) | Does |
|---|---|---|
| `ComfyUI-Cleaner` | Free Memory (passthrough) · Free Memory After Run · Memory Status | **Resource Reaper.** Frees VRAM/RAM automatically before every run (adaptive light/standard/deep), plus a sidebar panel listing other programs holding your GPU. Optional: `pip install psutil nvidia-ml-py`. |
| `ComfyUI-TextEditPause` | Text Edit (Pause) | Stops the workflow, opens a popup with the incoming text, continues with your edited version (Resume) or cancels (Abort). |
| `simple_image_upload_resize` | Simple Image Upload + Resize · Multi Image Upload + Resize (9) · Image Set Splitter (9) · Image BBox Layout (MOSS) | Drag-and-drop images, downscaled to a megapixel limit. Simple Upload also outputs a `mask` (alpha channel / mask editor). BBox Layout: draw boxes on the image, prompt each, get MOSS-Image layout JSON (no boxes = no output). Details in `simple_image_upload_resize/README.md`. |
| `comfyui-airpg-saturation-match` | AIRPG Saturation Match | Matches an edited image's colour intensity to its reference (Oklab chroma only, hue and lightness untouched). Auto mode measures only unchanged pixels. |
| `comfyui-lora-vault` | Lora Vault Loader | Stores LoRA metadata and previews, downloads weights on first use, evicts old ones, keeps a snapshot if Civitai removes a model. Uses an optional Firefox helper extension; see its README. |
