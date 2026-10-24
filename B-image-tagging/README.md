# Pack B: Image Tagging

Nodes (`image/tagging`) for captioning a folder of images into training-ready `.txt` files.

| Node | Does |
|---|---|
| **Batch Image Loader** | One image per run from a folder: glob patterns, shuffle + seed, auto-advance. Outputs image, filename, index, total. |
| **Hybrid Tagging Node** | WD14 booru tags, a Qwen3-VL caption, or both (hybrid mode feeds the tags to Qwen for a natural-language caption). |
| **Same Directory Text Saver** | Writes the tags / caption / combined text as `<image>.txt` beside the source image. |
| **Image Batch Queue** | Advances / resets a batch counter and flags `is_complete`. |

**Requirements:** `transformers` (Qwen3-VL), `onnxruntime` (WD14). The WD14 mode reads `.onnx` + `.csv` model files from the `comfyui-wd14-tagger` models folder (install that pack or drop the files there). The Qwen model field takes a local folder path or a Hugging Face ID (default `Qwen/Qwen3-VL-4B-Instruct`, downloaded on first use).

Install: copy `comfyui-midfr-image-tagging` into `ComfyUI/custom_nodes`.
