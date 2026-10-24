# MidFR ComfyUI Nodes

Three small packs of custom [ComfyUI](https://github.com/comfyanonymous/ComfyUI) nodes: everyday workflow tools, dataset tagging, and MiniMax H3 prompt building.

> **Is this safe?** Yes, and you don't have to take my word for it: it's 100% plain-text Python and JavaScript, no compiled files, no installers, no obfuscation. See [SECURITY.md](SECURITY.md) for exactly what each pack touches (files, network, processes) and how to check it yourself in 5 minutes.

| Pack | Name | Nodes |
|---|---|---|
| **A** | [Toolbox](A-toolbox) | 12 across 8 tools |
| **B** | [Image Tagging](B-image-tagging) | 4 |
| **C** | [H3 AutoPrompt](C-h3-autoprompt) | 3 |

## Pack A: Toolbox
| Tool | Nodes | Does |
|---|---|---|
| **Resource Reaper** (`ComfyUI-Cleaner`) | Free Memory (passthrough), Free Memory After Run, Memory Status | Frees VRAM/RAM before each run; sidebar panel shows what else is hogging the GPU. |
| **Text Edit (Pause)** (`ComfyUI-TextEditPause`) | Text Edit (Pause) | Pauses the workflow, lets you edit the text in a popup, then continues. |
| **Image Upload + Resize** (`simple_image_upload_resize`) | Simple Image Upload + Resize (image + **mask**), Multi Image Upload + Resize (9), Image Set Splitter (9) | Drag-and-drop images, auto-resized to a megapixel cap. The mask comes from the image's alpha / the mask editor. |
| **Saturation Match** (`comfyui-airpg-saturation-match`) | AIRPG Saturation Match | Fixes over-saturated edits by matching colour intensity to the reference image. |
| **Strength to Denoise** (`comfyui-flow-strength`) | Strength to Denoise (flow models) | Invoke-style img2img strength converted to KSampler `denoise` (on MOSS-Image `denoise 0.9` keeps only ~2% of the source). |
| **Image BBox Layout** (`image_bbox_layout`) | Image BBox Layout (MOSS) | Draw and prompt boxes on an image; outputs a MOSS-Image layout prompt, the boxes and a soft mask. |
| **Invoke Inpainting** (`invoke_inpainting`) | Invoke Inpainting (MOSS) | InvokeAI-style box + mask inpainting in one node. |
| **Lora Vault** (`comfyui-lora-vault`) | Lora Vault Loader | LoRA manager: saves Civitai metadata, downloads weights on first use, tracks removals. |

## Pack B: Image Tagging
| Node | Does |
|---|---|
| **Batch Image Loader** | Loads one image per run from a folder (glob, shuffle, auto-advance). |
| **Hybrid Tagging Node** | Makes WD14 booru tags and/or a Qwen3-VL caption, plus a combined version. |
| **Same Directory Text Saver** | Saves the caption as a `.txt` next to the source image. |
| **Image Batch Queue** | Counts through a batch and flags when it's finished. |

## Pack C: H3 AutoPrompt
Writes MiniMax H3's strict structured video prompt from your reference media plus a short brief.
| Node | Does |
|---|---|
| **H3 AutoPrompt: Structured Media Loader** | Panel for loading and tagging pictures, video and audio references per shot. |
| **H3 AutoPrompt: All-in-One** | Local VLM + Whisper turn the references into a format-valid H3 prompt, with a self-check report. |
| **H3 AutoPrompt: Reference Splitter** | Splits the bundled references into individual picture / video / audio outputs. |

## Install

**Recommended: clone the whole repo into `custom_nodes`.** A root `__init__.py` loads every tool; if one fails (e.g. a missing dependency) it's reported in the console and the rest still load.
```
cd ComfyUI/custom_nodes
git clone https://github.com/MidFR-CAI/midfr-comfyui-nodes
```
Restart ComfyUI. The console prints `[MidFR] loaded N tool(s), N node(s)`.

- **Don't also keep the old standalone copies** of these tools in `custom_nodes` (duplicate node names).
- **Skip a tool:** list its folder name (e.g. `comfyui-lora-vault`) in a `disabled.txt` next to `__init__.py`, one per line, or set `MIDFR_DISABLE=comfyui-lora-vault,ComfyUI-Cleaner`.
- **Defaults to know:** Resource Reaper frees memory before every run (turn off in its sidebar panel), and Lora Vault starts a local-only helper on `127.0.0.1`.

**Just want one tool?** Run `python install.py "C:\path\to\ComfyUI\custom_nodes" A` (pack letters A, B, C; none = all) or copy a single tool folder (e.g. `A-toolbox/ComfyUI-TextEditPause`) into `custom_nodes`. Each tool also works on its own.

Some tools have optional Python dependencies; see each pack's `requirements.txt` / README. Use ComfyUI's own Python (portable build: `python_embeded\python.exe -m pip install ...`).

## License
[MIT](LICENSE). Pack C is standalone and needs no other node pack. Its reference numbering conventions and the prompt grammar it encodes follow Adudeguyman's MIT-licensed [ComfyUI-Fantastic-MiniMaxH3-PromptBuilder](https://github.com/Adudeguyman/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder) and MiniMax's H3 prompt-writing guide; thanks to both.
