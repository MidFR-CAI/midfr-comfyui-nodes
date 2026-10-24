# MidFR ComfyUI Nodes

Three small packs of custom [ComfyUI](https://github.com/comfyanonymous/ComfyUI) nodes: everyday workflow tools, dataset tagging, and MiniMax H3 prompt building.

> **Is this safe?** Yes, and you don't have to take my word for it: it's 100% plain-text Python and JavaScript, no compiled files, no installers, no obfuscation. See [SECURITY.md](SECURITY.md) for exactly what each pack touches (files, network, processes) and how to check it yourself in 5 minutes.

| Pack | Name | Nodes |
|---|---|---|
| **A** | [Toolbox](A-toolbox) | 9 across 5 tools |
| **B** | [Image Tagging](B-image-tagging) | 4 |
| **C** | [H3 AutoPrompt](C-h3-autoprompt) | 3 |

## Pack A: Toolbox
| Tool | Nodes | Does |
|---|---|---|
| **Resource Reaper** (`ComfyUI-Cleaner`) | Free Memory (passthrough), Free Memory After Run, Memory Status | Frees VRAM/RAM before each run; sidebar panel shows what else is hogging the GPU. |
| **Text Edit (Pause)** (`ComfyUI-TextEditPause`) | Text Edit (Pause) | Pauses the workflow, lets you edit the text in a popup, then continues. |
| **Image Upload + Resize** (`simple_image_upload_resize`) | Simple Image Upload + Resize, Multi Image Upload + Resize (9), Image Set Splitter (9) | Drag-and-drop images, auto-resized to a megapixel cap (multiples of 32). |
| **Saturation Match** (`comfyui-airpg-saturation-match`) | AIRPG Saturation Match | Fixes over-saturated edits by matching colour intensity to the reference image. |
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

Each tool is a normal ComfyUI custom-node folder. Either:

**Easy:** run the installer, pick packs, it copies the folders into `ComfyUI/custom_nodes`:
```
python install.py "C:\path\to\ComfyUI\custom_nodes"          # all packs
python install.py "C:\path\to\ComfyUI\custom_nodes" A C      # only packs A and C
```
**Manual:** copy the folder(s) you want from inside a pack folder (e.g. `A-toolbox/comfyui-lora-vault`) into `ComfyUI/custom_nodes`, then restart ComfyUI.

Some tools have optional Python dependencies; see each pack's `requirements.txt` / README. Use ComfyUI's own Python (portable build: `python_embeded\python.exe -m pip install ...`).

## License
[MIT](LICENSE). Pack C is designed to work alongside, but does not include, Adudeguyman's [ComfyUI-Fantastic-MiniMaxH3-PromptBuilder](https://github.com/Adudeguyman/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder).
