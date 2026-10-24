# Security & Trust

These nodes are plain, readable source code. There are no binaries, no `.exe`/`.dll`, no obfuscated or minified code, no install scripts that run on their own, and nothing is downloaded and executed.

## What each pack does on your machine

| Pack | Network | Writes files | Runs other programs |
|---|---|---|---|
| B Image Tagging | Only if the Qwen model field is a Hugging Face ID (downloads the model via `transformers`) | `.txt` captions you ask it to save | No |
| C H3 AutoPrompt | Optional model downloads from Hugging Face (WD tagger, Whisper, diarization) | Temp audio files (for Whisper) | `ffmpeg` (fallback video/audio decoding), nothing else |
| A Resource Reaper | None | `config.json` (its own settings) | Only if you enable "allow terminate" (off by default) and click it, to close a program hogging VRAM; never automatic |
| A Text Edit (Pause) | None (local ComfyUI routes only) | None | No |
| A Image Upload + Resize | None | Copies images into ComfyUI's `input` folder | No |
| A Saturation Match | None | None | No |
| A Lora Vault | `civitai.com` only, and only through the optional Firefox extension; ComfyUI itself never connects to Civitai | Its own data folder (settings, SQLite DB, previews, LoRAs you choose to download); deletes only its own cached files | No |

Notes worth knowing:
- **Resource Reaper** wraps ComfyUI's `PromptExecutor.execute` so it can free memory before each run. It's a standard monkey-patch, visible in `hook.py` / `cleanup.py`.
- **Pack B/C `trust_remote_code=True`**: loading a Hugging Face model through `transformers` can run code shipped with that model. Only load models from sources you trust. Local GGUF models (llama.cpp) don't have this.
- **Lora Vault's Firefox extension** requests access to `civitai.com`/`civitai.red` and downloads, plus `localhost:8199` (to talk to ComfyUI). Its full source is in `extension/`. No signed `.xpi` is shipped; build or sign your own (see its README) so you know exactly what you're running.
- **API keys:** the repo contains none. Lora Vault's optional Civitai key is stored locally in its own settings file (git-ignored) or read from `CIVITAI_API_KEY`.

## Check it yourself (5 minutes)
1. Search the repo for `subprocess`, `os.system`, `eval(`, `exec(`, `socket`, `urlopen`, `requests`: every hit is listed above and in context.
2. Scan with [VirusTotal](https://www.virustotal.com) or just download the ZIP and open the files in a text editor; they're all text.
3. Run ComfyUI with a firewall prompt on; the only outbound connections should be the ones in the table.

## Reporting a problem
Open a GitHub issue, or use *Security → Report a vulnerability* on the repo.
