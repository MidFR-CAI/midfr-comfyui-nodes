# Pack C: H3 AutoPrompt

Builds [MiniMax H3](https://www.minimax.io)'s strict structured video prompt from your reference pictures, video and audio plus a short plain-language brief. Category: `conditioning/video_models/h3_autoprompt`.

| Node | Does |
|---|---|
| **H3 AutoPrompt: Structured Media Loader** | Load references and tag each one (subject, frame anchor, motion reference, dialogue, voice timbre, ambience...) in an in-graph panel. |
| **H3 AutoPrompt: All-in-One** | Composes the final prompt. A local VLM writes the content, Whisper transcribes dialogue, and plain Python builds the format. Returns `prompt` + media, with a validation report against H3's own output checklist. |
| **H3 AutoPrompt: Reference Splitter** | Splits the bundled references into `picture_N` / `video_N` / `audio_N` outputs for native H3 nodes. |

Why it's reliable: the model only decides *what* appears; `grammar.py` guarantees *format* (timestamps, speaker IDs, section order). See [`docs/design-notes.md`](ComfyUI-H3-AutoPrompt/docs/design-notes.md).

**Requirements:** see `requirements.txt` (core: `torch`, `Pillow`, `numpy`; plus `av` or `ffmpeg`, `faster-whisper`, and either `llama-cpp-python` (GGUF VLM) or `transformers`). Optional: `pyannote.audio` (speaker diarization), `opencv-python` + `pytesseract` + `onnxruntime` (comic mode). Standalone: needs no other node pack, and its outputs feed ComfyUI's built-in MiniMax H3 nodes.

Install: copy `ComfyUI-H3-AutoPrompt` into `ComfyUI/custom_nodes`, `pip install -r requirements.txt`.
