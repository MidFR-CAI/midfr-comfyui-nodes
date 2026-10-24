> **Archived design notes.** Written when the Subject Categorizer / Categorized Composer nodes were still registered. Today the pack exposes only the nodes listed in the pack README; the grammar/VLM/Whisper design below still applies.

# ComfyUI-H3-AutoPrompt

Automates writing MiniMax H3's rigid, structured video prompt from your
reference images/video/audio and a short plain-language brief.

## Two pipelines

### Recommended: the categorized pipeline

```
Fantastic H3 Media Loader (the original pack, unmodified)
  -> H3 AutoPrompt: Subject Categorizer   (tag each item explicitly)
  -> H3 AutoPrompt: Categorized Composer  (deterministic build + narrow VLM)
```

Everything format-critical -- which pictures are the same subject, whether a
picture is an identity reference or a literal frame anchor, whether an audio
clip is dialogue or a voice-timbre donor, which subject speaks which line --
is explicit input from the Subject Categorizer panel, not something a VLM
infers from your brief. The VLM's only remaining job is writing a short
visual description per subject and turning your brief into an ordered shot
list. This exists because the earlier all-VLM pipeline (below) kept
misclassifying exactly these things -- character replacement, voice-donor
pairing, frame anchors -- no matter how much the system prompt was
strengthened, because a small local VLM has a real ceiling on this kind of
judgment call. Moving it to explicit user input isn't a workaround for that
ceiling, it's just not asking the model to guess something the person
loading the media already knows.

**Setup**: install the original pack
([ComfyUI-Fantastic-MiniMaxH3-PromptBuilder](https://github.com/Adudeguyman/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder))
alongside this one -- this pipeline reads its Media Loader's live state
directly off the graph (the same trick that pack's own Prompt Builder uses
for tag previews), so no data flows through Python before you open the
categorizer panel. Wire the Loader's `references` output into **H3
AutoPrompt: Subject Categorizer**'s `references` input, click "Open
categorizer...", tag each item, Save. Wire that node's `references` output
(now carrying your tags) into **H3 AutoPrompt: Categorized Composer**
alongside your brief and duration.

**Categorization options**, per item:
- **Pictures**: `identity reference` (contributes to a Subject's
  description, with a retention marker you choose directly) or `concrete
  frame anchor` (this exact picture IS a specific frame -- first/last
  frame, keyframe, or storyboard, for a specific shot number).
- **Videos**: `identity/environment` (its visual content is tracked as a
  Subject) or `structure/motion only` (source/motion reference for a
  *different* subject -- e.g. "this subject's look comes from a photo, but
  its motion comes from this video").
- **Audio** (standalone or a video's paired soundtrack): `dialogue`
  (transcribed with Whisper, attributed to a subject, optionally marked
  off-screen/narrator) or `voice timbre` (never transcribed -- only lends
  its vocal sound to a *different* subject's dialogue, which you specify
  directly) or `ambience`/`music` (cited in the sound sections) or `off`.

Leave an item's subject number blank to exclude it from the prompt
entirely -- there's no need to categorize media you don't want cited.

### Legacy: the all-VLM pipeline

```
H3 AutoPrompt: Media Loader  ->  H3 AutoPrompt: All-in-One
```

Self-sufficient: does **not** require Adudeguyman's
[ComfyUI-Fantastic-MiniMaxH3-PromptBuilder](https://github.com/Adudeguyman/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder)
to be installed. Its `prompt` output and `picture_N`/`video_N`/`video_audio_N`/
`audio_N` pass-throughs are wire-compatible with that pack's editor and with
ComfyUI's native `MiniMax H3 Image to Video` / `MiniMax H3 Reference to Video`
nodes, so you can drop this in front of either.

## Why this design

MiniMax ships a written grammar (`Video_Prompt_Writing_Guide.pdf`, extracted
and encoded into [`grammar.py`](./grammar.py)) with hard rules: strictly
increasing cut times in `MM:SS.mmm`, a fixed instruction line per mode,
`<d>[Language] ...</d>` dialogue blocks, stable `(Sx)` speaker IDs assigned in
first-vocal-event order, exactly six ordered sections in full-reference mode,
`<Subject/Picture/Video/Audio N>` labels that must stay consistent across
every section, etc.

Models are good at deciding **what** a shot should show or what a voice says.
They are unreliable at guaranteeing **formatting** invariants like "this
timestamp must be strictly greater than the last one." So this pack keeps
those two jobs in separate code paths:

```
              ┌─────────────────────┐
brief + media │  VLM (vlm_engine.py) │──▶ scene JSON (content, camera, style)
   ──────────▶│  content only, never │
              │  formatting          │
              └─────────────────────┘
reference     ┌─────────────────────┐
   audio  ───▶│ Whisper (whisper_    │──▶ dialogue JSON (who/what/when/lang)
              │ engine.py) + turn    │
              │ segmentation         │
              └─────────────────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │ grammar.py           │──▶ format-legal H3 prompt string
              │ pure Python, zero    │    + a self-check against the
              │ model calls          │    guide's own output checklist
              └─────────────────────┘
```

`grammar.py`'s `validate()` re-implements the guide's Section 3 "Output
Checklist" mechanically (sequential shot numbers, strictly increasing cut
times inside the video's duration, `[Shot 1]` never carrying a timestamp,
balanced `<d>` tags, no `(Sx)` inside `retention_analysis`, all six REF
sections present and in order, ...) so a broken merge fails loudly in
`validation_report` instead of silently producing a prompt H3 will choke on.

## Nodes

| Node | Job |
|---|---|
| **H3 AutoPrompt: Media Loader** | Loads pictures/video/audio from disk, assigns `<Picture/Video/Audio N>` tags using the same numbering convention as the native H3 nodes expect (a video's paired soundtrack takes the audio slot right before its own video). |
| **H3 AutoPrompt: VLM Scene Analyzer** | Feeds your brief + sampled frames to a local VLM, gets back structured JSON (style, subjects, per-shot content/camera/on-screen text, ambient sound, music). |
| **H3 AutoPrompt: Whisper Dialogue** | Transcribes every audio reference with `large-v3-turbo`, word-level timestamps, language detection, and speaker-turn grouping. |
| **H3 AutoPrompt: Prompt Composer** | Pure-Python merge of the above into a grammar-legal prompt, plus a validation report. |
| **H3 AutoPrompt: All-in-One** | The four above, chained, with the same media pass-through outputs as the native nodes -- one node, brief in, prompt + media out. |

## Using the other pack's Media Loader instead of this one

If you already have Adudeguyman's ComfyUI-Fantastic-MiniMaxH3-PromptBuilder
installed and prefer its drag-and-drop Media Loader UI, keep using it --
their `H3_REFS` output wires **directly** into this pack, no visible
adapter node needed:

```
MiniMax H3 Media Loader  →  H3 AutoPrompt: Media Loader (mmh3_refs input)
     (H3_REFS)                        or
                              H3 AutoPrompt: All-in-One (mmh3_refs input)
```

Both of this pack's entry points have an `mmh3_refs` optional input typed
`H3_REFS` -- connect their Media Loader's `references` output straight to
it. `H3_REFS` is that pack's own private bundle type with no documented
internal shape, so rather than guessing its dict layout, this pack locates
that pack's own **MiniMax H3 Reference Splitter** class through ComfyUI's
node registry at runtime and calls it directly to do the unpacking --
reading their real unpacking code every time, not a format assumption that
could go stale. This requires ComfyUI-Fantastic-MiniMaxH3-PromptBuilder to
be installed (their Reference Splitter node itself doesn't need to be
placed anywhere in your graph); if it isn't found, you get a clear error
naming the missing node rather than a silently empty or wrong bundle.

If you'd rather route through their **Reference Splitter** node yourself
(e.g. you're also feeding their native H3 nodes from the same media and
want one shared upstream bundle), that path still works too via the
`ext_picture_1..9` / `ext_video_1..3` / `ext_video_audio_1..3` /
`ext_audio_1..3` optional inputs on **H3 AutoPrompt: Media Loader**. If
more than one source is wired at once, priority is `mmh3_refs` > `ext_*`
slots > `media_json`.



```
cd ComfyUI/custom_nodes
git clone <this repo>   # or unzip
pip install -r ComfyUI-H3-AutoPrompt/requirements.txt --break-system-packages
```

You need `ffmpeg` on PATH (or `pip install av`) for video/audio reference
handling -- same requirement as the original PromptBuilder pack.

### VLM model

Default backend is a local **GGUF** vision model via `llama-cpp-python`
(lowest VRAM footprint, and the easiest route to an already-decensored
checkpoint). Point `vlm_model_path_or_id` / `vlm_mmproj_path` at:

- **Qwen2-VL-2B-Instruct** (or a community **abliterated**/"heretic" build of
  it) -- best JSON-following-to-size ratio of the small VLMs, good multi-image
  reasoning for matching a voice reference to a face reference.
- **Qwen2.5-VL-3B-Instruct-abliterated** -- a step up if you have VRAM to
  spare alongside the video model.
- **InternVL2.5-2B/4B** abliterated builds, if you prefer that family.

Search Hugging Face for `"abliterated" <model name> GGUF` -- these are
community releases (frequently under names like `Huihui-*-abliterated`), not
official vendor builds, so read the model card and evaluate the specific
release yourself before use. The point of using a decensored checkpoint here
is narrow: so the captioner doesn't refuse to plainly describe ordinary
reference media (a swimsuit photo, a bar scene, a horror-styled clip). It is
not, and should not be used as, a way to route around this pack's own rules
(see `vlm_engine.SYSTEM_PROMPT`, which explicitly forbids naming real people
or reproducing copyrighted characters regardless of which checkpoint you
load).

If you'd rather load an HF-format checkpoint directly instead of GGUF, switch
`vlm_backend` to `transformers_hf` and put a model id in
`vlm_model_path_or_id` -- `vlm_engine.TransformersVLM` handles it, uses more
VRAM, but needs no separate mmproj file.

### Whisper model

`faster-whisper`'s `large-v3-turbo` downloads automatically on first run
(CTranslate2 format, pulled from Hugging Face). Set `device` to `cpu` if you
don't want it competing with the VLM/video model for VRAM -- turbo is fast
enough on CPU for the short (2-15s) clips H3 reference audio is limited to
anyway.

Diarization (`use_diarization`) tries `pyannote/speaker-diarization-3.1` if
installed and falls back to a silence-gap heuristic otherwise -- see
`whisper_engine.py` for the trade-off. Given H3's own 2-15s clip length limit,
most reference audio has exactly one voice and diarization is a no-op.

## What's still heuristic, and where to hand-tune

- **Shot timing**: the VLM gives each shot a rough `duration_hint_s`; the
  composer scales those proportionally to fill `duration_seconds`. It doesn't
  understand pacing the way a human editor does -- if a beat needs to land at
  an exact time, edit `cut_time_s` on the `Shot` before composing, or just
  edit the finished prompt's timestamps directly (they're plain text).
- **Retention markers** (`fully_preserved` / `partially_preserved` /
  `attribute_transfer` / `weak_reference`) default to `fully_preserved` for
  every subject. The guide's own definitions of these are judgment calls
  (does this shot actually change the subject's defined characteristics?)
  that a 2B VLM will get wrong often enough to be worth a manual pass on
  anything that isn't a straightforward "copy the reference" case.
- **Speaker identity phrases** ("the quiet, breathy young woman...") default
  to a generic placeholder. Feed richer descriptions into
  `H3AutoPromptWhisperNode`'s `speaker_labels_json` (`{"Audio 1::0": "the
  quiet, breathy young woman"}`) if you want the guide's recommended level of
  first-appearance detail without hand-editing afterward.

None of this blocks the pipeline -- `validation_report` only checks the
mechanically-checkable rules (numbering, tag balance, ordering). Creative
judgment calls like the ones above are exactly what's worth reviewing in the
few seconds it takes to read the composed prompt before you queue.

## Credits

Grammar rules encoded in `grammar.py` are transcribed from MiniMax's own
*Video Prompt Writing Guide*, which ships inside Adudeguyman's
ComfyUI-Fantastic-MiniMaxH3-PromptBuilder (MIT licensed) and is the same
guide that pack's live JS checker enforces. `media_io.py`'s reference-tag
numbering convention (video-soundtrack-before-its-video) mirrors that pack's
Media Loader for drop-in compatibility. No code from that repository was
copied verbatim; this is an independent implementation built for a
GPU-local, model-driven pipeline rather than a hand-driven editor.
