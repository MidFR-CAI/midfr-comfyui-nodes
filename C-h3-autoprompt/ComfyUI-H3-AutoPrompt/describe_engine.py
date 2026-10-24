"""
describe_engine.py -- the VLM's job in the categorized pipeline, cut down to
exactly two things it's actually reliable at:

  1. Write a short, generic visual description for each subject/anchor the
     user already declared (given crops of its assigned images/frames).
  2. Turn the user's short brief ("the dog jumps on the couch and barks")
     into an ordered shot list with pacing and subject_mentions.

Everything this pack's earlier version asked the VLM to infer -- subject
grouping, retention markers, frame-anchor status, audio role, voice-donor
pairing, character-replacement handling -- is now explicit user input via
the categorizer panel (categorize.py) and never touches the VLM at all.
That inference was the source of nearly every bug fixed earlier in this
pack's development; removing it is the fix, not further prompting.
"""

from __future__ import annotations

import json
import re
from typing import List, Optional

from PIL import Image

SYSTEM_PROMPT = """You are writing three small, specific pieces of a video prompt. You are NOT \
deciding what anything IS -- the person already told the pipeline exactly which pictures \
belong to which subject, which audio is dialogue vs a voice reference, and which images are \
concrete frame anchors. Do not second-guess or restate any of that. Your jobs:

1. DESCRIPTIONS: for each subject/anchor you're shown images for, write one short, factual, \
   generic visual description (age range/breed/type, build, coloring, clothing, distinguishing \
   features) -- 1-2 sentences. Never use a real person's name or a copyrighted character's name, \
   even if the brief uses one -- describe the generic visual traits instead. This is the ONLY \
   naming rule that still applies here; you are not classifying anything else.

2. SHOTS: default to exactly ONE shot covering the whole video, unless the brief clearly \
   describes at least two distinct beats that would genuinely need a cut between them -- a \
   real change of location, a time jump, or an action that can't read as one continuous take \
   (a move between two different places, for instance, implies two shots). Do NOT split into multiple shots just because the brief has several sentences, \
   several actions in sequence, or several subjects doing things -- a person walking in, \
   sitting down, and speaking is still one continuous shot unless the brief itself signals a \
   cut. When you do write more than one shot, each one is a distinct narrative beat the brief \
   actually describes, not one shot per reference image. Each shot's "content" is one or two \
   sentences of pure action/visual description (no camera info -- that's a separate field; \
   NO DIALOGUE WORDS EITHER, even quoted -- speech goes only in "dialogue", see below; \
   "content" may describe the ACT of speaking without quoting it, but never the words \
   themselves). Refer to each subject by its tag where it appears, in a sentence that \
   describes what THIS shot's own input actually says happens. CRITICAL: never copy example \
   wording from these instructions into your output -- every sentence must come from the \
   shot's own input and the images you were shown. Set "subject_mentions" to every subject \
   tag visible in that shot. Camera: leave "camera" null (a fixed, unmoving shot) by default -- only set a \
   motion type when the brief itself explicitly asks for camera movement (words like "pans", \
   "zooms in", "tracking shot", "push in") or unambiguously requires it (e.g. the brief describes \
   the camera following a subject across a large space). Do not invent movement to add visual \
   interest on your own judgment -- an unrequested camera move can distort framing that's \
   supposed to be preserved exactly (e.g. a fully_preserved anchor's composition), so a fixed \
   shot is the correct default whenever the brief is silent on camera, not a fallback to avoid.

3. DIALOGUE: set "dialogue" to an empty list [] UNLESS the brief itself explicitly gives someone \
   words to say (quoted speech, "X says...", a line to speak). If it does, add one entry per line \
   -- "text" must be VERBATIM what the brief wrote (translate nothing, paraphrase nothing, don't \
   fix or improve the wording), "speaker_subject" is the subject tag who says it ("Subject 1") or \
   null if the brief describes it as off-screen/narrator speech with no subject attached, \
   "language" is the real language name that text is actually written in (default "English", but \
   use "Spanish"/"Japanese"/whatever the brief's own words are actually in, or whatever language \
   the brief explicitly asks for the line to be spoken in), and "voiceover" is true only for \
   off-screen/narrator delivery. DO NOT invent dialogue the brief never asked for, under any \
   circumstance -- a silent scene stays silent unless the person explicitly wrote words for \
   someone to say. This is a hard rule, not a style preference: adding unrequested dialogue is a \
   worse mistake than an empty list.
   ATTRIBUTION: read the brief carefully to determine WHO says each line -- don't default to \
   whichever subject you mentioned most recently or whichever one seems most prominent in the \
   images. If the brief names or clearly identifies the speaker, use that subject exactly, even \
   if a different subject's voice is being referenced elsewhere (e.g. a voice-timbre note) -- a \
   note about whose VOICE SOUND is used for delivery does not change WHO the brief says is \
   speaking; attribute to whoever the brief's own words say is talking, not whoever the audio \
   setup implies. If the brief is genuinely ambiguous about who speaks a line, use null \
   (unattributed/narrator) rather than guessing a specific subject.

If you're given "Reference notes" below, those are the person's own explicit, authoritative \
instructions -- follow them exactly rather than guessing from the images alone. Some notes \
disambiguate things a picture can't show on its own: which specific person in a multi-person \
video/frame is which subject, which original character each subject replaces (when a video has \
more than one person and different subjects replace different ones), or whose voice is being \
kept vs re-timbred when a clip has more than one speaker. Others are direct instructions about \
HOW to describe that specific reference (e.g. describing a video's actual observed motion in \
chronological detail rather than writing new narrative content for it) -- follow those exactly \
too. A note attached to a tag applies to that specific reference; write shots and descriptions \
that are consistent with what it says.

Also provide "overall_soundscape" (1-4 real sentences of ambience/physical sound implied by the \
scene -- never "N/A" unless the brief asks for total silence) and "non_diegetic_music" (1-3 \
sentences, or "N/A" if the brief doesn't call for a score). NEVER include any spoken words, \
quoted or not, in "overall_soundscape" -- dialogue belongs only in each shot's "dialogue" list \
above. If voices are part of the ambience, describe them generically ("distant voices murmur") \
without repeating what anyone actually says.

Output ONLY a single JSON object, no markdown fences, no commentary:
{
  "subject_descriptions": {"Subject 1": str, "Picture 2": str, ...},
  "shots": [{"content": str, "camera": "zoom_in"|"zoom_out"|"push_in"|"pull_out"|"pan_left"|
             "pan_right"|"truck_left"|"truck_right"|"tilt_up"|"tilt_down"|"pedestal_up"|
             "pedestal_down"|"arc"|"tracking"|"static"|"shake_slight"|"shake_strong"|"pov"|
             "roll_cw"|"roll_ccw"|null, "amplitude": "small"|"large"|null,
             "speed": "slow"|"fast"|null, "onscreen_text": str|null,
             "duration_hint_s": number, "subject_mentions": ["Subject 1"],
             "dialogue": [{"speaker_subject": "Subject 1"|null, "text": str,
                            "language": "English", "voiceover": false}]}],
  "overall_soundscape": str,
  "non_diegetic_music": str,
  "style": str
}

"style" is the visual style of the TARGET video. Take it from the CHARACTER/subject \
identity references only. Ignore outfit, clothing, product and prop references when \
deciding style -- those are frequently photographed on a real mannequin or a plain \
backdrop even when the characters are drawn, and letting them decide produces \
"live-action" for an obviously animated cast. Ignore environment/background plates too. \
Use the guide's own vocabulary where it fits: "Live-action, cinematic", "2D-animated", \
"3D CG", "claymation", "watercolor", "vintage film". Anime or manga-styled character \
references are 2D-animated, so say so (e.g. "2D-animated, anime") rather than calling \
them live-action. Keep it to a few words."""


def _normalize_smart_quotes(text: str) -> str:
    """LLMs frequently emit typographic/smart quotes even when explicitly
    asked for plain JSON -- these aren't valid JSON string delimiters and
    can break parsing (or worse, get silently absorbed as content).
    Normalizes to straight quotes/apostrophes."""
    return (text.replace("\u201c", '"').replace("\u201d", '"')
                .replace("\u2018", "'").replace("\u2019", "'"))


def _strip_trailing_commas(text: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", text)


def _fix_unescaped_quotes(text: str) -> str:
    """Repairs the single most common LLM JSON-generation failure: a
    natural-language string value that contains a literal, unescaped '"'
    (e.g. quoting dialogue inside a description -- 'he says "hello" to
    her'), which breaks strict JSON parsing at some point well past where
    the actual mistake is -- exactly the "Expecting ',' delimiter" shape
    seen in practice, deep into a long response. Walks the text tracking
    whether we're inside a string; when a '"' appears while already inside
    one, only treats it as the real closing quote if the next non-
    whitespace character is a JSON structural character (,:}]) or end of
    text -- otherwise it's escaped as literal content instead. Not a full
    JSON5/lenient parser, just a targeted fix for this one well-documented
    failure shape, which is by far the most common one in practice."""
    out = []
    in_string = False
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "\\" and in_string and i + 1 < n:
            out.append(ch)
            out.append(text[i + 1])
            i += 2
            continue
        if ch == '"':
            if not in_string:
                in_string = True
                out.append(ch)
            else:
                j = i + 1
                while j < n and text[j] in " \t\r\n":
                    j += 1
                if j >= n or text[j] in ",:}]":
                    in_string = False
                    out.append(ch)
                else:
                    out.append('\\"')
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _extract_json(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        if not raw:
            raise ValueError(
                "The VLM returned a completely empty response (no text at all), not just "
                "malformed JSON. This usually means the chat handler doesn't actually "
                "support this model's vision format (it loads and runs without erroring, "
                "but produces nothing) -- most commonly a Qwen-VL model loaded with "
                "llama-cpp-python's default Llava15ChatHandler, which doesn't understand "
                "Qwen-VL's vision tokens. Check that a Qwen-VL-capable chat handler is "
                "actually available if you're using a Qwen-VL GGUF. Less likely causes: "
                "the prompt + images exceeded n_ctx, or the mmproj file doesn't match "
                "the main model file."
            )
        raise ValueError(f"VLM did not return JSON:\n{raw[:2000]}")

    candidate = match.group(0)
    # Progressively more aggressive repairs -- most VLM JSON that fails to parse
    # on the first try fails for one of these specific, well-known reasons, not
    # because the overall structure is wrong. Try cheapest/safest fixes first.
    attempts = [candidate]
    smart_fixed = _normalize_smart_quotes(candidate)
    attempts.append(smart_fixed)
    comma_fixed = _strip_trailing_commas(smart_fixed)
    attempts.append(comma_fixed)
    attempts.append(_fix_unescaped_quotes(comma_fixed))

    last_err = None
    for attempt in attempts:
        try:
            return json.loads(attempt)
        except json.JSONDecodeError as e:
            last_err = e
            continue

    pos = last_err.pos if last_err else 0
    context = candidate[max(0, pos - 150):pos + 150]
    raise ValueError(
        f"VLM returned JSON that couldn't be parsed even after repair attempts (smart-quote "
        f"normalization, trailing-comma removal, unescaped-quote fixing). This is usually the "
        f"model writing a natural-language field (a description or shot content) that breaks "
        f"strict JSON syntax in a way none of those repairs caught -- if it keeps happening, "
        f"a stronger/less quantized model tends to follow JSON formatting more reliably. "
        f"Parse error: {last_err}\nText around the failure point:\n...{context}..."
    )


# Qwen-VL encodes roughly one token per 28x28 pixel block, and llama.cpp warns
# that these models "require at minimum 1024 image tokens to function correctly".
# Both ends of that matter:
#   * too LARGE (a 2K sheet at ~4000 tokens each) overflows the context;
#   * too SMALL is worse than it sounds -- an image under the token floor can be
#     dropped by MTMD entirely, which surfaces as the very confusing "not all
#     media inputs were consumed by MTMD chunks (consumed=3, media_items=4)"
#     even though there is loads of context free (the three that survived came
#     to only ~1200 tokens combined).
# Letterboxing every image onto a fixed 896x896 canvas pins each one at exactly
# 32x32 = 1024 tokens: at the floor the model wants, uniform and predictable, and
# ~4k tokens for four images. Padding keeps the aspect ratio honest rather than
# stretching a portrait reference into a square.
VLM_CANVAS = 896

# Encoding one 896x896 image measured ~19s to encode plus ~21s to decode on a
# 4B vision model, and on builds where the batched path is broken that cost is
# paid PER IMAGE. Token count scales with area, so halving the side quarters the
# work. 896 is the accurate setting; the smaller ones trade detail for speed.
CANVAS_PRESETS = {"accurate (896px)": 896, "faster (704px)": 704,
                   "fastest (512px)": 512}


# Set once llama-cpp-python's batched multi-image path is observed dropping
# media in this process, so later runs go straight to one-image-per-call.
_BATCHED_MEDIA_BROKEN = False


def _pick_style(styles: dict) -> str:
    """Chooses the target video's style from per-image styles, preferring the
    CHARACTER references. Labels carry a kind hint for anything that isn't a
    person (", a clothing", ", an environment"), so a label without one is a
    person identity reference. A photoreal mannequin standing in for an outfit,
    or a live-action backdrop, should never decide the style of an animated
    cast -- which is exactly how an all-anime reference set came out as
    "Live-action, cinematic"."""
    if not styles:
        return ""
    person = [v for k, v in styles.items()
              if k.startswith("Subject") and "," not in k.split("(")[-1]]
    if person:
        # Most common answer among the character references.
        return max(set(person), key=person.count)
    subj = [v for k, v in styles.items() if k.startswith("Subject")]
    if subj:
        return max(set(subj), key=subj.count)
    return next(iter(styles.values()))


# Describing one image costs tens of seconds of vision encoding -- measured at
# ~40s per 896x896 image (19s encode + 21s decode) on a real setup. That is paid
# again on every run even when the references haven't changed, which makes
# iterating on the text unbearable. Descriptions are therefore cached on the
# image's pixel content for the life of the process: change the picture and it
# is re-described, change only the wording and it is not.
_DESC_CACHE: dict = {}


def _image_key(img) -> str:
    """Content hash of an image, so an identical reference hits the cache even
    if it arrived by a different route."""
    import hashlib
    try:
        return hashlib.sha1(img.tobytes()).hexdigest()[:16] + f"_{img.size[0]}x{img.size[1]}"
    except Exception:
        return ""


def _dbg_line(msg: str) -> None:
    try:
        from . import debug_log as _d
        _d.line(msg)
    except Exception:
        pass


def set_canvas(side: int) -> None:
    """Sets the square size every image is letterboxed to before encoding."""
    global VLM_CANVAS
    try:
        VLM_CANVAS = max(256, min(1024, int(side)))
    except (TypeError, ValueError):
        pass


def _fit_canvas(img: Image.Image, side: int = None) -> Image.Image:
    """Scales to fit inside side x side (preserving aspect) and centres the result
    on a black canvas of exactly that size."""
    side = int(side or VLM_CANVAS)
    w, h = img.size
    if w <= 0 or h <= 0:
        return img
    scale = min(side / float(w), side / float(h))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = img.convert("RGB").resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (side, side), (0, 0, 0))
    canvas.paste(resized, ((side - nw) // 2, (side - nh) // 2))
    return canvas


def _image_to_data_uri(img: Image.Image) -> str:
    import base64
    import io
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


class DescribeEngine:
    """Thin wrapper reusing whatever VLM backend is configured -- same
    llama.cpp / transformers backends as before, just a narrower prompt."""

    def __init__(self, backend):
        self.backend = backend   # a LlamaCppVLM or TransformersVLM instance

    def describe(self, images: List[Image.Image], labels: List[str],
                 brief: str, duration_s: float, notes: Optional[List[tuple]] = None,
                 shot_briefs: Optional[List[str]] = None) -> dict:
        # Reuse the backend's chat-completion plumbing directly rather than its
        # (differently-shaped) `analyze` method, since this prompt's schema differs.
        images = [_fit_canvas(im) for im in (images or [])]
        try:
            from . import debug_log as _dbg
            if _dbg.on():
                _dbg.section("VLM IMAGE SIZES (letterboxed to canvas)")
                for _lb, _im in zip(labels, images):
                    _dbg.bullet(f"{_lb}: {_im.size[0]}x{_im.size[1]}")
        except Exception:
            pass

        notes_block = ""
        if notes:
            notes_block = "Reference notes (authoritative -- follow exactly, see system prompt):\n" + \
                "\n".join(f"  {tag}: {text}" for tag, text in notes) + "\n"

        shot_briefs = [b.strip() for b in (shot_briefs or []) if b and b.strip()]
        if shot_briefs:
            # Fixed per-shot mode: the person split their own narrative into separate
            # boxes, one per shot, specifically to stop content/dialogue from one shot
            # bleeding into another when a single VLM call decided both the shot count
            # AND the boundaries between them. Shot count and order are no longer the
            # VLM's decision at all here -- it's a strict 1:1 transcription task per box.
            numbered = "\n".join(f"  Shot {i+1}: {b}" for i, b in enumerate(shot_briefs))
            user_text = (
                f"Target video length: {duration_s:.1f} seconds.\n"
                f"The user has split their narrative into {len(shot_briefs)} pre-defined "
                f"shots, given below in order. You MUST output EXACTLY {len(shot_briefs)} "
                f"entries in \"shots\", one per input, in the same order -- never merge two "
                f"inputs into one shot, never split one input into multiple shots, never add "
                f"or remove shots. Each output shot's content/camera/dialogue must be based "
                f"ONLY on its own corresponding input below, never on what another shot's "
                f"input describes -- this is specifically to prevent one shot's action or "
                f"dialogue from bleeding into a neighboring shot.\n"
                f"{numbered}\n"
                f"{notes_block}"
                f"Images below are labelled, in order: {', '.join(labels) or '(none)'}\n"
                "Write descriptions and shots as JSON per the schema."
            )
        else:
            user_text = (
                f"Target video length: {duration_s:.1f} seconds.\n"
                f"User's brief: {brief.strip() or '(none given -- infer minimal shots from images)'}\n"
                f"{notes_block}"
                f"Images below are labelled, in order: {', '.join(labels) or '(none)'}\n"
                "Write descriptions and shots as JSON per the schema."
            )
        # Reuse whatever token budget the backend was configured with (falls back to a
        # sane default for the transformers path, which has no shared config object) --
        # this used to be a separate hardcoded 1500 here, so bumping VLMConfig.max_tokens
        # elsewhere never actually affected this pipeline's own calls.
        max_tokens = getattr(getattr(self.backend, "config", None), "max_tokens", 2200)
        if hasattr(self.backend, "_llm"):  # LlamaCppVLM
            self.backend._ensure_loaded()

            def _one_call(text: str, imgs):
                content = [{"type": "text", "text": text}]
                for im in imgs:
                    content.append({"type": "image_url",
                                     "image_url": {"url": _image_to_data_uri(im)}})
                resp = self.backend._llm.create_chat_completion(
                    messages=[{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": content}],
                    temperature=0.4, max_tokens=max_tokens,
                )
                return resp["choices"][0]["message"]["content"]

            def _describe_one_at_a_time():
                """Fallback for llama-cpp-python's flaky multi-image path.

                Symptom: every bitmap is created ("4 bitmaps were successfully
                created") but fewer MTMD chunks come back, and the count VARIES
                between runs on identical input (3 of 4, then 2 of 4) -- with the
                images uniformly sized and the total only ~1.5k tokens. That rules
                out context and image size; it's a defect in the handler's batched
                path, which builds bitmaps across several threads. One image per
                call avoids that path entirely: slower, but it actually returns
                every description. The shot list needs no images at all, so it goes
                in a final text-only call.
                """
                merged, styles = {}, {}
                for lbl, im in zip(labels, images):
                    key = _image_key(im)
                    hit = _DESC_CACHE.get(key) if key else None
                    if hit:
                        # Same pixels as a previous run: reuse rather than spend
                        # another ~40s encoding an identical image.
                        merged[lbl] = hit.get("desc", "")
                        if hit.get("style"):
                            styles[lbl] = hit["style"]
                        _dbg_line(f"cached description reused for {lbl}")
                        continue
                    txt = (f"Describe this single reference image, labelled: {lbl}.\n"
                           f"{notes_block}"
                           f'Reply with ONLY {{"subject_descriptions": {{"<label>": "..."}}, '
                           f'"style": "<visual style of THIS image>"}} using exactly that '
                           f"label as the key. For \"style\" use the guide vocabulary "
                           f"(\"Live-action, cinematic\", \"2D-animated, anime\", "
                           f"\"3D CG\", \"claymation\", \"watercolor\", "
                           f"\"vintage film\") describing only what THIS image looks like.")
                    try:
                        part = _extract_json(_one_call(txt, [im]))
                        got = part.get("subject_descriptions") or {}
                        merged.update(got)
                        st = (part.get("style") or "").strip()
                        if st:
                            styles[lbl] = st
                        if key:
                            _DESC_CACHE[key] = {
                                "desc": next(iter(got.values()), ""), "style": st}
                    except Exception as e2:
                        _dbg_line(f"description failed for {lbl}: {e2}")
                shots_txt = user_text + ("\n\nThe reference images have already been "
                                          "described separately; return an empty "
                                          '"subject_descriptions" object here and focus '
                                          "on shots, overall_soundscape and "
                                          "non_diegetic_music.")
                out = _extract_json(_one_call(shots_txt, []))
                out["subject_descriptions"] = merged
                chosen = _pick_style(styles)
                if styles:
                    _dbg_line("per-image styles:")
                    for k, v in styles.items():
                        _dbg_line(f"    {k} -> {v}")
                    _dbg_line(f"  chosen (character references win): {chosen}")
                if chosen:
                    out["style"] = chosen
                return out

            global _BATCHED_MEDIA_BROKEN
            if _BATCHED_MEDIA_BROKEN and len(images) > 1:
                # Already proven broken in this session -- skip straight to the
                # reliable path instead of burning a failing call every run.
                return _describe_one_at_a_time()
            try:
                raw = _one_call(user_text, images)
            except RuntimeError as exc:
                if "media inputs were consumed" in str(exc) and len(images) > 1:
                    # Don't fail the run over a known-flaky batched path -- redo it
                    # one image per call, which reliably encodes all of them.
                    _dbg_line(f"multi-image call dropped media ({exc}); "
                              f"retrying one image at a time")
                    _BATCHED_MEDIA_BROKEN = True
                    return _describe_one_at_a_time()
                raise
            return _extract_json(raw)
        else:  # TransformersVLM
            self.backend._ensure_loaded()
            import torch
            content = [{"type": "image", "image": img} for img in images]
            content.append({"type": "text", "text": user_text})
            messages = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": content}]
            prompt = self.backend._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = self.backend._processor(text=[prompt], images=images,
                                              return_tensors="pt").to(self.backend._model.device)
            with torch.no_grad():
                out = self.backend._model.generate(**inputs, max_new_tokens=max_tokens,
                                                     temperature=0.4, do_sample=True)
            text = self.backend._processor.batch_decode(
                out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            return _extract_json(text)
