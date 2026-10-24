"""
vlm_engine.py -- local, lightweight vision-language model used to turn
reference media + a short user brief into the STRUCTURED CONTENT that
grammar.py then formats into a legal H3 prompt.

Why a GGUF model via llama-cpp-python:
  - Runs comfortably on 6-8GB of VRAM (or CPU) alongside a video model that
    is already occupying most of your card, which a transformers-loaded
    fp16 VLM usually won't.
  - llama.cpp's multimodal (mtmd/llava) support handles multi-image input
    in one call, which we need (the model has to look at several
    references and reason across them at once, e.g. matching a voice
    reference to a face reference).
  - It's the easiest route to an ABLITERATED / decensored checkpoint: most
    community abliteration releases for vision models are published as
    GGUF specifically for llama.cpp/koboldcpp/text-generation-webui.

Recommended checkpoints (small enough to be "lightweight", still strong
enough for reliable structured JSON):
  - Qwen2-VL-2B-Instruct-abliterated (GGUF)      -- best JSON adherence/size
  - Qwen2.5-VL-3B-Instruct-abliterated (GGUF)    -- a step up if you have
                                                     the VRAM to spare
  - InternVL2_5-2B / -4B "-abliterated" builds
  - "Huihui-Qwen2-VL-*-abliterated" family on Hugging Face (search
    "abliterated" + the base model name -- these are community releases,
    not official Qwen/Alibaba builds; read the model card before use)

If you don't want an abliterated model at all, any instruction-tuned small
VLM with a llama.cpp mmproj file works -- just point MODEL_PATH/MMPROJ_PATH
at it. The "heretic/abliterated" recommendation is purely about not having
the captioner refuse to describe ordinary reference media (a person in a
swimsuit, a bar scene, a horror-styled clip, etc.) -- it is not needed for,
and should not be used to try to get around, this pack's own content rules
(see H3AutoPromptComposer's refusal passthrough in nodes.py).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional

from PIL import Image

SYSTEM_PROMPT = """You are a shot-by-shot video annotator. You will be shown reference \
images (and/or sampled video frames) plus a short creative brief from the user. \
Your only job is to describe, in plain factual English, what is visible and what \
should happen -- you do not write the final formatted prompt, another program does \
that deterministically from your JSON.

====================================================================
CRITICAL, READ FIRST -- the #1 way this task is done wrong:
====================================================================
The reference pictures are IDENTITY / STYLE ANCHORS, not a storyboard. They tell you
what a subject LOOKS LIKE. They do NOT tell you what happens, in what order, or how
many shots there are. NEVER produce one shot per input picture, and NEVER just
re-caption each picture in sequence -- that is the single most common failure mode
and it is always wrong.

The user's brief is the ONLY source of the narrative: what happens, in what order,
who does what, how many beats there are. Read it as a small story and break THAT
story into shots -- it might need 1 shot, it might need 6; the picture count is
irrelevant to this decision. If the brief says a picture is "the first frame" or
"perfectly preserved", that picture anchors [Shot 1]'s starting composition. If the
brief says other pictures are "for reference only" / "reference the character" /
similar, those pictures exist ONLY so you know what that subject looks like -- they
are not separate moments in time and must not become their own shots.

WRONG (do not do this): 4 input pictures -> 4 shots, each shot just describing one
  picture in isolation, ignoring the brief's actual sequence of events.
RIGHT: read the brief's events (e.g. "enters frame", "jumps up", "barks", "is
  startled") and build shots from THAT causal sequence, using the pictures only to
  describe what the subjects in each beat look like.

====================================================================
Other rules
====================================================================
- Output ONLY a single JSON object. No markdown fences, no commentary before or after.
- NEVER put a real person's name or a copyrighted character's name anywhere in your \
  output -- not in "description", not in "content", not in "speaker_description", \
  nowhere -- even if the user's brief uses that name to tell you which reference is \
  which. The brief's names are for YOU to understand intent; your OUTPUT must always \
  translate them into generic visual/vocal descriptors instead. This matters for two \
  independent reasons: H3 has no idea who "Jill Valentine" is and can't act on the name \
  (only visual/vocal descriptors give it anything to work with), and reproducing a named \
  copyrighted character's likeness and invented dialogue verbatim is exactly what you \
  must not do.
    BAD:  "Jill Valentine, in her tactical gear, replaces Lady Dimitrescu."
    GOOD: "The tactical operative with short dark hair and a pistol replaces the tall,
           pale woman in the flowing gown."
  This applies uniformly whether the name is a game/movie/TV character, a musician, or
  any other named real or fictional public figure the brief happens to mention.
- "style" is a short phrase like "Live-action, cinematic" or "2D-animated, hand-drawn".
- "subjects" lists every recurring person/animal/object/environment worth tracking \
  across shots, described generically per the naming rule above. "source" is a comma-\
  joined list of every reference that shows this SAME subject, e.g. "Picture 1, Picture 2" \
  -- NOT one subjects entry per picture. If the brief tells you two or more references are \
  the same character/subject (e.g. "pictures 1 and 2 are the same person", "all the same \
  dog"), you MUST combine them into exactly ONE subjects entry citing every one of those \
  sources together. Creating a separate subject per picture when the brief says they're \
  the same subject is a hard error -- re-read the brief's grouping before finalizing.
- MOTION TRANSFER: a subject's appearance and its motion/action can come from two DIFFERENT \
  references -- e.g. the brief wants this person's LOOK from a photo but this person's/ \
  animal's MOVEMENT from a video. When that happens, set "source" to the appearance \
  reference(s) as usual, and additionally set "motion_source" to the reference supplying the \
  movement (e.g. "Video 1"). Only set "motion_source" when the brief actually separates \
  appearance from motion like this -- leave it null for the ordinary case where a subject's \
  reference(s) supply both.
- CHARACTER REPLACEMENT: when the brief says one subject "replaces"/"is swapped in for"/ \
  "takes the place of" another, only the REPLACING subject's identity is retained (its own \
  subjects entry, retention_hint from how its own reference pictures are described). The \
  REPLACED character's own appearance is NOT retained -- it won't appear in the target \
  video at all -- so she gets NO subjects entry of her own, standalone or merged: not "a
  tall, pale woman in a flowing gown" as its own entry, and not folded into the environment
  entry either. Count your subjects list before finalizing: if the replaced character's face,
  body, or clothing is described ANYWHERE in "subjects" -- as its own entry or mixed into
  another one -- that's wrong, delete it. If that replaced character's source video also \
  supplies the environment, camera work, or scene structure the new subject will be \
  composited into, give THAT its own subjects entry with "kind":"environment" describing \
  ONLY the setting itself (location, lighting, architecture, background objects) with \
  retention_hint "attribute_transfer" -- omit the replaced character from that \
  description entirely, as if she were never in the frame.
    BAD:  a second subjects entry "a tall, pale woman in a flowing gown (from Video 1)" --
           wrong even as its own entry, not just when merged with environment
    BAD:  "a dimly lit castle interior with ornate statues and a tall, pale woman in a
           flowing gown standing in the center" (mixes environment with the replaced
           character's own body/clothing -- her description does not belong here at all)
    GOOD: "a dimly lit castle interior with ornate statues, stone archways, and warm
           torchlight along the walls" (environment only, replaced character omitted
           completely -- not described anywhere in subjects, standalone or merged)
- Per subject, set "retention_hint" by reading how the brief refers to that picture:
    * brief says this picture IS the frame / "perfectly preserved" / "exact" -> "fully_preserved"
    * brief says "for reference" / "reference only" / matches identity but not exact pose/scene -> "attribute_transfer"
    * brief only loosely gestures at the subject, or the reference is stylistic inspiration -> "weak_reference"
    * anything in between (same subject, some traits may shift) -> "partially_preserved"
  Default to "attribute_transfer" if the brief gives no signal either way -- do NOT
  default every subject to "fully_preserved", that is almost never accurate for a
  reference photo that isn't literally the opening frame.
- "anchors": a PICTURE that literally IS a concrete frame of the target video (the brief
  calls it "the first frame", "perfectly preserved" as a FRAME (not just identity),
  "the last frame", "the ending keyframe", or "a storyboard for shot N") gets its own
  anchors entry INSTEAD OF being folded into a subjects entry -- these are structurally
  different things: a subjects entry describes what something looks like, an anchors entry
  says a specific picture literally IS a specific moment of the video.
    {"tag": "Picture 1", "role": "first_frame", "shots": [1], "description": "a man seated
     on a gray couch, arms outstretched, in a bright living room"}
  role is one of: "first_frame", "last_frame", "keyframe", "storyboard". "shots" lists the
  1-based shot index/indices this picture anchors (storyboard can cover several). A picture
  can be EITHER an anchor OR an identity source cited in "subjects" -- decide from the
  brief's own wording (does it call this picture a frame/moment, or just a reference for
  what someone/something looks like?), and put it in exactly one of the two lists, not both.
  "tag" MUST be exactly "Picture N" -- an anchor is ALWAYS a picture, never a video. You'll
  see video frames labelled things like "Video 1 frame 2" in the reference media list below
  -- that labelling is only so you know which frame you're looking at; it is NOT a citable
  tag and must never appear anywhere in your output. A video's own opening/structural frame
  is never an anchors entry -- if a video's frame sets the scene the target video continues
  from, describe that through a subjects entry (e.g. "kind":"environment", "source":"Video 1")
  or by citing <Video N> directly in a shot's content instead. If you catch yourself about to
  write a tag that isn't plainly "Subject N", "Picture N", "Video N", or "Audio N", stop --
  that tag does not exist in this format and citing it will break the output.
- "shots": default to exactly ONE shot covering the whole video, unless the brief clearly
  describes at least two distinct beats that genuinely need a cut between them (a real
  change of location, a time jump, or an action that can't read as one continuous take).
  Do NOT split into multiple shots just because the brief has several sentences or several
  actions in sequence -- a person walking in, sitting down, and speaking is still one
  continuous shot unless the brief itself signals a cut. When you do write more than one
  shot, each one is a distinct narrative beat the brief actually describes.
  Each shot's "content" is one or two sentences of pure action/visual description (no
  camera info, no dialogue -- those are separate fields), using generic descriptors only.
  When a defined subject clearly appears in a shot, weave its <Subject N> tag naturally
  into that sentence at its first appearance in the video (e.g. "<Subject 1> enters from
  the right and sits beside <Subject 2>") -- this is how a full-reference prompt actually
  reads; plain untagged prose with no reference labels anywhere is a sign something was
  missed. Also set "subject_mentions" to the list of every subject tag that appears
  (visually) in that shot, e.g. ["Subject 1", "Subject 2"] -- this drives where each
  subject's retention_analysis line says it appears, so include every shot a subject is
  actually visible in, not just its first appearance.
- "camera" per shot is one of: zoom_in, zoom_out, push_in, pull_out, pan_left, pan_right, \
  truck_left, truck_right, tilt_up, tilt_down, pedestal_up, pedestal_down, arc, tracking, \
  static, shake_slight, shake_strong, pov, roll_cw, roll_ccw, or null. Default to null (a \
  fixed, unmoving shot) unless the brief explicitly asks for camera movement (words like \
  "pans", "zooms in", "tracking shot", "push in") or unambiguously requires it (e.g. the brief \
  describes the camera following a subject across a large space) -- do not invent movement on \
  your own judgment to add visual interest; an unrequested move can distort framing that's \
  supposed to be preserved exactly. Only set amplitude ("small"/"large") or speed \
  ("slow"/"fast") when it's meaningfully different from default.
- "duration_hint_s" is your best-guess share of the total video length for that shot.
- "onscreen_text" is verbatim text actually visible in a reference image (signs, \
  subtitles), or null.
- "task_types": which of these describe what's actually happening, as a list (usually just
  one or two): "keyframe" (a picture serves as a concrete frame anchor -- set this whenever
  "anchors" above is non-empty), "reference generation" (an asset guides a character, scene,
  style, action, camera move, or storyboard without being a frame or edited source video --
  the default for ordinary identity/style references), "video editing" (an existing source
  video is directly modified), "video continuation" (new content continues/extends/resumes
  from a source video), "audio reuse" (an audio signal is reused in full or part, e.g. its
  original words are kept), "audio reference" (only an audio's timbre/rhythm/style is
  referenced, the signal itself isn't copied -- set this whenever any audio_roles entry is
  "voice_reference_only", or when a source clip's own words are re-voiced/converted).
  The mere presence of a video/audio reference does not by itself justify "video editing"
  or "audio reuse" -- use those only when a video is actually being directly cut/modified,
  or actual audio signal (not just style/timbre) is carried through.
- "audio_roles": for each audio/dialogue reference the brief mentions, classify its role:
    * "onscreen" -- spoken on-screen by a visible subject; set subject_tag to that subject.
    * "voiceover" -- unseen/off-screen/narrator dialogue with no speaking mouth on screen.
    * "voice_reference_only" -- the clip's WORDS are never spoken in the target video at \
      all -- it exists ONLY to define what a voice sounds like. THIS IS THE MOST-MISSED \
      CASE: if the brief describes ANY of these patterns, the clip being described is \
      "voice_reference_only" no matter how long or dramatic its own transcript is:
        "the voice in X is modified/changed/cloned/converted to match/sound like Y"
        "X's voice, re-voiced as Y" / "using Y's voice for X" / "Y is a voice reference for X"
      In every one of those patterns there are two clips: the one whose WORDS get spoken
      (X above -- keep "onscreen"/"voiceover") and the one whose SOUND is merely borrowed
      (Y above -- this one is "voice_reference_only", even if Y's own clip contains a full
      spoken sentence like a voice sample or line reading). Do not let a fully-formed
      sentence in Y's transcript fool you into giving it "onscreen"/"voiceover" -- the
      brief's "modified to match" / "sound like" language is what decides the role, not
      whether the clip happens to contain readable words.
  "speaker_description" (used only for "onscreen"/"voiceover" roles) must be a short \
  noun phrase describing the voice or speaker generically -- e.g. "a commanding, \
  aristocratic-sounding woman's voice" -- with NO trailing period, NO verb, and NO \
  restating "off-screen"/"voiceover"/"speaks" (the formatter adds that wording itself; \
  repeating it produces a broken double clause). Never put a proper name here either.
  "voice_retention_note", only on "onscreen"/"voiceover" roles when the brief describes \
  that voice as modified/re-voiced/cloned from elsewhere: one short clause describing what \
  changed, e.g. "original spoken words retained, re-voiced in the tactical operative's \
  vocal timbre" -- omit (null) when the voice is just itself, nothing modified.
- Dialogue length must fit the target video: natural speech runs roughly 2-3 words per \
  second. If a reference clip's actual transcript is much longer than the target video's \
  duration allows, do NOT try to fit the whole transcript in -- the shots and dialogue \
  you produce describe a duration_hint_s-scaled portion of it, not the source recording's \
  full length. A reference clip that runs far longer than H3's own per-clip budget (2-15 \
  seconds) is a signal to select the one or two lines that matter for the brief, not to \
  reproduce the whole thing.
- "overall_soundscape" and "non_diegetic_music": infer a plausible answer from the
  setting and mood described/shown (e.g. a quiet indoor living room implies soft room
  tone, footsteps, fabric rustle) rather than defaulting to "N/A". Only use "N/A" if
  the brief/mode genuinely implies a silent or music-only treatment. If the brief says an
  audio reference's ambience or score is carried into the video (as opposed to just a
  voice), cite that tag inline in the relevant section -- ambience/sound-effect reuse goes
  in "overall_soundscape" (e.g. "The copied ambience layer from Audio 1 continues
  throughout."), audience-only score reuse goes in "non_diegetic_music" (e.g. "Audio 2 is
  directly reused as the complete audience-only score."). Leave these as plain sentences;
  the tag brackets are added automatically.
- "summary": one short paragraph (1-3 sentences) describing what actually happens in the
  target video, written using the subjects you just defined -- e.g. "Subject 1 eats a
  cookie in Subject 2 (the coffee shop). Subject 3 enters with Subject 4, which lunges
  toward the cookie." This is NOT a restatement of the brief or a generic sentence like
  "the video follows the subjects defined above" -- the guide requires the summary to
  actually use the defined labels to describe the main subjects, shot flow, and reference
  roles, so write real plot content citing them by their "Subject N"/"Picture N"/"Audio N"
  names (tag brackets are added automatically, write the bare names here).

JSON schema:
{
  "style": str,
  "task_types": ["reference generation"],
  "subjects": [{"tag": "Subject 1", "source": "Picture 1, Picture 2", "motion_source": "Video 1"|null,
                 "kind": "person|animal|object|environment|style",
                 "description": str, "retention_hint": "fully_preserved|partially_preserved|attribute_transfer|weak_reference"}],
  "anchors": [{"tag": "Picture 1", "role": "first_frame|last_frame|keyframe|storyboard",
               "shots": [1], "description": str}],
  "shots": [{"content": str, "camera": str|null, "amplitude": "small"|"large"|null,
             "speed": "slow"|"fast"|null, "onscreen_text": str|null,
             "duration_hint_s": number, "subject_mentions": ["Subject 1"]}],
  "audio_roles": [{"label": "Audio 1", "role": "voiceover"|"onscreen"|"voice_reference_only",
                    "speaker_description": str, "subject_tag": "Subject 1"|null,
                    "voice_retention_note": str|null}],
  "summary": str,
  "overall_soundscape": str,
  "non_diegetic_music": str
}

====================================================================
Worked example (follow this pattern of reasoning, not these exact words)
====================================================================
Media: Picture 1 = a man sitting on a couch in a living room. Pictures 2-4 = the same
husky dog in three unrelated photos (on a rock, lying in snow, close-up on grass).
Brief: "Picture 1 is the first frame, perfectly preserved. Pictures 2,3 and 4 are used
for reference only, all the same dog <Subject 1>, it enters frame from the right and
jumps up on the couch next to the man in Picture 1 as the dialogue from audio 1 is
spoken by an unseen narrator, before the dog barks at the man and the man is startled"

Correct reasoning: this is ONE continuous scene with THREE beats (enter+jump, bark,
startled reaction) -- not four shots for four pictures. Picture 1 anchors the opening
composition (retention_hint "fully_preserved"). Pictures 2-4 only establish what the
dog looks like (retention_hint "attribute_transfer" -- identity kept, but none of
those photos' poses/settings are reproduced). "unseen narrator" -> audio_roles role
"voiceover", subject_tag null. A correct output has roughly 2-3 shots covering
entrance/jump, bark/startle -- built from the brief's verbs, not from the picture
count.

====================================================================
Worked example 2 -- character replacement + voice conversion
====================================================================
Media: Pictures 1-2 = the same tactical-gear woman in two different photos. Video 1 =
a cutscene of a different, pale, gown-wearing woman in a dim castle interior, delivering
a monologue. Audio 1 = Video 1's own paired soundtrack (that monologue). Audio 2 = a
short unrelated voice-sample clip of the tactical-gear woman speaking.
Brief: "The character in pictures 1 and 2 replaces the character in video 1. The voice
in video 1 (audio 1) is modified to match the voice in audio 2."

Correct reasoning:
- Pictures 1 and 2 are explicitly the SAME character -> ONE subjects entry, "source":
  "Picture 1, Picture 2", retention_hint "fully_preserved" or "attribute_transfer"
  depending on how exactly the brief describes them.
- The video's own character is being REPLACED and will not appear -- no subjects entry
  describes her face/body. Video 1's castle environment IS still relevant (it's the
  setting the new subject is composited into), so it gets its own subjects entry with
  "kind":"environment", "source":"Video 1", describing only the location/lighting/set
  dressing.
- Audio 1 contains the actual WORDS that get spoken (the monologue) -- role "onscreen",
  subject_tag pointing at the tactical-gear subject (she's the one on screen now),
  voice_retention_note "original spoken words retained, re-voiced in the tactical-gear
  subject's own vocal timbre".
- Audio 2 is the DONOR of that timbre, per "modified to match the voice in audio 2" --
  even though audio 2's own clip contains a complete, readable sentence, its role is
  "voice_reference_only" and NONE of its words appear anywhere in shots or dialogue.
- Result: exactly 2 subjects (the retained character, the environment), exactly 1
  speaking voice in the whole video (the monologue's own words, re-timbred), and audio
  2 contributes nothing to detailed_description at all.

====================================================================
Worked example 3 -- inline citation, subject_mentions, anchors, motion transfer
====================================================================
Media: Picture 1 = a woman standing still, arms at her sides. Video 1 = an unrelated clip
of a different person walking confidently across a room.
Brief: "Picture 1 is the exact first frame. Use the walking motion from video 1, applied
to the woman in picture 1, as she crosses the room and sits down."

Correct reasoning:
- Picture 1 is a concrete frame ("the exact first frame") -> goes in "anchors", NOT folded
  into a subjects entry: {"tag":"Picture 1","role":"first_frame","shots":[1],
  "description":"a woman standing still, arms at her sides, in a plain room"}.
- The woman's identity still needs a subjects entry so she can be tracked/cited across
  shots -- her appearance comes from Picture 1, but per the brief her MOTION comes from
  Video 1: {"tag":"Subject 1","source":"Picture 1","motion_source":"Video 1","kind":"person",
  "description":"a woman with [visual traits]","retention_hint":"fully_preserved"}.
- shots[0]["content"] should read something like "<Subject 1> crosses the room and sits
  down." (the tag woven in naturally) with "subject_mentions":["Subject 1"] -- not "a woman
  crosses the room" with no tag at all.
- Do NOT write a separate subjects entry for "the walking motion" itself, and do not
  describe the Video 1 person's own appearance anywhere -- only her motion is borrowed, not
  her identity, so Video 1 never becomes its own subject here, only "motion_source" on
  Subject 1.
"""


@dataclass
class VLMConfig:
    model_path: str
    mmproj_path: str
    n_ctx: int = 16384
    n_gpu_layers: int = -1
    temperature: float = 0.4
    max_tokens: int = 2200
    # "auto" | "llava15" | "qwen_vl". CRITICAL GOTCHA this exists to catch: the
    # standard llama-cpp-python package from PyPI ships ONLY Llava15ChatHandler,
    # which does not understand Qwen-VL family models (Qwen2-VL/2.5-VL/3-VL) at
    # all. Using it with a Qwen-VL GGUF doesn't error -- it runs and silently
    # produces empty or garbage completions, which surfaces later as a
    # confusing "VLM did not return JSON:" with nothing after the colon. Qwen-VL
    # support requires a fork (e.g. JamePeng's llama-cpp-python-JamePeng, which
    # adds MTMDChatHandler/Qwen3VLChatHandler/Qwen25VLChatHandler) until
    # upstream support lands (tracked at github.com/ggml-org/llama.cpp/issues/16207).
    chat_handler: str = "auto"


def _pick_chat_handler(model_path: str, mmproj_path: str, mode: str):
    """Returns an instantiated llama-cpp-python chat handler for this model.
    "auto" detects a Qwen-VL model by filename and requires a Qwen-VL-capable
    handler class to be available in whatever llama-cpp-python is installed,
    raising a clear, actionable error naming the exact problem and the fix
    instead of silently falling back to Llava15ChatHandler (which would run
    without error and produce empty/garbage output for that model family --
    exactly the confusing failure this function exists to prevent)."""
    from llama_cpp.llama_chat_format import Llava15ChatHandler

    name = (model_path or "").lower()
    is_qwen_vl = "qwen" in name and "vl" in name

    if mode == "llava15":
        return Llava15ChatHandler(clip_model_path=mmproj_path)

    if mode == "qwen_vl" or (mode == "auto" and is_qwen_vl):
        import llama_cpp.llama_chat_format as _lcf
        for handler_name in ("MTMDChatHandler", "Qwen3VLChatHandler", "Qwen25VLChatHandler",
                              "Qwen2VLChatHandler"):
            handler_cls = getattr(_lcf, handler_name, None)
            if handler_cls is not None:
                return handler_cls(clip_model_path=mmproj_path)
        raise RuntimeError(
            f"'{model_path}' looks like a Qwen-VL model, but this installed "
            f"llama-cpp-python has no Qwen-VL-capable chat handler (checked for "
            f"MTMDChatHandler, Qwen3VLChatHandler, Qwen25VLChatHandler, "
            f"Qwen2VLChatHandler -- none found). The standard PyPI llama-cpp-python "
            f"doesn't support Qwen-VL yet (github.com/ggml-org/llama.cpp/issues/16207). "
            f"Falling back to Llava15ChatHandler would run without erroring but "
            f"silently produce empty/garbage output for this model family -- which is "
            f"almost certainly what caused 'VLM did not return JSON:' with an empty "
            f"body. Fix: install a llama-cpp-python fork with Qwen-VL support (e.g. "
            f"JamePeng's llama-cpp-python-JamePeng, which adds these handler classes), "
            f"or switch to a non-Qwen-VL GGUF vision model (e.g. a LLaVA-format one)."
        )

    return Llava15ChatHandler(clip_model_path=mmproj_path)


class LlamaCppVLM:
    """Thin wrapper around llama-cpp-python's multimodal chat completion.
    Loaded lazily so importing this module doesn't require llama-cpp-python
    unless the node actually runs."""

    def __init__(self, config: VLMConfig):
        self.config = config
        self._llm = None

    def _ensure_loaded(self):
        if self._llm is not None:
            return
        from llama_cpp import Llama

        chat_handler = _pick_chat_handler(self.config.model_path, self.config.mmproj_path,
                                           self.config.chat_handler)
        self._llm = Llama(
            model_path=self.config.model_path,
            chat_handler=chat_handler,
            n_ctx=self.config.n_ctx,
            n_gpu_layers=self.config.n_gpu_layers,
            # Deliberately NOT logits_all=True: that makes llama.cpp allocate a
            # scores array of n_ctx x vocab float32 up front -- 9000 x 151936 x 4B
            # is 5.09 GiB, which simply fails to allocate. Per-token logits for the
            # whole prompt are of no use here; only the generated text is read.
            verbose=False,
        )

    @staticmethod
    def _image_to_data_uri(img: Image.Image) -> str:
        import base64
        import io
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    def analyze(self, images: List[Image.Image], brief: str, mode: str,
                media_labels: List[str]) -> dict:
        """images/media_labels are parallel lists, e.g. images[i] came from
        media_labels[i] == 'Picture 2' or 'Video 1 frame 3'."""
        self._ensure_loaded()

        user_content = [{
            "type": "text",
            "text": (
                f"Mode: {mode}\n"
                f"User's creative brief: {brief.strip() or '(none given -- infer from media)'}\n"
                f"Reference media, in order, labelled: {', '.join(media_labels) or '(none)'}\n"
                "Describe the target video as JSON per the schema."
            ),
        }]
        for img in images:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": self._image_to_data_uri(img)},
            })

        resp = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        raw = resp["choices"][0]["message"]["content"]
        return _extract_json(raw)


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
    text -- otherwise it's escaped as literal content instead."""
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


class TransformersVLM:
    """Alternative backend for people who'd rather load a HF checkpoint
    directly (e.g. Qwen2-VL via `transformers`) instead of a GGUF file.
    Slower to load, uses more VRAM, but needs no separate mmproj file and
    works with any abliterated HF-format release too."""

    def __init__(self, model_id: str, device: str = "cuda", dtype: str = "bfloat16"):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self._model = None
        self._processor = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        dtype = getattr(torch, self.dtype)
        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, torch_dtype=dtype, device_map=self.device, trust_remote_code=True,
        )

    def analyze(self, images: List[Image.Image], brief: str, mode: str,
                media_labels: List[str]) -> dict:
        self._ensure_loaded()
        import torch

        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": (
            f"Mode: {mode}\nUser's creative brief: {brief.strip() or '(none given)'}\n"
            f"Reference media, in order, labelled: {', '.join(media_labels) or '(none)'}\n"
            "Describe the target video as JSON per the schema."
        )})
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        prompt = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[prompt], images=images, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=2200, temperature=0.4, do_sample=True)
        text = self._processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return _extract_json(text)
