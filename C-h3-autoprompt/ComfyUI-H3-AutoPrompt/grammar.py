"""
grammar.py -- Deterministic MiniMax H3 prompt compiler.

This module contains ZERO model calls. It only assembles structured data
(produced by a VLM and by Whisper elsewhere in this pack) into prompt text
that is byte-for-byte compliant with MiniMax's own "Video Prompt Writing
Guide" -- the PDF that ships inside Adudeguyman/ComfyUI-Fantastic-MiniMaxH3-
PromptBuilder (web/Video_Prompt_Writing_Guide.pdf) and that this pack's
prompts are checked against.

Rigid formatting should never be delegated to a generative model -- models
are good at deciding WHAT to say, not at guaranteeing punctuation, ordering,
and monotonically increasing timestamps. So the pipeline is:

    VLM        -> content   (what's in the shot, what the camera should do)
    Whisper    -> dialogue  (who says what, when, in what language)
    grammar.py -> FORMAT    (turns the above into a legal H3 prompt, always)

Everything in here is pure and testable without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict, Tuple
import re

Mode = Literal["T2VA", "I2VA", "FL2VA", "L2VA", "REF"]

# --------------------------------------------------------------------------
# Time / camera formatting
# --------------------------------------------------------------------------

def fmt_mmss(seconds: float) -> str:
    """'00:03.500' -- MM:SS.mmm, used for in-body [Shot N] cut times."""
    seconds = max(0.0, float(seconds))
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m:02d}:{s:06.3f}"


def fmt_2dp(seconds: float) -> str:
    """'6.00' -- plain seconds to two decimal places, used in anchor lines
    (I2VA / FL2VA / L2VA instruction sentences)."""
    return f"{max(0.0, float(seconds)):.2f}"


CAMERA_VERBS = {
    "zoom_in": "zooms in", "zoom_out": "zooms out",
    "push_in": "pushes in", "pull_out": "pulls out",
    "pan_left": "pans left", "pan_right": "pans right",
    "truck_left": "trucks left", "truck_right": "trucks right",
    "tilt_up": "tilts up", "tilt_down": "tilts down",
    "pedestal_up": "pedestals up", "pedestal_down": "pedestals down",
    "arc": "arcs around the subject",
    "tracking": "tracks the moving subject",
    "static": "holds a static shot",
    "shake_slight": "shakes slightly",
    "shake_strong": "shakes strongly",
    "pov": "adopts the subject's point of view",
    "roll_cw": "rolls clockwise",
    "roll_ccw": "rolls counterclockwise",
}

CUT_VERBS = [
    "the camera cuts to", "the shot cuts to", "the shot transitions to",
    "the shot changes to", "the shot switches to",
]
EXPLICIT_CUT_VERBS = {"dissolve": "cross-dissolves to", "fade": "fades to", "wipe": "wipes to"}


def camera_phrase(motion: Optional[str], amplitude: Optional[str] = None,
                   speed: Optional[str] = None) -> str:
    """Camera motion written as natural action, per guide S1.4.3.
    Medium amplitude / normal speed are omitted (guide says to add them
    'only when meaningful')."""
    if not motion or motion == "none":
        return ""
    verb = CAMERA_VERBS.get(motion, motion.replace("_", " "))
    bits = [f"The camera {verb}"]
    if amplitude in ("small", "large"):
        bits.append(f"with {amplitude} amplitude")
    if speed in ("slow", "fast"):
        bits.append(f"at {speed} speed")
    return " ".join(bits)


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class DialogueEvent:
    speaker_label: str          # e.g. "the middle-aged baker with a calm, slightly raspy voice"
    sx: str                     # "S1", "S2", or compound "S1,S2"
    lang: str                   # "English"
    text: str                   # verbatim, untranslated
    voiceover: bool = False
    first_mention: bool = True  # guide only requires the full descriptive identity phrase
                                 # the first time a speaker appears; later lines by the same
                                 # (Sx) just need the stable ID (guide S1.4.4)
    scenetrans_out: bool = False   # this event's audio continues into the NEXT shot
    scenetrans_in: bool = False    # this event's audio continued FROM the previous shot
    cutoff: bool = False           # truncated by end of video
    subject_ref: Optional[str] = None   # "<Subject 3>" in REF mode, else None
    sung: bool = False             # guide treats lyrics/dialogue identically via <d>,
                                    # but the verb outside <d> should read "sings" not "says"
    position: str = "after"        # "before" | "after" -- whether this line happens before
                                    # or after the shot's action content, based on its actual
                                    # timing within the shot. Dialogue used to always render
                                    # after the full content text regardless of when it really
                                    # happens, which could put a greeting after the walk-away
                                    # it was supposed to precede.


@dataclass
class Shot:
    index: int                                # 1-based
    content: str                              # action/description prose, NOT including camera/dialogue
    cut_time_s: Optional[float] = None        # None only for shot 1
    cut_verb: str = "the camera cuts to"
    camera_motion: Optional[str] = None
    camera_amplitude: Optional[str] = None
    camera_speed: Optional[str] = None
    onscreen_text: Optional[str] = None       # verbatim text in quotes, if any
    dialogue: List[DialogueEvent] = field(default_factory=list)
    picture_anchor: Optional[str] = None      # e.g. "<Picture 1>" when this shot begins/ends on a picture
    style: Optional[str] = None    # only meaningful on shot 1 (base modes) / before shot 1 (REF)
    establish: str = ""            # rendered before dialogue; see render_shot_body
    subject_mentions: List[str] = field(default_factory=list)  # "<Subject 2>" first-appearance text, REF mode


@dataclass
class SubjectDef:
    tag: str            # "Subject 1"
    text: str            # full defining sentence, e.g. "is the young woman in <Picture 1>, ..."


@dataclass
class RetentionLine:
    tag: str                     # "Subject 1" / "Picture 2" / "Video 1" / "Audio 1"
    where: str                   # "(appears in [Shot 1], [Shot 3])" or "([Shot 1] first frame)" or ""
    marker: str                  # fully_preserved / partially_preserved / attribute_transfer / weak_reference
                                  # or for audio: fully_copy / partially_copy / reference / weak_reference
    explanation: str


# --------------------------------------------------------------------------
# Shared shot-body renderer (used by both base and full-reference modes)
# --------------------------------------------------------------------------

def _normalize_dialogue_text(text: str) -> str:
    """Guide S2.5.4: 'Standardize punctuation to the basic written marks
    needed to express the sentence... remove repeated tildes, emoji,
    bullets, and repeated or decorative punctuation. End complete
    statements, questions, and exclamations with ., ?, or ! respectively
    before </d>.' This is a distinct rule from 'preserve every original
    word... verbatim' just above it in the guide -- that governs the WORDS,
    this one governs punctuation formatting, and both apply simultaneously.
    Whisper (especially after sentence-splitting on word timestamps, or
    when a segment got cut by VAD/a hard clip trim rather than a natural
    pause) frequently omits trailing punctuation; this adds it
    deterministically rather than leaving an unpunctuated <d> block, which
    the guide explicitly disallows. Defaults to '.' since a fragment's
    intended sentence type (statement/question/exclamation) usually can't
    be inferred reliably from the words alone."""
    text = text.strip()
    if not text:
        return text
    text = re.sub(r"~{2,}", "", text)
    text = re.sub(r"([.!?]){2,}", r"\1", text)
    if text[-1] not in ".!?":
        text += "."
    return text


def _render_dialogue(ev: DialogueEvent) -> str:
    d = f"<d>[{ev.lang}] {_normalize_dialogue_text(ev.text)}</d>"
    # Guide S2.5.4: "<Subject N> identifies the referenced subject, while (Sx) identifies
    # the actual speaker. When the subject speaks, write <Subject N> (Sx)." -- this applies
    # on every mention, not just the first, and takes priority over a plain description.
    if ev.subject_ref:
        who_full = f"<{ev.subject_ref}> ({ev.sx})"
        who_terse = who_full
    else:
        who_full = f"{ev.speaker_label} ({ev.sx})"
        who_terse = f"({ev.sx})"
    who = who_full if (ev.first_mention or ev.subject_ref) else who_terse
    if ev.voiceover:
        verb_phrase = "sings in an off-screen voiceover" if ev.sung else "says in an off-screen voiceover"
        lead = f"{who} {verb_phrase}: {d} while their lips remain completely closed."
    else:
        verb = ("sing" if ev.sung else "say") if "," in ev.sx else ("sings" if ev.sung else "says")
        lead = f"{who} {verb}, {d}"
    if ev.scenetrans_in:
        lead = f"<scenetrans> {lead}, continuing seamlessly across the cut."
    if ev.scenetrans_out:
        lead = f"{lead} <scenetrans> The line continues uninterrupted into the next shot."
    if ev.cutoff:
        lead = f"{lead} <cutoff>"
    return lead


_SPEECH_CUE_RE = re.compile(
    r"\b(says?|said|speaks?|spoke|talks?|talking|replies|replied|answers?|answered|"
    r"asks?|asked|shouts?|shouted|calls?\s+out|called\s+out|yells?|yelled|"
    r"whispers?|whispered|mutters?|muttered|exclaims?|exclaimed|adds?|added|"
    r"tells?|told|begins?\s+to\s+speak|responds?|responded)\b", re.IGNORECASE)


def _split_at_speech(content: str):
    """Splits a shot's action text around the sentence that mentions speaking.

    Returns (head, tail) where the dialogue belongs between them, or ("", "")
    when nothing in the text refers to speech -- in which case the caller keeps
    its existing before/after placement.
    """
    if not content:
        return "", ""
    # Sentence boundaries, keeping the terminator attached.
    parts = re.findall(r"[^.!?]+[.!?]?", content)
    parts = [p for p in (x.strip() for x in parts) if p]
    for i, part in enumerate(parts):
        if _SPEECH_CUE_RE.search(part):
            head = " ".join(parts[:i + 1]).strip()
            tail = " ".join(parts[i + 1:]).strip()
            return head, tail
    return "", ""


def render_shot_body(shot: Shot) -> str:
    parts: List[str] = []

    header = f"[Shot {shot.index}]"
    if shot.index == 1:
        pass  # no timestamp on shot 1, ever
    else:
        cv = shot.cut_verb if shot.cut_verb in CUT_VERBS or shot.cut_verb in EXPLICIT_CUT_VERBS.values() else "the camera cuts to"
        header = f"[Shot {shot.index}] At {fmt_mmss(shot.cut_time_s)}, {cv}"

    body_bits = []
    if shot.style:
        body_bits.append(shot.style)
    # Establishing facts (what a subject is wearing, where the shot takes place)
    # render before ANY dialogue. detailed_description is read as a timeline, so
    # these have to be true from the first frame -- placed after a line of speech
    # they read as a change happening partway through the shot.
    if getattr(shot, "establish", ""):
        body_bits.append(shot.establish.strip())

    before_dialogue = [ev for ev in shot.dialogue if ev.position == "before"]
    after_dialogue = [ev for ev in shot.dialogue if ev.position != "before"]

    # If the action text itself says someone speaks ("she turns and says..."),
    # put the lines at THAT moment rather than bolting them onto one end. The
    # description reads as a timeline, so dialogue parked at the front happens
    # before the action that was meant to lead into it.
    head, tail = _split_at_speech(shot.content.strip()) if shot.dialogue else ("", "")
    if head:
        body_bits.append(head)
        for ev in shot.dialogue:
            body_bits.append(_render_dialogue(ev))
        if tail:
            body_bits.append(tail)
        before_dialogue = after_dialogue = []
    else:
        for ev in before_dialogue:
            body_bits.append(_render_dialogue(ev))
        body_bits.append(shot.content.strip())

    cam = camera_phrase(shot.camera_motion, shot.camera_amplitude, shot.camera_speed)
    if cam:
        body_bits.append(cam.rstrip() + ("" if cam.endswith((".", ",")) else "."))

    for ev in after_dialogue:
        body_bits.append(_render_dialogue(ev))

    if shot.onscreen_text:
        body_bits.append(f'On-screen text reads "{shot.onscreen_text}".')

    body = " ".join(b for b in body_bits if b).strip()
    return f"{header} {body}".strip()


# --------------------------------------------------------------------------
# Base modes: T2VA / I2VA / FL2VA / L2VA
# --------------------------------------------------------------------------

def compose_instruction(mode: Mode, last_shot_index: int, duration_s: float,
                         second_picture_time_s: Optional[float] = None) -> str:
    if mode == "T2VA":
        return ""
    if mode == "I2VA":
        return ("For the target video, at 0.00 seconds into the target video, "
                "<Picture 1> (from [Shot 1]) is fully referenced.")
    if mode == "FL2VA":
        t2 = fmt_2dp(second_picture_time_s if second_picture_time_s is not None else duration_s)
        return ("How the reference pictures align with the target video "
                f"\u2014 Picture 1 (from Shot 1) aligns with the 0.00-second mark of the "
                f"target video; Picture 2 (from Shot {last_shot_index}) aligns with the "
                f"{t2}-second mark of the target video.")
    if mode == "L2VA":
        return ("How the reference pictures align with the target video "
                f"\u2014 <Picture 1> (from [Shot {last_shot_index}]) aligns with the "
                f"{fmt_2dp(duration_s)}-second mark of the target video.")
    raise ValueError(mode)


def compose_base(mode: Mode, shots: List[Shot], overall_soundscape: str,
                  non_diegetic_music: str, duration_s: float) -> str:
    """Builds the full T2VA/I2VA/FL2VA/L2VA prompt string."""
    if not shots:
        raise ValueError("At least one shot is required.")
    last_idx = shots[-1].index
    instruction = compose_instruction(mode, last_idx, duration_s)

    body = " ".join(render_shot_body(s) for s in shots)

    lines = []
    if instruction:
        lines.append(instruction)
        lines.append("")  # blank line required between instruction and core fields
    lines.append(f"integrated_multimodal_description: {body}")
    lines.append(f"overall_soundscape: {overall_soundscape.strip() or 'N/A'}")
    lines.append(f"non_diegetic_music: {non_diegetic_music.strip() or 'N/A'}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Full-reference mode
# --------------------------------------------------------------------------

def compose_subject_definitions(subjects: List[SubjectDef]) -> str:
    return "\n".join(f"<{s.tag}> {s.text}" for s in subjects)


def compose_summary(task_types: List[str], text: str) -> str:
    prefix = "[" + " + ".join(dict.fromkeys(task_types)) + "]"  # de-dupe, keep order
    return f"{prefix} {text.strip()}"


def compose_retention_analysis(lines: List[RetentionLine]) -> str:
    out = []
    for r in lines:
        where = f" {r.where}" if r.where else ""
        out.append(f"<{r.tag}>{where}: {r.marker} - {r.explanation.strip()}")
    return "\n".join(out)


def compose_detailed_description(style_intro: str, shots: List[Shot]) -> str:
    body = " ".join(render_shot_body(s) for s in shots)
    return f"{style_intro.strip()} {body}".strip()


def compose_reference(subjects: List[SubjectDef], task_types: List[str], summary_text: str,
                       retention: List[RetentionLine], style_intro: str, shots: List[Shot],
                       overall_soundscape: str, non_diegetic_music: str) -> str:
    """Builds the full six-section reference-mode prompt string, in order:
    subject_definitions, summary, retention_analysis, detailed_description,
    overall_soundscape, non_diegetic_music."""
    sections = [
        ("subject_definitions", compose_subject_definitions(subjects)),
        ("summary", compose_summary(task_types, summary_text)),
        ("retention_analysis", compose_retention_analysis(retention)),
        ("detailed_description", compose_detailed_description(style_intro, shots)),
        ("overall_soundscape", overall_soundscape.strip() or "N/A"),
        ("non_diegetic_music", non_diegetic_music.strip() or "N/A"),
    ]
    return "\n".join(f"{name}:\n{content}" for name, content in sections)


# --------------------------------------------------------------------------
# Self-check against the guide's own "Output Checklist" (Section 3)
# --------------------------------------------------------------------------

_D_TAG_RE = re.compile(r"<d>\[[^\]]+\]\s.*?</d>", re.DOTALL)
_SHOT_HEADER_RE = re.compile(r"\[Shot (\d+)\](?:\sAt\s(\d\d):(\d\d\.\d{3}),)?")


def validate(prompt_text: str, mode: Mode, duration_s: float) -> List[str]:
    """Returns a list of human-readable violations. Empty list == passes the
    guide's own checklist to the extent it's mechanically checkable."""
    problems: List[str] = []

    if mode == "REF":
        required = ["subject_definitions:", "summary:", "retention_analysis:",
                    "detailed_description:", "overall_soundscape:", "non_diegetic_music:"]
        positions = [prompt_text.find(r) for r in required]
        if any(p == -1 for p in positions):
            missing = [r for r, p in zip(required, positions) if p == -1]
            problems.append(f"missing section(s): {missing}")
        elif positions != sorted(positions):
            problems.append("the six sections are present but out of order")
        if re.search(r"retention_analysis:.*?(?=detailed_description:)",
                      prompt_text, re.DOTALL):
            ra = re.search(r"retention_analysis:(.*?)detailed_description:", prompt_text, re.DOTALL)
            if ra and re.search(r"\(S\d", ra.group(1)):
                problems.append("(Sx) speaker ID found inside retention_analysis -- forbidden")
        # Guide S2.5.3: detailed_description should cite <Subject N>/<Picture N>/<Video N>
        # tags inline at their appearances -- a REF prompt that defines subjects but never
        # cites any of them inline has likely fallen back to untagged plain prose (a
        # regression we've hit before), which loses the guide's actual defining feature of
        # full-reference mode.
        dd = re.search(r"detailed_description:(.*?)(?=overall_soundscape:)", prompt_text, re.DOTALL)
        sd = re.search(r"subject_definitions:(.*?)(?=summary:)", prompt_text, re.DOTALL)
        if dd and sd and re.search(r"<(Subject|Picture|Video) \d", sd.group(1)):
            if not re.search(r"<(Subject|Picture|Video) \d", dd.group(1)):
                problems.append("subject_definitions defines reference labels but "
                                 "detailed_description never cites any of them inline")
    else:
        for f in ("integrated_multimodal_description:", "overall_soundscape:", "non_diegetic_music:"):
            if f not in prompt_text:
                problems.append(f"missing field: {f}")

    if mode == "REF":
        body_match = re.search(r"detailed_description:(.*?)(?=overall_soundscape:)", prompt_text, re.DOTALL)
    else:
        body_match = re.search(r"integrated_multimodal_description:(.*?)(?=overall_soundscape:)",
                                prompt_text, re.DOTALL)
    body_text = body_match.group(1) if body_match else prompt_text

    shots = _SHOT_HEADER_RE.findall(body_text)
    if not shots:
        problems.append("no [Shot N] markers found")
    else:
        indices = [int(i) for i, _, _ in shots]
        if indices != list(range(1, len(indices) + 1)):
            problems.append(f"shot numbers are not sequential from 1: {indices}")
        if shots[0][1] or shots[0][2]:
            problems.append("[Shot 1] carries a timestamp -- it must not")
        last_t = -1.0
        for i, mm, ss in shots[1:]:
            if not mm:
                problems.append(f"[Shot {i}] is missing its cut time")
                continue
            t = int(mm) * 60 + float(ss)
            if t <= last_t:
                problems.append(f"[Shot {i}] cut time {mm}:{ss} does not strictly increase")
            if t > duration_s:
                problems.append(f"[Shot {i}] cut time {mm}:{ss} exceeds the video duration ({duration_s}s)")
            last_t = t

    for m in re.finditer(r"<d>", prompt_text):
        chunk = prompt_text[m.start():m.start() + 4000]
        if "</d>" not in chunk:
            problems.append("unclosed <d> tag")
            break
    if prompt_text.count("<d>") != prompt_text.count("</d>"):
        problems.append("<d> / </d> tag count mismatch")

    for m in re.finditer(r"<d>\[([^\]]*)\]", prompt_text):
        if not m.group(1).strip():
            problems.append("a <d> block has an empty language tag")

    # Guide S1.4.6/S1.4.7 (and S2.6 for REF mode) call for "one to four sentences" /
    # "one to three sentences" -- a noun-phrase fragment with no verb or terminal
    # punctuation ("ambient room tone with subtle footsteps") technically satisfies every
    # other check here but isn't actually a sentence. Cheap proxy: field is "N/A" or ends
    # with terminal punctuation.
    for field in ("overall_soundscape", "non_diegetic_music"):
        m = re.search(rf"{field}:\s*\n?(.*?)(?=\n\w+:|\Z)", prompt_text, re.DOTALL)
        if m:
            text = m.group(1).strip()
            if text and text != "N/A" and not text.endswith((".", "!", "?")):
                problems.append(f"{field} reads as a sentence fragment, not a complete "
                                 f"sentence: {text!r}")

    # Guide's own checklist: "overall_soundscape is one to four sentences with no dialogue,
    # singing, or diegetic music" -- an explicit rule nothing was actually checking. If any
    # actual spoken line (from a <d> block) shows up verbatim in overall_soundscape, that's
    # dialogue leaking into a section the guide says never contains it.
    soundscape_m = re.search(r"overall_soundscape:\s*\n?(.*?)(?=\n\w+:|\Z)", prompt_text, re.DOTALL)
    if soundscape_m:
        soundscape_text = soundscape_m.group(1)
        for dm in re.finditer(r"<d>\[[^\]]*\]\s*(.*?)</d>", prompt_text, re.DOTALL):
            line = dm.group(1).strip()
            if line and len(line) > 3 and line in soundscape_text:
                problems.append(f"overall_soundscape appears to repeat a spoken line "
                                 f"verbatim ({line!r}) -- dialogue belongs only in "
                                 f"detailed_description's <d> tags, per the guide's own rule")
                break

    return problems
