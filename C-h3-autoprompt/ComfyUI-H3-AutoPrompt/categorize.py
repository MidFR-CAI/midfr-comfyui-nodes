"""
categorize.py -- turns explicit, user-declared media categorization into the
same subjects / anchors / audio_roles structures the composer already knows
how to render, WITHOUT asking a VLM to guess any of it.

This replaces the single biggest source of bugs across this whole pack's
development: a small local VLM misclassifying voice-donor audio, inventing
frame anchors the brief never mentioned, or re-describing a replaced
character's own appearance. All of that was the VLM guessing at things the
user actually already knows when they load the media. So now they just say
so, directly, per item, in the loader -- and this module turns that into
exactly the same deterministic data shapes nodes.py already builds a legal
H3 prompt from.

Input shape (per item, read from an H3_REFS bundle's `items` list, each
dict extended with new `cat_*` fields written by the categorizer panel):

  Picture / Video item:
    cat_role:      "identity" | "anchor"   (anchor: picture only)
    cat_subject:   int | "1,3" | None  -- which Subject slot(s) this
                                           contributes to (a comma list when
                                           one asset defines more than one
                                           subject -- guide S2.2.1)
    cat_kind:      "person"|"animal"|"object"|"environment"|"style"|None
    cat_retention: "fully_preserved" | "partially_preserved" |
                   "attribute_transfer" | "weak_reference" | None
    cat_anchor_role: "first_frame" | "last_frame" | "keyframe" | "storyboard"
    cat_anchor_shots: [int, ...]           -- 1-based shot indices
    cat_motion_for_subject: int | None     -- this VIDEO supplies MOTION for
                                               a different subject (appearance
                                               comes from elsewhere)
    cat_structural_role: "editing"|"continuation"|"camera_reference"|None
                                            -- video only, INDEPENDENT of
                                               cat_role/cat_subject (guide
                                               S2.2.3: a video's visible
                                               content can be a Subject AND
                                               separately get a structural
                                               <Video N> citation)
    cat_structural_retention: same four-value visual enum, for the
                                               structural citation above
    cat_character_swap_subject: int | "1,2" | None -- video only, the SIMPLE path
                                               for the single most common editing
                                               case: this subject (or these subjects,
                                               for multiple simultaneous swaps)
                                               replaces this video's own performer(s),
                                               everything else about the footage kept
                                               exactly as-is. Auto-derives structural
                                               role=editing, retention=fully_preserved,
                                               and that subject's replaces_original --
                                               previously needed those three set
                                               correctly by hand across two different
                                               items. Takes priority over
                                               cat_structural_role if both are set.

  Audio / video_audio item:
    cat_audio_role: "dialogue" | "voice_timbre" | "ambience" | "music" | "off"
    cat_subject:    int | None  -- on-screen speaker (dialogue) or the
                                    subject this timbre re-voices (voice_timbre)
    cat_voiceover:  bool        -- dialogue is off-screen narration
    cat_sung:       bool        -- renders as "sings"/"sing" instead of
                                    "says"/"say" (guide treats lyrics and
                                    dialogue identically via <d>, but the
                                    verb outside <d> should still match)
    cat_speaker_description: str | None    -- optional, else a generic default
    cat_retention:  "fully_copy"|"partially_copy"|"reference"|"weak_reference"
                                -- user-selectable; audio has its OWN marker
                                   vocabulary, distinct from the visual one
                                   (guide S2.4.2)

None of this module touches models. It is pure data transformation, exactly
like grammar.py -- testable without a GPU.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


def _parse_subject_slots(raw) -> List[int]:
    """Accepts an int, a comma-separated string ('1,3'), a list, or text
    that includes non-numeric words around the number ('Subject 1',
    'subject 1', 'S1') -- confirmed bug: someone naturally typed "Subject 1"
    into the field (that's literally the label used everywhere else in the
    UI/output), and the old int()-only parser silently discarded the whole
    thing on a ValueError with no error surfaced anywhere -- a picture's
    entire subject assignment just vanished. Extracting the first digit
    run from each comma-separated piece handles "1", "Subject 1", "S1",
    "subject  12" alike, rather than requiring an exact bare integer."""
    if raw is None or raw == "":
        return []
    if isinstance(raw, int):
        return [raw]
    if isinstance(raw, (list, tuple)):
        out: List[int] = []
        for x in raw:
            out.extend(_parse_subject_slots(x))
        return out
    slots = []
    for piece in str(raw).split(","):
        m = re.search(r"\d+", piece)
        if m:
            slots.append(int(m.group(0)))
    return slots


@dataclass
class SubjectPlan:
    slot: int                       # 1-based -> becomes "Subject {slot}"
    kind: str = "person"            # person | animal | object | environment | style
    sources: List[str] = field(default_factory=list)     # ["Picture 1", "Picture 2"]
    motion_source: Optional[str] = None                  # "Video 1"
    retention: str = "attribute_transfer"
    description: str = ""           # filled in by the VLM later (short, generic)
    # [(tag, label)] -- outfit references kept SEPARATE from `sources` so the
    # subject definition can cite them explicitly ("wearing the outfit shown in
    # <Picture 3>") instead of silently folding them into the identity source
    # list, where the outfit was never actually mentioned in the prompt.
    outfit_refs: List[tuple] = field(default_factory=list)
    # Shot this subject is a scene/prop reference for, when it came from a shot's
    # Reference Images rather than from a Subject row.
    ref_for_shot: Optional[int] = None
    # Set when this Subject IS an outfit: the slot of the subject who wears it, and
    # the shot label it applies to. The guide lists "clothing, props, interfaces"
    # as valid <Subject N> content (S2.2.1), and giving the outfit its own subject
    # gives it a real attribute_transfer retention line onto the wearer instead of
    # a sub-clause that competes with the identity image's own clothing.
    outfit_for_subject: Optional[int] = None
    outfit_scope: Optional[str] = None
    # Set for a comic panel used as a shot's source artwork, so its definition
    # says it becomes live motion with no lettering rather than reading like an
    # ordinary photographed location.
    is_comic_panel: bool = False
    # True when the panel keeps its speech balloons, which changes how the shot
    # is described: animated artwork with timed lettering rather than live action.
    keep_bubbles: bool = False
    # Which person in the source video this subject takes over, in plain words
    # ("the person on the left"). With more than one person on screen the model
    # has no way to know who to swap unless it's said outright.
    replaces_person: Optional[str] = None
    # Which traits carry over from this subject's own references
    # (face / hair / body / wardrobe). Empty means all of them.
    replace_keep: dict = field(default_factory=dict)
    replaces_original: bool = False # this subject replaces whoever originally appears
                                     # in the video/scene it's composited into -- explicit
                                     # instead of guessed from the brief (the legacy
                                     # pipeline's approach), since a categorized subject
                                     # doesn't need the brief to state this at all.
    # {"position": ..., "description": ...} naming the person in the source
    # footage this subject takes the place of, so the prompt can point at a
    # specific figure instead of leaving the model to guess.
    replaces_person: Optional[dict] = None
    swap_video: Optional[str] = None   # "Video 1" -- set when replaces_original came from
                                        # the single character-swap field (see build_plan),
                                        # so the composer knows which video's choreography
                                        # this subject is mapped onto for the rich retention
                                        # wording and dense frame sampling a swap needs.


@dataclass
class AnchorPlan:
    tag: str                        # "Picture 1"
    role: str                       # first_frame | last_frame | keyframe | storyboard
    shots: List[int]
    description: str = ""           # filled in by the VLM later


@dataclass
class VideoStructurePlan:
    """A <Video N> structural citation -- editing, continuation, or a pure
    camera/cut/rhythm reference. Independent of whether that same video ALSO
    feeds a Subject (guide S2.2.3: '<Video N> identifies the asset or
    structural source and does not replace subject labels')."""
    tag: str                        # "Video 1"
    role: str                       # editing | continuation | camera_reference
    retention: str = "weak_reference"
    description: str = ""
    is_character_swap: bool = False # set by the single character-swap field -- signals the
                                     # composer to sample this video densely across its
                                     # timeline and ask the VLM to describe its ACTUAL
                                     # observed motion/choreography in chronological detail,
                                     # rather than writing brief-driven narrative content
                                     # (guide's own example for this: a shot-by-shot account
                                     # of exactly what the source footage does, since that's
                                     # what has to be preserved with only identity swapped).


@dataclass
class AudioPlan:
    tag: str                        # "Audio 1"
    role: str                       # dialogue | voice_timbre | ambience | music
    subject_slot: Optional[int] = None      # speaker (dialogue) or re-voiced subject (voice_timbre)
    voiceover: bool = False
    sung: bool = False
    speaker_description: Optional[str] = None
    retention: Optional[str] = None         # fully_copy | partially_copy | reference | weak_reference
    # "<Video N>" when this audio is that video's own synchronized track. The
    # guide names the shared source explicitly (S2.2.5) so the provenance isn't
    # ambiguous: "<Audio 2> is the synchronized audio track of <Video 1>".
    from_video_tag: Optional[str] = None
    item_index: int = -1            # index into the original items list, for tensor lookup


@dataclass
class CategorizationPlan:
    subjects: List[SubjectPlan]
    anchors: List[AnchorPlan]
    video_structures: List[VideoStructurePlan]
    audio: List[AudioPlan]
    donor_pairs: List[tuple]        # [(source_audio_tag, donor_audio_tag), ...]
    soundscape_citations: List[str]
    music_citations: List[str]
    notes: List[tuple] = field(default_factory=list)   # [(tag, note_text), ...] -- free-text
                                     # disambiguation the VLM can't get from pixels alone:
                                     # which character in a multi-person video/frame is
                                     # which subject, which original character each subject
                                     # replaces, which speaker's voice is kept vs re-timbred.


def _picture_video_tags(items: List[dict]) -> Dict[int, str]:
    """item id() -> 'Picture N' / 'Video N', in the same order/numbering the
    original pack's own computeTags() (web_medialoader.js) uses: enabled
    items only, pictures numbered first, videos numbered separately."""
    tags = {}
    p = v = 0
    for it in items:
        if it.get("enabled") is False:
            continue
        if it.get("kind") == "picture":
            p += 1
            tags[id(it)] = f"Picture {p}"
    for it in items:
        if it.get("enabled") is False:
            continue
        if it.get("kind") == "video":
            v += 1
            tags[id(it)] = f"Video {v}"
    return tags


def _audio_tags(items: List[dict]) -> Dict[int, str]:
    """item id() -> 'Audio N', matching the original pack's own numbering:
    a video's paired soundtrack gets the low numbers (in video order), then
    standalone audio (and any video audio explicitly marked standalone)."""
    tags = {}
    a = 0
    for it in items:
        if it.get("enabled") is False:
            continue
        if it.get("kind") == "video" and it.get("has_audio") and \
                (it.get("audio_mode") or "paired") == "paired":
            a += 1
            tags[id(it)] = f"Audio {a}"
    for it in items:
        if it.get("enabled") is False:
            continue
        if it.get("kind") == "audio":
            a += 1
            tags[id(it)] = f"Audio {a}"
        elif it.get("kind") == "video" and it.get("has_audio") and \
                it.get("audio_mode") == "standalone":
            a += 1
            tags[id(it)] = f"Audio {a}"
    return tags


def build_plan(items: List[dict]) -> CategorizationPlan:
    pv_tags = _picture_video_tags(items)
    au_tags = _audio_tags(items)

    subjects: Dict[int, SubjectPlan] = {}
    anchors: List[AnchorPlan] = []
    video_structures: List[VideoStructurePlan] = []
    audio: List[AudioPlan] = []
    donor_pairs: List[tuple] = []
    soundscape_citations: List[str] = []
    music_citations: List[str] = []
    notes: List[tuple] = []

    def subject(slot: int) -> SubjectPlan:
        if slot not in subjects:
            subjects[slot] = SubjectPlan(slot=slot)
        return subjects[slot]

    def add_source(slots: List[int], tag: str, kind, retention, is_motion=False, replaces=False):
        for slot in slots:
            s = subject(slot)
            if is_motion:
                s.motion_source = tag
            else:
                s.sources.append(tag)
                if kind:
                    s.kind = kind
                if retention:
                    s.retention = retention
                if replaces:
                    s.replaces_original = True

    # -- pictures & videos: identity / anchor -------------------------------
    for it in items:
        if it.get("enabled") is False or it.get("kind") not in ("picture", "video"):
            continue
        tag = pv_tags.get(id(it))
        if not tag:
            continue
        note = (it.get("cat_notes") or "").strip()
        if note:
            notes.append((tag, note))
        role = it.get("cat_role") or "identity"
        slots = _parse_subject_slots(it.get("cat_subject"))

        if role == "anchor" and it.get("kind") == "picture":
            a_role = it.get("cat_anchor_role") or "keyframe"
            a_shots = _parse_subject_slots(it.get("cat_anchor_shots") or [1])
            anchors.append(AnchorPlan(tag=tag, role=a_role, shots=a_shots or [1]))
            # An anchor can ALSO feed one or more tracked subjects if its content
            # persists beyond just being the opening/closing frame -- guide S2.2.1
            # allows one asset to define multiple subjects, so this is a list too.
            add_source(slots, tag, it.get("cat_kind"), it.get("cat_retention"),
                       replaces=bool(it.get("cat_replaces_original")))
            continue

        # Ordinary identity/environment source -- may feed multiple subjects.
        add_source(slots, tag, it.get("cat_kind"), it.get("cat_retention"),
                   replaces=bool(it.get("cat_replaces_original")))

        if it.get("kind") == "video":
            # Motion-transfer target (guide S2.2.1: appearance from X, motion from Y)
            # and structural <Video N> citation (guide S2.2.3: editing/continuation/
            # camera-rhythm reference) are BOTH independent of the identity/subject
            # assignment above -- a video can feed a Subject's appearance-adjacent
            # motion AND separately be cited as the edited/continued source video,
            # at the same time. Neither used to be possible; only one narrow
            # "structure = motion donor only" mode existed before.
            motion_slots = _parse_subject_slots(it.get("cat_motion_for_subject"))
            add_source(motion_slots, tag, None, None, is_motion=True)

            # Single simplified field for the single most common case: swap this
            # video's performer for a subject, keep everything else about the
            # footage exactly as-is. Auto-derives what previously needed three
            # separate fields set correctly together (structural role=editing,
            # structural retention=fully_preserved, that subject's
            # replaces_original=True) -- getting per-field wiring right by hand
            # was the actual UX complaint this replaces, not any wrong default.
            swap_slots = _parse_subject_slots(it.get("cat_character_swap_subject"))
            if swap_slots:
                video_structures.append(VideoStructurePlan(
                    tag=tag, role="editing", retention="fully_preserved",
                    is_character_swap=True))
                for slot in swap_slots:
                    s = subject(slot)
                    s.replaces_original = True
                    s.swap_video = tag
                    # attribute_transfer is SubjectPlan's own default already, so this
                    # only overrides an explicit cat_retention the user set on the
                    # swap subject's OWN identity picture -- otherwise leaves it alone.
                continue  # character swap fully covers this video; skip the generic
                          # cat_structural_role path below so the two can't conflict.

            struct_role = it.get("cat_structural_role")
            if struct_role in ("editing", "continuation", "camera_reference"):
                video_structures.append(VideoStructurePlan(
                    tag=tag, role=struct_role,
                    retention=it.get("cat_structural_retention") or "weak_reference"))

    # -- audio: dialogue / voice_timbre / ambience / music -------------------
    dialogue_subject_audio: Dict[int, str] = {}   # subject slot -> Audio tag (its own dialogue)
    for idx, it in enumerate(items):
        if it.get("enabled") is False:
            continue
        is_audio_bearing = it.get("kind") == "audio" or (
            it.get("kind") == "video" and it.get("has_audio")
            and (it.get("audio_mode") or "off") != "off")
        if not is_audio_bearing:
            continue
        tag = au_tags.get(id(it))
        if not tag:
            continue
        note = (it.get("cat_notes") or "").strip()
        if note:
            notes.append((tag, note))
        a_role = it.get("cat_audio_role") or "off"
        if a_role == "off":
            continue

        # Audio's own retention vocabulary (guide S2.4.2) is user-selectable, not
        # hardcoded -- it also decides the task-type prefix (fully_copy/
        # partially_copy -> "audio reuse"; reference/weak_reference -> "audio
        # reference"), so ambience/music now distinguish reused-verbatim from
        # merely-referenced instead of always assuming reuse.
        # Deliberately cat_audio_retention, NOT cat_retention -- a video item's
        # cat_retention already means "how much of THIS VIDEO's visual identity
        # is preserved" (visual vocabulary: fully_preserved/etc). Its paired
        # audio needs its OWN marker from the audio vocabulary (fully_copy/etc),
        # which is a different question with a different answer -- reusing
        # cat_retention caused a visual marker to get attached to an audio
        # explanation, exactly the same class of bug as the cat_subject collision.
        retention = it.get("cat_audio_retention")
        if a_role == "ambience":
            soundscape_citations.append((tag, retention or "partially_copy"))
            continue
        if a_role == "music":
            music_citations.append((tag, retention or "fully_copy"))
            continue

        # Deliberately a SEPARATE field from cat_subject: a video item's own
        # cat_subject means "this video's visual content belongs to Subject N";
        # its paired audio's speaker is an independent fact that can disagree
        # (e.g. a video categorized with no subject at all can still have paired
        # dialogue attributed to a subject defined elsewhere). Reusing cat_subject
        # for both caused the video to wrongly get listed as an appearance source.
        # Parsed the same lenient way as cat_subject -- a raw int() here would
        # crash the whole compose() call on anything but a clean integer/numeric
        # string, rather than the confirmed silent-drop failure that hit
        # cat_subject; both are worth avoiding the same way.
        audio_slots = _parse_subject_slots(it.get("cat_audio_subject"))
        slot = audio_slots[0] if audio_slots else None
        plan = AudioPlan(
            tag=tag, role=a_role,
            subject_slot=slot,
            voiceover=bool(it.get("cat_voiceover")),
            sung=bool(it.get("cat_sung")),
            speaker_description=it.get("cat_speaker_description") or None,
            retention=retention or ("partially_copy" if a_role == "dialogue" else "reference"),
            item_index=idx,
        )
        audio.append(plan)
        if a_role == "dialogue" and slot:
            dialogue_subject_audio[slot] = tag

    # Pair voice-timbre donors with whichever audio item carries that
    # subject's actual dialogue -- explicit now (cat_subject on the donor
    # IS the subject being re-voiced), no brief-text guessing needed at all.
    for plan in audio:
        if plan.role == "voice_timbre" and plan.subject_slot:
            source_tag = dialogue_subject_audio.get(plan.subject_slot)
            if source_tag:
                donor_pairs.append((source_tag, plan.tag))

    return CategorizationPlan(
        subjects=[subjects[k] for k in sorted(subjects)],
        anchors=anchors, video_structures=video_structures, audio=audio,
        donor_pairs=donor_pairs,
        soundscape_citations=soundscape_citations, music_citations=music_citations,
        notes=notes,
    )
