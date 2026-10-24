"""
replace_prompt.py -- the prompt for a character replacement.

Replacement is a rigid, narrow job, and the general composer over-serves it:
it narrates action that doesn't exist, transcribes dialogue that isn't
changing, and cites the source clip as a structural <Video N> edit. Community
results that actually land treat the source clip as a SUBJECT -- "the
environment and motion layout in video1" -- and keep the whole thing terse.

This module renders that shape directly instead of routing through the
general pipeline.
"""

from __future__ import annotations

from typing import List, Optional

# Traits that can carry over from a replacing subject's own references. Order
# matters: it's the order they're listed in the prompt.
KEEP_TRAITS = ["face", "hair", "body", "wardrobe"]
_TRAIT_WORDS = {
    "face": "facial features",
    "hair": "hair",
    "body": "build",
    "wardrobe": "wardrobe",
}


def _kept(keep: dict) -> List[str]:
    """Which traits transfer. An empty/absent mapping means all of them, so a
    person who never touched the checkboxes gets the full swap."""
    if not keep:
        return list(KEEP_TRAITS)
    chosen = [t for t in KEEP_TRAITS if keep.get(t)]
    return chosen or list(KEEP_TRAITS)


def _trait_phrase(keep: dict) -> str:
    words = [_TRAIT_WORDS[t] for t in _kept(keep)]
    if len(words) == 1:
        return words[0]
    return ", ".join(words[:-1]) + " and " + words[-1]


def render(plan, source_tag: str, source_desc: str, duration: float,
            style: str = "cinematic, ultra-realistic",
            audio_tag: Optional[str] = None,
            audio_mode: str = "keep",
            people: Optional[List[dict]] = None,
            shots: Optional[List[str]] = None) -> str:
    """Builds the finished prompt.

    Every person marked in the source frame becomes a <Subject N> of their own,
    whether they're being replaced or not. That's what lets an unreplaced person
    survive: they get a fully_preserved line instead of being swept up by a
    blanket "nothing from the original remains". The clip is ALSO cited as
    <Video N>, the edit source, since the two roles are different (guide S2.2.3).

    `people` entries: {"slot", "where", "replaced_by", "voice_tag", "keep"}.
    """
    people = people or []
    swaps = {p["replaced_by"]: p for p in people if p.get("replaced_by")}
    by_slot = {s.slot: s for s in plan.subjects}

    defs, ret = [], []

    # 1. The replacing subjects, from their own reference pictures.
    for slot, person in sorted(swaps.items()):
        sp = by_slot.get(slot)
        if not sp:
            continue
        srcs = ", ".join(f"<{t}>" for t in sp.sources) or "the reference image"
        desc = (sp.description or "").strip().rstrip(".")
        defs.append(f"<Subject {slot}> is {desc or 'the replacement subject'}, whose "
                    f"appearance comes from {srcs}.")
        ret.append(f"<Subject {slot}> (appears in [Shot 1]): attribute_transfer - only "
                   f"the subject's defining visual traits, not the pose or setting "
                   f"shown, are carried into the target video.")

    # 2. Each person already in the clip, replaced or not.
    for person in people:
        pslot, where = person["slot"], person.get("where") or "a person in the shot"
        if person.get("replaced_by"):
            defs.append(f"<Subject {pslot}> is {where} in <{source_tag}>.")
            ret.append(f"<Subject {pslot}> (appears in [Shot 1]): partially_preserved - "
                       f"their position, motion and timing are kept exactly; their "
                       f"visual identity is replaced.")
        else:
            defs.append(f"<Subject {pslot}> is {where} in <{source_tag}>, who is not "
                        f"replaced.")
            ret.append(f"<Subject {pslot}> (appears in [Shot 1]): fully_preserved - "
                       f"reproduced exactly as in the source; appearance, motion and "
                       f"voice are unchanged.")

    defs.append(f"<{source_tag}> is the source video for the target video edit"
                + (f": {source_desc.strip().rstrip('.')}." if source_desc else "."))
    for person in people:
        if person.get("voice_tag"):
            defs.append(f"<{person['voice_tag']}> is the voice timbre reference for "
                        f"<Subject {person['replaced_by'] or person['slot']}>.")

    ret.append(f"<{source_tag}> (edit source): partially_preserved - the camera, "
               f"framing, cuts, lighting and environment are unchanged; only the "
               f"replaced identities differ.")
    for person in people:
        if person.get("voice_tag"):
            ret.append(f"<{person['voice_tag']}>: reference - only timbre, rhythm and "
                       f"delivery are referenced, without copying the original signal.")

    # -- summary --------------------------------------------------------
    tasks = ["reference generation", "video editing"]
    if any(p.get("voice_tag") for p in people):
        tasks.append("audio reference")
    bits = []
    for slot, person in sorted(swaps.items()):
        bits.append(f"Subject {slot} replaces Subject {person['slot']}")
    for person in people:
        if not person.get("replaced_by"):
            bits.append(f"Subject {person['slot']} is preserved unchanged")
    summary = (f"[{' + '.join(tasks)}] The target video is an edited version of "
               f"<{source_tag}>: " + "; ".join(bits) + ".")

    # -- body -----------------------------------------------------------
    # The guide is explicit: detailed_description must establish composition,
    # appearance and position, environment and lighting, actions and state
    # changes, and camera movement -- and "avoid reducing the description to a
    # plot summary or a list of reference relationships". The earlier version
    # here WAS a list of reference relationships, which left the model with no
    # described scene to render and is very likely why it fell back to showing
    # reference stills. The VLM's description of the source footage now carries
    # the body, with the replacements woven into it.
    body = [f"The target video is an edited version of <{source_tag}> in a "
            f"{style.lower()} style, identical to the source in every respect "
            f"except the identities noted below."]
    for i, shot_text in enumerate(shots or [], start=1):
        head = "[Shot 1]" if i == 1 else f"[Shot {i}]"
        body.append(f"{head} {shot_text.strip()}")
    if not shots:
        body.append(f"[Shot 1] The shot plays exactly as in <{source_tag}>.")

    # Replacement statements come after the scene is described, so they modify a
    # scene that exists rather than standing in for one.
    for slot, person in sorted(swaps.items()):
        sp = by_slot.get(slot)
        traits = _trait_phrase(person.get("keep") or getattr(sp, "replace_keep", {}))
        body.append(f"Throughout, <Subject {person['slot']}> is replaced by "
                    f"<Subject {slot}>, which supplies only {traits}. "
                    f"<Subject {slot}> holds <Subject {person['slot']}>'s exact "
                    f"position, pose, gesture and expression in every frame; the "
                    f"performance, timing and blocking are unchanged.")
    kept = [p for p in people if not p.get("replaced_by")]
    if kept:
        names = " and ".join(f"<Subject {p['slot']}>" for p in kept)
        body.append(f"{names} {'are' if len(kept) > 1 else 'is'} unchanged from "
                    f"<{source_tag}> in appearance, motion and voice.")
    body.append(f"The camera, framing, cuts, lighting, shadows and environment of "
                f"<{source_tag}> are preserved exactly, and the clip runs as one "
                f"continuous take from its first frame to its last with no still "
                f"image, intro, outro or added cut.")

    # -- sound ----------------------------------------------------------
    voiced = [p for p in people if p.get("voice_tag")]
    if voiced:
        parts = [f"every line spoken by <Subject {p['replaced_by'] or p['slot']}> is "
                 f"re-dubbed in the voice timbre of <{p['voice_tag']}>" for p in voiced]
        sound = (f"The wording, timing and lip sync of <{source_tag}> are kept "
                 f"exactly; " + "; ".join(parts) + ".")
        if len(voiced) < len(people):
            sound += " All other speech is retained unchanged."
    elif audio_mode == "keep":
        sound = (f"The original audio of <{source_tag}> is retained unchanged, "
                 f"including all spoken dialogue, in sync with the performance.")
    else:
        sound = f"Ambience and incidental sound follow <{source_tag}>."

    music = (f"Any music present in <{source_tag}> is matched exactly."
             if audio_mode != "off" else "N/A")

    return (
        "subject_definitions:\n" + "\n".join(defs)
        + "\nsummary:\n" + summary
        + "\nretention_analysis:\n" + "\n".join(ret)
        + "\ndetailed_description:\n" + "\n".join(body)
        + "\noverall_soundscape:\n" + sound
        + "\nnon_diegetic_music:\n" + music
    )
