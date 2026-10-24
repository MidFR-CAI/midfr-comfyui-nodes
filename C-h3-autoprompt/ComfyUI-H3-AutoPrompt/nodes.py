"""
nodes.py -- ComfyUI node definitions for H3 AutoPrompt.

Five nodes, all under conditioning/video_models/h3_autoprompt:

  H3AutoPromptMediaLoader    - loads pictures/videos/audio, same slot/tag
                               numbering convention as the native H3 nodes
                               expect, and the same output contract as
                               Adudeguyman's MiniMax H3 Media Loader
                               (picture_1..9 / video_1..3 / video_audio_1..3
                               / audio_1..3) so it drops into existing H3
                               workflows unchanged.
  H3AutoPromptVLMAnalyzer    - local VLM -> structured scene JSON
  H3AutoPromptWhisperNode    - local Whisper -> structured dialogue JSON
  H3AutoPromptComposer       - grammar.py -> final, format-checked prompt
  H3AutoPromptAllInOne       - the three above, wired together, for people
                               who just want one node: brief + media in,
                               finished prompt out.

This pack does not require Adudeguyman/ComfyUI-Fantastic-MiniMaxH3-
PromptBuilder to be installed -- it is self-sufficient -- but its output
`prompt` STRING and media pass-throughs are wire-compatible with that
pack's Prompt Builder / native MiniMaxH3ReferenceToVideo nodes if you do
have it installed and want to open the result in its editor for a final
manual pass.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional, Dict

try:                       # available inside ComfyUI, absent in bare tests
    import folder_paths
except Exception:          # pragma: no cover
    folder_paths = None

from . import media_io
from . import grammar as g
from . import composer as c
from . import categorize as cat
from .vlm_engine import LlamaCppVLM, TransformersVLM, VLMConfig
from .whisper_engine import FasterWhisperTranscriber, transcribe_clip


def _gguf_dir():
    """models/gguf -- where llama.cpp-style VLM weights and their mmproj files
    live. Registered with ComfyUI's own folder system so get_full_path works,
    and created if missing so the dropdown has somewhere to look."""
    if folder_paths is None:
        return None
    try:
        base = os.path.join(folder_paths.models_dir, "gguf")
    except Exception:
        return None
    try:
        os.makedirs(base, exist_ok=True)
    except Exception:
        pass
    try:
        adder = getattr(folder_paths, "add_model_folder_path", None)
        if callable(adder):
            adder("gguf", base)
    except Exception:
        pass
    return base


def _gguf_choices():
    """Filenames under models/gguf, for the model/mmproj dropdowns. Evaluated at
    node-registration time (like every other ComfyUI model list), so a newly
    added file needs a ComfyUI restart or a node refresh to appear. Returns an
    empty list when the folder is missing or empty -- INPUT_TYPES then falls
    back to a plain text field so a path can still be typed."""
    base = _gguf_dir()
    if not base or not os.path.isdir(base):
        return []
    out = []
    for root, _dirs, files in os.walk(base):
        for fn in files:
            if fn.lower().endswith(".gguf"):
                rel = os.path.relpath(os.path.join(root, fn), base)
                out.append(rel.replace(os.sep, "/"))
    return sorted(out)


def _resolve_gguf(name: str) -> str:
    """Turns a dropdown selection back into a real path. Absolute/existing paths
    are passed through untouched so a typed path still works."""
    if not name:
        return name
    if os.path.isabs(name) and os.path.exists(name):
        return name
    if folder_paths is not None:
        try:
            full = folder_paths.get_full_path("gguf", name)
            if full and os.path.exists(full):
                return full
        except Exception:
            pass
    base = _gguf_dir()
    if base:
        cand = os.path.join(base, name)
        if os.path.exists(cand):
            return cand
    return name


_GGUF_CHOICES = _gguf_choices()

PICTURES, VIDEOS, VIDEO_AUDIOS, AUDIOS = 9, 3, 3, 3
CATEGORY = "conditioning/video_models/h3_autoprompt"


def _find_reference_splitter_class():
    """Locates Adudeguyman's 'MiniMax H3 Reference Splitter' node class
    through ComfyUI's own global node registry (populated at startup from
    every installed custom_nodes package), rather than assuming a fixed
    import path. Matched by key substring since the exact registration key
    isn't part of that pack's public docs -- 'referencesplitter' alone, or
    'minimaxh3' + 'splitter' together, covers the naming pattern their
    other nodes use (MiniMaxH3MediaLoader, MiniMaxH3PromptBuilder, ...)."""
    try:
        import nodes as comfy_nodes  # ComfyUI's own top-level nodes.py
    except ImportError:
        return None
    mapping = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}) or {}
    for key, cls in mapping.items():
        lk = key.lower()
        if "referencesplitter" in lk or ("minimaxh3" in lk and "splitter" in lk):
            return cls
    return None


def _unpack_h3_refs(mmh3_refs):
    """Feeds the other pack's opaque H3_REFS bundle through their own
    Reference Splitter node (found via _find_reference_splitter_class) and
    returns a {slot_name: tensor_or_audio} dict built from that node's own
    RETURN_NAMES -- so this reads whatever slot layout that pack actually
    ships, by name, instead of hardcoding an assumed position or format."""
    splitter_cls = _find_reference_splitter_class()
    if splitter_cls is None:
        raise RuntimeError(
            "H3 AutoPrompt couldn't find 'MiniMax H3 Reference Splitter' in "
            "ComfyUI's node registry, so it can't unpack the H3_REFS bundle "
            "from Adudeguyman's Media Loader. Make sure ComfyUI-Fantastic-"
            "MiniMaxH3-PromptBuilder is installed and ComfyUI has been fully "
            "restarted. Alternatively, wire that pack's Media Loader through "
            "its own Reference Splitter node into this pack's ext_* inputs "
            "manually, or use H3 AutoPrompt: Media Loader's media_json input "
            "instead."
        )
    splitter = splitter_cls()
    func_name = getattr(splitter_cls, "FUNCTION")
    result = getattr(splitter, func_name)(mmh3_refs)
    names = getattr(splitter_cls, "RETURN_NAMES", None) or getattr(splitter_cls, "RETURN_TYPES")
    return dict(zip(names, result))


def _bundle_from_h3_refs(mmh3_refs):
    """H3_REFS bundle -> this pack's H3AP_REFS bundle, by delegating the
    unpacking to the other pack's own Reference Splitter (see above) and
    then reusing the same slot-alignment logic as the ext_* inputs."""
    slots = _unpack_h3_refs(mmh3_refs)
    picture_slots = [slots.get(f"picture_{i}") for i in range(1, PICTURES + 1)]
    video_slots = [slots.get(f"video_{i}") for i in range(1, VIDEOS + 1)]
    video_audio_slots = [slots.get(f"video_audio_{i}") for i in range(1, VIDEO_AUDIOS + 1)]
    audio_slots = [slots.get(f"audio_{i}") for i in range(1, AUDIOS + 1)]
    return H3AutoPromptMediaLoader._bundle_from_external_slots(
        picture_slots, video_slots, video_audio_slots, audio_slots)


# --------------------------------------------------------------------------
# 1. Media Loader
# --------------------------------------------------------------------------

def _audio_tags(references: dict) -> list:
    """The single source of truth for Audio N numbering: a video's paired
    soundtrack takes the low slots (in video order), standalone audio
    references fill the remaining slots after them -- matching the native
    H3 nodes' own convention. Every place that needs to know what 'Audio 2'
    refers to, or whether it exists, must call this rather than
    recomputing the numbering itself: the two independent copies of this
    logic that used to exist disagreed with each other and silently broke
    the voice-conversion safety net (see H3AutoPromptVLMAnalyzer.analyze)."""
    tm = references["tag_map"]
    n_paired = sum(1 for a in tm["video_audios"] if a)
    return [f"Audio {i+1}" for i in range(n_paired + len(references["audios_t"]))]


class H3AutoPromptMediaLoader:
    """`media_json` is a JSON list of
    {"kind":"picture|video|audio","file":"<abs path>","enabled":true,
     "audio_mode":"paired|standalone|off"}
    in the order you want them numbered. Point `file` at anything on disk
    (e.g. paths under ComfyUI's input/ directory); this node does not ship
    a drag-and-drop widget -- build media_json with a Note node, a small
    upstream script, or by hand.

    ALTERNATIVE, if you're already using Adudeguyman's ComfyUI-Fantastic-
    MiniMaxH3-PromptBuilder for its media handling: feed this node from
    that pack's own "MiniMax H3 Reference Splitter" node instead of
    filling in media_json. The Splitter unpacks that pack's opaque
    H3_REFS bundle into individually-typed picture_1..9 / video_1..3 /
    video_audio_1..3 / audio_1..3 outputs -- exactly the same slot layout
    this node uses -- so wire Splitter -> these optional inputs and this
    node builds its own H3AP_REFS bundle straight from them, no disk
    paths or JSON involved. When any of these optional inputs are
    connected, media_json is ignored entirely. This is deliberately built
    against their public, documented slot contract rather than their
    internal bundle format, so it keeps working even if that pack's
    internals change.
    ALTERNATIVE, if you're already using Adudeguyman's ComfyUI-Fantastic-
    MiniMaxH3-PromptBuilder for its media handling: wire that pack's
    "MiniMax H3 Media Loader" `references` output (type H3_REFS) straight
    into this node's `mmh3_refs` input -- no Splitter needed to place in
    the graph yourself, no media_json needed. Under the hood this node
    finds that pack's own "MiniMax H3 Reference Splitter" class through
    ComfyUI's node registry and calls it directly to do the unpacking, so
    it's reading their real code rather than a guessed dict format -- it
    requires that pack to be installed, but not for you to wire its
    Splitter node visibly.

    ALTERNATIVE #2: route through their Reference Splitter node yourself
    (e.g. because you're also feeding their native H3 nodes from the same
    media) and wire its outputs into the ext_* slot inputs below instead.

    Priority when more than one is wired: mmh3_refs > ext_* slots >
    media_json.
    """

    CATEGORY = CATEGORY
    RETURN_TYPES = ("H3AP_REFS", "STRING") + ("IMAGE",) * PICTURES + ("IMAGE",) * VIDEOS \
        + ("AUDIO",) * VIDEO_AUDIOS + ("AUDIO",) * AUDIOS
    RETURN_NAMES = ("references", "tag_map_json") \
        + tuple(f"picture_{i}" for i in range(1, PICTURES + 1)) \
        + tuple(f"video_{i}" for i in range(1, VIDEOS + 1)) \
        + tuple(f"video_audio_{i}" for i in range(1, VIDEO_AUDIOS + 1)) \
        + tuple(f"audio_{i}" for i in range(1, AUDIOS + 1))
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        optional = {"mmh3_refs": ("H3_REFS",)}
        for i in range(1, PICTURES + 1):
            optional[f"ext_picture_{i}"] = ("IMAGE",)
        for i in range(1, VIDEOS + 1):
            optional[f"ext_video_{i}"] = ("IMAGE",)
        for i in range(1, VIDEO_AUDIOS + 1):
            optional[f"ext_video_audio_{i}"] = ("AUDIO",)
        for i in range(1, AUDIOS + 1):
            optional[f"ext_audio_{i}"] = ("AUDIO",)
        return {
            "required": {
                "media_json": ("STRING", {"multiline": True, "default": "[]"}),
                "video_frame_samples": ("INT", {"default": 6, "min": 1, "max": 24}),
            },
            "optional": optional,
        }

    def load(self, media_json: str, video_frame_samples: int = 6,
              mmh3_refs: Optional[dict] = None, **kwargs):
        if mmh3_refs is not None:
            bundle = _bundle_from_h3_refs(mmh3_refs)
        else:
            picture_slots = [kwargs.get(f"ext_picture_{i}") for i in range(1, PICTURES + 1)]
            video_slots = [kwargs.get(f"ext_video_{i}") for i in range(1, VIDEOS + 1)]
            video_audio_slots = [kwargs.get(f"ext_video_audio_{i}") for i in range(1, VIDEO_AUDIOS + 1)]
            audio_slots = [kwargs.get(f"ext_audio_{i}") for i in range(1, AUDIOS + 1)]
            using_external = any(x is not None for x in
                                  picture_slots + video_slots + video_audio_slots + audio_slots)
            if using_external:
                bundle = self._bundle_from_external_slots(picture_slots, video_slots,
                                                            video_audio_slots, audio_slots)
            else:
                bundle = self._bundle_from_media_json(media_json, video_frame_samples)

        def pad(seq, n):
            return list(seq) + [None] * (n - len(seq))

        out = (bundle, json.dumps(bundle["tag_map"]))
        out += tuple(pad(bundle["pictures_t"], PICTURES))
        out += tuple(pad(bundle["videos_t"], VIDEOS))
        out += tuple(pad(bundle["video_audios_t"], VIDEO_AUDIOS))
        out += tuple(pad(bundle["audios_t"], AUDIOS))
        return out

    @staticmethod
    def _bundle_from_external_slots(picture_slots, video_slots, video_audio_slots, audio_slots):
        """Builds an H3AP_REFS bundle straight from already-unpacked slot
        tensors (e.g. from the other pack's Reference Splitter). video_slots
        and video_audio_slots stay positionally aligned before compaction --
        video_audio_i is that pack's paired soundtrack for video_i, exactly
        mirroring how compose_reference/build_tag_map expect video_audios_t
        to line up with videos_t elsewhere in this file."""
        pictures_t = [t for t in picture_slots if t is not None]
        kept_video_idx = [i for i, t in enumerate(video_slots) if t is not None]
        videos_t = [video_slots[i] for i in kept_video_idx]
        video_audios_t = [video_audio_slots[i] for i in kept_video_idx]
        audios_t = [t for t in audio_slots if t is not None]

        tag_map = {
            "pictures": [f"ext:picture:{i}" for i in range(len(pictures_t))],
            "videos": [f"ext:video:{i}" for i in range(len(videos_t))],
            "video_audios": [f"ext:video_audio:{i}" if a is not None else None
                              for i, a in enumerate(video_audios_t)],
            "audios": [f"ext:audio:{i}" for i in range(len(audios_t))],
        }
        return {
            "items": [], "tag_map": tag_map, "video_durations": {},
            "pictures_t": pictures_t, "videos_t": videos_t,
            "video_audios_t": video_audios_t, "audios_t": audios_t,
        }

    @staticmethod
    def _bundle_from_media_json(media_json: str, video_frame_samples: int):
        items = json.loads(media_json or "[]")
        media_items = []
        for it in items:
            kind = it["kind"]
            has_audio = it.get("has_audio")
            if kind == "video" and has_audio is None:
                has_audio = media_io.has_audio_stream(it["file"])
            media_items.append(c.MediaItem(
                kind=kind, file=it["file"], enabled=it.get("enabled", True),
                has_audio=bool(has_audio), audio_mode=it.get("audio_mode", "paired"),
            ))
        tag_map = c.build_tag_map(media_items)

        pictures = [media_io.load_image(p) for p in tag_map.pictures[:PICTURES]]
        videos = [media_io.load_video_frames(p, max_frames=video_frame_samples)
                  for p in tag_map.videos[:VIDEOS]]
        video_audios = [media_io.extract_audio(p) if p else None
                         for p in tag_map.video_audios[:VIDEO_AUDIOS]]
        audios = [media_io.load_audio(p) for p in tag_map.audios[:AUDIOS]]

        durations = {p: media_io.video_duration(p) for p in tag_map.videos}

        return {
            "items": [it.__dict__ for it in media_items],
            "tag_map": {
                "pictures": tag_map.pictures, "videos": tag_map.videos,
                "video_audios": tag_map.video_audios, "audios": tag_map.audios,
            },
            "video_durations": durations,
            "pictures_t": pictures, "videos_t": videos,
            "video_audios_t": video_audios, "audios_t": audios,
        }



_TAG_RE = re.compile(r"\b(audio|video|picture)\s*#?\s*(\d+)\b", re.IGNORECASE)
_VOICE_MATCH_VERBS_RE = re.compile(
    r"(?:modified|changed|converted|cloned|altered|re-?voiced|re-?synthesi[sz]ed)\s+"
    r"(?:to\s+)?(?:match|sound(?:s)?\s+like|resemble)", re.IGNORECASE)


def _canonicalize_tag(raw: str, known_tags) -> Optional[str]:
    """Matches a tag string against known canonical tags case/spacing-
    insensitively -- e.g. VLM output 'audio 1' or 'Audio1' both resolve to
    the real 'Audio 1' from the media bundle, rather than being rendered
    verbatim and silently breaking the guide's <Tag N> citation format."""
    norm = re.sub(r"[^a-z0-9]", "", raw.lower())
    for kt in known_tags:
        if re.sub(r"[^a-z0-9]", "", kt.lower()) == norm:
            return kt
    return None


def _find_voice_donor_pairs(brief: str, known_audio_tags: set) -> list:
    """Finds (source_tag, donor_tag) pairs from the brief's own "voice in X
    is modified/cloned/converted to match/sound like Y" phrasing -- source
    is whichever Audio tag was mentioned most recently before the verb
    phrase, donor is the Audio tag right after it. Same conservative,
    brief-text-only approach as the role override below, just capturing
    both sides of the relationship instead of only the donor."""
    pairs = []
    for m in _VOICE_MATCH_VERBS_RE.finditer(brief):
        head = brief[max(0, m.start() - 120):m.start()]
        tail = brief[m.end():m.end() + 100]
        tail_match = _TAG_RE.search(tail)
        if not tail_match or tail_match.group(1).lower() != "audio":
            continue
        donor = _canonicalize_tag(f"Audio {tail_match.group(2)}", known_audio_tags)
        if not donor:
            continue
        source = None
        for hm in reversed(list(_TAG_RE.finditer(head))):
            if hm.group(1).lower() != "audio":
                continue
            cand = _canonicalize_tag(f"Audio {hm.group(2)}", known_audio_tags)
            if cand and cand != donor:
                source = cand
                break
        if source:
            pairs.append((source, donor))
    return pairs


def _apply_voice_conversion_safety_net(scene: dict, brief: str, known_audio_tags: set) -> dict:
    """Deterministic backstop for the single most common VLM misclassification
    seen in practice: a brief that says "the voice in X is modified/cloned/
    converted to match/sound like the voice in Y" describes Y as a pure voice-
    timbre DONOR whose own words must never be spoken in the target video --
    but a small local VLM will often still tag Y "onscreen"/"voiceover"
    whenever Y's own clip happens to contain a complete, readable sentence,
    even after being shown a worked example of exactly this pattern.
    Two rounds of prompt strengthening didn't reliably fix that, so this
    scans the user's own brief text (not a guess at anything undocumented --
    it's the person's own words, already fully in context) for that specific
    relational phrasing and force-corrects the role regardless of what the
    VLM said. Conservative by design: it only fires on this one explicit
    "modified/cloned/converted ... to match/sound like ..." pattern and only
    touches the tag immediately following it -- it does not try to guess
    intent from anything vaguer than that.

    Also stores the detected (source, donor) pairs on scene["_voice_donor_pairs"]
    regardless of whether an override was needed -- the guide requires a donor
    clip to get its own "<Audio N> is the voice-timbre reference for <Subject
    N> (Sx)" line in subject_definitions plus a "reference" retention_analysis
    line (guide S2.2.4/S2.4.2), which needs this pairing to construct even
    when the VLM classified the role correctly on its own."""
    audio_roles = scene.get("audio_roles", [])
    by_tag = {}
    for r in audio_roles:
        canon = _canonicalize_tag(str(r.get("label", "")), known_audio_tags) or r.get("label")
        r["label"] = canon
        by_tag[canon] = r

    pairs = _find_voice_donor_pairs(brief, known_audio_tags)
    donor_tags = {donor for _, donor in pairs}

    corrected_any = False
    for tag in donor_tags:
        entry = by_tag.get(tag)
        if entry is None:
            entry = {"label": tag}
            audio_roles.append(entry)
            by_tag[tag] = entry
        if entry.get("role") != "voice_reference_only":
            entry["role"] = "voice_reference_only"
            entry["_safety_net_override"] = True
            corrected_any = True

    if corrected_any:
        # The VLM chose task_types based on its OWN (wrong) role classification, before
        # this override happened -- if it never thought Audio 2 was voice-reference-only,
        # it had no reason to include "audio reference" in the summary's task-type prefix
        # either. The guide's own checklist requires the prefix to match the asset's
        # actual role, so the correction has to cascade here too, not just into the role.
        task_types = scene.get("task_types") or ["reference generation"]
        if "audio reference" not in task_types:
            task_types.append("audio reference")
        scene["task_types"] = task_types

    scene["audio_roles"] = audio_roles
    scene["_voice_donor_pairs"] = [[s, d] for s, d in pairs]
    return scene


_REPLACE_VERBS_RE = re.compile(
    r"(?:replaces|replace|is\s+swapped\s+in\s+for|takes?\s+the\s+place\s+of|swaps?\s+in\s+for)",
    re.IGNORECASE)


def _find_replaced_tags(brief: str, known_tags: set) -> set:
    """Finds which Picture/Video tag the brief says gets REPLACED -- e.g. "the
    character in video 1 (...) replaces" or "X replaces the character in
    video 1" both mention a tag near a replacement verb. Same conservative,
    brief-text-only approach as the voice-donor pairing above."""
    replaced = set()
    for m in _REPLACE_VERBS_RE.finditer(brief):
        window = brief[max(0, m.start() - 100):m.end() + 100]
        for tm in _TAG_RE.finditer(window):
            canon = _canonicalize_tag(f"{tm.group(1).capitalize()} {tm.group(2)}", known_tags)
            if canon:
                replaced.add(canon)
    return replaced


def _apply_character_replacement_safety_net(scene: dict, brief: str,
                                             known_picture_tags: set, known_video_tags: set) -> dict:
    """Deterministic backstop for the other recurring VLM miss: the guide is
    explicit that a replaced character's own appearance gets NO subjects
    entry at all (guide's character-replacement handling, reinforced twice
    in the system prompt already) -- but a small local VLM keeps re-adding
    one anyway, describing the replaced character's body either standalone
    or merged into the environment entry, across multiple attempts even
    after being shown bad/good examples of exactly this mistake. Same
    reasoning as the voice-conversion safety net: after repeated prompt
    strengthening doesn't reliably fix a specific misclassification, a
    deterministic check against the brief's own words is more reliable than
    a fourth rewording. This drops (does not just warn about) any subjects
    entry sourced from a tag the brief identifies as "replaced", unless
    that entry is explicitly an environment/structural citation -- an
    environment entry sourced from the same video is legitimate and kept."""
    known_tags = known_picture_tags | known_video_tags
    replaced_tags = _find_replaced_tags(brief, known_tags)
    if not replaced_tags:
        return scene

    subjects = scene.get("subjects", [])
    kept, dropped = [], []
    for s in subjects:
        sources = {t.strip() for t in str(s.get("source", "")).split(",")}
        sources = {_canonicalize_tag(t, known_tags) or t for t in sources}
        if sources & replaced_tags and s.get("kind") != "environment":
            dropped.append(s.get("tag", "(unnamed)"))
            continue
        kept.append(s)

    scene["subjects"] = kept
    if dropped:
        scene["_replacement_safety_net_dropped"] = dropped
        # Dropping a subject here must not leave a dangling "<Subject 2>" (or, in the
        # summary field specifically, a bare unbracketed "Subject 2") citation anywhere
        # else -- an undefined reference label is worse than the original problem (guide's
        # own checklist: "every label defined once and used consistently"). Scrub shot
        # content, subject_mentions, AND summary of the dropped tag(s), rewriting the
        # common "replaces <Tag>" phrasing to stay grammatical rather than just excising
        # the citation.
        def _scrub(content: str, tag: str) -> str:
            esc = re.escape(tag)
            bracketed_or_bare = rf"<{esc}>|\b{esc}\b"
            content = re.sub(rf"\breplaces?\s+(?:{bracketed_or_bare})",
                              "replaces the original occupant", content)
            content = re.sub(rf"\bin\s+place\s+of\s+(?:{bracketed_or_bare})",
                              "in place of the original occupant", content)
            content = re.sub(bracketed_or_bare, "the original occupant of the scene", content)
            return re.sub(r"\s{2,}", " ", content).strip()

        for shot in scene.get("shots", []) or []:
            content = shot.get("content", "")
            for tag in dropped:
                content = _scrub(content, tag)
            shot["content"] = content
            if shot.get("subject_mentions"):
                shot["subject_mentions"] = [t for t in shot["subject_mentions"] if t not in dropped]

        if scene.get("summary"):
            summary = scene["summary"]
            for tag in dropped:
                summary = _scrub(summary, tag)
            scene["summary"] = summary
    return scene


_ANCHOR_SIGNAL_RE = re.compile(
    r"first\s+frame|last\s+frame|final\s+frame|opening\s+frame|ending\s+frame|"
    r"perfectly\s+preserved|exact(?:ly)?\s+frame|keyframe|key\s+frame|storyboard",
    re.IGNORECASE)


def _apply_anchor_safety_net(scene: dict, brief: str, known_picture_tags: set) -> dict:
    """Drops any anchor the VLM invented without the brief actually saying that
    picture is a concrete frame of the video. Same failure shape as the other
    two safety nets: "anchors" is a real, documented pattern (guide S2.2.2),
    but the VLM applies it speculatively to an ordinary identity reference the
    brief never calls a frame/keyframe/storyboard at all -- observed directly:
    a brief that only ever describes "Picture 1" as an identity source got a
    fabricated "is the first frame of [Shot 1]" anchor entry anyway. Kept
    conservative: only requires anchor language to appear somewhere near that
    specific picture's own tag mention in the brief, not an exact phrase."""
    anchors = scene.get("anchors", []) or []
    kept, dropped = [], []
    for a in anchors:
        tag = _canonicalize_tag(str(a.get("tag", "")), known_picture_tags)
        if tag is None:
            dropped.append(str(a.get("tag", "(unnamed)")))
            continue
        num = tag.split()[-1]
        signalled = any(
            _ANCHOR_SIGNAL_RE.search(brief[max(0, tm.start() - 100):tm.end() + 100])
            for tm in re.finditer(rf"\bpicture\s*#?\s*{num}\b", brief, re.IGNORECASE)
        )
        if signalled:
            a["tag"] = tag
            kept.append(a)
        else:
            dropped.append(tag)

    scene["anchors"] = kept
    if dropped:
        scene["_anchor_safety_net_dropped"] = dropped
        # A dropped anchor's shot-content citation ("The shot begins from <Picture 1>.")
        # never gets generated in the first place, since this runs before the Composer
        # ever reads scene["anchors"] -- unlike the character-replacement case, there's
        # nothing already rendered here to clean up.
    return scene


# --------------------------------------------------------------------------
# 2. VLM Analyzer
# --------------------------------------------------------------------------

class H3AutoPromptVLMAnalyzer:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("scene_json",)
    FUNCTION = "analyze"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "references": ("H3AP_REFS",),
            "brief": ("STRING", {"multiline": True, "default": ""}),
            "mode": (["T2VA", "I2VA", "FL2VA", "L2VA", "REF"],),
            "backend": (["llama_cpp_gguf", "transformers_hf"],),
            "model_path_or_id": ("STRING", {"multiline": False,
                "default": "models/vlm/qwen2-vl-2b-instruct-abliterated-q4_k_m.gguf"}),
            "mmproj_path": ("STRING", {"multiline": False,
                "default": "models/vlm/qwen2-vl-2b-instruct-mmproj-f16.gguf"}),
            "device": (["cuda", "cpu"],),
            "max_images": ("INT", {"default": 12, "min": 1, "max": 24}),
        }}

    def analyze(self, references: dict, brief: str, mode: str, backend: str,
                model_path_or_id: str, mmproj_path: str, device: str, max_images: int):
        images, labels = [], []
        tm = references["tag_map"]
        for i, t in enumerate(references["pictures_t"]):
            images.append(media_io.image_tensor_to_pil(t))
            labels.append(f"Picture {i + 1}")
        for i, frames in enumerate(references["videos_t"]):
            for f in range(min(frames.shape[0], 3)):
                images.append(media_io.image_tensor_to_pil(frames[f]))
                labels.append(f"Video {i + 1} frame {f + 1}")
        images, labels = images[:max_images], labels[:max_images]

        if backend == "llama_cpp_gguf":
            vlm = LlamaCppVLM(VLMConfig(model_path=model_path_or_id, mmproj_path=mmproj_path,
                                         n_gpu_layers=-1 if device == "cuda" else 0))
        else:
            vlm = TransformersVLM(model_id=model_path_or_id, device=device)

        result = vlm.analyze(images, brief, mode, labels)

        known_audio_tags = set(_audio_tags(references))
        result = _apply_voice_conversion_safety_net(result, brief, known_audio_tags)

        known_picture_tags = {f"Picture {i+1}" for i in range(len(references["pictures_t"]))}
        known_video_tags = {f"Video {i+1}" for i in range(len(references["videos_t"]))}
        result = _apply_character_replacement_safety_net(result, brief, known_picture_tags, known_video_tags)
        result = _apply_anchor_safety_net(result, brief, known_picture_tags)

        return (json.dumps(result),)


# --------------------------------------------------------------------------
# 3. Whisper node
# --------------------------------------------------------------------------

class H3AutoPromptWhisperNode:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("dialogue_json",)
    FUNCTION = "transcribe"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "references": ("H3AP_REFS",),
            "model_size": ("STRING", {"default": "large-v3-turbo"}),
            "device": (["cuda", "cpu"],),
            "use_diarization": ("BOOLEAN", {"default": True}),
            "forced_language": ("STRING", {"default": ""}),
            "speaker_labels_json": ("STRING", {"multiline": True, "default": "{}"}),
        }}

    def transcribe(self, references: dict, model_size: str, device: str,
                    use_diarization: bool, forced_language: str, speaker_labels_json: str):
        tm = references["tag_map"]
        speaker_labels = json.loads(speaker_labels_json or "{}")
        transcriber = FasterWhisperTranscriber(model_size=model_size, device=device,
            compute_type="float16" if device == "cuda" else "int8")

        # Audio numbering follows the same rule the native H3 nodes use: a video's
        # paired soundtrack takes the Audio slot that appears immediately before its
        # own video's slot; standalone audio references fill the remaining slots in
        # their own order. See _audio_tags -- the single shared implementation of
        # this rule (a second, divergent copy of it used to live here and quietly
        # broke tag matching elsewhere; don't reintroduce a second copy).
        video_audio_paths = [p for p in tm["video_audios"] if p]
        n_paired = len(video_audio_paths)
        all_tags = _audio_tags(references)

        clips = []
        tag_i = 0
        for i, audio in enumerate(references["video_audios_t"]):
            if audio is None:
                continue
            tag = all_tags[tag_i]
            tag_i += 1
            wav_path = _write_temp_wav(audio)
            dur = audio["waveform"].shape[-1] / audio["sample_rate"]
            clip = transcribe_clip(transcriber, wav_path, tag, dur,
                                    forced_language=forced_language or None,
                                    use_pyannote=use_diarization)
            clips.append(clip)
        for i, audio in enumerate(references["audios_t"]):
            tag = all_tags[tag_i]
            tag_i += 1
            wav_path = _write_temp_wav(audio)
            dur = audio["waveform"].shape[-1] / audio["sample_rate"]
            clip = transcribe_clip(transcriber, wav_path, tag, dur,
                                    forced_language=forced_language or None,
                                    use_pyannote=use_diarization)
            clips.append(clip)

        payload = []
        for clip in clips:
            payload.append({
                "media_tag": clip.media_tag,
                "clip_duration_s": clip.clip_duration_s,
                "turns": [{"start": t.start, "end": t.end, "text": t.text,
                           "lang": t.lang, "cluster": t.cluster} for t in clip.turns],
            })
        return (json.dumps(payload),)


def _write_temp_wav(audio: dict) -> str:
    import tempfile
    data = media_io.audio_to_wav_bytes(audio)
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    f.write(data)
    f.close()
    return f.name


# --------------------------------------------------------------------------
# 4. Composer
# --------------------------------------------------------------------------

class H3AutoPromptComposer:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "validation_report")
    FUNCTION = "compose"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mode": (["T2VA", "I2VA", "FL2VA", "L2VA", "REF"],),
            "scene_json": ("STRING", {"multiline": True}),
            "dialogue_json": ("STRING", {"multiline": True, "default": "[]"}),
            "duration_seconds": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 30.0}),
            "references": ("H3AP_REFS",),
        }}

    def compose(self, mode: str, scene_json: str, dialogue_json: str,
                duration_seconds: float, references: dict):
        scene = json.loads(scene_json)
        dialogue = json.loads(dialogue_json or "[]")
        tm = references["tag_map"]

        dialogue_clips = []
        for d in dialogue:
            from .whisper_engine import DialogueClip, SpeakerTurn
            turns = [SpeakerTurn(start=t["start"], end=t["end"], text=t["text"],
                                  lang=t["lang"], cluster=t["cluster"]) for t in d["turns"]]
            dialogue_clips.append(DialogueClip(media_tag=d["media_tag"], turns=turns,
                                                clip_duration_s=d["clip_duration_s"]))

        known_audio_tags = {d.media_tag for d in dialogue_clips}
        audio_roles = {}
        for r in scene.get("audio_roles", []):
            canon = _canonicalize_tag(str(r.get("label", "")), known_audio_tags) or r.get("label")
            audio_roles[canon] = r
        voiceover_clips = {tag for tag, r in audio_roles.items() if r.get("role") == "voiceover"}
        excluded_clips = {tag for tag, r in audio_roles.items()
                           if r.get("role") == "voice_reference_only"}
        subject_of_clip = {tag: r["subject_tag"] for tag, r in audio_roles.items()
                            if r.get("subject_tag")}

        # Clips the VLM marked "voice_reference_only" lend their SOUND to another
        # speaker but never actually speak in the target video -- drop them before
        # assigning (Sx) IDs too, or a donor-voice clip that happens to start earliest
        # would wrongly claim S1 despite never being heard.
        speaking_clips = [d for d in dialogue_clips if d.media_tag not in excluded_clips]
        speaker_ids = c.assign_speaker_ids(speaking_clips)

        speaker_labels = {}
        for key in speaker_ids:
            clip_tag = key.split("::", 1)[0]
            role = audio_roles.get(clip_tag, {})
            default = "an unseen narrator" if clip_tag in voiceover_clips else "the speaker"
            desc = (role.get("speaker_description") or "").strip().rstrip(".! ")
            speaker_labels[key] = desc or default

        vlm_shots = scene.get("shots", [])
        shots = c.build_shots(vlm_shots, dialogue_clips, duration_seconds,
                               speaker_labels, speaker_ids, subject_of_clip,
                               voiceover_clips=voiceover_clips, excluded_clips=excluded_clips)

        def _as_sentence(text) -> str:
            """Turns a noun-phrase fragment ('ambient room tone with subtle
            footsteps') into a grammatically complete sentence the guide's own
            'N sentences' wording actually requires, without depending on the
            VLM to remember terminal punctuation. Leaves already-complete text
            and 'N/A' untouched. Accepts None: scene.get(key, "N/A") only uses
            the default when the KEY is missing, not when the VLM's JSON
            explicitly set it to null -- text.strip() on that None crashed."""
            text = (text or "").strip()
            if not text or text == "N/A":
                return text or "N/A"
            if not text.endswith((".", "!", "?")):
                text = text[0].upper() + text[1:] + "."
            return text

        overall_soundscape = _as_sentence(scene.get("overall_soundscape", "N/A"))
        non_diegetic_music = _as_sentence(scene.get("non_diegetic_music", "N/A"))
        # Guide S2.6: when reference audio's ambience/score is actually reused (not just a
        # voice), state that relationship in the matching section, tag included.
        soundscape_cite = (scene.get("soundscape_audio_citation") or "").strip()
        if soundscape_cite and mode == "REF":
            overall_soundscape = _as_sentence(f"{soundscape_cite} {overall_soundscape}")
        music_cite = (scene.get("music_audio_citation") or "").strip()
        if music_cite and mode == "REF":
            non_diegetic_music = _as_sentence(f"{music_cite} {non_diegetic_music}")
            if non_diegetic_music.rstrip().endswith("N/A."):
                non_diegetic_music = _as_sentence(music_cite)
        style_text = scene.get("style", "Cinematic, live-action").rstrip(". ") 

        if mode != "REF":
            # Base modes: style is the opening clause of [Shot 1] itself (guide S1.4.1).
            if shots:
                shots[0].style = style_text + ","
            prompt = g.compose_base(mode, shots, overall_soundscape, non_diegetic_music,
                                     duration_seconds)
        else:
            RETENTION_EXPLANATIONS = {
                "fully_preserved": "this reference is reproduced exactly -- pose, framing, and "
                                    "setting are carried into the target video as shown.",
                "partially_preserved": "core identifying traits are kept, but pose, framing, or "
                                        "setting may differ from this exact reference.",
                "attribute_transfer": "only the subject's defining visual traits (not the specific "
                                       "pose/setting shown) are carried into the target video.",
                "weak_reference": "this reference is loose creative inspiration only; only a "
                                   "general resemblance should carry through.",
            }
            vlm_subjects = scene.get("subjects", [])
            known_video_tags = {f"Video {i+1}" for i in range(len(references["videos_t"]))}

            def _bracket_tags(csv: str) -> str:
                """'Picture 1, Picture 2' -> '<Picture 1>, <Picture 2>'. The guide
                brackets a reference label everywhere it's mentioned, not only in
                its own standalone line -- every worked example does this even
                for an ordinary identity citation inside a Subject definition
                ("is the young woman in <Picture 1>, with...", "in <Picture 2>,
                <Picture 3>, and <Picture 4>, with..."). This had been missing
                here, rendering plain unbracketed "(from Picture 1, Picture 2)"."""
                parts = [p.strip() for p in csv.split(",") if p.strip()]
                return ", ".join(f"<{p}>" if not p.startswith("<") else p for p in parts) or csv

            def _subject_text(s: dict) -> str:
                desc = (s.get("description") or "").strip()
                motion_src_raw = (s.get("motion_source") or "").strip()
                motion_src = _canonicalize_tag(motion_src_raw, known_video_tags) if motion_src_raw else None
                src_bracketed = _bracket_tags(s.get("source", "reference media"))
                if motion_src:
                    # Guide S2.2.1: appearance and motion can come from different assets --
                    # "is the woman whose appearance comes from <Picture 1> and whose walking
                    # motion comes from <Video 1>." Built deterministically so the tags always
                    # land correctly rather than trusting freeform VLM prose to get both tags
                    # right in one sentence. motion_source is validated against known Video
                    # tags the same way anchors are validated against Picture tags -- a
                    # fabricated or malformed tag here is dropped rather than rendered.
                    return (f"is {desc}, whose appearance comes from {src_bracketed} and whose "
                            f"motion comes from <{motion_src}>.")
                return f"is {desc} (from {src_bracketed})."

            subjects = [g.SubjectDef(tag=f"Subject {i+1}", text=_subject_text(s))
                        for i, s in enumerate(vlm_subjects)]

            # "where" (appears in [Shot X], [Shot Y]) comes from each shot's own
            # subject_mentions rather than being left blank -- guide S2.4.1 examples all
            # include it ("<Subject 1> (appears in [Shot 1], [Shot 3]): ...").
            appearances: Dict[str, list] = {}
            for shot_idx, vs in enumerate(vlm_shots, start=1):
                for tag in vs.get("subject_mentions", []) or []:
                    appearances.setdefault(tag, []).append(shot_idx)

            retention = []
            for i, s in enumerate(vlm_subjects):
                tag = f"Subject {i+1}"
                marker = s.get("retention_hint") or "attribute_transfer"
                if marker not in RETENTION_EXPLANATIONS:
                    marker = "attribute_transfer"
                shot_list = appearances.get(tag) or appearances.get(s.get("tag", ""), [])
                where = (f"(appears in {', '.join(f'[Shot {n}]' for n in shot_list)})"
                         if shot_list else "")
                retention.append(g.RetentionLine(tag=tag, where=where, marker=marker,
                                                  explanation=RETENTION_EXPLANATIONS[marker]))

            # Picture anchors (guide S2.2.2): a picture that literally IS a concrete frame
            # gets its own subject_definitions + retention_analysis entry, distinct from an
            # ordinary identity-source citation, and the shot(s) it anchors get an explicit
            # natural-language citation woven in -- built deterministically, not left to the
            # VLM's prose, since getting "[Shot N]" indices and tags right inside a sentence
            # is exactly the kind of formatting detail this pack keeps out of model hands.
            ANCHOR_PHRASING = {
                "first_frame": "The shot begins from <{tag}>.",
                "last_frame": "The shot ends on <{tag}>.",
                "keyframe": "The shot's keyframe corresponds to <{tag}>.",
                "storyboard": "<{tag}> serves as the storyboard reference for this shot.",
            }
            known_picture_tags = {f"Picture {i+1}" for i in range(len(references["pictures_t"]))}
            invalid_anchors = []
            for anchor in scene.get("anchors", []) or []:
                a_tag = _canonicalize_tag(str(anchor.get("tag", "")), known_picture_tags)
                if a_tag is None:
                    # Guards against exactly the failure seen in practice: the VLM citing
                    # one of the internal "Video 1 frame N" labels used to give it visual
                    # context, as if it were a real, citable H3 tag. Only "Picture N" is
                    # ever a valid anchor per the guide (S2.2.2) -- anything else, including
                    # any Video-derived tag, is dropped here rather than rendered, since a
                    # fabricated label in the output is worse than a missing anchor.
                    invalid_anchors.append(str(anchor.get("tag", "(missing)")))
                    continue
                a_role = anchor.get("role") if anchor.get("role") in ANCHOR_PHRASING else "keyframe"
                a_shots = [i for i in (anchor.get("shots") or []) if isinstance(i, int)]
                a_desc = (anchor.get("description") or "").strip()
                role_phrase = {"first_frame": "the first frame", "last_frame": "the last frame",
                                "keyframe": "a keyframe", "storyboard": "a storyboard reference"}[a_role]
                where_shots = ", ".join(f"[Shot {n}]" for n in a_shots) or "[Shot 1]"
                subjects.append(g.SubjectDef(
                    tag=a_tag, text=f"is {role_phrase} of {where_shots}, showing {a_desc}."))
                retention.append(g.RetentionLine(
                    tag=a_tag, where=f"({where_shots} {a_role.replace('_', ' ')})",
                    marker="fully_preserved",
                    explanation=f"{a_tag} is reproduced exactly as the anchor for {where_shots}."))
                phrase = ANCHOR_PHRASING[a_role].format(tag=a_tag)
                for shot_idx in a_shots:
                    if 1 <= shot_idx <= len(shots):
                        shots[shot_idx - 1].content = (shots[shot_idx - 1].content.rstrip()
                                                         + " " + phrase)
            # Audio uses its OWN retention marker vocabulary, distinct from the visual one
            # above (guide S2.4.2): fully_copy / partially_copy / reference / weak_reference.
            # A donor clip whose voice is borrowed but whose own words are never spoken also
            # needs its own subject_definitions line -- "<Audio N> is the voice-timbre
            # reference for <Subject N> (Sx)" (guide S2.2.4) -- not just a retention_analysis
            # entry. Both halves of that citation are built deterministically here from the
            # (source, donor) pairs _apply_voice_conversion_safety_net detected, rather than
            # trusted to the VLM's own prose: getting <Audio N> tags exactly right inside a
            # freeform sentence is exactly the kind of formatting detail this pack keeps out
            # of model hands everywhere else.
            donor_pairs = [tuple(p) for p in scene.get("_voice_donor_pairs", [])]
            donor_tags_seen = {donor for _, donor in donor_pairs}

            for tag, r in audio_roles.items():
                if tag in donor_tags_seen or tag in excluded_clips:
                    continue  # donors get their dedicated block below; excluded clips never speak at all
                role = r.get("role")
                if role not in ("onscreen", "voiceover"):
                    continue
                # Every audio tag whose dialogue actually appears in detailed_description
                # needs its own subject_definitions line too, not just a retention_analysis
                # entry -- "every label defined once and used consistently across all
                # sections" (guide checklist) means defined SOMEWHERE, and retention_analysis
                # alone doesn't count as a definition. This used to only happen for clips
                # with a voice_retention_note (the voice-conversion case); a plain reused
                # dialogue clip with no modification got no citation anywhere at all.
                sx = next((v for k, v in speaker_ids.items() if k.startswith(f"{tag}::")), None)
                subj_tag = r.get("subject_tag")
                if sx and subj_tag:
                    who = f"<{subj_tag}> ({sx})"
                elif sx:
                    who = f"{speaker_labels.get(f'{tag}::0', 'the speaker')} ({sx})"
                else:
                    who = "the target speaker"
                subjects.append(g.SubjectDef(
                    tag=tag, text=f"is the original dialogue track spoken by {who} "
                                  f"in the target video."))
                note = (r.get("voice_retention_note") or "").strip()
                explanation = note or ("the source audio's spoken words are retained; timing "
                                        "is trimmed to fit the target video's duration.")
                retention.append(g.RetentionLine(tag=tag, where="", marker="partially_copy",
                                                   explanation=explanation))

            for source_tag, donor_tag in donor_pairs:
                sx = next((v for k, v in speaker_ids.items()
                           if k.startswith(f"{source_tag}::")), None)
                role_entry = audio_roles.get(source_tag, {})
                subj_tag = role_entry.get("subject_tag")
                if sx and subj_tag:
                    who = f"<{subj_tag}> ({sx})"
                elif sx:
                    who = f"{speaker_labels.get(f'{source_tag}::0', 'the speaker')} ({sx})"
                else:
                    who = "the target speaker"
                subjects.append(g.SubjectDef(tag=donor_tag,
                                              text=f"is the voice-timbre reference for {who}."))
                retention.append(g.RetentionLine(
                    tag=donor_tag, where="", marker="reference",
                    explanation=f"the target speaker follows <{donor_tag}>'s voice timbre and "
                                f"delivery without copying the original signal."))
                # Make sure the source's own line (built above from voice_retention_note, if
                # any) actually names its donor -- if the VLM's note omitted it, add a plain
                # fallback line so the relationship isn't silently one-directional.
                if not any(rl.tag == source_tag for rl in retention):
                    retention.append(g.RetentionLine(
                        tag=source_tag, where="", marker="partially_copy",
                        explanation=f"original spoken words retained; the voice itself is "
                                    f"replaced with <{donor_tag}>'s timbre."))

            task_types = scene.get("task_types") or ["reference generation"]

            # Guide S2.3: "the summary uses the previously defined labels to describe the
            # main subjects, shot flow, and roles of the reference assets" -- this had been
            # hardcoded generic filler that never cited a single tag, which doesn't do what
            # this section is for. The VLM now writes real content citing bare tag names
            # (see its own instructions); bracket them deterministically here rather than
            # trust freeform prose to place brackets correctly, and only bracket tags that
            # are actually defined -- guide checklist forbids introducing new labels in the
            # summary, so an invented tag is left as plain unbracketed text instead of
            # becoming a citation to something that doesn't exist.
            known_summary_tags = {s.tag for s in subjects}
            _bare_tag_re = re.compile(r"\b(subject|picture|video|audio)\s+(\d+)\b", re.IGNORECASE)

            def _finalize_summary(raw: str) -> str:
                fallback = ("The target video follows the user's brief using the subjects "
                            "defined above.")
                text = (raw or "").strip()
                if not text:
                    return fallback

                # Guard against a lazy or visibly incomplete VLM response ("Subject 1 ...")
                # -- a bland but valid fallback line is better than a broken fragment
                # reaching the final prompt. This is not a token-truncation case (a
                # genuinely truncated JSON response would have failed to parse at all,
                # see _extract_json's closing-brace requirement) -- it's the model
                # choosing weak content for a harder-than-usual combination (e.g. a
                # picture that's both an anchor and a subject source), and the result
                # is unusable either way. Trailing ellipsis with very few real words
                # before it is the tell.
                core = re.sub(r"\.{2,}\s*$", "", text).strip()
                if len(re.findall(r"\w+", core)) < 4:
                    return fallback
                text = core

                def repl(m):
                    canon = _canonicalize_tag(f"{m.group(1)} {m.group(2)}", known_summary_tags)
                    return f"<{canon}>" if canon else m.group(0)

                text = _bare_tag_re.sub(repl, text)
                if not text.endswith((".", "!", "?")):
                    text += "."
                return text

            summary_text = _finalize_summary(scene.get("summary", ""))
            # REF mode: style is established in its own sentence BEFORE [Shot 1]
            # (guide S2.5.2) -- never also set on shots[0].style, or it repeats. Both of the
            # guide's own examples wrap it in a full sentence -- "The target video is in a
            # cinematic, literary music-video style with soft lighting..." (S2.5.2) and "The
            # target video uses a realistic multi-camera sitcom style with warm indoor
            # lighting." (S2.7) -- never the bare phrase base mode uses inside [Shot 1]
            # itself. This had been reusing the base-mode bare-fragment form ("Live-action,
            # cinematic.") for REF mode too, which doesn't match either example.
            style_lower = (style_text[0].lower() + style_text[1:]) if style_text else \
                "cinematic, live-action"
            # Don't append "style" again if the style text already contains it naturally
            # (e.g. a richer VLM-written phrase like "realistic multi-camera sitcom style
            # with warm indoor lighting") -- otherwise this doubles up into "... lighting
            # style.", which the first version of this fix did.
            style_intro = (f"The target video uses a {style_lower}." if re.search(r"\bstyle\b", style_lower)
                            else f"The target video uses a {style_lower} style.")
            prompt = g.compose_reference(subjects, task_types, summary_text, retention,
                                          style_intro, shots,
                                          overall_soundscape, non_diegetic_music)

        problems = g.validate(prompt, mode, duration_seconds)

        advisories = []
        if locals().get("invalid_anchors"):
            advisories.append(
                f"The VLM tried to cite {', '.join(invalid_anchors)} as a frame anchor, but "
                f"that isn't a valid reference label (only \"Picture N\" can be an anchor) -- "
                f"it was dropped rather than included in the prompt. If a picture genuinely "
                f"anchors a shot, check the raw scene_json to see why it wasn't tagged as a "
                f"plain \"Picture N\"."
            )
        if scene.get("_replacement_safety_net_dropped"):
            advisories.append(
                f"{', '.join(scene['_replacement_safety_net_dropped'])} was removed "
                f"automatically -- your brief describes its source as being replaced by "
                f"another subject, so it shouldn't have its own appearance retained. The "
                f"VLM defined it anyway; this is usually correct to drop, but worth a glance "
                f"at subject_definitions above to confirm nothing legitimate was removed."
            )
        if scene.get("_anchor_safety_net_dropped"):
            advisories.append(
                f"{', '.join(scene['_anchor_safety_net_dropped'])} was NOT treated as a "
                f"concrete frame anchor -- your brief never actually calls it a first/last "
                f"frame, keyframe, or storyboard near its own mention. The VLM tagged it as "
                f"one anyway; it's still cited as an ordinary identity reference in "
                f"subject_definitions, just not as an anchor. If you did mean for it to "
                f"literally be a specific frame, say so explicitly in the brief (e.g. "
                f"\"Picture 1 is the exact first frame\")."
            )
        overridden = [tag for tag, r in audio_roles.items() if r.get("_safety_net_override")]
        if overridden:
            advisories.append(
                f"{', '.join(overridden)} was reclassified as voice_reference_only "
                f"automatically, based on 'modified/cloned/converted ... to match/sound "
                f"like ...' language found in your brief -- the VLM itself tagged it "
                f"differently. This is usually correct, but worth a glance at the "
                f"detailed_description above to confirm."
            )
        for clip in dialogue_clips:
            if not clip.turns:
                continue
            span = max(t.end for t in clip.turns)
            if clip.media_tag in excluded_clips:
                continue
            if span > duration_seconds:
                dropped = sum(1 for t in clip.turns if t.start >= duration_seconds)
                advisories.append(
                    f"{clip.media_tag}'s transcribed speech runs to ~{span:.1f}s, longer "
                    f"than the {duration_seconds:.1f}s target video -- "
                    f"{dropped} line(s) starting past the video's end were dropped rather "
                    f"than crammed in. If that's not the excerpt you wanted spoken, trim "
                    f"the reference clip or shorten it to just the lines that matter."
                )
            if clip.clip_duration_s > 15.0:
                advisories.append(
                    f"{clip.media_tag} is {clip.clip_duration_s:.1f}s long, over H3's own "
                    f"15-second single-reference-clip budget -- consider trimming it."
                )

        report_lines = []
        report_lines.append("OK - passes the mechanical checklist." if not problems else
                             "ISSUES FOUND:\n" + "\n".join(f"- {p}" for p in problems))
        if advisories:
            report_lines.append("ADVISORY:\n" + "\n".join(f"- {a}" for a in advisories))
        report = "\n\n".join(report_lines)
        return (prompt, report)


# --------------------------------------------------------------------------
# 5. All-in-one convenience node
# --------------------------------------------------------------------------

class H3AutoPromptAllInOne:
    """Brief + media in, finished H3 prompt out. Internally chains the VLM
    analyzer, the Whisper node (only run when audio references are
    present), and the composer. Also passes media straight through with
    the same slot names the native MiniMax H3 nodes expect, so you can
    wire this node directly in front of MiniMax H3 Image to Video /
    MiniMax H3 Reference to Video with nothing else in between."""

    CATEGORY = CATEGORY
    # Media leaves as ONE bundle rather than 18 separate slots: that many outputs
    # forced the node to be absurdly tall on the canvas. Feed the bundle into
    # "H3 AutoPrompt: Reference Splitter" to fan it back out into the individual
    # ref_* slots the native MiniMax H3 nodes expect.
    RETURN_TYPES = ("STRING", "H3_MEDIA_BUNDLE")
    RETURN_NAMES = ("prompt", "media")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Only the structured loader feeds this node now. The old H3AP_REFS
                # and H3_REFS inputs (and the categorizer's item JSON) are gone along
                # with the nodes that produced them -- the Subject/Shot/Audio
                # categories are already explicit, so there is nothing left to infer.
                "references": ("H3_STRUCTURED_REFS",),
                # A normal widget that also accepts a link: type a value here, or wire
                # the loader's duration output in and that wins. forceInput used to make
                # it link-ONLY, which left no way to set it by hand.
                "duration_seconds": ("FLOAT", {"default": 10.0, "min": 5.0, "max": 15.0}),
                "vlm_backend": (["llama_cpp_gguf", "transformers_hf"],),
                # Dropdowns listing models/gguf when anything is there, so there's no
                # path to copy and paste; a plain text field otherwise, which also
                # covers the transformers_hf backend (that wants a HF repo id, not a
                # file). _resolve_gguf turns a selection back into a real path.
                "vlm_model_path_or_id": ((_GGUF_CHOICES,) if _GGUF_CHOICES
                                          else ("STRING", {"default": ""})),
                "vlm_mmproj_path": ((_GGUF_CHOICES,) if _GGUF_CHOICES
                                     else ("STRING", {"default": ""})),
                "whisper_model_size": ("STRING", {"default": "large-v3-turbo"}),
                "device": (["cuda", "cpu"],),
                "use_diarization": ("BOOLEAN", {"default": True}),
                # Empirically-grounded default: a real crash (MTMDChatHandler "not all
                # media inputs were consumed") showed this exact llama-cpp-python/mmproj
                # combo reliably handling 6 images in one call but not 9.
                # Each image costs roughly 40s on a 4B vision model, and on builds
                # with a broken batched path that is per image, so this is the
                # single biggest lever on run time.
                "max_vlm_images": ("INT", {"default": 4, "min": 1, "max": 24}),
                "vlm_detail": (["accurate (896px)", "faster (704px)",
                                 "fastest (512px)"],),
                # Off skips the vision model completely: your own descriptions are
                # used and references get generic placeholder descriptions.
                "use_vlm": ("BOOLEAN", {"default": True}),
                "vlm_n_ctx": ("INT", {"default": 16384, "min": 2048, "max": 131072}),
                # Visual style of the target video. "auto" reads it off the
                # reference images, which is what an all-anime reference set needs
                # -- this was previously hardcoded to live-action.
                # STRING rather than a fixed combo: the preset list can't cover
                # every look ("1970s grindhouse", "stop-motion felt"), and the
                # panel offers the presets plus a custom box on top of this.
                "style": ("STRING", {"multiline": False,
                                      "default": "auto (detect from references)"}),
                # Traces the whole pipeline to the ComfyUI console: which file got
                # which tag, what the plan made of it, what the VLM was shown and
                # returned, how shots and speakers were assembled, and the final
                # prompt. Off by default; costs nothing when off.
                "debug": ("BOOLEAN", {"default": False}),
                # Whole-video guide, used when the loader defines no shots (pairs
                # naturally with a storyboard). The panel shows it only in that
                # case and swaps to per-shot boxes once shots exist. Single-line
                # for the same DOM-backed-widget reason as the fields below; the
                # panel renders a real textarea over it.
                "brief": ("STRING", {"multiline": False, "default": ""}),
                # Per-shot prompt boxes, written by web/allinone_panel.js. Each entry is
                # {"text": str, "dialogue": [...]} -- bare strings are also accepted for
                # anything saved before dialogue lines existed. multiline MUST stay False:
                # a multiline STRING is DOM-backed and loses its value when ComfyUI
                # rebuilds the DOM on a workflow tab switch.
                "shot_briefs": ("STRING", {"multiline": False, "default": "[]"}),
            },
        }

    def run(self, references: dict, duration_seconds: float,
            vlm_backend: str, vlm_model_path_or_id: str, vlm_mmproj_path: str,
            whisper_model_size: str, device: str, use_diarization: bool,
            max_vlm_images: int = 6, vlm_n_ctx: int = 16384,
            shot_briefs: str = "[]", debug: bool = False,
            style: str = "auto (detect from references)", brief: str = "",
            vlm_detail: str = "accurate (896px)", use_vlm: bool = True):
        from . import debug_log as dbg
        from .structured_nodes import H3StructuredComposer

        dbg.set_enabled(debug)
        dbg.section("H3 AUTOPROMPT RUN")
        dbg.kv("duration", f"{duration_seconds}s")
        dbg.kv("shot_briefs raw", shot_briefs)
        dbg.kv("brief", brief or "(none)")

        composer = H3StructuredComposer()
        # The composer still runs its checklist internally; the report just isn't an
        # output slot anymore. Anything it flags is printed to the ComfyUI console.
        prompt, report = composer.compose(
            references, brief, duration_seconds, vlm_backend,
            _resolve_gguf(vlm_model_path_or_id), _resolve_gguf(vlm_mmproj_path),
            whisper_model_size, device, use_diarization,
            max_vlm_images=max_vlm_images, vlm_n_ctx=vlm_n_ctx,
            shot_prompts=shot_briefs, style=style,
            vlm_detail=vlm_detail, use_vlm=use_vlm)

        # Pass the already-decoded media straight out on the slot names the native
        # MiniMax H3 nodes expect, so this node can sit directly in front of
        # MiniMaxH3ReferenceToVideo with nothing in between. The structured loader
        # numbers every audio asset as "Audio N" regardless of whether it came from a
        # standalone file or a video's track, so they all land on audio_* and the
        # video_audio_* slots stay empty rather than guessing a split.
        if report and "ISSUES FOUND" in report:
            print(f"[H3 AutoPrompt] {report}")
        ti = getattr(composer, "_last_tensor_index", {}) or {}
        dbg.section("MEDIA PASSED THROUGH")
        for _tag in sorted(ti):
            dbg.bullet(f"{_tag} -> splitter slot")
        dbg.section("DONE")
        dbg.set_enabled(False)
        return (prompt, {"tensor_index": ti})


class H3ReferenceSplitter:
    """Fans the All-in-One node's single media bundle back out into the
    individual ref_* slots MiniMaxH3ReferenceToVideo expects. Exists so the
    All-in-One node doesn't need 18 output slots of its own, which made it
    unusably tall on the canvas."""

    CATEGORY = CATEGORY
    RETURN_TYPES = ("IMAGE",) * PICTURES + ("IMAGE",) * VIDEOS \
        + ("AUDIO",) * VIDEO_AUDIOS + ("AUDIO",) * AUDIOS
    RETURN_NAMES = tuple(f"picture_{i}" for i in range(1, PICTURES + 1)) \
        + tuple(f"video_{i}" for i in range(1, VIDEOS + 1)) \
        + tuple(f"video_audio_{i}" for i in range(1, VIDEO_AUDIOS + 1)) \
        + tuple(f"audio_{i}" for i in range(1, AUDIOS + 1))
    FUNCTION = "split"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"media": ("H3_MEDIA_BUNDLE",)}}

    def split(self, media: dict):
        ti = (media or {}).get("tensor_index") or {}
        out = tuple(ti.get(f"Picture {i}") for i in range(1, PICTURES + 1))
        out += tuple(ti.get(f"Video {i}") for i in range(1, VIDEOS + 1))
        # The structured loader numbers every audio asset "Audio N" whether it came
        # from a standalone file or a video's track, so they all land on audio_* and
        # video_audio_* stays empty rather than guessing a split.
        out += (None,) * VIDEO_AUDIOS
        out += tuple(ti.get(f"Audio {i}") for i in range(1, AUDIOS + 1))
        return out

# Only two nodes are registered now: the structured loader and this all-in-one.
# The earlier granular nodes (separate media loader, VLM analyzer, whisper node,
# standalone composer, subject categorizer) and the categorized composer are gone
# -- their classes were either deleted or are still imported internally as plain
# helpers, but none of them appear in the node menu anymore.
NODE_CLASS_MAPPINGS = {
    "H3AutoPromptAllInOne": H3AutoPromptAllInOne,
    "H3ReferenceSplitter": H3ReferenceSplitter,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3AutoPromptAllInOne": "H3 AutoPrompt: All-in-One",
    "H3ReferenceSplitter": "H3 AutoPrompt: Reference Splitter",
}
