"""
composer.py -- the glue between vlm_engine output, whisper_engine output,
and grammar.py's deterministic formatter. This is where "a VLM helps
interpret the media" and "Whisper transcribes dialogue" turn into shots,
speaker IDs, and reference-label bookkeeping.

Everything that affects legality of the final prompt (numbering, tag
consistency, cut-time math) happens here in plain Python, not in the model
prompts -- the VLM and Whisper are only ever asked for content, never for
formatting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import grammar as g
from .whisper_engine import DialogueClip, SpeakerTurn

# --------------------------------------------------------------------------
# Reference tag numbering (mirrors Adudeguyman's MediaLoader convention:
# gaps close up, a video's paired soundtrack takes the audio slot that
# appears immediately before its own video's slot).
# --------------------------------------------------------------------------

@dataclass
class MediaItem:
    kind: str                 # "picture" | "video" | "audio"
    file: str
    enabled: bool = True
    has_audio: bool = False
    audio_mode: str = "paired"   # "paired" | "standalone" | "off"


@dataclass
class TagMap:
    pictures: List[str]              # ordered list of file paths -> Picture 1..N
    videos: List[str]
    video_audios: List[Optional[str]]  # parallel to videos; None if that video's audio isn't a separate ref
    audios: List[str]                  # standalone audio files (incl. "alone" video audio)

    def picture_tag(self, path: str) -> Optional[str]:
        return f"Picture {self.pictures.index(path) + 1}" if path in self.pictures else None

    def video_tag(self, path: str) -> Optional[str]:
        return f"Video {self.videos.index(path) + 1}" if path in self.videos else None

    def audio_tag_for_video(self, path: str) -> Optional[str]:
        if path not in self.videos:
            return None
        vi = self.videos.index(path)
        if self.video_audios[vi] is None:
            return None
        # count how many audio slots precede this one (paired video-audios + standalone audios,
        # in the order MiniMax H3 actually receives them)
        n_before = sum(1 for a in self.video_audios[:vi] if a is not None)
        return f"Audio {n_before + 1}"

    def audio_tag_standalone(self, path: str) -> Optional[str]:
        n_paired = sum(1 for a in self.video_audios if a is not None)
        if path not in self.audios:
            return None
        return f"Audio {n_paired + self.audios.index(path) + 1}"


def build_tag_map(items: List[MediaItem]) -> TagMap:
    items = [i for i in items if i.enabled]
    pictures = [i.file for i in items if i.kind == "picture"]
    videos = [i.file for i in items if i.kind == "video"]
    video_audios: List[Optional[str]] = []
    audios: List[str] = []
    for i in items:
        if i.kind == "video":
            if i.has_audio and i.audio_mode == "paired":
                video_audios.append(i.file)
            else:
                video_audios.append(None)
            if i.has_audio and i.audio_mode == "standalone":
                audios.append(i.file)
        elif i.kind == "audio":
            audios.append(i.file)
    return TagMap(pictures=pictures, videos=videos, video_audios=video_audios, audios=audios)


# --------------------------------------------------------------------------
# Shots: distribute VLM shot list across the timeline, then drop in dialogue
# --------------------------------------------------------------------------

def _speaker_key(clip_tag: str, cluster: int) -> str:
    return f"{clip_tag}::{cluster}"


def assign_speaker_ids(dialogue_clips: List[DialogueClip]) -> Dict[str, str]:
    """Stable (S1), (S2)... IDs in order of first vocal event across ALL
    clips, exactly as the guide requires ('assign (Sx) once according to
    the order of actual vocal events in the target video')."""
    events: List[Tuple[float, str]] = []
    for clip in dialogue_clips:
        for turn in clip.turns:
            events.append((turn.start, _speaker_key(clip.media_tag, turn.cluster)))
    events.sort(key=lambda e: e[0])
    ids: Dict[str, str] = {}
    counter = 0
    for _, key in events:
        if key not in ids:
            counter += 1
            ids[key] = f"S{counter}"
    return ids


def build_shots(vlm_shots: List[dict], dialogue_clips: List[DialogueClip],
                 duration_s: float, speaker_labels: Dict[str, str],
                 speaker_ids: Dict[str, str], subject_of_clip: Dict[str, str],
                 voiceover_clips: Optional[set] = None,
                 excluded_clips: Optional[set] = None,
                 sung_clips: Optional[set] = None,
                 style_on_shot_one: bool = True) -> List[g.Shot]:
    """speaker_labels: clip_tag::cluster -> human description e.g. 'the
    young woman with long blonde hair'. subject_of_clip: clip_tag ->
    '<Subject 3>' if that clip's voice maps to a defined REF-mode subject
    (leave empty dict for base modes). voiceover_clips: set of media_tag
    whose dialogue should render as an off-screen voiceover (e.g. the VLM
    classified it as narration) rather than as an on-screen speaker.
    sung_clips: set of media_tag whose dialogue is sung lyrics rather than
    spoken dialogue -- guide treats both identically inside <d>, but the
    verb outside it ("sings" vs "says") should still match.
    excluded_clips: set of media_tag whose turns are skipped entirely --
    for clips the VLM classified 'voice_reference_only' (the clip lends
    its voice's SOUND to another speaker but its own words are never
    actually spoken in the target video)."""
    voiceover_clips = voiceover_clips or set()
    excluded_clips = excluded_clips or set()
    sung_clips = sung_clips or set()
    n = len(vlm_shots)
    if n == 0:
        raise ValueError("VLM returned no shots.")

    hints = [max(0.1, float(s.get("duration_hint_s") or (duration_s / n))) for s in vlm_shots]
    total_hint = sum(hints)
    cum = 0.0
    boundaries = []  # (start, end) per shot, scaled to duration_s
    for h in hints:
        start = cum
        cum += h / total_hint * duration_s
        boundaries.append((start, cum))
    boundaries[-1] = (boundaries[-1][0], duration_s)

    shots: List[g.Shot] = []
    for idx, (vs, (start, end)) in enumerate(zip(vlm_shots, boundaries), start=1):
        shot = g.Shot(
            index=idx,
            content=vs.get("content", "").strip(),
            cut_time_s=None if idx == 1 else round(start, 3),
            camera_motion=vs.get("camera"),
            camera_amplitude=vs.get("amplitude"),
            camera_speed=vs.get("speed"),
            onscreen_text=vs.get("onscreen_text"),
            picture_anchor=vs.get("picture_anchor"),
            style=(None),
        )
        shots.append(shot)

    if style_on_shot_one and vlm_shots:
        # style is prepended separately by grammar.compose_* callers via shots[0].style
        pass

    # Bucket dialogue turns into the shot whose [start,end) contains the turn's midpoint.
    # Turns starting at or after the video's own duration are dropped outright -- if a
    # reference clip's real transcript runs far longer than the target video (a common
    # case: a full monologue used as a voice/word source for a much shorter shot), only
    # the portion that actually falls inside the video gets used, rather than stuffing
    # every remaining line into the last shot with a <cutoff> tag on each one. Only the
    # single turn that's genuinely the last one kept, and that itself runs past the
    # video's end, gets marked <cutoff>.
    kept_by_shot: Dict[int, List[tuple]] = {}
    for clip in dialogue_clips:
        if clip.media_tag in excluded_clips:
            continue
        for turn in clip.turns:
            if turn.start >= duration_s:
                continue
            mid = (turn.start + turn.end) / 2
            shot_idx = next((i for i, (s, e) in enumerate(boundaries) if s <= mid < e),
                             len(boundaries) - 1)
            kept_by_shot.setdefault(shot_idx, []).append((clip, turn))

    last_shot_with_dialogue = max(kept_by_shot.keys()) if kept_by_shot else None
    seen_sx: set = set()
    for shot_idx in sorted(kept_by_shot.keys()):
        entries = kept_by_shot[shot_idx]
        entries.sort(key=lambda ct: ct[1].start)
        for pos, (clip, turn) in enumerate(entries):
            key = _speaker_key(clip.media_tag, turn.cluster)
            sx = speaker_ids.get(key, "S1")
            label = speaker_labels.get(key, "the speaker")
            spans_next = shot_idx < len(boundaries) - 1 and turn.end > boundaries[shot_idx][1]
            spans_prev = shot_idx > 0 and turn.start < boundaries[shot_idx][0]
            is_last_kept = (shot_idx == last_shot_with_dialogue and pos == len(entries) - 1)
            cutoff = is_last_kept and turn.end > duration_s - 0.05
            first_mention = sx not in seen_sx
            seen_sx.add(sx)
            shot_start, shot_end = boundaries[shot_idx]
            midpoint = (shot_start + shot_end) / 2
            position = "before" if turn.start < midpoint else "after"
            ev = g.DialogueEvent(
                speaker_label=label, sx=sx, lang=turn.lang, text=turn.text,
                voiceover=clip.media_tag in voiceover_clips, first_mention=first_mention,
                sung=clip.media_tag in sung_clips,
                scenetrans_out=spans_next, scenetrans_in=spans_prev, cutoff=cutoff,
                subject_ref=subject_of_clip.get(clip.media_tag), position=position,
            )
            shots[shot_idx].dialogue.append(ev)

    return shots
