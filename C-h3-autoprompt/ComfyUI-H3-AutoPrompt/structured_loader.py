"""
structured_loader.py -- backend for the new H3 Structured Media Loader.

Replaces dependence on the other pack's flat "items + cat_* fields" model
with three explicit categories (Subject, Shot, Audio) built directly into
the loader itself, matching how a person actually thinks about assembling
a reference set: "here's my character," "here's what happens in this shot,"
"here's audio that spans the whole thing" -- rather than loading a pile of
generic media and tagging each piece's role afterward.

This module is pure data transformation (like categorize.py) -- no model
calls, fully testable without a GPU. It builds the SAME CategorizationPlan
shape categorize.py already produces, so categorized_nodes.py's rendering
logic (subject_definitions, retention_analysis, summary, the narrow VLM
call) is reused as-is; only how the plan gets built is new.

Data model (JSON, written by the loader's JS panel):

{
  "subjects": [
    {
      "complete": {"file": str, "kind": "picture"|"video"} | null,
      "visual_refs": [{"file": str}, ...],
      "outfit_refs": [{"file": str}, ...],   # index 0 = "Outfit Shot(s) 1 or All",
                                              # index N = "Outfit Shot N+1"
      "vocal_ref": {"file": str, "kind": "audio"|"video"} | null
    }, ...
  ],
  "shots": [
    {
      "complete": {"file": str} | null,      # replacement-type; clears everything else
      "first_frame": {"file": str} | null,
      "last_frame": {"file": str} | null,
      "motion": {"file": str} | null,        # text-only motion citation (guide's own
                                              # "<Video N> (motion)" vocabulary) -- NOT
                                              # real ControlNet conditioning, which this
                                              # pack has no way to wire in at all (a
                                              # ControlNet-Union checkpoint for MiniMax H3
                                              # exists, but it's a wholly separate
                                              # VideoX-Fun inference pipeline with no
                                              # ComfyUI node and a different generation
                                              # path than the text-prompt-plus-references
                                              # pipeline this pack targets)
      "music": {"file": str} | null,
      "sfx": {"file": str} | null,
      "reference_images": [{"file": str}, ...]  # generic, VLM-interpreted freely
    }, ...
  ],
  "audio": {
    "dialogue": [{"file": str}, ...],   # spans all shots, Whisper-transcribed
    "music": [{"file": str}, ...]       # spans all shots, full-video score
  }
}

Shared caps enforced across ALL categories combined, matching the native
MiniMaxH3ReferenceToVideo node's actual slot limits: 9 images, 3 videos,
3 audio.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import debug_log as dbg
from .categorize import (
    AnchorPlan, AudioPlan, CategorizationPlan, SubjectPlan, VideoStructurePlan,
)

MAX_PICTURES = 9
MAX_VIDEOS = 3
MAX_AUDIO = 3

try:                       # available inside ComfyUI, absent in bare tests
    import folder_paths
except Exception:          # pragma: no cover
    folder_paths = None


def resolve_media_path(annotated: str) -> str:
    """Turns the JS panel's annotated path into a real filesystem path.

    The panel stores what ComfyUI's /view endpoint wants -- "subfolder/name.png
    [input]" -- where the bracketed suffix names WHICH root directory the file
    lives under (input/output/temp) rather than being part of the filename. That
    string is correct for building a view URL, but passing it straight to open()
    fails with a confusing "No such file or directory: '... [input]'" (which is
    exactly what happened in practice). ComfyUI's own get_annotated_filepath
    understands this format, so prefer it; fall back to splitting the suffix and
    joining against the right root when it's unavailable (bare tests), and last
    of all treat the string as a literal path so already-plain paths still work.
    """
    if not annotated:
        return annotated
    text = str(annotated)

    if folder_paths is not None:
        getter = getattr(folder_paths, "get_annotated_filepath", None)
        if callable(getter):
            try:
                resolved = getter(text)
                if resolved and os.path.exists(resolved):
                    return resolved
            except Exception:
                pass

    name, kind = text, "input"
    m = re.match(r"^(.*)\s\[(input|output|temp)\]$", text)
    if m:
        name, kind = m.group(1), m.group(2)

    base = None
    if folder_paths is not None:
        fn = getattr(folder_paths, f"get_{kind}_directory", None)
        if callable(fn):
            try:
                base = fn()
            except Exception:
                base = None
    if base:
        candidate = os.path.join(base, name)
        if os.path.exists(candidate):
            return candidate
        return candidate
    return name


@dataclass
class ShotMediaContext:
    """Per-shot media that isn't captured by the CategorizationPlan's own
    shapes (which are subject/anchor/audio-centric, not shot-centric) --
    carried alongside the plan so the composer can feed each shot's own
    reference images to the narrow VLM and apply anchor phrasing at the
    right shot index."""
    index: int                      # 1-based shot number
    reference_image_tags: List[str] = field(default_factory=list)  # "Picture N" tags
    is_replacement: bool = False
    continues_from: Optional[str] = None   # "<Video N>" this shot resumes after
    # Comic mode fills these in: the shot's own dialogue, a text brief, and a
    # duration derived from how long that dialogue takes to speak. Empty for
    # ordinary shots, which take those from the node's shot boxes instead.
    dialogue: List[dict] = field(default_factory=list)
    sounds: List[str] = field(default_factory=list)
    subject_slots: List[int] = field(default_factory=list)
    brief_text: str = ""
    # True when the person wrote the description themselves, in which case the
    # VLM must format it rather than reinterpret it.
    user_authored: bool = False
    duration: Optional[float] = None    # "complete" video was used -- guide's own
                                     # video-editing/character-swap treatment applies


def _count_media(state: dict) -> Tuple[int, int, int]:
    """Returns (pictures, videos, audio) counts across every category, so the
    caller can enforce the shared 9/3/3 limit before spending time loading
    tensors for media that would exceed it anyway."""
    pics = vids = auds = 0

    def bump(item: Optional[dict], kind_hint: Optional[str] = None):
        nonlocal pics, vids, auds
        if not item:
            return
        kind = item.get("kind") or kind_hint
        if kind == "picture":
            pics += 1
        elif kind == "video":
            vids += 1
        elif kind == "audio":
            auds += 1

    bump(state.get("storyboard"), "picture")
    # Replace mode's source clip is a real video reference (plus its audio), but
    # it lives outside state["shots"], so the shot loop below never sees it.
    _rep = state.get("replace") or {}
    if _rep.get("enabled") and (_rep.get("video") or {}).get("file"):
        vids += 1
        if _rep.get("keep_audio", True):
            auds += 1
        # Each per-person voice reference costs an audio slot too.
        for _pp in (_rep.get("people") or []):
            if (_pp.get("voice") or {}).get("file"):
                auds += 1
    # Comic mode: the active chunk's panels are the images. The page itself isn't
    # a reference -- it's only the source the panels are cropped out of.
    comic_state = state.get("comic") or {}
    if comic_state.get("enabled"):
        chunks = comic_state.get("chunks") or []
        idx = int(comic_state.get("chunk_index") or 0)
        if 0 <= idx < len(chunks):
            pics += len(chunks[idx])
        else:
            pics += len(comic_state.get("panels") or [])
    for s in state.get("subjects") or []:
        bump(s.get("complete"))
        for r in s.get("visual_refs") or []:
            bump(r, "picture")
        for r in s.get("outfit_refs") or []:
            bump(r, "picture")
        bump(s.get("vocal_ref"), "audio")

    for sh in state.get("shots") or []:
        bump(sh.get("complete"), "video")
        if sh.get("complete") and (sh["complete"] or {}).get("keep_audio", True):
            auds += 1
        bump(sh.get("continue_from"), "video")
        if (sh.get("continue_from") or {}).get("continue_audio"):
            auds += 1
        bump(sh.get("first_frame"), "picture")
        bump(sh.get("last_frame"), "picture")
        bump(sh.get("motion"), "video")
        bump(sh.get("music"), "audio")
        bump(sh.get("sfx"), "audio")
        for r in sh.get("reference_images") or []:
            bump(r, "picture")

    audio_section = state.get("audio") or {}
    for r in audio_section.get("dialogue") or []:
        bump(r, "audio")
    for r in audio_section.get("music") or []:
        bump(r, "audio")

    return pics, vids, auds


DEFAULT_DURATION = 10.0
H3_MAX_DURATION = 15.0      # H3 generates 5-15s clips
H3_MIN_DURATION = 5.0


def resolve_duration(state: dict) -> float:
    """The duration this loader hands to the prompt builder.

    Normally the manual value set on the loader. In comic mode the figure is
    computed from the panels in the active chunk (estimated speech time), which
    overrides the manual value -- a comic's pacing is dictated by how long its
    dialogue actually takes to say, not by a number typed in advance.
    """
    comic = state.get("comic") or {}
    if comic.get("enabled"):
        d = comic_chunk_duration(comic)
        if d:
            return max(H3_MIN_DURATION, min(H3_MAX_DURATION, d))
        try:
            d = float(comic.get("computed_duration") or 0)
            if d > 0:
                return max(H3_MIN_DURATION, min(H3_MAX_DURATION, d))
        except (TypeError, ValueError):
            pass
    try:
        d = float(state.get("duration_seconds") or DEFAULT_DURATION)
    except (TypeError, ValueError):
        d = DEFAULT_DURATION
    return max(H3_MIN_DURATION, min(H3_MAX_DURATION, d))


def _comic_panels(comic_cfg: dict):
    """Panel objects from the stored config -- no image decoding, so this is
    safe to call from duration/cap paths that must not touch the disk."""
    from . import comic as cmc
    out = []
    for i, p in enumerate(comic_cfg.get("panels") or [], start=1):
        pn = cmc.Panel(index=i, x=int(p.get("x", 0)), y=int(p.get("y", 0)),
                        w=int(p.get("w", 1)), h=int(p.get("h", 1)))
        pn.bubbles = [cmc.Bubble(x=0, y=0, w=0, h=0,
                                  text=str(b.get("text") or ""),
                                  speaker_subject=b.get("speaker_subject"))
                       for b in (p.get("bubbles") or [])]
        out.append(pn)
    return out


def comic_chunks(comic_cfg: dict):
    from . import comic as cmc
    return cmc.chunk_panels(
        _comic_panels(comic_cfg),
        split_on_aspect=not bool(comic_cfg.get("merge_aspect")))


def comic_chunk_duration(comic_cfg: dict) -> float:
    """Seconds for the active chunk, from its dialogue. 0 when there's nothing
    to measure yet."""
    from . import comic as cmc
    chunks = comic_chunks(comic_cfg)
    if not chunks:
        return 0.0
    idx = int(comic_cfg.get("chunk_index") or 0)
    idx = max(0, min(idx, len(chunks) - 1))
    return cmc.chunk_duration(chunks[idx])


def resolve_aspect(state: dict, media_io_module=None) -> Tuple[Optional[float], str]:
    """The aspect ratio to generate at, and where it came from.

    Precedence: any mode with a real source frame dictates the ratio, because
    generating at a different shape would letterbox or crop the very thing being
    matched. Only when there's no such source does the node's own setting apply.
    """
    from . import resolution as rz
    import os

    def _img_ratio(item):
        if not item or not item.get("file"):
            return None
        try:
            from PIL import Image
            with Image.open(resolve_media_path(item["file"])) as im:
                return rz.ratio_of(*im.size)
        except Exception:
            return None

    def _vid_ratio(item):
        if not item or not item.get("file"):
            return None
        try:
            import cv2
            cap = cv2.VideoCapture(resolve_media_path(item["file"]))
            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            cap.release()
            return rz.ratio_of(w, h)
        except Exception:
            return None

    rep = state.get("replace") or {}
    if rep.get("enabled"):
        r = _vid_ratio(rep.get("video"))
        if r:
            return r, "the source video being edited"

    comic = state.get("comic") or {}
    if comic.get("enabled"):
        panels = comic.get("panels") or []
        chunks = comic.get("chunks") or []
        idx = int(comic.get("chunk_index") or 0)
        active = chunks[idx] if 0 <= idx < len(chunks) else list(range(1, len(panels) + 1))
        rs = [rz.ratio_of(panels[i - 1].get("w"), panels[i - 1].get("h"))
              for i in active if 0 < i <= len(panels)]
        rs = [x for x in rs if x]
        if rs:
            return sum(rs) / len(rs), "the comic panels in this chunk"

    for sh in state.get("shots") or []:
        for key, label in (("first_frame", "the first-frame image"),
                            ("last_frame", "the last-frame image")):
            r = _img_ratio(sh.get(key))
            if r:
                return r, label
        r = _vid_ratio(sh.get("continue_from"))
        if r:
            return r, "the clip being continued from"

    r = rz.parse_ratio(state.get("aspect_ratio") or "16:9")
    return (r or 1.777), "the loader's aspect setting"


def resolve_dimensions(state: dict) -> Tuple[int, int, str]:
    """Width, height and a one-line explanation of where the shape came from."""
    from . import resolution as rz
    ratio, source = resolve_aspect(state)
    try:
        mp = float(state.get("megapixels") or 2.0)
    except (TypeError, ValueError):
        mp = 2.0
    w, h = rz.dimensions(mp, ratio)
    return w, h, f"{w}x{h} ({rz.describe(ratio)}, {w * h / 1e6:.2f} MP) from {source}"


def validate_modes(state: dict) -> Optional[str]:
    """Comic and Replace both replace the Shot tab, so only one can be active.
    Guarded here as well as in the UI because a workflow saved before the check
    existed can still have both set."""
    if (state.get("comic") or {}).get("enabled") and \
            (state.get("replace") or {}).get("enabled"):
        return ("Comic mode and Replace mode are both on. They each build the shots, "
                "so turn one off in the loader.")
    return None


def validate_caps(state: dict) -> Optional[str]:
    """Returns an error string if the shared 9/3/3 caps are exceeded, else
    None. Checked before loading anything -- matches the native node's own
    hard slot limits, so this is a real constraint, not a style preference."""
    pics, vids, auds = _count_media(state)
    if pics > MAX_PICTURES:
        return f"{pics} images loaded across Subject/Shot/Audio; H3 accepts {MAX_PICTURES}."
    if vids > MAX_VIDEOS:
        return f"{vids} videos loaded across Subject/Shot/Audio; H3 accepts {MAX_VIDEOS}."
    if auds > MAX_AUDIO:
        return f"{auds} audio files loaded across Subject/Shot/Audio; H3 accepts {MAX_AUDIO}."
    return None


def _resize_image_tensor(t, max_side):
    """Downscales an IMAGE tensor [N,H,W,3] so its longest side is at most
    `max_side`. This governs what the actual H3 generation receives (via the
    Reference Splitter), which is separate from the much smaller copies the
    description VLM gets -- useful for VRAM headroom and for matching H3's own
    expected input sizes. `max_side` of 0/None means leave the media alone."""
    if not max_side or t is None:
        return t
    try:
        import torch.nn.functional as F
        h, w = int(t.shape[1]), int(t.shape[2])
        longest = max(h, w)
        if longest <= max_side:
            return t
        scale = max_side / float(longest)
        nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        # [N,H,W,C] -> [N,C,H,W] for interpolate, then back.
        chw = t.permute(0, 3, 1, 2)
        out = F.interpolate(chw, size=(nh, nw), mode="bilinear", align_corners=False)
        return out.permute(0, 2, 3, 1).contiguous().clamp(0.0, 1.0)
    except Exception as exc:      # never let a resize failure lose the media
        dbg.line(f"resize skipped ({exc})")
        return t


def _trim_audio(audio: dict, trim) -> dict:
    """Slices an AUDIO dict {waveform,[B,C,S]; sample_rate} to a trim window.
    Matches the reference loader's model: trim = {start, end} in seconds, either
    side optional."""
    if not audio or not trim:
        return audio
    try:
        start = float(trim.get("start") or 0.0)
        end = trim.get("end")
        end = float(end) if end not in (None, "") else None
        if start <= 0 and end is None:
            return audio
        sr = int(audio["sample_rate"])
        wav = audio["waveform"]
        total = wav.shape[-1]
        a = max(0, min(total, int(start * sr)))
        b = total if end is None else max(a, min(total, int(end * sr)))
        if b <= a:
            return audio
        return {"waveform": wav[..., a:b].contiguous(), "sample_rate": sr}
    except Exception as exc:
        dbg.line(f"audio trim skipped ({exc})")
        return audio


def _trim_video_frames(t, trim, duration):
    """Keeps only the sampled frames that fall inside the trim window. Frames are
    sampled evenly across the clip, so a window maps cleanly onto an index range
    without re-decoding. Falls back to the untrimmed tensor if the window would
    leave nothing."""
    if t is None or not trim or not duration:
        return t
    try:
        start = float(trim.get("start") or 0.0)
        end = trim.get("end")
        end = float(end) if end not in (None, "") else float(duration)
        if start <= 0 and end >= duration:
            return t
        n = int(t.shape[0])
        a = max(0, min(n - 1, int(round(start / duration * n))))
        b = max(a + 1, min(n, int(round(end / duration * n))))
        return t[a:b].contiguous()
    except Exception as exc:
        dbg.line(f"video trim skipped ({exc})")
        return t


class _TagAllocator:
    """Assigns Picture N / Video N / Audio N tags in a stable order (subjects
    first, then shots, then the global audio section) and loads each item's
    tensor exactly once via media_io, matching the same numbering discipline
    categorize.py's own tag allocator uses for the flat-items model."""

    def __init__(self, media_io_module, max_side=0):
        self._m = media_io_module
        self._max_side = max_side
        self.p = self.v = self.a = 0
        self.tensor_index: Dict[str, object] = {}
        self.item_by_tag: Dict[str, dict] = {}

    def _side_for(self, item) -> int:
        """Each file carries its own resize setting; the loader-wide value is only
        a fallback for files that never had one set. A character sheet left on
        "original" therefore stays full-resolution even when a default cap is on."""
        raw = (item or {}).get("max_side")
        if raw in (None, ""):
            return self._max_side
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return self._max_side

    def register_picture(self, tensor, label: str) -> str:
        """Registers an already-decoded image (a cleaned comic panel) under the
        next Picture tag. Comic panels are cropped and inpainted in memory, so
        there is no file on disk for picture() to load."""
        self.p += 1
        tag = f"Picture {self.p}"
        self.tensor_index[tag] = _resize_image_tensor(tensor, self._max_side)
        self.item_by_tag[tag] = {"name": label}
        dbg.bullet(f"{tag}  <-  {label}  (comic panel, bubbles inpainted)")
        return tag

    def picture(self, item: Optional[dict]) -> Optional[str]:
        if not item or not item.get("file"):
            return None
        self.p += 1
        tag = f"Picture {self.p}"
        real = resolve_media_path(item["file"])
        self.tensor_index[tag] = _resize_image_tensor(
            self._m.load_image(real), self._side_for(item))
        self.item_by_tag[tag] = item
        dbg.bullet(f"{tag}  <-  {item.get('name') or item.get('file')}   [{real}]")
        return tag

    def video(self, item: Optional[dict]) -> Optional[str]:
        if not item or not item.get("file"):
            return None
        self.v += 1
        tag = f"Video {self.v}"
        real = resolve_media_path(item["file"])
        _frames = self._m.load_video_frames(real)
        _dur = (item or {}).get("duration")
        if not _dur:
            try:
                _dur = self._m.video_duration(real)
            except Exception:
                _dur = None
        _frames = _trim_video_frames(_frames, (item or {}).get("trim"), _dur)
        self.tensor_index[tag] = _resize_image_tensor(_frames, self._side_for(item))
        self.item_by_tag[tag] = item
        dbg.bullet(f"{tag}  <-  {item.get('name') or item.get('file')}   [{real}]")
        return tag

    def audio(self, item: Optional[dict], from_video: bool = False) -> Optional[str]:
        if not item or not item.get("file"):
            return None
        self.a += 1
        tag = f"Audio {self.a}"
        real = resolve_media_path(item["file"])
        if from_video or item.get("kind") == "video":
            self.tensor_index[tag] = _trim_audio(
                self._m.extract_audio(real), (item or {}).get("trim"))
            src = "extracted from video track"
        else:
            self.tensor_index[tag] = _trim_audio(
                self._m.load_audio(real), (item or {}).get("trim"))
            src = "standalone file"
        self.item_by_tag[tag] = item
        dbg.bullet(f"{tag}  <-  {item.get('name') or item.get('file')}   ({src})")
        return tag


def _comic_shot_contexts(comic_cfg: dict, alloc, notes: List[tuple],
                          subjects: List, _next_slot, audio: List):
    """Turns the active chunk's panels into ShotMediaContexts.

    Each panel is cropped from the page, its speech balloons inpainted away
    (so the lettering can't be reproduced in the video), and registered as a
    reference image for its own shot. The dialogue that was IN those balloons
    is carried as structured dialogue instead, which is how it ends up properly
    <d>-tagged rather than as text baked into the picture.
    """
    from . import comic as cmc
    from PIL import Image
    import numpy as _np

    page_path = resolve_media_path(comic_cfg["page"]["file"])
    panels_raw = comic_cfg.get("panels") or []
    if not panels_raw:
        return []

    panels = []
    for i, p in enumerate(panels_raw, start=1):
        pn = cmc.Panel(index=i, x=int(p.get("x", 0)), y=int(p.get("y", 0)),
                        w=int(p.get("w", 0)), h=int(p.get("h", 0)))
        pn.bubbles = []
        for b in (p.get("bubbles") or []):
            bub = cmc.Bubble(x=int(b.get("x", 0)), y=int(b.get("y", 0)),
                              w=int(b.get("w", 0)), h=int(b.get("h", 0)),
                              text=str(b.get("text") or ""),
                              speaker_subject=b.get("speaker_subject"))
            bub.is_sound = bool(b.get("is_sound"))
            pn.bubbles.append(bub)
        panels.append(pn)

    # Comic mode defines its own cast: a subject list lives in the Comic tab
    # rather than the Subject tab, because who appears in which PANEL is part of
    # the page's structure and can't be expressed by a global subject row.
    comic_subjects = comic_cfg.get("subjects") or []
    subject_slot_of = {}
    for ci, cs in enumerate(comic_subjects):
        ref = cs.get("ref") or {}
        slot = _next_slot()
        sp = SubjectPlan(slot=slot, kind=cs.get("kind") or "person",
                          retention="attribute_transfer")
        if ref.get("file"):
            r_tag = alloc.picture(ref)
            if r_tag:
                sp.sources.append(r_tag)
        if cs.get("name"):
            notes.append((f"Subject {slot}",
                          f"Subject {slot} is {cs['name']}. Use this name's described "
                          f"appearance consistently in every panel they appear in."))
        subjects.append(sp)
        subject_slot_of[ci + 1] = slot          # the tab numbers them from 1
        voice = cs.get("voice")
        if voice and voice.get("file"):
            v_tag = alloc.audio(voice)
            if v_tag:
                audio.append(AudioPlan(tag=v_tag, role="voice_timbre",
                                        subject_slot=slot, retention="reference"))

    chunks = cmc.chunk_panels(
        panels, split_on_aspect=not bool(comic_cfg.get("merge_aspect")))
    idx = int(comic_cfg.get("chunk_index") or 0)
    idx = max(0, min(idx, len(chunks) - 1)) if chunks else 0
    chunk = chunks[idx] if chunks else []
    dbg.kv("comic chunk", f"{idx + 1} of {len(chunks)} "
                           f"(panels {[p.index for p in chunk]})")

    import cv2
    page = cv2.imread(page_path)
    if page is None:
        dbg.line(f"comic page could not be read: {page_path}")
        return []

    out = []
    for shot_no, pn in enumerate(chunk, start=1):
        crop = page[pn.y:pn.y + pn.h, pn.x:pn.x + pn.w]
        if crop.size == 0:
            continue
        # Keeping the balloons is the default. H3 animates comic pages natively,
        # lettering included -- published workflows literally instruct it to
        # "time the speech bubbles" -- whereas inpainting them out smears the art
        # badly over large regions. Removal stays available for pages where the
        # lettering genuinely has to go.
        bubble_mode = (comic_cfg.get("bubble_mode") or "keep").lower()
        cleaned = cmc.clean_panel(crop, pn.bubbles) if bubble_mode == "remove" else crop
        rgb = cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)
        tensor = _to_image_tensor(rgb)
        tag = alloc.register_picture(tensor, f"panel {pn.index}")

        # Register the panel as a Subject too. Without this the artwork reached
        # the VLM but appeared in no section of the prompt, so subject_definitions
        # and retention_analysis came out empty and H3 had nothing to anchor to.
        panel_sp = SubjectPlan(slot=_next_slot(), kind="environment", sources=[tag],
                                retention="partially_preserved")
        panel_sp.ref_for_shot = shot_no
        panel_sp.is_comic_panel = True
        panel_sp.keep_bubbles = (bubble_mode != "remove")
        subjects.append(panel_sp)

        ctx = ShotMediaContext(index=shot_no)
        ctx.reference_image_tags.append(tag)
        ctx.duration = cmc.panel_duration(pn)
        ctx.dialogue = [
            {"speaker_subject": (f"Subject {subject_slot_of.get(int(b.speaker_subject), b.speaker_subject)}"
                                  if b.speaker_subject else None),
             "speaker_label": None,
             "text": b.text.strip(),
             "language": "English",
             "voiceover": False}
            for b in pn.bubbles
            if (b.text or "").strip() and not getattr(b, "is_sound", False)
        ]
        # Sound-effect balloons ("cough", "CRASH") are sounds, not dialogue --
        # putting them in <d> would have a character speak the word aloud.
        ctx.sounds = [b.text.strip() for b in pn.bubbles
                      if (b.text or "").strip() and getattr(b, "is_sound", False)]
        if ctx.sounds:
            notes.append((tag, f"Panel {pn.index} non-speech sounds: "
                                f"{', '.join(ctx.sounds)}. These are sound effects, "
                                f"not spoken lines -- describe the action that makes "
                                f"each sound; never put them in dialogue."))
        # A description the person wrote wins outright: the auto tagger and the
        # duration estimate are BACKUPS for when they haven't written one.
        raw = (comic_cfg.get("panels") or [])[pn.index - 1] \
            if pn.index - 1 < len(comic_cfg.get("panels") or []) else {}
        # Marks carry WHERE each character stands in the panel, which is what
        # lets the shot say "on the left" instead of leaving the model to place
        # them. Plain presence numbers are still accepted.
        marks = raw.get("marks") or []
        present, placed = [], []
        for mk in marks:
            n = mk.get("n")
            if n not in subject_slot_of:
                continue
            slot = subject_slot_of[n]
            present.append(slot)
            fx = float(mk.get("fx") or 0.5)
            fy = float(mk.get("fy") or 0.5)
            where = ("on the left" if fx < 0.38
                      else "on the right" if fx > 0.62 else "in the centre")
            depth = ("in the foreground" if fy > 0.66
                      else "in the background" if fy < 0.34 else "")
            placed.append(f"Subject {slot} {where}"
                           + (f", {depth}" if depth else ""))
        for n in (raw.get("present") or []):
            if n in subject_slot_of and subject_slot_of[n] not in present:
                present.append(subject_slot_of[n])
        if present:
            ctx.subject_slots = present
            detail = ("; ".join(placed) if placed
                       else ", ".join(f"Subject {x}" for x in present))
            notes.append((tag, f"Panel {pn.index} contains exactly these characters, "
                                f"in these positions: {detail}. No one else appears "
                                f"in this shot, and each stays where stated."))
        ctx.brief_text = (raw.get("description") or "").strip()
        ctx.user_authored = bool(ctx.brief_text)
        try:
            manual = float(raw.get("duration") or 0)
        except (TypeError, ValueError):
            manual = 0.0
        if manual > 0:
            ctx.duration = manual
        # Tag the CLEANED panel and hand the tags to the VLM as the authoritative
        # content. A booru tagger reads comic art far more reliably than a small
        # general VLM, so the VLM's job shrinks to turning a list of facts into a
        # sentence rather than perceiving the panel itself.
        _written = ((comic_cfg.get("panels") or [])[pn.index - 1].get("description")
                    if pn.index - 1 < len(comic_cfg.get("panels") or []) else "") or ""
        _tags = []
        if not _written.strip():
            # Tagging is the fallback for an undescribed panel only.
            try:
                from . import tagger as _tg
                from PIL import Image as _Im
                _tags = _tg.tag_image(_Im.fromarray(rgb))
            except Exception as _exc:
                dbg.line(f"panel tagging skipped ({_exc})")
        if _tags:
            dbg.bullet(f"panel {pn.index} tags: {', '.join(_tags[:12])}")
            notes.append((tag, f"Panel {pn.index} content tags (authoritative -- these "
                                f"come from an image tagger, trust them over your own "
                                f"reading of the artwork): {', '.join(_tags)}. Write "
                                f"[Shot {shot_no}] from these tags, as live action."))

        if bubble_mode == "remove":
            notes.append((tag, f"Comic panel {pn.index}, already cleaned of speech "
                                f"balloons and lettering. Describe the action and "
                                f"staging it depicts as LIVE MOTION for "
                                f"[Shot {shot_no}] -- no speech bubbles, captions, "
                                f"lettering or panel borders appear in the video. "
                                f"Any dialogue is spoken aloud, not shown as text."))
        else:
            notes.append((tag, f"Comic panel {pn.index}, with its speech balloons "
                                f"intact. Animate the panel as drawn: the artwork "
                                f"comes alive while each balloon stays part of the "
                                f"frame, appearing as its line is spoken and clearing "
                                f"once the line finishes. Keep the lettering legible "
                                f"and unchanged -- do not redraw or translate it."))
        out.append(ctx)

    # Comics cut between frozen moments; animation has to move through them.
    for a, b in zip(out, out[1:]):
        notes.append((f"Shot {a.index}->{b.index}",
                      f"[Shot {a.index}] and [Shot {b.index}] are consecutive comic "
                      f"panels. Carry the action continuously from the first into the "
                      f"second, inventing the in-between movement the comic skips."))
    return out


def _to_image_tensor(rgb_array):
    """HxWx3 uint8 RGB -> the [1,H,W,3] float tensor the rest of the pack uses."""
    import torch
    import numpy as _np
    arr = _np.asarray(rgb_array).astype("float32") / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def build_plan_from_structured(state: dict, media_io_module) -> Tuple[
        CategorizationPlan, Dict[str, object], List[ShotMediaContext]]:
    """The structured-loader equivalent of categorize.build_plan(): takes the
    loader's JSON state, loads every referenced file's tensor exactly once,
    and returns (plan, tensor_index, shot_contexts). Reuses the exact same
    CategorizationPlan shape the flat-items model produces, so none of
    categorized_nodes.py's rendering logic needs to know which loader a
    given prompt came from."""
    dbg.section("LOADER STATE -> TAGS")
    dbg.kv("subject rows", len(state.get("subjects") or []))
    dbg.kv("shot rows", len(state.get("shots") or []))
    _aud = state.get("audio") or {}
    dbg.kv("global audio", f"{len(_aud.get('dialogue') or [])} dialogue, "
                            f"{len(_aud.get('music') or [])} music")
    _p, _v, _a = _count_media(state)
    dbg.kv("media totals", f"{_p}/{MAX_PICTURES} images, {_v}/{MAX_VIDEOS} videos, "
                            f"{_a}/{MAX_AUDIO} audio")

    _max_side = state.get("max_media_side") or 0
    try:
        _max_side = int(_max_side)
    except (TypeError, ValueError):
        _max_side = 0
    if _max_side:
        dbg.kv("resize", f"longest side capped at {_max_side}px")
    alloc = _TagAllocator(media_io_module, max_side=_max_side)
    subjects: List[SubjectPlan] = []

    _outfit_rows: List[tuple] = []   # (picture_tag, scope_label, wearer_slot)

    def _outfit_scope(item: dict, fallback_shot: int) -> str:
        """Which shot(s) an outfit applies to. Read from the item's own `scope`
        field ("all" or a shot number) so any outfit can target any shot -- the
        old index-derived scheme forced outfit N to mean shot N, so you couldn't
        dress Shot 2 without also adding a Shot 1 outfit."""
        scope = item.get("scope")
        if scope in (None, "", "auto"):
            return "all shots" if fallback_shot <= 1 else f"Shot {fallback_shot}"
        if str(scope).lower() in ("all", "all shots"):
            return "all shots"
        m = re.search(r"\d+", str(scope))
        return f"Shot {m.group(0)}" if m else "all shots"

    def _next_slot() -> int:
        # Subject numbering continues past the Subject rows so a shot's scene
        # reference gets its own <Subject N> rather than colliding with a person.
        return max((sp.slot for sp in subjects), default=0) + 1

    anchors: List[AnchorPlan] = []
    video_structures: List[VideoStructurePlan] = []
    audio: List[AudioPlan] = []
    donor_pairs: List[tuple] = []
    soundscape_citations: List[tuple] = []
    music_citations: List[tuple] = []
    notes: List[tuple] = []
    shot_contexts: List[ShotMediaContext] = []

    # -- Subjects -------------------------------------------------------
    for i, s in enumerate(state.get("subjects") or [], start=1):
        sp = SubjectPlan(slot=i)
        complete = s.get("complete")

        if complete and complete.get("kind") == "picture":
            # Complete = image: identity/appearance/outfit all from one photo.
            # Per spec, only a Vocal Reference may still be added alongside it.
            tag = alloc.picture(complete)
            if tag:
                sp.sources.append(tag)
            vocal = s.get("vocal_ref")
            if vocal and vocal.get("file"):
                a_tag = alloc.audio(vocal)
                if a_tag:
                    audio.append(AudioPlan(tag=a_tag, role="voice_timbre",
                                            subject_slot=i, retention="reference"))

        elif complete and complete.get("kind") == "video":
            # Complete = video: appearance, outfit, AND the subject's own natural
            # voice all come from one clip -- no separate Vocal Reference needed
            # or allowed (their real voice is already in this video's own audio,
            # used as a voice-timbre reference for their own dialogue elsewhere,
            # not assumed to be footage that gets directly edited/reused).
            tag = alloc.video(complete)
            if tag:
                sp.sources.append(tag)
                sp.motion_source = tag
            if complete.get("has_audio", True):
                a_tag = alloc.audio(complete, from_video=True)
                if a_tag:
                    audio.append(AudioPlan(tag=a_tag, role="voice_timbre",
                                            subject_slot=i, retention="reference",
                                            from_video_tag=tag))
            # "Outfit Shot 2 and onward" -- additional outfit refs for shots where
            # the outfit changes from what the Complete video itself shows.
            for j, outfit in enumerate(s.get("outfit_refs") or [], start=2):
                o_tag = alloc.picture(outfit)
                if o_tag:
                    label = _outfit_scope(outfit, j)
                    sp.outfit_refs.append((o_tag, label))
                    _outfit_rows.append((o_tag, label, i))
                    notes.append((f"Subject {i}", f"Subject {i} has a separate outfit "
                                   f"reference. Describe ONLY face, hair, build and age "
                                   f"-- do NOT mention clothing, colors of garments, or "
                                   f"accessories for this subject at all."))
                    notes.append((o_tag, f"Outfit reference for Subject {i} in [Shot {j}] "
                                          f"specifically -- the outfit changes from what "
                                          f"Complete (Video) shows there. Describe the "
                                          f"clothing itself, and mention it in that shot."))

        else:
            # No Complete asset -- the granular path: Visual/Outfit/Vocal are all
            # independent and freely combinable.
            for r in s.get("visual_refs") or []:
                tag = alloc.picture(r)
                if tag:
                    sp.sources.append(tag)
            for j, outfit in enumerate(s.get("outfit_refs") or []):
                o_tag = alloc.picture(outfit)
                if not o_tag:
                    continue
                label = _outfit_scope(outfit, j + 1)
                sp.outfit_refs.append((o_tag, label))
                _outfit_rows.append((o_tag, label, i))
                notes.append((f"Subject {i}", f"Subject {i} has a separate outfit "
                               f"reference. Describe ONLY face, hair, build and age -- do "
                               f"NOT mention clothing, colors of garments, or accessories "
                               f"for this subject at all."))
                notes.append((o_tag, f"Outfit reference for Subject {i}, applying to "
                                      f"{label}. Describe the clothing itself; it must be "
                                      f"worn by that subject where the label says."))
            vocal = s.get("vocal_ref")
            if vocal and vocal.get("file"):
                a_tag = alloc.audio(vocal)
                if a_tag:
                    audio.append(AudioPlan(tag=a_tag, role="voice_timbre",
                                            subject_slot=i, retention="reference"))

        if sp.sources:
            subjects.append(sp)

    for _o_tag, _o_label, _wearer in _outfit_rows:
        _osp = SubjectPlan(slot=_next_slot(), kind="clothing", sources=[_o_tag],
                            retention="attribute_transfer")
        _osp.outfit_for_subject = _wearer
        _osp.outfit_scope = _o_label
        subjects.append(_osp)

    # -- Replace mode: a source video with an explicit person mapping -------
    # Its own tab rather than a Shot field, because a character-only edit is a
    # different job from composing shots: the shot structure comes entirely from
    # the source, and the only real decision is who replaces whom.
    rep = state.get("replace") or {}
    if rep.get("enabled") and (rep.get("video") or {}).get("file"):
        state = dict(state)
        state["shots"] = [{
            "complete": dict(rep["video"], keep_audio=rep.get("keep_audio", True)),
        }]
        # Only markers with a Subject assigned mean anything; an unassigned one is
        # a click the user hasn't finished, not a replacement.
        rep_keep = rep.get("keep") or {}
        _replace_map = []
        for _p in (rep.get("people") or []):
            if not _p.get("subject"):
                continue
            _where = str(_p.get("where") or _p.get("label") or "").strip()
            if not _where and _p.get("fx") is not None:
                # Fall back to the marker's own position if it was never described.
                _fx = float(_p["fx"])
                _where = ("the person on the left" if _fx < 0.38 else
                          "the person on the right" if _fx > 0.62 else
                          "the person in the centre")
            _replace_map.append((int(_p["subject"]), _where,
                                  _p.get("keep") or rep_keep or {}))
            # Per-person voice reference: the replacing subject can speak in a
            # timbre of its own rather than everyone sharing one global setting.
            _voice = _p.get("voice")
            if (not (_voice or {}).get("file")) and _p.get("use_subject_voice"):
                # Reuse the Vocal Reference already set on that Subject rather
                # than making the person load the same file a second time.
                _sub = (state.get("subjects") or [])
                _idx = int(_p["subject"]) - 1
                if 0 <= _idx < len(_sub):
                    _voice = (_sub[_idx] or {}).get("vocal_ref")
            if _voice and _voice.get("file"):
                _v_tag = alloc.audio(_voice)
                if _v_tag:
                    audio.append(AudioPlan(tag=_v_tag, role="voice_timbre",
                                            subject_slot=int(_p["subject"]),
                                            retention="reference"))
    else:
        _replace_map = []

    # -- Comic mode: panels become shots ----------------------------------
    # Replaces the Shots tab entirely. Each panel of the ACTIVE CHUNK becomes one
    # shot, carrying its own cleaned artwork, its dialogue in reading order, and
    # a duration derived from how long that dialogue takes to say. A page's
    # panels routinely exceed H3's 15s/9-image envelope, which is why only one
    # chunk is emitted per run.
    comic_cfg = state.get("comic") or {}
    if comic_cfg.get("enabled") and comic_cfg.get("page", {}).get("file"):
        for ctx in _comic_shot_contexts(comic_cfg, alloc, notes,
                                          subjects, _next_slot, audio):
            shot_contexts.append(ctx)
        state = dict(state)
        state["shots"] = []      # the Shots tab is inert while comic mode is on

    for i, sh in enumerate(state.get("shots") or [], start=1):
        ctx = ShotMediaContext(index=i)
        complete = sh.get("complete")

        if complete and complete.get("file"):
            # Replacement-type: this shot is a straight video edit, everything
            # else about the shot is dropped (guide's own video-editing model --
            # same treatment as the composer's existing character-swap feature).
            tag = alloc.video(complete)
            if tag:
                # Work out the mapping FIRST: it decides whether this is a
                # character swap, and that has to be known before the structure
                # plan is built rather than patched on afterwards.
                replace_map = complete.get("replace_map") or []
                mapped_any = False
                for m in replace_map:
                    slot = m.get("subject")
                    if not slot:
                        continue
                    sp_t = subject(int(slot))
                    sp_t.replaces_original = True
                    sp_t.swap_video = tag
                    sp_t.replaces_person = {
                        "position": m.get("position") or "",
                        "description": m.get("description") or "",
                    }
                    mapped_any = True
                # No explicit mapping and exactly one subject: unambiguous, so
                # treat it as that subject's swap. With several subjects and no
                # mapping there is nothing to infer, and guessing is what made
                # replacement silently do the wrong thing.
                if not mapped_any and len(subjects) == 1:
                    subjects[0].replaces_original = True
                    subjects[0].swap_video = tag
                video_structures.append(VideoStructurePlan(
                    tag=tag, role="editing", retention="fully_preserved",
                    is_character_swap=mapped_any))
                ctx.is_replacement = True
                # Guide S2.3: "When editing a source video, use audio reuse as well
                # if its original audio remains audible." Kept as dialogue so the
                # spoken words are transcribed and survive verbatim.
                if complete.get("keep_audio", True):
                    ca_tag = alloc.audio(complete, from_video=True)
                    if ca_tag:
                        audio.append(AudioPlan(tag=ca_tag, role="dialogue",
                                                retention="fully_copy",
                                                from_video_tag=tag))
                        notes.append((ca_tag, "Original audio of the video being "
                                               "edited. Only the performer's appearance "
                                               "changes, so it carries over unchanged."))
            shot_contexts.append(ctx)
            continue

        # Video continuation -- shot 1 only. The target video resumes from the END
        # of this clip (guide S2.3: "New content continues, extends, resumes, or
        # transitions from an existing source video"), so it's only meaningful on
        # the opening shot. Distinct from Complete/replacement, which edits the
        # source in place rather than carrying on after it.
        cont = sh.get("continue_from")
        if cont and cont.get("file") and i == 1:
            c_tag = alloc.video(cont)
            if c_tag:
                video_structures.append(VideoStructurePlan(
                    tag=c_tag, role="continuation", retention="partially_preserved"))
                ctx.continues_from = c_tag
                notes.append((c_tag, "The target video CONTINUES from the end of this "
                                      "clip. Describe the state it finishes in -- final "
                                      "positions, framing, lighting, action in progress -- "
                                      "so [Shot 1] can resume from exactly there. Do not "
                                      "re-narrate the clip itself."))
                if cont.get("continue_audio"):
                    # Guide S2.3: continuing a source video without copying its audio
                    # signal is "audio reference", not "audio reuse".
                    a_tag = alloc.audio(cont, from_video=True)
                    if a_tag:
                        audio.append(AudioPlan(tag=a_tag, role="ambience",
                                                retention="reference",
                                                from_video_tag=c_tag))
                        soundscape_citations.append((a_tag, "reference"))
                        notes.append((a_tag, "Audio of the clip being continued from. The "
                                              "new audio should carry on its audible "
                                              "character (room tone, ambience, music feel) "
                                              "without copying the signal."))

        ff = sh.get("first_frame")
        if ff and ff.get("file"):
            tag = alloc.picture(ff)
            if tag:
                anchors.append(AnchorPlan(tag=tag, role="first_frame", shots=[i]))
        lf = sh.get("last_frame")
        if lf and lf.get("file"):
            tag = alloc.picture(lf)
            if tag:
                anchors.append(AnchorPlan(tag=tag, role="last_frame", shots=[i]))
        motion = sh.get("motion")
        if motion and motion.get("file"):
            tag = alloc.video(motion)
            if tag:
                # Text-only motion citation -- guide's own "<Video N> (motion)"
                # vocabulary, not real ControlNet conditioning (see module docstring).
                video_structures.append(VideoStructurePlan(
                    tag=tag, role="camera_reference", retention="weak_reference"))
                notes.append((tag, f"This video is the motion reference for [Shot {i}] "
                                    f"specifically -- describe its pacing and movement "
                                    f"quality as guidance for that shot's own camera and "
                                    f"action, not as footage being directly edited."))
        music = sh.get("music")
        if music and music.get("file"):
            tag = alloc.audio(music)
            if tag:
                music_citations.append((tag, "partially_copy"))
                notes.append((tag, f"This music applies to [Shot {i}] specifically."))
        sfx = sh.get("sfx")
        if sfx and sfx.get("file"):
            tag = alloc.audio(sfx)
            if tag:
                soundscape_citations.append((tag, "partially_copy"))
                notes.append((tag, f"This sound effect applies to [Shot {i}] specifically."))
        for r in sh.get("reference_images") or []:
            tag = alloc.picture(r)
            if not tag:
                continue
            ctx.reference_image_tags.append(tag)
            # Promote each shot reference image to its own Subject. Previously these
            # were only fed to the VLM as unlabelled context and had no
            # subject_definitions entry at all, so the setting/prop they showed was
            # never cited anywhere in the prompt -- it just silently vanished. The
            # guide treats scenes, backgrounds and props as <Subject N> content
            # (S2.2.1), so this is the shape they're supposed to take.
            ref_slot = _next_slot()
            ref_sp = SubjectPlan(slot=ref_slot, kind="environment", sources=[tag],
                                  retention="fully_preserved")
            ref_sp.ref_for_shot = i
            subjects.append(ref_sp)
            notes.append((tag, f"Scene/prop reference for [Shot {i}]. Describe what it "
                                f"actually shows (location, setting, or object) -- it is "
                                f"not a character. It must appear in [Shot {i}]."))

        shot_contexts.append(ctx)

    # -- Storyboard (spans shots) -----------------------------------------
    sb_item = state.get("storyboard")
    if sb_item and sb_item.get("file"):
        sb_tag = alloc.picture(sb_item)
        if sb_tag:
            n_shots = max(1, len(shot_contexts))
            anchors.append(AnchorPlan(tag=sb_tag, role="storyboard",
                                       shots=list(range(1, n_shots + 1))))
            notes.append((sb_tag, "Storyboard / shot-planning reference. Describe the "
                                   "viewpoint, subject placement and shot order it lays "
                                   "out -- it is a plan for the shots, not a frame that "
                                   "appears in the video."))

    # -- Global audio (spans all shots) -----------------------------------
    audio_section = state.get("audio") or {}
    for r in audio_section.get("dialogue") or []:
        tag = alloc.audio(r)
        if tag:
            audio.append(AudioPlan(tag=tag, role="dialogue", retention="partially_copy"))
            notes.append((tag, "This dialogue spans the whole video, not one shot -- "
                                "attribute it to whichever subject the content "
                                "actually shows speaking."))
    for r in audio_section.get("music") or []:
        tag = alloc.audio(r)
        if tag:
            music_citations.append((tag, "fully_copy"))

    dbg.section("PLAN")
    for _sp in subjects:
        _role = ("scene/prop reference for [Shot %d]" % _sp.ref_for_shot
                 if _sp.ref_for_shot else _sp.kind)
        dbg.bullet(f"Subject {_sp.slot} ({_role})")
        dbg.kv("identity from", ", ".join(_sp.sources) or "(none)")
        if _sp.motion_source:
            dbg.kv("motion from", _sp.motion_source)
        if _sp.outfit_refs:
            dbg.kv("outfit", ", ".join(f"{t} ({l})" for t, l in _sp.outfit_refs))
        dbg.kv("retention", _sp.retention)
    for _an in anchors:
        dbg.bullet(f"{_an.tag} -> {_an.role} of shot(s) {_an.shots}")
    for _vs in video_structures:
        dbg.bullet(f"{_vs.tag} -> video {_vs.role} ({_vs.retention})"
                    + ("  [character swap]" if _vs.is_character_swap else ""))
    for _ap in audio:
        _tgt = f" -> Subject {_ap.subject_slot}" if _ap.subject_slot else ""
        dbg.bullet(f"{_ap.tag} -> {_ap.role}{_tgt} ({_ap.retention})")
    for _t, _m in soundscape_citations:
        dbg.bullet(f"{_t} -> overall_soundscape ({_m})")
    for _t, _m in music_citations:
        dbg.bullet(f"{_t} -> non_diegetic_music ({_m})")
    for _sc in shot_contexts:
        dbg.bullet(f"[Shot {_sc.index}] refs={_sc.reference_image_tags or '(none)'}"
                    + ("  [replacement]" if _sc.is_replacement else ""))
    if notes:
        dbg.section("NOTES SENT TO VLM")
        for _t, _n in notes:
            dbg.bullet(f"{_t}: {_n}")

    # Name who each subject replaces, and which traits carry over from its own
    # references, so a multi-person swap isn't ambiguous.
    for _slot, _where, _keep in _replace_map:
        _sp = next((x for x in subjects if x.slot == _slot), None)
        if _sp is None:
            continue
        if _where:
            _sp.replaces_person = _where
        _sp.replace_keep = _keep or {}

    plan = CategorizationPlan(
        subjects=subjects, anchors=anchors, video_structures=video_structures,
        audio=audio, donor_pairs=donor_pairs,
        soundscape_citations=soundscape_citations, music_citations=music_citations,
        notes=notes,
    )
    return plan, alloc.tensor_index, shot_contexts
