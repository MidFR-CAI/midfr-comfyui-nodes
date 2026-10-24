"""
whisper_engine.py -- transcribes reference audio (standalone <Audio N> clips,
and the split-off soundtrack of reference videos) into timestamped, per-
speaker dialogue events that grammar.py can drop straight into <d> tags.

Uses faster-whisper (CTranslate2) running `large-v3-turbo`: it's ~6x faster
than the original openai-whisper implementation at comparable accuracy, and
turbo specifically trims the decoder layers that matter least for
transcription (as opposed to translation), which is all this pack needs --
dialogue is kept verbatim and untranslated per the guide's own rules.

Speaker attribution: H3 reference clips are short (2-15s per the guide's
own limits) and usually contain one voice. So:
  - If the audio came from a video whose reference role is a single known
    subject, every segment in that clip is attributed to that subject --
    no diarization needed.
  - If a clip contains more than one voice (rare, given the length limit,
    but possible for a two-person snippet), a lightweight embedding-based
    turn clustering step splits it. pyannote.audio is used if installed;
    otherwise a lightweight silence/pitch-change heuristic groups
    Whisper's own segments into turns, which is coarser but has zero extra
    model weight and works fully offline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class WhisperSegment:
    start: float
    end: float
    text: str
    lang: str
    words: List[dict] = field(default_factory=list)  # [{"word","start","end"}, ...] if available


@dataclass
class SpeakerTurn:
    start: float
    end: float
    text: str
    lang: str
    cluster: int   # 0-based local speaker index within this clip


LANG_CODE_TO_NAME = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "ja": "Japanese", "ko": "Korean",
    "zh": "Chinese", "ru": "Russian", "ar": "Arabic", "hi": "Hindi",
    "nl": "Dutch", "tr": "Turkish", "pl": "Polish", "vi": "Vietnamese",
    "th": "Thai", "id": "Indonesian", "uk": "Ukrainian", "sv": "Swedish",
}


def _split_segment_by_sentence(seg: WhisperSegment) -> List[WhisperSegment]:
    """Splits an over-merged Whisper segment into individual sentences using
    its own word-level timestamps, so a VAD segmentation choice (or a hard
    mid-clip trim that removed the natural pause between two lines) doesn't
    force multiple distinct lines of dialogue into a single <d> block.
    Splits after any word ending in ./!/?, using that word's own end
    timestamp as the sentence boundary -- gives each resulting line its own
    accurate start/end for shot-bucketing and <scenetrans>/<cutoff> logic,
    rather than inheriting the whole original segment's wider span. Falls
    back to returning the segment unchanged if word timestamps aren't
    available or nothing sentence-final is found inside it (i.e. it was
    already just one sentence)."""
    words = seg.words
    if not words or len(words) < 2:
        return [seg]

    sentences: List[List[dict]] = []
    current: List[dict] = []
    for w in words:
        current.append(w)
        token = (w.get("word") or "").strip()
        if token and token[-1] in ".!?":
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    if len(sentences) <= 1:
        return [seg]

    out = []
    for sent_words in sentences:
        text = "".join(w.get("word", "") for w in sent_words).strip()
        if not text:
            continue
        out.append(WhisperSegment(
            start=sent_words[0]["start"], end=sent_words[-1]["end"],
            text=text, lang=seg.lang, words=sent_words,
        ))
    return out or [seg]


class FasterWhisperTranscriber:
    def __init__(self, model_size: str = "large-v3-turbo", device: str = "cuda",
                 compute_type: str = "float16"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from faster_whisper import WhisperModel
        self._model = WhisperModel(self.model_size, device=self.device,
                                    compute_type=self.compute_type)

    def transcribe(self, wav_path: str, forced_language: Optional[str] = None,
                    split_sentences: bool = True) -> List[WhisperSegment]:
        self._ensure_loaded()
        segments, info = self._model.transcribe(
            wav_path,
            language=forced_language,
            word_timestamps=True,
            vad_filter=True,               # skip the silent padding H3 clips often have
            vad_parameters={"min_silence_duration_ms": 300},
        )
        lang_name = LANG_CODE_TO_NAME.get(info.language, info.language.capitalize())
        out = []
        for seg in segments:
            words = [{"word": w.word, "start": w.start, "end": w.end} for w in (seg.words or [])]
            out.append(WhisperSegment(start=seg.start, end=seg.end, text=seg.text.strip(),
                                       lang=lang_name, words=words))
        if split_sentences:
            split_out = []
            for seg in out:
                split_out.extend(_split_segment_by_sentence(seg))
            out = split_out
        return out


# --------------------------------------------------------------------------
# Speaker turn grouping
# --------------------------------------------------------------------------

def _diarize_pyannote(wav_path: str, segments: List[WhisperSegment]) -> List[SpeakerTurn]:
    from pyannote.audio import Pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    diarization = pipeline(wav_path)

    def cluster_at(t: float) -> int:
        for turn, _, label in diarization.itertracks(yield_label=True):
            if turn.start <= t <= turn.end:
                return hash(label) % 997  # stable-enough small int per label
        return 0

    return [SpeakerTurn(start=s.start, end=s.end, text=s.text, lang=s.lang,
                         cluster=cluster_at((s.start + s.end) / 2)) for s in segments]


def _diarize_single_speaker(segments: List[WhisperSegment]) -> List[SpeakerTurn]:
    """No extra model, no guessing: every segment in the clip is the same
    speaker (cluster 0). This used to instead split on any silence gap
    over ~1.2s, on the theory that a pause suggests a speaker change --
    but a pause is just as often a single speaker taking a breath between
    lines of a monologue, and H3 reference clips are short (2-15s per the
    guide's own budget) specifically because they're meant to carry ONE
    voice. In practice the gap heuristic reliably shattered a single
    dramatic monologue into five or six false speakers, each grabbing its
    own (Sx) ID and breaking the guide's 'one (Sx) per real speaker' rule.
    Real multi-speaker separation needs an actual model -- use pyannote
    (use_pyannote=True, the default) for clips that genuinely contain more
    than one voice."""
    return [SpeakerTurn(start=s.start, end=s.end, text=s.text, lang=s.lang, cluster=0)
            for s in segments]


def diarize(wav_path: str, segments: List[WhisperSegment], use_pyannote: bool = True) -> List[SpeakerTurn]:
    if not segments:
        return []
    if use_pyannote:
        try:
            return _diarize_pyannote(wav_path, segments)
        except Exception:
            pass  # pyannote not installed / no HF token accepted for the gated model -- fall back
    return _diarize_single_speaker(segments)


# --------------------------------------------------------------------------
# High-level entry point used by the ComfyUI node
# --------------------------------------------------------------------------

@dataclass
class DialogueClip:
    media_tag: str          # "Audio 1", "Video 2" (paired soundtrack), etc.
    turns: List[SpeakerTurn]
    clip_duration_s: float


def transcribe_clip(transcriber: FasterWhisperTranscriber, wav_path: str, media_tag: str,
                     clip_duration_s: float, forced_language: Optional[str] = None,
                     use_pyannote: bool = True) -> DialogueClip:
    segments = transcriber.transcribe(wav_path, forced_language=forced_language)
    turns = diarize(wav_path, segments, use_pyannote=use_pyannote)
    return DialogueClip(media_tag=media_tag, turns=turns, clip_duration_s=clip_duration_s)
