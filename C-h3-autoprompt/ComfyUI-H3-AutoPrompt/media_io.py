"""
media_io.py -- image / video-frame / audio loading for the H3 AutoPrompt pack.

Deliberately mirrors the conventions used by Adudeguyman's
ComfyUI-Fantastic-MiniMaxH3-PromptBuilder (MIT licensed) so that reference
tag numbering ("a video's soundtrack takes the audio slot right before its
own video") behaves identically and prompts written for one pack read
naturally if you ever move to the other. No code was copied verbatim --
this is a from-scratch, minimal implementation covering only what the
auto-prompt pipeline needs (it does not need the drag-and-drop UI, presets,
or the prompt library).

Video/audio decoding tries PyAV first, then falls back to invoking ffmpeg
as a subprocess so the pack works even without the `av` package installed.
"""

from __future__ import annotations

import io
import os
import subprocess
import tempfile
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

try:
    import av  # PyAV, optional
    _HAS_AV = True
except Exception:
    _HAS_AV = False


def _ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# Images  (IMAGE tensor convention: float32, shape [1, H, W, 3], range 0..1)
# --------------------------------------------------------------------------

def pil_to_image_tensor(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]


def load_image(path: str) -> torch.Tensor:
    return pil_to_image_tensor(Image.open(path))


def image_tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Accepts a [1,H,W,3] or [H,W,3] float tensor in 0..1."""
    if t.dim() == 4:
        t = t[0]
    arr = (t.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


# --------------------------------------------------------------------------
# Video frames
# --------------------------------------------------------------------------

def _decode_frames_av(path: str, max_frames: int) -> Tuple[List[Image.Image], float]:
    container = av.open(path)
    stream = container.streams.video[0]
    duration = float(stream.duration * stream.time_base) if stream.duration else 0.0
    frames = []
    for frame in container.decode(stream):
        frames.append(frame.to_image())
    container.close()
    if not frames:
        return [], duration
    if len(frames) > max_frames:
        idxs = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        frames = [frames[i] for i in idxs]
    return frames, duration


def _decode_frames_ffmpeg(path: str, max_frames: int) -> Tuple[List[Image.Image], float]:
    dur = _ffprobe_duration(path)
    fps = max_frames / dur if dur > 0 else 1.0
    with tempfile.TemporaryDirectory() as td:
        pattern = os.path.join(td, "f_%04d.png")
        subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-vf", f"fps={fps:.4f}", pattern],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        )
        files = sorted(f for f in os.listdir(td) if f.endswith(".png"))[:max_frames]
        frames = [Image.open(os.path.join(td, f)).copy() for f in files]
    return frames, dur


def _ffprobe_duration(path: str) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, check=True,
        )
        return float(out.stdout.strip())
    except Exception:
        return 0.0


def load_video_frames(path: str, max_frames: int = 6) -> torch.Tensor:
    """Returns a stacked IMAGE tensor [N,H,W,3] of up to max_frames evenly
    sampled frames -- enough context for a small VLM without decoding the
    whole clip into VRAM."""
    if _HAS_AV:
        frames, _ = _decode_frames_av(path, max_frames)
    elif _ffmpeg_available():
        frames, _ = _decode_frames_ffmpeg(path, max_frames)
    else:
        raise RuntimeError(
            "Neither PyAV nor ffmpeg is available; install one to read video "
            "references (`pip install av` is usually the quickest route)."
        )
    tensors = [pil_to_image_tensor(f)[0] for f in frames]
    return torch.stack(tensors, dim=0)


def video_duration(path: str) -> float:
    if _HAS_AV:
        try:
            container = av.open(path)
            stream = container.streams.video[0]
            d = float(stream.duration * stream.time_base) if stream.duration else 0.0
            container.close()
            return d
        except Exception:
            pass
    return _ffprobe_duration(path)


# --------------------------------------------------------------------------
# Audio  (AUDIO convention matches ComfyUI core: {"waveform": [1,C,T] float32
# tensor at `sample_rate`, "sample_rate": int})
# --------------------------------------------------------------------------

def _decode_audio_ffmpeg(path: str, sample_rate: int = 16000) -> Tuple[torch.Tensor, int]:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out_path = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", str(sample_rate), out_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        )
        import wave
        with wave.open(out_path, "rb") as wf:
            sr = wf.getframerate()
            n = wf.getnframes()
            raw = wf.readframes(n)
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        wav = torch.from_numpy(arr)[None, None, :]  # [1,1,T]
        return wav, sr
    finally:
        try:
            os.remove(out_path)
        except OSError:
            pass


def load_audio(path: str, sample_rate: int = 16000) -> dict:
    wav, sr = _decode_audio_ffmpeg(path, sample_rate)
    return {"waveform": wav, "sample_rate": sr}


def extract_audio(video_path: str, sample_rate: int = 16000) -> dict:
    """Split the audio track off a video file -- used for a video's paired
    or standalone soundtrack reference."""
    return load_audio(video_path, sample_rate)


def audio_to_wav_bytes(audio: dict) -> bytes:
    """AUDIO dict -> in-memory 16-bit mono WAV bytes, for handing to Whisper."""
    import wave
    wav = audio["waveform"]
    if wav.dim() == 3:
        wav = wav[0, 0]
    elif wav.dim() == 2:
        wav = wav[0]
    arr = (wav.clamp(-1, 1).cpu().numpy() * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(audio["sample_rate"]))
        wf.writeframes(arr.tobytes())
    return buf.getvalue()


def has_audio_stream(path: str) -> bool:
    if _HAS_AV:
        try:
            container = av.open(path)
            has = len(container.streams.audio) > 0
            container.close()
            return has
        except Exception:
            return False
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
             "stream=index", "-of", "csv=p=0", path],
            capture_output=True, text=True, check=True,
        )
        return bool(out.stdout.strip())
    except Exception:
        return False
