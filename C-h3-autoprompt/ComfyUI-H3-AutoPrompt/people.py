"""
people.py -- finds the people in a source video's first frame.

Character replacement fails when the model is left to infer WHO is being
replaced. With one subject it can usually be guessed; with two it cannot, and
in practice even the single-subject case gets ignored because nothing in the
prompt points at a specific person in the footage.

So the people are detected up front, the user says which of their subjects
replaces which detected person, and the prompt names each one by where it
stands and what it looks like. Detection is deliberately cheap and local:
OpenCV's bundled HOG pedestrian detector and Haar face cascade, no model
downloads, no VRAM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

try:
    import cv2
    import numpy as np
except Exception:                       # pragma: no cover
    cv2 = None
    np = None


@dataclass
class Person:
    index: int                    # 1-based, left to right
    x: int
    y: int
    w: int
    h: int
    source: str = "hog"           # which detector found it
    description: str = ""         # filled by the VLM from the crop
    replaced_by: Optional[int] = None     # Subject slot that takes this person's place

    def position_words(self, frame_w: int, frame_h: int) -> str:
        """Plain-language position, so the prompt can point at a person without
        coordinates -- "the woman on the left of frame" is something the model
        can actually act on."""
        cx = self.x + self.w / 2.0
        third = frame_w / 3.0
        if cx < third:
            horiz = "on the left of frame"
        elif cx > 2 * third:
            horiz = "on the right of frame"
        else:
            horiz = "in the centre of frame"
        big = self.h > frame_h * 0.6
        return f"{horiz}, {'in the foreground' if big else 'further back'}"


def _require():
    if cv2 is None:
        raise RuntimeError("Person detection needs OpenCV. Install opencv-python.")


def first_frame(video_path: str):
    """Grabs frame 0. Replacement is keyed off the opening frame because that's
    where the subject mapping has to hold from."""
    _require()
    cap = cv2.VideoCapture(video_path)
    try:
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def detect_people(frame, min_frac: float = 0.03) -> List[Person]:
    """Detects people, preferring full bodies and falling back to faces.

    HOG finds standing figures in live-action footage; the Haar cascade catches
    people HOG misses (seated, close-up, partially framed). A face box is grown
    into a rough body box so both kinds describe the same thing. Neither works
    well on stylised or animated footage -- that's what the manual boxes in the
    UI are for.
    """
    _require()
    h, w = frame.shape[:2]
    area = float(h * w)
    boxes: List[Tuple[Tuple[int, int, int, int], str]] = []

    try:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        found, _weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8),
                                                scale=1.05)
        for (x, y, bw, bh) in found:
            boxes.append(((int(x), int(y), int(bw), int(bh)), "hog"))
    except Exception:
        pass

    try:
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5)
        for (x, y, fw, fh) in faces:
            # Grow a face into an approximate body so both detectors yield
            # comparable regions: roughly 3 face-widths across, 7 tall.
            bx = max(0, int(x - fw))
            by = max(0, int(y - fh * 0.4))
            bw = min(w - bx, int(fw * 3))
            bh = min(h - by, int(fh * 7))
            boxes.append(((bx, by, bw, bh), "face"))
    except Exception:
        pass

    kept: List[Tuple[Tuple[int, int, int, int], str]] = []
    for box, src in sorted(boxes, key=lambda b: b[0][2] * b[0][3], reverse=True):
        if box[2] * box[3] < area * min_frac:
            continue
        if any(_iou(box, k[0]) > 0.4 for k in kept):
            continue
        kept.append((box, src))

    kept.sort(key=lambda b: b[0][0])      # left to right, so numbering is predictable
    return [Person(index=i + 1, x=b[0], y=b[1], w=b[2], h=b[3], source=src)
            for i, (b, src) in enumerate(kept)]


def detect_in_video(video_path: str):
    """Returns (people, frame_width, frame_height). Empty list when the frame
    can't be read or nothing is found -- the caller offers manual boxes."""
    frame = first_frame(video_path)
    if frame is None:
        return [], 0, 0
    h, w = frame.shape[:2]
    return detect_people(frame), w, h
