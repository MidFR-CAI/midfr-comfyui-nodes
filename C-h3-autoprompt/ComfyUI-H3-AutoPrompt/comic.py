"""
comic.py -- turns a comic page into shots.

Everything here is deterministic image processing and arithmetic; nothing calls
a model. The VLM's only job downstream is describing what a (cleaned) panel
shows and filling the gap between consecutive panels, which the normal shot
pipeline already handles.

Pipeline:
    detect_panels()   -- find panel rectangles, in reading order
    detect_bubbles()  -- find speech balloons inside a panel
    clean_panel()     -- inpaint the balloons away so the video has no text
    estimate_speech() -- how long the dialogue in a panel takes to say
    chunk_panels()    -- split into runs that respect H3's 15s / 9-image limits

Auto-detection handles regular gutter-separated grids well. Splash pages,
overlapping panels and full-bleed art are genuinely hard, so a caller can pass
manual rectangles instead -- detect_panels reports a confidence flag so the UI
knows when to suggest that.
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

# H3's own envelope; chunking never produces a run that breaks these.
MAX_SECONDS = 15.0
MIN_SECONDS = 5.0
MAX_IMAGES = 9

# Speech pacing. 2.5 words/sec is an unhurried delivery -- comics are usually
# read faster than they'd be performed, and rushing dialogue is the failure mode
# that actually ruins a generation.
WORDS_PER_SECOND = 2.5
MIN_PANEL_SECONDS = 1.2          # a silent panel still needs time to register
PANEL_PADDING_SECONDS = 0.4      # beat between lines/panels


@dataclass
class Bubble:
    """A speech balloon inside a panel, in panel-local pixels."""
    x: int
    y: int
    w: int
    h: int
    text: str = ""                 # filled by OCR/VLM, always user-editable
    speaker_subject: Optional[int] = None


@dataclass
class Panel:
    index: int                     # 1-based, in reading order
    x: int
    y: int
    w: int
    h: int
    bubbles: List[Bubble] = field(default_factory=list)
    description: str = ""

    @property
    def area(self) -> int:
        return self.w * self.h


def _require_cv2():
    if cv2 is None or np is None:
        raise RuntimeError(
            "Comic mode needs OpenCV and NumPy. ComfyUI normally ships both; "
            "install with: pip install opencv-python numpy")


def _reading_order(panels: List[Tuple[int, int, int, int]], page_w: int,
                    rtl: bool = False) -> List[Tuple[int, int, int, int]]:
    """Sorts rectangles into reading order: top row first, then across it.

    Rows are grouped by vertical overlap rather than exact y, because panels in
    the same row rarely share a top edge. `rtl` handles manga, which reads
    right-to-left -- getting this wrong silently reverses every conversation.
    """
    if not panels:
        return []
    remaining = sorted(panels, key=lambda r: r[1])
    rows: List[List[Tuple[int, int, int, int]]] = []
    while remaining:
        first = remaining.pop(0)
        row = [first]
        f_top, f_bot = first[1], first[1] + first[3]
        still = []
        for r in remaining:
            r_top, r_bot = r[1], r[1] + r[3]
            overlap = min(f_bot, r_bot) - max(f_top, r_top)
            # Same row if they overlap vertically by at least half the shorter one.
            if overlap > 0.5 * min(first[3], r[3]):
                row.append(r)
            else:
                still.append(r)
        remaining = still
        row.sort(key=lambda r: r[0], reverse=rtl)
        rows.append(row)
    return [r for row in rows for r in row]


def detect_panels(image_path: str, rtl: bool = False,
                   min_area_frac: float = 0.02) -> Tuple[List[Panel], bool]:
    """Finds panel rectangles on a page.

    Returns (panels, confident). `confident` is False when the result looks
    like detection failed -- one panel covering the whole page, or none at all --
    so the UI can prompt for manual rectangles instead of silently proceeding
    with a wrong split.
    """
    _require_cv2()
    img = cv2.imread(image_path)
    if img is None:
        return [], False
    page_h, page_w = img.shape[:2]
    page_area = float(page_h * page_w)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Panels are content; gutters are a flat bright (or flat dark) margin.
    # Otsu splits content from gutter without assuming a specific page colour.
    _th, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Close small gaps so line art and lettering inside a panel merge into one
    # blob rather than fragmenting into dozens of contours.
    k = max(3, int(min(page_h, page_w) * 0.01) | 1)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < page_area * min_area_frac:
            continue
        # Reject slivers -- gutter artefacts and stray borders.
        if w < page_w * 0.06 or h < page_h * 0.04:
            continue
        rects.append((x, y, w, h))

    rects = _dedupe_nested(rects)
    ordered = _reading_order(rects, page_w, rtl=rtl)
    panels = [Panel(index=i + 1, x=x, y=y, w=w, h=h)
              for i, (x, y, w, h) in enumerate(ordered)]

    whole_page = (len(panels) == 1
                  and panels[0].area > page_area * 0.85)
    confident = bool(panels) and not whole_page and len(panels) <= 24
    return panels, confident


def _dedupe_nested(rects, overlap_frac: float = 0.8):
    """Drops rectangles almost entirely inside another -- a panel border often
    yields both an outer and an inner contour."""
    keep = []
    for r in sorted(rects, key=lambda r: r[2] * r[3], reverse=True):
        rx, ry, rw, rh = r
        nested = False
        for k in keep:
            kx, ky, kw, kh = k
            ix = max(0, min(rx + rw, kx + kw) - max(rx, kx))
            iy = max(0, min(ry + rh, ky + kh) - max(ry, ky))
            if ix * iy > overlap_frac * rw * rh:
                nested = True
                break
        if not nested:
            keep.append(r)
    return keep


def detect_bubbles(panel_img, min_area_frac: float = 0.01) -> List[Bubble]:
    """Finds speech balloons: bright, roughly convex blobs that contain dark
    marks (the lettering). The dark-content test is what separates a balloon
    from a plain bright background like sky or snow."""
    _require_cv2()
    gray = cv2.cvtColor(panel_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    area = float(h * w)
    # Balloons are near-white; 200 keeps off-white paper stock in range.
    _t, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    out = []
    for c in contours:
        bx, by, bw, bh = cv2.boundingRect(c)
        if bw * bh < area * min_area_frac or bw * bh > area * 0.6:
            continue
        # Absolute floor as well as a relative one: on a large panel 1% of the
        # area is still big enough to catch highlights and specular blobs, which
        # is where the nonsense OCR reads were coming from.
        if bw < 40 or bh < 24:
            continue
        # Balloons are never extreme slivers.
        ar = bw / float(max(1, bh))
        if ar < 0.5 or ar > 8.0:
            continue
        roi = gray[by:by + bh, bx:bx + bw]
        dark_frac = float((roi < 100).sum()) / max(1, roi.size)
        # Lettering occupies a small but non-trivial share of a balloon.
        if not (0.02 < dark_frac < 0.45):
            continue
        # Balloons are far more convex than incidental bright shapes.
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0 or cv2.contourArea(c) / hull_area < 0.7:
            continue
        out.append(Bubble(x=bx, y=by, w=bw, h=bh))

    # Reading order within the panel, same rule as panels on a page.
    out.sort(key=lambda b: (b.y // max(1, int(h * 0.15)), b.x))
    return out


def ocr_bubble(panel_img, bubble: "Bubble", upscale: int = 2) -> str:
    """Reads one speech balloon with Tesseract.

    Tesseract is weak on a full comic page but strong on an ISOLATED balloon
    with a clean background -- which is exactly what bubble detection produces,
    and what the published comic-OCR pipelines rely on. It's CPU-only and takes
    milliseconds, so it's tried before spending a VLM call. PSM 6 ("a uniform
    block of text") suits balloon lettering; the default mode assumes a page
    layout and fragments short lines.

    Returns "" when Tesseract is unavailable or reads nothing, so the caller can
    fall back to the VLM.
    """
    try:
        import pytesseract
    except Exception:
        return ""
    if not _locate_tesseract():
        return ""
    _require_cv2()
    # Pad the crop: a tight bounding box clips ascenders/descenders on the first
    # and last lines, and Tesseract drops a line it thinks is cut off -- which
    # showed up as balloons reading as empty.
    pad = max(4, int(0.06 * max(bubble.w, bubble.h)))
    ph, pw = panel_img.shape[:2]
    y0, y1 = max(0, bubble.y - pad), min(ph, bubble.y + bubble.h + pad)
    x0, x1 = max(0, bubble.x - pad), min(pw, bubble.x + bubble.w + pad)
    crop = panel_img[y0:y1, x0:x1]
    if crop.size == 0:
        return ""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if upscale > 1:
        gray = cv2.resize(gray, None, fx=upscale, fy=upscale,
                           interpolation=cv2.INTER_CUBIC)

    # Try several preprocessings and keep whichever Tesseract is most confident
    # about. One fixed pipeline produced confident-looking garbage ("&d,", "FS.")
    # on real pages: comic lettering varies a lot in weight and colour, and
    # captions are sometimes white-on-black, so the right binarisation differs
    # from balloon to balloon.
    _t, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    best_text, best_conf = "", -1.0
    for img in (otsu, cv2.bitwise_not(otsu), gray):
        try:
            data = pytesseract.image_to_data(img, config="--psm 6",
                                              output_type=pytesseract.Output.DICT)
        except Exception:
            continue
        words, confs = [], []
        for txt, cf in zip(data.get("text", []), data.get("conf", [])):
            try:
                cf = float(cf)
            except (TypeError, ValueError):
                continue
            word = str(txt).strip()
            if cf < 0 or not word:
                continue
            # Drop individual low-confidence tokens instead of only judging the
            # mean: a couple of garbage fragments inside a good read were
            # surviving as stray characters between lines.
            if cf < 60:
                continue
            # A lone letter or symbol between words is almost always a balloon
            # edge or tail picked up as text.
            if len(word) == 1 and word.upper() not in ("A", "I"):
                continue
            words.append(word)
            confs.append(cf)
        if not words:
            continue
        mean_conf = sum(confs) / len(confs)
        if mean_conf > best_conf:
            best_conf, best_text = mean_conf, " ".join(words)

    cleaned = _tidy_ocr(best_text)
    # Reject rather than surface junk. An empty box you type into is far more
    # useful than three wrong characters that look deliberate -- and a wrong word
    # here would be spoken verbatim in the generated video.
    if not _looks_like_text(cleaned, best_conf):
        return ""
    return cleaned


def _looks_like_text(text: str, conf: float) -> bool:
    """Whether an OCR result is plausible dialogue rather than noise.

    Tesseract reports high confidence on nonsense more often than you'd expect,
    so the shape of the string is checked too: real lettering is mostly letters,
    and a line of dialogue is more than one or two stray characters."""
    if not text or conf < 65:
        return False
    letters = sum(ch.isalpha() for ch in text)
    if letters < 3:
        return False
    return letters / max(1, len(text.replace(" ", ""))) >= 0.6


def _tidy_ocr(text: str) -> str:
    """Comic lettering is usually all-caps and hard-wrapped inside the balloon.
    Joins the wrapped lines back into sentences and drops the stray marks
    Tesseract emits for balloon outlines."""
    import re
    lines = [ln.strip() for ln in (text or "").splitlines()]
    lines = [ln for ln in lines if len(ln) > 1 or ln.isalnum()]
    joined = " ".join(lines)
    # Balloon outlines and tails are read as stray punctuation; curly quotes in
    # particular came through as leading junk on almost every line.
    joined = re.sub(r"[|_~^`\u2018\u2019\u201c\u201d\u00b4]+", " ", joined)
    joined = re.sub(r"^[\s'\",.;:-]+", "", joined)
    joined = re.sub(r"\s+'\s+", " ", joined)
    joined = re.sub(r"\s+([,.!?;:])", r"\1", joined)
    joined = re.sub(r"\s{2,}", " ", joined).strip()
    return joined


def ocr_panel(panel_img, panel: "Panel") -> int:
    """Fills in every empty bubble on a panel. Returns how many were read, so
    the caller knows whether to fall back to the VLM for the rest."""
    filled = 0
    for b in panel.bubbles:
        if (b.text or "").strip():
            continue
        got = ocr_bubble(panel_img, b)
        if got:
            b.text = got
            filled += 1
    return filled


def detect_people(image_path: str, frame_index: int = 0) -> List[dict]:
    """Finds the people in a video's frame, so a replacement can be pointed at a
    specific one.

Best-effort only. Uses OpenCV's bundled frontal-face Haar cascade, which
    works on photographed head-and-shoulders framing but NOT on drawn or anime
    faces (verified: zero detections on an anime character sheet). It is a hint
    to save the user some clicking, never a requirement -- an empty result is
    normal and the caller should let the user mark the person instead.

    Returns boxes left to right, each with a plain-language position so a person
    can be identified in words as well as coordinates.
    """
    _require_cv2()
    if image_path.lower().endswith((".mp4", ".webm", ".mov", ".mkv", ".avi")):
        cap = cv2.VideoCapture(image_path)
        if frame_index:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return []
    else:
        frame = cv2.imread(image_path)
    if frame is None:
        return []

    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boxes = []

    try:
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                          minSize=(int(w * 0.04), int(h * 0.04)))
        for (fx, fy, fw, fh) in faces:
            # Expand the face box into a rough head-and-torso region, which is
            # what someone is actually pointing at when they say "this person".
            bx = max(0, int(fx - fw * 0.6))
            by = max(0, int(fy - fh * 0.4))
            bw = min(w - bx, int(fw * 2.2))
            bh = min(h - by, int(fh * 4.0))
            boxes.append((bx, by, bw, bh, "face"))
    except Exception:
        pass

    # NOTE: OpenCV's HOG people detector was tried here as a fallback and
    # SEGFAULTED on a real image -- a native crash Python cannot catch, which
    # would take ComfyUI down with it. It is deliberately not used. Face
    # detection is best-effort only; when it finds nothing the caller falls back
    # to the user pointing at the person, which is reliable and needs no model.

    boxes = _dedupe_nested([(b[0], b[1], b[2], b[3]) for b in boxes])
    boxes.sort(key=lambda b: b[0])       # left to right, as people describe them
    out = []
    for i, (x, y, bw, bh) in enumerate(boxes, start=1):
        cx = (x + bw / 2.0) / max(1, w)
        where = "on the left" if cx < 0.38 else ("on the right" if cx > 0.62
                                                  else "in the centre")
        out.append({"index": i, "x": x, "y": y, "w": bw, "h": bh,
                     "where": where,
                     "label": f"person {i} ({where})"})
    return out


# Common Windows install locations. ComfyUI's embedded Python often does NOT
# inherit the user's PATH, so "tesseract is on my PATH" and "pytesseract can find
# tesseract" are different claims -- pytesseract shells out and gets a
# FileNotFoundError. Pointing it at the binary directly avoids that entirely.
_TESSERACT_CANDIDATES = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    "/usr/bin/tesseract",
    "/usr/local/bin/tesseract",
    "/opt/homebrew/bin/tesseract",
]


def _locate_tesseract() -> Optional[str]:
    """Finds the tesseract binary and tells pytesseract about it. Checks PATH
    first, then the usual install locations."""
    import os
    import shutil
    try:
        import pytesseract
    except Exception:
        return None
    cmd = getattr(pytesseract.pytesseract, "tesseract_cmd", "") or ""
    if cmd and (os.path.isabs(cmd) and os.path.exists(cmd)):
        return cmd
    found = shutil.which("tesseract")
    if not found:
        for cand in _TESSERACT_CANDIDATES:
            if os.path.exists(cand):
                found = cand
                break
    if found:
        pytesseract.pytesseract.tesseract_cmd = found
    return found


def estimate_style(image_path: str) -> dict:
    """Decides photographic vs drawn from pixel statistics, no model involved.

    Cel/anime art and photographs differ in ways that are trivially measurable
    and that a small VLM gets wrong surprisingly often:
      * flat fills -- drawn art has large areas of a single exact colour, while a
        photo has sensor noise everywhere, so local standard deviation in the
        smoothest regions is near zero for art and clearly non-zero for a photo;
      * palette size -- drawn art quantises to far fewer distinct colours;
      * edges -- line art produces a small number of very strong gradients,
        photography produces many weak ones.

    Returns the verdict plus the measurements, so a caller can show its working
    rather than asserting an answer.
    """
    _require_cv2()
    img = cv2.imread(image_path)
    if img is None:
        return {"style": None, "reason": "unreadable"}
    # Work at a consistent scale so the thresholds mean the same thing for a
    # thumbnail and a 4K still.
    h, w = img.shape[:2]
    scale = 512.0 / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                          interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Local variation: mean of the lowest decile of 5x5 standard deviations.
    blur = cv2.blur(gray.astype("float32"), (5, 5))
    sq = cv2.blur((gray.astype("float32") ** 2), (5, 5))
    local_sd = np.sqrt(np.maximum(sq - blur * blur, 0))
    flat_sd = float(np.percentile(local_sd, 10))

    # Palette: distinct colours after a light quantisation. //24 was far too
    # coarse -- it caps at 1331 buckets, so every image scored as "few colours"
    # and the test was meaningless. //4 gives 64^3 buckets, enough to separate a
    # flat cel fill from a photograph's continuous tone.
    q = (img // 4).astype("uint8")
    palette = int(len(np.unique(q.reshape(-1, 3), axis=0)))

    edges = cv2.Canny(gray, 80, 200)
    edge_frac = float((edges > 0).mean())

    # Drawn art: flat fills, small palette. Photo: noise everywhere, big palette.
    # Only flat_sd survived testing. Palette size and edge density both misfire:
    # a smooth-gradient photograph has almost no hard edges and a modest palette,
    # so they voted "drawn" for an obvious photo. Sensor noise is the one thing
    # cameras always add and flat cel fills never have.
    #
    # Reported with a CONFIDENCE rather than as a verdict, because it is only
    # decisive at the extremes: heavily denoised, graded or CG-rendered footage
    # sits in the middle and genuinely can't be called from pixel statistics.
    if flat_sd < 0.5:
        style, confidence = "2D-animated", "high"
    elif flat_sd > 3.0:
        style, confidence = "Live-action, cinematic", "high"
    elif flat_sd < 1.5:
        style, confidence = "2D-animated", "low"
    else:
        style, confidence = "Live-action, cinematic", "low"
    return {"style": style, "confidence": confidence, "flat_sd": round(flat_sd, 2),
             "palette": palette, "edge_frac": round(edge_frac, 4)}


def tesseract_status() -> dict:
    """Why OCR is or isn't usable, in enough detail to act on. A bare False sent
    people hunting for a PATH problem when the real cause was usually that the
    embedded interpreter can't see the binary."""
    try:
        import pytesseract
    except Exception as exc:
        return {"ok": False,
                "reason": f"pytesseract not importable in this Python ({exc}). "
                          f"Install it into ComfyUI's python_embeded."}
    path = _locate_tesseract()
    if not path:
        return {"ok": False,
                "reason": "the tesseract binary wasn't found. ComfyUI's embedded "
                          "Python often doesn't inherit your PATH, so install it "
                          "to the default location or set "
                          "pytesseract.pytesseract.tesseract_cmd."}
    try:
        version = str(pytesseract.get_tesseract_version())
    except Exception as exc:
        return {"ok": False, "path": path,
                "reason": f"found tesseract at {path} but running it failed: {exc}"}
    return {"ok": True, "path": path, "version": version}


def tesseract_available() -> bool:
    return bool(tesseract_status().get("ok"))


def clean_panel(panel_img, bubbles: List[Bubble], dilate: int = 6):
    """Inpaints the balloons away so the panel can be used as a reference
    without the lettering being reproduced in the video. Dilation covers the
    balloon outline, which otherwise survives as a visible ring."""
    _require_cv2()
    if not bubbles:
        return panel_img
    mask = np.zeros(panel_img.shape[:2], dtype=np.uint8)
    for b in bubbles:
        cv2.rectangle(mask, (b.x, b.y), (b.x + b.w, b.y + b.h), 255, -1)
    if dilate > 0:
        mask = cv2.dilate(mask, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate * 2 + 1, dilate * 2 + 1)))
    return cv2.inpaint(panel_img, mask, 3, cv2.INPAINT_TELEA)


def estimate_speech(texts: List[str]) -> float:
    """Seconds needed to speak these lines, with a beat between them and a floor
    so a silent panel still gets screen time."""
    total = 0.0
    for t in texts:
        words = len((t or "").split())
        if words:
            total += words / WORDS_PER_SECOND + PANEL_PADDING_SECONDS
    return max(MIN_PANEL_SECONDS, total)


def panel_duration(panel: Panel) -> float:
    return estimate_speech([b.text for b in panel.bubbles])


def chunk_panels(panels: List[Panel],
                  max_seconds: float = MAX_SECONDS,
                  max_images: int = MAX_IMAGES,
                  split_on_aspect: bool = True) -> List[List[Panel]]:
    """Splits panels into runs that each fit one H3 generation.

    A page's dialogue routinely exceeds 15 seconds, and its panels can exceed
    the 9-image reference cap, so a page usually cannot be a single generation.
    Panels are never split across runs, and a single panel longer than the cap
    gets a run of its own (its duration is clamped later rather than silently
    dropping it).
    """
    from . import resolution as rz

    chunks: List[List[Panel]] = []
    current: List[Panel] = []
    current_secs = 0.0
    current_ratio = None
    for p in panels:
        d = panel_duration(p)
        ratio = rz.ratio_of(p.w, p.h)
        too_long = current and (current_secs + d > max_seconds)
        too_many = current and (len(current) + 1 > max_images)
        # One generation produces ONE frame size, so panels of different shapes
        # can't share a chunk without letterboxing or cropping some of them.
        # Only an orientation change forces a split, and the caller can waive
        # even that when the person has chosen to merge.
        shape_change = (split_on_aspect and current
                        and not rz.same_ratio(current_ratio, ratio))
        if too_long or too_many or shape_change:
            chunks.append(current)
            current, current_secs, current_ratio = [], 0.0, None
        current.append(p)
        current_secs += d
        if current_ratio is None:
            current_ratio = ratio
    if current:
        chunks.append(current)
    return chunks


def chunk_aspect(chunk: List[Panel]):
    """Mean aspect ratio of a chunk, for reporting and for sizing the output."""
    from . import resolution as rz
    rs = [r for r in (rz.ratio_of(p.w, p.h) for p in chunk) if r]
    return (sum(rs) / len(rs)) if rs else None


def chunk_duration(chunk: List[Panel]) -> float:
    """Total seconds for a chunk, clamped into H3's accepted range."""
    total = sum(panel_duration(p) for p in chunk)
    return max(MIN_SECONDS, min(MAX_SECONDS, round(total, 1)))
