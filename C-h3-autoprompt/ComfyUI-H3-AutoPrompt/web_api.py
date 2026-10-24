"""
web_api.py -- server routes for the comic tab.

Panel detection, bubble OCR and chunking all live in Python (OpenCV +
Tesseract), so the browser needs somewhere to send a page and get structure
back. Registered on ComfyUI's own aiohttp server; import failures are swallowed
in __init__ so a missing dependency can never take the nodes down with it.
"""

from __future__ import annotations

import json
import traceback

try:
    from server import PromptServer
    from aiohttp import web
except Exception:                       # pragma: no cover
    PromptServer = None
    web = None


def _analyze(payload: dict) -> dict:
    """Runs the full comic analysis for one page and returns plain JSON.

    Deliberately returns partial results with a warning rather than failing
    outright: a page whose panels detect badly is still useful if the caller can
    see that and switch to drawing rectangles by hand.
    """
    from . import comic as cmc
    from .structured_loader import resolve_media_path
    import cv2

    path = resolve_media_path(payload.get("file") or "")
    rtl = bool(payload.get("rtl"))
    page = cv2.imread(path)
    if page is None:
        return {"error": f"could not read image: {path}"}

    manual = payload.get("panels") or []
    if manual:
        # Manual rectangles win: they're the fallback for pages auto-detection
        # gets wrong, so re-detecting over the top would defeat the point.
        panels = [cmc.Panel(index=i + 1, x=int(r["x"]), y=int(r["y"]),
                             w=int(r["w"]), h=int(r["h"]))
                  for i, r in enumerate(manual)]
        confident = True
    else:
        panels, confident = cmc.detect_panels(path, rtl=rtl)

    from . import tagger as tg
    tag_status = tg.status()
    ocr_status = cmc.tesseract_status()
    ocr_ok = bool(ocr_status.get("ok"))
    for p in panels:
        crop = page[p.y:p.y + p.h, p.x:p.x + p.w]
        if crop.size == 0:
            continue
        p.bubbles = cmc.detect_bubbles(crop)
        if ocr_ok:
            cmc.ocr_panel(crop, p)

    chunks = cmc.chunk_panels(
        panels, split_on_aspect=not bool(payload.get("merge_aspect")))
    return {
        "confident": confident,
        "ocr": ocr_ok,
        "ocr_detail": ocr_status,
        "tagger": tag_status,
        "page": {"w": int(page.shape[1]), "h": int(page.shape[0])},
        "panels": [
            {"x": p.x, "y": p.y, "w": p.w, "h": p.h,
             "seconds": round(cmc.panel_duration(p), 1),
             "bubbles": [{"x": b.x, "y": b.y, "w": b.w, "h": b.h, "text": b.text,
                           "speaker_subject": b.speaker_subject}
                          for b in p.bubbles]}
            for p in panels
        ],
        "chunks": [[p.index for p in ch] for ch in chunks],
        "chunk_seconds": [cmc.chunk_duration(ch) for ch in chunks],
        "chunk_aspects": [_aspect_label(cmc.chunk_aspect(ch)) for ch in chunks],
    }


def _aspect_label(ratio):
    from . import resolution as rz
    return rz.describe(ratio) if ratio else "unknown"


def _replace_frame(payload: dict) -> dict:
    """Pulls the first frame out of a source video so the user can point at the
    person being replaced.

    Written into ComfyUI's temp directory and returned as a viewable name, since
    the browser can't decode an arbitrary video container reliably. Face
    detection is included as a HINT only -- it finds nothing on drawn/animated
    footage, so the UI must work without it.
    """
    from . import comic as cmc
    from .structured_loader import resolve_media_path
    import cv2, os, time

    path = resolve_media_path(payload.get("file") or "")
    at = float(payload.get("at") or 0.0)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"error": f"could not open video: {path}"}
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    if at > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(at * fps))
    ok, frame = cap.read()
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    if not ok or frame is None:
        return {"error": "could not read a frame from that video"}

    try:
        import folder_paths
        outdir = folder_paths.get_temp_directory()
    except Exception:
        outdir = os.path.join(os.path.dirname(path), "_h3_frames")
    os.makedirs(outdir, exist_ok=True)
    name = f"h3_replace_{int(time.time() * 1000)}.png"
    cv2.imwrite(os.path.join(outdir, name), frame)

    faces = []
    try:
        faces = cmc.detect_people(os.path.join(outdir, name))
    except Exception:
        faces = []

    h, w = frame.shape[:2]
    return {"frame": f"{name} [temp]", "w": int(w), "h": int(h),
             "duration": round(total / fps, 2) if fps else None,
             "faces": faces,
             "hint": ("Face detection found nothing -- normal for animated or "
                      "stylised footage. Click the frame to mark each person "
                      "instead.") if not faces else ""}


def _ocr_box(payload: dict) -> dict:
    """OCRs one hand-drawn box on the page.

    This is the reliable case for Tesseract: the crop is exactly the balloon the
    user outlined, with no detection guesswork in front of it. Returns the text
    or an empty string -- never a guess dressed up as a reading.
    """
    from . import comic as cmc
    from .structured_loader import resolve_media_path
    import cv2

    page = cv2.imread(resolve_media_path(payload.get("file") or ""))
    if page is None:
        return {"error": "could not read the page image"}
    b = payload.get("box") or {}
    x, y = int(b.get("x", 0)), int(b.get("y", 0))
    w, h = int(b.get("w", 0)), int(b.get("h", 0))
    if w < 4 or h < 4:
        return {"text": ""}
    crop = page[max(0, y):y + h, max(0, x):x + w]
    if crop.size == 0:
        return {"text": ""}
    bub = cmc.Bubble(x=0, y=0, w=crop.shape[1], h=crop.shape[0])
    status = cmc.tesseract_status()
    if not status.get("ok"):
        return {"text": "", "reason": status.get("reason", "")}
    return {"text": cmc.ocr_bubble(crop, bub)}


def _retime(payload: dict) -> dict:
    """Recomputes chunking and durations after the text has been edited, without
    touching the image again -- editing a line changes how long it takes to say,
    which can move panels between chunks."""
    from . import comic as cmc
    panels = []
    for i, p in enumerate(payload.get("panels") or [], start=1):
        pn = cmc.Panel(index=i, x=int(p.get("x", 0)), y=int(p.get("y", 0)),
                        w=int(p.get("w", 1)), h=int(p.get("h", 1)))
        pn.bubbles = [cmc.Bubble(x=0, y=0, w=0, h=0, text=str(b.get("text") or ""))
                       for b in (p.get("bubbles") or [])]
        panels.append(pn)
    chunks = cmc.chunk_panels(
        panels, split_on_aspect=not bool(payload.get("merge_aspect")))
    return {
        "seconds": [round(cmc.panel_duration(p), 1) for p in panels],
        "chunks": [[p.index for p in ch] for ch in chunks],
        "chunk_seconds": [cmc.chunk_duration(ch) for ch in chunks],
        "chunk_aspects": [_aspect_label(cmc.chunk_aspect(ch)) for ch in chunks],
    }


if PromptServer is not None and web is not None:     # pragma: no cover
    routes = PromptServer.instance.routes

    @routes.post("/h3_autoprompt/comic/analyze")
    async def comic_analyze(request):
        try:
            return web.json_response(_analyze(await request.json()))
        except Exception as exc:
            traceback.print_exc()
            return web.json_response({"error": f"{type(exc).__name__}: {exc}"},
                                      status=500)

    @routes.post("/h3_autoprompt/comic/ocr")
    async def comic_ocr(request):
        try:
            return web.json_response(_ocr_box(await request.json()))
        except Exception as exc:
            traceback.print_exc()
            return web.json_response({"error": f"{type(exc).__name__}: {exc}"},
                                      status=500)

    @routes.post("/h3_autoprompt/replace/frame")
    async def replace_frame(request):
        try:
            return web.json_response(_replace_frame(await request.json()))
        except Exception as exc:
            traceback.print_exc()
            return web.json_response({"error": f"{type(exc).__name__}: {exc}"},
                                      status=500)

    @routes.post("/h3_autoprompt/comic/retime")
    async def comic_retime(request):
        try:
            return web.json_response(_retime(await request.json()))
        except Exception as exc:
            traceback.print_exc()
            return web.json_response({"error": f"{type(exc).__name__}: {exc}"},
                                      status=500)
