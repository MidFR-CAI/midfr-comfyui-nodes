"""
resolution.py -- turns a megapixel budget plus an aspect ratio into concrete
width/height integers for the generator.

Two things decide the output size:

  * a total pixel budget, given in megapixels, which the person sets once;
  * an aspect ratio, which is usually theirs to choose but is DICTATED by the
    input whenever there is a real source frame to match -- a replacement edit,
    a comic panel, or a first/last frame anchor. Generating a 16:9 video from a
    4:5 panel would letterbox or crop the very thing being matched, so in those
    modes the input wins and the node's own setting is ignored.

Dimensions are rounded to a multiple of 32, which video models generally
require, while keeping the ratio and the pixel budget as close as possible.
"""

from __future__ import annotations

import math
import re
from typing import Optional, Tuple

MULTIPLE = 32
MIN_SIDE = 256
MAX_SIDE = 3840

# Ratios offered as presets; anything parseable as "W:H" also works.
COMMON_RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "4:5", "5:4", "3:2", "2:3"]


def parse_ratio(text: str) -> Optional[float]:
    """"16:9" -> 1.777..., tolerating "16x9", "16/9" and plain decimals."""
    if text is None:
        return None
    t = str(text).strip().lower().replace("x", ":").replace("/", ":")
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\s*$", t)
    if m:
        w, h = float(m.group(1)), float(m.group(2))
        return (w / h) if h else None
    try:
        v = float(t)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def _round_to_multiple(v: float) -> int:
    return max(MULTIPLE, int(round(v / MULTIPLE)) * MULTIPLE)


def dimensions(megapixels: float, ratio: float) -> Tuple[int, int]:
    """Width/height hitting `megapixels` as closely as a /32 grid allows.

    Solved exactly first, then snapped: w = sqrt(MP * ratio), h = w / ratio.
    Snapping both independently would drift the ratio, so height is recomputed
    from the snapped width and only then snapped itself.
    """
    mp = max(0.05, float(megapixels)) * 1_000_000.0
    ratio = float(ratio) if ratio and ratio > 0 else 1.0
    w = math.sqrt(mp * ratio)
    w = _round_to_multiple(w)
    h = _round_to_multiple(w / ratio)
    # Clamp without letting either side collapse or blow past the model's range.
    w = max(MIN_SIDE, min(MAX_SIDE, w))
    h = max(MIN_SIDE, min(MAX_SIDE, h))
    return int(w), int(h)


def ratio_of(width: float, height: float) -> Optional[float]:
    try:
        w, h = float(width), float(height)
        return (w / h) if (w > 0 and h > 0) else None
    except (TypeError, ValueError):
        return None


def orientation(ratio: Optional[float]) -> str:
    """Buckets a ratio into landscape / portrait / square.

    Comparing exact ratios was far too strict: a 16:9 panel and a 2:1 panel are
    both landscape and sit happily in one clip, but a small numeric difference
    forced a whole extra generation. Only a change of ORIENTATION is worth
    splitting on, so these three buckets are the real boundary.
    """
    if not ratio or ratio <= 0:
        return "unknown"
    if ratio >= 1.2:
        return "landscape"
    if ratio <= 0.83:
        return "portrait"
    return "square"


def same_ratio(a: Optional[float], b: Optional[float], tol: float = 0.06) -> bool:
    """Whether two ratios can share one generation.

    Orientation is the test, not numeric closeness -- see orientation().
    """
    if a is None or b is None or a <= 0 or b <= 0:
        return True
    return orientation(a) == orientation(b)


def describe(ratio: Optional[float]) -> str:
    """Nearest common ratio name, for telling the person what they'll get."""
    if not ratio or ratio <= 0:
        return "unknown"
    best, best_err = None, None
    for name in COMMON_RATIOS:
        r = parse_ratio(name)
        if r is None:
            continue
        err = abs(r - ratio) / max(r, ratio)
        if best_err is None or err < best_err:
            best, best_err = name, err
    return f"{best} ({ratio:.2f}:1)" if best_err is not None and best_err < 0.06 \
        else f"{ratio:.2f}:1"
