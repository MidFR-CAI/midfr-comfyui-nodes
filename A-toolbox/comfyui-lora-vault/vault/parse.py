"""Parse `<lora:REF:model_strength[:clip_strength][:layers=START-END]>` text and resolve REF to a vault entry.

The panel always writes canonical refs (`v12345`), but hand-typed names, aliases and
file names work too.
"""
import os
import re

from . import config, db

TAG_RE = re.compile(r"<lora:([^<>]+)>", re.IGNORECASE)
_NUM = re.compile(r"^-?\d+(\.\d+)?$")
_VID = re.compile(r"^(?:v|id:|civitai:|version:)?(\d+)$", re.IGNORECASE)


_LAYERS = re.compile(r"^layers?\s*=\s*(\d*)\s*-\s*(\d*)$", re.IGNORECASE)


def parse_layers(token: str):
    """'14-41' / '14-' / '-20' -> (14, 41) / (14, None) / (0, 20); anything else -> None."""
    m = _LAYERS.match("layers=" + token.strip())
    if not m or (m.group(1) == "" and m.group(2) == ""):
        return None
    start = int(m.group(1)) if m.group(1) else 0
    end = int(m.group(2)) if m.group(2) else None
    if end is not None and end < start:
        start, end = end, start
    return start, end


def parse(text: str):
    """-> [{raw, ref, strength, clip_strength, layer_start, layer_end}]  (layer_* None = all layers)"""
    specs = []
    for m in TAG_RE.finditer(text or ""):
        body = m.group(1).strip()
        parts = body.split(":")
        start = end = None
        while len(parts) > 1:  # trailing key=value options, e.g. layers=14-41
            tok = parts[-1].strip()
            lm = _LAYERS.match(tok)
            if not lm:
                break
            parts.pop()
            got = parse_layers(tok.split("=", 1)[1])
            if got:
                start, end = got
        nums = []
        while len(parts) > 1 and _NUM.match(parts[-1].strip()) and len(nums) < 2:
            nums.insert(0, float(parts.pop().strip()))
        ref = ":".join(parts).strip()
        strength = nums[0] if nums else 1.0
        clip = nums[1] if len(nums) > 1 else strength
        if ref:
            specs.append({"raw": m.group(0), "ref": ref, "strength": strength, "clip_strength": clip,
                          "layer_start": start, "layer_end": end})
    return specs


def tag_ref(model_name: str, version_name: str, vid: int) -> str:
    """Human-readable ref used in tags: 'Model Name - Version Name' (falls back to v<id> for odd names)."""
    ref = re.sub(r"\s+", " ", re.sub(r"[<>:|]+", " ", f"{model_name or ''} - {version_name or ''}")).strip(" -")
    return ref if ref and not _VID.match(ref) else f"v{vid}"


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


class ResolveError(Exception):
    pass


def resolve(ref: str):
    """-> ('vault', record) | ('file', absolute_path). Raises ResolveError."""
    m = _VID.match(ref.strip())
    if m:
        rec = db.get_version(int(m.group(1)))
        if rec:
            return "vault", rec
        raise ResolveError(f"Version {m.group(1)} is not in your vault yet - save it first (search or the browser extension).")

    recs = db.list_versions(light=True)
    n = _norm(ref)
    stem = _norm(os.path.splitext(ref)[0])
    hits = [r for r in recs if r.get("alias") and _norm(r["alias"]) == n]
    if not hits:
        hits = [r for r in recs if r.get("file_name") and _norm(os.path.splitext(r["file_name"])[0]) == stem]
    if not hits:
        hits = [r for r in recs if _norm(f"{r['model_name']} {r['version_name']}") == n]
    if not hits:
        hits = [r for r in recs if _norm(r["model_name"]) == n]
    if len(hits) == 1:
        return "vault", db.get_version(hits[0]["version_id"])
    if len(hits) > 1:
        ids = ", ".join(f"v{h['version_id']} ({h['model_name']} / {h['version_name']})" for h in hits[:6])
        raise ResolveError(f"'{ref}' matches several vault entries: {ids}. Use the v<id> form.")

    # Not in the vault: maybe a plain LoRA file you already have on disk.
    path = _find_local_file(ref)
    if path:
        return "file", path
    raise ResolveError(f"'{ref}' isn't in your vault or loras folder.")


def _find_local_file(ref: str):
    if os.path.isabs(ref) and os.path.isfile(ref):
        return ref
    try:
        import folder_paths

        folders = folder_paths.get_folder_paths("loras")
    except Exception:
        folders = [config.lora_root()]
    root = os.path.abspath
    for f in folders:
        for cand in (ref, ref + ".safetensors"):
            p = os.path.abspath(os.path.join(f, cand))
            # never let a ref escape the loras folder
            if os.path.commonpath([p, root(f)]) == root(f) and os.path.isfile(p):
                return p
    return None
