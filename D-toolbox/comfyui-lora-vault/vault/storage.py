"""LoRA files on disk: on-demand download, hash check, pin/unpin, LRU eviction.

Layout (inside your loras folder, so other loader nodes can see these files too):
  loras/_vault/cache/   - downloaded on demand, evicted oldest-used-first past the size cap
  loras/_vault/pinned/  - never evicted (your permanent backups)
"""
import hashlib
import os
import re
import shutil
import threading
import time
import urllib.parse

from . import civitai, config, db, relay

_locks: dict = {}
_locks_guard = threading.Lock()
jobs: dict = {}  # version_id -> {state, done, total, error}


class Unavailable(Exception):
    pass


def _lock_for(vid: int) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(int(vid), threading.Lock())


def _safe(name: str) -> str:
    return re.sub(r"[^\w.\- ]+", "_", name or "model.safetensors").strip() or "model.safetensors"


_BAD_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def display_filename(rec: dict) -> str:
    """'Model Name - Version Name.safetensors' - what ComfyUI's own LoRA dropdowns show."""
    ext = os.path.splitext(rec.get("file_name") or "")[1].lower()
    if ext not in (".safetensors", ".pt", ".ckpt", ".bin", ".pth"):
        ext = ".safetensors"
    base = f"{rec.get('model_name') or ''} - {rec.get('version_name') or ''}".strip(" -")
    base = re.sub(r"\s+", " ", _BAD_CHARS.sub(" ", base)).strip(" .")[:140].rstrip(" .")
    return (base or f"lora {rec['version_id']}") + ext


def _unique_path(dest_dir: str, rec: dict) -> str:
    path = os.path.join(dest_dir, display_filename(rec))
    mine = rec.get("local_path")
    if os.path.exists(path) and not (mine and os.path.abspath(mine) == os.path.abspath(path)) or os.path.exists(path + ".part"):
        stem, ext = os.path.splitext(display_filename(rec))
        path = os.path.join(dest_dir, f"{stem} [{rec['version_id']}]{ext}")
    return path


def rename_local(rec: dict):
    """Give an existing file (and its export next to it) the readable name. Best effort."""
    p = rec.get("local_path")
    if not p or not os.path.isfile(p):
        return
    new = _unique_path(os.path.dirname(p), rec)
    if os.path.abspath(new) == os.path.abspath(p):
        return
    old_base, new_base = os.path.splitext(p)[0], os.path.splitext(new)[0]
    os.replace(p, new)
    for suffix in (".vault.json", ".txt", ".previews"):
        if os.path.exists(old_base + suffix) and not os.path.exists(new_base + suffix):
            try:
                os.replace(old_base + suffix, new_base + suffix)
            except OSError:
                pass
    db.set_fields(rec["version_id"], local_path=new)


def local_state(rec) -> str:
    """'pinned' | 'cached' | 'none'"""
    p = rec.get("local_path")
    if p and os.path.isfile(p):
        return "pinned" if rec.get("pinned") else "cached"
    return "none"


def touch_used(vid: int):
    db.set_fields(vid, last_used=time.time())


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def _candidate_urls(rec: dict) -> list:
    """Ways to ask for the same file: with the API key, without it (Firefox's own Civitai login),
    and on the other configured Civitai host. A paid / early-access file can answer differently."""
    url = rec["download_url"]
    p = urllib.parse.urlparse(url)
    urls = [civitai.with_token(url), url]
    for h in config.get("civitai_hosts") or []:
        hp = urllib.parse.urlparse(h)
        if hp.hostname and hp.hostname != p.hostname:
            alt = p._replace(scheme=hp.scheme or "https", netloc=hp.netloc).geturl()
            urls += [civitai.with_token(alt), alt]
    return list(dict.fromkeys(urls))


def access_hint(rec: dict) -> str:
    v = ((rec.get("snapshot") or {}).get("version")) or {}
    av = v.get("availability")
    ends = v.get("earlyAccessEndsAt")
    if av and av != "Public":
        return (f" Civitai lists this version as '{av}'" + (f" (early access ends {str(ends)[:10]})" if ends else "") +
                ". Early-access / Buzz files need the purchase on the SAME account as your API key and the account "
                "logged in to Civitai in Firefox. You can also download it by hand and use 'I downloaded it myself'.")
    return " If you bought this with Buzz, make sure the API key and the Firefox login are both for the account that bought it."


def _download_via_firefox(rec: dict, dest: str, progress=None):
    """Try each way of asking for the file until one works; report every refusal if none do."""
    errors = []
    for url in _candidate_urls(rec):
        try:
            return _download_via_firefox_one(rec, url, dest, progress)
        except relay.NotConnected:
            raise
        except (civitai.AuthRequired, civitai.NetworkError) as e:
            msg = str(e)
            if "not connected" in msg.lower() and not errors:
                raise
            errors.append(f"{urllib.parse.urlparse(url).hostname}{' (with key)' if 'token=' in url else ''}: {msg}")
    raise civitai.AuthRequired("Civitai refused every download attempt - " + " | ".join(errors) + "." + access_hint(rec))


def _download_via_firefox_one(rec: dict, url: str, dest: str, progress=None):
    """Firefox downloads the file with its own network route + Civitai login (so it handles
    redirects, resume and progress), then tells us where it saved it. We move it into place
    and verify the SHA-256."""
    part = dest + ".part"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        res = relay.submit("download", url, headers={}, idle_timeout=600, progress=progress,
                           filename="LoraVault/" + os.path.basename(dest))
    except relay.RelayError as e:
        raise civitai.NetworkError(str(e))
    src = res.get("path")
    if not src or not os.path.isfile(src):
        raise Unavailable(
            f"Firefox says the download finished ('{src}') but ComfyUI can't see that file. "
            "Firefox and ComfyUI need to be on the same PC."
        )
    shutil.move(src, part)
    expected = (rec.get("sha256") or "").upper()
    if expected:
        if _sha256_file(part) != expected:
            small = os.path.getsize(part) < 1024 * 1024
            os.remove(part)
            if small:
                raise civitai.AuthRequired(
                    "Firefox saved a tiny file instead of the LoRA - Civitai probably wants a login. "
                    "Log in to civitai.com in Firefox (or add an API key in the vault settings), then retry."
                )
            raise Unavailable("Downloaded file failed its SHA-256 check (corrupt or tampered) - deleted. Try again.")
    os.replace(part, dest)


def _download(rec: dict, dest: str, progress=None):
    """Stream to dest.part (resumable), verify SHA-256, then rename into place."""
    url = rec.get("download_url")
    if not url:
        raise Unavailable("No download URL stored for this version.")
    if civitai.transport() == "firefox":
        if config.get("download_mode") == "firefox":
            return _download_via_firefox(rec, dest, progress)
        return _download_resolved(rec, dest, progress)
    return _stream_to_file(rec, url, dest, progress)


def _download_resolved(rec: dict, dest: str, progress=None):
    """Firefox (with your Civitai login + route) turns the Civitai link into the signed file link;
    ComfyUI then downloads the bytes itself, straight from the CDN, with resume + SHA-256.
    If that can't be done (old extension, expired link, CDN blocked) fall back to a Firefox download."""
    notes = []
    for url in _candidate_urls(rec):
        try:
            res = relay.submit("resolve", url, headers={}, timeout=60)
        except relay.NotConnected:
            raise
        except relay.RelayError as e:
            notes.append(str(e)); continue
        cdn = res.get("resolved")
        if not cdn:
            notes.append(res.get("note") or "no direct link"); continue
        try:
            return _stream_to_file(rec, cdn, dest, progress)
        except (civitai.NetworkError, civitai.AuthRequired) as e:
            notes.append(f"direct download: {e}")
    print("[LoraVault] direct download not possible (" + " | ".join(notes)[:400] + ") - using Firefox to download")
    return _download_via_firefox(rec, dest, progress)


def _stream_to_file(rec: dict, url: str, dest: str, progress=None):
    part = dest + ".part"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    expected = (rec.get("sha256") or "").upper()

    h = hashlib.sha256()
    have = 0
    if os.path.isfile(part):
        with open(part, "rb") as f:  # re-hash what we already have so we can resume
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
                have += len(chunk)

    try:
        resp = civitai.open_stream(url, range_start=have)
    except civitai.CivitaiError as e:
        if "range-not-satisfiable" in str(e):
            os.remove(part)
            return _stream_to_file(rec, url, dest, progress)
        raise
    with resp:
        resumed = getattr(resp, "status", 200) == 206
        if have and not resumed:  # server ignored Range -> start over
            have = 0
            h = hashlib.sha256()
        total = have + int(resp.headers.get("Content-Length") or 0)
        mode = "ab" if resumed else "wb"
        last = 0.0
        with open(part, mode) as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                h.update(chunk)
                have += len(chunk)
                if progress and time.time() - last > 0.25:
                    progress(have, total)
                    last = time.time()
    if progress:
        progress(have, total or have)
    if expected and h.hexdigest().upper() != expected:
        os.remove(part)
        raise Unavailable("Downloaded file failed its SHA-256 check (corrupt or tampered) - deleted. Try again.")
    os.replace(part, dest)


def _policy(vid: int):
    from . import snapshot  # lazy: snapshot imports storage
    try:
        snapshot.apply_preview_policy(vid)
    except Exception as e:
        db.add_event(vid, "preview_policy_failed", str(e))


def ensure_local(vid: int, pin: bool = False, progress=None, protect=()) -> str:
    """Return a local path for this version, downloading it first if needed. Pinning also keeps
    every preview + all metadata locally (see snapshot.apply_preview_policy)."""
    path = _ensure_local(vid, pin, progress, protect)
    if pin:
        _policy(vid)
    return path


def _ensure_local(vid: int, pin: bool = False, progress=None, protect=()) -> str:
    vid = int(vid)
    with _lock_for(vid):
        rec = db.get_version(vid)
        if not rec:
            raise Unavailable(f"Version {vid} is not in the vault.")
        state = local_state(rec)
        if state != "none":
            path = rec["local_path"]
            if pin and not rec["pinned"]:
                path = _move(rec, config.pinned_dir())
                db.set_fields(vid, pinned=1, local_path=path)
            touch_used(vid)
            return path
        if rec["status"] in ("removed", "archived", "taken_down", "unpublished"):
            raise Unavailable(
                f"'{rec['model_name']}' was removed from Civitai ({rec['status']}) and no local copy was kept."
            )
        dest_dir = config.pinned_dir() if pin else config.cache_dir()
        dest = _unique_path(dest_dir, rec)
        jobs[vid] = {"state": "downloading", "done": 0, "total": 0, "error": None, "name": rec["model_name"]}

        def prog(done, total):
            jobs[vid].update(done=done, total=total)
            if progress:
                progress(done, total)

        try:
            _download(rec, dest, prog)
        except Exception as e:
            jobs[vid].update(state="error", error=str(e))
            raise
        jobs[vid].update(state="done")
        db.set_fields(vid, local_path=dest, pinned=1 if pin else 0, last_used=time.time())
    evict(protect=set(protect) | {vid})
    return dest


def _move(rec: dict, dest_dir: str) -> str:
    src = rec["local_path"]
    dest = os.path.join(dest_dir, os.path.basename(src))
    if os.path.abspath(src) != os.path.abspath(dest):
        os.makedirs(dest_dir, exist_ok=True)
        os.replace(src, dest)
    return dest


def import_file(vid: int, path: str, pin: bool = False) -> str:
    """Adopt a file the user downloaded by hand (e.g. a Buzz-locked one): check it, move it in."""
    with _lock_for(vid):
        rec = db.get_version(vid)
        if not rec:
            raise Unavailable(f"Version {vid} is not in the vault.")
        path = (path or "").strip().strip('"')
        if not os.path.isfile(path):
            raise Unavailable(f"No file at: {path}")
        expected = (rec.get("sha256") or "").upper()
        if expected and _sha256_file(path) != expected:
            raise Unavailable("That file doesn't match this version's SHA-256 - it is a different or damaged file.")
        dest = _unique_path(config.pinned_dir() if pin else config.cache_dir(), rec)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.move(path, dest)
        jobs.pop(vid, None)
        db.set_fields(vid, local_path=dest, pinned=1 if pin else 0, last_used=time.time())
    if pin:
        _policy(vid)
    return dest


def pin(vid: int, progress=None) -> str:
    return ensure_local(vid, pin=True, progress=progress)


def unpin(vid: int):
    with _lock_for(vid):
        rec = db.get_version(vid)
        if not rec:
            return
        if rec["status"] != "ok" and local_state(rec) != "none":
            raise Unavailable(
                "This model was removed from Civitai and your local file is the only copy - it stays pinned."
            )
        old_path = rec.get("local_path")
        if local_state(rec) != "none":
            db.set_fields(vid, pinned=0, local_path=_move(rec, config.cache_dir()))
        else:
            db.set_fields(vid, pinned=0)
    from . import snapshot
    snapshot.prune_previews(vid)
    snapshot.remove_sidecar(old_path)
    evict()


def delete_local(vid: int, force: bool = False):
    with _lock_for(vid):
        rec = db.get_version(vid)
        if not rec or local_state(rec) == "none":
            return
        if rec["pinned"] and not force:
            raise Unavailable("Unpin it first (or force delete).")
        if rec["status"] != "ok" and not force:
            raise Unavailable("This is the only surviving copy of a removed model - refusing without force.")
        os.remove(rec["local_path"])
        from . import snapshot
        snapshot.remove_sidecar(rec["local_path"])
        db.set_fields(vid, local_path=None, pinned=0)
    from . import snapshot as _s
    _s.prune_previews(vid)


def cache_usage() -> dict:
    cached = pinned = n_c = n_p = 0
    for r in db.list_versions(light=True):
        p = r.get("local_path")
        if p and os.path.isfile(p):
            sz = os.path.getsize(p)
            if r["pinned"]:
                pinned += sz
                n_p += 1
            else:
                cached += sz
                n_c += 1
    return {"cache_bytes": cached, "pinned_bytes": pinned, "cache_files": n_c, "pinned_files": n_p,
            "limit_bytes": int(config.get("cache_limit_gb") * 1024**3)}


def evict(protect=()):
    """Delete least-recently-used *unpinned, still-live-on-Civitai* files past the cap."""
    limit = int(config.get("cache_limit_gb") * 1024**3)
    if limit <= 0:
        return []
    protect = {int(p) for p in protect}
    cands, total = [], 0
    for r in db.list_versions(light=True):
        p = r.get("local_path")
        if r["pinned"] or not p or not os.path.isfile(p):
            continue
        sz = os.path.getsize(p)
        total += sz
        # a file whose model vanished is the last copy - never evict it
        if r["version_id"] not in protect and r["status"] == "ok":
            cands.append((r.get("last_used") or r.get("saved_at") or 0, r["version_id"], p, sz))
    removed = []
    for _, vid, p, sz in sorted(cands):
        if total <= limit:
            break
        with _lock_for(vid):
            try:
                os.remove(p)
            except OSError:
                continue
            db.set_fields(vid, local_path=None)
        total -= sz
        removed.append(vid)
    return removed


def reconcile():
    """Forget local paths whose files were deleted outside the vault; fill in newer columns."""
    db.backfill_thumbs()
    db.backfill_tags()
    for r in db.list_versions(light=True):
        p = r.get("local_path")
        if p and not os.path.isfile(p):
            db.set_fields(r["version_id"], local_path=None)
        elif p:
            try:  # older downloads were saved as '<number>__file.safetensors'
                rename_local(r)
            except Exception:
                pass
