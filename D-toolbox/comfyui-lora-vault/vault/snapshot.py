"""Save a Civitai model version into the vault: metadata snapshot + local preview images.

The snapshot (full API payloads + thumbnails) is what survives if the model is later
deleted - a few hundred KB per LoRA, versus hundreds of MB for the weights.
"""
import json
import os
import re
import shutil
import time

from . import civitai, config, db, storage

LORA_TYPES = {"LORA", "LOCON", "DORA"}


def pick_file(version: dict):
    files = [f for f in (version.get("files") or []) if f.get("type", "Model") == "Model"] or (version.get("files") or [])
    if not files:
        return None
    for pred in (
        lambda f: f.get("primary") and f.get("name", "").endswith(".safetensors"),
        lambda f: f.get("name", "").endswith(".safetensors"),
        lambda f: f.get("primary"),
    ):
        for f in files:
            if pred(f):
                return f
    return files[0]


def thumb_url(url: str, width: int) -> str:
    return re.sub(r"width=\d+", f"width={width}", url) if "width=" in url else url


def _collect_images(vdata: dict) -> list:
    """The version's own gallery first (the uploader's curated order), then extra images from
    the images endpoint. The extras also supply generation metadata the gallery often lacks, and
    the mature images that civitai.com hides from a version's own list."""
    own = [dict(i) for i in (vdata.get("images") or []) if i.get("url")]
    try:
        extra = civitai.get_images(vdata["id"])
    except Exception:
        extra = []
    by_url = {e["url"]: e for e in extra}
    for i in own:
        e = by_url.get(i["url"])
        if e and not i.get("meta") and e.get("meta"):
            i["meta"] = e["meta"]
    seen = {i["url"] for i in own}
    return own + [e for e in extra if e["url"] not in seen]


MAX_ENTRIES = 60  # example images whose text metadata we keep (local copies are limited separately)


def _keep_local(pinned: bool) -> int:
    return int(config.get("pinned_previews" if pinned else "unpinned_previews"))


def _download_previews(vid: int, images: list, keep_local: int, width: int, existing: dict | None = None) -> list:
    """One entry per example image (url + generation metadata = text, always kept); local image
    files only for the first `keep_local`. Builds the new set aside and swaps it in, reusing an
    old local file when a fetch fails - a refresh must never destroy the only copy of a preview."""
    out_dir = os.path.join(config.previews_dir(), str(vid))
    tmp_dir = out_dir + ".new"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
    existing = existing or {}
    out = []
    for img in images[:MAX_ENTRIES]:
        url = img.get("url")
        if not url:
            continue
        entry = {
            "idx": len(out), "url": url, "type": img.get("type", "image"),
            "nsfwLevel": img.get("nsfwLevel", 0), "width": img.get("width"), "height": img.get("height"),
            "meta": img.get("meta"), "local": None,
        }
        if entry["type"] == "image" and sum(1 for e in out if e["local"]) < keep_local:
            ext = os.path.splitext(url.split("?")[0])[1].lower()
            ext = ext if ext in (".jpg", ".jpeg", ".png", ".webp", ".gif") else ".jpeg"
            fname = f"{entry['idx']}{ext}"
            dest = os.path.join(tmp_dir, fname)
            old = existing.get(url)
            try:
                if old and os.path.isfile(old):
                    shutil.copyfile(old, dest)        # already have it: no download
                else:
                    with open(dest, "wb") as f:
                        f.write(civitai.fetch_bytes(thumb_url(url, width)))
                entry["local"] = fname
            except Exception:
                if os.path.exists(dest):
                    os.remove(dest)  # a missing preview must never block saving (the UI falls back to the web URL)
        out.append(entry)
    shutil.rmtree(out_dir, ignore_errors=True)
    os.replace(tmp_dir, out_dir)
    return out


def _existing_locals(vid: int, previews: list | None) -> dict:
    d = os.path.join(config.previews_dir(), str(vid))
    return {p["url"]: os.path.join(d, p["local"]) for p in (previews or []) if p.get("local")}


def refresh_previews(vid: int) -> dict:
    """Re-fetch a version's example images + generation metadata (repairs entries saved with
    missing or mature-filtered previews). How many local copies are kept depends on whether the
    LoRA is pinned. Works from the stored snapshot if Civitai no longer has the model."""
    rec = db.get_version(vid)
    if not rec:
        raise ValueError("Not in the vault.")
    snap = rec.get("snapshot") or {}
    vdata = snap.get("version") or {"id": vid}
    try:
        fresh = civitai.get_version(vid)
        if fresh.get("images") is not None:
            vdata = fresh
    except civitai.NotFound:
        pass  # removed on Civitai: re-use what we stored
    vdata.setdefault("id", vid)
    new_imgs = _collect_images(vdata)
    seen = {i["url"] for i in new_imgs}
    # keep images we already knew about even if Civitai no longer lists them (filtered / removed)
    old_imgs = [{k: p.get(k) for k in ("url", "type", "nsfwLevel", "width", "height", "meta")}
                for p in snap.get("previews") or [] if p["url"] not in seen]
    snap["previews"] = _download_previews(vid, new_imgs + old_imgs, _keep_local(rec["pinned"]),
                                          config.get("preview_width"), _existing_locals(vid, snap.get("previews")))
    snap["version"] = {**snap.get("version", {}), **{k: v for k, v in vdata.items() if k != "images"},
                       "images": vdata.get("images", [])}
    db.set_fields(vid, snapshot=snap, **thumb_fields(snap["previews"]))
    return db.get_version(vid)


def prune_previews(vid: int):
    """Unpinned: keep only the first few local preview files (text metadata stays)."""
    rec = db.get_version(vid)
    if not rec or rec["pinned"]:
        return
    snap = rec.get("snapshot") or {}
    keep, d, n = _keep_local(False), os.path.join(config.previews_dir(), str(vid)), 0
    for p in snap.get("previews") or []:
        if p.get("local"):
            n += 1
            if n > keep:
                try:
                    os.remove(os.path.join(d, p["local"]))
                except OSError:
                    pass
                p["local"] = None
    db.set_fields(vid, snapshot=snap, **thumb_fields(snap.get("previews") or []))


def _text_export(rec: dict) -> str:
    snap, lines = rec.get("snapshot") or {}, []
    version, model = snap.get("version") or {}, snap.get("model") or {}
    plain = lambda h: re.sub(r"\n{3,}", "\n\n", re.sub(r"<[^>]+>", "", re.sub(r"<\s*(br|/p|/li|/h\d)\s*/?>", "\n", h or "", flags=re.I))).strip()
    lines += [f"{rec['model_name']} - {rec['version_name']}", f"Base model: {rec['base_model']}  |  Creator: {rec['creator']}",
              f"https://civitai.com/models/{rec['model_id']}?modelVersionId={rec['version_id']}",
              f"File: {rec['file_name']}  SHA256: {rec['sha256']}", "",
              "Trigger words: " + ", ".join(rec.get("trigger_words") or []), "", "== Version description ==",
              plain(version.get("description")), "", "== Model description ==", plain(model.get("description")), ""]
    for p in snap.get("previews") or []:
        m = p.get("meta") or {}
        if m:
            lines += [f"== Example {p['idx'] + 1} ==", *(f"{k}: {v}" for k, v in m.items() if not isinstance(v, (dict, list))), ""]
    return "\n".join(lines)


def write_sidecar(vid: int):
    """Pinned LoRAs get a self-contained export next to the weights: <name>.vault.json (everything
    we know), <name>.txt (readable) and <name>.previews/ - so the metadata survives even if the
    vault database or Civitai page is gone."""
    rec = db.get_version(vid)
    if not rec or not rec["pinned"] or not rec.get("local_path"):
        return
    base = os.path.splitext(rec["local_path"])[0]
    data = {"exported_by": "lora-vault", "exported_at": time.time(),
            "record": {k: v for k, v in rec.items() if k != "local_path"}}
    with open(base + ".vault.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    with open(base + ".txt", "w", encoding="utf-8") as f:
        f.write(_text_export(rec))
    pdir, src = base + ".previews", os.path.join(config.previews_dir(), str(vid))
    os.makedirs(pdir, exist_ok=True)
    for p in (rec.get("snapshot") or {}).get("previews") or []:
        if p.get("local") and os.path.isfile(os.path.join(src, p["local"])):
            shutil.copyfile(os.path.join(src, p["local"]), os.path.join(pdir, p["local"]))


def remove_sidecar(local_path: str | None):
    if not local_path:
        return
    base = os.path.splitext(local_path)[0]
    for ext in (".vault.json", ".txt"):
        try:
            os.remove(base + ext)
        except OSError:
            pass
    shutil.rmtree(base + ".previews", ignore_errors=True)


def apply_preview_policy(vid: int):
    """Pinned -> keep EVERYTHING locally (all previews + their generation data + export next to the
    weights). Unpinned -> one preview + text metadata. Best-effort: never blocks pinning."""
    rec = db.get_version(vid)
    if not rec:
        return
    if rec["pinned"]:
        try:
            refresh_previews(vid)
        except Exception as e:  # Firefox closed / Civitai down: keep what we have
            db.add_event(vid, "pin_metadata_partial", f"Couldn't re-fetch all previews now: {e}")
        write_sidecar(vid)
    else:
        prune_previews(vid)
        remove_sidecar(rec.get("local_path"))


def thumb_fields(previews: list) -> dict:
    """Small columns the library list can read without loading every snapshot."""
    local = next((p for p in previews if p.get("local")), None)
    remote = next((p for p in previews if p.get("type") == "image"), None)
    pick = local or remote
    cover = previews[0] if previews else None
    video = cover["url"] if cover and cover.get("type") != "image" else ""
    return {"thumb_video": video, "thumb_idx": local["idx"] if local else None,
            "thumb_remote": thumb_url(remote["url"], 300) if (remote and not local) else None,
            "thumb_nsfw": (pick or {}).get("nsfwLevel", 0) or 0,
            "preview_count": len(previews)}


def tag_names(m: dict) -> list:
    """Civitai tags on a model as lower-case strings (the API gives strings or {name} objects)."""
    out = []
    for t in (m or {}).get("tags") or []:
        n = t.get("name") if isinstance(t, dict) else t
        if n and str(n).strip().lower() not in out:
            out.append(str(n).strip().lower())
    return out


def _light_version(v: dict) -> dict:
    """What we remember about a sibling version, enough to show it without going to Civitai."""
    imgs = [{k: i.get(k) for k in ("url", "type", "nsfwLevel", "width", "height", "meta")}
            for i in (v.get("images") or [])[:6] if i.get("url")]
    f = pick_file(v) or {}
    out = {k: v.get(k) for k in ("id", "name", "baseModel", "publishedAt", "stats", "trainedWords", "description")}
    out.update(images=imgs, size_kb=f.get("sizeKB") or 0)
    return out


def _light_model(m: dict) -> dict:
    m = dict(m or {})
    m["modelVersions"] = [
        _light_version(v) for v in (m.get("modelVersions") or [])
    ]
    return m


def build_record(vdata: dict, mdata: dict | None, previews: list) -> dict:
    f = pick_file(vdata) or {}
    hashes = f.get("hashes") or {}
    model = vdata.get("model") or {}
    return {
        "version_id": vdata["id"],
        "model_id": vdata.get("modelId") or (mdata or {}).get("id"),
        "model_name": model.get("name") or (mdata or {}).get("name") or f"model {vdata.get('modelId')}",
        "version_name": vdata.get("name") or "",
        "model_type": model.get("type") or (mdata or {}).get("type") or "LORA",
        "base_model": vdata.get("baseModel") or "",
        "creator": ((mdata or {}).get("creator") or {}).get("username") or "",
        "file_name": f.get("name") or f"{vdata['id']}.safetensors",
        "file_size_kb": f.get("sizeKB") or 0,
        "sha256": hashes.get("SHA256") or "",
        "download_url": f.get("downloadUrl") or vdata.get("downloadUrl") or "",
        "trigger_words": vdata.get("trainedWords") or [],
        "nsfw_level": vdata.get("nsfwLevel") or 0,
        "tags": tag_names(mdata or {}),
        "status": "ok", "status_detail": "", "miss_count": 0,
        "last_checked": time.time(), "last_ok": time.time(),
        "snapshot": {"version": vdata, "model": _light_model(mdata), "previews": previews,
                     "captured_at": time.time()},
        **thumb_fields(previews),
    }


def refresh_from_api(vdata: dict, mdata: dict | None, keep_previews_of: dict | None = None) -> dict:
    """Build + store a fresh record from API payloads (used by save and by the checker)."""
    vid = vdata["id"]
    previews = (keep_previews_of or {}).get("snapshot", {}).get("previews")
    if not previews:
        pinned = bool((keep_previews_of or {}).get("pinned"))
        previews = _download_previews(vid, _collect_images(vdata), _keep_local(pinned), config.get("preview_width"))
    rec = build_record(vdata, mdata, previews)
    db.upsert_version(rec)
    return db.get_version(vid)


def save_version(version_id=None, model_id=None, pin=False, download=False, progress=None) -> dict:
    if not version_id and not model_id:
        raise ValueError("Need a version_id or model_id")
    mdata = None
    if not version_id:
        mdata = civitai.get_model(model_id)
        versions = mdata.get("modelVersions") or []
        if not versions:
            raise civitai.NotFound("This model has no published versions.")
        version_id = versions[0]["id"]  # newest first
    vdata = civitai.get_version(version_id)
    model_type = ((vdata.get("model") or {}).get("type") or "LORA").upper()
    if model_type not in LORA_TYPES:
        raise ValueError(f"Only LoRAs can be saved (this is a '{model_type}').")
    if mdata is None:
        try:
            mdata = civitai.get_model(vdata["modelId"])
        except civitai.CivitaiError:
            mdata = None  # version data alone is enough to save
    if not pick_file(vdata):
        raise civitai.CivitaiError("Civitai returned no downloadable file for this version (archived or restricted).")
    existing = db.get_version(version_id)
    rec = refresh_from_api(vdata, mdata, keep_previews_of=existing)
    if mdata:
        try:
            store_model_versions(rec["model_id"], mdata)
        except Exception:
            pass
    rec = db.get_version(version_id)
    if existing and existing["status"] != "ok":
        db.add_event(version_id, "restored", "Back on Civitai")
    if pin or download:
        storage.ensure_local(version_id, pin=pin, progress=progress)
        rec = db.get_version(version_id)
    return rec


def store_model_versions(model_id: int, mdata: dict):
    """Remember the model's full version list inside every saved sibling's snapshot."""
    light = _light_model(mdata).get("modelVersions") or []
    for r in db.versions_for_model(model_id):
        rec = db.get_version(r["version_id"])
        snap = rec.get("snapshot") or {}
        model = snap.get("model") or {}
        model["modelVersions"] = light
        if mdata.get("tags"):
            model["tags"] = mdata["tags"]
        snap["model"] = model
        db.set_fields(rec["version_id"], snapshot=snap, **({"tags": tag_names(mdata)} if mdata.get("tags") else {}))
