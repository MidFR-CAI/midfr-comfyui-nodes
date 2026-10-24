"""Detect saved LoRAs that were deleted/archived/taken down on Civitai.

A single "not found" is only a *warning* (`possibly_removed`): Civitai hides mature
models from some hosts/regions, and the API can hiccup. Only a second consecutive
miss, on a later run, marks a version `removed`. Explicit signals from Civitai itself
(`mode: Archived` / `TakenDown`) are trusted immediately.

When a version goes bad we (a) keep its last good snapshot + previews, (b) copy the
snapshot JSON into backups/, and (c) auto-pin the cached weights so they can't be
evicted - the weights are the only part you cannot get back.
"""
import json
import os
import threading
import time

from . import civitai, config, db, snapshot, storage

BAD = {"possibly_removed", "removed", "archived", "taken_down", "unpublished"}


def _set_status(rec: dict, status: str, detail: str, miss_count: int):
    vid, old = rec["version_id"], rec["status"]
    fields = dict(status=status, status_detail=detail, miss_count=miss_count, last_checked=time.time())
    if status == "ok":
        fields.update(last_ok=time.time(), removed_at=None)
    elif status != "possibly_removed":
        fields["removed_at"] = rec.get("removed_at") or time.time()
    db.set_fields(vid, **fields)
    if status != old:
        db.add_event(vid, status, f"{rec['model_name']} / {rec['version_name']}: {old} -> {status}. {detail}")
        if status in ("removed", "archived", "taken_down", "unpublished"):
            _preserve(vid)


def _preserve(vid: int):
    rec = db.get_version(vid)
    d = os.path.join(config.backups_dir(), str(vid))
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    if config.get("auto_pin_on_removal") and storage.local_state(rec) == "cached":
        try:
            storage.pin(vid)
            db.add_event(vid, "auto_pinned", "Weights were cached locally - pinned so they can't be evicted.")
        except Exception as e:
            db.add_event(vid, "auto_pin_failed", str(e))


def check_version(vid: int) -> str:
    rec = db.get_version(vid)
    if not rec:
        return "missing"
    try:
        vdata = civitai.get_version(vid)
    except civitai.NotFound:
        return _handle_not_found(rec)
    except civitai.CivitaiError as e:  # network, rate limit, auth: say nothing about the model
        db.set_fields(vid, last_checked=time.time())
        return "unknown: " + str(e)

    mode = (vdata.get("model") or {}).get("mode")
    vstatus = vdata.get("status")
    if mode in ("Archived", "TakenDown"):
        status = "archived" if mode == "Archived" else "taken_down"
        _set_status(rec, status, f"Civitai marked the model {mode}.", 0)
        return status
    if vstatus and vstatus != "Published":
        _set_status(rec, "unpublished", f"Version status is '{vstatus}'.", 0)
        return "unpublished"
    # Healthy: refresh metadata (stats, trigger words, description) but keep our previews.
    mdata = None
    try:
        mdata = civitai.get_model(vdata["modelId"])
    except civitai.CivitaiError:
        pass
    if snapshot.pick_file(vdata):
        snapshot.refresh_from_api(vdata, mdata, keep_previews_of=rec)
    _set_status(db.get_version(vid), "ok", "", 0)
    return "ok"


def _handle_not_found(rec: dict) -> str:
    misses = (rec.get("miss_count") or 0) + 1
    # Is the whole model gone, or just this version?
    scope = "version"
    try:
        civitai.get_model(rec["model_id"])
    except civitai.NotFound:
        scope = "model"
    except civitai.CivitaiError:
        pass
    detail = f"Civitai returned 404 for the {scope} (miss #{misses})."
    if misses >= 2:
        _set_status(rec, "removed", detail, misses)
        return "removed"
    _set_status(rec, "possibly_removed", detail + " Will confirm on the next check.", misses)
    return "possibly_removed"


class Scheduler(threading.Thread):
    """Checks every saved version about once per `check_interval_hours` (slowly, politely)."""

    def __init__(self):
        super().__init__(daemon=True, name="lora-vault-checker")
        self._wake = threading.Event()
        self.running_now = False
        self.progress = {"done": 0, "total": 0}

    def trigger_now(self):
        self._wake.set()

    def run(self):
        time.sleep(60)  # let ComfyUI finish starting
        while True:
            try:
                interval = config.get("check_interval_hours") * 3600
                last = float(db.get_meta("last_full_check", 0) or 0)
                if self._wake.is_set() or (interval > 0 and time.time() - last >= interval):
                    self._wake.clear()
                    self.check_all()
            except Exception:
                pass
            self._wake.wait(timeout=1800)

    def check_all(self):
        if not civitai.available():  # e.g. Firefox helper not running: try again later, don't log noise
            return
        self.running_now = True
        try:
            ids = [r["version_id"] for r in db.list_versions(light=True)]
            self.progress = {"done": 0, "total": len(ids)}
            for vid in ids:
                check_version(vid)
                self.progress["done"] += 1
                time.sleep(1.5)
            db.set_meta("last_full_check", time.time())
        finally:
            self.running_now = False


scheduler = Scheduler()
