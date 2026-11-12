"""Keeps private Civitai collections '<Username>-Vault - <Base Model>' (one per base model, like the local
folders) in step with what is pinned in the vault, so they are a backup a fresh install can restore.

  pinned here, not in the collection -> added to the collection
  in a collection, not here          -> saved + pinned here (new install, or pinned on another PC)
  unpinned/removed here since the last sync -> removed from the collection

The single '<Username>-Vault' collection of earlier versions is only read (to restore from) and emptied.
"""
import json
import threading

from . import civitai, db, snapshot, storage

state = {"running": False, "done": 0, "total": 0, "errors": [], "username": None, "summary": ""}
_guard = threading.Lock()


def _fail(what, e):
    state["errors"].append(f"{what}: {e}")


def sync():
    """Returns False if a sync is already running."""
    if not _guard.acquire(blocking=False):
        return False
    try:
        state.update(running=True, done=0, total=0, errors=[], username=None, summary="")
        user = civitai.get_session_user()
        prefix = user["username"] + "-Vault"
        state["username"] = user["username"]
        recs = [r for r in db.list_versions() if r["pinned"]]
        pinned = {(r["base_model"] or "Other", r["model_id"]) for r in recs}
        live = {(r["base_model"] or "Other", r["model_id"]) for r in recs if r["status"] == "ok"}  # removed models can't be collected
        pinned_models = {m for _, m in pinned}
        stored = json.loads(db.get_meta("collection_synced", "[]"))
        previous = {tuple(x) for x in stored if isinstance(x, list)}  # the old single-collection format held bare ids

        cids, remote, legacy_cid, legacy = {}, set(), None, set()
        mine = civitai.get_user_collections(user["id"])
        for c in mine:
            if c["name"] == prefix:
                legacy_cid, legacy = c["id"], set(civitai.collection_model_ids(c["id"]))
            elif c["name"].startswith(prefix + " - "):
                base = c["name"][len(prefix) + 3:]
                cids[base] = c["id"]
                remote |= {(base, m) for m in civitai.collection_model_ids(c["id"])}

        to_add, to_remove = live - remote, previous & remote - pinned
        to_restore = {m for _, m in remote - pinned - previous} | (legacy - pinned_models)
        state["total"] = len(to_add) + len(to_remove) + len(to_restore)
        synced = remote & pinned

        for base, mid in to_add:
            try:
                if base not in cids:
                    cids[base] = civitai.create_collection(f"{prefix} - {base}")
                civitai.add_to_collection(cids[base], mid)
                remote.add((base, mid))
                synced.add((base, mid))
            except civitai.CivitaiError as e:
                _fail(f"add {base} {mid}", e)
            state["done"] += 1
        for base, mid in to_remove:
            try:
                civitai.remove_from_collection(cids[base], mid)
            except civitai.CivitaiError as e:
                _fail(f"remove {base} {mid}", e)
                synced.add((base, mid))  # still in the collection: retry next time
            state["done"] += 1
        for mid in to_restore:
            try:
                for r in db.versions_for_model(mid) or [snapshot.save_version(model_id=mid)]:
                    storage.pin(r["version_id"])
                    synced.add((r["base_model"] or "Other", mid))
            except Exception as e:
                _fail(f"restore {mid}", e)
            state["done"] += 1
        for mid in legacy & {m for _, m in remote}:  # now covered by a per-base collection
            try:
                civitai.remove_from_collection(legacy_cid, mid)
            except civitai.CivitaiError as e:
                _fail(f"clear old collection {mid}", e)
        state["summary"] = (f"{len(pinned)} pinned, {len(remote)} in {len(cids)} collections ({len(mine)} of yours found on Civitai); "
                            f"added {len(to_add)}, removed {len(to_remove)}, restored {len(to_restore)}")
        db.set_meta("collection_synced", json.dumps(sorted(synced)))
        return True
    finally:
        state["running"] = False
        _guard.release()
