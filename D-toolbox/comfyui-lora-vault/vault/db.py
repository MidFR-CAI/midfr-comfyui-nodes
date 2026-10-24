"""SQLite store. One short-lived connection per call (WAL) so ComfyUI's worker
thread, our background checker and the web handlers never share a connection."""
import json
import os
import sqlite3
import time

from . import config

JSON_COLS = ("trigger_words", "snapshot", "tags")

SCHEMA = """
CREATE TABLE IF NOT EXISTS versions (
  version_id INTEGER PRIMARY KEY,
  model_id INTEGER NOT NULL,
  model_name TEXT, version_name TEXT, model_type TEXT, base_model TEXT, creator TEXT,
  file_name TEXT, file_size_kb REAL, sha256 TEXT, download_url TEXT,
  trigger_words TEXT DEFAULT '[]',
  nsfw_level INTEGER DEFAULT 0,
  status TEXT DEFAULT 'ok', status_detail TEXT, miss_count INTEGER DEFAULT 0,
  pinned INTEGER DEFAULT 0, local_path TEXT, last_used REAL,
  alias TEXT, notes TEXT,
  saved_at REAL, last_checked REAL, last_ok REAL, removed_at REAL,
  snapshot TEXT,
  thumb_idx INTEGER, thumb_remote TEXT, thumb_nsfw INTEGER DEFAULT 0, preview_count INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_versions_model ON versions(model_id);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL, version_id INTEGER, kind TEXT, detail TEXT
);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
"""


_migrated: set = set()
NEW_COLS = {"thumb_idx": "INTEGER", "thumb_remote": "TEXT", "thumb_nsfw": "INTEGER DEFAULT 0",
            "preview_count": "INTEGER DEFAULT 0", "thumb_video": "TEXT", "tags": "TEXT"}


def connect() -> sqlite3.Connection:
    path = os.path.join(config.data_dir(), "vault.db")
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    if path not in _migrated:  # older databases: add columns introduced later
        have = {r["name"] for r in conn.execute("PRAGMA table_info(versions)")}
        for col, typ in NEW_COLS.items():
            if col not in have:
                conn.execute(f"ALTER TABLE versions ADD COLUMN {col} {typ}")
        conn.commit()
        _migrated.add(path)
    return conn


def backfill_thumbs():
    """Fill thumbnail columns for rows saved before they existed (reads each snapshot once)."""
    from . import snapshot
    with connect() as c:
        rows = c.execute("SELECT version_id, snapshot FROM versions "
                         "WHERE thumb_video IS NULL AND snapshot IS NOT NULL").fetchall()
        for r in rows:
            try:
                previews = (json.loads(r["snapshot"]) or {}).get("previews") or []
            except Exception:
                continue
            f = snapshot.thumb_fields(previews)
            c.execute("UPDATE versions SET thumb_idx=?, thumb_remote=?, thumb_nsfw=?, preview_count=?, thumb_video=? WHERE version_id=?",
                      (f["thumb_idx"], f["thumb_remote"], f["thumb_nsfw"], f["preview_count"], f["thumb_video"], r["version_id"]))


def backfill_tags():
    """Older rows: copy the model's Civitai tags out of the snapshot so the library can filter by category."""
    from . import snapshot
    with connect() as c:
        rows = c.execute("SELECT version_id, snapshot FROM versions WHERE tags IS NULL AND snapshot IS NOT NULL").fetchall()
        for r in rows:
            try:
                model = (json.loads(r["snapshot"]) or {}).get("model") or {}
            except Exception:
                model = {}
            c.execute("UPDATE versions SET tags=? WHERE version_id=?",
                      (json.dumps(snapshot.tag_names(model)), r["version_id"]))


def _row(r):
    if r is None:
        return None
    d = dict(r)
    for c in JSON_COLS:
        if c in d:
            try:
                d[c] = json.loads(d[c]) if d[c] else ([] if c in ("trigger_words", "tags") else {})
            except Exception:
                d[c] = [] if c in ("trigger_words", "tags") else {}
    d["pinned"] = bool(d.get("pinned"))
    return d


def get_version(vid: int):
    with connect() as c:
        return _row(c.execute("SELECT * FROM versions WHERE version_id=?", (int(vid),)).fetchone())


def list_versions(light: bool = True):
    cols = "*" if not light else (
        "version_id,model_id,model_name,version_name,model_type,base_model,creator,file_name,"
        "file_size_kb,trigger_words,nsfw_level,status,status_detail,pinned,local_path,last_used,"
        "alias,saved_at,last_checked,removed_at,thumb_idx,thumb_remote,thumb_nsfw,preview_count,thumb_video,tags"
    )
    with connect() as c:
        rows = c.execute(f"SELECT {cols} FROM versions ORDER BY saved_at DESC").fetchall()
    return [_row(r) for r in rows]


def versions_for_model(model_id: int):
    with connect() as c:
        rows = c.execute("SELECT * FROM versions WHERE model_id=?", (int(model_id),)).fetchall()
    return [_row(r) for r in rows]


def upsert_version(rec: dict):
    """Insert or refresh a version. User-owned fields (pinned, local_path, alias,
    notes, last_used, saved_at) are never overwritten by a refresh."""
    rec = dict(rec)
    for c in JSON_COLS:
        if c in rec and not isinstance(rec[c], str):
            rec[c] = json.dumps(rec[c], ensure_ascii=False)
    now = time.time()
    with connect() as c:
        cur = c.execute("SELECT version_id FROM versions WHERE version_id=?", (rec["version_id"],)).fetchone()
        if cur is None:
            rec.setdefault("saved_at", now)
            cols = ",".join(rec)
            qs = ",".join("?" * len(rec))
            c.execute(f"INSERT INTO versions ({cols}) VALUES ({qs})", list(rec.values()))
        else:
            protected = {"pinned", "local_path", "alias", "notes", "last_used", "saved_at", "version_id"}
            sets = {k: v for k, v in rec.items() if k not in protected}
            if sets:
                c.execute(
                    f"UPDATE versions SET {','.join(k + '=?' for k in sets)} WHERE version_id=?",
                    list(sets.values()) + [rec["version_id"]],
                )


def set_fields(vid: int, **fields):
    if not fields:
        return
    for c in JSON_COLS:
        if c in fields and not isinstance(fields[c], str):
            fields[c] = json.dumps(fields[c], ensure_ascii=False)
    with connect() as c:
        c.execute(
            f"UPDATE versions SET {','.join(k + '=?' for k in fields)} WHERE version_id=?",
            list(fields.values()) + [int(vid)],
        )


def delete_version(vid: int):
    with connect() as c:
        c.execute("DELETE FROM versions WHERE version_id=?", (int(vid),))


def add_event(vid, kind: str, detail: str = ""):
    with connect() as c:
        c.execute(
            "INSERT INTO events (ts, version_id, kind, detail) VALUES (?,?,?,?)",
            (time.time(), vid, kind, detail),
        )


def list_events(limit: int = 100):
    with connect() as c:
        rows = c.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_meta(k: str, default=None):
    with connect() as c:
        r = c.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
    return r["v"] if r else default


def set_meta(k: str, v):
    with connect() as c:
        c.execute("INSERT INTO meta (k,v) VALUES (?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, str(v)))


def status_for_versions(ids):
    ids = [int(i) for i in ids]
    if not ids:
        return {}
    with connect() as c:
        rows = c.execute(
            f"SELECT version_id,status,pinned,local_path FROM versions WHERE version_id IN ({','.join('?' * len(ids))})",
            ids,
        ).fetchall()
    return {r["version_id"]: _status_view(r) for r in rows}


def status_for_models(ids):
    ids = [int(i) for i in ids]
    if not ids:
        return {}
    with connect() as c:
        rows = c.execute(
            f"SELECT model_id,version_id,status,pinned,local_path FROM versions WHERE model_id IN ({','.join('?' * len(ids))})",
            ids,
        ).fetchall()
    out = {}
    for r in rows:
        cur = out.setdefault(r["model_id"], {"versions": [], "status": "ok", "pinned": False})
        cur["versions"].append(r["version_id"])
        cur["pinned"] = cur["pinned"] or bool(r["pinned"])
        if r["status"] != "ok":
            cur["status"] = r["status"]
    return out


def _status_view(r):
    return {
        "saved": True,
        "status": r["status"],
        "pinned": bool(r["pinned"]),
        "local": bool(r["local_path"] and os.path.exists(r["local_path"])),
    }
