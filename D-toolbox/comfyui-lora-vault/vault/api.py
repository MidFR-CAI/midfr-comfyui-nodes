"""HTTP API shared by the ComfyUI route table (same-origin, used by the node panel and
vault page) and the small side server (used by the browser extension)."""
import asyncio
import html
import os
import re
import shutil
import threading

from aiohttp import web

from . import checker, civitai, config, db, parse, relay, snapshot, storage


async def run(fn, *a, **kw):
    return await asyncio.get_running_loop().run_in_executor(None, lambda: fn(*a, **kw))


def _err(e: Exception):
    code = 404 if isinstance(e, civitai.NotFound) else 401 if isinstance(e, civitai.AuthRequired) else \
        429 if isinstance(e, civitai.RateLimited) else 400 if isinstance(e, (ValueError, parse.ResolveError)) else 500
    return web.json_response({"ok": False, "error": str(e) or e.__class__.__name__}, status=code)


def _text(h: str) -> str:
    h = re.sub(r"<\s*(br|/p|/li|/h\d)\s*/?>", "\n", h or "", flags=re.I)
    return html.unescape(re.sub(r"<[^>]+>", "", h)).strip()


def _versions_of(rec: dict, model: dict) -> list:
    """Every known version of this model (Civitai's list, plus any saved ones it no longer lists)."""
    vs = {}
    for v in (model.get("modelVersions") or []):
        imgs = v.get("images") or []
        t = next((i for i in imgs if i.get("type", "image") == "image"), None)
        vs[v["id"]] = {"id": v["id"], "name": v.get("name"), "baseModel": v.get("baseModel"),
                       "publishedAt": v.get("publishedAt"), "downloads": (v.get("stats") or {}).get("downloadCount"),
                       "trainedWords": v.get("trainedWords") or [], "size_kb": v.get("size_kb") or 0,
                       "tag_ref": parse.tag_ref(rec.get("model_name"), v.get("name"), v["id"]),
                       "description_html": v.get("description") or "",
                       "thumb": snapshot.thumb_url(t["url"], 300) if t else None, "thumb_nsfw": (t or {}).get("nsfwLevel") or 0,
                       "images": [{"url": i["url"], "type": i.get("type", "image"), "nsfwLevel": i.get("nsfwLevel") or 0,
                                   "meta": i.get("meta"), "src": snapshot.thumb_url(i["url"], 640) if i.get("type", "image") == "image" else i["url"]}
                                  for i in imgs]}
    for r in db.versions_for_model(rec["model_id"]):
        vs.setdefault(r["version_id"], {"id": r["version_id"], "name": r["version_name"],
                                        "baseModel": r["base_model"], "publishedAt": None, "downloads": None,
                                        "trainedWords": r.get("trigger_words") or [],
                                        "tag_ref": parse.tag_ref(r.get("model_name"), r["version_name"], r["version_id"])})
        if r.get("thumb_idx") is not None:
            vs[r["version_id"]].setdefault("thumb", f"/lora_vault/api/preview/{r['version_id']}/{r['thumb_idx']}")
    out = []
    for v in vs.values():
        saved = db.get_version(v["id"])
        v.update(saved=bool(saved), status=saved["status"] if saved else None,
                 pinned=bool(saved and saved["pinned"]), local=storage.local_state(saved) if saved else "none")
        out.append(v)
    return out


def view(rec: dict, detail: bool = False) -> dict:
    if not rec:
        return None
    snap = rec.get("snapshot") or {}
    previews = snap.get("previews") or []
    vid = rec["version_id"]
    out = {k: v for k, v in rec.items() if k not in ("snapshot", "local_path", "download_url")}
    out["local"] = storage.local_state(rec)
    if rec.get("thumb_idx") is not None:
        out["thumb"] = f"/lora_vault/api/preview/{vid}/{rec['thumb_idx']}"
    elif rec.get("thumb_remote"):  # no local copy (download failed): the browser can still load it from Civitai
        out["thumb"] = rec["thumb_remote"]
    else:
        out["thumb"] = None
    out["thumb_video"] = rec.get("thumb_video") or None   # cover is a video: tiles play it silently
    out["thumb_nsfw"] = rec.get("thumb_nsfw") or 0   # level of *this* image, not the model
    out["preview_count"] = rec.get("preview_count") or 0
    out["tag_ref"] = parse.tag_ref(rec.get("model_name"), rec.get("version_name"), vid)
    out["civitai_url"] = f"https://civitai.com/models/{rec['model_id']}?modelVersionId={vid}"
    if detail:
        version, model = snap.get("version") or {}, snap.get("model") or {}
        out["previews"] = [
            {"idx": p["idx"], "type": p["type"], "nsfwLevel": p.get("nsfwLevel", 0), "meta": p.get("meta"),
             "width": p.get("width"), "height": p.get("height"), "full": p["url"], "local": bool(p.get("local")),
             "src": f"/lora_vault/api/preview/{vid}/{p['idx']}" if p.get("local")
             else (snapshot.thumb_url(p["url"], 640) if p["type"] == "image" else p["url"])}
            for p in previews
        ]
        out["stats"] = {**(model.get("stats") or {}), **(version.get("stats") or {})}
        out["description"] = _text(model.get("description") or "")
        out["version_description"] = _text(version.get("description") or "")
        out["description_html"] = model.get("description") or ""          # sanitised client-side
        out["version_description_html"] = version.get("description") or ""
        out["tags"] = model.get("tags") or []
        out["license"] = {k: model.get(k) for k in ("allowNoCredit", "allowCommercialUse", "allowDerivatives") if k in model}
        out["published_at"] = version.get("publishedAt")
        out["availability"] = version.get("availability")
        out["early_access_ends"] = version.get("earlyAccessEndsAt")
        out["file"] = {"name": rec.get("file_name"), "size_kb": rec.get("file_size_kb"), "sha256": rec.get("sha256")}
        out["other_versions"] = out["versions"] = _versions_of(rec, model)
        out["versions_known"] = len(model.get("modelVersions") or []) > 0
    return out


# ---------------------------------------------------------------- handlers
async def ping(request):
    return web.json_response({"ok": True, "version": config.VERSION, "count": len(db.list_versions()),
                              "transport": civitai.transport(), "helper_connected": relay.connected()})


async def bases(request):
    return web.json_response({"bases": await run(civitai.get_base_models)})


# Civitai's category row. A model belongs to a category when it carries that tag.
CATEGORIES = [("character", "Character"), ("style", "Style"), ("concept", "Concept"), ("clothing", "Clothing"),
              ("poses", "Poses"), ("background", "Background"), ("tool", "Tool"), ("vehicle", "Vehicle"),
              ("buildings", "Buildings"), ("objects", "Objects"), ("animal", "Animal"), ("action", "Action"),
              ("assets", "Assets"), ("base model", "Base Model")]


def _norm_tag(t: str) -> str:
    t = re.sub(r"[\s_-]+", " ", (t or "").strip().lower())
    return t[:-1] if t.endswith("s") and not t.endswith("ss") else t


def _cats_of(rec: dict) -> set:
    have = {_norm_tag(t) for t in (rec.get("tags") or [])}
    return {key for key, _ in CATEGORIES if _norm_tag(key) in have}


async def list_(request):
    q = request.query
    recs = db.list_versions(light=True)
    cat_counts = {}
    for key in {k for r in recs for k in _cats_of(r)}:
        cat_counts[key] = len({r["model_id"] for r in recs if key in _cats_of(r)})
    needle = (q.get("q") or "").lower()
    if needle:  # every word must appear somewhere (name, version, creator, nickname, trigger words, tags)
        words = needle.split()
        recs = [r for r in recs if all(w in f"{r['model_name']} {r['version_name']} {r['creator']} {r.get('alias') or ''} {' '.join(r['trigger_words'])} {' '.join(r.get('tags') or [])}".lower() for w in words)]
    if q.get("base"):
        recs = [r for r in recs if r["base_model"] == q["base"]]
    wanted = [c for c in (q.get("cat") or "").split(",") if c]
    if wanted:  # any of the chosen categories (like Civitai's chips)
        recs = [r for r in recs if _cats_of(r) & set(wanted)]
    if q.get("type"):
        recs = [r for r in recs if (r["model_type"] or "").lower() == q["type"].lower()]
    if q.get("status") == "problem":
        recs = [r for r in recs if r["status"] != "ok"]
    if q.get("pinned") == "1":
        recs = [r for r in recs if r["pinned"]]
    return web.json_response({"items": [view(r) for r in recs], "categories": [{"key": k, "label": l, "count": cat_counts.get(k, 0)} for k, l in CATEGORIES],
                              "types": sorted({r["model_type"] for r in db.list_versions(light=True) if r["model_type"]}),
                              "bases": sorted({r["base_model"] for r in db.list_versions(light=True) if r["base_model"]})})


async def version(request):
    rec = db.get_version(int(request.match_info["vid"]))
    if not rec:
        return web.json_response({"ok": False, "error": "not in vault"}, status=404)
    return web.json_response({"ok": True, "item": view(rec, detail=True)})


async def resolve(request):
    body = await request.json()
    items = []
    for s in parse.parse(body.get("text", "")):
        entry = {"ref": s["ref"], "strength": s["strength"], "clip_strength": s["clip_strength"], "raw": s["raw"]}
        try:
            kind, obj = parse.resolve(s["ref"])
            if kind == "vault":
                entry.update(ok=True, kind="vault", item=view(db.get_version(obj["version_id"]), detail=True))
            else:
                entry.update(ok=True, kind="file", name=os.path.basename(obj))
        except parse.ResolveError as e:
            entry.update(ok=False, error=str(e))
        items.append(entry)
    return web.json_response({"items": items})


def _bg(vid: int, fn):
    def work():
        try:
            fn()
        except Exception as e:
            storage.jobs[int(vid)] = {"state": "error", "error": str(e), "done": 0, "total": 0}
    storage.jobs[int(vid)] = {"state": "queued", "done": 0, "total": 0, "error": None}
    threading.Thread(target=work, daemon=True).start()


def _do_save(body: dict) -> int:
    model_id, version_id = body.get("model_id"), body.get("version_id")
    if body.get("url"):
        model_id, version_id = civitai.parse_civitai_url(body["url"])
        if not (model_id or version_id):
            raise ValueError("That doesn't look like a Civitai model link.")
    rec = snapshot.save_version(version_id=version_id, model_id=model_id)
    vid = rec["version_id"]
    if body.get("pin") or body.get("download"):
        pin = bool(body.get("pin"))
        _bg(vid, lambda: storage.ensure_local(vid, pin=pin))
    return vid


save_jobs: dict = {}   # job id -> {"state": running|done|error, "error", "item"}


async def save(request):
    body = await request.json()
    if body.get("async"):
        # Saving can take a while (previews are fetched through Firefox one by one). The browser
        # extension must not hold a message open that long, so it polls /save_job/<id> instead.
        import uuid
        job = uuid.uuid4().hex[:12]
        save_jobs[job] = {"state": "running", "error": None, "item": None}

        def work():
            try:
                vid = _do_save(body)
                save_jobs[job].update(state="done", item=view(db.get_version(vid)))
            except Exception as e:
                save_jobs[job].update(state="error", error=str(e) or e.__class__.__name__)
        threading.Thread(target=work, daemon=True).start()
        return web.json_response({"ok": True, "job": job})
    try:
        vid = await run(_do_save, body)
        return web.json_response({"ok": True, "item": view(db.get_version(vid))})
    except Exception as e:
        return _err(e)


async def save_job(request):
    j = save_jobs.get(request.match_info["job"])
    if not j:
        return web.json_response({"ok": False, "error": "unknown save job"}, status=404)
    return web.json_response({"ok": True, **j})


def _remove_one(vid: int, delete_files: bool):
    """Take one version out of the vault: optionally its downloaded weights, always its stored previews."""
    rec = db.get_version(vid)
    if not rec:
        return False
    if delete_files and storage.local_state(rec) != "none":
        storage.delete_local(vid, True)
    shutil.rmtree(os.path.join(config.previews_dir(), str(vid)), ignore_errors=True)
    db.delete_version(vid)
    return True


async def remove_many(request):
    """Batch remove: {"version_ids": [...]} and/or {"model_ids": [...]} (every saved version of those models), plus delete_files. Returns how many went."""
    body = await request.json()
    ids = [int(x) for x in body.get("version_ids") or []]
    mids = {str(m) for m in (body.get("model_ids") or [])} | ({str(body["model_id"])} if body.get("model_id") else set())
    if mids:
        ids += [r["version_id"] for r in db.list_versions() if str(r.get("model_id")) in mids]
    ids = list(dict.fromkeys(ids))
    delete_files = bool(body.get("delete_files"))

    def work():
        done, errors = 0, []
        for vid in ids:
            try:
                done += bool(_remove_one(vid, delete_files))
            except Exception as e:  # keep going; report at the end
                errors.append(f"{vid}: {e}")
        return done, errors
    done, errors = await run(work)
    return web.json_response({"ok": True, "removed": done, "errors": errors})


def _action(fn_name):
    async def h(request):
        body = await request.json()
        vid = int(body["version_id"])
        try:
            if fn_name == "pin":
                _bg(vid, lambda: storage.pin(vid))
            elif fn_name == "download":
                _bg(vid, lambda: storage.ensure_local(vid))
            elif fn_name == "import_file":
                await run(storage.import_file, vid, body.get("path"), bool(body.get("pin")))
            elif fn_name == "unpin":
                await run(storage.unpin, vid)
            elif fn_name == "delete_local":
                await run(storage.delete_local, vid, bool(body.get("force")))
            elif fn_name == "remove":
                await run(_remove_one, vid, bool(body.get("delete_files")))
            elif fn_name == "update":
                db.set_fields(vid, **{k: body[k] for k in ("alias", "notes") if k in body})
            elif fn_name == "check":
                res = await run(checker.check_version, vid)
                return web.json_response({"ok": True, "result": res, "item": view(db.get_version(vid))})
            return web.json_response({"ok": True, "item": view(db.get_version(vid))})
        except Exception as e:
            return _err(e)
    return h


bulk: dict = {}  # model_id -> progress of a "save all versions" job


async def save_model(request):
    """Save several versions of one model page (a collection page can hold dozens of LoRAs)."""
    body = await request.json()
    model_id = int(body["model_id"])
    ids = [int(i) for i in (body.get("version_ids") or [])]

    def work():
        st = bulk[model_id] = {"done": 0, "total": len(ids), "errors": [], "state": "running"}
        try:
            todo = ids or [v["id"] for v in civitai.get_model(model_id).get("modelVersions", [])]
            st["total"] = len(todo)
            for vid in todo:
                try:
                    snapshot.save_version(version_id=vid)
                except Exception as e:
                    st["errors"].append(f"{vid}: {e}")
                st["done"] += 1
            st["state"] = "done"
        except Exception as e:
            st.update(state="error", error=str(e))

    bulk[model_id] = {"done": 0, "total": len(ids), "errors": [], "state": "running"}
    threading.Thread(target=work, daemon=True).start()
    return web.json_response({"ok": True})


async def refresh_previews(request):
    body = await request.json()
    try:
        rec = await run(snapshot.refresh_previews, int(body["version_id"]))
        return web.json_response({"ok": True, "item": view(rec, detail=True)})
    except Exception as e:
        return _err(e)


async def status(request):
    q = request.query
    ids = [int(x) for x in (q.get("version_ids") or "").split(",") if x.strip().isdigit()]
    mids = [int(x) for x in (q.get("model_ids") or "").split(",") if x.strip().isdigit()]
    return web.json_response({"versions": db.status_for_versions(ids), "models": db.status_for_models(mids)})


async def jobs(request):
    return web.json_response({"jobs": {str(k): v for k, v in storage.jobs.items()},
                              "bulk": {str(k): v for k, v in bulk.items()},
                              "check": {"running": checker.scheduler.running_now, **checker.scheduler.progress}})


async def check_all(request):
    checker.scheduler.trigger_now()
    return web.json_response({"ok": True})


async def test_connection(request):
    return web.json_response(await run(civitai.test_connection))


async def relay_next(request):
    """Long-poll used by the Firefox extension: hands it the next queued request."""
    relay.poller_enter()
    relay.note_version(request.query.get("v"))
    try:
        for _ in range(80):  # ~20 s
            t = relay.claim_next()
            if t:
                return web.json_response({"task": t})
            await asyncio.sleep(0.25)
        return web.json_response({"task": None})
    finally:
        relay.poller_leave()


async def relay_result(request):
    b = await request.json()
    return web.json_response({"ok": relay.complete(b.get("id"), b)})


async def relay_progress(request):
    b = await request.json()
    relay.report_progress(b.get("id"), b.get("received", 0), b.get("total", 0))
    return web.json_response({"ok": True})


async def relay_status(request):
    return web.json_response({**relay.status(), "transport": civitai.transport()})


async def events(request):
    return web.json_response({"items": db.list_events(100)})


async def usage(request):
    return web.json_response(storage.cache_usage())


async def get_settings(request):
    return web.json_response(config.public_settings())


async def post_settings(request):
    body = await request.json()
    if not body.get("civitai_api_key") and not body.get("clear_api_key"):
        body.pop("civitai_api_key", None)  # masked field posted back empty -> keep stored key
    if body.pop("clear_api_key", False):
        body["civitai_api_key"] = ""
    config.update(body)
    return web.json_response(config.public_settings())


async def search(request):
    q = request.query
    try:
        data = await run(civitai.search_models, q.get("q", ""), q.get("base", ""), q.get("sort", "Most Downloaded"),
                         q.get("period", "AllTime"), q.get("nsfw") == "1", q.get("cursor", ""))
    except Exception as e:
        return _err(e)
    base = q.get("base", "")
    items = []
    for m in data.get("items", []):
        vers = m.get("modelVersions") or []
        if base:
            vers = [v for v in vers if v.get("baseModel") == base] or []
        out_v = []
        model_thumb = next((i for v in (m.get("modelVersions") or []) for i in (v.get("images") or [])
                            if i.get("type", "image") == "image"), None)
        for v in vers[:30]:
            img = next((i for i in (v.get("images") or []) if i.get("type", "image") == "image"), None)
            f = snapshot.pick_file(v) or {}
            out_v.append({
                "id": v["id"], "name": v.get("name"), "baseModel": v.get("baseModel"),
                "trainedWords": v.get("trainedWords") or [], "size_kb": f.get("sizeKB"),
                "thumb": snapshot.thumb_url((img or model_thumb)["url"], 300) if (img or model_thumb) else None,
                "thumb_nsfw": (img or model_thumb or {}).get("nsfwLevel", 0),
                "published": v.get("publishedAt"), "downloads": (v.get("stats") or {}).get("downloadCount"),
                "saved": db.get_version(v["id"]) is not None,
            })
        if not out_v:
            continue
        st = m.get("stats") or {}
        items.append({"model_id": m["id"], "name": m.get("name"), "type": m.get("type"),
                      "creator": (m.get("creator") or {}).get("username"), "nsfw": m.get("nsfw"),
                      "downloads": st.get("downloadCount"), "likes": st.get("thumbsUpCount"),
                      "tags": (m.get("tags") or [])[:6], "versions": out_v})
    return web.json_response({"ok": True, "items": items, "next_cursor": (data.get("metadata") or {}).get("nextCursor")})


async def model_versions(request):
    """Fetch the model page's full version list from Civitai (via Firefox) and remember it, so the
    other versions can be browsed - with previews, trigger words and descriptions - without opening Civitai."""
    mid = int(request.match_info["mid"])
    try:
        mdata = await asyncio.get_running_loop().run_in_executor(None, civitai.get_model, mid)
        await asyncio.get_running_loop().run_in_executor(None, snapshot.store_model_versions, mid, mdata)
        recs = db.versions_for_model(mid)
        if not recs:
            return web.json_response({"ok": True, "versions": []})
        rec = db.get_version(recs[0]["version_id"])
        return web.json_response({"ok": True, "versions": view(rec, detail=True)["versions"]})
    except Exception as e:
        return _err(e)


async def preview(request):
    vid, idx = int(request.match_info["vid"]), int(request.match_info["idx"])
    rec = db.get_version(vid)
    p = next((p for p in ((rec or {}).get("snapshot") or {}).get("previews", []) if p["idx"] == idx and p.get("local")), None)
    if not p:
        raise web.HTTPNotFound()
    path = os.path.join(config.previews_dir(), str(vid), os.path.basename(p["local"]))
    if not os.path.isfile(path):
        raise web.HTTPNotFound()
    return web.FileResponse(path, headers={"Cache-Control": "public, max-age=86400"})


def table(prefix: str):
    """[(method, path, handler)] with every path under `prefix`."""
    P = prefix
    return [
        ("GET", f"{P}/ping", ping), ("GET", f"{P}/bases", bases), ("GET", f"{P}/list", list_),
        ("GET", f"{P}/version/{{vid}}", version), ("GET", f"{P}/model_versions/{{mid}}", model_versions), ("POST", f"{P}/resolve", resolve),
        ("POST", f"{P}/save", save), ("GET", f"{P}/save_job/{{job}}", save_job), ("POST", f"{P}/save_model", save_model),
        ("POST", f"{P}/refresh_previews", refresh_previews), ("GET", f"{P}/status", status), ("GET", f"{P}/jobs", jobs),
        ("POST", f"{P}/check_all", check_all), ("POST", f"{P}/remove_many", remove_many), ("GET", f"{P}/events", events), ("GET", f"{P}/relay/next", relay_next),
        ("POST", f"{P}/relay/result", relay_result), ("POST", f"{P}/relay/progress", relay_progress),
        ("GET", f"{P}/relay/status", relay_status), ("POST", f"{P}/test_connection", test_connection), ("GET", f"{P}/usage", usage),
        ("GET", f"{P}/settings", get_settings), ("POST", f"{P}/settings", post_settings),
        ("GET", f"{P}/search", search), ("GET", f"{P}/preview/{{vid}}/{{idx}}", preview),
    ] + [("POST", f"{P}/{n}", _action(n)) for n in ("pin", "unpin", "download", "delete_local", "remove", "update", "check", "import_file")]
