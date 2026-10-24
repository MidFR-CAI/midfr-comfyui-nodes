import importlib
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from fake_civitai import Fake, PAYLOAD  # noqa: E402


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LORA_VAULT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LORA_VAULT_LORA_DIR", str(tmp_path / "loras"))
    fake = Fake().start()
    from vault import config
    config.update({"civitai_hosts": [f"http://127.0.0.1:{fake.api_port}"], "civitai_api_key": "KEY123",
                   "cache_limit_gb": 0, "transport": "direct"})
    return fake, tmp_path


def mods():
    from vault import checker, civitai, db, parse, snapshot, storage
    return checker, civitai, db, parse, snapshot, storage


def test_save_and_snapshot(env):
    fake, _ = env
    checker, civitai, db, parse, snapshot, storage = mods()
    rec = snapshot.save_version(version_id=111)
    assert rec["model_name"] == "Cool Style" and rec["trigger_words"] == ["coolstyle"]
    assert rec["snapshot"]["previews"][0]["local"] == "0.jpeg"
    assert storage.local_state(rec) == "none"  # saved but NOT downloaded
    # saving by model id picks the newest version
    assert snapshot.save_version(model_id=22)["version_id"] == 222


def test_download_on_demand_verifies_hash_and_strips_auth_on_cdn(env):
    fake, tmp = env
    checker, civitai, db, parse, snapshot, storage = mods()
    snapshot.save_version(version_id=111)
    path = storage.ensure_local(111)
    assert open(path, "rb").read() == PAYLOAD
    assert "_vault" in path and os.sep + "cache" + os.sep in path
    assert fake.cdn_auth_seen == [None], "API key must not be sent to the CDN"
    assert storage.ensure_local(111) == path  # second call: no re-download
    assert len(fake.cdn_auth_seen) == 1


def test_download_without_key_gives_clear_error(env):
    fake, _ = env
    checker, civitai, db, parse, snapshot, storage = mods()
    from vault import config
    snapshot.save_version(version_id=111)
    config.update({"civitai_api_key": "wrong"})
    with pytest.raises(civitai.AuthRequired):
        storage.ensure_local(111)


def test_corrupt_download_rejected(env):
    fake, _ = env
    checker, civitai, db, parse, snapshot, storage = mods()
    snapshot.save_version(version_id=111)
    fake.payload = b"tampered" * 100
    with pytest.raises(storage.Unavailable, match="SHA-256"):
        storage.ensure_local(111)
    assert db.get_version(111)["local_path"] is None


def test_resume_partial_download(env):
    fake, tmp = env
    checker, civitai, db, parse, snapshot, storage = mods()
    from vault import config
    snapshot.save_version(version_id=111)
    dest = os.path.join(config.cache_dir(), "111__Cool_Style.safetensors")
    with open(dest + ".part", "wb") as f:
        f.write(PAYLOAD[:1000])
    path = storage.ensure_local(111)
    assert open(path, "rb").read() == PAYLOAD


def test_pin_unpin_and_eviction(env):
    fake, tmp = env
    checker, civitai, db, parse, snapshot, storage = mods()
    from vault import config
    snapshot.save_version(version_id=111)
    snapshot.save_version(version_id=222)
    p1 = storage.ensure_local(111)
    storage.ensure_local(222)
    # make 111 the least recently used
    db.set_fields(111, last_used=1.0)
    config.update({"cache_limit_gb": len(PAYLOAD) * 1.5 / 1024**3})
    removed = storage.evict()
    assert removed == [111] and not os.path.exists(p1)
    # pinned files are never evicted
    storage.pin(111)
    assert db.get_version(111)["pinned"] and os.sep + "pinned" + os.sep in db.get_version(111)["local_path"]
    config.update({"cache_limit_gb": 0.000001})
    storage.evict()
    assert os.path.exists(db.get_version(111)["local_path"])
    storage.unpin(111)
    lp = db.get_version(111)["local_path"]  # unpinned -> back in cache, and the tiny cap evicts it
    assert not db.get_version(111)["pinned"] and (lp is None or os.sep + "cache" + os.sep in lp)


def test_removal_detection_needs_two_misses_and_preserves(env):
    fake, tmp = env
    checker, civitai, db, parse, snapshot, storage = mods()
    snapshot.save_version(version_id=111)
    storage.ensure_local(111)  # cached weights
    assert checker.check_version(111) == "ok"
    fake.hidden.add(111)
    fake.dead_models.add(11)
    assert checker.check_version(111) == "possibly_removed"
    assert db.get_version(111)["pinned"] is False
    assert checker.check_version(111) == "removed"
    rec = db.get_version(111)
    assert rec["status"] == "removed" and rec["pinned"], "cached weights auto-pinned on removal"
    assert rec["snapshot"]["version"]["name"] == "v1.0", "last good snapshot retained"
    assert os.path.exists(os.path.join(os.environ["LORA_VAULT_DATA_DIR"], "backups", "111", "snapshot.json"))
    # removed models: eviction/unpin must protect the last copy
    with pytest.raises(storage.Unavailable):
        storage.unpin(111)
    # comes back -> ok again
    fake.hidden.clear(); fake.dead_models.clear()
    assert checker.check_version(111) == "ok"


def test_archived_flag_trusted_immediately(env):
    fake, _ = env
    checker, civitai, db, parse, snapshot, storage = mods()
    snapshot.save_version(version_id=222)
    fake.versions[222]["model"]["mode"] = "Archived"
    assert checker.check_version(222) == "archived"


def test_network_error_does_not_change_status(env):
    fake, _ = env
    checker, civitai, db, parse, snapshot, storage = mods()
    from vault import config
    snapshot.save_version(version_id=111)
    config.update({"civitai_hosts": ["http://127.0.0.1:1"]})
    assert checker.check_version(111).startswith("unknown")
    assert db.get_version(111)["status"] == "ok"


def test_unavailable_when_removed_and_not_local(env):
    fake, _ = env
    checker, civitai, db, parse, snapshot, storage = mods()
    snapshot.save_version(version_id=111)
    db.set_fields(111, status="removed")
    with pytest.raises(storage.Unavailable, match="removed from Civitai"):
        storage.ensure_local(111)


def test_parse_and_resolve(env):
    checker, civitai, db, parse, snapshot, storage = mods()
    s = parse.parse("hi <lora:v111:0.8> and <lora:C:\\x\\foo.safetensors:1:0.5> <lora:Cool Style>")
    assert [(x["ref"], x["strength"], x["clip_strength"]) for x in s] == [
        ("v111", 0.8, 0.8), ("C:\\x\\foo.safetensors", 1.0, 0.5), ("Cool Style", 1.0, 1.0)]
    snapshot.save_version(version_id=111)
    assert parse.resolve("v111")[1]["version_id"] == 111
    assert parse.resolve("cool style")[1]["version_id"] == 111
    assert parse.resolve("Cool_Style.safetensors")[1]["version_id"] == 111
    with pytest.raises(parse.ResolveError):
        parse.resolve("v999")
    with pytest.raises(parse.ResolveError):
        parse.resolve("../../etc/passwd")


def test_host_fallback_only_404_if_all_hosts_404(env):
    fake, _ = env
    checker, civitai, db, parse, snapshot, storage = mods()
    from vault import config
    config.update({"civitai_hosts": ["http://127.0.0.1:1", f"http://127.0.0.1:{fake.api_port}"]})
    with pytest.raises(civitai.NetworkError):  # dead first host is an error, not a 404
        civitai.get_version(111)
    config.update({"civitai_hosts": [f"http://127.0.0.1:{fake.api_port}"]})
    with pytest.raises(civitai.NotFound):
        civitai.get_version(99999)


def test_search_and_bases(env):
    fake, _ = env
    checker, civitai, db, parse, snapshot, storage = mods()
    assert "Pony" in civitai.get_base_models()
    r = civitai.search_models(query="pony", base_model="Pony")
    assert [m["name"] for m in r["items"]] == ["Pony Thing"]
    assert civitai.search_models(base_model="SDXL 1.0")["items"][0]["name"] == "Cool Style"


def test_parse_url():
    from vault import civitai
    assert civitai.parse_civitai_url("https://civitai.com/models/123/foo?modelVersionId=456") == (123, 456)
    assert civitai.parse_civitai_url("https://civitai.red/models/123") == (123, None)
    assert civitai.parse_civitai_url("789") == (None, 789)


# ---------------------------------------------------------------- previews, versions, detail
def test_mature_previews_come_from_images_endpoint_with_meta(env):
    fake, _ = env
    checker, civitai, db, parse, snapshot, storage = mods()
    rec = snapshot.save_version(version_id=333)       # version's own image list is empty
    pv = rec["snapshot"]["previews"]
    assert [p["nsfwLevel"] for p in pv] == [8, 4]
    assert [bool(p["local"]) for p in pv] == [True, False], "unpinned: one local preview, text metadata for all"
    assert pv[0]["meta"]["prompt"] == "mature one" and pv[0]["meta"]["sampler"] == "Euler a"
    assert rec["thumb_idx"] == 0 and rec["thumb_nsfw"] == 8 and rec["preview_count"] == 2


def test_gallery_images_get_meta_filled_in_from_images_endpoint(env):
    fake, _ = env
    checker, civitai, db, parse, snapshot, storage = mods()
    # the version's own image lacks meta -> the images endpoint copy supplies it
    fake.versions[111]["images"][0]["meta"] = None
    fake.extra_images[111] = [{"url": "http://127.0.0.1:{API}/img/width=450/111.jpeg", "type": "image", "browsingLevel": 1,
                               "meta": {"prompt": "filled in"}}]
    rec = snapshot.save_version(version_id=111)
    assert len(rec["snapshot"]["previews"]) == 1 and rec["snapshot"]["previews"][0]["meta"] == {"prompt": "filled in"}


def test_refresh_previews_repairs_entry_with_no_previews(env):
    fake, _ = env
    checker, civitai, db, parse, snapshot, storage = mods()
    saved = fake.extra_images.pop(333)
    rec = snapshot.save_version(version_id=333)
    assert rec["preview_count"] == 0 and rec["thumb_idx"] is None
    fake.extra_images[333] = saved                    # e.g. user turned on mature content / switched host
    rec = snapshot.refresh_previews(333)
    assert rec["preview_count"] == 2 and rec["thumb_idx"] == 0


def test_missing_local_preview_falls_back_to_web_thumbnail(env, monkeypatch):
    fake, _ = env
    checker, civitai, db, parse, snapshot, storage = mods()
    from vault import api
    monkeypatch.setattr(civitai, "fetch_bytes", lambda *a, **k: (_ for _ in ()).throw(civitai.NetworkError("x")))
    rec = snapshot.save_version(version_id=111)
    v = api.view(db.get_version(111))
    assert rec["snapshot"]["previews"][0]["local"] is None and v["thumb"].startswith("http") and "width=300" in v["thumb"]


def test_list_view_has_thumbs_without_loading_snapshots_and_old_rows_are_backfilled(env):
    fake, _ = env
    checker, civitai, db, parse, snapshot, storage = mods()
    from vault import api
    snapshot.save_version(version_id=111)
    # simulate a row saved by the previous version (no thumb columns filled in)
    db.set_fields(111, thumb_idx=None, thumb_remote=None, thumb_nsfw=0, preview_count=0, thumb_video=None)
    assert api.view(db.list_versions(light=True)[0])["thumb"] is None
    db.backfill_thumbs()
    item = api.view(db.list_versions(light=True)[0])
    assert item["thumb"] == "/lora_vault/api/preview/111/0" and item["preview_count"] == 1


def test_detail_view_has_versions_meta_and_html(env):
    fake, _ = env
    checker, civitai, db, parse, snapshot, storage = mods()
    from vault import api
    snapshot.save_version(version_id=333); snapshot.save_version(version_id=331)
    d = api.view(db.get_version(333), detail=True)
    assert [v["id"] for v in d["versions"]] == [333, 332, 331]
    saved = {v["id"]: v["saved"] for v in d["versions"]}
    assert saved == {333: True, 332: False, 331: True}
    assert d["previews"][0]["meta"]["steps"] == 28 and d["previews"][0]["full"].endswith("m1.jpeg")
    assert "Model <b>desc</b>" in d["description_html"] and d["description"].startswith("Model desc")
    assert d["file"]["name"] == "Collection_Pack.safetensors"


def test_save_every_version_of_a_collection(env):
    fake, _ = env
    checker, civitai, db, parse, snapshot, storage = mods()
    from vault import api
    import asyncio, time

    class Req:
        async def json(self): return {"model_id": 33}
    asyncio.run(api.save_model(Req()))
    for _ in range(100):
        if api.bulk.get(33, {}).get("state") == "done": break
        time.sleep(0.1)
    assert api.bulk[33]["done"] == 3 and not api.bulk[33]["errors"]
    assert {r["version_id"] for r in db.versions_for_model(33)} == {331, 332, 333}
    # the node can use several versions from the same model page together
    assert parse.resolve("v331")[1]["version_name"] == "Alpha" and parse.resolve("v333")[1]["version_name"] == "Gamma (mature)"
    with pytest.raises(parse.ResolveError, match="several"):
        parse.resolve("Collection Pack")


def test_blur_setting_defaults_on_and_persists(env):
    from vault import config
    assert config.public_settings()["blur_mature"] is True
    config.update({"blur_mature": False})
    assert config.public_settings()["blur_mature"] is False


# ---------------------------------------------------------------- two-tier preview/metadata policy
def test_pin_keeps_everything_locally_and_unpin_trims_back(env):
    fake, tmp = env
    checker, civitai, db, parse, snapshot, storage = mods()
    from vault import config
    snapshot.save_version(version_id=333)
    pdir = os.path.join(config.previews_dir(), "333")
    assert sorted(os.listdir(pdir)) == ["0.jpeg"]                      # unpinned: 1 preview file
    path = storage.pin(333)
    rec = db.get_version(333)
    assert all(p["local"] for p in rec["snapshot"]["previews"]) and sorted(os.listdir(pdir)) == ["0.jpeg", "1.jpeg"]
    base = os.path.splitext(path)[0]                                    # self-contained export beside the weights
    import json
    side = json.load(open(base + ".vault.json"))
    assert side["record"]["snapshot"]["previews"][0]["meta"]["prompt"] == "mature one"
    assert "Trigger words: gamma" in open(base + ".txt").read() and "prompt: mature one" in open(base + ".txt").read()
    assert sorted(os.listdir(base + ".previews")) == ["0.jpeg", "1.jpeg"]
    storage.unpin(333)
    rec = db.get_version(333)
    assert [bool(p["local"]) for p in rec["snapshot"]["previews"]] == [True, False]
    assert rec["snapshot"]["previews"][1]["meta"] is None and rec["snapshot"]["previews"][0]["meta"]["steps"] == 28, "text metadata kept"
    assert not os.path.exists(base + ".vault.json") and not os.path.exists(base + ".previews")


def test_removal_autopin_never_destroys_existing_previews(env):
    fake, tmp = env
    checker, civitai, db, parse, snapshot, storage = mods()
    from vault import config
    snapshot.save_version(version_id=111); storage.ensure_local(111)
    assert db.get_version(111)["snapshot"]["previews"][0]["local"]
    fake.hidden.add(111); fake.dead_models.add(11)
    checker.check_version(111); checker.check_version(111)               # -> removed -> auto-pin -> policy runs
    rec = db.get_version(111)
    assert rec["status"] == "removed" and rec["pinned"]
    assert rec["snapshot"]["previews"][0]["local"] and os.path.isfile(os.path.join(config.previews_dir(), "111", "0.jpeg"))
    assert os.path.isfile(os.path.splitext(rec["local_path"])[0] + ".vault.json")


def test_refresh_keeps_previously_known_images_that_civitai_stops_listing(env):
    fake, _ = env
    checker, civitai, db, parse, snapshot, storage = mods()
    snapshot.save_version(version_id=333)
    fake.extra_images[333] = fake.extra_images[333][:1]                  # Civitai now lists only one
    rec = snapshot.refresh_previews(333)
    assert len(rec["snapshot"]["previews"]) == 2 and rec["snapshot"]["previews"][1]["nsfwLevel"] == 4


def test_model_versions_endpoint_lists_unsaved_siblings_with_details(env):
    import asyncio
    fake, _ = env
    checker, civitai, db, parse, snapshot, storage = mods()
    from vault import api
    snapshot.save_version(version_id=332)           # only Beta of the 3-version collection is saved
    class R:  match_info = {"mid": "33"}
    resp = asyncio.run(api.model_versions(R()))
    import json
    vs = {v["id"]: v for v in json.loads(resp.body)["versions"]}
    assert set(vs) == {331, 332, 333}
    assert vs[331]["saved"] is False and vs[331]["trainedWords"] == ["alpha"] and vs[331]["thumb"]
    assert vs[332]["saved"] is True
    # and it is remembered: a plain detail view now lists them with no network
    again = api.view(db.get_version(332), detail=True)
    assert {v["id"] for v in again["versions"]} == {331, 332, 333} and again["versions_known"]


def test_unpinned_keeps_one_local_image_even_when_cover_is_a_video(env):
    fake, _ = env
    checker, civitai, db, parse, snapshot, storage = mods()
    fake.versions[111]["images"] = [
        {"url": "http://127.0.0.1:{API}/img/width=450/v.mp4", "type": "video", "nsfwLevel": 1, "meta": None},
        {"url": "http://127.0.0.1:{API}/img/width=450/a.jpeg", "type": "image", "nsfwLevel": 1, "meta": {"prompt": "x"}}]
    rec = snapshot.save_version(version_id=111)
    assert rec["thumb_video"].endswith("v.mp4") and rec["thumb_idx"] == 1


def test_category_filter_uses_civitai_tags(env):
    import asyncio, json
    fake, _ = env
    checker, civitai, db, parse, snapshot, storage = mods()
    from vault import api
    fake.models[11]["tags"] = ["Character", "anime"]
    fake.models[22]["tags"] = ["style"]
    snapshot.save_version(version_id=111); snapshot.save_version(version_id=222)
    class R:
        def __init__(self, **q): self.query = q
    out = json.loads(asyncio.run(api.list_(R(cat="character"))).body)
    assert [i["version_id"] for i in out["items"]] == [111]
    cats = {c["key"]: c["count"] for c in out["categories"]}
    assert cats["character"] == 1 and cats["style"] == 1 and cats["tool"] == 0
    both = json.loads(asyncio.run(api.list_(R(cat="character,style"))).body)
    assert len(both["items"]) == 2
    # old rows without the column get it backfilled from the snapshot
    with db.connect() as c:
        c.execute("UPDATE versions SET tags=NULL WHERE version_id=111")
    db.backfill_tags()
    assert "character" in db.get_version(111)["tags"]


def test_files_get_readable_names_and_old_ones_are_renamed(env):
    import os
    fake, tmp = env
    checker, civitai, db, parse, snapshot, storage = mods()
    snapshot.save_version(version_id=111)
    rec = db.get_version(111)
    assert storage.display_filename(rec) == "Cool Style - v1.0.safetensors"
    assert parse.tag_ref("Cool Style", "v1.0", 111) == "Cool Style - v1.0"
    assert parse.tag_ref("Odd: <Name>", "2", 5) == "Odd Name - 2"
    assert parse.tag_ref("1234", "", 9) == "v9"
    # a file saved by the old scheme gets renamed (with its sidecar) and still resolves by name
    old = os.path.join(config_cache(), "111__Cool_Style.safetensors")
    os.makedirs(os.path.dirname(old), exist_ok=True)
    open(old, "wb").write(b"x"); open(os.path.splitext(old)[0] + ".txt", "w").write("t")
    db.set_fields(111, local_path=old)
    storage.reconcile()
    new = db.get_version(111)["local_path"]
    assert os.path.basename(new) == "Cool Style - v1.0.safetensors" and os.path.isfile(new) and not os.path.exists(old)
    assert os.path.isfile(os.path.splitext(new)[0] + ".txt")
    assert parse.resolve("Cool Style - v1.0")[1]["version_id"] == 111


def config_cache():
    from vault import config
    return config.cache_dir()


def test_import_file_checks_hash_and_adopts_it(env):
    import os
    fake, tmp = env
    checker, civitai, db, parse, snapshot, storage = mods()
    from fake_civitai import PAYLOAD
    snapshot.save_version(version_id=111)
    good = os.path.join(str(tmp), "dl.safetensors"); open(good, "wb").write(PAYLOAD)
    bad = os.path.join(str(tmp), "bad.safetensors"); open(bad, "wb").write(b"nope")
    import pytest
    with pytest.raises(storage.Unavailable, match="SHA-256"):
        storage.import_file(111, bad)
    p = storage.import_file(111, good)
    assert os.path.basename(p) == "Cool Style - v1.0.safetensors" and storage.local_state(db.get_version(111)) == "cached"


def test_async_save_job_reports_progress_and_errors(env):
    import asyncio, json, time
    fake, _ = env
    checker, civitai, db, parse, snapshot, storage = mods()
    from vault import api

    class Req:
        def __init__(self, body=None, **mi):
            self._b, self.match_info = body, mi
        async def json(self): return self._b

    def run_job(body):
        r = json.loads(asyncio.run(api.save(Req(body))).body)
        assert r["ok"] and r["job"]
        for _ in range(100):
            j = json.loads(asyncio.run(api.save_job(Req(job=r["job"]))).body)
            if j["state"] != "running":
                return j
            time.sleep(0.1)
    ok = run_job({"version_id": 111, "async": True})
    assert ok["state"] == "done" and ok["item"]["version_id"] == 111
    fake.hidden.add(222)
    bad = run_job({"version_id": 222, "async": True})
    assert bad["state"] == "error" and bad["error"]


def test_remove_many_by_model_and_ids(env):
    import asyncio, json, os
    fake, tmp = env
    checker, civitai, db, parse, snapshot, storage = mods()
    from vault import api, config
    from fake_civitai import PAYLOAD
    snapshot.save_version(version_id=111)
    snapshot.save_version(version_id=222)
    mid = db.get_version(111)["model_id"]
    pdir = os.path.join(config.previews_dir(), "111")
    assert os.path.isdir(pdir)

    class Req:
        def __init__(self, b): self._b = b
        async def json(self): return self._b
    r = asyncio.run(api.remove_many(Req({"model_ids": [mid]})))
    out = json.loads(r.body)
    assert out["ok"] and out["removed"] >= 1 and not out["errors"]
    assert db.get_version(111) is None and not os.path.exists(pdir), "previews must go too"
    r = asyncio.run(api.remove_many(Req({"version_ids": [222, 999999]})))
    assert json.loads(r.body)["removed"] <= 1 and db.get_version(222) is None
