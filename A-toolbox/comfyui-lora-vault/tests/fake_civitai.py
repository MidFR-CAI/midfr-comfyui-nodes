"""A fake Civitai (API on 127.0.0.1:A, file CDN on localhost:B) for offline tests."""
import asyncio
import hashlib
import threading

from aiohttp import web

import base64
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
PAYLOAD = b"fake-safetensors-" * 5000


def sha(b=PAYLOAD):
    return hashlib.sha256(b).hexdigest().upper()


class Fake:
    def __init__(self):
        self.versions = {}      # id -> version dict
        self.models = {}        # id -> model dict
        self.hidden = set()     # version ids that 404
        self.dead_models = set()
        self.extra_images = {}   # version id -> items served by /api/v1/images only
        self.cdn_auth_seen = []  # Authorization headers the CDN received
        self.api_auth_seen = []
        self.payload = PAYLOAD
        self.api_port = self.cdn_port = None
        self.add(111, 11, "Cool Style", "v1.0", "SDXL 1.0", words=["coolstyle"])
        self.add(222, 22, "Pony Thing", "v2", "Pony", words=["pt"])
        # a "collection" page: three LoRAs under one model, the newest is mature
        self.add(331, 33, "Collection Pack", "Alpha", "SDXL 1.0", words=["alpha"])
        self.add(332, 33, "Collection Pack", "Beta", "Pony", words=["beta"])
        self.add(333, 33, "Collection Pack", "Gamma (mature)", "Pony", words=["gamma"])
        # civitai.com hides mature images from the version's own list; only the images endpoint has them
        self.versions[333]["images"] = []
        self.extra_images = {333: [
            {"url": "http://127.0.0.1:{API}/img/width=450/m1.jpeg", "type": "image", "browsingLevel": 8, "nsfwLevel": "X",
             "width": 450, "height": 600, "meta": {"prompt": "mature one", "negativePrompt": "bad", "steps": 28,
                                                    "sampler": "Euler a", "cfgScale": 6.5, "seed": 42, "Size": "832x1216"}},
            {"url": "http://127.0.0.1:{API}/img/width=450/m2.jpeg", "type": "image", "browsingLevel": 4, "nsfwLevel": "Mature",
             "width": 450, "height": 600, "meta": None}]}

    def add(self, vid, mid, mname, vname, base, words=()):
        self.versions[vid] = {
            "id": vid, "modelId": mid, "name": vname, "baseModel": base, "status": "Published",
            "trainedWords": list(words), "description": "<p>hello <b>there</b></p>", "availability": "Public",
            "stats": {"downloadCount": 10, "thumbsUpCount": 5},
            "model": {"name": mname, "type": "LORA", "nsfw": False, "poi": False},
            "files": [{"id": vid, "name": f"{mname.replace(' ', '_')}.safetensors", "type": "Model",
                       "sizeKB": len(self.payload) / 1024, "primary": True,
                       "hashes": {"SHA256": sha(self.payload)},
                       "downloadUrl": f"http://127.0.0.1:{{API}}/api/download/models/{vid}"}],
            "images": [{"url": f"http://127.0.0.1:{{API}}/img/width=450/{vid}.jpeg", "type": "image",
                        "nsfwLevel": 1, "width": 450, "height": 600, "meta": {"prompt": "a cat", "seed": 1}}],
        }
        if mid in self.models:  # another version of the same model page (newest first)
            self.models[mid]["modelVersions"].insert(0, self.versions[vid])
        else:
            self.models[mid] = {"id": mid, "name": mname, "type": "LORA", "nsfw": False,
                                "description": "<p>Model <b>desc</b> <script>alert(1)</script></p>", "tags": ["style"],
                                "creator": {"username": "tester"}, "stats": {"downloadCount": 99, "thumbsUpCount": 7},
                                "modelVersions": [self.versions[vid]]}

    def _fix(self, obj):
        import json
        return json.loads(json.dumps(obj).replace("{API}", str(self.api_port)))

    def start(self):
        f = self

        async def get_version(r):
            f.api_auth_seen.append(r.headers.get("Authorization"))
            vid = int(r.match_info["id"])
            if vid in f.hidden or vid not in f.versions:
                return web.json_response({"error": "nope"}, status=404)
            return web.json_response(f._fix(f.versions[vid]))

        async def get_model(r):
            mid = int(r.match_info["id"])
            if mid in f.dead_models or mid not in f.models:
                return web.json_response({"error": "nope"}, status=404)
            m = dict(f.models[mid])
            m["modelVersions"] = [v for v in m["modelVersions"] if v["id"] not in f.hidden]
            return web.json_response(f._fix(m))

        async def search(r):
            q = r.query
            items = []
            for m in f.models.values():
                if q.get("query") and q["query"].lower() not in m["name"].lower():
                    continue
                if q.get("baseModels") and not any(v["baseModel"] == q["baseModels"] for v in m["modelVersions"]):
                    continue
                items.append(m)
            return web.json_response(f._fix({"items": items, "metadata": {"nextCursor": None}}))

        async def images(r):
            vid = int(r.query.get("modelVersionId", 0))
            items = list(f.extra_images.get(vid, []))
            for img in f.versions.get(vid, {}).get("images", []):
                items.append({**img, "nsfwLevel": "None", "browsingLevel": 1})
            return web.json_response(f._fix({"items": items, "metadata": {}}))

        async def enums(r):
            return web.json_response({"BaseModel": ["SD 1.5", "SDXL 1.0", "Pony"]})

        async def download(r):
            auth = r.headers.get("Authorization")
            if auth != "Bearer KEY123" and r.query.get("token") != "KEY123":
                return web.Response(text="<html>login</html>", content_type="text/html")
            raise web.HTTPFound(f"http://localhost:{f.cdn_port}/file/{r.match_info['id']}")

        async def img(r):
            return web.Response(body=PNG, content_type="image/jpeg")

        async def cdn(r):
            f.cdn_auth_seen.append(r.headers.get("Authorization"))
            data = f.payload
            rng = r.headers.get("Range")
            if rng:
                start = int(rng.split("=")[1].split("-")[0])
                return web.Response(body=data[start:], status=206,
                                    headers={"Content-Range": f"bytes {start}-{len(data)-1}/{len(data)}"})
            return web.Response(body=data, content_type="application/octet-stream")

        @web.middleware
        async def cors(request, handler):
            if request.method == "OPTIONS":
                resp = web.Response(status=204)
            else:
                resp = await handler(request)
            # (the real extension has host permissions for civitai.com, which bypasses CORS; the
            # fake lives on 127.0.0.1 so it has to speak credentialed CORS instead)
            origin = request.headers.get("Origin")
            if origin:
                resp.headers["Access-Control-Allow-Origin"] = origin
                resp.headers["Access-Control-Allow-Credentials"] = "true"
                resp.headers["Access-Control-Allow-Headers"] = "authorization, content-type, range"
            return resp

        api = web.Application(middlewares=[cors])
        api.router.add_get("/api/v1/model-versions/{id}", get_version)
        api.router.add_get("/api/v1/models/{id}", get_model)
        api.router.add_get("/api/v1/models", search)
        api.router.add_get("/api/v1/enums", enums)
        api.router.add_get("/api/v1/images", images)
        api.router.add_get("/api/download/models/{id}", download)
        api.router.add_get("/img/{p:.*}", img)
        cdn_app = web.Application(middlewares=[cors])
        cdn_app.router.add_get("/file/{id}", cdn)

        ready = threading.Event()

        def run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def go():
                for app, attr in ((api, "api_port"), (cdn_app, "cdn_port")):
                    runner = web.AppRunner(app)
                    await runner.setup()
                    site = web.TCPSite(runner, "127.0.0.1", 0)
                    await site.start()
                    setattr(f, attr, site._server.sockets[0].getsockname()[1])
                ready.set()

            loop.run_until_complete(go())
            loop.run_forever()

        threading.Thread(target=run, daemon=True).start()
        ready.wait(10)
        return self
