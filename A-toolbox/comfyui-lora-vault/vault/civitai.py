"""Civitai REST client.

Two transports (setting `transport`):
  "firefox" (default) - every request is relayed through the Firefox extension, so it uses
                        your browser's network route and Civitai login. ComfyUI itself
                        never connects to Civitai.
  "direct"            - ComfyUI's Python talks to Civitai itself (stdlib only).

Auth: an optional API key goes in an `Authorization: Bearer` header and is only ever sent
to Civitai hosts. In Firefox mode the browser's logged-in session is used too.
"""
import base64
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from . import config, relay

UA = "comfyui-lora-vault/" + config.VERSION
RETRY_STATUS = {429, 500, 502, 503, 504}


class CivitaiError(Exception):
    pass


class NotFound(CivitaiError):
    pass


class AuthRequired(CivitaiError):
    pass


class RateLimited(CivitaiError):
    pass


class NetworkError(CivitaiError):
    pass


def transport() -> str:
    return "direct" if config.get("transport") == "direct" else "firefox"


def available() -> bool:
    """Can we reach Civitai right now? (Firefox mode: is the extension polling?)"""
    return transport() == "direct" or relay.connected()


def _trusted(host: str) -> bool:
    host = (host or "").lower()
    if host in ("civitai.com", "civitai.red", "civitai.green") or host.endswith(
        (".civitai.com", ".civitai.red", ".civitai.green")
    ):
        return True
    return any((urllib.parse.urlparse(b).hostname or "").lower() == host for b in config.get("civitai_hosts"))


def _auth_headers(url: str) -> dict:
    key = config.get("civitai_api_key")
    if key and _trusted(urllib.parse.urlparse(url).hostname):
        return {"Authorization": "Bearer " + key}
    return {}


def with_token(url: str) -> str:
    """Browser downloads can't set headers reliably across Civitai's CDN redirect, so (and only
    when an API key is set) the key rides on the Civitai download URL itself."""
    key = config.get("civitai_api_key")
    if key and _trusted(urllib.parse.urlparse(url).hostname) and "token=" not in url:
        return url + ("&" if "?" in url else "?") + "token=" + urllib.parse.quote(key)
    return url


def _status_error(status: int, url: str):
    if status == 404:
        return NotFound(url)
    if status in (401, 403):
        return AuthRequired(f"HTTP {status} from Civitai - log in to Civitai in Firefox / check your API key")
    if status == 429:
        return RateLimited("Civitai rate limit hit (HTTP 429)")
    return CivitaiError(f"HTTP {status} from {url}")


# ---------------------------------------------------------------- direct transport
class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # we follow redirects ourselves
        return None


_opener = urllib.request.build_opener(_NoRedirect)


def _direct_text(url: str, timeout: int):
    for _ in range(6):
        h = {"User-Agent": UA, "Accept": "application/json", **_auth_headers(url)}
        try:
            with _opener.open(urllib.request.Request(url, headers=h), timeout=timeout) as r:
                return r.status, r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307, 308) and e.headers.get("Location"):
                url = urllib.parse.urljoin(url, e.headers["Location"])
                continue
            return e.code, ""
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            raise NetworkError(str(e))
    raise CivitaiError("too many redirects")


def open_stream(url: str, range_start: int = 0, timeout: int = 60):
    """Direct transport only: open a file download, following redirects manually with the
    auth header stripped off-site. Caller closes the response."""
    for _ in range(8):
        extra = {"Accept": "*/*"}
        if range_start:
            extra["Range"] = f"bytes={range_start}-"
        h = {"User-Agent": UA, **_auth_headers(url), **extra}
        try:
            resp = _opener.open(urllib.request.Request(url, headers=h), timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307, 308) and e.headers.get("Location"):
                url = urllib.parse.urljoin(url, e.headers["Location"])
                continue
            if e.code == 416:
                raise CivitaiError("range-not-satisfiable")
            raise _status_error(e.code, url)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise NetworkError(str(e))
        if "text/html" in (resp.headers.get("Content-Type") or "").lower():
            resp.close()
            raise AuthRequired(
                "Civitai sent a web page instead of the file. Add a valid API key in the vault "
                "settings - or this model needs login / early-access (Buzz) to download."
            )
        return resp
    raise CivitaiError("too many redirects")


# ---------------------------------------------------------------- shared
def _fetch_text(url: str, timeout: int):
    if transport() == "firefox":
        try:
            res = relay.submit("json", url, headers=_auth_headers(url), timeout=timeout + 20)
        except relay.RelayError as e:
            raise NetworkError(str(e))
        return int(res.get("status", 0)), res.get("body", "")
    return _direct_text(url, timeout)


def _get_json(url: str, timeout: int = 30, retries: int = 3):
    delay, last = 1.0, None
    for attempt in range(retries + 1):
        try:
            status, text = _fetch_text(url, timeout)
        except NetworkError as e:
            last = e
            if transport() == "firefox" and not relay.connected():
                raise  # nothing to retry against
            if attempt < retries:
                time.sleep(delay)
                delay = min(delay * 2, 30)
                continue
            raise
        if 200 <= status < 300:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                raise CivitaiError("Civitai didn't return JSON (a Cloudflare check page? open civitai.com in Firefox once).")
        if status in RETRY_STATUS and attempt < retries:
            last = _status_error(status, url)
            time.sleep(delay)
            delay = min(delay * 2, 30)
            continue
        raise _status_error(status, url)
    raise last or CivitaiError("request failed")


def _api(path: str):
    """GET path on each configured host; only NotFound if *every* host says 404."""
    last_nf = None
    for base in config.get("civitai_hosts"):
        try:
            return _get_json(f"{base}/api/v1/{path}")
        except NotFound as e:
            last_nf = e
    raise last_nf or NotFound(path)


def get_version(version_id: int) -> dict:
    return _api(f"model-versions/{int(version_id)}")


def get_model(model_id: int) -> dict:
    return _api(f"models/{int(model_id)}")


def get_version_by_hash(h: str) -> dict:
    return _api(f"model-versions/by-hash/{urllib.parse.quote(h)}")


_LEVEL_NAMES = {"none": 1, "soft": 2, "mature": 4, "x": 8, "xxx": 16}


def get_images(version_id: int, limit: int = 30) -> list:
    """Example images (with generation metadata) for a version, from the images endpoint,
    merged over every configured host. civitai.com filters mature images out of a version's
    own image list; civitai.red doesn't, so ask all hosts and combine. Never raises."""
    out, seen = [], set()
    q = urllib.parse.urlencode({"modelVersionId": int(version_id), "browsingLevel": 31,
                                "limit": limit, "sort": "Most Reactions"})
    for base in config.get("civitai_hosts"):
        try:
            data = _get_json(f"{base}/api/v1/images?{q}", retries=1)
        except CivitaiError:
            continue
        for it in (data.get("items") or []):
            url = it.get("url")
            if not url or url in seen:
                continue
            seen.add(url)
            lvl = it.get("browsingLevel")
            if not isinstance(lvl, int):
                nl = it.get("nsfwLevel")
                lvl = nl if isinstance(nl, int) else _LEVEL_NAMES.get(str(nl).lower(), 1)
            out.append({"url": url, "type": it.get("type") or "image", "nsfwLevel": lvl,
                        "width": it.get("width"), "height": it.get("height"), "meta": it.get("meta")})
    return out


FALLBACK_BASE_MODELS = [
    "SD 1.5", "SD 2.1", "SDXL 1.0", "Pony", "Illustrious", "NoobAI", "Flux.1 D", "Flux.1 S",
    "Flux.1 Krea", "Flux.1 Kontext", "SD 3", "SD 3.5", "SD 3.5 Large", "Qwen", "Wan Video 14B t2v",
    "Wan Video 14B i2v 480p", "Wan Video 2.2 T2V-A14B", "Wan Video 2.2 I2V-A14B", "HiDream",
    "Lumina", "Hunyuan Video", "LTXV", "Chroma", "ZImageTurbo", "Anima", "Other",
]
_enum_cache = {"t": 0.0, "bases": None}


def get_base_models() -> list:
    """Base-model names Civitai accepts for filtering (GET /enums -> BaseModel). Cached for a
    day; falls back to a built-in list if Civitai/Firefox can't be reached."""
    if _enum_cache["bases"] and time.time() - _enum_cache["t"] < 86400:
        return _enum_cache["bases"]
    if available():
        try:
            data = _api("enums")
            bases = data.get("BaseModel") if isinstance(data, dict) else None
            if bases:
                _enum_cache.update(t=time.time(), bases=[str(b) for b in bases])
                return _enum_cache["bases"]
        except CivitaiError:
            pass
    return FALLBACK_BASE_MODELS


def search_models(query: str = "", base_model: str = "", sort: str = "Most Downloaded",
                  period: str = "AllTime", nsfw: bool = False, cursor: str = "",
                  limit: int = 20, types: str = "LORA") -> dict:
    """Search Civitai LoRAs. `query` can't be combined with page=, so we page by cursor.
    types="" or "all" searches every model type (for LoRAs the uploader tagged as something else)."""
    p = [("limit", str(max(1, min(int(limit), 50)))), ("sort", sort), ("period", period)]
    if types and types.lower() != "all":
        p.insert(0, ("types", types))
    if query:
        p.append(("query", query))
    if base_model:
        p.append(("baseModels", base_model))
    if nsfw:
        p.append(("nsfw", "true"))
    if cursor:
        p.append(("cursor", cursor))
    return _api("models?" + urllib.parse.urlencode(p))


def test_connection() -> dict:
    t = time.time()
    try:
        _get_json(f"{config.get('civitai_hosts')[0]}/api/v1/enums", timeout=15, retries=0)
        return {"ok": True, "ms": int((time.time() - t) * 1000), "transport": transport()}
    except CivitaiError as e:
        return {"ok": False, "error": str(e), "transport": transport()}


_URL_RE = re.compile(r"/models/(\d+)")


def parse_civitai_url(text: str):
    """'https://civitai.com/models/123/x?modelVersionId=456' -> (123, 456).
    A bare number is treated as a model-version id."""
    text = (text or "").strip()
    if text.isdigit():
        return None, int(text)
    m = _URL_RE.search(text)
    if not m:
        return None, None
    model_id = int(m.group(1))
    q = urllib.parse.parse_qs(urllib.parse.urlparse(text).query)
    vid = q.get("modelVersionId", [None])[0]
    return model_id, int(vid) if vid and vid.isdigit() else None


def fetch_bytes(url: str, max_bytes: int = 8 * 1024 * 1024, timeout: int = 30) -> bytes:
    """Small binary fetch (preview images)."""
    if transport() == "firefox":
        try:
            res = relay.submit("bytes", url, headers={}, timeout=timeout + 20)
        except relay.RelayError as e:
            raise NetworkError(str(e))
        if int(res.get("status", 0)) != 200:
            raise _status_error(int(res.get("status", 0)), url)
        data = base64.b64decode(res.get("body_b64", ""))
    else:
        resp = open_stream(url, timeout=timeout)
        try:
            data = resp.read(max_bytes + 1)
        finally:
            resp.close()
    if len(data) > max_bytes:
        raise CivitaiError("file too large")
    return data
