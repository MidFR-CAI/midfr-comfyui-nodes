"""Firefox relay: ComfyUI never touches the internet itself.

When Civitai traffic has to go through your browser, the vault queues each request here
and the Firefox extension (which long-polls /relay/next) runs it with Firefox's own
network route and Civitai login, then posts the result back. Three kinds of task:

  json      GET a Civitai API URL, return status + text
  bytes     GET a small file (preview image), return base64
  resolve   follow a Civitai download link and return the signed CDN URL (ComfyUI downloads it itself)
  download  have Firefox save a big file (LoRA) with its download manager, then report
            where it landed so the vault can move it into place
"""
import threading
import time
import uuid
from collections import deque

CONNECTED_WINDOW = 45  # seconds since the extension's last poll that still counts as "connected"


class RelayError(Exception):
    pass


class NotConnected(RelayError):
    pass


class Task:
    def __init__(self, kind: str, url: str, headers: dict | None, **extra):
        self.id = uuid.uuid4().hex
        self.kind, self.url, self.headers, self.extra = kind, url, headers or {}, extra
        self.event = threading.Event()
        self.result: dict | None = None
        self.received = 0
        self.total = 0
        self.last_activity = time.time()
        self.claimed = False
        self.cancelled = False

    def public(self) -> dict:
        return {"id": self.id, "kind": self.kind, "url": self.url, "headers": self.headers, **self.extra}


_lock = threading.Lock()
_pending: deque = deque()
_tasks: dict = {}
_last_poll = 0.0


def connected() -> bool:
    return time.time() - _last_poll < CONNECTED_WINDOW


_pollers = 0   # long-polls currently open; more than one means two helpers are running


def poller_enter():
    global _pollers
    with _lock:
        _pollers += 1


def poller_leave():
    global _pollers
    with _lock:
        _pollers = max(0, _pollers - 1)


_ext_version = None


def note_version(v):
    global _ext_version
    if v:
        _ext_version = str(v)[:20]


def status() -> dict:
    with _lock:
        return {"connected": connected(), "pollers": _pollers, "extension_version": _ext_version, "last_poll_age": None if not _last_poll else round(time.time() - _last_poll, 1),
                "pending": len(_pending), "active": sum(1 for t in _tasks.values() if t.claimed)}


def note_poll():
    global _last_poll
    _last_poll = time.time()


def claim_next():
    """Called by the extension's long-poll. Returns the next task dict or None."""
    note_poll()
    with _lock:
        while _pending:
            t = _pending.popleft()
            if t.cancelled:
                continue
            t.claimed = True
            t.last_activity = time.time()
            return t.public()
    return None


def submit(kind: str, url: str, headers: dict | None = None, timeout: float = 60,
           idle_timeout: float | None = None, progress=None, **extra) -> dict:
    """Queue a task and block until the extension finishes it. Raises NotConnected right away
    if no extension is polling, rather than leaving the caller hanging."""
    if not connected():
        raise NotConnected(
            "The Firefox helper isn't connected. Open Firefox with the Lora Vault extension "
            "installed (and keep it running) - Civitai traffic goes through it."
        )
    t = Task(kind, url, headers, **extra)
    with _lock:
        _tasks[t.id] = t
        _pending.append(t)
    deadline = time.time() + timeout
    try:
        while not t.event.wait(0.5):
            if progress and t.total:
                progress(t.received, t.total)
            if idle_timeout is not None:  # long downloads: only fail if nothing is happening
                if time.time() - t.last_activity > idle_timeout:
                    raise RelayError("Firefox stopped reporting progress on this download.")
            elif time.time() > deadline:
                if not t.claimed:
                    raise NotConnected("The Firefox helper didn't pick up the request - is Firefox running?")
                with _lock:
                    many = _pollers > 1
                print(f"[LoraVault] relay: Firefox claimed a '{kind}' request but never answered: {url}")
                raise RelayError(
                    f"Firefox took too long to answer ({kind} {url[:140]})."
                    + (" More than one Lora Vault helper is polling - disable or remove the older Lora Vault extension in about:addons, "
                       "because an old copy can pick up requests and never answer them." if many else
                       " Firefox picked the request up but never returned it (Civitai slow, a stuck connection, or the helper was reloaded mid-request)."))
        if progress and t.total:
            progress(t.received, t.total)
    except BaseException:
        t.cancelled = True
        raise
    finally:
        with _lock:
            _tasks.pop(t.id, None)
    res = t.result or {}
    if not res.get("ok"):
        raise RelayError(res.get("error") or "Firefox reported a failure")
    return res


def report_progress(task_id: str, received: int, total: int):
    t = _tasks.get(task_id)
    if t:
        t.received, t.total, t.last_activity = int(received), int(total), time.time()


def complete(task_id: str, payload: dict):
    t = _tasks.get(task_id)
    if not t:
        return False
    t.result = payload
    t.last_activity = time.time()
    t.event.set()
    return True
