"""Settings and paths for Lora Vault.

Everything user-specific lives in <ComfyUI>/user/lora_vault/ (override with the
LORA_VAULT_DATA_DIR env var), so updating or re-installing the node never wipes
your library, previews or API key.
"""
import json
import os
import threading

VERSION = "0.1.0"

_lock = threading.RLock()

DEFAULTS = {
    # Free key from civitai.com -> Account settings -> API Keys.
    "civitai_api_key": "",
    # "firefox": every Civitai request (API, search, previews, downloads) goes through the
    # Firefox extension, using your browser's network route + Civitai login.
    # "direct": ComfyUI's Python connects to Civitai itself.
    "transport": "firefox",
    "download_mode": "direct",   # "direct": ComfyUI downloads the file itself once Firefox has resolved the link
    # API hosts to try, in order. A 404 on one host is retried on the next before
    # a model is ever treated as missing (civitai.com hides mature models).
    "civitai_hosts": ["https://civitai.com", "https://civitai.red"],
    # Where LoRA files go. Blank = ComfyUI's first "loras" folder.
    "lora_dir": "",
    # Max size of the *unpinned* download cache. 0 = unlimited.
    "cache_limit_gb": 10.0,
    "check_interval_hours": 24,
    # If a saved model disappears and we have the file cached, pin it automatically.
    "auto_pin_on_removal": True,
    # Local preview images kept per LoRA. Unpinned: just a few (text metadata - descriptions,
    # trigger words, every example's prompt/steps/sampler - is always kept). Pinned: everything.
    "unpinned_previews": 1,
    "pinned_previews": 40,
    "preview_width": 640,
    # Blur mature previews (R and above) everywhere. Turn off to always show them.
    "blur_mature": True,
    "side_server_enabled": True,
    "side_server_port": 8199,
}


def data_dir() -> str:
    d = os.environ.get("LORA_VAULT_DATA_DIR")
    if not d:
        try:
            import folder_paths  # ComfyUI

            d = os.path.join(folder_paths.get_user_directory(), "lora_vault")
        except Exception:
            d = os.path.join(os.path.expanduser("~"), ".lora_vault")
    os.makedirs(d, exist_ok=True)
    return d


def _settings_path() -> str:
    return os.path.join(data_dir(), "settings.json")


def load() -> dict:
    with _lock:
        s = dict(DEFAULTS)
        try:
            with open(_settings_path(), "r", encoding="utf-8") as f:
                stored = json.load(f)
            if isinstance(stored, dict):
                s.update({k: v for k, v in stored.items() if k in DEFAULTS})
        except FileNotFoundError:
            pass
        except Exception:
            pass  # corrupt file -> fall back to defaults rather than crash ComfyUI
        env_key = os.environ.get("CIVITAI_API_KEY")
        if env_key and not s["civitai_api_key"]:
            s["civitai_api_key"] = env_key
        return s


def get(key: str):
    return load()[key]


def update(changes: dict) -> dict:
    """Validate + persist a partial settings dict. Returns the new full settings."""
    with _lock:
        s = load()
        for k, v in (changes or {}).items():
            if k not in DEFAULTS:
                continue
            default = DEFAULTS[k]
            if isinstance(default, bool):
                v = bool(v)
            elif isinstance(default, int):
                v = int(v)
            elif isinstance(default, float):
                v = max(0.0, float(v))
            elif isinstance(default, list):
                if isinstance(v, str):
                    v = [x.strip() for x in v.replace(",", "\n").splitlines() if x.strip()]
                v = [str(x).rstrip("/") for x in v if str(x).strip()] or list(default)
            else:
                v = str(v).strip()
            if k == "transport" and v not in ("firefox", "direct"):
                v = "firefox"
            if k == "download_mode" and v not in ("direct", "firefox"):
                v = "direct"
            s[k] = v
        path = _settings_path()
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(s, f, indent=2)
        os.replace(tmp, path)
        return s


def public_settings() -> dict:
    """Settings safe to send to the browser (API key masked)."""
    s = load()
    key = s.pop("civitai_api_key", "")
    s["has_api_key"] = bool(key)
    s["api_key_hint"] = ("…" + key[-4:]) if len(key) >= 8 else ""
    return s


def lora_root() -> str:
    d = get("lora_dir") or os.environ.get("LORA_VAULT_LORA_DIR", "")
    if not d:
        try:
            import folder_paths

            d = folder_paths.get_folder_paths("loras")[0]
        except Exception:
            d = os.path.join(data_dir(), "loras")
    return os.path.abspath(d)


def cache_dir() -> str:
    d = os.path.join(lora_root(), "_vault", "cache")
    os.makedirs(d, exist_ok=True)
    return d


def pinned_dir() -> str:
    d = os.path.join(lora_root(), "_vault", "pinned")
    os.makedirs(d, exist_ok=True)
    return d


def previews_dir() -> str:
    d = os.path.join(data_dir(), "previews")
    os.makedirs(d, exist_ok=True)
    return d


def backups_dir() -> str:
    d = os.path.join(data_dir(), "backups")
    os.makedirs(d, exist_ok=True)
    return d
