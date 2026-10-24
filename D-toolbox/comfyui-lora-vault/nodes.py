"""Lora Vault Loader - applies LoRAs listed as <lora:v12345:0.8> text, downloading any
that aren't on disk yet (from your vault) right before they're needed."""
from .vault import parse, storage, db


def _send(event: str, data: dict):
    try:
        from server import PromptServer

        PromptServer.instance.send_sync(event, data)
    except Exception:
        pass


class LoraVaultLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_text": ("STRING", {"multiline": True, "default": "", "dynamicPrompts": False}),
            },
            "optional": {
                "download_missing": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING")
    RETURN_NAMES = ("model", "clip", "trigger_words", "loaded")
    FUNCTION = "load"
    CATEGORY = "loaders/lora vault"
    DESCRIPTION = (
        "Apply LoRAs from your Lora Vault. LoRAs you've saved but not downloaded are fetched "
        "from Civitai on first use. Syntax: <lora:v12345:0.8> (model strength) or <lora:v12345:0.8:0.6> (model:clip)."
    )

    def load(self, model, clip, lora_text, download_missing=True):
        import comfy.sd
        import comfy.utils

        specs = parse.parse(lora_text)
        resolved = []
        for s in specs:  # resolve everything first so a typo fails before any download starts
            kind, obj = parse.resolve(s["ref"])
            resolved.append((s, kind, obj))
        protect = {o["version_id"] for _, k, o in resolved if k == "vault"}

        words, loaded = [], []
        for s, kind, obj in resolved:
            if kind == "vault":
                vid = obj["version_id"]
                label = f"{obj['model_name']} / {obj['version_name']}"
                if storage.local_state(obj) == "none" and not download_missing:
                    raise RuntimeError(f"LoRA '{label}' isn't downloaded and download_missing is off.")

                def progress(done, total, vid=vid, label=label):
                    _send("lora_vault.progress", {"version_id": vid, "done": done, "total": total, "name": label})

                path = storage.ensure_local(vid, progress=progress, protect=protect)
                _send("lora_vault.progress", {"version_id": vid, "done": 1, "total": 1, "name": label, "finished": True})
                words.extend(w for w in (db.get_version(vid) or obj)["trigger_words"] if w not in words)
            else:
                path, label = obj, obj.replace("\\", "/").split("/")[-1]
            lora = comfy.utils.load_torch_file(path, safe_load=True)
            model, clip = comfy.sd.load_lora_for_models(model, clip, lora, s["strength"], s["clip_strength"])
            loaded.append(f"{label} @ {s['strength']:g}" + (f"/{s['clip_strength']:g}" if s["clip_strength"] != s["strength"] else ""))
        return (model, clip, ", ".join(words), "\n".join(loaded))


NODE_CLASS_MAPPINGS = {"LoraVaultLoader": LoraVaultLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"LoraVaultLoader": "Lora Vault Loader"}
