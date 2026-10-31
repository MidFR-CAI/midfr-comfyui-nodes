"""Lora Vault Loader - applies LoRAs listed as <lora:v12345:0.8> text, downloading any
that aren't on disk yet (from your vault) right before they're needed."""
from .vault import parse, storage, db, lora_fix


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
        "from Civitai on first use. Syntax: <lora:v12345:0.8> (strength for model and clip), "
        "<lora:v12345:0.8:0.6> (model:clip), add :layers=14-41 to apply only those transformer layers. "
        "Fused qkv LoRAs are split automatically when the model needs it."
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
            lora, notes = lora_fix.prepare(lora, model, s.get("layer_start"), s.get("layer_end"))
            model, clip = comfy.sd.load_lora_for_models(model, clip, lora, s["strength"], s["clip_strength"])
            loaded.append(f"{label} @ {s['strength']:g}" + (f"/{s['clip_strength']:g}" if s["clip_strength"] != s["strength"] else "")
                          + (f" [{'; '.join(notes)}]" if notes else ""))
        return (model, clip, ", ".join(words), "\n".join(loaded))


class LoraLoaderLayers:
    """Plain model-only LoRA loader with a layer range (no vault)."""

    CATEGORY = "loaders"
    FUNCTION = "load"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    DESCRIPTION = "Load a LoRA onto the model, optionally only on transformer layers start..end (inclusive, end -1 = last)."

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths

        return {"required": {
            "model": ("MODEL",),
            "lora_name": (folder_paths.get_filename_list("loras"),),
            "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
            "start_layer": ("INT", {"default": 0, "min": 0, "max": 999, "tooltip": "First layer the LoRA applies to."}),
            "end_layer": ("INT", {"default": -1, "min": -1, "max": 999, "tooltip": "Last layer (inclusive); -1 = all the way to the end."}),
        }}

    def load(self, model, lora_name, strength, start_layer=0, end_layer=-1):
        import comfy.sd
        import comfy.utils
        import folder_paths

        if strength == 0:
            return (model,)
        lora = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("loras", lora_name), safe_load=True)
        end = None if end_layer < 0 else end_layer
        lora, _ = lora_fix.prepare(lora, model, start_layer or None, end)
        model, _ = comfy.sd.load_lora_for_models(model, None, lora, strength, 0)
        return (model,)


NODE_CLASS_MAPPINGS = {"LoraVaultLoader": LoraVaultLoader, "LoraLoaderLayers": LoraLoaderLayers}
NODE_DISPLAY_NAME_MAPPINGS = {"LoraVaultLoader": "Lora Vault Loader", "LoraLoaderLayers": "Load LoRA (layers, model only)"}
