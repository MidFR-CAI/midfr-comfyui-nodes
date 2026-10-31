"""Scene Context (Qwen VL): looks at each box of the picture and writes a short "keep these unchanged" note (skin tone, lip size,
eye colour ...) for that box's edit prompt. Edit models drift on whatever the prompt does not say (skin comes back paler, lips
thinner); naming those attributes pins them. The note is PER BOX and only covers what is visible inside that box: a global note
("dark curly hair") makes the model grow hair into a box that has none. Wire the bbox node's `passes` through this node into
Invoke Inpainting's `passes`."""
import gc
import hashlib
import json
import os
import re

import numpy as np
import torch
from PIL import Image

import folder_paths

# Per-box denoise strength by edit type, from testing (the user does not pick it): an edit that has to draw a new shape needs ~1.0
# (0.9-0.95 leaves the old outfit); an edit that keeps the shape and only changes colour/material must stay close to the original,
# or the pose, skin texture and background around it get redrawn too. Shoe recolour tests (silver -> black, same shape): 0.6/0.75
# change nothing, 0.9 only greys them, so every kind needs ~1.0 today; lower a kind here if a test shows it works lower.
TYPE_STRENGTH = {"replace": 1.0, "remove": 1.0, "add": 1.0, "detail": 1.0, "recolour": 1.0}
_CACHE = {}  # (crop hash, settings) -> text, so re-queuing the same picture does not rerun the model

SYSTEM = ("You write short, factual appearance notes for an image editor. Reply with one line only: comma-separated attributes, "
          "no sentences, no markdown, no names, nothing you cannot clearly see.")

EDIT_SYSTEM = ("You write short, factual notes for an image editor, in exactly the lines asked for. No markdown, no names, "
               "nothing you cannot clearly see.")


def _models():
    root = os.path.join(folder_paths.models_dir, "LLM")
    out = []
    if os.path.isdir(root):
        for d in sorted(os.listdir(root)):
            p = os.path.join(root, d)
            if os.path.isfile(os.path.join(p, "config.json")) and "vl" in d.lower():
                out.append(d)
    return out or ["(no Qwen-VL model found in models/LLM)"]


def _crop_pil(image, rect, pad=16, long_side=768):
    arr = (image[0].detach().float().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    h, w = arr.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in rect)
    x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
    im = Image.fromarray(arr[y0:y1, x0:x1, :3])
    s = long_side / max(im.size)
    if s < 1 or max(im.size) < 512:  # small crops are enlarged so the VLM sees iris, strands and fingers
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
    return im


class SceneContextQwenVL:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "passes": ("STRING", {"forceInput": True, "tooltip": "The bbox node's passes output"}),
                "model": (_models(), {"tooltip": "A Qwen3-VL folder in models/LLM"}),
                "describe": ("STRING", {"multiline": True, "default":
                    "skin tone and ethnicity, eye colour, lip size and shape, hair colour",
                    "tooltip": "What to write down. Each one is only written if it is clearly visible inside the box"}),
                "exclude": ("STRING", {"multiline": True, "default": "clothing, accessories, expression, pose, background, body build",
                    "tooltip": "What to leave out (the things you are likely to edit). The box's own edit text is also left out"}),
            },
            "optional": {
                "max_new_tokens": ("INT", {"default": 64, "min": 16, "max": 256}),
                "keep_loaded": ("BOOLEAN", {"default": False, "tooltip": "Keep the VLM in memory (uses VRAM while MOSS samples)"}),
                "auto_strength": ("BOOLEAN", {"default": True, "tooltip":
                    "Pick each box's denoise strength from the kind of edit the VLM sees (replace/remove/add need full strength, a "
                    "recolour of the same shape needs much less so the pose and surroundings stay). A strength set on the box itself wins"}),
                "edit_aware": ("BOOLEAN", {"default": True, "tooltip":
                    "Write each box's notes against its edit text, as padding for it: only the problem points inside the redrawn area "
                    "(skin tone, lip size, tattoos...), a description of the old item being replaced or removed so none of it is left "
                    "behind, and a note of the surrounding scene so the edit does not bleed into it. Off = the same attribute list for every box"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("passes", "notes")
    FUNCTION = "run"
    CATEGORY = "image/midfr"
    _loaded = None  # (name, model, processor)

    @classmethod
    def IS_CHANGED(cls, image, passes, model, describe, exclude, max_new_tokens=64, keep_loaded=False, edit_aware=True, auto_strength=True):
        return hashlib.sha1(image[0].cpu().numpy().tobytes()[::97] + repr((passes, model, describe, exclude, max_new_tokens, edit_aware, auto_strength)).encode()).hexdigest()

    def _load(self, name):
        cls = type(self)
        if cls._loaded and cls._loaded[0] == name:
            return cls._loaded[1], cls._loaded[2]
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        path = os.path.join(folder_paths.models_dir, "LLM", name)
        cuda = torch.cuda.is_available()
        dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else (torch.float16 if cuda else torch.float32)
        model = Qwen3VLForConditionalGeneration.from_pretrained(path, torch_dtype=dtype, device_map="cuda" if cuda else "cpu",
                                                                trust_remote_code=True)
        proc = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        cls._loaded = (name, model, proc)
        return model, proc

    @classmethod
    def _unload(cls):
        cls._loaded = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _gen(vlm, proc, crop, system, ask, max_new_tokens):
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": ask}]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[crop], padding=True, return_tensors="pt")
        inputs = {k: (v.to(vlm.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        with torch.no_grad():
            out = vlm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                               pad_token_id=proc.tokenizer.eos_token_id, eos_token_id=proc.tokenizer.eos_token_id)
        return proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def _edit_note(self, vlm, proc, crop, caption, describe, leave):
        """Padding for the box's edit text: what the text leaves out and the model would otherwise get wrong. Only what sits in the
        area the edit redraws: the person's problem points there (skin tone, lips, tattoos...), a description of the old item being
        replaced or removed (so none of it is left behind), and the neighbours that could take on the new item's colour."""
        ask = ("This crop of a picture is being edited with the instruction: \"" + caption.strip() + "\"\n"
               "Answer with exactly these five lines (write 'none' after a label if nothing applies). KEEP and OLD are "
               "short comma-separated noun phrases; SCENE is described below. Only what is clearly visible in the crop, no markdown.\n"
               "KEEP: first work out, from what you see, which parts of the person the edit will redraw or reveal: look at where the old "
               "item sits and what it covers, touches or hides (anything it hides will be redrawn). List the attributes of those parts "
               "that an edit tends to change by accident. Pick only from: " + describe.strip() + ". Skip parts the edit will not "
               "touch, even if they are visible in the crop. For hair give only its colour and curl texture. Write the actual "
               "value for each, never the label (for example 'deep warm brown skin'). Judge colours by the main body of the thing, not by highlights, "
               "reflections or make-up, and say 'black' for anything very dark instead of brown. Do not mention: " + "; ".join(leave) + ", or anything the instruction itself changes.\n"
               "OLD: describe in detail the existing item that the instruction replaces or removes, as it looks now: colour, material, "
               "trim, and every visible part of it (a dress: bodice, straps, hem; headwear: band, frills), saying where each part is, "
               "including any part that spreads over or lies on something else (for example a hem spread out over a sofa). Never include anything the "
               "instruction adds, and never the person's body. If the instruction only adds something, write none.\n"
               "POSE: for each hand, arm or leg inside the redrawn area, its exact position, finger positions and what it touches or holds "
               "(for example 'right hand resting on the knee, fingers loosely extended downward'). If it holds or touches something "
               "the instruction removes, say the hand keeps the same relaxed pose.\n"
               "TYPE: the kind of edit the instruction is (if it does several, list every kind that applies, separated by commas): recolour (the same item in the same shape, only its colour, material or "
               "pattern changes), replace (a different item or a different shape takes its place), remove (it is taken away and nothing "
               "new is put there), add (a new item where there was none), or detail (a small change to a face, hair or body detail).\n"
               "SCENE: everything else inside the crop that the instruction does not change: the setting, each other object and any other "
               "person. One short sentence per item, 20 words at most each, with its colour and where it is relative to the changed "
               "item, separated by semicolons, so the edit does not bleed into them.")
        raw = self._gen(vlm, proc, crop, EDIT_SYSTEM, ask, 320)
        got = {}
        for ln in raw.splitlines():
            if ":" in ln:
                k, v = ln.split(":", 1)
                k = k.strip(" *-#").upper()
                v = " ".join(v.split()).strip(" .*")
                if k in ("KEEP", "OLD", "POSE", "TYPE", "SCENE") and v and not v.lower().startswith(("none", "n/a", "not ")):
                    got[k] = v
        bad = ("skin", "hair", "eye", "face", "lip", "body", "arm", "leg", "hand", "finger", "foot", "feet", "ethnic", "person", "woman")
        if "OLD" in got:  # the person is never what gets removed
            items = [t.strip() for t in got["OLD"].split(",")]
            items = [t for t in items if t and not any(w in t.lower() for w in bad)]
            if items:
                got["OLD"] = ", ".join(items)
            else:
                del got["OLD"]
        kind = ""
        if "TYPE" in got:
            w = re.findall(r"[a-z]+", got["TYPE"].lower())
            found = [t for t in w if t in TYPE_STRENGTH]
            kind = max(found, key=lambda t: TYPE_STRENGTH[t]) if found else ""  # several edits in one box: the strongest one wins
        out = []
        if "KEEP" in got:
            out.append("Keep these unchanged: " + got["KEEP"] + ".")
        if "OLD" in got:
            if kind == "remove":
                out.append("The old " + got["OLD"] + " is removed completely, every part of it, wherever it lies; the space it leaves is filled with whatever naturally continues from its surroundings (the person's hair, skin, clothes or the background around it).")
            else:  # the item is swapped for the new one, not taken away
                out.append("The old " + got["OLD"] + " is replaced by the new item, no part of the old one remains.")
        if "POSE" in got:
            out.append("Keep the pose exactly: " + got["POSE"] + ".")
        if "SCENE" in got:
            out.append("Around it, not part of the edit and left as it is: " + got["SCENE"] + ".")
        return " ".join(out), kind

    def _note(self, vlm, proc, crop, describe, leave, max_new_tokens):
        ask = ("This is a crop from a larger picture. List ONLY these attributes of the person, and ONLY the ones that are clearly "
               "visible in this crop: " + describe.strip() + ". "
               + ("Do NOT mention: " + "; ".join(leave) + ". " if leave else "")
               + "Be specific about colours and sizes (for example 'deep warm brown skin', 'full lips'). One line, comma-separated. "
               "If none of them are visible, reply with exactly: none")
        raw = self._gen(vlm, proc, crop, SYSTEM, ask, max_new_tokens)
        parts = [t.strip(" .") for t in " ".join(raw.split()).split(",")]
        line = ", ".join(t for t in parts if t and t.lower() not in ("none", "n/a", "not visible"))
        if not line:
            return ""
        return "Keep these unchanged: " + line + "."

    def run(self, image, passes, model, describe, exclude, max_new_tokens=64, keep_loaded=False, edit_aware=True, auto_strength=True):
        try:
            todo = json.loads(passes) if (passes or "").strip() else []
        except Exception:
            todo = []
        if not todo:
            return (passes, "")
        if model.startswith("("):
            raise RuntimeError("Scene Context: no Qwen-VL model in models/LLM")
        loaded = False
        notes = []
        for p in todo:
            crop = _crop_pil(image, p["rect"])
            leave = [t for t in (exclude.strip(), (p.get("caption") or "").strip()) if t]
            cap = (p.get("caption") or "").strip() if edit_aware else ""
            key = (hashlib.sha1(np.asarray(crop).tobytes()[::53]).hexdigest(), model, describe, tuple(leave), max_new_tokens, cap)
            if key not in _CACHE:
                if not loaded:
                    try:
                        import comfy.model_management as mm
                        mm.unload_all_models()  # make room: MOSS is reloaded by the sampler afterwards
                        mm.soft_empty_cache()
                    except Exception:
                        pass
                    vlm, proc = self._load(model)
                    loaded = True
                _CACHE[key] = (self._edit_note(vlm, proc, crop, cap, describe, leave) if cap
                               else (self._note(vlm, proc, crop, describe, leave, max_new_tokens), ""))
            text, kind = _CACHE[key]
            if p.get("items"):  # the mask node's note on every worn or held item that is replaced or removed
                text = (text + " " + p["items"]).strip()
            p["context"] = text
            if auto_strength and kind and p.get("strength") is None:
                p["strength"] = TYPE_STRENGTH[kind]
            notes.append(text + ("  [%s%s]" % (kind, ", denoise %.2f" % p["strength"] if p.get("strength") is not None else "") if kind else ""))
        if loaded and not keep_loaded:
            del vlm, proc
            self._unload()
        return (json.dumps(todo, separators=(",", ":")), "\n".join("box %d: %s" % (i + 1, n or "(nothing visible)") for i, n in enumerate(notes)))


NODE_CLASS_MAPPINGS = {"SceneContextQwenVL": SceneContextQwenVL}
NODE_DISPLAY_NAME_MAPPINGS = {"SceneContextQwenVL": "Scene Context (Qwen VL)"}
