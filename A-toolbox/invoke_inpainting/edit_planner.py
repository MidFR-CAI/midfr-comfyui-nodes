"""Edit Planner (Qwen VL) + lazy gates.

The planner looks at the picture and the plain edit instruction and decides, instead of fixed settings:
  * what the instruction really does (outfit swap / recolour / small local edit / background / pose) and so how much control is needed:
    only an outfit change that alters the silhouette needs the clothed-figure depth control, everything else is a plain reference edit;
  * which segmentation classes to drop from the silhouette (hair, glasses...) and which items the edit removes (-> negative prompt);
  * what must stay the same, which it writes into the prompt (face, hair, build, pose, camera, framing, background) whenever the
    instruction does not mention it.
The gates are lazy: when the planner switches the control off, nothing upstream of the gate (Sapiens depth) runs."""
import hashlib
import json
import re

import numpy as np
import torch
from PIL import Image

from .context_vlm import SceneContextQwenVL, _models

SYS = ("You answer questions about a picture and an edit instruction with a short phrase or one word. No full sentences, no markdown. "
       "Left and right always mean the VIEWER's left and right. Say only what you clearly see.")

FACTS = [
    ("face", "Describe the person's face and skin: skin tone, eye colour, lip shape, expression. Short comma-separated attributes."),
    ("hair", "Describe only the person's own hair, ignoring anything worn on the head such as a helmet, hat or headband: colour, length, "
             "style (braid, ponytail, loose). Short comma-separated attributes."),
    ("build", "Describe the person's body shape as visible: bust size, waist, hips, shoulder width, muscle tone, height impression. "
              "Short comma-separated attributes."),
    ("pose", "Describe only the posture of the person's body in one phrase: head tilt and facing, torso turn, where each arm and hand "
             "is (viewer's left / right), how the legs stand. Do not name any clothing, armor or held object."),
    ("camera", "Where is the camera relative to the person? Answer with one of: low angle looking up, eye level, high angle looking down."),
    ("framing", "The frame crops the person at which part of the body? Answer as 'from <top> to <bottom>' and how much of the frame "
                "the person fills."),
    ("scene", "List the notable objects and architecture in the background, as short comma-separated nouns."),
    ("style", "Name the picture's medium and art style in a few words (for example: photorealistic 3d render, anime cel shading, "
              "digital painting)."),
]

_COLOURS = {"grey", "gray", "white", "black", "blue", "red", "green", "brown", "blonde", "pink", "navy", "beige", "yellow",
            "purple", "orange", "silver", "cream", "tan"}
_ACCESSORY = {"watch", "watches", "earring", "earrings", "ring", "rings", "bracelet", "bracelets", "bangle", "bangles", "necklace",
              "pendant", "chain", "piercing", "piercings"}


def _filter_removed(items, iw):
    """Keep only what the edit really removes. `iw` = words of the instruction + the described additions.
    - an item whose distinguishing words (colours ignored) are all named by the instruction is a NEW item, not a removal;
    - jewelry and watches stay unless the instruction names them."""
    out = []
    for x in items:
        ws = re.findall(r"[a-z]+", x.lower())
        key = [w for w in ws if len(w) > 3 and w not in _COLOURS]
        if key and all(w in iw for w in key):
            continue
        acc = [w for w in ws if w in _ACCESSORY]
        if acc and not any(w in iw for w in acc):
            continue
        out.append(x)
    return out


ETYPES = ("outfit", "recolor", "local", "background", "pose", "other")


def _pil(image, long_side=1024):
    arr = (image[0].detach().float().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    im = Image.fromarray(arr[..., :3])
    s = long_side / max(im.size)
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    return im


def _clean(a):
    return a.strip().strip('"').strip().rstrip(".")


class EditPlannerQwenVL:
    CATEGORY = "invoke_inpainting"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING", "FLOAT", "FLOAT", "STRING", "STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("prompt", "negative_prompt", "control_strength", "control_end", "drop_classes", "plan", "control_mode", "use_control")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "instruction": ("STRING", {"multiline": True, "default": "", "tooltip": "The edit, in plain words. Name only what changes."}),
            "model": (_models(), {"tooltip": "A Qwen3-VL folder in models/LLM"}),
        }, "optional": {
            "base_negative": ("STRING", {"multiline": True, "default": "deformed, extra limbs, blurry, low quality"}),
            "control_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05,
                                           "tooltip": "Union control strength when the planner switches the depth control on"}),
            "control_end": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
            "keep_loaded": ("BOOLEAN", {"default": False}),
            "control_mode": (["auto", "none", "depth", "raw"], {"default": "auto", "tooltip":
                             "raw: no planner and no control, your instruction goes to Qwen unchanged. "
                             "auto: the planner decides. none: plain reference edit. depth: the clothed figure's depth as control"}),
        }}

    @classmethod
    def IS_CHANGED(cls, image, instruction, model, **k):
        return hashlib.sha1(image[0].cpu().numpy().tobytes()[::97] + repr((instruction, model, sorted(k.items()))).encode()).hexdigest()

    def run(self, image, instruction, model, base_negative="", control_strength=0.8, control_end=0.8,
            keep_loaded=False, control_mode="auto"):
        if control_mode == "raw":  # mode 0: straight Qwen edit, no VLM call, no control, nothing upstream is evaluated
            neg = base_negative.strip().rstrip(",")
            return (instruction.strip(), neg, 0.0, 0.0, "", json.dumps({"control_mode": "raw"}), "none", False)
        if model.startswith("("):
            raise RuntimeError("Edit Planner: no Qwen-VL model in models/LLM")
        ins = instruction.strip()
        loader = SceneContextQwenVL()
        vlm, proc = loader._load(model)
        img = _pil(image)
        small = _pil(image, 512)

        def ask(q, im=img, n=80, with_ins=False):
            q = ("Edit instruction: \"" + ins + "\"\n" + q) if with_ins else q
            return _clean(SceneContextQwenVL._gen(vlm, proc, im, SYS, q, n))

        def yes(q):
            return ask(q + " Answer yes or no.", small, 4, True).lower().startswith("yes")

        facts = {k: ask(q) for k, q in FACTS}
        et = ask("Which best describes what the instruction does to the picture? Answer with exactly one word: "
                 "outfit (replaces worn clothes or armor with different clothes), recolor (same shapes, new colours or materials), "
                 "local (a small change: an expression, one accessory, one object), background (only the setting), "
                 "pose (changes how the person stands or moves), other.", small, 6, True).lower()
        et = next((e for e in ETYPES if e in et), "other")
        silhouette = yes("Would the new result change the outline or volume of the person's body or outfit (for example heavy armor "
                         "to a slim suit, a long dress to shorts)?")
        changes_bg = yes("Does the instruction change the background or setting of the picture?")
        removed = ask("List the clothing, armor and held items VISIBLE IN THE PICTURE that the instruction removes or replaces. Include "
                      "parts of the same outfit that go with it (replacing a suit of armor removes its helmet, plates, belt, cape and "
                      "sword; replacing a dress removes the dress, its straps and ruffles only. Jewelry, watches, shoes and headwear stay unless "
                      "the instruction names them). Also list any top, shirt or other garment worn under or beneath it that the new outfit replaces. Never list the new things the instruction adds and never list background objects "
                      "or furniture. Short comma-separated nouns; answer 'none' if nothing is removed.", small, 60, True)
        adds = ask("Describe each new item, garment or setting that the instruction adds, in concrete visible terms: garment length, fit and "
                   "colour; what a held object looks like and which hand holds it; what the new setting contains. One short phrase "
                   "per item, separated by semicolons. Answer 'none' if it adds nothing.", small, 90, True)
        adds = "; ".join(a.strip() for a in adds.split(";") if a.strip() and a.strip().lower().rstrip(".") not in ("none", "n/a")) or "none"
        parts = ask("Which of these body-adjacent parts does the instruction change or remove? Answer with a comma-separated subset of: "
                    "hair, glasses, none.", small, 12, True).lower()
        keep_hair = "hair" not in parts

        drop = []
        if not keep_hair:
            drop.append("Hair")
        if "glass" in parts or "eyeglass" in removed.lower():
            drop.append("Eyeglass")
        hl = removed.lower()
        if any(w in hl for w in ("helmet", "hat", "headband", "crown", "cap", "hood")) and "Hair" not in drop:
            drop.append("Hair")  # headwear is segmented with the hair

        if control_mode in ("none", "depth"):
            mode = control_mode
        else:
            mode = "depth" if (et == "outfit" or (et == "other" and silhouette)) else "none"
        use_control = mode != "none"
        removed_l = [] if removed.lower() in ("none", "") else [x.strip() for x in removed.split(",") if x.strip()]
        # the VLM sometimes lists the NEW items; anything whose words the instruction names is not being removed
        iw = set(re.findall(r"[a-z]+", (ins + " " + (adds if adds.lower() != "none" else "")).lower()))
        removed_l = _filter_removed(removed_l, iw)
        if changes_bg and facts["scene"].lower() not in ("none", ""):  # a new setting replaces the old one, so the old scenery must not stay
            removed_l += _filter_removed([x.strip() for x in facts["scene"].split(",") if x.strip()], iw)
        removed_l = list(dict.fromkeys(removed_l))

        keep = ["the same character", "face (" + facts["face"] + ")"]
        keep.append("hairstyle (" + facts["hair"] + ")" if keep_hair else "hair colour and identity")
        keep += ["body shape and proportions (" + facts["build"] + ")", "pose (" + facts["pose"] + ")",
                 "camera angle (" + facts["camera"] + ")", "framing (" + facts["framing"] + ")", "art style (" + facts["style"] + ")"]
        if not changes_bg:
            keep.append("background (" + facts["scene"] + ")")
        sent = "Edit <image1>: " + ins.rstrip(".") + "."
        if adds.lower() not in ("none", ""):
            sent += " Details of what is added: " + adds.rstrip(".") + "."
        if removed_l:
            sent += (" Remove completely: " + ", ".join(removed_l) + " (nothing of the old outfit, held items or old setting may remain; "
                     "anything held is replaced by the new item in the same hand and grip).")
        sent += " Keep exactly as in <image1>: " + "; ".join(keep) + "."
        neg = [base_negative.strip().rstrip(",")] if base_negative.strip() else []
        neg += removed_l
        negative = ", ".join(x for x in neg if x)

        strength = float(control_strength) if use_control else 0.0
        plan = json.dumps({"edit_type": et, "changes_silhouette": silhouette, "changes_background": changes_bg,
                           "control_mode": mode, "control_strength": strength, "removes": removed_l,
                           "drop_classes": drop, "facts": facts}, indent=1, ensure_ascii=False)
        if not keep_loaded:
            SceneContextQwenVL._unload()
        return (sent, negative, strength, float(control_end), ",".join(drop), plan, mode, bool(use_control))


class _Gate:
    CATEGORY = "invoke_inpainting"
    FUNCTION = "run"
    KIND = "IMAGE"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"enable": ("BOOLEAN", {"default": True})},
                "optional": {"value": (cls.KIND, {"lazy": True})}}

    def check_lazy_status(self, enable, value=None):
        return ["value"] if enable and value is None else []


class GateImage(_Gate):
    """Passes `value` only when enabled; the nodes upstream of `value` do not run when disabled (output: a 64x64 black image)."""
    RETURN_TYPES = ("IMAGE",)

    def run(self, enable, value=None):
        return (value if enable and value is not None else torch.zeros(1, 64, 64, 3),)


class GateText(_Gate):
    KIND = "STRING"
    RETURN_TYPES = ("STRING",)

    def run(self, enable, value=None):
        return (value if enable and value is not None else "(control off)",)


class ControlSelect:
    CATEGORY = "invoke_inpainting"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mode": ("STRING", {"default": "depth", "forceInput": True})},
                "optional": {"depth_img": ("IMAGE", {"lazy": True})}}

    def check_lazy_status(self, mode, depth_img=None):
        return ["depth_img"] if mode == "depth" and depth_img is None else []

    def run(self, mode, depth_img=None):
        return (depth_img if mode == "depth" and depth_img is not None else torch.zeros(1, 64, 64, 3),)


class ImagePick:
    """Picks image a or b by a switch; only the chosen branch runs."""
    CATEGORY = "invoke_inpainting"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pick": (["a", "b"], {"default": "a"})},
                "optional": {"a": ("IMAGE", {"lazy": True}), "b": ("IMAGE", {"lazy": True})}}

    def check_lazy_status(self, pick, a=None, b=None):
        if pick == "a" and a is None:
            return ["a"]
        if pick == "b" and b is None:
            return ["b"]
        return []

    def run(self, pick, a=None, b=None):
        x = a if pick == "a" else b
        return (x[..., :3] if x is not None and x.shape[-1] > 3 else x,)  # drop alpha (the Qwen VAE decodes 4 channels)


class GateModel:
    """Picks the model that carries the control (enable) or the plain model; only the chosen branch is evaluated."""
    CATEGORY = "invoke_inpainting"
    FUNCTION = "run"
    RETURN_TYPES = ("MODEL",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"enable": ("BOOLEAN", {"default": True})},
                "optional": {"with_control": ("MODEL", {"lazy": True}), "plain": ("MODEL", {"lazy": True})}}

    def check_lazy_status(self, enable, with_control=None, plain=None):
        if enable:
            return ["with_control"] if with_control is None else []
        return ["plain"] if plain is None else []

    def run(self, enable, with_control=None, plain=None):
        return (with_control if enable else plain,)


NODE_CLASS_MAPPINGS = {"EditPlannerQwenVL": EditPlannerQwenVL, "GateImage": GateImage, "GateText": GateText, "GateModel": GateModel, "ControlSelect": ControlSelect, "ImagePick": ImagePick}
NODE_DISPLAY_NAME_MAPPINGS = {"EditPlannerQwenVL": "Edit Planner (Qwen VL)", "GateImage": "Gate Image (lazy)",
                              "GateText": "Gate Text (lazy)", "ControlSelect": "Control Select (lazy)", "ImagePick": "Image Pick a/b (lazy)", "GateModel": "Gate Model (lazy)"}
