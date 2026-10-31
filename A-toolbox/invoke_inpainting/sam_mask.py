"""Edit Mask (Qwen VL + SAM 2): turns each box's edit text into a pixel mask of the thing being changed.

The edit model redraws everything inside a box at full strength, so anything the box overlaps but the edit does not touch (an eye
under a headband box) drifts. Here Qwen3-VL (the model Scene Context already uses) is asked which existing item in the box the
instruction changes or removes and returns its bounding box; SAM 2.1 turns that box into an outline; the outline is cut to the
user's box, grown a little (the new item may be a different shape) and feathered. Wire the `mask` output into Invoke Inpainting's
`inpaint_mask`. A box where nothing sensible is found keeps the whole box (no mask), exactly as before.

SAM 2.1 Hiera-Large (Apache-2.0, ~0.9 GB) is fetched from Hugging Face into models/sam2 on first use."""
import hashlib
import json
import os
import re

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import folder_paths

from .context_vlm import SceneContextQwenVL, _models, EDIT_SYSTEM
from . import segment_layers as SL

SAM_REPO = "facebook/sam2.1-hiera-large"
SAM_DIR = os.path.join(folder_paths.models_dir, "sam2", "sam2.1-hiera-large")
_SAM = {}
_CACHE = {}

TYPE_ASK = ('The picture is being edited with the instruction: "{cap}"\n'
            "What kind of edit is it? Answer with one word: replace (one thing is changed into another), add (something new is put in), "
            "remove (something is taken away), recolour (only its colour changes), adjust (a small local change to part of a person or "
            "object, such as a facial expression, leaving everything else exactly as it is) or background (the surroundings are changed).")
SEG_YN = ('You are an image editor. The edit request is: "{cap}"\nConsider only the {label} in the image. Would the final edited '
          "image show the {label} differently from the original image? Answer yes or no.")
SEG_BG = ('The edit request is: "{cap}"\nDoes this request ask to change the setting, location or background of the picture? '
          "Answer yes or no.")
SEG_LIST = ('Edit instruction: "{cap}"\nThings in the picture: {names}.\nWhich of these things would be removed, replaced, recoloured or look '
            "different after the edit? Answer with a comma-separated list of the names only.")
SEG_KEEP = ('Edit instruction: "{cap}"\nThings in the picture: {names}.\nWhich of these things does the instruction say must stay the same '
            '(for example "keep", "stays", "do not change", "do not edit")? Only list a thing if the instruction names it with its own word in such a '
            'statement. A pose, proportions or position do not count: never list hands, arms or legs because of them. Answer with a comma-separated '
            'list of the names only, or none.')
SEG_WORN = ('Is a "{label}" something a person wears (clothing, armour, footwear, headwear, an accessory or jewellery)? Answer yes or no.')
SEG_POSE = ("The edit request is: \"{cap}\"\nDoes this request ask to change the pose, posture, stance or body or head position of the person? "
            "Answer yes or no.")
SEG_BODY = ('Is a "{label}" a part of a person\'s own body (not something they wear, hold or carry)? Answer yes or no.')
SEG_FATE = ('Edit instruction: "{cap}"\nWhat happens to the {label} in the picture? Answer with exactly one of: "removed", '
            '"replaced by <what takes its place>", "recoloured to <colour>", or "stays".')
SEG_OUTFIT = ('The edit request is: "{cap}"\nDoes this request ask to change what the person is wearing (their clothes, armour, shoes or '
              "accessories)? Answer yes or no.")
SEG_TARGET = ('Edit instruction: "{cap}"\nName the thing or things in the picture that this instruction changes, as short names separated '
              "by commas. Names only.")
GROUND = ('This crop is from a picture that is being edited with the instruction: "{cap}"\n'
          "Locate the existing item that this instruction changes or removes (for example the piece of clothing, accessory, hair "
          "or object it names) and give its bounding box. Include all of that item that is visible in the crop, and nothing else. "
          'Output only JSON like [{{"bbox_2d": [x1, y1, x2, y2], "label": "short name"}}] in 0-1000 relative coordinates. '
          "If you cannot find it, output [].")


PARTS_ASK = ("List every separate part of each person's body that is clearly visible in this crop (such as the face, hair, hands) and every "
             "separate item they wear or hold, as short names separated by commas. Use the most specific name for each piece and never "
             "list a group together with its members. Names only.")
GROUND_PART = ('Locate the "{name}" in this crop with a tight box. Output only JSON like [{{"bbox_2d": [x1, y1, x2, y2], '
               '"label": "{name}"}}] in 0-1000 relative coordinates. If you cannot find it, output [].')
INV_ASK = ("List every distinct visible part of the person and every distinct object inside this crop, one entry each, with a tight "
           'box around it. Output only JSON like [{{"bbox_2d": [x1, y1, x2, y2], "label": "short name"}}] in 0-1000 relative '
           "coordinates. {cap_unused}")
YN_ASK = ('An image-editing instruction for a photo of a person: "{cap}"\n'
          'After this edit is done, will any part of the person\'s {label} look different from before, even partly (because the instruction covers it with something '
          "new, replaces it, removes it or recolours it)? Answer with one word: yes or no.")


def _parse_boxes(raw, w, h):
    """Every {"bbox_2d": [...], "label": ...} object in the text, even if the list is cut off at the end."""
    out = []
    for m in re.finditer(r"\{[^{}]*\}", raw):
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        b = d.get("bbox_2d") or d.get("bbox")
        if b and len(b) == 4:
            x1, y1, x2, y2 = b[0] / 1000 * w, b[1] / 1000 * h, b[2] / 1000 * w, b[3] / 1000 * h
            if x2 > x1 and y2 > y1:
                out.append(([x1, y1, x2, y2], str(d.get("label", ""))[:40]))
    return out


def _gen_text(vlm, proc, system, ask, ntok):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": [{"type": "text", "text": ask}]}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], padding=True, return_tensors="pt")
    inputs = {k: (v.to(vlm.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    with torch.no_grad():
        out = vlm.generate(**inputs, max_new_tokens=ntok, do_sample=False, pad_token_id=proc.tokenizer.eos_token_id)
    return proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def _yes_prob(vlm, proc, crop, system, ask):
    """Probability (0-1) that the VLM's first answer word is 'yes' rather than 'no'."""
    content = ([{"type": "image"}] if crop is not None else []) + [{"type": "text", "text": ask}]
    messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if crop is not None:
        inputs = proc(text=[text], images=[crop], padding=True, return_tensors="pt")
    else:
        inputs = proc(text=[text], padding=True, return_tensors="pt")
    inputs = {k: (v.to(vlm.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    with torch.no_grad():
        out = vlm.generate(**inputs, max_new_tokens=1, do_sample=False, output_scores=True, return_dict_in_generate=True,
                           pad_token_id=proc.tokenizer.eos_token_id)
    lg = out.scores[0][0].float()
    tk = proc.tokenizer

    def ids(words):
        r = []
        for w in words:
            t = tk.encode(w, add_special_tokens=False)
            if t:
                r.append(t[0])
        return r

    y, n = ids(["yes", "Yes", "YES"]), ids(["no", "No", "NO"])
    py = torch.logsumexp(lg[y], 0)
    pn = torch.logsumexp(lg[n], 0)
    return float(torch.sigmoid(py - pn))


def _sam_files():
    return [os.path.join(SAM_DIR, f) for f in ("config.json", "model.safetensors")]


def _ensure_sam():
    if all(os.path.isfile(f) for f in _sam_files()):
        return SAM_DIR
    from huggingface_hub import snapshot_download

    print("[MidFR] downloading %s (about 0.9 GB) to %s" % (SAM_REPO, SAM_DIR), flush=True)
    os.makedirs(SAM_DIR, exist_ok=True)
    snapshot_download(SAM_REPO, local_dir=SAM_DIR, allow_patterns=["*.json", "model.safetensors"])
    return SAM_DIR


def _load_sam(device):
    if "m" not in _SAM:
        from transformers import Sam2Model, Sam2Processor

        path = _ensure_sam()
        _SAM["p"] = Sam2Processor.from_pretrained(path)
        _SAM["m"] = Sam2Model.from_pretrained(path).eval()
    _SAM["m"].to(device)
    return _SAM["m"], _SAM["p"]


def _unload_sam():
    if "m" in _SAM:
        _SAM["m"].to("cpu")
    _SAM.clear()
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _window(rect, W, H, long_side=1024):
    """The box plus as much context again on every side (3x the box), clipped to the picture."""
    x0, y0, x1, y1 = rect
    pad = max(48, x1 - x0, y1 - y0)
    return max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)


def _parse_box(raw, w, h):
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    for d in data if isinstance(data, list) else [data]:
        b = d.get("bbox_2d") or d.get("bbox") if isinstance(d, dict) else None
        if b and len(b) == 4:
            x1, y1, x2, y2 = b[0] / 1000 * w, b[1] / 1000 * h, b[2] / 1000 * w, b[3] / 1000 * h
            if x2 > x1 and y2 > y1:
                return [x1, y1, x2, y2]
    return None


def _hull_fill(m, b):
    """Convex hull of a part's mask, cut to the part's box. SAM often returns only the hair of a head (the face sits in the
    hole) or one finger group of a hand; the hull covers the whole part. Pure numpy/PIL."""
    from PIL import ImageDraw
    ys, xs = torch.nonzero(m > 0.5, as_tuple=True)
    if len(xs) < 20:
        return m
    st = max(1, len(xs) // 4000)
    pts = sorted(set(zip(xs[::st].tolist(), ys[::st].tolist())))

    def half(P):
        h = []
        for q in P:
            while len(h) >= 2 and (h[-1][0] - h[-2][0]) * (q[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (q[0] - h[-2][0]) <= 0:
                h.pop()
            h.append(q)
        return h

    hull = half(pts)[:-1] + half(pts[::-1])[:-1]
    if len(hull) < 3:
        return m
    im = Image.new("L", (m.shape[1], m.shape[0]), 0)
    ImageDraw.Draw(im).polygon(hull, fill=255)
    h = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)
    cut = torch.zeros_like(h)
    cut[max(0, int(b[1])):int(b[3]) + 1, max(0, int(b[0])):int(b[2]) + 1] = 1.0
    return torch.maximum(m, h * cut)


def _blur(a, sigma):
    if sigma <= 0:
        return a
    k = int(sigma * 3) * 2 + 1
    x = torch.arange(k, dtype=torch.float32) - k // 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    t = a[None, None]
    t = F.conv2d(F.pad(t, (k // 2, k // 2, 0, 0), mode="replicate"), g.view(1, 1, 1, k))
    t = F.conv2d(F.pad(t, (0, 0, k // 2, k // 2), mode="replicate"), g.view(1, 1, k, 1))
    return t[0, 0]


def _grow(m, px):
    if px <= 0:
        return m
    k = 2 * px + 1
    return F.max_pool2d(m[None, None], k, 1, px)[0, 0]


def _eyedrop_grow(arr, base, box, reach, tol=3.0):
    """Eyedropper: sample the colours deep inside `base` (the old item), then grow the mask into neighbouring pixels of that colour
    that are connected to it, inside `box`, within `reach` px. Catches thin or uncertain parts the outline missed."""
    try:
        import cv2
    except Exception:
        return base
    core = -F.max_pool2d(-base[None, None], 13, 1, 6)[0, 0] > 0.5
    if int(core.sum()) < 50:
        core = base > 0.5
    if int(core.sum()) < 50:
        return base
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
    cm = core.numpy()
    samples = lab[cm]
    if len(samples) > 20000:
        samples = samples[np.random.RandomState(0).choice(len(samples), 20000, replace=False)]
    mean = samples.mean(0)
    cov = np.cov(samples.T) + np.eye(3) * 4.0
    inv = np.linalg.inv(cov)
    d = lab - mean
    d2 = np.einsum("hwi,ij,hwj->hw", d, inv, d)
    cand = torch.from_numpy(d2 < tol * tol).float() * box
    cur = (base > 0.5).float()
    for _ in range(int(reach)):
        nxt = torch.maximum(cur, F.max_pool2d(cur[None, None], 3, 1, 1)[0, 0] * cand)
        if float((nxt - cur).sum()) == 0:
            break
        cur = nxt
    return cur


def _colour_d2(arr, core_src, erode=6):
    """Squared Mahalanobis distance of every pixel's colour (Lab) from the colours deep inside `core_src`; None if too little to sample."""
    try:
        import cv2
    except Exception:
        return None
    core = (-F.max_pool2d(-core_src[None, None], 2 * erode + 1, 1, erode)[0, 0]) > 0.5
    if int(core.sum()) < 50:
        core = core_src > 0.5
    if int(core.sum()) < 50:
        return None
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
    samples = lab[core.numpy()]
    if len(samples) > 20000:
        samples = samples[np.random.RandomState(0).choice(len(samples), 20000, replace=False)]
    mean = samples.mean(0)
    inv = np.linalg.inv(np.cov(samples.T) + np.eye(3) * 4.0)
    d = lab - mean
    return np.einsum("hwi,ij,hwj->hw", d, inv, d)


def _close_and_clean(m, close_px=3, min_frac=0.002):
    t = m[None, None]
    k = 2 * close_px + 1
    t = -F.max_pool2d(-F.max_pool2d(t, k, 1, close_px), k, 1, close_px)
    m = t[0, 0]
    if float(m.sum()) < min_frac * m.numel():
        return m
    return m


class EditMaskQwenSAM:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "passes": ("STRING", {"forceInput": True, "tooltip": "The bbox node's passes output (or Scene Context's)"}),
                "model": (_models(), {"tooltip": "A Qwen3-VL folder in models/LLM (finds the item each instruction changes)"}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional: a mask you painted; it is added to what the VLM finds"}),
                "prompt": ("STRING", {"forceInput": True, "tooltip":
                    "Used when there are no boxes: the whole image is then treated as one box with this instruction"}),
                "grow": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip":
                    "Percent of the picture's short side to grow the found outline by (the new item can be a different shape; "
                    "thin items like bands and straps always get at least 6 px)"}),
                "feather": ("INT", {"default": 4, "min": 0, "max": 64, "tooltip": "Soft edge of the mask in px"}),
                "mode": (["segment", "auto", "mask", "region"], {"tooltip":
                    "segment (default): the picture is split into named components (scenery by OneFormer, parts and items by SAM 3, cleaned up "
                    "by SAM 2.1) and the VLM rules each in or out for the edit. The older modes: "
                    "auto: the VLM reads the edit type. 'change X to Y' / add repaint the whole box (minus the parts that stay); remove, "
                    "recolour and background repaint only the outline of the item. mask / region force one for every box"}),
                "protect": ("BOOLEAN", {"default": True, "tooltip":
                    "Big boxes: the VLM lists the parts inside the box that the instruction does not change (e.g. a face the box overlaps) "
                    "and SAM cuts them out of the repaint area (blue in the preview)"}),
                "eyedrop": ("BOOLEAN", {"default": True, "tooltip":
                    "After SAM, sample the old item's colours and grow the mask into connected pixels of the same colour (catches thin "
                    "parts like frills that SAM missed)"}),
                "grow_box": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.5, "tooltip":
                    "Region mode: grow the box by this percent of its size on every side, so the new item has room to continue past it "
                    "(a collar, a hem). The parts that stay (face, hands) are still protected"}),
                "side_above": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip":
                    "Region mode: if the VLM thinks the new item must continue past a box edge with at least this chance, that side of the "
                    "box grows by 12% (the grown box is returned in `passes`, wire it on to Scene Context / Invoke)"}),
                "keep_below": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip":
                    "Region mode: a part the VLM thinks has less than this chance of being changed by the edit is protected"}),
                "keep_loaded": ("BOOLEAN", {"default": False, "tooltip": "Keep the VLM in memory"}),
                "lock_labels": ("STRING", {"default": "face, hair", "tooltip": "Words that name the parts the `kept` output protects (any detected label containing one of them). Everything else is left for the next stage to regenerate"}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE", "STRING", "STRING", "MASK", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("mask", "preview", "log", "passes", "margin", "hide", "hide_scene", "kept")
    FUNCTION = "run"
    CATEGORY = "image/midfr"

    @classmethod
    def IS_CHANGED(cls, image, passes, model, mask=None, prompt="", grow=1.5, feather=4, mode="segment", protect=True, eyedrop=True, keep_below=0.5, side_above=0.7, grow_box=6.0, keep_loaded=False, lock_labels="face, hair"):
        h = hashlib.sha1(image[0].cpu().numpy().tobytes()[::97] + repr((passes, prompt, model, grow, feather, mode, protect, eyedrop, keep_below, side_above, grow_box, lock_labels)).encode())
        if mask is not None:
            h.update(mask.cpu().numpy().tobytes()[::97])
        return h.hexdigest()

    def _sam_one(self, sam, sproc, full, box, rect, W, H, dev):
        """SAM outline of `box`, computed on an enlarged crop (box + context) so thin parts are resolved; returns a full-size 0/1 mask."""
        # tight window around the target itself (SAM resizes its input to 1024 px, so a tighter window = more detail per pixel),
        # enlarged to about 1.5 MP
        bw, bh = box[2] - box[0], box[3] - box[1]
        pad = max(24, int(0.3 * max(bw, bh)))
        wx0, wy0 = max(0, int(box[0]) - pad), max(0, int(box[1]) - pad)
        wx1, wy1 = min(W, int(box[2]) + pad), min(H, int(box[3]) + pad)
        crop = full.crop((wx0, wy0, wx1, wy1))
        k = min(4.0, max(1.0, (1.5e6 / (crop.width * crop.height)) ** 0.5))
        crop = crop.resize((max(1, round(crop.width * k)), max(1, round(crop.height * k))), Image.LANCZOS)
        cb = [(box[0] - wx0) * k, (box[1] - wy0) * k, (box[2] - wx0) * k, (box[3] - wy0) * k]
        inp = sproc(images=crop, input_boxes=[[[float(v) for v in cb]]], return_tensors="pt")
        inp = {kk: (v.to(dev) if isinstance(v, torch.Tensor) else v) for kk, v in inp.items()}
        with torch.inference_mode():
            if dev == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = sam(**inp, multimask_output=False)
            else:
                o = sam(**inp, multimask_output=False)
        m = sproc.post_process_masks(o.pred_masks.float().cpu(), inp["original_sizes"])[0]
        m = m.reshape(-1, m.shape[-2], m.shape[-1])[0].float()
        m = (m > 0).float() if (m.max() > 1 or m.min() < 0) else (m > 0.5).float()
        m = F.interpolate(m[None, None], size=(wy1 - wy0, wx1 - wx0), mode="bilinear")[0, 0]
        full_m = torch.zeros((H, W))
        full_m[wy0:wy1, wx0:wx1] = (m > 0.5).float()
        return full_m

    def _run_segment(self, image, todo, model, full, arr, out, W, H, dev, feather, keep_below, grow_box, keep_loaded):
        log = []
        ents = []
        for p in todo:
            r = [int(v) for v in p["rect"]]
            r0 = list(r)
            bw, bh = r[2] - r[0], r[3] - r[1]
            mx, my = int(grow_box / 100.0 * bw), int(grow_box / 100.0 * bh)
            r = [max(0, r[0] - mx), max(0, r[1] - my), min(W, r[2] + mx), min(H, r[3] + my)]
            p["rect"] = r
            ents.append({"rect": r, "rect0": r0, "cap": (p.get("caption") or "").strip()})
        # layer 0: the scenery
        bg, person = SL.background_layer(full)
        bgmask = np.zeros((H, W), bool)
        for m, _ in bg:
            bgmask |= m
        bgnames = []
        for _, n in bg:
            if n not in bgnames:
                bgnames.append(n)
        # the VLM: names the parts and items it sees, reads each box's edit type, and rules every name in or out
        vlm, proc = SceneContextQwenVL()._load(model)
        s = min(3.0, (1.5e6 / (W * H)) ** 0.5)
        crop = full.resize((max(1, round(W * s)), max(1, round(H * s))), Image.LANCZOS) if abs(s - 1) > 0.02 else full
        names = []
        for nm in re.split(r"[,\n;]", SceneContextQwenVL._gen(vlm, proc, crop, EDIT_SYSTEM, PARTS_ASK, 120)):
            nm = re.sub(r"^[\s\-\*\d\.\)]+", "", nm).strip(" .\"'").lower()
            if 2 <= len(nm) <= 30 and len(nm.split()) <= 3 and nm != "person" and nm not in names:
                names.append(nm)
        names = names[:16]
        for e in ents:
            e["targets"] = []
            if e["cap"]:
                kt = (model, e["cap"], "targets")
                if kt not in _CACHE:
                    _CACHE[kt] = _gen_text(vlm, proc, EDIT_SYSTEM, SEG_TARGET.format(cap=e["cap"]), 40)
                for nm in re.split(r"[,\n;]", _CACHE[kt]):
                    nm = re.sub(r"^[\s\-\*\d\.\)]+", "", nm).strip(" .\"'").lower()
                    if 2 <= len(nm) <= 30 and len(nm.split()) <= 3 and nm != "person":
                        e["targets"].append(nm)
                        if nm not in names and len(names) < 20:
                            names.append(nm)
        allnames = list(names) + [n for n in bgnames if n not in names]
        for e in ents:
            e["p"] = {}
            if not e["cap"]:
                continue
            wx0, wy0, wx1, wy1 = _window(e["rect"], W, H)
            c2 = full.crop((wx0, wy0, wx1, wy1))
            s2 = min(3.0, (1.5e6 / (c2.width * c2.height)) ** 0.5)
            if abs(s2 - 1) > 0.02:
                c2 = c2.resize((max(1, round(c2.width * s2)), max(1, round(c2.height * s2))), Image.LANCZOS)
            e["kind"] = SceneContextQwenVL._gen(vlm, proc, c2, EDIT_SYSTEM, TYPE_ASK.format(cap=e["cap"]), 6).strip().lower()
            qb = SEG_BG.format(cap=e["cap"])
            kb = (model, qb, "p-text")
            if kb not in _CACHE:
                _CACHE[kb] = _yes_prob(vlm, proc, None, EDIT_SYSTEM, qb)
            e["pbg"] = _CACHE[kb]
            # the VLM sees the whole list together with the instruction (so a skirt replacing a red dress takes the dress with it)
            ql = SEG_LIST.format(cap=e["cap"], names=", ".join(allnames))
            kl = (model, ql, "list")
            if kl not in _CACHE:
                _CACHE[kl] = _gen_text(vlm, proc, EDIT_SYSTEM, ql, 100)
            listed = {re.sub(r"^[\s\-\*\d\.\)]+", "", x).strip(" .\"'").lower() for x in re.split(r"[,\n;]", _CACHE[kl])}
            qk = SEG_KEEP.format(cap=e["cap"], names=", ".join(allnames))
            kk_ = (model, qk, "keep")
            if kk_ not in _CACHE:
                _CACHE[kk_] = _gen_text(vlm, proc, EDIT_SYSTEM, qk, 60)
            stay = {re.sub(r"^[\s\-\*\d\.\)]+", "", x).strip(" .\"'").lower() for x in re.split(r"[,\n;]", _CACHE[kk_])}
            qo = SEG_OUTFIT.format(cap=e["cap"])
            ko = (model, qo, "p-text")
            if ko not in _CACHE:
                _CACHE[ko] = _yes_prob(vlm, proc, None, EDIT_SYSTEM, qo)
            e["poutfit"] = _CACHE[ko]
            for nm in allnames:
                kw = (model, nm, "worn")
                if kw not in _CACHE:
                    _CACHE[kw] = _yes_prob(vlm, proc, None, EDIT_SYSTEM, SEG_WORN.format(label=nm))
                q = SEG_YN.format(cap=e["cap"], label=nm)
                k = (model, q, "p-text")
                if k not in _CACHE:
                    _CACHE[k] = _yes_prob(vlm, proc, None, EDIT_SYSTEM, q)
                pv = _CACHE[k]
                if nm in bgnames and nm not in names and e["pbg"] >= 0.5:
                    pv = 1.0  # the setting changes: all the scenery goes
                elif nm in listed:
                    pv = max(pv, 1.0)  # named by the VLM when it saw the whole list
                elif e["poutfit"] >= 0.5 and _CACHE[kw] >= 0.5:
                    pv = max(pv, 1.0)  # the outfit changes: anything worn is the same kind of thing and goes with it
                if nm in stay and nm not in ("none", ""):
                    pv = 0.0  # the instruction says it stays: that wins over every guess above
                e["p"][nm] = pv
            # every worn or held item that is replaced or removed is named in the edit prompt, so the model does not keep it
            cand = []
            for nm in names:
                kd = (model, nm, "body")
                if kd not in _CACHE:
                    _CACHE[kd] = _yes_prob(vlm, proc, None, EDIT_SYSTEM, SEG_BODY.format(label=nm))
                if e["p"].get(nm, 0.0) < keep_below:
                    continue
                if _CACHE[kd] < 0.5:
                    cand.append(nm)
            cand = [n for n in cand if not any(o != n and o in n for o in cand)][:8]
            gone, swapped = [], []
            for nm in cand:
                qf = SEG_FATE.format(cap=e["cap"], label=nm)
                kf = (model, qf, "fate")
                if kf not in _CACHE:
                    _CACHE[kf] = _gen_text(vlm, proc, EDIT_SYSTEM, qf, 24).strip().strip(".\"' ").lower()
                f = _CACHE[kf]
                if f.startswith("remov"):
                    gone.append(nm)
                elif f.startswith("replaced by"):
                    swapped.append("the %s is replaced by %s, in exactly the same position and grip as the old one" % (nm, f[len("replaced by"):].strip(" <>\"'") or "the new item"))
                elif f.startswith("recoloured to"):
                    swapped.append("the %s becomes %s" % (nm, f[len("recoloured to"):].strip(" <>\"'")))
            e["hide_names"] = list(gone) + [n for n in cand if any(x.startswith("the %s is replaced" % n) or x.startswith("the %s becomes" % n) for x in swapped)]
            txt = []
            if gone:
                txt.append("%s %s removed completely (anything new the instruction puts there takes %s place, otherwise what %s covered is shown naturally)" % (
                    "the " + ", the ".join(gone), "is" if len(gone) == 1 else "are", "its" if len(gone) == 1 else "their", "it" if len(gone) == 1 else "they"))
            txt += swapped
            e["items"] = ("Worn and held items: " + "; ".join(txt) + ".") if txt else ""
            kp = (model, e["cap"], "pose")
            if kp not in _CACHE:
                _CACHE[kp] = _yes_prob(vlm, proc, None, EDIT_SYSTEM, SEG_POSE.format(cap=e["cap"]))
            if _CACHE[kp] < 0.5:  # no pose change asked for: she stays exactly where she is, so the kept face still lines up
                e["items"] = (e["items"] + " Keep the person's pose, head size and position, hair volume, body proportions and the place of both hands exactly as in the original; removing an item never makes the head or hair bigger.").strip()
        if not keep_loaded:
            del vlm, proc
            SceneContextQwenVL._unload()
        # layer 1: SAM 3 paints the parts and items; layer 2: SAM 2.1 cleans them up
        items, dropped = SL.sam3_layer(full, names, dev, bg) if names else ([], [])
        raw = {}
        for m, c in items:
            raw[c] = raw.get(c, np.zeros((H, W), bool)) | m
        log.append("SAM 3 found: " + ", ".join("%s x%d" % (c, sum(1 for _, cc in items if cc == c)) for c in dict.fromkeys(c for _, c in items)) + (" | dropped as scenery: " + ", ".join(dropped) if dropped else ""))
        lab, lnames = SL.label_map(items, (H, W))
        sam, sproc = _load_sam(dev)
        segs = SL.sam2_segments(sam, sproc, full, dev)
        _unload_sam()
        lab = SL.snap(lab, segs)
        # (person pixels nobody labelled stay unlabelled: repainted in region mode, left alone in mask mode)
        byname = {}
        for i, n in enumerate(lnames):
            byname[n] = lab == i
        for m, n in bg:
            byname[n] = byname.get(n, np.zeros((H, W), bool)) | (m & (lab < 0))
        # per box: repaint area
        keepout = torch.zeros((H, W))
        hidem = torch.zeros((H, W))
        hides = torch.zeros((H, W))
        keptm = torch.zeros((H, W))
        for e in ents:
            x0, y0, x1, y1 = e["rect"]
            box = torch.zeros((H, W))
            box[y0:y1, x0:x1] = 1.0
            barea = (x1 - x0) * (y1 - y0)
            if not e["cap"] or not e["p"]:
                log.append("rect %s: no instruction -> whole box" % (e["rect"],))
                out = torch.maximum(out, box)
                continue
            chg = np.zeros((H, W), bool)
            kep = np.zeros((H, W), bool)
            chg_n, kep_n = [], []
            for nm, pr in e["p"].items():
                m = byname.get(nm)
                if m is None or not (m[y0:y1, x0:x1]).any():
                    continue
                if pr >= keep_below:
                    chg |= m
                    chg_n.append("%s=%.2f" % (nm, pr))
                else:
                    kep |= m
                    kep_n.append("%s=%.2f" % (nm, pr))
            region = e["kind"].startswith(("replace", "add", "change"))
            if region:
                kep = np.zeros((H, W), bool)
                for nm, pr in e["p"].items():
                    m = byname.get(nm)
                    if m is not None and pr < keep_below and nm in names:
                        kep |= m  # only parts and items are protected; scenery inside the box is simply repainted
            else:
                tg = [t for t in e.get("targets", []) if t in byname and byname[t][y0:y1, x0:x1].any()]
                if tg:
                    chg = np.zeros((H, W), bool)
                    for t in tg:
                        chg |= byname[t] | raw.get(t, np.zeros((H, W), bool))  # the whole detection, even where another name sits on top
                    chg_n = ["%s (target)" % t for t in tg]
            if region and e.get("hide_names"):
                # a kept part (the face) stops short of an item that is being removed (the helmet): its shadow and edge sit on the
                # part's border, and pasting those original pixels back would keep a mark of the old item
                gone_m = np.zeros((H, W), bool)
                for nm in e["hide_names"]:
                    gone_m |= byname.get(nm, np.zeros((H, W), bool)) | raw.get(nm, np.zeros((H, W), bool))
                k_ = max(3, int(round(0.012 * min(H, W))))
                gm = F.max_pool2d(torch.from_numpy(gone_m.astype(np.float32))[None, None], 2 * k_ + 1, 1, k_)[0, 0].numpy() > 0.5
                kep = kep & ~gm
            _kb = torch.zeros((H, W))
            _kb[y0:y1, x0:x1] = 1.0
            keptm = torch.maximum(keptm, _blur(torch.from_numpy(kep.astype(np.float32)), max(1.0, feather / 2.0)) * _kb)
            if region:
                ex = torch.from_numpy(kep.astype(np.float32))
                ex = -F.max_pool2d(-ex[None, None], 5, 1, 2)[0, 0]  # shrink 2 px so the kept edge is repainted
                ex = _blur(ex, max(1.0, feather / 2.0))
                m = box * (1 - ex)
                keepout = torch.maximum(keepout, ex * box)
            else:
                g = max(6, int(round(1.5 / 100.0 * min(H, W))))
                m = _grow(_close_and_clean(torch.from_numpy(chg.astype(np.float32)) * box), g) * box
                m = _blur(m, feather / 2.0) * box
            inbox = np.zeros((H, W), bool)
            inbox[y0:y1, x0:x1] = True
            scene_b = (bgmask & (lab < 0) & inbox) if e.get("pbg", 0.0) >= 0.5 else np.zeros((H, W), bool)  # the setting changes: old scenery
            body_m = np.zeros((H, W), bool)
            for nm in names:
                if _CACHE.get((model, nm, "body"), 0.0) >= 0.5 and nm in byname:
                    body_m |= byname[nm]  # body parts (hands) keep their shape in the reference
            hb = (m.numpy() > 0.5) & ~scene_b & ~body_m & inbox  # everything else that is repainted: items, clothes, unlabelled bits
            for nm in e.get("hide_names", []):
                hb |= (byname.get(nm, np.zeros((H, W), bool)) | raw.get(nm, np.zeros((H, W), bool))) & inbox
            hidem = torch.maximum(hidem, torch.from_numpy(hb.astype(np.float32)))
            hides = torch.maximum(hides, torch.from_numpy(scene_b.astype(np.float32)))
            out = torch.maximum(out, m)
            log.append("rect %s (%s): change [%s] keep [%s] repaint %.0f%% of box" % (
                e["rect"], e["kind"], ", ".join(chg_n), ", ".join(kep_n), 100 * float((m > 0.5).sum()) / barea))
        for p_, e_ in zip(todo, ents):
            if e_.get("items"):
                p_["items"] = e_["items"]
            log.append("items: " + e_.get("items", ""))
        toks = [t.strip().lower() for t in (getattr(self, "_lock", "face, hair") or "").split(",") if t.strip()]
        ident = np.zeros((H, W), bool)
        for nm_, m_ in byname.items():
            if any(t in nm_.lower() for t in toks):
                ident |= m_
        keptm = _blur(torch.from_numpy(ident.astype(np.float32)), max(1.0, feather / 2.0))
        mm_ = out.numpy()[..., None]
        kk = keepout.numpy()[..., None]
        overlay = arr.copy().astype(np.float32)
        overlay = overlay * (1 - 0.55 * mm_) + np.array([255, 40, 40], dtype=np.float32) * 0.55 * mm_
        overlay = overlay * (1 - 0.55 * kk) + np.array([40, 120, 255], dtype=np.float32) * 0.55 * kk
        prev = torch.from_numpy(np.clip(overlay, 0, 255).astype(np.float32) / 255.0)[None]
        print("[MidFR] edit mask (segment):\n" + "\n".join(log), flush=True)
        user = torch.zeros((H, W))
        for e in ents:
            x0, y0, x1, y1 = e["rect0"]
            user[y0:y1, x0:x1] = 1.0
        margin = (out * (1 - user)).clamp(0, 1)
        return (out[None].clamp(0, 1), prev, "\n".join(log), json.dumps(todo, separators=(",", ":")), margin[None], hidem[None], hides[None], keptm[None])

    def run(self, image, passes, model, mask=None, prompt="", grow=1.5, feather=4, mode="segment", protect=True, eyedrop=True, keep_below=0.5, side_above=0.7, grow_box=6.0, keep_loaded=False, lock_labels="face, hair"):
        self._lock = lock_labels
        try:
            todo = json.loads(passes) if (passes or "").strip() else []
        except Exception:
            todo = []
        _, H, W, _ = image.shape
        if not todo and (prompt or "").strip():
            todo = [{"rect": [0, 0, W, H], "caption": prompt.strip()}]  # no boxes: the whole image is the region
        out = torch.zeros((H, W), dtype=torch.float32)
        if mask is not None:
            m = mask.reshape(-1, mask.shape[-2], mask.shape[-1])[0].float().cpu()
            if m.shape != (H, W):
                m = F.interpolate(m[None, None], size=(H, W), mode="bilinear")[0, 0]
            out = torch.maximum(out, m)
        log = []
        if not todo:
            return (out[None], image, "", passes, torch.zeros_like(out)[None], torch.zeros_like(out)[None], torch.zeros_like(out)[None], torch.zeros_like(out)[None])
        if model.startswith("("):
            raise RuntimeError("Edit Mask: no Qwen-VL model in models/LLM")
        try:
            import comfy.model_management as mm
            mm.unload_all_models()
            mm.soft_empty_cache()
        except Exception:
            pass
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        arr = (image[0].detach().float().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)[..., :3]
        full = Image.fromarray(arr)
        if mode == "segment":
            return self._run_segment(image, todo, model, full, arr, out, W, H, dev, feather, keep_below, grow_box, keep_loaded)
        # 1) the VLM looks at each box: small box -> the item the instruction changes; big box -> the parts it does not change
        vlm = proc = None
        found = []
        for p in todo:
            rect = [int(v) for v in p["rect"]]
            ent = {"rect": rect, "inc": None, "item": None, "exc": [], "asked": [], "status": "ok", "big": False, "rect0": list(rect)}
            found.append(ent)
            cap = (p.get("caption") or "").strip()
            if not cap:
                ent["status"] = "no instruction"
                continue
            wx0, wy0, wx1, wy1 = _window(rect, W, H)
            crop = full.crop((wx0, wy0, wx1, wy1))
            s = min(3.0, (1.5e6 / (crop.width * crop.height)) ** 0.5)  # about 1.5 MP for the VLM
            if abs(s - 1) > 0.02:
                crop = crop.resize((max(1, round(crop.width * s)), max(1, round(crop.height * s))), Image.LANCZOS)
            def _ask(tmpl, ntok):
                nonlocal vlm, proc
                a = tmpl.format(cap=cap)
                k = (hashlib.sha1(np.asarray(crop).tobytes()[::53]).hexdigest(), model, a)
                if k not in _CACHE:
                    if vlm is None:
                        vlm, proc = SceneContextQwenVL()._load(model)
                    _CACHE[k] = SceneContextQwenVL._gen(vlm, proc, crop, EDIT_SYSTEM, a, ntok)
                return _CACHE[k]

            # edit type decides the mask: change X to Y / add -> the whole box minus what stays; remove / recolour / background -> just the item
            kind = _ask(TYPE_ASK, 6).strip().lower() if mode == "auto" else mode
            ent["kind"] = kind
            ent["big"] = mode == "region" or (mode == "auto" and kind.startswith(("replace", "add", "change")))
            if ent["big"]:
                # does the new item need room beyond the box? ask per side; a flagged side grows by 12% of the box size
                grow_log = []
                bw0, bh0 = rect[2] - rect[0], rect[3] - rect[1]
                rel = [round((rect[0] - wx0) / (wx1 - wx0) * 1000), round((rect[1] - wy0) / (wy1 - wy0) * 1000),
                       round((rect[2] - wx0) / (wx1 - wx0) * 1000), round((rect[3] - wy0) / (wy1 - wy0) * 1000)]
                newr = list(rect)
                for si, side in enumerate(("left", "top", "right", "bottom")):
                    if (si == 0 and rect[0] <= 0) or (si == 1 and rect[1] <= 0) or (si == 2 and rect[2] >= W) or (si == 3 and rect[3] >= H):
                        continue
                    q = ('The picture is being edited inside the box [{a}, {b}, {c}, {d}] (0-1000 relative coordinates) with the '
                         'instruction: "{cap}"\nThe edit can only change pixels inside the box. Would the new item need to continue beyond '
                         'the {side} edge of the box to look natural? Answer with one word: yes or no.').format(
                        a=rel[0], b=rel[1], c=rel[2], d=rel[3], cap=cap, side=side)
                    kq = (hashlib.sha1(np.asarray(crop).tobytes()[::53]).hexdigest(), model, q, "side")
                    if kq not in _CACHE:
                        if vlm is None:
                            vlm, proc = SceneContextQwenVL()._load(model)
                        _CACHE[kq] = _yes_prob(vlm, proc, crop, EDIT_SYSTEM, q)
                    grow_log.append("%s=%.2f" % (side, _CACHE[kq]))
                    if _CACHE[kq] > side_above:
                        if si == 0:
                            newr[0] = max(0, int(rect[0] - 0.12 * bw0))
                        elif si == 1:
                            newr[1] = max(0, int(rect[1] - 0.12 * bh0))
                        elif si == 2:
                            newr[2] = min(W, int(rect[2] + 0.12 * bw0))
                        else:
                            newr[3] = min(H, int(rect[3] + 0.12 * bh0))
                if grow_box > 0:
                    mx, my = int(grow_box / 100.0 * bw0), int(grow_box / 100.0 * bh0)
                    newr = [max(0, newr[0] - mx), max(0, newr[1] - my), min(W, newr[2] + mx), min(H, newr[3] + my)]
                ent["grow"] = grow_log
                if newr != rect:
                    rect = newr
                    ent["rect"] = rect
                    p["rect"] = rect
            if ent["big"] and not protect:
                ent["status"] = "%s: whole box" % kind
                continue
            if ent["big"]:
                ib = _parse_boxes(_ask(GROUND, 160), wx1 - wx0, wy1 - wy0)
                if ib:
                    b = ib[0][0]
                    ent["item"] = [b[0] + wx0, b[1] + wy0, b[2] + wx0, b[3] + wy0]
            if ent["big"]:
                def _g(a, ntok):
                    nonlocal vlm, proc
                    k = (hashlib.sha1(np.asarray(crop).tobytes()[::53]).hexdigest(), model, a)
                    if k not in _CACHE:
                        if vlm is None:
                            vlm, proc = SceneContextQwenVL()._load(model)
                        _CACHE[k] = SceneContextQwenVL._gen(vlm, proc, crop, EDIT_SYSTEM, a, ntok)
                    return _CACHE[k]

                # the VLM names the separate parts it sees (plain text is easier for a small model than boxes), then boxes each one
                names = []
                for nm in re.split(r"[,\n;]", _g(PARTS_ASK, 90)):
                    nm = re.sub(r"^[\s\-\*\d\.\)]+", "", nm).strip(" .\"'")
                    if 2 <= len(nm) <= 30 and nm.lower() not in [n.lower() for n in names]:
                        names.append(nm)
                boxes = []
                for nm in names[:16]:
                    gb = _parse_boxes(_g(GROUND_PART.format(name=nm), 80), wx1 - wx0, wy1 - wy0)
                    if gb:
                        b = gb[0][0]
                        boxes.append(([b[0] + wx0, b[1] + wy0, b[2] + wx0, b[3] + wy0], nm))
                raw = "names: " + ", ".join(names)
            else:
                raw = _ask(GROUND, 160)
                boxes = _parse_boxes(raw, wx1 - wx0, wy1 - wy0)
                boxes = [([b[0] + wx0, b[1] + wy0, b[2] + wx0, b[3] + wy0], lab) for b, lab in boxes]
            if ent["big"]:
                win_area = (wx1 - wx0) * (wy1 - wy0)
                for b, lab in boxes:
                    ix0, iy0, ix1, iy1 = max(b[0], rect[0]), max(b[1], rect[1]), min(b[2], rect[2]), min(b[3], rect[3])
                    if ix1 - ix0 < 8 or iy1 - iy0 < 8:
                        continue  # not inside this edit box
                    if (b[2] - b[0]) * (b[3] - b[1]) > 0.5 * win_area:
                        continue  # a whole-picture answer, not a part
                    if len(ent["exc"]) + len(ent["asked"]) >= 16:
                        break
                    rb = [b[0] - wx0, b[1] - wy0, b[2] - wx0, b[3] - wy0]
                    rb = [round(rb[0] / (wx1 - wx0) * 1000), round(rb[1] / (wy1 - wy0) * 1000),
                          round(rb[2] / (wx1 - wx0) * 1000), round(rb[3] / (wy1 - wy0) * 1000)]
                    q = YN_ASK.format(cap=cap, label=lab)
                    k = (model, q, "p-text")
                    if k not in _CACHE:
                        if vlm is None:
                            vlm, proc = SceneContextQwenVL()._load(model)
                        _CACHE[k] = _yes_prob(vlm, proc, None, EDIT_SYSTEM, q)
                    ent["asked"].append("%s=%.2f" % (lab, _CACHE[k]))
                    # only compact parts (face, hands, feet: not long limbs or the torso) are protected; those are the ones the edit smudges
                    if _CACHE[k] < keep_below and (b[2] - b[0]) * (b[3] - b[1]) <= 0.25 * (rect[2] - rect[0]) * (rect[3] - rect[1]) and max(b[2] - b[0], b[3] - b[1]) <= 1.6 * min(b[2] - b[0], b[3] - b[1]):
                        ent["exc"].append((b, lab))
                ent["status"] = ent["kind"] + " sides [" + ", ".join(ent.get("grow", [])) + "]" + (" grown to %s" % ent["rect"] if ent.get("grow") else "") + ": parts [%s] -> keep [%s]" % (", ".join(ent["asked"]) or "none: " + raw[:160].replace("\n", " "), ", ".join(l for _, l in ent["exc"]))
            else:
                if not boxes:
                    ent["status"] = "VLM found nothing (%s)" % raw[:60].replace("\n", " ")
                    continue
                b = boxes[0][0]
                sl = 8
                b = [max(b[0], rect[0] - sl), max(b[1], rect[1] - sl), min(b[2], rect[2] + sl), min(b[3], rect[3] + sl)]
                if b[2] - b[0] < 8 or b[3] - b[1] < 8:
                    ent["status"] = "box outside the user's box"
                    continue
                ent["inc"] = b
                ent["status"] = ent["kind"]
        if vlm is not None and not keep_loaded:
            del vlm, proc
            SceneContextQwenVL._unload()
        # 2) SAM outlines each item box
        need = [(e["rect"], b) for e in found for b in ([e["inc"]] if e["inc"] else []) + ([e["item"]] if e.get("item") else []) + [x for x, _ in e["exc"]]]
        masks = {}
        if need:
            sam, sproc = _load_sam(dev)
            for r, b in need:
                masks[tuple(b)] = self._sam_one(sam, sproc, full, b, r, W, H, dev)
            _unload_sam()
        # 3) per box: repaint area = outline (small box) or the whole box (big box), minus the parts that stay
        g = max(6, int(round(grow / 100.0 * min(H, W))))
        overlay = arr.copy().astype(np.float32)
        keepout = torch.zeros((H, W))
        for e in found:
            x0, y0, x1, y1 = e["rect"]
            box = torch.zeros((H, W))
            box[y0:y1, x0:x1] = 1.0
            barea = (x1 - x0) * (y1 - y0)
            m = box.clone()
            note = e["status"]
            if e["inc"] is not None:
                im = masks[tuple(e["inc"])]
                if float((im * box).sum()) < 0.01 * barea:
                    note = "SAM mask tiny -> whole box"
                else:
                    base = (im * box) > 0.5
                    base = base.float()
                    if eyedrop:
                        reach = max(8, int(0.03 * min(H, W)))
                        gm = _eyedrop_grow(arr, base, box, reach)
                        added = 100 * float((gm - base).sum()) / max(1.0, float(base.sum()))
                        base = gm
                    else:
                        added = 0
                    m = _grow(_close_and_clean(base), g) * box
                    m = _blur(m, feather / 2.0) * box
                    note = "item box %s, eyedropper +%.0f%%" % ([int(v) for v in e["inc"]], added)
            if e["exc"]:
                ex = torch.zeros((H, W))
                item = masks[tuple(e["item"])] if e.get("item") else torch.zeros((H, W))
                if float((item * box).sum()) > 0.35 * barea:
                    item = torch.zeros((H, W))  # an item covering that much is a bad grounding (multi-edit prompt), not one thing
                dropped = []
                for b, lab in e["exc"]:
                    pm = masks[tuple(b)] * box
                    area = float(pm.sum())
                    pm = _hull_fill(pm, b) * box
                    if False and area > 0:  # (the yes/no question decides; the grounded item is unreliable for multi-edit prompts)
                        dropped.append(lab)  # the VLM listed the item the instruction changes
                        continue
                    ex = torch.maximum(ex, pm * (1 - item))
                if dropped:
                    note += " (ignored, it is the changed item: %s)" % ", ".join(dropped)
                ex = -F.max_pool2d(-ex[None, None], 5, 1, 2)[0, 0]  # shrink 2 px so the protected edge is repainted
                if float(item.sum()) > 200 and float(ex.sum()) > 0:
                    d2 = _colour_d2(arr, item)
                    if d2 is not None:
                        ring = (F.max_pool2d(ex[None, None], 15, 1, 7)[0, 0] - (-F.max_pool2d(-ex[None, None], 15, 1, 7)[0, 0])) > 0.5
                        same = torch.from_numpy(d2 < 4.0).float()
                        ex = ex * (1 - ring.float() * same)  # old-item colour on the edge of a kept part is not kept
                ex = _blur(ex, max(1.0, feather / 2.0))
                if float((ex * box).sum()) < 0.6 * barea:
                    m = m * (1 - ex)
                    keepout = torch.maximum(keepout, ex * box)
                else:
                    note += " (keep area too large, ignored)"
            out = torch.maximum(out, m)
            log.append("rect %s: %s, repaint %.0f%% of box" % (e["rect"], note, 100 * float((m > 0.5).sum()) / barea))
        mm_ = out.numpy()[..., None]
        kk = keepout.numpy()[..., None]
        overlay = overlay * (1 - 0.55 * mm_) + np.array([255, 40, 40], dtype=np.float32) * 0.55 * mm_
        overlay = overlay * (1 - 0.55 * kk) + np.array([40, 120, 255], dtype=np.float32) * 0.55 * kk
        prev = torch.from_numpy(np.clip(overlay, 0, 255).astype(np.float32) / 255.0)[None]
        print("[MidFR] edit mask:\n" + "\n".join(log), flush=True)
        user = torch.zeros((H, W))
        for e in found:
            x0, y0, x1, y1 = e["rect0"]
            user[y0:y1, x0:x1] = 1.0
        margin = (out * (1 - user)).clamp(0, 1)
        return (out[None].clamp(0, 1), prev, "\n".join(log), json.dumps(todo, separators=(",", ":")), margin[None], torch.zeros_like(margin)[None], torch.zeros_like(margin)[None], torch.zeros_like(margin)[None])


def _blend_core(o, e, mk, mg, strictness=3.0, seam_px=40, big=45):
    import cv2

    H, W = o.shape[:2]
    diff = np.abs(e - o).mean(-1) * 255.0
    inner = (mk > 0.5) & (mg < 0.5)
    marg = mg > 0.5
    revert = np.zeros((H, W), np.float32)
    note = "no margin"
    if marg.any():
        new = inner & (diff > 35)
        if int(new.sum()) < 400:
            note = "too little new item to compare, margin kept"
        else:
            lab = cv2.cvtColor(np.clip(e * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
            smp = lab[new]
            if len(smp) > 30000:
                smp = smp[np.random.RandomState(0).choice(len(smp), 30000, replace=False)]
            K = 4
            _, lbl, _ = cv2.kmeans(smp, K, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5), 2, cv2.KMEANS_PP_CENTERS)
            lbl = lbl.ravel()
            d2 = np.full((H, W), 1e9, np.float32)
            flat = lab.reshape(-1, 3)
            for k in range(K):
                c = smp[lbl == k]
                if len(c) < 60:
                    continue
                mu = c.mean(0)
                cov = np.cov(c.T) + np.eye(3) * 4.0
                inv = np.linalg.inv(cov)
                dd = flat - mu
                dk = np.einsum("ij,jk,ik->i", dd, inv, dd).reshape(H, W)
                d2 = np.minimum(d2, dk)
            good = (d2 < strictness ** 2) & (diff > 12)
            keep = (good & marg) | inner
            keep = cv2.morphologyEx(keep.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
            n, cc = cv2.connectedComponents(keep, connectivity=8)
            ids = np.unique(cc[inner & (cc > 0)])
            conn = np.isin(cc, ids) & (cc > 0)
            # margin pixels the edit changed that are not connected new-item colour go back to the original
            bad = marg & (diff > big) & ~conn
            bad = cv2.dilate(bad.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            bad &= marg
            revert = cv2.GaussianBlur(bad.astype(np.float32), (0, 0), 2.0)
            note = "margin: kept %.0f%% of the changed pixels" % (100.0 * ((marg & (diff > big)) & ~bad).sum() / max(1, (marg & (diff > big)).sum()))
    out = e * (1 - revert[..., None]) + o * revert[..., None]
    # seams: smooth the colour difference along the edge of the edited area inward
    if seam_px > 0:
        changed = (np.abs(out - o).mean(-1) * 255.0 > 6).astype(np.uint8)
        changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        support = (mk > 0.5) & (revert < 0.5) & (changed > 0)
        sup = support.astype(np.uint8)
        dist = cv2.distanceTransform(sup, cv2.DIST_L2, 3)
        ring = (dist > 0) & (dist <= 3)
        delta = (o - out)
        gate = ring & (np.abs(delta).mean(-1) * 255.0 < 30)
        if gate.sum() > 50:
            w = gate.astype(np.float32)
            sg = max(3.0, seam_px / 3.0)
            num = cv2.GaussianBlur(delta * w[..., None], (0, 0), sg)
            den = cv2.GaussianBlur(w, (0, 0), sg)[..., None]
            corr = num / np.maximum(den, 1e-4)
            fall = np.clip(1.0 - dist / float(seam_px), 0, 1)[..., None] * (den > 1e-3) * support[..., None]
            out = np.clip(out + corr * fall, 0, 1)
    return out, revert, note


class EditBlendTrim:
    """After the edit: the box was grown so the new item had room (a collar, a hem). Here the part of the edit that landed in that
    extra margin is kept only where it looks like the new item (its colours match what the edit drew inside the box, and it is
    connected to it); anything else in the margin goes back to the original. Then colour differences along the edge of the edited
    area (a neck that came out a shade off) are smoothed away over a short distance."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original": ("IMAGE",),
                "edited": ("IMAGE",),
                "mask": ("MASK", {"tooltip": "The Edit Mask node's mask"}),
                "margin": ("MASK", {"tooltip": "The Edit Mask node's margin output (the grown part of each box)"}),
            },
            "optional": {
                "strictness": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 8.0, "step": 0.25, "tooltip":
                    "How far (in colour spread) margin pixels may be from the new item's colours and still be kept; lower trims more"}),
                "seam_px": ("INT", {"default": 40, "min": 0, "max": 200, "tooltip": "Distance over which edge colour differences are smoothed (0 = off)"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "trim_preview")
    FUNCTION = "run"
    CATEGORY = "image/midfr"

    def run(self, original, edited, mask, margin, strictness=3.0, seam_px=40):
        import cv2

        o = original[0].detach().float().cpu().numpy()[..., :3]
        H, W = o.shape[:2]
        e = edited[0].detach().float().cpu().numpy()[..., :3]
        if e.shape[:2] != (H, W):
            e = cv2.resize(e, (W, H), interpolation=cv2.INTER_LANCZOS4)

        def _m(t):
            a = t.reshape(-1, t.shape[-2], t.shape[-1])[0].float().cpu()
            if tuple(a.shape) != (H, W):
                a = F.interpolate(a[None, None], size=(H, W), mode="bilinear")[0, 0]
            return a.numpy()

        mk, mg = _m(mask), _m(margin)
        out, revert, note = _blend_core(o, e, mk, mg, strictness, seam_px)
        prev = np.clip(0.6 * o + 0.4 * np.stack([revert, np.zeros_like(revert), np.zeros_like(revert)], -1), 0, 1)
        print("[MidFR] edit blend: " + note, flush=True)
        return (torch.from_numpy(out.astype(np.float32))[None], torch.from_numpy(prev.astype(np.float32))[None])


NODE_CLASS_MAPPINGS = {"EditMaskQwenSAM": EditMaskQwenSAM, "EditBlendTrim": EditBlendTrim}
NODE_DISPLAY_NAME_MAPPINGS = {"EditMaskQwenSAM": "Edit Mask (Qwen VL + SAM 2)", "EditBlendTrim": "Edit Blend (trim + seams)"}
