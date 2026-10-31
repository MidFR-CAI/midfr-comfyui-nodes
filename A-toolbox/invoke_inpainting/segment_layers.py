"""Layered segmentation for Edit Mask's `segment` mode.

Layer 0  OneFormer (COCO panoptic): the scenery (wall, sofa, window, floor ...) with names; the person is left out.
Layer 1  SAM 3 with one text prompt per part / item the VLM listed (face, hair, helmet, sword ...), never painting onto layer 0.
Layer 2  SAM 2.1 automatic segments clean up layer 1: each small segment takes the label most of its pixels already have.
Everything comes out as a label map of named components that the VLM then rules in or out for the edit."""
import gc
import re

import numpy as np
import torch
from PIL import Image

ONEFORMER = "shi-labs/oneformer_coco_swin_large"
SAM3_REPOS = ["facebook/sam3", "barxudaryangor/sam3-mirror-ungated"]


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def clean_name(n):
    n = re.sub(r"-(other-)?merged$|-stuff$|-other$", "", n)
    return n.replace("-", " ").replace("_", " ").strip().lower()


def background_layer(full):
    """-> ([(bool mask, name)], person bool mask)"""
    from transformers import pipeline

    p = pipeline("image-segmentation", model=ONEFORMER, device=0 if torch.cuda.is_available() else -1, subtask="panoptic")
    res = p(full)
    del p
    _free()
    W, H = full.size
    person = np.zeros((H, W), bool)
    bg = []
    for r in res:
        m = np.asarray(r["mask"].convert("L").resize((W, H))) > 127
        name = clean_name(r["label"])
        if name == "person":
            person |= m
        elif m.mean() > 0.0005:
            bg.append((m, name))
    return bg, person


def sam3_layer(full, concepts, dev, bg):
    """-> [(bool mask, concept)] for every instance SAM 3 finds. An instance that sits (over 60%) inside one scenery segment is scenery
    itself and is dropped; otherwise it is kept whole, even where OneFormer called part of it scenery."""
    from transformers import Sam3Model, Sam3Processor

    model = proc = None
    err = None
    for repo in SAM3_REPOS:
        try:
            model = Sam3Model.from_pretrained(repo).to(dev).eval()
            proc = Sam3Processor.from_pretrained(repo)
            break
        except Exception as e:  # gated / missing: try the next source
            err = e
            model = None
    if model is None:
        raise RuntimeError("SAM 3 could not be loaded: %s" % err)
    items = []
    dropped = []
    for c in concepts:
        inp = proc(images=full, text=c, return_tensors="pt").to(dev)
        with torch.inference_mode():
            o = model(**inp)
        r = proc.post_process_instance_segmentation(o, threshold=0.45, mask_threshold=0.5, target_sizes=inp.get("original_sizes").tolist())[0]
        for m in r["masks"]:
            m = m.cpu().numpy().astype(bool)
            a = int(m.sum())
            if a < 0.0004 * m.size:
                continue
            hit = [n for b, n in bg if (m & b).sum() > 0.6 * a]
            if hit:
                dropped.append("%s (inside %s)" % (c, hit[0]))
                continue
            items.append((m, c))
    del model, proc
    _free()
    return items, dropped


def label_map(items, shape):
    """Bigger masks first so smaller parts sit on top. -> (int map, [names])"""
    names = []
    lab = -np.ones(shape, np.int32)
    for m, c in sorted(items, key=lambda t: -int(t[0].sum())):
        if c not in names:
            names.append(c)
        lab[m] = names.index(c)
    return lab, names


def sam2_segments(sam, sproc, full, dev, n=24):
    """Automatic SAM 2.1 segments from a grid of points -> list of bool masks (deduplicated)."""
    W, H = full.size
    xs = np.linspace(W * 0.02, W * 0.98, n)
    ys = np.linspace(H * 0.02, H * 0.98, n)
    pts = [[float(x), float(y)] for y in ys for x in xs]
    cands = []
    for i in range(0, len(pts), 24):
        chunk = pts[i:i + 24]
        inp = sproc(images=full, input_points=[[[p] for p in chunk]], input_labels=[[[1] for _ in chunk]], return_tensors="pt")
        inp = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in inp.items()}
        with torch.inference_mode():
            if dev == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = sam(**inp, multimask_output=True)
            else:
                o = sam(**inp, multimask_output=True)
        ms = sproc.post_process_masks(o.pred_masks.float().cpu(), inp["original_sizes"])[0]
        sc = o.iou_scores.float().cpu()[0]
        for a in range(ms.shape[0]):
            for b in range(ms.shape[1]):
                if sc[a, b] > 0.82:
                    m = (ms[a, b] > 0)
                    ar = float(m.float().mean())
                    if 0.002 < ar < 0.6:
                        cands.append((float(sc[a, b]), m.numpy()))
    cands.sort(key=lambda t: -t[0])
    keep = []
    for _, m in cands:
        ms_ = m[::4, ::4]
        if any((ms_ & k[::4, ::4]).sum() / max(1, (ms_ | k[::4, ::4]).sum()) > 0.7 for k in keep):
            continue
        keep.append(m)
    return keep


def snap(lab, segs):
    """Each SAM 2.1 segment (smallest first) takes the label most of its already-labelled pixels have."""
    out = lab.copy()
    done = np.zeros(lab.shape, bool)
    near = {}
    for m in sorted(segs, key=lambda s: int(s.sum())):
        inside = m
        free = inside & ~done
        if free.sum() < 30:
            continue
        v = lab[inside]
        v = v[v >= 0]
        if len(v) < 0.5 * int(inside.sum()):
            continue
        if int(inside.sum()) > 1.6 * len(v):
            continue  # a segment much bigger than the part SAM 3 found is a neighbour spilling over, not a refinement
        cnt = np.bincount(v)
        k = int(cnt.argmax())
        if cnt[k] < 0.6 * len(v):
            continue
        # a refinement stays close to what SAM 3 found: it fills edges and gaps, it does not take in a neighbour (the chest under a face)
        if k not in near:
            from scipy import ndimage

            a = int((lab == k).sum())
            near[k] = ndimage.distance_transform_edt(lab != k) <= max(6.0, 0.12 * a ** 0.5)
        free = free & near[k]
        if not free.any():
            continue
        out[free] = k
        done |= free
    return out


def fill_person(lab, person):
    """Person pixels nobody labelled join the nearest labelled part."""
    from scipy import ndimage

    unl = person & (lab < 0)
    if not unl.any() or not (lab >= 0).any():
        return lab
    _, (iy, ix) = ndimage.distance_transform_edt(lab < 0, return_indices=True)
    out = lab.copy()
    out[unl] = lab[iy[unl], ix[unl]]
    return out
