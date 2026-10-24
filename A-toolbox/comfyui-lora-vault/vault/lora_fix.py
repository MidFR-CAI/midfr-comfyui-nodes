"""LoRA clean-up applied before a LoRA is patched into the model.

1. Fused attention: some trainers (ai-toolkit, diffusers exports) save q, k and v as one `qkv` weight. A model
   that keeps them separate (MOSS-Image) doesn't recognise those keys, ComfyUI prints "lora key not loaded"
   and the whole attention part of the LoRA is silently dropped. We split the fused weight into q, k and v
   (exact: A is shared, B is cut into three row blocks), but only when the model has separate q/k/v and no
   fused qkv, so models that really are fused are left alone.
2. Layer range: keep only the transformer layers START..END (inclusive).

Works on any dict of tensors (torch or numpy): it only slices and copies, no maths.
"""
import re

LAYER_RE = re.compile(r"(?:^|\.)(?:layers|blocks|transformer_blocks|double_blocks|single_blocks)\.(\d+)\.")
_FUSED = re.compile(r"^(?P<base>.+)\.qkv\.(?P<rest>lora_A\.weight|lora_B\.weight|lora_down\.weight|lora_up\.weight|alpha|dora_scale)$")
_COPY = ("lora_A.weight", "lora_down.weight", "alpha")  # same for q, k and v
_SLICE = ("lora_B.weight", "lora_up.weight", "dora_scale")  # one row block per q, k, v


def layer_of(key):
    m = LAYER_RE.search(key)
    return int(m.group(1)) if m else None


def _rows(t):
    shape = getattr(t, "shape", None)
    return int(shape[0]) if shape is not None and len(shape) else None


def split_fused_qkv(lora, key_map, dims_for=None):
    """-> (new_lora, number of fused weights split).

    key_map: LoRA base key -> model weight (ComfyUI's model_lora_keys_unet), or None when unknown (then nothing
    is split). dims_for(base) may return the three output sizes (q, k, v); default is an equal split.
    Each fused weight is split completely or not at all."""
    if not key_map:
        return lora, 0
    groups = {}  # base -> {rest: key}
    for key in lora:
        m = _FUSED.match(key)
        if m:
            groups.setdefault(m.group("base"), {})[m.group("rest")] = key
    out = dict(lora)
    done = 0
    for base, keys in groups.items():
        if (base + ".qkv") in key_map or not all((base + s) in key_map for s in (".q", ".k", ".v")):
            continue  # model has a fused qkv itself (or no q/k/v at all): leave as it is
        rows = [_rows(lora[k]) for r, k in keys.items() if r in _SLICE and r != "dora_scale"]
        dims = None
        if dims_for is not None:
            try:
                dims = dims_for(base)
            except Exception:
                dims = None
        n = rows[0] if rows else None
        if not dims or (n is not None and sum(dims) != n):
            if n is None or n % 3:
                continue  # can't tell where to cut: leave it (ComfyUI will report it as before)
            dims = [n // 3] * 3
        for rest, key in keys.items():
            value = lora[key]
            if rest in _COPY:
                parts = [value, value, value]
            else:
                parts, o = [], 0
                for d in dims:
                    parts.append(value[o:o + d])
                    o += d
            for suffix, part in zip((".q", ".k", ".v"), parts):
                out[f"{base}{suffix}.{rest}"] = part
            del out[key]
        done += 1
    return out, done


def limit_layers(lora, start=None, end=None):
    """Keep transformer layers start..end (inclusive; None = open). Keys with no layer number are kept.
    -> (new_lora, layers_kept, layers_dropped)"""
    if start is None and end is None:
        return lora, 0, 0
    lo = 0 if start is None else start
    out, kept, dropped = {}, set(), set()
    for key, value in lora.items():
        n = layer_of(key)
        if n is None or (n >= lo and (end is None or n <= end)):
            out[key] = value
            if n is not None:
                kept.add(n)
        else:
            dropped.add(n)
    return out, len(kept), len(dropped - kept)


def describe_range(start, end):
    if start is None and end is None:
        return ""
    return f"layers {0 if start is None else start}-{'' if end is None else end}"


def _model_key_map(model):
    try:
        import comfy.lora

        return comfy.lora.model_lora_keys_unet(model.model, {})
    except Exception:
        return None


def _dims_fn(model, key_map):
    def dims(base):
        out = []
        for s in (".q", ".k", ".v"):
            target = key_map[base + s]
            if isinstance(target, (tuple, list)):
                target = target[0]
            w = model.get_model_object(target if str(target).endswith(".weight") else str(target) + ".weight")
            out.append(int(w.shape[0]))
        return out
    return dims


def prepare(lora, model, start=None, end=None):
    """Split fused qkv (when the model needs it) and apply the layer range. -> (lora, [notes for the log line])"""
    notes = []
    key_map = _model_key_map(model)
    lora, n = split_fused_qkv(lora, key_map, _dims_fn(model, key_map) if key_map else None)
    if n:
        notes.append(f"split {n} fused qkv")
    lora, kept, dropped = limit_layers(lora, start, end)
    r = describe_range(start, end)
    if r:
        notes.append(f"{r} ({kept} kept, {dropped} off)")
    return lora, notes
