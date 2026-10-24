"""ComfyUI-free logic of the MOSS control adapter: reading the adapter file, building the control tensor, the embedder add-on.
Everything here is plain torch so it is unit-tested without ComfyUI (tests/ in this folder)."""
from __future__ import annotations

import json

import torch
import torch.nn.functional as F

CHANNELS = {"fill": 384, "lineart": 128, "depth": 128, "colour": 128, "pose": 128, "tile": 128}


def split_adapter(sd, metadata):
    """-> (lora_sd, embedder dict {weight,bias,ctrl_weight}, kinds list). The adapter is ONE safetensors file."""
    if metadata.get("format") != "moss-control-adapter-v1":
        raise ValueError("not a moss-control-adapter file")
    kinds = json.loads(metadata["controls"])
    emb = {k.split(".", 1)[1]: v for k, v in sd.items() if k.startswith("video_embedder.")}
    lora = {k: v for k, v in sd.items() if not k.startswith("video_embedder.")}
    if "ctrl_weight" not in emb or emb["ctrl_weight"].shape[1] != sum(CHANNELS[k] for k in kinds):
        raise ValueError("embedder control columns do not match the control list")
    return lora, emb, kinds


def unshuffle_mask(mask, stride=16):
    """(B,1,H,W) -> (B,stride^2,H/stride,W/stride), the same layout the trainer used."""
    b, _, h, w = mask.shape
    m = mask.view(b, h // stride, stride, w // stride, stride)
    return m.permute(0, 2, 4, 1, 3).reshape(b, stride * stride, h // stride, w // stride)


def crop16(t):
    """(B,C,H,W) -> crop to multiples of 16."""
    h, w = t.shape[-2] // 16 * 16, t.shape[-1] // 16 * 16
    return t[..., :h, :w]


def build_control(kinds, encode, image=None, mask=None, **aux):
    """encode: callable (B,3,H,W) in [-1,1] -> packed latent (B,128,H/16,W/16). ``image`` (B,3,H,W) in [-1,1], ``mask`` (B,1,H,W)
    {0,1} 1 = repaint, aux: lineart/depth/colour/pose/tile images in [-1,1]. A missing input becomes zeros, which the adapter was
    trained to treat as 'no control'. Returns (B, total, H/16, W/16)."""
    ref = next(t for t in (image, *aux.values()) if t is not None)
    ref = crop16(ref)
    b, _, h, w = ref.shape
    parts = []
    for kind in kinds:
        ch = CHANNELS[kind]
        if kind == "fill":
            if image is None or mask is None:
                parts.append(torch.zeros(b, ch, h // 16, w // 16))
            else:
                img, m = crop16(image), crop16(mask).float()
                parts.append(torch.cat([encode(img * (1 - m)), unshuffle_mask(m)], 1))
        else:
            src = aux.get(kind)
            parts.append(torch.zeros(b, ch, h // 16, w // 16) if src is None else encode(crop16(src)))
    return torch.cat(parts, 1).float()


def add_control(embedded, tokens_in, control_tokens, ctrl_weight, strength=1.0):
    """What the patched embedder adds: ``embedded`` (B,N,dim) + control(B|1,N,extra) @ ctrl_weight^T. Only applied when the token
    count matches the image tokens (reference/aux sequences pass through untouched)."""
    if control_tokens is None or control_tokens.shape[1] != embedded.shape[1] or strength == 0:
        return embedded
    c = control_tokens.to(device=embedded.device, dtype=ctrl_weight.dtype)
    if c.shape[0] != embedded.shape[0]:
        c = c.expand(embedded.shape[0], -1, -1) if c.shape[0] == 1 else c.repeat(embedded.shape[0] // c.shape[0], 1, 1)
    return embedded + (F.linear(c, ctrl_weight.to(embedded.device)) * strength).to(embedded.dtype)


def tokens_of(control):
    """(B,extra,H,W) -> (B,H*W,extra)."""
    return control.flatten(2).transpose(1, 2).contiguous()
