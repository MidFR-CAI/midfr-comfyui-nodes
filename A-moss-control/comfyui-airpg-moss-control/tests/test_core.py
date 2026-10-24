import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from core import build_control, unshuffle_mask, add_control, tokens_of, split_adapter

def enc(px):
    return torch.nn.functional.avg_pool2d(px, 16).repeat(1, 43, 1, 1)[:, :128]

def test_fill_layout_and_zero_missing():
    img = torch.randn(1, 3, 40, 50); m = torch.zeros(1, 1, 40, 50); m[..., 5:20, 5:20] = 1
    c = build_control(["fill", "lineart", "depth"], enc, img, m, lineart=img)
    assert c.shape == (1, 384 + 128 + 128, 2, 3)
    assert torch.count_nonzero(c[:, 512:]) == 0, "absent depth must be zeros"
    assert torch.count_nonzero(c[:, 128:384]) > 0

def test_unshuffle_matches_trainer():
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "training"))
    from moss_ctrl.masks import mask_to_tokens
    m = (torch.rand(2, 1, 32, 48) > 0.5).float()
    assert torch.equal(unshuffle_mask(m), mask_to_tokens(m))

def test_add_control():
    out = torch.randn(2, 6, 8); tok = torch.randn(1, 6, 5); w = torch.randn(8, 5)
    got = add_control(out, None, tok, w)
    assert torch.allclose(got, out + torch.nn.functional.linear(tok.expand(2, -1, -1), w), atol=1e-5)
    assert torch.equal(add_control(out, None, tok[:, :4], w), out), "mismatched token count passes through"
    assert torch.equal(add_control(out, None, tok, w, 0.0), out)

def test_split_adapter_checks():
    sd = {"video_embedder.weight": torch.zeros(8, 128), "video_embedder.ctrl_weight": torch.zeros(8, 384), "diffusion_model.layers.0.proj.q.alpha": torch.tensor(1.0)}
    lora, emb, kinds = split_adapter(sd, {"format": "moss-control-adapter-v1", "controls": '["fill"]'})
    assert kinds == ["fill"] and "ctrl_weight" in emb and len(lora) == 1
    try:
        split_adapter(sd, {"format": "moss-control-adapter-v1", "controls": '["fill","depth"]'}); assert False
    except ValueError:
        pass

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
