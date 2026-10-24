"""Plain unittest (numpy only):  python -m unittest discover -s tests -p "test_lora_fix.py"  from the lora-vault folder."""
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("LORA_VAULT_DATA_DIR", tempfile.mkdtemp())
os.environ.setdefault("LORA_VAULT_LORA_DIR", tempfile.mkdtemp())
from vault import lora_fix, parse  # noqa: E402

P = "diffusion_model.layers.{}.proj"
SEPARATE = {f"{P.format(i)}.{x}": f"{P.format(i)}.{x}.weight" for i in range(4) for x in ("q", "k", "v", "o")}


def fused_lora(layers=(0, 1, 2, 3), rank=4, dim=6):
    sd = {}
    for i in layers:
        sd[f"{P.format(i)}.qkv.lora_A.weight"] = np.random.rand(rank, 5).astype("float32")
        sd[f"{P.format(i)}.qkv.lora_B.weight"] = np.random.rand(3 * dim, rank).astype("float32")
        sd[f"{P.format(i)}.o.lora_A.weight"] = np.random.rand(rank, 5).astype("float32")
        sd[f"{P.format(i)}.o.lora_B.weight"] = np.random.rand(dim, rank).astype("float32")
    return sd


class SplitTests(unittest.TestCase):
    def test_split_is_exact(self):
        sd = fused_lora()
        out, n = lora_fix.split_fused_qkv(sd, SEPARATE)
        self.assertEqual(n, 4)
        self.assertFalse(any(".qkv." in k for k in out))
        b = sd[f"{P.format(1)}.qkv.lora_B.weight"]
        a = sd[f"{P.format(1)}.qkv.lora_A.weight"]
        for i, name in enumerate("qkv"):
            np.testing.assert_array_equal(out[f"{P.format(1)}.{name}.lora_B.weight"], b[i * 6:(i + 1) * 6])
            np.testing.assert_array_equal(out[f"{P.format(1)}.{name}.lora_A.weight"], a)
        # delta of the fused weight == stacked deltas of q, k, v
        fused = sd[f"{P.format(1)}.qkv.lora_B.weight"] @ a
        stacked = np.vstack([out[f"{P.format(1)}.{x}.lora_B.weight"] @ out[f"{P.format(1)}.{x}.lora_A.weight"] for x in "qkv"])
        np.testing.assert_allclose(fused, stacked, rtol=1e-6)
        self.assertIn(f"{P.format(1)}.o.lora_A.weight", out)  # untouched keys stay

    def test_alpha_is_copied(self):
        sd = fused_lora(layers=(0,)); sd[f"{P.format(0)}.qkv.alpha"] = np.float32(8)
        out, _ = lora_fix.split_fused_qkv(sd, SEPARATE)
        self.assertTrue(all(f"{P.format(0)}.{x}.alpha" in out for x in "qkv"))

    def test_model_with_fused_qkv_is_left_alone(self):
        key_map = dict(SEPARATE); key_map.update({f"{P.format(i)}.qkv": "x" for i in range(4)})
        sd = fused_lora(); out, n = lora_fix.split_fused_qkv(sd, key_map)
        self.assertEqual(n, 0); self.assertEqual(set(out), set(sd))

    def test_unknown_key_map_or_model_without_qkv_targets_does_nothing(self):
        sd = fused_lora()
        self.assertEqual(lora_fix.split_fused_qkv(sd, None)[1], 0)
        self.assertEqual(lora_fix.split_fused_qkv(sd, {"other": "x"})[1], 0)

    def test_uneven_dims_from_model(self):
        sd = fused_lora(layers=(0,), dim=6)
        out, n = lora_fix.split_fused_qkv(sd, SEPARATE, dims_for=lambda base: [8, 5, 5])
        self.assertEqual(n, 1)
        self.assertEqual([out[f"{P.format(0)}.{x}.lora_B.weight"].shape[0] for x in "qkv"], [8, 5, 5])

    def test_indivisible_without_dims_is_skipped(self):
        sd = fused_lora(layers=(0,)); sd[f"{P.format(0)}.qkv.lora_B.weight"] = np.zeros((10, 4), "float32")
        out, n = lora_fix.split_fused_qkv(sd, SEPARATE)
        self.assertEqual(n, 0); self.assertIn(f"{P.format(0)}.qkv.lora_B.weight", out)


class LayerTests(unittest.TestCase):
    def test_limit(self):
        sd = fused_lora(); sd["diffusion_model.final.lora_A.weight"] = 1
        out, kept, dropped = lora_fix.limit_layers(sd, 2, 3)
        self.assertEqual((kept, dropped), (2, 2))
        self.assertTrue(all(lora_fix.layer_of(k) in (2, 3) for k in out if lora_fix.layer_of(k) is not None))
        self.assertIn("diffusion_model.final.lora_A.weight", out)

    def test_open_ended_and_none(self):
        sd = fused_lora()
        self.assertIs(lora_fix.limit_layers(sd)[0], sd)
        self.assertEqual(lora_fix.limit_layers(sd, 2, None)[1], 2)
        self.assertEqual(lora_fix.limit_layers(sd, None, 1)[1], 2)

    def test_describe(self):
        self.assertEqual(lora_fix.describe_range(None, None), "")
        self.assertEqual(lora_fix.describe_range(14, 41), "layers 14-41")
        self.assertEqual(lora_fix.describe_range(14, None), "layers 14-")


class ParseTests(unittest.TestCase):
    def one(self, t):
        (s,) = parse.parse(t); return s

    def test_old_syntax_unchanged(self):
        s = self.one("<lora:Cool Style - v1:0.8>")
        self.assertEqual((s["ref"], s["strength"], s["clip_strength"], s["layer_start"], s["layer_end"]), ("Cool Style - v1", 0.8, 0.8, None, None))
        s = self.one("<lora:v123:1:0.5>")
        self.assertEqual((s["strength"], s["clip_strength"]), (1.0, 0.5))
        self.assertEqual(self.one("<lora:v123>")["strength"], 1.0)

    def test_layers(self):
        for tag, want in (("<lora:v1:0.4:layers=14-41>", (14, 41)), ("<lora:v1:0.4:layers=14->", (14, None)),
                          ("<lora:v1:0.4:layers=-20>", (0, 20)), ("<lora:v1:0.4:0.2:layers=3-9>", (3, 9)),
                          ("<lora:v1:layers=9-3>", (3, 9))):
            s = self.one(tag)
            self.assertEqual((s["layer_start"], s["layer_end"]), want, tag)
        s = self.one("<lora:v1:0.4:0.2:layers=3-9>")
        self.assertEqual((s["strength"], s["clip_strength"], s["ref"]), (0.4, 0.2, "v1"))
        self.assertEqual(self.one("<lora:v1:0.4:layers=->")["layer_start"], None)  # junk ignored


if __name__ == "__main__":
    unittest.main()
