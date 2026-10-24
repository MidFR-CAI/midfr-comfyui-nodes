"""Run: python -m unittest discover -s tests   (from the simple_image_upload_resize folder; needs only Pillow)."""
import importlib.util
import json
import os
import tempfile
import unittest

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _load(name):  # import the ComfyUI-free modules directly, without the package __init__
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


layout = _load("layout")
common = _load("common")


def boxes(*b):
    return json.dumps([dict(x=x, y=y, w=w, h=h, caption=c) for x, y, w, h, c in b])


class LayoutTests(unittest.TestCase):
    def test_no_boxes_returns_plain_prompt(self):
        for data in ("[]", "", "not json", "{}", "null", boxes((0.1, 0.1, 0, 0.2, "x"))):
            self.assertEqual(layout.build_prompt(data, 1024, 1024, "a cat on a sofa"), "a cat on a sofa", data)

    def test_centred_yxyx_pixels(self):
        # box covering the left half of a 1024x512 canvas
        out = json.loads(layout.build_prompt(boxes((0, 0, 0.5, 1, "a cat")), 1024, 512, "scene"))
        self.assertEqual(out["items"][0]["bbox"], [-256, -512, 256, 0])  # [ymin, xmin, ymax, xmax]
        self.assertEqual(out["bbox_coord_space"], "abs_centralized")
        self.assertEqual(out["bbox_format"], "yxyx")
        self.assertEqual(out["summary"], {"caption": "scene"})

    def test_matches_core_node_shape(self):
        s = layout.build_prompt(boxes((0.1, 0.2, 0.5, 0.25, " title ")), 1024, 1024, " scene ")
        out = json.loads(s)
        self.assertEqual(list(out), ["summary", "items", "bbox_coord_space", "bbox_format"])
        self.assertEqual(out["items"][0], {"label": "object", "caption": "title", "bbox": [-308, -410, -51, 103]})
        self.assertNotIn(" ", s.replace("title", "").replace("scene", ""))  # compact, like core

    def test_legacy_label_and_text_fields_are_ignored(self):
        data = json.dumps([{"x": 0, "y": 0, "w": 1, "h": 1, "caption": "c", "label": "text", "text": "HELLO"}])
        item = json.loads(layout.build_prompt(data, 64, 64))["items"][0]
        self.assertEqual(list(item), ["label", "caption", "bbox"])
        self.assertEqual(item["label"], "object")

    def test_clamp_and_swap(self):
        out = json.loads(layout.build_prompt(boxes((-0.2, 0.5, 1.5, -0.25, "c")), 100, 100))
        self.assertEqual(out["items"][0]["bbox"], [-25, -50, 0, 50])

    def test_unicode_not_escaped(self):
        self.assertIn("é", layout.build_prompt(boxes((0, 0, 1, 1, "é")), 64, 64))


class ImageTests(unittest.TestCase):
    def _png(self, size, mode, color):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "t.png")
        Image.new(mode, size, color).save(p)
        return p

    def test_alpha_becomes_mask_source(self):
        rgb, alpha = common.load_rgb_and_alpha(self._png((8, 8), "RGBA", (10, 20, 30, 64)))
        self.assertEqual(rgb.mode, "RGB")
        self.assertEqual(alpha.getpixel((0, 0)), 64)

    def test_no_alpha(self):
        rgb, alpha = common.load_rgb_and_alpha(self._png((8, 8), "RGB", (1, 2, 3)))
        self.assertIsNone(alpha)

    def test_resize_keeps_old_behaviour_and_resizes_alpha(self):
        rgb = Image.new("RGB", (4000, 3000)); alpha = Image.new("L", (4000, 3000), 255)
        r, a = common.resize_pair(rgb, alpha, 1.0, 32, False)
        self.assertEqual(r.size, a.size)
        self.assertTrue(r.size[0] % 32 == 0 and r.size[1] % 32 == 0)
        self.assertLessEqual(r.size[0] * r.size[1], 1.1e6)
        small = Image.new("RGB", (500, 333))
        self.assertEqual(common.resize_pair(small, None, 1.0, 32, False)[0].size, (500, 333))  # untouched

    def test_always_snap_16(self):
        r, _ = common.resize_pair(Image.new("RGB", (500, 333)), None, 1.0, 16, True)
        self.assertEqual(r.size, (496, 336))
        self.assertEqual(common.snap(1, 16), 16)


if __name__ == "__main__":
    unittest.main()
