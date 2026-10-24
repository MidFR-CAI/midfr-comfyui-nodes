"""Run: python -m unittest discover -s tests   (from the invoke_inpainting folder; numpy + Pillow + scipy/cv2 optional)."""
import importlib.util
import os
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("masks", os.path.join(ROOT, "masks.py"))
masks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(masks)


class SizeTests(unittest.TestCase):
    def test_auto_grows_small_boxes_to_one_megapixel(self):
        self.assertEqual(masks.processing_size(300, 300), (1024, 1024))
        w, h = masks.processing_size(400, 200)
        self.assertGreaterEqual(w * h, 1024 * 1024)
        self.assertEqual((w % 16, h % 16), (0, 0))
        self.assertAlmostEqual(w / h, 2.0, delta=0.1)

    def test_large_boxes_only_snap(self):
        self.assertEqual(masks.processing_size(1500, 1100), (1504, 1104))
        self.assertEqual(masks.processing_size(300, 300, "off"), (304, 304))


class GradientTests(unittest.TestCase):
    def setUp(self):
        self.m = np.zeros((256, 256), "float32")
        self.m[96:160, 96:160] = 1

    def test_gradient_grows_and_falls_off(self):
        w, area = masks.gradient_mask(self.m, 16, "Gaussian Blur", 0.0)
        self.assertEqual(w[128, 128], 1.0)
        self.assertGreater(area.sum(), (self.m > 0).sum())  # grown
        self.assertTrue(((w > 0) == area).all())
        edge = w[128, 88]  # just outside the mask
        self.assertTrue(0 < edge < 1)
        self.assertEqual(w[0, 0], 0)

    def test_zero_edge_keeps_mask(self):
        w, area = masks.gradient_mask(self.m, 0)
        self.assertTrue(area[self.m > 0.5].all())  # mask kept
        self.assertLessEqual(int(area.sum()), 80 * 80)  # at most one 8px cell of bilinear softening

    def test_min_denoise_floors_the_edge(self):
        w, _ = masks.gradient_mask(self.m, 16, "Gaussian Blur", 0.5)
        outside = np.ones_like(w, bool)
        outside[88:168, 88:168] = False  # beyond the mask and its softened edge
        grown = w[outside & (w > 0)]
        self.assertTrue(grown.size and np.allclose(grown, 0.5, atol=0.01))
        self.assertEqual(w[128, 128], 1.0)

    def test_modes_run(self):
        for mode in masks.COHERENCE_MODES:
            w, area = masks.gradient_mask(self.m, 32, mode, 0.2)
            self.assertTrue(area.any() and w.max() <= 1.0)


class FadeTests(unittest.TestCase):
    def test_fade(self):
        area = np.zeros((128, 128), bool)
        area[48:80, 48:80] = True
        a = masks.fade_alpha(area, 16)
        self.assertEqual(a[64, 64], 1.0)
        self.assertEqual(a[0, 0], 0.0)
        self.assertTrue(0 < a[44, 64] < 1)
        self.assertTrue(a[47, 64] > a[40, 64])  # monotone away from the area
        self.assertTrue((masks.fade_alpha(area, 0) == area).all())


class NoCropTests(unittest.TestCase):
    def test_no_crop_processes_whole_image_and_repaints_box(self):
        box = np.zeros((600, 800), "float32"); box[200:500, 300:700] = 1
        p = masks.plan_inpaint(600, 800, box=box, crop=False)
        self.assertEqual(p["rect"], (0, 0, 800, 600))
        self.assertTrue(p["masked"] and p["weight"] is not None)
        self.assertEqual(p["alpha"][300, 500], 1.0)
        self.assertEqual(p["alpha"][50, 50], 0.0)  # outside the box: never pasted
        self.assertEqual(p["alpha"][300, 301], 1.0)  # full strength right at the box edge
        self.assertTrue(0 < p["alpha"][300, 295] < 1)  # and fades out just beyond it

    def test_no_crop_mask_inside_box(self):
        box = np.zeros((600, 800), "float32"); box[100:500, 100:700] = 1
        inp = np.zeros_like(box); inp[250:350, 350:450] = 1
        p = masks.plan_inpaint(600, 800, box=box, inpaint=inp, crop=False)
        self.assertEqual(p["alpha"][300, 400], 1.0)
        self.assertEqual(p["alpha"][120, 120], 0.0)


class StrengthTests(unittest.TestCase):
    def test_strength(self):
        self.assertAlmostEqual(masks.strength_to_denoise(0.75, 5.0), 0.75 / (5 - 4 * 0.75), places=6)
        self.assertEqual(masks.strength_to_denoise(1.0, 5.0), 1.0)
        self.assertEqual(masks.strength_to_denoise(0.0, 5.0), 0.0)


if __name__ == "__main__":
    unittest.main()


class PlanTests(unittest.TestCase):
    def test_box_only_is_img2img_on_the_box(self):
        box = np.zeros((600, 800), "float32"); box[100:300, 200:500] = 1
        p = masks.plan_inpaint(600, 800, box=box)
        self.assertEqual(p["rect"], (200, 100, 500, 300))
        self.assertEqual(p["crop_size"], (300, 200))
        self.assertIsNone(p["weight"])
        self.assertTrue((p["alpha"] == 1).all())  # a lone box is repainted right up to its edge
        self.assertGreaterEqual(p["proc_size"][0] * p["proc_size"][1], 1024 * 1024)

    def test_mask_inside_box(self):
        box = np.zeros((600, 800), "float32"); box[100:500, 100:700] = 1
        inp = np.zeros_like(box); inp[250:350, 350:450] = 1
        inp[0:50, 0:50] = 1  # outside the box: must be ignored
        p = masks.plan_inpaint(600, 800, box=box, inpaint=inp)
        self.assertTrue(p["masked"] and p["weight"] is not None)
        self.assertEqual(p["alpha"].shape, (400, 600))
        self.assertEqual(p["alpha"][200, 300], 1.0)  # centre of the mask
        self.assertEqual(p["alpha"][0, 0], 0.0)  # far corner of the box is untouched
        self.assertEqual(p["weight"].shape[::-1], p["proc_size"])

    def test_no_box_uses_whole_image(self):
        inp = np.zeros((256, 256), "float32"); inp[100:150, 100:150] = 1
        p = masks.plan_inpaint(256, 256, box=np.zeros((256, 256), "float32"), inpaint=inp)
        self.assertEqual(p["rect"], (0, 0, 256, 256))
        self.assertTrue(p["masked"])

    def test_two_boxes_only_paste_inside_boxes(self):
        box = np.zeros((300, 300), "float32"); box[10:60, 10:60] = 1; box[200:250, 200:250] = 1
        p = masks.plan_inpaint(300, 300, box=box)
        self.assertEqual(p["rect"], (10, 10, 250, 250))
        self.assertEqual(p["alpha"][100, 100], 0.0)
        self.assertEqual(p["alpha"][20, 20], 1.0)

    def test_context_padding_widens_crop_but_repaints_only_the_box(self):
        box = np.zeros((400, 400), "float32"); box[100:200, 150:250] = 1
        p = masks.plan_inpaint(400, 400, box=box, padding=64)
        self.assertEqual(p["rect"], (86, 36, 314, 264))
        self.assertTrue(p["masked"] and p["weight"] is not None)  # box is repainted, the padding is context
        self.assertEqual(p["alpha"][0, 0], 0.0)                     # padding corner never pasted
        self.assertEqual(p["alpha"][100 - 36 + 50, 150 - 86 + 50], 1.0)  # box centre
        # clamped at the image edge, and padding 0 is unchanged
        box2 = np.zeros((400, 400), "float32"); box2[0:50, 0:50] = 1
        self.assertEqual(masks.plan_inpaint(400, 400, box=box2, padding=64)["rect"], (0, 0, 114, 114))
        self.assertEqual(masks.plan_inpaint(400, 400, box=box, padding=0)["rect"], (150, 100, 250, 200))
        self.assertFalse(masks.plan_inpaint(400, 400, box=box, padding=0)["masked"])

    def test_empty_inpaint_mask_means_whole_box(self):
        box = np.zeros((128, 128), "float32"); box[:64, :64] = 1
        p = masks.plan_inpaint(128, 128, box=box, inpaint=np.zeros((128, 128), "float32"))
        self.assertFalse(p["masked"])
