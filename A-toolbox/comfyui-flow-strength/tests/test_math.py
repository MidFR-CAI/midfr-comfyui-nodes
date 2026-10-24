"""Run: python -m unittest discover -s tests   (from the comfyui-flow-strength folder)."""
import importlib.util
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("math_utils", os.path.join(ROOT, "math_utils.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Tests(unittest.TestCase):
    def test_round_trip_start_sigma_equals_strength(self):
        for shift in (1.0, 3.0, 5.0):
            for s in (0.1, 0.35, 0.5, 0.8, 0.95):
                d = m.strength_to_denoise(s, shift)
                self.assertAlmostEqual(m.shift_sigma(d, shift), s, places=9)

    def test_known_values_shift5(self):
        self.assertAlmostEqual(m.strength_to_denoise(0.8, 5.0), 0.8 / 1.8, places=9)   # 0.444
        self.assertAlmostEqual(m.strength_to_denoise(0.5, 5.0), 1 / 6, places=9)
        self.assertAlmostEqual(m.shift_sigma(0.9, 5.0), 0.9783, places=3)              # KSampler denoise 0.9

    def test_edges_and_no_shift(self):
        self.assertEqual(m.strength_to_denoise(0, 5.0), 0.0)
        self.assertEqual(m.strength_to_denoise(1, 5.0), 1.0)
        self.assertEqual(m.strength_to_denoise(2, 5.0), 1.0)
        self.assertAlmostEqual(m.strength_to_denoise(0.37, 1.0), 0.37)


if __name__ == "__main__":
    unittest.main()
