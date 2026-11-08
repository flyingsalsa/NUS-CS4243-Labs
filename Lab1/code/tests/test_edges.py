import unittest
import numpy as np
from texturelab.config import EdgeConfig
from texturelab.edge_branch import adaptive_thresholds, bilinear_sample, hysteresis, nms_interpolated


class TestEdges(unittest.TestCase):
    def test_bilinear_plane_is_exact(self):
        y, x = np.mgrid[:8, :9]; plane = 2*y + 3*x
        yy = np.array([1.2, 4.5]); xx = np.array([2.4, 5.25])
        np.testing.assert_allclose(bilinear_sample(plane, yy, xx), 2*yy + 3*xx, atol=1e-6)

    def test_interpolated_nms_thins_horizontal_ridge(self):
        magnitude = np.zeros((7, 7), np.float32); magnitude[:, 2:5] = [1, 3, 1]
        result = nms_interpolated(magnitude, np.zeros_like(magnitude))
        self.assertTrue(np.all(result[1:-1, 3] == 3))
        self.assertEqual(np.count_nonzero(result[:, [2, 4]]), 0)

    def test_hysteresis_connectivity(self):
        strong = np.zeros((4, 4), bool); strong[0, 0] = True
        weak = np.zeros_like(strong); weak[1, 1] = weak[2, 2] = True
        self.assertEqual(hysteresis(strong, weak, 4).sum(), 1)
        self.assertEqual(hysteresis(strong, weak, 8).sum(), 3)

    def test_thresholds_are_reproducible(self):
        values = np.arange(100, dtype=float).reshape(10, 10)
        a = adaptive_thresholds(values, EdgeConfig(high_percentile=80))
        b = adaptive_thresholds(values, EdgeConfig(high_percentile=80))
        self.assertEqual(a, b); self.assertLess(a[0], a[1])


if __name__ == "__main__": unittest.main()

