import unittest
import numpy as np
from texturelab.config import GaborConfig
from texturelab.gabor_branch import gabor_energy_maps, make_gabor_bank


class TestGabor(unittest.TestCase):
    def setUp(self):
        self.config = GaborConfig(frequencies=(.12, .25), orientations=(0, np.pi/2), kernel_size=9)

    def test_bank_contract(self):
        bank = make_gabor_bank(self.config)
        self.assertEqual(len(bank), 4)
        for kernel, metadata in bank:
            self.assertAlmostEqual(float(kernel.mean()), 0, places=6)
            self.assertAlmostEqual(float(np.linalg.norm(kernel)), 1, places=5)
            self.assertIn("orientation", metadata)
            self.assertEqual(metadata["phase"], 0.0)

    def test_configured_phase_offsets(self):
        config = GaborConfig(
            frequencies=(.12, .25),
            orientations=(0, np.pi / 2),
            phases=(0.0, np.pi / 2),
            kernel_size=9,
        )
        bank = make_gabor_bank(config)
        self.assertEqual(len(bank), 8)
        self.assertEqual({metadata["phase"] for _, metadata in bank}, {0.0, np.pi / 2})

    def test_energy_shape_and_finite(self):
        image = np.random.default_rng(1).random((25, 31), dtype=np.float32)
        result = gabor_energy_maps(image, make_gabor_bank(self.config), 5)
        self.assertEqual(result.shape, (25, 31, 4))
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue((result >= 0).all())
if __name__ == "__main__": unittest.main()
