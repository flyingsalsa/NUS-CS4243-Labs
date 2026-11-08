import unittest
import numpy as np
from texturelab.config import FeatureConfig, GaborConfig, NormalityConfig
from texturelab.features import (
    extract_local_features,
    feature_family_indices,
    global_pool,
)
from texturelab.normality import (
    fit_normal_model,
    predict_anomaly,
    select_mask_threshold,
)
from texturelab.synthetic import texture


class TestFeaturesAndNormality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.feature_config = FeatureConfig(
            gabor=GaborConfig(
                frequencies=(0.15,),
                orientations=(0, np.pi / 2),
                kernel_size=9,
                pool_size=5,
            )
        )

    def test_aligned_named_features(self):
        image, _ = texture("wood", 32)
        features, names = extract_local_features(image, self.feature_config)
        self.assertEqual(features.shape[:2], image.shape[:2])
        self.assertEqual(features.shape[-1], len(names))
        self.assertTrue(np.isfinite(features).all())
        families = feature_family_indices(names)
        self.assertTrue(
            all(families[k] for k in ("colour", "gabor", "gradient", "edge"))
        )
        self.assertEqual(global_pool(features).shape, (3 * len(names),))

    def test_defect_scores_more_highly(self):
        normal_maps = [
            extract_local_features(texture("grid", 36, i)[0], self.feature_config)[0]
            for i in range(2)
        ]
        model = fit_normal_model(normal_maps, NormalityConfig(threshold_percentile=99))
        defective, truth = texture("grid", 36, 9, True)
        fmap, _ = extract_local_features(defective, self.feature_config)
        score, image_score, mask = predict_anomaly(fmap, model, NormalityConfig())
        self.assertGreater(float(score[truth].mean()), float(score[~truth].mean()))
        self.assertGreater(image_score, 0)
        self.assertEqual(mask.dtype, bool)
        self.assertFalse(mask[:5].any())
        self.assertFalse(mask[:, :5].any())

    def test_validation_threshold(self):
        score = np.array([[0.0, 1.0], [2.0, 4.0]])
        truth = np.array([[0, 0], [1, 1]], bool)
        threshold = select_mask_threshold([score], [truth], candidates=20)
        self.assertGreaterEqual(threshold, 1)
        self.assertLessEqual(threshold, 2.1)

    def test_validation_threshold_rejects_mismatched_shapes(self):
        with self.assertRaises(ValueError):
            select_mask_threshold([np.zeros((2, 2))], [np.zeros((3, 3), bool)])


if __name__ == "__main__":
    unittest.main()
