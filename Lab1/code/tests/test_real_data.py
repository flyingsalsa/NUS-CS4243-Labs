import unittest
import csv
from pathlib import Path
import numpy as np
from texturelab.config import FeatureConfig, GaborConfig
from texturelab.data import load_mask, read_manifest, validate_manifest
from texturelab.features import extract_local_features
from texturelab.supplied import load_image

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" if (ROOT / "data").is_dir() else ROOT.parent / "data"


@unittest.skipUnless(DATA_ROOT.is_dir(), "real course datasets are not installed")
class TestRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mvtec = read_manifest(ROOT / "manifests" / "mvtec_textures.csv")
        cls.dtd = read_manifest(ROOT / "manifests" / "dtd_split1.csv")

    def test_complete_manifest_counts_and_files(self):
        validate_manifest(self.mvtec)
        validate_manifest(self.dtd)
        self.assertEqual(len(self.mvtec), 1781)
        self.assertEqual(len(self.dtd), 5640)
        self.assertEqual(sum(r.mask_path != "" for r in self.mvtec), 382)
        self.assertEqual(sum(r.split == "validation" for r in self.mvtec), 267)
        self.assertEqual(sum(r.split == "test" for r in self.mvtec), 248)
        self.assertEqual({r.split for r in self.mvtec}, {"train", "validation", "test"})
        self.assertEqual({r.split for r in self.dtd}, {"train", "validation", "test"})

        # Every defect type is represented in both public validation and test.
        groups = {
            (r.material, r.defect_type) for r in self.mvtec if r.defect_type != "good"
        }
        for material, defect in groups:
            splits = {
                r.split
                for r in self.mvtec
                if r.material == material and r.defect_type == defect
            }
            self.assertEqual(splits, {"validation", "test"})

    def test_real_mvtec_feature_and_mask_alignment(self):
        record = next(r for r in self.mvtec if r.mask_path)
        image = load_image(record.path, (48, 48))
        mask = load_mask(record.mask_path, (48, 48))
        config = FeatureConfig(
            gabor=GaborConfig(
                frequencies=(0.15,), orientations=(0,), kernel_size=9, pool_size=5
            )
        )
        features, names = extract_local_features(image, config)
        self.assertEqual(features.shape[:2], mask.shape)
        self.assertEqual(features.shape[-1], len(names))
        self.assertTrue(np.isfinite(features).all())
        self.assertTrue(mask.any())

    def test_dtd_joint_annotations_are_present(self):
        self.assertTrue(all(r.attributes for r in self.dtd))
        self.assertEqual(len({r.attributes[0] for r in self.dtd}), 47)

    def test_task_d_uses_test_queries_and_validation_support(self):
        with (ROOT / "manifests" / "task_d_queries.csv").open(newline="") as handle:
            queries = list(csv.DictReader(handle))
        with (ROOT / "manifests" / "task_d_support.csv").open(newline="") as handle:
            support = list(csv.DictReader(handle))
        self.assertEqual(len(queries), 5)
        self.assertEqual(len(support), 25)
        by_path = {r.path: r for r in self.mvtec}
        self.assertTrue(
            all(
                by_path[str((ROOT / "manifests" / q["path"]).resolve())].split == "test"
                for q in queries
            )
        )
        self.assertTrue(
            all(
                by_path[str((ROOT / "manifests" / s["path"]).resolve())].split
                == "validation"
                for s in support
            )
        )


if __name__ == "__main__":
    unittest.main()
