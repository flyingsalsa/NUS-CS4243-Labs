import unittest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from texturelab.metrics import binary_pixel_metrics, confusion_matrix, expected_calibration_error, macro_f1, roc_auc
from texturelab.models import fit_attribute_classifier, fit_material_classifier


class TestModelsMetrics(unittest.TestCase):
    def test_multiclass_and_probabilities(self):
        x = np.array([[0,0],[.1,0],[2,2],[2.1,2],[0,3],[.1,3.]])
        y = np.array(["a","a","b","b","c","c"])
        model = fit_material_classifier(x, y, C=100); probability = model.predict_proba(x)
        self.assertIsInstance(model, Pipeline)
        self.assertIsInstance(model.named_steps["standardscaler"], StandardScaler)
        self.assertIsInstance(model.named_steps["logisticregression"], LogisticRegression)
        self.assertTrue(np.allclose(probability.sum(1), 1))
        self.assertGreaterEqual(np.mean(model.predict(x) == y), .8)

    def test_multilabel(self):
        x = np.array([[0],[1],[2],[3]], float); y = np.array([[0,1],[0,1],[1,0],[1,0]])
        model = fit_attribute_classifier(x, y, ["high", "low"], C=100)
        self.assertIsInstance(model.named_steps["onevsrestclassifier"], OneVsRestClassifier)
        self.assertEqual(model.attribute_names_, ["high", "low"])
        self.assertEqual(model.predict_proba(x).shape, y.shape)

    def test_metrics(self):
        y = np.array(["a","a","b"]); p = np.array(["a","b","b"])
        self.assertEqual(confusion_matrix(y,p,["a","b"]).tolist(), [[1,1],[0,1]])
        self.assertGreater(macro_f1(y,p,["a","b"]), .6)
        self.assertEqual(roc_auc([0,0,1,1],[.1,.2,.8,.9]), 1)
        self.assertEqual(binary_pixel_metrics([0,1],[0,1])["iou"], 1)
        self.assertGreaterEqual(expected_calibration_error(y, np.array([[.8,.2],[.7,.3],[.1,.9]]), ["a","b"]), 0)

    def test_ece_includes_confidence_one(self):
        error = expected_calibration_error(["b"], [[1.0, 0.0]], ["a", "b"])
        self.assertEqual(error, 1.0)


if __name__ == "__main__": unittest.main()
