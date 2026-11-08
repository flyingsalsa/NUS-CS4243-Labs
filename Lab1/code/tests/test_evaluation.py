import unittest
import numpy as np
from texturelab.evaluation import evaluate_attributes, evaluate_detection, evaluate_material


class TestEvaluation(unittest.TestCase):
    def test_material_groups(self):
        result = evaluate_material(["a","a","b","b"], np.array([[.9,.1],[.6,.4],[.2,.8],[.7,.3]]),
                                   ["a","b"], ["good","defective","good","defective"])
        self.assertEqual(set(result), {"overall","good","defective"})
        self.assertEqual(result["good"]["accuracy"], 1)

    def test_detection_breakdown(self):
        truth = [np.zeros((2,2),bool), np.array([[1,0],[0,0]],bool)]
        pred = [np.zeros((2,2),bool), np.array([[1,0],[0,0]],bool)]
        result = evaluate_detection([0,1],[.1,.9],truth,pred,["good","scratch"])
        self.assertEqual(result["image_auroc"], 1); self.assertEqual(result["pixel"]["iou"], 1)

    def test_attributes(self):
        y = np.array([[1,0],[0,1]],bool); p = np.array([[.9,.1],[.2,.8]])
        result = evaluate_attributes(y,p,["x","y"])
        self.assertEqual(result["macro_average_precision"], 1); self.assertEqual(result["macro_f1"], 1)

    def test_attribute_average_precision_groups_tied_scores(self):
        y = np.array([[0], [1]], bool)
        p = np.array([[0.5], [0.5]])
        result = evaluate_attributes(y, p, ["x"])
        self.assertEqual(result["macro_average_precision"], 0.5)


if __name__ == "__main__": unittest.main()
