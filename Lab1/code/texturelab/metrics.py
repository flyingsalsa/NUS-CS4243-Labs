"""Metrics used in reports; binary labels are always explicit."""

from __future__ import annotations
import numpy as np


def confusion_matrix(y_true, y_pred, labels):
    index = {v: i for i, v in enumerate(labels)}; matrix = np.zeros((len(labels), len(labels)), int)
    for truth, pred in zip(y_true, y_pred): matrix[index[truth], index[pred]] += 1
    return matrix


def macro_f1(y_true, y_pred, labels) -> float:
    scores = []
    for label in labels:
        yt, yp = np.asarray(y_true) == label, np.asarray(y_pred) == label
        tp = np.sum(yt & yp); fp = np.sum(~yt & yp); fn = np.sum(yt & ~yp)
        scores.append(2 * tp / max(1, 2 * tp + fp + fn))
    return float(np.mean(scores))


def binary_pixel_metrics(truth: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    truth, pred = np.asarray(truth, bool), np.asarray(pred, bool)
    tp = np.sum(truth & pred); fp = np.sum(~truth & pred); fn = np.sum(truth & ~pred)
    return {"f1": float(2*tp / max(1, 2*tp+fp+fn)), "iou": float(tp / max(1, tp+fp+fn))}


def roc_auc(y_true, scores) -> float:
    y, scores = np.asarray(y_true, bool), np.asarray(scores, float)
    pos, neg = scores[y], scores[~y]
    if not len(pos) or not len(neg): return float("nan")
    return float((np.sum(pos[:, None] > neg) + .5*np.sum(pos[:, None] == neg)) / (len(pos)*len(neg)))


def expected_calibration_error(y_true, probabilities, classes, bins: int = 10) -> float:
    probabilities = np.asarray(probabilities); classes = np.asarray(classes)
    pred = classes[probabilities.argmax(1)]; conf = probabilities.max(1); correct = pred == np.asarray(y_true)
    if bins < 1:
        raise ValueError("bins must be positive")
    if np.any((conf < 0) | (conf > 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    # Clipping assigns confidence == 1 to the final bin instead of dropping it.
    bin_index = np.minimum((conf * bins).astype(int), bins - 1)
    result = 0.0
    for index in range(bins):
        mask = bin_index == index
        if mask.any(): result += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(result)
