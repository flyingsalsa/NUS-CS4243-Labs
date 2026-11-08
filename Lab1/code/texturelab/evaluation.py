"""Task-level reports with the assignment's required group breakdowns."""

from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score
from .metrics import binary_pixel_metrics, confusion_matrix, expected_calibration_error, macro_f1, roc_auc


def evaluate_material(y_true, probabilities, classes, conditions=None) -> dict:
    """Report accuracy/F1/confusion/ECE overall and for good/defective groups."""
    y_true, probabilities, classes = np.asarray(y_true), np.asarray(probabilities), np.asarray(classes)
    predicted = classes[probabilities.argmax(1)]
    conditions = np.asarray(conditions if conditions is not None else ["all"] * len(y_true))
    report = {}
    groups = ["overall"] + list(dict.fromkeys(conditions.tolist()))
    for group in groups:
        mask = np.ones(len(y_true), bool) if group == "overall" else conditions == group
        if not mask.any(): continue
        report[group] = {
            "count": int(mask.sum()), "accuracy": float(np.mean(predicted[mask] == y_true[mask])),
            "macro_f1": macro_f1(y_true[mask], predicted[mask], classes),
            "confusion_matrix": confusion_matrix(y_true[mask], predicted[mask], classes).tolist(),
            "ece": expected_calibration_error(y_true[mask], probabilities[mask], classes),
        }
    return report


def evaluate_detection(image_labels, image_scores, true_masks, predicted_masks, defect_types) -> dict:
    """Report image AUROC and pixel metrics overall and per defect type."""
    image_labels = np.asarray(image_labels, bool); image_scores = np.asarray(image_scores, float)
    defect_types = np.asarray(defect_types)
    report = {"image_auroc": roc_auc(image_labels, image_scores), "by_defect_type": {}}
    truth_all = np.concatenate([np.asarray(m, bool).ravel() for m in true_masks])
    pred_all = np.concatenate([np.asarray(m, bool).ravel() for m in predicted_masks])
    report["pixel"] = binary_pixel_metrics(truth_all, pred_all)
    for defect in dict.fromkeys(defect_types.tolist()):
        selected = np.flatnonzero(defect_types == defect)
        truth = np.concatenate([np.asarray(true_masks[i], bool).ravel() for i in selected])
        pred = np.concatenate([np.asarray(predicted_masks[i], bool).ravel() for i in selected])
        report["by_defect_type"][str(defect)] = binary_pixel_metrics(truth, pred)
    return report


def _average_precision(y_true, score) -> float:
    y_true, score = np.asarray(y_true, bool), np.asarray(score, float)
    positives = int(y_true.sum())
    if positives == 0: return float("nan")
    # Group equal scores at the same decision threshold. Ranking positives
    # individually makes AP depend on arbitrary input order within ties.
    return float(average_precision_score(y_true, score))


def evaluate_attributes(y_true, probabilities, attributes, threshold: float = .5) -> dict:
    """Return per-attribute AP/F1 and macro summaries for Task C."""
    y_true, probabilities = np.asarray(y_true, bool), np.asarray(probabilities, float)
    predicted = probabilities >= threshold; rows = {}
    for i, attribute in enumerate(attributes):
        metrics = binary_pixel_metrics(y_true[:, i], predicted[:, i])
        rows[attribute] = {"average_precision": _average_precision(y_true[:, i], probabilities[:, i]),
                           "f1": metrics["f1"], "positives": int(y_true[:, i].sum())}
    return {"macro_average_precision": float(np.nanmean([v["average_precision"] for v in rows.values()])),
            "macro_f1": float(np.mean([v["f1"] for v in rows.values()])), "per_attribute": rows}
