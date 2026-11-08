"""Scikit-learn model factories used by Tasks A and C."""

from __future__ import annotations
from typing import Sequence
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


def make_material_classifier(C: float = 1.0, random_state: int = 7) -> Pipeline:
    """Create a scaled, regularised five-way logistic-regression classifier."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, solver="lbfgs", max_iter=2_000, random_state=random_state),
    )


def fit_material_classifier(x: np.ndarray, y: Sequence[str], C: float = 1.0,
                            random_state: int = 7) -> Pipeline:
    """Fit and return the Task A scikit-learn pipeline."""
    return make_material_classifier(C=C, random_state=random_state).fit(np.asarray(x), np.asarray(y))


def make_attribute_classifier(C: float = 1.0, random_state: int = 7) -> Pipeline:
    """Create independent scaled logistic predictors through OneVsRestClassifier."""
    binary_model = LogisticRegression(C=C, solver="liblinear", max_iter=2_000,
                                      random_state=random_state)
    return make_pipeline(StandardScaler(), OneVsRestClassifier(binary_model))


def fit_attribute_classifier(x: np.ndarray, y: np.ndarray, attributes: Sequence[str],
                             C: float = 1.0, random_state: int = 7) -> Pipeline:
    """Fit Task C and attach the stable attribute order used by its output columns."""
    if np.asarray(y).ndim != 2 or np.asarray(y).shape[1] != len(attributes):
        raise ValueError("y must be an NxA indicator matrix matching attributes")
    model = make_attribute_classifier(C=C, random_state=random_state)
    model.fit(np.asarray(x), np.asarray(y, dtype=int))
    model.attribute_names_ = list(attributes)
    return model
