"""End-to-end Task A/B/C and personal-photo deployment helpers."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .config import FeatureConfig, NormalityConfig
from .features import extract_local_features, global_pool
from sklearn.pipeline import Pipeline
from .models import fit_attribute_classifier, fit_material_classifier
from .normality import NormalModel, fit_normal_model, predict_anomaly


def descriptors(images: list[np.ndarray], config: FeatureConfig) -> tuple[np.ndarray, list[str]]:
    vectors, names = [], None
    for image in images:
        fmap, current = extract_local_features(image, config)
        if names is not None and names != current: raise RuntimeError("feature channel order changed")
        names = current; vectors.append(global_pool(fmap))
    return np.stack(vectors), names or []


def train_material_classifier(images: list[np.ndarray], labels, config: FeatureConfig) -> Pipeline:
    x, _ = descriptors(images, config)
    return fit_material_classifier(x, np.asarray(labels))


def train_attribute_predictors(images: list[np.ndarray], labels: np.ndarray, attributes: list[str],
                               config: FeatureConfig) -> Pipeline:
    x, _ = descriptors(images, config)
    return fit_attribute_classifier(x, labels, attributes)


def train_material_normal_models(images: list[np.ndarray], materials, feature_config: FeatureConfig,
                                 normal_config: NormalityConfig) -> dict[str, NormalModel]:
    grouped: dict[str, list[np.ndarray]] = {}
    feature_names = None
    for image, material in zip(images, materials):
        fmap, feature_names = extract_local_features(image, feature_config)
        grouped.setdefault(str(material), []).append(fmap)
    return {material: fit_normal_model(maps, normal_config, feature_names) for material, maps in grouped.items()}


@dataclass
class TexturePassport:
    material: str
    material_confidence: float
    attributes: list[tuple[str, float]]
    anomaly_score: float
    anomaly_map: np.ndarray
    anomaly_mask: np.ndarray
    description: str
    limitation: str


def make_texture_passport(image: np.ndarray, material_model: Pipeline,
                          attribute_model: Pipeline, normal_models: dict[str, NormalModel],
                          feature_config: FeatureConfig, normal_config: NormalityConfig,
                          top_k: int = 5) -> TexturePassport:
    fmap, _ = extract_local_features(image, feature_config); vector = global_pool(fmap)[None]
    material_prob = material_model.predict_proba(vector)[0]; material_i = int(material_prob.argmax())
    material = str(material_model.classes_[material_i])
    attr_prob = attribute_model.predict_proba(vector)[0]
    ranked = np.argsort(attr_prob)[::-1][:top_k]
    attrs = [(attribute_model.attribute_names_[i], float(attr_prob[i])) for i in ranked]
    amap, ascore, mask = predict_anomaly(fmap, normal_models[material], normal_config)
    evidence = ", ".join(name for name, score in attrs[:3] if score >= .5) or "no high-confidence attributes"
    description = f"Predicted {material} texture with visible evidence consistent with {evidence}."
    limitation = ("Scores come from handcrafted colour, frequency, and edge cues; viewpoint, lighting, "
                  "and unfamiliar materials can cause confident errors. Inspect the heatmap before acting.")
    return TexturePassport(material, float(material_prob[material_i]), attrs, ascore, amap, mask,
                           description, limitation)
