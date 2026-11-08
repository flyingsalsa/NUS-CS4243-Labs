"""Per-material diagonal normal model and dense anomaly prediction."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .config import NormalityConfig
from .supplied import box_mean


@dataclass
class NormalModel:
    mean: np.ndarray
    std: np.ndarray
    threshold: float
    feature_names: list[str] | None = None


def _scores(
    features: np.ndarray, mean: np.ndarray, std: np.ndarray, epsilon: float
) -> np.ndarray:
    return np.sqrt(np.mean(((features - mean) / (std + epsilon)) ** 2, axis=-1)).astype(
        np.float32
    )


def _pool_score(score: np.ndarray, config: NormalityConfig) -> np.ndarray:
    """Suppress isolated patch responses while preserving coherent anomalies."""
    if config.score_pool_size < 1 or config.score_pool_size % 2 == 0:
        raise ValueError("score_pool_size must be a positive odd integer")
    return box_mean(score, config.score_pool_size).astype(np.float32)


def _interior(score: np.ndarray, border: int) -> np.ndarray:
    if border <= 0:
        return score
    if 2 * border >= min(score.shape):
        raise ValueError("ignore_border is too large for the score map")
    return score[border:-border, border:-border]


def fit_normal_model(feature_maps: list[np.ndarray] | np.ndarray, config: NormalityConfig,
                     feature_names: list[str] | None = None) -> NormalModel:
    """Fit normal feature statistics using normal training maps only."""
    # YOUR CODE HERE
    #
    # TODO 1 — Normalise and validate the collection
    #   - Accept either one H x W x D array or a non-empty collection of maps.
    #   - Reject an empty collection; preserve the common feature dimension.
    #
    # TODO 2 — Build the normal patch sample
    #   - Flatten each map to (H*W) x D and concatenate across training images.
    #   - If there are more than config.max_samples rows, select exactly that
    #     many without replacement using config.random_seed.
    #
    # TODO 3 — Fit diagonal normal statistics
    #   - Compute a D-vector mean and population standard deviation.
    #
    # TODO 4 — Estimate a normal-only provisional threshold
    #   - Score every original map with _scores, smooth it with _pool_score, crop
    #     the configured unreliable border with _interior, then concatenate.
    #   - Use config.mask_threshold when explicitly supplied; otherwise take
    #     config.threshold_percentile of the pooled interior normal scores.
    #
    # TODO 5 — Return the model
    #   - Store float32 mean/std, a Python-float threshold, and feature_names.
    if isinstance(feature_maps, np.ndarray):
        maps = [feature_maps]
    else:
        maps = list(feature_maps)
    if not maps:
        raise ValueError("feature_maps must be non-empty")

    normal_maps: list[np.ndarray] = []
    feature_dim: int | None = None
    for fmap in maps:
        fmap = np.asarray(fmap)
        if fmap.ndim != 3 or fmap.shape[0] == 0 or fmap.shape[1] == 0:
            raise ValueError("each feature map must be a non-empty HxWxD array")
        if feature_dim is None:
            feature_dim = fmap.shape[-1]
        elif fmap.shape[-1] != feature_dim:
            raise ValueError("all feature maps must have the same feature dimension")
        if not np.isfinite(fmap).all():
            raise ValueError("feature maps must contain only finite values")
        normal_maps.append(fmap)

    samples = np.concatenate(
        [fmap.reshape(-1, feature_dim) for fmap in normal_maps], axis=0
    )
    if config.max_samples < 1:
        raise ValueError("max_samples must be positive")
    if samples.shape[0] > config.max_samples:
        rng = np.random.default_rng(config.random_seed)
        indices = rng.choice(samples.shape[0], size=config.max_samples, replace=False)
        samples = samples[indices]

    mean = np.mean(samples, axis=0).astype(np.float32)
    std = np.std(samples, axis=0, ddof=0).astype(np.float32)
    pooled_normal_scores = np.concatenate(
        [
            _interior(
                _pool_score(_scores(fmap, mean, std, config.epsilon), config),
                config.ignore_border,
            ).ravel()
            for fmap in normal_maps
        ]
    )
    if pooled_normal_scores.size == 0:
        raise ValueError("ignore_border leaves no normal score pixels")
    threshold = (
        float(config.mask_threshold)
        if config.mask_threshold is not None
        else float(np.percentile(pooled_normal_scores, config.threshold_percentile))
    )
    return NormalModel(mean, std, threshold, feature_names)


def predict_anomaly(feature_map: np.ndarray, model: NormalModel, config: NormalityConfig
                    ) -> tuple[np.ndarray, float, np.ndarray]:
    """Return dense score, image-level score, and predicted mask."""
    # YOUR CODE HERE
    #
    # TODO 1 — Validate feature compatibility
    #   - The final feature dimension must equal the fitted model mean length.
    #
    # TODO 2 — Compute the dense anomaly score
    #   - Use _scores for RMS standardized distance with config.epsilon.
    #   - Smooth coherent evidence with _pool_score.
    #
    # TODO 3 — Aggregate an image-level score
    #   - Crop the unreliable border with _interior and take
    #     config.image_percentile over that interior only.
    #
    # TODO 4 — Produce the binary localization mask
    #   - Threshold the full score map at model.threshold.
    #   - Force the configured outer border to False without altering scores.
    #   - Return (float32 score map, Python-float image score, Boolean mask).
    feature_map = np.asarray(feature_map)
    if feature_map.ndim != 3:
        raise ValueError("feature_map must be an HxWxD array")
    if feature_map.shape[-1] != model.mean.shape[0]:
        raise ValueError("feature dimension does not match the fitted model")
    if not np.isfinite(feature_map).all():
        raise ValueError("feature_map must contain only finite values")

    score = _pool_score(
        _scores(feature_map, model.mean, model.std, config.epsilon), config
    ).astype(np.float32)
    image_score = float(
        np.percentile(_interior(score, config.ignore_border), config.image_percentile)
    )
    mask = score >= model.threshold
    border = config.ignore_border
    if border > 0:
        if 2 * border >= min(mask.shape):
            raise ValueError("ignore_border is too large for the score map")
        mask[:border, :] = False
        mask[-border:, :] = False
        mask[:, :border] = False
        mask[:, -border:] = False
    return score, image_score, mask.astype(bool)


def select_mask_threshold(score_maps: list[np.ndarray], masks: list[np.ndarray],
                          candidates: int = 80) -> float:
    """Choose the pixel-F1-optimal threshold on public validation only."""
    # YOUR CODE HERE
    #
    # TODO 1 — Validate the validation inputs
    #   - Require matching non-empty score/mask collections and candidates >= 2.
    #   - Convert scores to float and masks to bool; require matching shapes and
    #     finite scores for every pair.
    #   - Callers handle border cropping, so do not crop again here.
    #
    # TODO 2 — Build candidate thresholds
    #   - Flatten and concatenate all validation pixels.
    #   - Use evenly spaced quantile levels from 0.5 through 0.999.
    #
    # TODO 3 — Select by pixel F1
    #   - For each threshold, predict score >= threshold and compute TP/FP/FN.
    #   - Use 2*TP / max(1, 2*TP + FP + FN) to avoid division by zero.
    #   - Keep the threshold with the greatest F1; deterministic ties should
    #     retain the first encountered candidate.
    #   - Return the selected threshold as a Python float.
    if len(score_maps) == 0 or len(score_maps) != len(masks):
        raise ValueError("score_maps and masks must be non-empty and have equal length")
    if candidates < 2:
        raise ValueError("candidates must be at least 2")

    scores: list[np.ndarray] = []
    truth_values: list[np.ndarray] = []
    for score_map, mask in zip(score_maps, masks):
        score_array = np.asarray(score_map, dtype=float)
        mask_array = np.asarray(mask, dtype=bool)
        if score_array.shape != mask_array.shape:
            raise ValueError("each score map and mask must have matching shapes")
        if not np.isfinite(score_array).all():
            raise ValueError("score maps must contain only finite values")
        scores.append(score_array.ravel())
        truth_values.append(mask_array.ravel())

    all_scores = np.concatenate(scores)
    all_truth = np.concatenate(truth_values)
    if all_scores.size == 0:
        raise ValueError("score maps must contain at least one pixel")
    thresholds = np.quantile(
        all_scores, np.linspace(0.5, 0.999, candidates)
    )

    best_threshold = float(thresholds[0])
    best_f1 = -1.0
    for threshold in thresholds:
        predicted = all_scores >= threshold
        true_positive = np.count_nonzero(predicted & all_truth)
        false_positive = np.count_nonzero(predicted & ~all_truth)
        false_negative = np.count_nonzero(~predicted & all_truth)
        denominator = max(1, 2 * true_positive + false_positive + false_negative)
        f1 = (2 * true_positive) / denominator
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold
