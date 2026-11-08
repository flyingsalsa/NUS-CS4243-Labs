"""Stage 3 Composite: occlusion handling and compositing.

Implement the functions marked ``YOUR CODE HERE``. Motion-compensated
differencing is the required occlusion signal; ``mode="colour"`` is an
optional ablation. Permitted library calls: ``cv2.morphologyEx``,
``cv2.connectedComponentsWithStats``, ``cv2.distanceTransform`` (feathering),
and your Lab 1 ``hysteresis``.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from .config import OcclusionConfig, CompositeConfig
from .supplied import to_gray, to_float
from .geometry import warp_artwork
from . import metrics


def hysteresis(strong: np.ndarray, weak: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """Keep weak pixels reachable from strong pixels (reuse your Lab 1 implementation).

    ``strong`` and ``weak`` are Boolean masks of the same shape (weak includes
    strong). Traverse with the 4- or 8-connected neighbourhood from every
    strong pixel, through chains of weak pixels, and return the Boolean mask
    of reached pixels. Validate shapes and the connectivity value.
    """
    # YOUR CODE HERE
    strong, weak = np.asarray(strong, dtype=bool), np.asarray(weak, dtype=bool)
    if strong.shape != weak.shape or strong.ndim != 2:
        raise ValueError("Masks must have the same two-dimensional shape")
    if connectivity not in (4, 8):
        raise ValueError("Connectivity must be 4 or 8")
    reached = strong.copy()
    stack = list(zip(*np.nonzero(strong)))
    neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        neighbours += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    h, w = weak.shape
    while stack:
        y, x = stack.pop()
        for dy, dx in neighbours:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and weak[yy, xx] and not reached[yy, xx]:
                reached[yy, xx] = True
                stack.append((yy, xx))
    return reached


def compensated_difference(frame: np.ndarray, warped_reference: np.ndarray, coverage: np.ndarray,
                           config: OcclusionConfig) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Per-pixel difference after a global gain/offset fit, zero outside ``coverage``.

    Fit I_frame ~ a * I_ref + b on the ``config.gain_quantile`` fraction of
    covered pixels whose raw difference is smallest (so that illumination
    change alone does not become an occluder), then return the mean absolute
    difference over the colour channels between the frame and the compensated
    reference, together with (a, b). Use colour, not grayscale alone: a
    coloured occluder over a coloured board can have almost the same luminance
    (the ball over the blue board differs by only ~0.13 in grayscale but ~0.3
    in mean channel difference), and the default thresholds assume colour.
    """
    # YOUR CODE HERE
    observed, reference = to_float(frame), to_float(warped_reference)
    coverage = np.asarray(coverage, dtype=bool)
    difference = np.zeros(coverage.shape, dtype=np.float32)
    if not coverage.any():
        return difference, (1.0, 0.0)
    raw = np.abs(observed - reference)
    raw = raw.mean(axis=-1) if raw.ndim == 3 else raw
    cutoff = np.quantile(raw[coverage], config.gain_quantile)
    agreeing = coverage & (raw <= cutoff)
    x, y = reference[agreeing].astype(float).ravel(), observed[agreeing].astype(float).ravel()
    centred = x - x.mean()
    denominator = np.dot(centred, centred)
    gain = float(np.dot(centred, y - y.mean()) / denominator) if denominator > 1e-12 else 1.0
    offset = float(y.mean() - gain * x.mean())
    error = np.abs(observed - (gain * reference + offset))
    error = error.mean(axis=-1) if error.ndim == 3 else error
    difference[coverage] = error[coverage]
    return difference, (gain, offset)


def occlusion_mask(frame: np.ndarray, H: np.ndarray, reference: np.ndarray, config: OcclusionConfig
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Foreground mask inside the board by motion-compensated differencing.

    1. Warp ``reference`` into the frame with ``H`` (``warp_artwork``) and
       compute the ``compensated_difference`` inside the coverage mask.
    2. Threshold at ``config.strong`` and ``config.weak`` and apply
       ``hysteresis`` with ``config.connectivity``.
    3. Open with a disc of radius ``config.open_radius`` (``cv2.morphologyEx``:
       erosion then dilation, which deletes anything thinner than 2r+1 pixels;
       0 disables) and keep connected components with at least
       ``config.min_area`` pixels.
    4. Return (mask bool H x W, difference map float H x W, component stats).

    ``component stats`` is an array with one row per kept component:
    [label, x, y, width, height, area]. With ``config.mode == "colour"`` an
    appearance model may be used instead (optional ablation).
    """
    # YOUR CODE HERE
    if config.mode == "colour":
        raise NotImplementedError("Optional colour-histogram ablation")
    if config.mode != "difference":
        raise ValueError("Unknown occlusion mode: " + config.mode)
    warped, coverage = warp_artwork(to_float(reference), H, frame.shape)
    difference, _ = compensated_difference(frame, warped, coverage, config)
    labels, stats = _difference_components(difference, coverage, config)
    keep = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= config.min_area)
    keep = keep[keep != 0]
    mask = np.isin(labels, keep) & coverage
    return mask, difference, np.column_stack((keep, stats[keep])).astype(int)


def select_difference_thresholds(frames: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                                 masks: Sequence[np.ndarray], config: OcclusionConfig
                                 ) -> Tuple[OcclusionConfig, List[Dict]]:
    """Choose strong/weak thresholds and the minimum area on validation masks only.

    ``frames`` is a sequence of (frame, H, reference) triples and ``masks`` the
    matching ground-truth occlusion masks. Grid-search the thresholds and the
    minimum area, score each setting by mean ``metrics.mask_iou`` over the
    validation frames, and return (frozen config, search log) where the log
    holds one dict per setting with its parameters and score.

    Several settings usually tie at the best score: the few validation masks
    do not separate the strong threshold. Break ties towards the middle of the
    tied range of ``strong`` values (not the first or the largest grid point),
    so that the frozen value sits inside the plateau rather than on its edge,
    and state the rule in the report.
    """
    # YOUR CODE HERE
    if len(frames) != len(masks) or not len(frames):
        raise ValueError("Supply matching nonempty validation frames and masks")
    if config.mode != "difference":
        raise ValueError("Threshold selection requires difference mode")
    prepared = []
    for (frame, H, reference), mask in zip(frames, masks):
        warped, coverage = warp_artwork(to_float(reference), H, frame.shape)
        difference, _ = compensated_difference(frame, warped, coverage, config)
        if mask.shape != coverage.shape:
            raise ValueError("Validation mask shape does not match its frame")
        prepared.append((difference, coverage, mask))
    strong_grid = np.unique(np.clip(config.strong * np.array([0.6, 0.8, 1.0, 1.2, 1.4]), 0, 1))
    weak_grid = np.unique(np.clip(config.weak * np.array([2 / 3, 1.0, 4 / 3]), 0, 1))
    area_grid = np.unique(np.maximum(1, np.rint(config.min_area * np.array([0.5, 1.0, 2.0])).astype(int)))
    log = []
    for strong in strong_grid:
        for weak in weak_grid:
            if weak >= strong:
                continue
            candidate = replace(config, strong=float(strong), weak=float(weak))
            components = [_difference_components(diff, cov, candidate) for diff, cov, _ in prepared]
            for area in area_grid:
                scores = []
                for (labels, stats), (_, coverage, gt) in zip(components, prepared):
                    keep = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= area)
                    keep = keep[keep != 0]
                    mask = np.isin(labels, keep) & coverage
                    score = metrics.mask_iou(mask, gt)
                    scores.append(score)
                finite = np.asarray(scores)[np.isfinite(scores)]
                log.append({"strong": float(strong), "weak": float(weak), "min_area": int(area),
                            "mean_iou": float(finite.mean()) if finite.size else float("nan")})
    scored = [row for row in log if np.isfinite(row["mean_iou"])]
    if not scored:
        raise ValueError("Validation masks produced no defined IoU scores")
    best_score = max(row["mean_iou"] for row in scored)
    tied = [row for row in scored if np.isclose(row["mean_iou"], best_score, rtol=0, atol=1e-12)]
    midpoint = (min(row["strong"] for row in tied) + max(row["strong"] for row in tied)) / 2
    chosen = min(tied, key=lambda row: (abs(row["strong"] - midpoint),
                                       abs(row["weak"] - config.weak), abs(row["min_area"] - config.min_area),
                                       row["strong"], row["weak"], row["min_area"]))
    return replace(config, **{key: chosen[key] for key in ("strong", "weak", "min_area")}), log


def composite(frame: np.ndarray, warped: np.ndarray, coverage: np.ndarray, occlusion: np.ndarray,
              config: CompositeConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Alpha-composite the warped artwork into the frame.

    alpha = C * (1 - O) * F, where C is the coverage mask, O the occlusion
    mask and F an optional feathering term that softens the quadrilateral and
    occluder boundaries over ``config.feather`` pixels (0 disables).
    I_out = alpha * W + (1 - alpha) * I. Optional brightness matching of the
    artwork to the board is controlled by ``config.match_gain``.
    Returns (composited RGB float32 frame, alpha map float32 in [0, 1]).
    """
    # YOUR CODE HERE
    original, artwork = to_float(frame), to_float(warped).copy()
    visible = np.asarray(coverage, dtype=bool) & ~np.asarray(occlusion, dtype=bool)
    alpha = visible.astype(np.float32)
    if config.feather > 0:
        padded = np.pad(visible.astype(np.uint8), 1)
        distance = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
        alpha *= np.clip(distance / config.feather, 0, 1)
    if config.match_gain and visible.any():
        mean_art = float(to_gray(artwork)[visible].mean())
        mean_frame = float(to_gray(original)[visible].mean())
        if mean_art > 1e-12:
            artwork = np.clip(artwork * (mean_frame / mean_art), 0, 1)
    blend = alpha[..., None] if original.ndim == 3 else alpha
    out = blend * artwork + (1 - blend) * original
    return out.astype(np.float32), alpha.astype(np.float32)


def _difference_components(difference, coverage, config):
    strong = coverage & (difference >= config.strong)
    weak = coverage & (difference >= config.weak)
    mask = hysteresis(strong, weak, config.connectivity)
    if config.open_radius > 0:
        radius = config.open_radius
        yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        kernel = (xx * xx + yy * yy <= radius * radius).astype(np.uint8)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
    mask &= coverage
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=config.connectivity)
    return labels, stats


__all__ = ["hysteresis", "compensated_difference", "occlusion_mask", "select_difference_thresholds", "composite"]
