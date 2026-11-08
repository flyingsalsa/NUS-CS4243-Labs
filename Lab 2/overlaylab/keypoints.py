"""Stage 1 Localise, keypoint branch.

Implement the four functions marked ``YOUR CODE HERE``. Do not change the
signatures or return types; the tests and the grading harness call them.
Use only numpy and the helpers in ``overlaylab.supplied`` (Sobel gradients,
Gaussian smoothing, bilinear sampling). OpenCV feature functions
(``cv2.cornerHarris``, ``cv2.SIFT`` and friends) are not allowed here; the
SIFT baseline lives only in the labelled ablation cell of Notebook 1.

Conventions: grayscale images are float32 in [0, 1]; ``x`` indexes columns and
``y`` rows; keypoints are N x 2 arrays of (x, y); descriptors are N x 128.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .config import DetectorConfig, DescriptorConfig, MatchConfig
from .supplied import sobel_gradients, gaussian_smooth
from .geometry import bilinear_sample


def harris_response(gray: np.ndarray, config: DetectorConfig) -> np.ndarray:
    """Harris response R = det(M) - kappa * tr(M)^2 at every pixel.

    M is the Gaussian-weighted (scale ``config.sigma_w``) second-moment matrix
    built from the supplied normalised Sobel gradients. Returns a float32 map
    with the same shape as ``gray``.
    """
    # YOUR CODE HERE
    gx, gy, _, _ = sobel_gradients(gray)
    xx = gaussian_smooth(gx * gx, config.sigma_w)
    yy = gaussian_smooth(gy * gy, config.sigma_w)
    xy = gaussian_smooth(gx * gy, config.sigma_w)
    return (xx * yy - xy * xy - config.kappa * (xx + yy) ** 2).astype(np.float32)


def detect_corners(gray: np.ndarray, config: DetectorConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Harris corners with non-maximum suppression.

    Steps: compute ``harris_response``; discard responses below
    ``config.response_fraction * max(R)``; apply NMS in mode ``config.nms_mode``
    with radius ``config.nms_radius`` ("window": keep a pixel only if it is the
    maximum of its (2r+1)x(2r+1) neighbourhood; "radius": sort survivors by R
    and greedily accept the strongest while discarding weaker candidates within
    distance r); drop corners within ``config.border`` pixels of the image edge.

    Returns (keypoints, responses): an N x 2 float array of (x, y) ordered by
    decreasing response (N <= config.max_corners) and the matching responses.
    """
    # YOUR CODE HERE
    if config.nms_mode not in ("window", "radius"):
        raise ValueError("Unknown NMS mode: " + config.nms_mode)
    R = harris_response(gray, config)
    h, w = R.shape
    r, border = config.nms_radius, config.border
    candidates = np.isfinite(R) & (R > 0) & (R >= config.response_fraction * R.max())
    yy, xx = np.indices(R.shape)
    candidates &= (xx >= border) & (xx < w - border) & (yy >= border) & (yy < h - border)
    if config.nms_mode == "window":
        padded = np.pad(R, r, constant_values=-np.inf)
        maxima = np.full_like(R, -np.inf)
        for dy in range(2 * r + 1):
            for dx in range(2 * r + 1):
                np.maximum(maxima, padded[dy:dy + h, dx:dx + w], out=maxima)
        candidates &= R == maxima
    ys, xs = np.nonzero(candidates)
    order = np.argsort(-R[ys, xs], kind="stable")
    accepted = []
    suppressed = np.zeros_like(candidates)
    for index in order:
        if len(accepted) >= config.max_corners:
            break
        y, x = ys[index], xs[index]
        if config.nms_mode == "radius":
            if suppressed[y, x]:
                continue
            y0, y1 = max(0, y - r), min(h, y + r + 1)
            x0, x1 = max(0, x - r), min(w, x + r + 1)
            dy, dx = np.ogrid[y0:y1, x0:x1]
            suppressed[y0:y1, x0:x1] |= (dx - x) ** 2 + (dy - y) ** 2 <= r * r
        accepted.append(index)
    accepted = np.asarray(accepted, dtype=int)
    return np.column_stack((xs[accepted], ys[accepted])).astype(np.float32), R[ys[accepted], xs[accepted]]


def describe_keypoints(gray: np.ndarray, keypoints: np.ndarray,
                       config: DescriptorConfig) -> Tuple[np.ndarray, np.ndarray]:
    """128-D gradient-histogram descriptors with dominant-orientation normalisation.

    For each keypoint take a 16 x 16 grid of samples spaced ``config.patch_scale``
    pixels apart (so the window is 16 * patch_scale pixels wide),
    build a ``config.orientation_bins``-bin orientation histogram weighted by
    gradient magnitude, assign the dominant orientation, rotate the sampling
    grid so that the first histogram bin is aligned to it, and accumulate a
    ``cells x cells`` grid of ``cell_bins``-bin histograms (4 x 4 x 8 = 128).
    Apply Gaussian spatial weighting when ``config.gaussian_weighting`` is set,
    normalise to unit L2 norm, clamp to ``config.clamp``, and renormalise.

    Keypoints whose (rotated) window leaves the image are dropped. Returns
    (descriptors N' x 128, kept_indices N') where kept_indices indexes the input.
    """
    # YOUR CODE HERE
    gx, gy, _, _ = sobel_gradients(gray)
    h, w = gray.shape
    yy, xx = np.meshgrid(np.arange(16) - 7.5, np.arange(16) - 7.5, indexing="ij")
    weight = np.exp(-(xx * xx + yy * yy) / (2 * 8.0 ** 2)) if config.gaussian_weighting else np.ones((16, 16))
    dx, dy = xx * config.patch_scale, yy * config.patch_scale
    cell_y = np.minimum(np.arange(16) * config.cells // 16, config.cells - 1)
    cell_index = (cell_y[:, None] * config.cells + cell_y[None, :]) * config.cell_bins
    descriptors, kept = [], []
    size = config.cells ** 2 * config.cell_bins

    def inside(x, y):
        return np.isfinite(x).all() and np.isfinite(y).all() and x.min() >= 0 and x.max() <= w - 1 and y.min() >= 0 and y.max() <= h - 1

    for index, (x, y) in enumerate(np.asarray(keypoints).reshape(-1, 2)):
        sx, sy = x + dx, y + dy
        if not inside(sx, sy):
            continue
        ix, iy = bilinear_sample(gx, sy, sx), bilinear_sample(gy, sy, sx)
        magnitude = np.hypot(ix, iy) * weight
        angle = np.mod(np.arctan2(iy, ix), 2 * np.pi)
        bins = angle * config.orientation_bins / (2 * np.pi)
        lower = np.floor(bins).astype(int)
        fraction = bins - lower
        histogram = (np.bincount((lower % config.orientation_bins).ravel(), (magnitude * (1 - fraction)).ravel(), minlength=config.orientation_bins)
                     + np.bincount(((lower + 1) % config.orientation_bins).ravel(), (magnitude * fraction).ravel(), minlength=config.orientation_bins))
        peak = int(np.argmax(histogram))
        left, middle, right = histogram[(peak - 1) % config.orientation_bins], histogram[peak], histogram[(peak + 1) % config.orientation_bins]
        denominator = left - 2 * middle + right
        shift = 0.5 * (left - right) / denominator if abs(denominator) > 1e-12 else 0.0
        theta = (peak + shift) * 2 * np.pi / config.orientation_bins
        c, s = np.cos(theta), np.sin(theta)
        sx, sy = x + c * dx - s * dy, y + s * dx + c * dy
        if not inside(sx, sy):
            continue
        ix, iy = bilinear_sample(gx, sy, sx), bilinear_sample(gy, sy, sx)
        magnitude = np.hypot(ix, iy) * weight
        bins = np.mod(np.arctan2(iy, ix) - theta, 2 * np.pi) * config.cell_bins / (2 * np.pi)
        lower = np.floor(bins).astype(int)
        fraction = bins - lower
        descriptor = (np.bincount((cell_index + lower % config.cell_bins).ravel(), (magnitude * (1 - fraction)).ravel(), minlength=size)
                      + np.bincount((cell_index + (lower + 1) % config.cell_bins).ravel(), (magnitude * fraction).ravel(), minlength=size))
        descriptor /= max(np.linalg.norm(descriptor), 1e-12)
        descriptor = np.minimum(descriptor, config.clamp)
        descriptor /= max(np.linalg.norm(descriptor), 1e-12)
        descriptors.append(descriptor)
        kept.append(index)
    return np.asarray(descriptors, dtype=np.float32).reshape(-1, size), np.asarray(kept, dtype=int)


def match_descriptors(desc_a: np.ndarray, desc_b: np.ndarray,
                      config: MatchConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest-neighbour matching with Lowe's ratio test (optionally mutual).

    For every descriptor in A find the nearest and second-nearest descriptor in
    B by Euclidean distance; accept the match when d1 / d2 < ``config.ratio``.
    With ``config.mutual`` the match must also be the nearest neighbour of the B
    descriptor in A. Returns (pairs M x 2 of (index_a, index_b), distances M,
    ratios M).
    """
    # YOUR CODE HERE
    a, b = np.asarray(desc_a, dtype=float), np.asarray(desc_b, dtype=float)
    if len(a) == 0 or len(b) < 2:
        return np.empty((0, 2), dtype=int), np.empty(0), np.empty(0)
    squared = np.maximum(np.sum(a * a, axis=1)[:, None] + np.sum(b * b, axis=1)[None, :] - 2 * a @ b.T, 0)
    nearest = np.argsort(squared, axis=1, kind="stable")[:, :2]
    rows = np.arange(len(a))
    first, second = np.sqrt(squared[rows, nearest[:, 0]]), np.sqrt(squared[rows, nearest[:, 1]])
    ratios = np.divide(first, second, out=np.full(len(a), np.inf), where=second > 0)
    accept = ratios < config.ratio
    if config.mutual:
        accept &= np.argmin(squared, axis=0)[nearest[:, 0]] == rows
    return np.column_stack((rows[accept], nearest[accept, 0])), first[accept], ratios[accept]


__all__ = ["harris_response", "detect_corners", "describe_keypoints", "match_descriptors"]
