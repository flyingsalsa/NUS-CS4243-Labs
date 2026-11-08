"""Stage 1 Localise, geometry branch.

Implement the functions marked ``YOUR CODE HERE``. Allowed: numpy (including
``numpy.linalg.svd`` and ``numpy.linalg.inv``) and your Lab 1
``bilinear_sample`` (pasted into this file). ``cv2.findHomography`` and ``cv2.warpPerspective``
may appear only in the labelled sanity-check and comparison cells of Notebook 1.

Conventions: points are N x 2 arrays of (x, y) with x indexing columns; a
homography H maps source homogeneous coordinates to destination ones,
x' ~ H x, and is scaled so that H[2, 2] = 1 whenever that entry is not close
to zero. Pixel centres sit at integer coordinates.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .config import RansacConfig


def bilinear_sample(image: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Sample a 2-D or H x W x C image at floating-point image coordinates (reuse Lab 1).

    Same signature as Lab 1 (``y`` first, then ``x``); extend it so that a
    colour image returns an array with a trailing channel axis. Coordinates
    are clamped to the valid closed interval, and an exact integer coordinate
    reproduces the corresponding pixel. ``y`` and ``x`` may be arrays of any
    matching shape so that a whole grid can be sampled at once.
    """
    # YOUR CODE HERE
    image = np.asarray(image)
    yy, xx = np.broadcast_arrays(np.asarray(y, dtype=float), np.asarray(x, dtype=float))
    yy = np.clip(yy, 0, image.shape[0] - 1)
    xx = np.clip(xx, 0, image.shape[1] - 1)
    y0, x0 = np.floor(yy).astype(int), np.floor(xx).astype(int)
    y1 = np.minimum(y0 + 1, image.shape[0] - 1)
    x1 = np.minimum(x0 + 1, image.shape[1] - 1)
    wy, wx = yy - y0, xx - x0
    if image.ndim == 3:
        wy, wx = wy[..., None], wx[..., None]
    return ((1 - wy) * ((1 - wx) * image[y0, x0] + wx * image[y0, x1])
            + wy * ((1 - wx) * image[y1, x0] + wx * image[y1, x1]))


def rectangle_to_rectangle(src_size: Tuple[int, int], dst_size: Tuple[int, int]) -> np.ndarray:
    """Supplied helper: affine scaling that maps a (w, h) pixel rectangle onto another.

    Corner (w-1, h-1) of the source maps to corner (w'-1, h'-1) of the
    destination (pixel-centre convention). Used to compose the artwork
    rectangle with the board-to-frame homography.
    """
    sw, sh = src_size
    dw, dh = dst_size
    return np.array([[(dw - 1) / max(sw - 1, 1), 0, 0],
                     [0, (dh - 1) / max(sh - 1, 1), 0],
                     [0, 0, 1.0]], dtype=np.float64)


def normalise_points(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Hartley normalisation.

    Returns (T, normalised) where T is the 3 x 3 similarity that translates the
    points to zero centroid and scales them so that the mean distance from the
    origin is sqrt(2), and ``normalised`` are the transformed N x 2 points.
    """
    # YOUR CODE HERE
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or not len(pts) or not np.isfinite(pts).all():
        raise ValueError("Expected a nonempty finite N x 2 array")
    centre = pts.mean(axis=0)
    distance = np.linalg.norm(pts - centre, axis=1).mean()
    scale = np.sqrt(2) / distance if distance > np.finfo(float).eps else 1.0
    T = np.array([[scale, 0, -scale * centre[0]],
                  [0, scale, -scale * centre[1]], [0, 0, 1]])
    return T, (pts - centre) * scale


def dlt_homography(src: np.ndarray, dst: np.ndarray
                   ) -> Tuple[Optional[np.ndarray], np.ndarray, bool]:
    """Normalised direct linear transform.

    Builds the 2n x 9 system A h = 0 from normalised correspondences, solves
    for h as the right singular vector with the smallest singular value,
    denormalises H = T'^{-1} H_hat T, and scales so that H[2, 2] = 1 when that
    entry is not close to zero.

    Raises ``ValueError`` for fewer than four correspondences. Returns
    (H, singular_values, status); when A is rank deficient (e.g. collinear
    source points) ``status`` is False and ``H`` is None rather than a
    numerically meaningless matrix. ``singular_values`` are those of A, so
    that conditioning can be inspected.
    """
    # YOUR CODE HERE
    src, dst = np.asarray(src, dtype=float), np.asarray(dst, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2 or len(src) < 4:
        raise ValueError("Expected at least four paired N x 2 points")
    if not (np.isfinite(src).all() and np.isfinite(dst).all()):
        return None, np.empty(0), False
    T, s = normalise_points(src)
    U, d = normalise_points(dst)
    sh = np.column_stack((s, np.ones(len(s))))
    A = np.zeros((2 * len(src), 9))
    A[0::2, :3] = -sh
    A[1::2, 3:6] = -sh
    A[0::2, 6:] = d[:, 0, None] * sh
    A[1::2, 6:] = d[:, 1, None] * sh
    _, singular_values, vt = np.linalg.svd(A, full_matrices=len(src) == 4)
    tol = singular_values[0] * max(A.shape) * np.finfo(float).eps
    if np.count_nonzero(singular_values > tol) < 8:
        return None, singular_values, False
    H = np.linalg.inv(U) @ vt[-1].reshape(3, 3) @ T
    H /= H[2, 2] if not np.isclose(H[2, 2], 0) else np.linalg.norm(H)
    if not np.isfinite(H).all() or np.linalg.matrix_rank(H) < 3:
        return None, singular_values, False
    return H, singular_values, True


def transfer_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a homography to N x 2 points and dehomogenise: pi(H x)."""
    # YOUR CODE HERE
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    projected = np.column_stack((pts, np.ones(len(pts)))) @ np.asarray(H).T
    out = np.full((len(pts), 2), np.nan)
    np.divide(projected[:, :2], projected[:, 2, None], out=out,
              where=np.abs(projected[:, 2, None]) > np.finfo(float).eps)
    return out


def symmetric_transfer_error(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Per-correspondence e_i = |x' - pi(H x)|^2 + |x - pi(H^-1 x')|^2."""
    # YOUR CODE HERE
    try:
        inverse = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.full(len(src), np.inf)
    errors = (np.sum((transfer_points(H, src) - dst) ** 2, axis=1)
              + np.sum((transfer_points(inverse, dst) - src) ** 2, axis=1))
    return np.where(np.isfinite(errors), errors, np.inf)


def _three_collinear(p: np.ndarray, tol: float = 1e-6) -> bool:
    """Supplied helper: True if any three of the four points are (nearly) collinear."""
    from itertools import combinations
    scale = max(np.ptp(p[:, 0]), np.ptp(p[:, 1]), 1.0)
    for i, j, k in combinations(range(len(p)), 3):
        a, b, c = p[i], p[j], p[k]
        area2 = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        if area2 < tol * scale * scale:
            return True
    return False


def ransac_homography(src: np.ndarray, dst: np.ndarray, config: RansacConfig
                      ) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray, int, bool]:
    """Robust homography with adaptive RANSAC and a final refit on all inliers.

    Each iteration draws four correspondences with ``config.rng()``, rejects
    the sample if any three source points are collinear, fits
    ``dlt_homography``, and scores every correspondence by the symmetric
    transfer error; correspondences under ``config.threshold`` are inliers.
    Keep the model with the most inliers (ties: smaller sum of inlier
    residuals); update N = log(1-p) / log(1-(1-eps)^4) from the current inlier
    fraction, capped by ``config.max_iterations``; refit on all inliers.

    Returns (H, inlier_mask, residuals, iterations_used, status) where status
    is False when no model reached ``config.min_inliers`` (H may then be None).
    """
    # YOUR CODE HERE
    src, dst = np.asarray(src, dtype=float), np.asarray(dst, dtype=float)
    n = len(src)
    best_H, best_mask = None, np.zeros(n, dtype=bool)
    best_residuals = np.full(n, np.inf)
    best_count, best_cost, iterations = 0, np.inf, 0
    if n < 4:
        return best_H, best_mask, best_residuals, iterations, False
    rng = config.rng()
    limit = config.max_iterations
    while iterations < limit:
        iterations += 1
        sample = rng.choice(n, 4, replace=False)
        if _three_collinear(src[sample]) or _three_collinear(dst[sample]):
            continue
        H, _, ok = dlt_homography(src[sample], dst[sample])
        if not ok:
            continue
        residuals = symmetric_transfer_error(H, src, dst)
        mask = residuals < config.threshold
        count, cost = int(mask.sum()), float(residuals[mask].sum())
        if count > best_count or (count == best_count and cost < best_cost):
            best_H, best_mask, best_residuals = H, mask, residuals
            best_count, best_cost = count, cost
            probability = (count / n) ** 4
            if probability >= 1:
                limit = iterations
            elif probability > 0:
                required = np.ceil(np.log1p(-config.confidence) / np.log1p(-probability))
                limit = min(limit, max(iterations, int(required)))
    if best_count >= 4:
        H, _, ok = dlt_homography(src[best_mask], dst[best_mask])
        if ok:
            best_H = H
            best_residuals = symmetric_transfer_error(H, src, dst)
            best_mask = best_residuals < config.threshold
    return best_H, best_mask, best_residuals, iterations, bool(
        best_H is not None and best_mask.sum() >= max(4, config.min_inliers))


def warp_artwork(artwork: np.ndarray, H: np.ndarray, frame_shape: Tuple[int, ...]
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Destination-to-source warp of ``artwork`` into a frame of shape ``frame_shape``.

    ``H`` maps artwork pixel coordinates to frame pixel coordinates. For every
    frame pixel inside the projected artwork quadrilateral, map it back through
    H^{-1} and sample the artwork bilinearly. Returns (warped H x W x C float32,
    coverage H x W bool); pixels that map outside the artwork receive no
    coverage and are zero in ``warped``.
    """
    # YOUR CODE HERE
    artwork = np.asarray(artwork)
    h, w = frame_shape[:2]
    ah, aw = artwork.shape[:2]
    warped = np.zeros((h, w) + artwork.shape[2:], dtype=np.float32)
    coverage = np.zeros((h, w), dtype=bool)
    yy, xx = np.indices((h, w))
    try:
        source = transfer_points(np.linalg.inv(H), np.column_stack((xx.ravel(), yy.ravel())))
    except np.linalg.LinAlgError:
        return warped, coverage
    sx, sy = source[:, 0].reshape(h, w), source[:, 1].reshape(h, w)
    eps = 1e-8
    coverage = (np.isfinite(sx) & np.isfinite(sy) & (sx >= -eps) & (sx <= aw - 1 + eps)
                & (sy >= -eps) & (sy <= ah - 1 + eps))
    warped[coverage] = bilinear_sample(artwork, sy[coverage], sx[coverage])
    return warped, coverage


__all__ = ["bilinear_sample", "rectangle_to_rectangle", "normalise_points", "dlt_homography", "transfer_points",
           "symmetric_transfer_error", "ransac_homography", "warp_artwork"]
