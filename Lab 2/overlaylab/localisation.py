"""Stage 1 Localise: board localisation from the reference image and single-frame overlay.

Implement the functions marked ``YOUR CODE HERE`` using your keypoint and
geometry branches. Optional feature families and alternative settings must be
controlled only through ``LocalisationConfig``.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .config import LocalisationConfig
from .supplied import to_gray, rect_corners
from .keypoints import detect_corners, describe_keypoints, match_descriptors
from .geometry import ransac_homography, transfer_points, warp_artwork, rectangle_to_rectangle
from . import metrics


def localise_board(reference: np.ndarray, frame: np.ndarray, config: LocalisationConfig
                   ) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Find the board in ``frame`` from the rectified ``reference`` image alone.

    Detect and describe keypoints in both images, match with the ratio test,
    estimate the reference-to-frame homography with RANSAC, and project the
    four reference corners (TL, TR, BR, BL) into the frame.

    Returns (H, corners 4 x 2, matches M x 2 (reference index, frame index),
    inlier_mask M, diagnostics). ``diagnostics`` must contain exactly these keys
    (the notebooks and the grading harness read them): ``n_keypoints_ref``,
    ``n_keypoints_frame``, ``n_matches``, ``n_inliers``, ``inlier_ratio``
    (inliers / ratio-test matches), ``median_residual``, ``iterations``,
    ``status``, ``residuals`` (the per-match symmetric transfer errors e_i from
    RANSAC, length M), ``keypoints_ref`` and ``keypoints_frame`` (the arrays the
    match indices refer to). ``median_residual`` is the median over inliers of
    sqrt(e_i / 2), i.e. the RMS one-way transfer error in pixels.
    When localisation fails, ``H`` is None, ``corners`` is NaN and ``status``
    is False.
    """
    # YOUR CODE HERE
    ref_gray, frame_gray = to_gray(reference), to_gray(frame)
    ref_points, _ = detect_corners(ref_gray, config.detector)
    frame_points, _ = detect_corners(frame_gray, config.detector)
    ref_desc, ref_kept = describe_keypoints(ref_gray, ref_points, config.descriptor)
    frame_desc, frame_kept = describe_keypoints(frame_gray, frame_points, config.descriptor)
    ref_points, frame_points = ref_points[ref_kept], frame_points[frame_kept]
    matches, _, _ = match_descriptors(ref_desc, frame_desc, config.matcher)
    H, inliers, residuals, iterations, status = ransac_homography(
        ref_points[matches[:, 0]], frame_points[matches[:, 1]], config.ransac)
    corners = np.full((4, 2), np.nan)
    if status:
        corners = transfer_points(H, rect_corners(reference.shape[1], reference.shape[0]))
        status = bool(np.isfinite(corners).all())
    if not status:
        H, corners = None, np.full((4, 2), np.nan)
    diagnostics = {
        "n_keypoints_ref": len(ref_points), "n_keypoints_frame": len(frame_points),
        "n_matches": len(matches), "n_inliers": int(inliers.sum()),
        "inlier_ratio": float(inliers.mean()) if len(inliers) else 0.0,
        "median_residual": float(np.median(np.sqrt(residuals[inliers] / 2))) if inliers.any() else float("inf"),
        "iterations": iterations, "status": status, "residuals": residuals,
        "keypoints_ref": ref_points, "keypoints_frame": frame_points,
    }
    return H, corners, matches, inliers, diagnostics


def artwork_to_frame(artwork: np.ndarray, H: np.ndarray, config: LocalisationConfig) -> np.ndarray:
    """Compose the artwork rectangle with the board-to-frame homography.

    The artwork is placed by mapping its own pixel rectangle onto the
    reference board rectangle (``config.warp.reference_shape``; the artwork's
    own shape when unset) and then applying ``H``. Document in your report how
    the two transformations compose.
    """
    # YOUR CODE HERE
    ah, aw = artwork.shape[:2]
    rh, rw = config.warp.reference_shape or (ah, aw)
    return np.asarray(H) @ rectangle_to_rectangle((aw, ah), (rw, rh))


def overlay_frame(frame: np.ndarray, artwork: np.ndarray, H: np.ndarray,
                  config: LocalisationConfig) -> np.ndarray:
    """Single overlaid frame without occlusion handling.

    Warp the artwork with the composed homography (``artwork_to_frame``) and
    paint it where the coverage mask is set. Returns an RGB float32 frame.
    """
    # YOUR CODE HERE
    result = np.asarray(frame, dtype=np.float32).copy()
    if frame.dtype == np.uint8:
        result /= 255.0
    art = np.asarray(artwork, dtype=np.float32)
    if artwork.dtype == np.uint8:
        art = art / 255.0
    warped, coverage = warp_artwork(art, artwork_to_frame(art, H, config), result.shape)
    result[coverage] = warped[coverage]
    return result


def evaluate_audit_frames(reference: np.ndarray, frames: Dict[int, np.ndarray],
                          gt_corners: Dict[int, np.ndarray], config: LocalisationConfig) -> Dict:
    """Run ``localise_board`` on annotated frames and summarise with ``metrics``.

    ``frames`` and ``gt_corners`` map frame index -> image / 4 x 2 corners.
    Returns a dict with ``per_frame`` (index -> dict of error, n_matches,
    n_inliers, inlier_ratio, median_residual, status, corners) and ``summary``
    from ``metrics.summarise_errors`` using ``config.success_threshold``.
    """
    # YOUR CODE HERE
    per_frame = {}
    for index in sorted(gt_corners):
        if index not in frames:
            continue
        _, corners, _, _, diag = localise_board(reference, frames[index], config)
        row = {key: diag[key] for key in ("n_matches", "n_inliers", "inlier_ratio", "median_residual", "status")}
        row.update(error=metrics.corner_error(corners, gt_corners[index]), corners=corners)
        per_frame[index] = row
    return {"per_frame": per_frame,
            "summary": metrics.summarise_errors((row["error"] for row in per_frame.values()), config.success_threshold)}


__all__ = ["localise_board", "artwork_to_frame", "overlay_frame", "evaluate_audit_frames"]
