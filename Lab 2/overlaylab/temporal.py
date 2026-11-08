"""Stage 2 Track: optical flow, temporal variants, and failure monitoring.

Implement the functions marked ``YOUR CODE HERE``. ``cv2.calcOpticalFlowPyrLK``
and ``cv2.goodFeaturesToTrack`` may appear only in the labelled comparison
cell of Notebook 2. Grayscale images are float32 in [0, 1].
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import FlowConfig, MonitorConfig, TrackConfig, LocalisationConfig
from .supplied import to_gray, sobel_gradients, gaussian_pyramid, quad_mask, rect_corners
from .geometry import ransac_homography, transfer_points, bilinear_sample
from .localisation import localise_board

OK, REINIT, HIDE = "OK", "REINIT", "HIDE"


def lk_flow(prev_gray: np.ndarray, next_gray: np.ndarray, points: np.ndarray, config: FlowConfig
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Iterative single-level Lucas-Kanade at sparse points.

    For each point use a (2w+1) x (2w+1) window (``config.window``), the
    supplied Sobel gradients of ``prev_gray``, bilinear sampling of the warped
    window in ``next_gray`` at every iteration, and at most
    ``config.iterations`` updates (stop early when the update is below
    ``config.convergence``). The structure tensor is normalised by the number
    of window pixels so that ``config.min_eigenvalue`` is scale-free.

    Returns (displacements N x 2, eigenvalues N x 2 with lambda_1 >= lambda_2,
    residuals N (mean absolute brightness residual), valid N bool). A point is
    invalid when lambda_2 < config.min_eigenvalue, the residual exceeds
    config.max_residual, the update failed to converge, or the point leaves
    the image. Coarse-to-fine estimation over ``config.pyramid_levels`` is an
    optional extension.
    """
    # YOUR CODE HERE
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    n = len(points)
    vectors = np.zeros((n, 2), dtype=float)
    eigenvalues = np.zeros((n, 2), dtype=float)
    residuals = np.full(n, np.inf)
    valid = np.zeros(n, dtype=bool)
    if not n:
        return vectors, eigenvalues, residuals, valid
    previous = gaussian_pyramid(prev_gray, max(1, config.pyramid_levels))
    following = gaussian_pyramid(next_gray, max(1, config.pyramid_levels))
    offsets = np.arange(-config.window, config.window + 1)
    oy, ox = np.meshgrid(offsets, offsets, indexing="ij")
    ox, oy = ox.ravel(), oy.ravel()
    for level in range(len(previous) - 1, -1, -1):
        scale = 2 ** level
        p = points / scale
        displacement = vectors / scale
        a, b = previous[level], following[level]
        gx, gy, _, _ = sobel_gradients(a)
        x, y = p[:, 0, None] + ox, p[:, 1, None] + oy
        source_ok = (np.isfinite(p).all(axis=1) & (x.min(axis=1) >= 0)
                     & (x.max(axis=1) <= a.shape[1] - 1) & (y.min(axis=1) >= 0)
                     & (y.max(axis=1) <= a.shape[0] - 1))
        x, y = np.nan_to_num(x), np.nan_to_num(y)
        template = bilinear_sample(a, y, x)
        gradients = np.stack((bilinear_sample(gx, y, x), bilinear_sample(gy, y, x)), axis=-1)
        tensor = np.einsum("npi,npj->nij", gradients, gradients) / len(ox)
        eigenvalues = np.linalg.eigvalsh(tensor)[:, ::-1]
        threshold = config.min_eigenvalue if level == 0 else np.finfo(float).eps
        active = source_ok & (eigenvalues[:, 1] >= threshold) & (eigenvalues[:, 1] > 0)
        converged = np.zeros(n, dtype=bool)
        for _ in range(config.iterations):
            sx, sy = x + displacement[:, 0, None], y + displacement[:, 1, None]
            active &= ((sx.min(axis=1) >= 0) & (sx.max(axis=1) <= b.shape[1] - 1)
                       & (sy.min(axis=1) >= 0) & (sy.max(axis=1) <= b.shape[0] - 1))
            ids = np.flatnonzero(active & ~converged)
            if not len(ids):
                break
            difference = template[ids] - bilinear_sample(b, sy[ids], sx[ids])
            rhs = np.einsum("npi,np->ni", gradients[ids], difference) / len(ox)
            update = np.linalg.solve(tensor[ids], rhs[..., None])[..., 0]
            displacement[ids] += update
            converged[ids] = np.linalg.norm(update, axis=1) < config.convergence
        sx, sy = x + displacement[:, 0, None], y + displacement[:, 1, None]
        destination_ok = ((sx.min(axis=1) >= 0) & (sx.max(axis=1) <= b.shape[1] - 1)
                          & (sy.min(axis=1) >= 0) & (sy.max(axis=1) <= b.shape[0] - 1))
        residuals = np.mean(np.abs(template - bilinear_sample(b, sy, sx)), axis=1)
        residuals[~(source_ok & destination_ok)] = np.inf
        valid = active & converged & destination_ok & (residuals <= config.max_residual)
        if level:
            vectors[valid] = displacement[valid] * scale
        else:
            vectors = displacement
    return vectors, eigenvalues, residuals, valid


@dataclass
class TrackInit:
    """What ``track_board`` needs to (re)initialise from the reference image."""
    reference: np.ndarray
    localisation: LocalisationConfig


@dataclass
class TrackResult:
    homographies: List[Optional[np.ndarray]]
    corners: List[np.ndarray]                 # 4 x 2 per frame (NaN when lost)
    status: List[str]                          # "ok" or "lost"
    stats: List[Dict]                          # consumed by failure_monitor
    reinit_frames: List[int] = field(default_factory=list)
    flow: List[Dict] = field(default_factory=list)  # per-frame vectors, eigenvalues, validity


def _frame_stats(prev_corners, corners, gray_prev, gray, inlier_count, inlier_ratio,
                 median_residual, mean_flow, n_valid, status, n_matches=None):
    """Supplied helper: assemble the per-frame statistics dictionary (``n_matches`` on (re)initialised frames)."""
    disp = float(np.mean(np.linalg.norm(corners - prev_corners, axis=1))) if (
        prev_corners is not None and np.all(np.isfinite(corners)) and np.all(np.isfinite(prev_corners))) else float("nan")
    if gray_prev is not None and np.all(np.isfinite(corners)):
        m = quad_mask(gray.shape, corners)
        bright = float(np.mean(np.abs(gray[m] - gray_prev[m]))) if m.any() else float("nan")
    else:
        bright = float("nan")
    return {"inlier_count": int(inlier_count), "inlier_ratio": float(inlier_ratio),
            "median_residual": float(median_residual), "corner_displacement": disp,
            "mean_flow": float(mean_flow), "brightness_change": bright,
            "n_valid_points": int(n_valid), "status": status,
            "n_matches": (int(n_matches) if n_matches is not None else None)}


def track_board(frames: Sequence[np.ndarray], init: TrackInit, config: TrackConfig) -> TrackResult:
    """Track the board through a sequence with one of two variants.

    ``config.variant == "independent"``: run ``localise_board`` on every frame
    from the reference image.

    ``config.variant == "flow_assisted"``: localise frame 0, then propagate the
    inlier correspondences (or the corners, ``config.propagate``) to the next
    frame with ``lk_flow``, drop invalid points, re-estimate the homography
    with RANSAC between the reference points and the propagated points, and
    reinitialise from the reference image when fewer than
    ``config.reinit_min_points`` valid points remain, every
    ``config.reinit_every`` frames (0 disables), or when RANSAC fails.

    Optional corner propagation has four points only: use reinit_min_points <= 4,
    RANSAC min_inliers = 4 and monitor min_inlier_count <= 4 if selecting it.

    Both variants return a ``TrackResult`` with a homography, projected
    corners, a status flag and the statistics the failure monitor consumes
    for every frame (build them with the supplied ``_frame_stats``; use
    ``inlier_ratio`` = inliers / valid propagated points for the flow variant
    and ``median_residual`` = median over inliers of sqrt(e_i / 2) px). The
    flow-assisted variant also stores, for every frame after the first, a dict
    in ``TrackResult.flow[t]`` with keys ``points`` (tracked points N x 2),
    ``vectors`` (displacements N x 2), ``eigenvalues`` (N x 2), ``residuals``
    (N) and ``valid`` (N bool); frames that were (re)initialised store {}.
    """
    # YOUR CODE HERE
    if config.variant not in ("independent", "flow_assisted"):
        raise ValueError("Unknown tracking variant: " + config.variant)
    if config.propagate not in ("inliers", "corners"):
        raise ValueError("Unknown propagation mode: " + config.propagate)
    result = TrackResult([], [], [], [])
    ref_corners = rect_corners(init.reference.shape[1], init.reference.shape[0])
    ref_points, points = np.empty((0, 2)), np.empty((0, 2))
    previous_gray, previous_corners = None, None
    previous_H, last_init = None, 0
    for t, frame in enumerate(frames):
        gray = to_gray(frame)
        reinitialise = (config.variant == "independent" or previous_H is None
                        or len(points) < config.reinit_min_points
                        or (config.reinit_every > 0 and t - last_init >= config.reinit_every))
        flow_record, mean_flow, n_matches = {}, float("nan"), None
        H, corners = None, np.full((4, 2), np.nan)
        inlier_count, inlier_ratio, median_residual, n_valid = 0, 0.0, float("inf"), 0
        if not reinitialise:
            vectors, eigenvalues, residuals, valid = lk_flow(previous_gray, gray, points, config.flow)
            n_valid = int(valid.sum())
            mean_flow = float(np.mean(np.linalg.norm(vectors[valid], axis=1))) if n_valid else float("nan")
            flow_record = {"points": points.copy(), "vectors": vectors, "eigenvalues": eigenvalues,
                           "residuals": residuals, "valid": valid}
            moved = points[valid] + vectors[valid]
            source = ref_points[valid]
            reinitialise = n_valid < max(4, config.reinit_min_points)
            if not reinitialise:
                H, inliers, errors, _, ok = ransac_homography(source, moved, init.localisation.ransac)
                reinitialise = not ok
                if ok:
                    corners = transfer_points(H, ref_corners)
                    reinitialise = not np.isfinite(corners).all()
                    inlier_count = int(inliers.sum())
                    inlier_ratio = inlier_count / n_valid
                    median_residual = float(np.median(np.sqrt(errors[inliers] / 2)))
                    ref_points, points = source[inliers], moved[inliers]
                    if config.propagate == "corners":
                        ref_points, points = ref_corners.copy(), corners.copy()
        if reinitialise:
            H, corners, matches, inliers, diag = localise_board(init.reference, frame, init.localisation)
            result.reinit_frames.append(t)
            last_init, flow_record, mean_flow = t, {}, float("nan")
            n_matches = diag["n_matches"]
            inlier_count, inlier_ratio = diag["n_inliers"], diag["inlier_ratio"]
            median_residual, n_valid = diag["median_residual"], inlier_count
            if H is None:
                ref_points, points = np.empty((0, 2)), np.empty((0, 2))
            elif config.propagate == "corners":
                ref_points, points = ref_corners.copy(), corners.copy()
            else:
                ref_points = diag["keypoints_ref"][matches[inliers, 0]]
                points = diag["keypoints_frame"][matches[inliers, 1]]
        status = "ok" if H is not None else "lost"
        result.homographies.append(H)
        result.corners.append(corners)
        result.status.append(status)
        result.stats.append(_frame_stats(previous_corners, corners, previous_gray, gray,
                                        inlier_count, inlier_ratio, median_residual, mean_flow,
                                        n_valid, status, n_matches))
        result.flow.append(flow_record)
        previous_gray, previous_corners, previous_H = gray, corners, H
    return result


def failure_monitor(stats: Sequence[Dict], config: MonitorConfig) -> List[Tuple[str, str]]:
    """Decide OK / REINIT / HIDE for every frame from the per-frame statistics.

    ``stats`` is the list produced by ``track_board``: dictionaries with
    ``inlier_count``, ``inlier_ratio``, ``median_residual``,
    ``corner_displacement``, ``mean_flow``, ``brightness_change`` and
    ``status``. Returns a list of (flag, trigger) where ``trigger`` names the
    statistic (or "none") responsible for the decision. Because a homography
    is defined only up to scale, compare projected-corner displacement and
    reprojection error, never homography matrices.

    Required policy (the tests check it; the thresholds are yours to set):
    1. ``status != "ok"`` -> HIDE with trigger "no_model";
    2. ``inlier_count`` < min_inlier_count or ``inlier_ratio`` < min_inlier_ratio
       -> REINIT with trigger "inlier_count" / "inlier_ratio", escalating to
       HIDE once this has happened on ``hide_after_consecutive`` consecutive
       frames (frames under rule 1 count towards the run);
    3. otherwise the first of ``median_residual``, ``corner_displacement``,
       ``mean_flow``, ``brightness_change`` that exceeds its maximum -> REINIT
       with that statistic as trigger (NaN statistics never fire);
    4. otherwise OK with trigger "none"; an OK frame resets the run.
    """
    # YOUR CODE HERE
    result, run = [], 0
    maxima = (("median_residual", config.max_median_residual),
              ("corner_displacement", config.max_corner_displacement),
              ("mean_flow", config.max_mean_flow),
              ("brightness_change", config.max_brightness_change))
    for row in stats:
        if row["status"] != "ok":
            run += 1
            result.append((HIDE, "no_model"))
            continue
        trigger = ("inlier_count" if row["inlier_count"] < config.min_inlier_count else
                   "inlier_ratio" if row["inlier_ratio"] < config.min_inlier_ratio else None)
        if trigger is not None:
            run += 1
            result.append((HIDE if run >= config.hide_after_consecutive else REINIT, trigger))
            continue
        trigger = next((key for key, limit in maxima if row.get(key, float("nan")) > limit), None)
        if trigger is not None:
            run += 1
            result.append((REINIT, trigger))
        else:
            run = 0
            result.append((OK, "none"))
    return result


__all__ = ["OK", "REINIT", "HIDE", "lk_flow", "TrackInit", "TrackResult", "track_board", "failure_monitor"]
