"""Plot frozen full-video tracking statistics and diagnostic flow. [AI-CODE] [AI-DESIGN]

Run after final videos: python student_files/tracking_evidence.py
No parameters are selected or changed by this script.
"""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from overlaylab import data, metrics, supplied as sp
from overlaylab.config import PipelineConfig
from overlaylab.temporal import lk_flow

OUT = ROOT / "student_files" / "tracking_evidence"
OUT.mkdir(exist_ok=True)
CONFIG = PipelineConfig.from_json((ROOT / "outputs/thresholds.json").read_text())


def read(cid, variant="flow_assisted"):
    return json.loads((ROOT / "outputs/videos" / f"{cid}_{variant}_stats.json").read_text())


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def timeline(cid):
    result = read(cid)
    gt = data.load_gt_corners() if cid == "common_clip" else data.load_diagnostic_corners(cid)
    keys = ["inlier_count", "inlier_ratio", "corner_displacement", "mean_flow", "brightness_change"]
    limits = [12, .4, 60, 25, .15]
    fig, axes = plt.subplots(7, 1, figsize=(15, 13), sharex=True)
    for ax, key, limit in zip(axes, keys, limits):
        values = [s[key] for s in result["stats"]]
        ax.plot(values, lw=.8)
        ax.axhline(limit, color="red", ls="--")
        ax.set_ylabel(key.replace("_", " "), fontsize=8)
        ax.grid(alpha=.2)
    errors = {f: metrics.corner_error(np.array(result["corners"][f]), q) for f, q in gt.items()}
    axes[5].plot(list(errors), [100 * e for e in errors.values()], "o-", ms=3)
    axes[5].axhline(2, color="red", ls="--")
    axes[5].set_ylabel("corner error %")
    flags = np.array([{"OK": 0, "REINIT": 1, "HIDE": 2}[x[0]] for x in result["flags"]])
    axes[6].step(range(len(flags)), flags, where="mid")
    axes[6].set_yticks([0, 1, 2], ["OK", "REINIT", "HIDE"])
    axes[6].scatter(result["reinit_frames"], np.full(len(result["reinit_frames"]), -.2), s=4, c="black", label="actual localiser call")
    axes[6].legend(loc="upper right")
    axes[6].set_xlabel("Source frame index (30 fps)")
    fig.suptitle(f"{cid}: frozen flow-assisted, native 1280 x 720; audit errors use all retained annotations")
    save(fig, f"{cid}_timeline.png")
    runs = []
    for f, flag in enumerate(result["flags"]):
        if not runs or runs[-1]["flag"] != flag:
            runs.append({"start": f, "end": f, "flag": flag})
        else:
            runs[-1]["end"] = f
    return {"runs": runs, "reinit_frames": result["reinit_frames"], "errors": errors,
            "timing": result["timing"], "stats": result["stats"], "flags": result["flags"]}


def common_comparison():
    gt = data.load_gt_corners()
    variants = {v: read("common_clip", v) for v in ("independent", "flow_assisted")}
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    results = {}
    for variant, result in variants.items():
        corners = np.array(result["corners"])
        axes[0].plot(corners[:, 0, 0], label=variant, lw=.8)
        errors = {f: metrics.corner_error(corners[f], q) for f, q in gt.items()}
        axes[1].plot(list(errors), [e * 100 for e in errors.values()], "o-", label=variant)
        age = np.arange(len(corners)) - np.maximum.accumulate(np.where(np.isin(np.arange(len(corners)), result["reinit_frames"]), np.arange(len(corners)), 0))
        axes[2].plot(age, label=variant, lw=.8)
        results[variant] = {"timing": result["timing"], "window_jitter": []}
        for start in range(0, len(corners)-31, 16):
            pred = {f: corners[f] for f in range(start, start+32) if np.isfinite(corners[f]).all()}
            results[variant]["window_jitter"].append({"start": start, "end": start+31,
                "finite_frames": len(pred), "jitter": metrics.jitter(pred, sorted(gt), [gt[f] for f in sorted(gt)])})
    axes[0].set_ylabel("Top-left x (px)")
    axes[1].set_ylabel("Audit corner error (%)")
    axes[1].axhline(2, color="red", ls="--")
    axes[2].set_ylabel("Frames since localiser call")
    axes[2].set_xlabel("Source frame index")
    for ax in axes:
        ax.legend(); ax.grid(alpha=.2)
    fig.suptitle("Common clip; both frozen variants, native 1280 x 720; all retained annotations")
    save(fig, "common_comparison.png")
    window = np.arange(16, 48)
    truth = metrics.interpolate_corners(sorted(gt), [gt[f] for f in sorted(gt)], window)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for variant, result in variants.items():
        corners = np.array(result["corners"])[window]
        axes[0].plot(window, corners[:, 0, 0] - truth[:, 0, 0], label=variant)
        delta = np.mean(np.linalg.norm(np.diff(corners, axis=0) - np.diff(truth, axis=0), axis=2), axis=1)
        axes[1].plot(window[1:], delta, label=variant)
    axes[0].set_ylabel("Top-left x minus interpolated GT (px)")
    axes[1].set_ylabel("Motion-subtracted corner step (px)")
    axes[1].set_xlabel("Source frame")
    for ax in axes:
        ax.legend(); ax.grid(alpha=.2)
    fig.suptitle("Common frames 16–47: local stabilisation example, native 1280 x 720")
    save(fig, "common_stable_window.png")
    return results


def blur():
    result = read("coke_blur")
    candidates = []
    for f, record in enumerate(result["flow"]):
        if not record:
            continue
        eig = np.array(record["eigenvalues"])
        residual = np.array(record["residuals"])
        small = eig[:, 1] < CONFIG.track.flow.min_eigenvalue
        high = (~small) & (residual > CONFIG.track.flow.max_residual) & np.isfinite(residual)
        candidates.append((int(small.sum()), int(high.sum()), f))
    if not candidates:
        raise RuntimeError("No flow records retained in blur run")
    choices = sorted(set([max(candidates, key=lambda v:v[0])[2], max(candidates, key=lambda v:v[1])[2]]))
    video, _ = data.clip_paths("coke_blur")
    summary = {"selected_frames": choices, "candidate_counts": candidates, "frames": []}
    for f in choices:
        r = result["flow"][f]
        points, eig, residual, valid = [np.array(r[key]) for key in ("points", "eigenvalues", "residuals", "valid")]
        small = eig[:, 1] < CONFIG.track.flow.min_eigenvalue
        high = (~small) & (residual > CONFIG.track.flow.max_residual) & np.isfinite(residual)
        other = ~valid & ~small & ~high
        frames = dict(sp.iter_video(video, start=f-1, stop=f+1))
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        axes[0].imshow(frames[f-1]); axes[1].imshow(frames[f])
        for mask, color, label in [(valid, "lime", "valid"), (small, "orange", "low lambda2"), (high, "magenta", "high residual, conditioned"), (other, "cyan", "other invalid")]:
            axes[0].scatter(points[mask, 0], points[mask, 1], s=20, c=color, label=f"{label} ({mask.sum()})")
            axes[2].scatter(eig[mask, 0], eig[mask, 1], s=15, c=color)
        axes[0].legend(fontsize=7)
        axes[0].set_title(f"Source {f-1}: tracked point locations")
        axes[1].set_title(f"Target {f}")
        axes[2].set(xscale="log", yscale="log", xlabel="lambda1 (larger)", ylabel="lambda2 (smaller)")
        axes[2].axhline(CONFIG.track.flow.min_eigenvalue, color="red", ls="--")
        axes[2].set_title("Window tensor eigenvalues")
        fig.suptitle("Blur stress clip, frozen flow-assisted, native 1280 x 720")
        save(fig, f"blur_flow_{f:06d}.png")
        summary["frames"].append({"frame": f, "low_lambda2": int(small.sum()), "conditioned_high_residual": int(high.sum()),
                                  "other_invalid": int(other.sum()), "valid": int(valid.sum()), "record": r})
    return summary


def contact(cid):
    video, _ = data.clip_paths(cid)
    fig, axes = plt.subplots(4, 8, figsize=(19, 7))
    for ax, (f, frame) in zip(axes.flat, sp.iter_video(video, step=16, width=320)):
        ax.imshow(frame); ax.set_title(str(f), fontsize=8); ax.axis("off")
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(f"{cid}, every 16th source frame; display width 320 px")
    save(fig, f"{cid}_contact.png")


def aperture_probe():
    """Supplement actual Harris tracks with a fixed grid; never feed it to tracking."""
    source_index = 2
    result = read("coke_blur")
    video, _ = data.clip_paths("coke_blur")
    frames = dict(sp.iter_video(video, start=source_index, stop=source_index + 2))
    source, target = frames[source_index], frames[source_index + 1]
    board = sp.quad_mask(source.shape[:2], np.array(result["corners"][source_index]))
    yy, xx = np.mgrid[12:source.shape[0]-12:12, 12:source.shape[1]-12:12]
    inside = board[yy, xx]
    points = np.column_stack([xx[inside], yy[inside]]).astype(float)
    vectors, eig, residual, valid = lk_flow(sp.to_gray(source), sp.to_gray(target), points, CONFIG.track.flow)
    threshold = CONFIG.track.flow.min_eigenvalue
    edge = (eig[:, 1] < threshold) & (eig[:, 0] >= threshold)
    flat = eig[:, 0] < threshold
    high = (eig[:, 1] >= threshold) & (residual > CONFIG.track.flow.max_residual)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].imshow(source)
    for mask, color, label in [(edge, "orange", "one weak direction"), (flat, "cyan", "both weak"), (high, "magenta", "conditioned / high residual")]:
        axes[0].scatter(points[mask, 0], points[mask, 1], s=10, c=color, label=f"{label}: {mask.sum()}")
        axes[1].scatter(eig[mask, 0], eig[mask, 1], s=10, c=color)
    q = np.array(result["corners"][source_index])
    axes[0].set_xlim(q[:, 0].min()-20, q[:, 0].max()+20)
    axes[0].set_ylim(q[:, 1].max()+20, q[:, 1].min()-20)
    axes[0].legend(fontsize=7)
    axes[1].set(xscale="log", yscale="log", xlabel="lambda1", ylabel="lambda2")
    axes[1].axhline(threshold, ls="--", c="red")
    axes[1].axvline(threshold, ls="--", c="red")
    fig.suptitle("Blur 2 to 3, native resolution: supplementary 12-pixel grid (NOT tracker input)")
    save(fig, "blur_aperture_probe.png")
    return {"source": 2, "target": 3, "spacing": 12, "points": points.tolist(), "eigenvalues": eig.tolist(),
            "residuals": residual.tolist(), "valid": valid.tolist(), "one_weak_direction": int(edge.sum()),
            "both_weak": int(flat.sum()), "conditioned_high_residual": int(high.sum()), "tracker_input": False}


if __name__ == "__main__":
    summary = {cid: timeline(cid) for cid in ("common_clip", "coke_blur", "coke_out_of_view", "coke_unconstrained")}
    summary["common_comparison"] = common_comparison()
    summary["blur_diagnostic"] = blur()
    summary["aperture_probe"] = aperture_probe()
    for cid in ("coke_unconstrained", "coke_out_of_view"):
        contact(cid)
    (OUT / "tracking.json").write_text(json.dumps(summary, indent=2))
    print("Saved native-resolution timelines, common comparison and blur eigenvalue evidence.")
