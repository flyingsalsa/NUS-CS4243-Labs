"""Reproducible supplementary diagnostics; never changes frozen settings. [AI-CODE] [AI-DESIGN]

Run from code/: python student_files/additional_evidence.py
All plots, raw measurements and experiment metadata stay beside this script.
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from overlaylab import data, metrics, supplied as sp
from overlaylab.config import PipelineConfig
from overlaylab.localisation import localise_board, artwork_to_frame
from overlaylab.geometry import warp_artwork
from overlaylab.compositing import occlusion_mask, composite, _difference_components

OUT = ROOT / "student_files" / "additional_evidence"
OUT.mkdir(exist_ok=True)
CONFIG = PipelineConfig.from_json((ROOT / "outputs/thresholds.json").read_text())


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def opening():
    cfg = replace(CONFIG.occlusion, min_area=round(CONFIG.occlusion.min_area / 4))
    reference = data.load_reference()
    reference = sp.resize_width(reference, round(reference.shape[1] / 2))
    masks = data.load_gt_masks("validation", width=640)
    rows = []
    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    for row_index, (f, frame) in enumerate((i, fr) for i, fr in sp.iter_video(data.COMMON_CLIP, width=640) if i in masks):
        H = localise_board(reference, frame, CONFIG.localisation)[0]
        if H is None:
            raise RuntimeError(f"Validation localisation failed at {f}")
        refwarp, cov = warp_artwork(reference, H, frame.shape)
        axes[row_index, 0].imshow(frame)
        axes[row_index, 0].set_title(f"Common validation {f}, 640 px")
        axes[row_index, 1].imshow(masks[f], cmap="gray")
        axes[row_index, 1].set_title("Ground truth")
        for col, radius in enumerate((0, cfg.open_radius), 2):
            mask, difference, components = occlusion_mask(frame, H, reference, replace(cfg, open_radius=radius))
            _, alpha = composite(frame, refwarp, cov, mask, CONFIG.composite)
            row = {"frame": f, "open_radius": radius, "iou": metrics.mask_iou(mask, masks[f]),
                   "f1": metrics.mask_f1(mask, masks[f]), "boundary_f1": metrics.boundary_f1(mask, masks[f]),
                   "alpha_leakage": metrics.leakage(alpha, masks[f]), "false_positive_pixels": int((mask & ~masks[f]).sum()),
                   "false_negative_pixels": int((~mask & masks[f]).sum()), "components": components.tolist()}
            rows.append(row)
            axes[row_index, col].imshow(mask, cmap="gray")
            axes[row_index, col].set_title(f"radius {radius}; IoU {row['iou']:.3f}")
    for ax in axes.flat:
        ax.axis("off")
    save(fig, "opening_validation.png")
    # Controlled vertical strips: weak body connected to a strong cap, long enough
    # that minimum-area removal is distinguishable from morphological erosion.
    thin = []
    fig, axes = plt.subplots(2, 8, figsize=(13, 5))
    for width in range(1, 9):
        diff = np.zeros((120, 40), dtype=np.float32)
        diff[10:110, 18:18 + width] = .18
        diff[10:25, 18:18 + width] = .5
        for row_index, radius in enumerate((0, CONFIG.occlusion.open_radius)):
            candidate = replace(CONFIG.occlusion, open_radius=radius)
            labels, stats = _difference_components(diff, np.ones(diff.shape, bool), candidate)
            keep = np.flatnonzero(stats[:, 4] >= candidate.min_area)
            mask = np.isin(labels, keep[keep != 0])
            thin.append({"width_px": width, "radius": radius, "remaining_pixels": int(mask.sum())})
            axes[row_index, width - 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
            axes[row_index, width - 1].set_title(f"w={width}, r={radius}\n{mask.sum()} px")
            axes[row_index, width - 1].axis("off")
    fig.suptitle("Synthetic 100-pixel-long weak strips with strong caps; min_area=150")
    save(fig, "thin_occluder.png")
    (OUT / "opening.json").write_text(json.dumps({"source_config": CONFIG.thresholds(), "validation_width": 640,
        "validation_min_area": cfg.min_area, "validation": rows, "synthetic": thin}, indent=2))


def boundaries():
    rows = []
    for cid, frame_id in (("common_clip", 144), ("coke_unconstrained", 304)):
        video, refpath = data.clip_paths(cid)
        reference = sp.read_image(refpath)
        reference = sp.resize_width(reference, round(reference.shape[1] / 2))
        frame = next(sp.iter_video(video, width=640, start=frame_id, stop=frame_id + 1))[1]
        truth = data.load_gt_masks("test", width=640)[frame_id] if cid == "common_clip" else sp.resize_width(data.load_diagnostic_masks(cid)[frame_id].astype(np.uint8) * 255, 640) > 127
        H = localise_board(reference, frame, CONFIG.localisation)[0]
        if H is None:
            raise RuntimeError(f"No estimated homography: {cid} {frame_id}; no GT fallback permitted here")
        lc = replace(CONFIG.localisation, warp=replace(CONFIG.localisation.warp, reference_shape=reference.shape[:2]))
        art = data.load_artwork()
        warped, cov = warp_artwork(art, artwork_to_frame(art, H, lc), frame.shape)
        mask, diff, components = occlusion_mask(frame, H, reference, replace(CONFIG.occlusion, min_area=round(CONFIG.occlusion.min_area / 4)))
        out, alpha = composite(frame, warped, cov, mask, CONFIG.composite)
        base, _ = composite(frame, warped, cov, np.zeros_like(mask), CONFIG.composite)
        ys, xs = np.where(truth | mask)
        crop = np.s_[max(0, ys.min()-12):min(frame.shape[0], ys.max()+13), max(0, xs.min()-12):min(frame.shape[1], xs.max()+13)]
        fig, axes = plt.subplots(2, 3, figsize=(11, 8))
        for ax, arr, title in zip(axes.flat, [frame, diff, mask.astype(int)+2*truth, alpha, base, out],
                                  ["Original", "Compensated difference", "Pred + 2*GT (3=agreement)", "Alpha", "No occlusion", "With occlusion"]):
            ax.imshow(arr[crop], cmap="viridis" if arr.ndim == 2 else None)
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle(f"{cid}, test/stress frame {frame_id}, independent H, 640-pixel preview")
        save(fig, f"boundary_{cid}.png")
        rows.append({"clip": cid, "frame": frame_id, "iou": metrics.mask_iou(mask, truth), "boundary_f1": metrics.boundary_f1(mask, truth), "alpha_leakage": metrics.leakage(alpha, truth), "components": components.tolist()})
    (OUT / "boundaries.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    opening()
    boundaries()
    print("Saved opening, thin-strip and boundary diagnostics to", OUT)
