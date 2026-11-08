"""Question 1 diagnostics at Notebook 1's 640-pixel preview resolution."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from overlaylab import data, supplied as sp, visualize as viz
from overlaylab.config import PipelineConfig
from overlaylab.keypoints import detect_corners, describe_keypoints, match_descriptors

config = PipelineConfig.from_json((ROOT / "outputs/thresholds.json").read_text())
cfg = config.localisation
video, ref_path = data.clip_paths("coke_scale")
info = sp.video_info(video)
factor = 640 / info["width"]
ref_full = sp.read_image(ref_path)
reference = sp.resize_width(ref_full, round(ref_full.shape[1] * factor))

def describe(image):
    gray = sp.to_gray(image)
    points, _ = detect_corners(gray, cfg.detector)
    desc, keep = describe_keypoints(gray, points, cfg.descriptor)
    return points[keep], desc

kr, dr = describe(reference)
gt = data.load_diagnostic_corners("coke_scale")
rows, images, features = [], {}, {}
for index, frame in sp.iter_video(video, step=data.GT_STRIDE, width=640):
    if index not in gt:
        continue
    k, d = describe(frame)
    pairs, _, _ = match_descriptors(dr, d, cfg.matcher)
    rows.append(dict(frame=index, matches=len(pairs), reference_descriptors=len(dr),
                     rate=len(pairs) / max(len(dr), 1)))
    images[index], features[index] = frame, (k, pairs)

best, worst = max(rows, key=lambda r: r["rate"]), min(rows, key=lambda r: r["rate"])
# Execute the baseline definition directly from Notebook 1's labelled SIFT cell.
notebook = json.loads((ROOT / "notebooks/01_keypoints_and_geometry.ipynb").read_text(encoding="utf-8"))
baseline = next("".join(c["source"]) for c in notebook["cells"]
                if c["cell_type"] == "code" and "sift = cv2.SIFT_create()" in "".join(c["source"]))
import cv2
exec(baseline.split("worst = rate_indices")[0])
for row in (best, worst):
    row["sift_rate"] = sift_rate(reference, images[row["frame"]])

out = ROOT / "student_files/question1"
out.mkdir(parents=True, exist_ok=True)
payload = dict(clip="coke_scale", subset="stress/test annotations", width=640,
               reference_shape=list(reference.shape), best=best, worst=worst, per_frame=rows)
(out / "matching.json").write_text(json.dumps(payload, indent=2))
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot([r["frame"] for r in rows], [r["rate"] for r in rows], "o-")
ax.set(xlabel="coke_scale frame", ylabel="matches / reference descriptors",
       title="Frozen custom descriptor, 640-pixel preview")
fig.tight_layout(); fig.savefig(out / "matching_rates.png"); plt.close(fig)
for name, row in (("best", best), ("worst", worst)):
    k, pairs = features[row["frame"]]
    panel = viz.draw_matches(reference, images[row["frame"]], kr, k, pairs)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(panel); ax.axis("off")
    ax.set_title(f"coke_scale reference vs frame {row['frame']}; 640 px; ratio matches {row['matches']}/{len(dr)}")
    fig.tight_layout(); fig.savefig(out / f"{name}_matches.png"); plt.close(fig)
print(json.dumps({k: v for k, v in payload.items() if k != "per_frame"}, indent=2))
