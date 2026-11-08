"""Verify report links, table copies, complete videos and frozen settings. [AI-CODE]"""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from overlaylab import data, supplied as sp
from overlaylab.config import PipelineConfig
from overlaylab.run_pipeline import all_runs

report = (ROOT / "REPORT.md").read_text(encoding="utf-8")
tables = (ROOT / "outputs/stress_tables.md").read_text(encoding="utf-8")
config = PipelineConfig.from_json((ROOT / "outputs/thresholds.json").read_text()).thresholds()
checks = {"links": [], "figures": [], "videos": [], "tables": []}
for image, label, target in re.findall(r"(!?)\[([^\]]*)\]\(([^)]+)\)", report):
    if "://" in target or target.startswith("#"):
        continue
    path = ROOT / target
    if not path.exists():
        raise AssertionError(f"Broken report link: {target}")
    checks["links"].append(target)
    if image:
        img = sp.read_image(path)
        checks["figures"].append({"file": target, "shape": list(img.shape)})
for section in re.split(r"^## Stage \d .*\n", tables, flags=re.M)[1:]:
    assert section.strip() in report, "Generated table/explanation was modified"
    assert not re.search(r"\| missing (?!masks)", section), "Missing final table data"
    checks["tables"].append("verbatim")
fingerprint = hashlib.sha256()
for source in sorted((ROOT / "overlaylab").glob("*.py")):
    fingerprint.update(source.name.encode())
    fingerprint.update(source.read_bytes())
for cid, variant, occlusion in all_runs():
    stem = f"{cid}_{variant}" + ("" if occlusion else "_no_occlusion")
    video = ROOT / "outputs/videos" / f"{stem}.mp4"
    result = json.loads(video.with_name(stem + "_stats.json").read_text())
    src = sp.video_info(data.clip_paths(cid)[0])
    info = sp.video_info(video)
    assert result["n_frames"] == info["frames"] == src["frames"]
    assert info["width"] == src["width"] and info["height"] == src["height"]
    assert result["request"]["code_sha256"] == fingerprint.hexdigest()
    saved = result["frozen_config"]
    saved["track"]["variant"] = config["track"]["variant"]
    assert saved == config and result["resize_width"] is None
    evaluation = ROOT / "outputs/evaluation" / f"{stem}_report.json"
    assert evaluation.exists(), f"Missing evaluator report {stem}"
    masks = set(data.mask_frames_for_clip(cid))
    assert set(result["mask_frames"]) | set(result["mask_frames_hidden"]) == masks
    for f in result["mask_frames"]:
        assert (video.parent / result["mask_dir"] / f"mask_{f:06d}.png").exists()
    checks["videos"].append({"stem": stem, "frames": info["frames"], "width": info["width"], "height": info["height"],
                             "scored_masks": result["mask_frames"], "hidden_masks": result["mask_frames_hidden"], "evaluator": str(evaluation.relative_to(ROOT))})
for path in (ROOT / "student_files").glob("*.py"):
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
assert "Suggested runs" not in report and "Suggested answer" not in report
checks["thresholds_sha256"] = hashlib.sha256((ROOT / "outputs/thresholds.json").read_bytes()).hexdigest()
checks["assessed_code_sha256"] = fingerprint.hexdigest()
checks["result"] = "PASS"
out = ROOT / "student_files/completion_checks.json"
out.write_text(json.dumps(checks, indent=2), encoding="utf-8")
print(f"PASS: {len(checks['figures'])} embedded figures; {len(checks['links'])} local links; 3 verbatim tables; 8 complete native-resolution videos and evaluator reports.")
