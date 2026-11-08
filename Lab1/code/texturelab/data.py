"""Dataset manifests and leakage-safe split validation."""

from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path
from .supplied import load_image
import numpy as np
from PIL import Image

MATERIALS = ("carpet", "grid", "leather", "tile", "wood")
DTD_ATTRIBUTES = ("banded", "blotchy", "braided", "bumpy", "cracked", "fibrous", "grid",
                  "marbled", "pitted", "porous", "stained", "striped", "woven")


@dataclass(frozen=True)
class Record:
    path: str
    split: str
    material: str = ""
    defect_type: str = "good"
    mask_path: str = ""
    attributes: tuple[str, ...] = ()


def read_manifest(path: str | Path) -> list[Record]:
    root = Path(path).resolve().parent; records = []
    with open(path, newline="", encoding="utf8") as handle:
        for row in csv.DictReader(handle):
            resolve = lambda value: str((root / value).resolve()) if value else ""
            records.append(Record(resolve(row["path"]), row["split"], row.get("material", ""),
                                  row.get("defect_type", "good"), resolve(row.get("mask_path", "")),
                                  tuple(filter(None, row.get("attributes", "").split(";")))))
    return records


def validate_manifest(records: list[Record], require_files: bool = True) -> None:
    seen = set()
    for record in records:
        if record.path in seen: raise ValueError(f"duplicate image: {record.path}")
        seen.add(record.path)
        if record.split not in {"train", "validation", "test", "personal"}: raise ValueError("invalid split")
        if record.material and record.material not in MATERIALS: raise ValueError(f"invalid material {record.material}")
        if record.defect_type != "good" and not record.mask_path: raise ValueError(f"defect has no mask: {record.path}")
        if record.defect_type == "good" and record.mask_path: raise ValueError(f"good image must not have a mask: {record.path}")
        if require_files and (not Path(record.path).is_file() or record.mask_path and not Path(record.mask_path).is_file()):
            raise FileNotFoundError(record.path)


def load_records(records: list[Record]):
    return [(record, load_image(record.path)) for record in records]


def load_mask(path: str | Path, size: tuple[int, int] | None = None) -> np.ndarray:
    """Load any non-zero mask value as foreground."""
    image = Image.open(path).convert("L")
    if size is not None:
        image = image.resize(size, Image.Resampling.NEAREST)
    return np.asarray(image) > 0


def validate_attribute_support(records: list[Record], attributes=DTD_ATTRIBUTES, minimum: int = 1) -> None:
    """Ensure each chosen DTD term has positives and negatives in every official split."""
    for split in ("train", "validation", "test"):
        rows = [r for r in records if r.split == split]
        for attribute in attributes:
            positives = sum(attribute in r.attributes for r in rows)
            negatives = len(rows) - positives
            if positives < minimum or negatives < minimum:
                raise ValueError(f"{attribute!r} lacks support in {split}: +{positives}/-{negatives}")
