"""Deterministic toy textures for tests and notebook dry-runs (not report data)."""

import numpy as np


def texture(kind: str, size: int = 64, seed: int = 0, defect: bool = False):
    rng = np.random.default_rng(seed); y, x = np.mgrid[:size, :size]
    patterns = {
        "carpet": rng.normal(.5, .10, (size, size)),
        "grid": .5 + .25*np.sin(x*.6)*np.sin(y*.6),
        "leather": .5 + .12*np.sin(x*.13 + np.sin(y*.2)) + rng.normal(0,.03,(size,size)),
        "tile": .45 + .15*((x//12 + y//12) % 2),
        "wood": .5 + .23*np.sin(y*.22 + .015*x),
    }
    base = np.clip(patterns[kind], 0, 1); image = np.stack([base*1.03, base, base*.92], -1)
    mask = np.zeros((size, size), bool)
    if defect:
        mask[size//3:2*size//3, size//2-3:size//2+3] = True; image[mask] = np.array([.95,.1,.1])
    return np.clip(image, 0, 1).astype(np.float32), mask
