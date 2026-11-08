"""Supplied image operations: students are not expected to reimplement these."""

from __future__ import annotations
import numpy as np
from PIL import Image


def load_image(path: str, size: tuple[int, int] | None = None) -> np.ndarray:
    """Load an image as float RGB in [0, 1], optionally resized to (width, height)."""
    image = Image.open(path).convert("RGB")
    if size is not None:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def rgb2gray(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        return image
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError("expected HxW or HxWx3 image")
    return np.tensordot(image[..., :3], np.array([0.2126, 0.7152, 0.0722]), axes=1).astype(np.float32)


def _sliding_windows(image: np.ndarray, kh: int, kw: int) -> np.ndarray:
    ph, pw = kh // 2, kw // 2
    padded = np.pad(image, ((ph, ph), (pw, pw)), mode="reflect")
    return np.lib.stride_tricks.sliding_window_view(padded, (kh, kw))


def correlate2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Same-size 2-D correlation with reflected boundaries."""
    image, kernel = np.asarray(image), np.asarray(kernel)
    if image.ndim != 2 or kernel.ndim != 2:
        raise ValueError("image and kernel must be two-dimensional")
    return np.einsum("ijxy,xy->ij", _sliding_windows(image, *kernel.shape), kernel, optimize=True)


def gaussian_kernel(sigma: float, truncate: float = 3.0) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    radius = max(1, int(truncate * sigma + 0.5))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x[:, None] ** 2 + x[None, :] ** 2) / (2 * sigma**2))
    return k / k.sum()


def gaussian_smooth(image: np.ndarray, sigma: float) -> np.ndarray:
    return correlate2d(np.asarray(image), gaussian_kernel(sigma)).astype(np.float32)


def sobel_gradients(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return gx, gy, magnitude, and angle; +y points down in image coordinates."""
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32) / 8
    sy = sx.T
    gx, gy = correlate2d(gray, sx), correlate2d(gray, sy)
    magnitude = np.hypot(gx, gy)
    return gx, gy, magnitude.astype(np.float32), np.arctan2(gy, gx).astype(np.float32)


def box_mean(image: np.ndarray, size: int) -> np.ndarray:
    if size < 1 or size % 2 == 0:
        raise ValueError("size must be a positive odd integer")
    return correlate2d(np.asarray(image), np.ones((size, size), np.float32) / size**2)


def nms_nearest(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Four-bin NMS baseline, supplied only for comparison."""
    a = (np.rad2deg(direction) + 180) % 180
    out = np.zeros_like(magnitude)
    for lo, hi, shifts in [(0, 22.5, ((0, -1), (0, 1))), (22.5, 67.5, ((-1, 1), (1, -1))),
                            (67.5, 112.5, ((-1, 0), (1, 0))), (112.5, 157.5, ((-1, -1), (1, 1))),
                            (157.5, 180, ((0, -1), (0, 1)))]:
        mask = (a >= lo) & (a < hi)
        n1 = np.roll(magnitude, shifts[0], axis=(0, 1)); n2 = np.roll(magnitude, shifts[1], axis=(0, 1))
        out[mask & (magnitude >= n1) & (magnitude >= n2)] = magnitude[mask & (magnitude >= n1) & (magnitude >= n2)]
    out[[0, -1], :] = 0; out[:, [0, -1]] = 0
    return out
