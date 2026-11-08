"""Student-facing exact-direction NMS, adaptive thresholds, and hysteresis."""

from __future__ import annotations
from collections import deque
import numpy as np
from .config import EdgeConfig
from .supplied import gaussian_smooth, sobel_gradients


def bilinear_sample(image: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Sample a 2-D image at floating-point image coordinates."""
    # YOUR CODE HERE
    #
    # TODO 1 — Resolve safe coordinates
    #   - Read image height/width and clamp y/x to the valid closed intervals.
    #   - Compute floor coordinates (y0, x0) and the next coordinates (y1, x1).
    #   - Clamp y1/x1 as well so samples at the last row/column remain valid.
    #
    # TODO 2 — Compute interpolation weights
    #   - wy and wx are the fractional offsets from (y0, x0).
    #   - Form the four complementary weights for top-left, top-right,
    #     bottom-left, and bottom-right pixels.
    #
    # TODO 3 — Return the weighted sample
    #   - Support array-valued y/x so a whole image grid can be sampled at once.
    #   - An exact integer coordinate should reproduce the corresponding pixel.
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("image must be two-dimensional")
    y, x = np.broadcast_arrays(np.asarray(y, dtype=float), np.asarray(x, dtype=float))
    height, width = image.shape
    if height == 0 or width == 0:
        return np.empty(y.shape, dtype=np.result_type(image, np.float32))

    safe_y = np.clip(y, 0, height - 1)
    safe_x = np.clip(x, 0, width - 1)
    y0 = np.floor(safe_y).astype(np.intp)
    x0 = np.floor(safe_x).astype(np.intp)
    y1 = np.minimum(y0 + 1, height - 1)
    x1 = np.minimum(x0 + 1, width - 1)
    wy = safe_y - y0
    wx = safe_x - x0

    top_left = image[y0, x0]
    top_right = image[y0, x1]
    bottom_left = image[y1, x0]
    bottom_right = image[y1, x1]
    return ((1 - wy) * (1 - wx) * top_left
            + (1 - wy) * wx * top_right
            + wy * (1 - wx) * bottom_left
            + wy * wx * bottom_right)


def nms_interpolated(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Thin gradient magnitude along the exact image-coordinate direction."""
    # YOUR CODE HERE
    #
    # TODO 1 — Validate inputs
    #   - Require matching two-dimensional magnitude and direction arrays.
    #
    # TODO 2 — Construct exact-direction comparison coordinates
    #   - Build row/column index grids for the complete image.
    #   - In image coordinates, the unit step is dy=sin(theta), dx=cos(theta).
    #   - Bilinearly sample magnitude one step forward and one step backward.
    #
    # TODO 3 — Suppress non-maxima
    #   - Keep the original magnitude when it is >= both interpolated neighbours.
    #   - Write zero elsewhere, return float32, and explicitly zero all four borders.
    magnitude = np.asarray(magnitude)
    direction = np.asarray(direction)
    if magnitude.ndim != 2 or direction.ndim != 2:
        raise ValueError("magnitude and direction must be two-dimensional")
    if magnitude.shape != direction.shape:
        raise ValueError("magnitude and direction must have matching shapes")

    height, width = magnitude.shape
    rows, columns = np.indices((height, width), dtype=float)
    dy = np.sin(direction)
    dx = np.cos(direction)
    forward = bilinear_sample(magnitude, rows + dy, columns + dx)
    backward = bilinear_sample(magnitude, rows - dy, columns - dx)
    result = np.where((magnitude >= forward) & (magnitude >= backward), magnitude, 0)
    result = np.asarray(result, dtype=np.float32)
    if height and width:
        result[0, :] = 0
        result[-1, :] = 0
        result[:, 0] = 0
        result[:, -1] = 0
    return result


def adaptive_thresholds(nms: np.ndarray, config: EdgeConfig) -> tuple[float, float]:
    """Select reproducible low/high thresholds without test labels."""
    # YOUR CODE HERE
    #
    # TODO 1 — Isolate useful responses
    #   - Estimate thresholds from strictly positive NMS values only.
    #   - If none exist, return a pair that produces no strong/weak edges.
    #
    # TODO 2 — Compute the high threshold
    #   - "percentile": use config.high_percentile on positive values.
    #   - "robust": use median + 2.5 * 1.4826 * median absolute deviation.
    #       - 1.4826 is a mathematically motivated conversion from MAD to a standard-deviation-like scale.
    #       - 2.5 is a scale multiplier for robust outlier rejection.
    #   - Reject unknown threshold_method values with ValueError.
    #
    # TODO 3 — Compute and return the low threshold
    #   - low is config.low_ratio multiplied by high.
    #   - Return ordinary Python floats in (low, high) order.
    if config.threshold_method not in {"percentile", "robust"}:
        raise ValueError(f"unknown threshold method: {config.threshold_method}")
    positive = np.asarray(nms)[np.asarray(nms) > 0]
    if positive.size == 0:
        return 0.0, 0.0

    if config.threshold_method == "percentile":
        high = float(np.percentile(positive, config.high_percentile))
    else:
        median = float(np.median(positive))
        mad = float(np.median(np.abs(positive - median)))
        high = median + 2.5 * 1.4826 * mad
    low = float(config.low_ratio * high)
    return low, high


def hysteresis(strong: np.ndarray, weak: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """Keep all weak pixels reachable from strong seeds."""
    # YOUR CODE HERE
    #
    # TODO 1 — Validate and initialise
    #   - Require matching mask shapes and connectivity equal to 4 or 8.
    #   - Convert inputs to Boolean masks without modifying the caller's arrays.
    #   - Seed the result and a deque with every strong-pixel coordinate.
    #
    # TODO 2 — Define the neighbourhood
    #   - Four-connectivity uses vertical/horizontal offsets.
    #   - Eight-connectivity additionally includes all diagonal offsets.
    #
    # TODO 3 — Traverse the full connected component
    #   - Pop a coordinate, check in-bounds allowed neighbours, and accept every
    #     unvisited weak pixel connected to a seed.
    #   - Enqueue newly accepted pixels so multi-pixel weak chains are retained.
    #   - Return the final Boolean mask after the queue is exhausted.
    strong = np.asarray(strong, dtype=bool)
    weak = np.asarray(weak, dtype=bool)
    if strong.shape != weak.shape:
        raise ValueError("strong and weak masks must have matching shapes")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")

    result = strong.copy()
    queue = deque(map(tuple, np.argwhere(strong)))
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    height, width = result.shape
    while queue:
        row, column = queue.popleft()
        for row_offset, column_offset in offsets:
            next_row = row + row_offset
            next_column = column + column_offset
            if (0 <= next_row < height and 0 <= next_column < width
                    and weak[next_row, next_column] and not result[next_row, next_column]):
                result[next_row, next_column] = True
                queue.append((next_row, next_column))
    return result


def detect_edges(gray: np.ndarray, config: EdgeConfig) -> dict[str, np.ndarray | float]:
    """Run the supplied smoothing/Sobel stages and student edge stages."""
    smoothed = gaussian_smooth(gray, config.gaussian_sigma)
    gx, gy, magnitude, direction = sobel_gradients(smoothed)
    # YOUR CODE HERE
    #
    # TODO 1 — Thin and threshold
    #   - Run nms_interpolated on magnitude/direction.
    #   - Obtain (low, high) from adaptive_thresholds.
    #   - Strong pixels meet/exceed high; weak pixels meet/exceed low but are not strong.
    #
    # TODO 2 — Link and package the result
    #   - Run hysteresis with config.connectivity.
    #   - Return gx, gy, magnitude, direction, nms, low, high, and edges using
    #     exactly those dictionary keys so downstream feature code remains stable.
    nms = nms_interpolated(magnitude, direction)
    low, high = adaptive_thresholds(nms, config)
    strong = (nms > 0) & (nms >= high)
    weak = (nms > 0) & (nms >= low) & ~strong
    edges = hysteresis(strong, weak, config.connectivity)
    return {
        "gx": gx,
        "gy": gy,
        "magnitude": magnitude,
        "direction": direction,
        "nms": nms,
        "low": low,
        "high": high,
        "edges": edges,
    }
