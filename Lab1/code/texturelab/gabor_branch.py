"""Student-facing Gabor branch. Core implementation is intentionally inspectable."""

from __future__ import annotations
from dataclasses import asdict
import numpy as np
from .config import GaborConfig
from .supplied import box_mean, correlate2d


def make_gabor_bank(config: GaborConfig) -> list[tuple[np.ndarray, dict]]:
    """Return ordered zero-mean, unit-norm kernels and their metadata."""
    # TODO 1 — Validate the configuration
    #   - Require kernel_size to be odd and at least 3. Raise ValueError if not.
    #
    # TODO 2 — Build the centred sampling grid
    #   - Construct float32 x/y coordinates spanning equally on both sides of 0.
    #   - The resulting grid must have shape (kernel_size, kernel_size).
    #
    # TODO 3 — Generate every filter in a stable order
    #   - Loop over frequency first, orientation second, and phase last.
    #   - Rotate the x coordinate into the current orientation.
    #   - Multiply a Gaussian envelope by a sine carrier with the current frequency and phase.
    #   - Remember to add the phase in the metadata.
    #
    # TODO 4 — Normalise and validate each kernel
    #   - Subtract the kernel mean so constant images have little response.
    #   - Divide by its Euclidean norm and reject a near-zero/degenerate norm.
    #   - Store the final kernel as float32.
    #
    # TODO 5 — Attach metadata and return
    #   - Include frequency, orientation, phase, and asdict(config) for each item.
    #   - Return a list of (kernel, metadata) tuples in the loop order above.
    
    if config.kernel_size < 3 or config.kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer of at least 3")

    half = config.kernel_size // 2
    coords = np.linspace(-half, half, config.kernel_size, dtype=np.float32)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")

    bank: list[tuple[np.ndarray, dict]] = []
    for frequency in config.frequencies:
        for orientation in config.orientations:
            for phase in config.phases:
                x_rot = xx * np.cos(orientation) + yy * np.sin(orientation)
                envelope = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * config.sigma ** 2))
                carrier = np.sin(2.0 * np.pi * frequency * x_rot + phase)
                kernel = (envelope * carrier).astype(np.float32)
                kernel = kernel - kernel.mean()

                norm = float(np.linalg.norm(kernel))
                if not np.isfinite(norm) or norm <= np.finfo(np.float32).eps:
                    raise ValueError("degenerate Gabor kernel: norm is zero or non-finite")
                kernel = kernel / norm

                metadata = {
                    "frequency": float(frequency),
                    "orientation": float(orientation),
                    "phase": float(phase),
                    **asdict(config),
                }
                bank.append((kernel.astype(np.float32), metadata))

    return bank


def gabor_energy_maps(gray: np.ndarray, bank: list[tuple[np.ndarray, dict]], pool_size: int = 9,
                      energy: str = "squared") -> np.ndarray:

    """Return finite H x W x K locally pooled energy maps."""
    # TODO 1 — Compute one response map per bank entry
    #   - Preserve bank order and correlate (do not convolve) gray with each kernel.
    #
    # TODO 2 — Convert responses to non-negative energy
    #   - For "squared", square the signed response.
    #   - For "absolute", take its absolute value.
    #   - Raise ValueError for any other energy name.
    #
    # TODO 3 — Pool and assemble the channels
    #   - Apply box_mean with pool_size to every energy map.
    #   - Pooling must preserve the original image height and width.
    #   - Stack maps on the final axis to obtain H x W x K float32 output.
    #
    # TODO 4 — Check numerical validity
    #   - Raise FloatingPointError if any returned value is NaN or infinite.

    gray = np.asarray(gray, dtype=np.float32)
    if gray.ndim != 2:
        raise ValueError("gray must be a 2D image")

    if energy == "squared":
        transform = lambda response: response ** 2
    elif energy == "absolute":
        transform = lambda response: np.abs(response)
    else:
        raise ValueError(f"unsupported energy mode: {energy}")

    maps = []
    for kernel, _ in bank:
        response = correlate2d(gray, kernel)
        energy_map = transform(response).astype(np.float32)
        pooled = box_mean(energy_map, pool_size).astype(np.float32)
        maps.append(pooled)

    stacked = np.stack(maps, axis=-1).astype(np.float32)
    if not np.isfinite(stacked).all():
        raise FloatingPointError("Gabor energy output contains NaN or infinity")
    return stacked
