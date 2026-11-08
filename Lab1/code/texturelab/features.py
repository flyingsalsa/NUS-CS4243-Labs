"""Aligned local descriptor shared by all three tasks."""

from __future__ import annotations
import numpy as np
from .config import FeatureConfig
from .edge_branch import detect_edges
from .gabor_branch import gabor_energy_maps, make_gabor_bank
from .supplied import box_mean, rgb2gray


def extract_local_features(image: np.ndarray, config: FeatureConfig) -> tuple[np.ndarray, list[str]]:
    """Return an aligned H x W x D map and one name per channel."""
    # YOUR CODE HERE
    #
    # TODO 1 — Normalise the input representation
    #   - Convert to float32; expand a 2-D image to three identical channels.
    #   - Scale 0--255-style inputs to 0--1 while leaving 0--1 inputs unchanged.
    #   - Compute grayscale once with rgb2gray.
    #
    # TODO 2 — Initialise aligned channel/name groups
    #   - Keep feature chunks and their names in matching, deterministic order.
    #   - If include_colour, append the first three image channels named
    #     colour_r, colour_g, and colour_b.
    #
    # TODO 3 — Add the Gabor branch when enabled
    #   - Build config.gabor's bank and compute pooled maps using its pool_size
    #     and energy mode.
    #   - Create one stable name from each kernel's frequency/orientation/phase.
    #
    # TODO 4 — Compute edge data only when needed
    #   - Call detect_edges once if gradient or edge channels are requested.
    #   - For gradient energy, square magnitude, box-pool it with density_size,
    #     add a singleton channel axis, and name it gradient_energy.
    #
    # TODO 5 — Add total and orientation-specific edge density
    #   - Pool the Boolean edge map for edge_density.
    #   - Map directions modulo pi, divide the unsigned orientation range into
    #     config.edge.n_orientations bins, and pool edges belonging to each bin.
    #   - Name bins edge_orientation_0, edge_orientation_1, and so on.
    #
    # TODO 6 — Assemble and optionally standardise
    #   - Reject a configuration with no enabled feature family.
    #   - Concatenate chunks along the final axis and return float32.
    #   - If standardise_per_image, standardise every channel over H/W using a
    #     small epsilon; return the feature map and equally long name list.
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError("image must be HxW or HxWx3")
    if image.size and float(np.nanmax(image)) > 1.0:
        image = image / 255.0
    gray = rgb2gray(image)

    chunks: list[np.ndarray] = []
    names: list[str] = []
    if config.include_colour:
        chunks.append(image[..., :3])
        names.extend(["colour_r", "colour_g", "colour_b"])

    if config.include_gabor:
        bank = make_gabor_bank(config.gabor)
        gabor_maps = gabor_energy_maps(gray, bank, config.gabor.pool_size,
                                       config.gabor.energy)
        chunks.append(gabor_maps)
        names.extend(
            f"gabor_f{metadata['frequency']:g}_o{metadata['orientation']:g}_p{metadata['phase']:g}"
            for _, metadata in bank
        )

    edge_data = None
    if config.include_gradient or config.include_edges:
        edge_data = detect_edges(gray, config.edge)

    if config.include_gradient:
        gradient_energy = box_mean(edge_data["magnitude"] ** 2,
                                   config.edge.density_size).astype(np.float32)
        chunks.append(gradient_energy[..., None])
        names.append("gradient_energy")

    if config.include_edges:
        edge_map = edge_data["edges"].astype(np.float32)
        edge_density = box_mean(edge_map, config.edge.density_size).astype(np.float32)
        edge_chunks = [edge_density[..., None]]
        edge_names = ["edge_density"]

        if config.edge.n_orientations < 1:
            raise ValueError("n_orientations must be positive")
        orientation = np.mod(edge_data["direction"], np.pi)
        bin_width = np.pi / config.edge.n_orientations
        for bin_index in range(config.edge.n_orientations):
            in_bin = ((orientation >= bin_index * bin_width)
                      & (orientation < (bin_index + 1) * bin_width))
            pooled = box_mean((edge_data["edges"] & in_bin).astype(np.float32),
                              config.edge.density_size)
            edge_chunks.append(pooled[..., None].astype(np.float32))
            edge_names.append(f"edge_orientation_{bin_index}")
        chunks.extend(edge_chunks)
        names.extend(edge_names)

    if not chunks:
        raise ValueError("at least one feature family must be enabled")
    features = np.concatenate(chunks, axis=-1).astype(np.float32)
    if config.standardise_per_image:
        mean = features.mean(axis=(0, 1), keepdims=True)
        std = features.std(axis=(0, 1), keepdims=True)
        features = (features - mean) / np.maximum(std, 1e-6)
    return features.astype(np.float32), names


def global_pool(feature_map: np.ndarray,
                statistics: tuple[str, ...] = ("mean", "std", "p90")) -> np.ndarray:
    """Pool local channels into one reproducible image descriptor."""
    # YOUR CODE HERE
    #
    # TODO 1 — Flatten spatial positions
    #   - Reshape H x W x D into (H*W) x D without changing channel order.
    #
    # TODO 2 — Implement the supported statistics
    #   - Support mean, std, p10, p50, and p90, each computed per channel.
    #   - Raise ValueError naming an unsupported statistic.
    #
    # TODO 3 — Assemble the descriptor
    #   - Evaluate statistics in the exact caller-supplied order.
    #   - Concatenate their D-vectors and return a one-dimensional float32 array.
    feature_map = np.asarray(feature_map)
    if feature_map.ndim != 3:
        raise ValueError("feature_map must be HxWxD")
    flat = feature_map.reshape(-1, feature_map.shape[-1])
    pooled = []
    for statistic in statistics:
        if statistic == "mean":
            values = np.mean(flat, axis=0)
        elif statistic == "std":
            values = np.std(flat, axis=0)
        elif statistic == "p10":
            values = np.percentile(flat, 10, axis=0)
        elif statistic == "p50":
            values = np.percentile(flat, 50, axis=0)
        elif statistic == "p90":
            values = np.percentile(flat, 90, axis=0)
        else:
            raise ValueError(f"unsupported statistic: {statistic}")
        pooled.append(values)
    return np.concatenate(pooled).astype(np.float32)


def feature_family_indices(names: list[str]) -> dict[str, list[int]]:
    """Map colour/Gabor/gradient/edge families to descriptor indices."""
    # YOUR CODE HERE
    #
    # TODO 1 — Create all required keys
    #   - Initialise colour, gabor, gradient, and edge to empty index lists.
    #
    # TODO 2 — Assign channels by their stable name prefix
    #   - Iterate through names in order, take the text before the first "_",
    #     and append the channel index to the matching family.
    #   - Return every family key even when that family has no channels.
    families = {"colour": [], "gabor": [], "gradient": [], "edge": []}
    for index, name in enumerate(names):
        family = name.split("_", 1)[0]
        if family in families:
            families[family].append(index)
    return families
