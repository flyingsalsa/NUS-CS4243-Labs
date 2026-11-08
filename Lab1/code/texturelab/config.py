"""Small serialisable configurations used by every branch."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GaborConfig:
    frequencies: tuple[float, ...] = (0.08, 0.16, 0.28)
    orientations: tuple[float, ...] = (0.0, 0.785398, 1.570796, 2.356194)
    # Phase offsets (radians) applied to a sine carrier.  The baseline uses one
    # sine phase; experiments may explicitly request additional offsets.
    phases: tuple[float, ...] = (0.0,)
    sigma: float = 3.0
    kernel_size: int = 15
    pool_size: int = 9
    energy: str = "squared"


@dataclass(frozen=True)
class EdgeConfig:
    gaussian_sigma: float = 1.2
    threshold_method: str = "percentile"
    high_percentile: float = 90.0
    low_ratio: float = 0.4
    connectivity: int = 8
    density_size: int = 9
    n_orientations: int = 8


@dataclass(frozen=True)
class FeatureConfig:
    gabor: GaborConfig = field(default_factory=GaborConfig)
    edge: EdgeConfig = field(default_factory=EdgeConfig)
    colour_space: str = "rgb"
    include_colour: bool = True
    include_gabor: bool = True
    include_gradient: bool = True
    include_edges: bool = True
    standardise_per_image: bool = False


@dataclass(frozen=True)
class NormalityConfig:
    epsilon: float = 1e-6
    image_percentile: float = 99.0
    mask_threshold: float | None = None
    threshold_percentile: float = 99.5
    max_samples: int = 200_000
    random_seed: int = 7
    score_pool_size: int = 7
    ignore_border: int = 5
