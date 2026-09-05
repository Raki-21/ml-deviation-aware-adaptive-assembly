import numpy as np

from src.config import (
    PROFILE_LENGTH_MM,
    N_POINTS,
    LOCATOR_POSITION,
    LOCATOR_SIGMA,
)


def create_normalized_axis(n_points: int = N_POINTS) -> np.ndarray:
    return np.linspace(0.0, 1.0, n_points)


def create_physical_axis(
    length_mm: float = PROFILE_LENGTH_MM,
    n_points: int = N_POINTS
) -> np.ndarray:
    return np.linspace(0.0, length_mm, n_points)


def generate_nominal_profile(n_points: int = N_POINTS) -> np.ndarray:
    return np.zeros(n_points)


def generate_deviation_profile(
    dev_offset: float,
    dev_tilt: float,
    dev_bend: float,
    dev_waviness: float,
    dev_noise: float,
    n_points: int = N_POINTS,
    length_mm: float = PROFILE_LENGTH_MM,
    waviness_frequency: int = 3,
    phase: float | None = None,
    random_seed: int | None = None
) -> np.ndarray:
    rng = np.random.default_rng(random_seed)
    s = create_normalized_axis(n_points)

    if phase is None:
        phase = rng.uniform(0.0, 2.0 * np.pi)

    dev_tilt_rad = np.deg2rad(dev_tilt)

    offset_component = dev_offset
    tilt_component = np.tan(dev_tilt_rad) * length_mm * (s - 0.5)
    bend_component = dev_bend * (4.0 * (s - 0.5) ** 2 - 1.0)
    waviness_component = dev_waviness * np.sin(
        2.0 * np.pi * waviness_frequency * s + phase
    )
    noise_component = rng.normal(0.0, dev_noise, size=n_points)

    z_b = (
        offset_component
        + tilt_component
        + bend_component
        + waviness_component
        + noise_component
    )

    return z_b


def locator_effect_shape(
    n_points: int = N_POINTS,
    s_loc: float = LOCATOR_POSITION,
    sigma: float = LOCATOR_SIGMA
) -> np.ndarray:
    s = create_normalized_axis(n_points)
    return np.exp(-((s - s_loc) ** 2) / (2.0 * sigma ** 2))