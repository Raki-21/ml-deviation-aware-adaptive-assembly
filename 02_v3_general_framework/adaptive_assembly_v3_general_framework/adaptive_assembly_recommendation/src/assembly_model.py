import numpy as np

from src.config import PROFILE_LENGTH_MM
from src.deviation_generator import create_normalized_axis, locator_effect_shape


def apply_assembly_correction(
    deviated_profile: np.ndarray,
    z_adj: float,
    theta_adj: float,
    locator_offset: float,
    length_mm: float = PROFILE_LENGTH_MM
) -> np.ndarray:
    n_points = len(deviated_profile)
    s = create_normalized_axis(n_points)

    theta_adj_rad = np.deg2rad(theta_adj)

    vertical_correction = z_adj
    rotation_correction = np.tan(theta_adj_rad) * length_mm * (s - 0.5)
    local_locator_correction = locator_offset * locator_effect_shape(n_points)

    corrected_profile = (
        deviated_profile
        + vertical_correction
        + rotation_correction
        + local_locator_correction
    )

    return corrected_profile


def calculate_gap(
    reference_profile: np.ndarray,
    corrected_profile: np.ndarray
) -> np.ndarray:
    return corrected_profile - reference_profile