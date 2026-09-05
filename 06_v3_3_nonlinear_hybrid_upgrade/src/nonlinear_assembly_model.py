"""
V3.3 nonlinear assembly model.

Purpose
-------
Provide a clean V3.3 assembly model that:

1. preserves V3.2 additive behaviour when nonlinearity_strength = 0.0,
2. supports configurable sequence length,
3. adds controlled nonlinear correction behaviour for future V3.3 studies,
4. keeps the frozen V3.2 implementation untouched.
"""

import numpy as np

from v33_config import (
    PROFILE_LENGTH_MM,
    N_PROFILE_POINTS,
    QUALITY_WEIGHT_MEAN_GAP,
    QUALITY_WEIGHT_MAX_GAP,
    QUALITY_WEIGHT_PARALLELISM,
    QUALITY_WEIGHT_RMS,
)


# ============================================================
# PROFILE AXIS
# ============================================================

s = np.linspace(
    0.0,
    1.0,
    N_PROFILE_POINTS,
)


# ============================================================
# INITIAL STATE
# ============================================================

def create_initial_state():
    """
    Return the nominal zero assembly state.
    """

    return np.zeros(
        N_PROFILE_POINTS,
        dtype=float,
    )


# ============================================================
# DEVIATION MODELS
# ============================================================

def deviation_offset(
    offset_mm,
):
    """
    Uniform vertical offset.
    """

    return np.full(
        N_PROFILE_POINTS,
        offset_mm,
        dtype=float,
    )


def deviation_tilt(
    angle_deg,
):
    """
    Linear angular deviation across the profile.
    """

    angle_rad = np.deg2rad(
        angle_deg
    )

    return (
        np.tan(
            angle_rad
        )
        * PROFILE_LENGTH_MM
        * (
            s
            -
            0.5
        )
    )


def deviation_bend(
    amplitude_mm,
):
    """
    Smooth global bending deviation.
    """

    return (
        amplitude_mm
        *
        np.sin(
            np.pi
            *
            s
        )
    )


def deviation_waviness(
    amplitude_mm,
    waves=3,
):
    """
    Periodic waviness.
    """

    return (
        amplitude_mm
        *
        np.sin(
            2.0
            *
            np.pi
            *
            waves
            *
            s
        )
    )


def deviation_twist(
    amplitude_mm,
):
    """
    Simplified one-dimensional twist-like deviation.

    This remains a controlled analytical representation,
    not a full three-dimensional torsional model.
    """

    return (
        amplitude_mm
        *
        np.sin(
            2.0
            *
            np.pi
            *
            s
        )
        *
        (
            s
            -
            0.5
        )
    )


def deviation_local_bump(
    amplitude_mm,
    center=0.50,
    sigma=0.08,
):
    """
    Localized geometric bump.
    """

    return (
        amplitude_mm
        *
        np.exp(
            -(
                (
                    s
                    -
                    center
                )
                ** 2
            )
            /
            (
                2.0
                *
                sigma
                ** 2
            )
        )
    )


# ============================================================
# V3.2-COMPATIBLE CORRECTION PROFILE
# ============================================================

def correction_profile_linear(
    z_adj_mm=0.0,
    theta_adj_deg=0.0,
    locator_offset_mm=0.0,
    locator_sigma=0.12,
):
    """
    V3.2-compatible linear correction profile.

    This function must preserve the original correction structure.
    """

    theta_rad = np.deg2rad(
        theta_adj_deg
    )

    vertical = np.full(
        N_PROFILE_POINTS,
        z_adj_mm,
        dtype=float,
    )

    angular = (
        np.tan(
            theta_rad
        )
        *
        PROFILE_LENGTH_MM
        *
        (
            s
            -
            0.5
        )
    )

    locator = (
        locator_offset_mm
        *
        np.exp(
            -(
                (
                    s
                    -
                    0.5
                )
                ** 2
            )
            /
            (
                2.0
                *
                locator_sigma
                ** 2
            )
        )
    )

    return (
        vertical
        +
        angular
        +
        locator
    )


# ============================================================
# V3.3 NONLINEAR CORRECTION RESPONSE
# ============================================================

def correction_profile_nonlinear(
    previous_state,
    z_adj_mm=0.0,
    theta_adj_deg=0.0,
    locator_offset_mm=0.0,
    nonlinearity_strength=0.0,
    locator_sigma=0.12,
):
    """
    Controlled V3.3 nonlinear correction response.

    Important:
    ----------
    If nonlinearity_strength == 0.0, this returns exactly
    the V3.2-compatible linear correction profile.

    The nonlinear extension contains two controlled effects:

    1. state-dependent correction effectiveness,
    2. mild interaction between vertical and angular correction.

    These are intentionally simple first-step nonlinearities.
    """

    linear_correction = correction_profile_linear(
        z_adj_mm=z_adj_mm,
        theta_adj_deg=theta_adj_deg,
        locator_offset_mm=locator_offset_mm,
        locator_sigma=locator_sigma,
    )

    if np.isclose(
        nonlinearity_strength,
        0.0,
    ):
        return linear_correction.copy()

    state_scale = np.max(
        np.abs(
            previous_state
        )
    )

    normalized_state_scale = (
        state_scale
        /
        (
            1.0
            +
            state_scale
        )
    )

    effectiveness_factor = (
        1.0
        +
        nonlinearity_strength
        *
        0.20
        *
        normalized_state_scale
    )

    interaction_shape = (
        s
        -
        0.5
    )

    interaction_term = (
        nonlinearity_strength
        *
        0.05
        *
        z_adj_mm
        *
        theta_adj_deg
        *
        interaction_shape
    )

    nonlinear_correction = (
        effectiveness_factor
        *
        linear_correction
        +
        interaction_term
    )

    return nonlinear_correction


# ============================================================
# SEQUENTIAL STATE UPDATE
# ============================================================

def update_assembly_state(
    previous_state,
    component_deviation,
    correction=None,
    fixture_drift=None,
):
    """
    Sequential state update.

    S_k =
        S_(k-1)
        + D_k
        + F_k
        + C_k
    """

    if correction is None:

        correction = np.zeros_like(
            previous_state
        )

    if fixture_drift is None:

        fixture_drift = np.zeros_like(
            previous_state
        )

    new_state = (
        previous_state
        +
        component_deviation
        +
        fixture_drift
        +
        correction
    )

    return new_state


# ============================================================
# QUALITY METRICS
# ============================================================

def calculate_quality_metrics(
    state,
):
    """
    Calculate V3.2-compatible geometric quality metrics.
    """

    gap = np.abs(
        state
    )

    mean_gap = float(
        np.mean(
            gap
        )
    )

    max_gap = float(
        np.max(
            gap
        )
    )

    rms_deviation = float(
        np.sqrt(
            np.mean(
                state
                ** 2
            )
        )
    )

    parallelism_error = float(
        abs(
            state[-1]
            -
            state[0]
        )
    )

    quality_score = float(
        QUALITY_WEIGHT_MEAN_GAP
        *
        mean_gap
        +
        QUALITY_WEIGHT_MAX_GAP
        *
        max_gap
        +
        QUALITY_WEIGHT_PARALLELISM
        *
        parallelism_error
        +
        QUALITY_WEIGHT_RMS
        *
        rms_deviation
    )

    return {
        "mean_gap":
            mean_gap,

        "max_gap":
            max_gap,

        "parallelism_error":
            parallelism_error,

        "rms_deviation":
            rms_deviation,

        "quality_score":
            quality_score,
    }


# ============================================================
# BASIC SANITY TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\n"
        "============================================================"
    )

    print(
        "V3.3 NONLINEAR ASSEMBLY MODEL SANITY TEST"
    )

    print(
        "============================================================"
    )

    initial_state = create_initial_state()

    component = (
        deviation_offset(
            0.20
        )
        +
        deviation_tilt(
            0.05
        )
        +
        deviation_bend(
            0.10
        )
    )

    correction_linear = correction_profile_nonlinear(
        previous_state=initial_state,
        z_adj_mm=-0.10,
        theta_adj_deg=-0.02,
        locator_offset_mm=0.00,
        nonlinearity_strength=0.0,
    )

    correction_nonlinear = correction_profile_nonlinear(
        previous_state=initial_state,
        z_adj_mm=-0.10,
        theta_adj_deg=-0.02,
        locator_offset_mm=0.00,
        nonlinearity_strength=0.50,
    )

    state_linear = update_assembly_state(
        previous_state=initial_state,
        component_deviation=component,
        correction=correction_linear,
    )

    state_nonlinear = update_assembly_state(
        previous_state=initial_state,
        component_deviation=component,
        correction=correction_nonlinear,
    )

    print(
        "\nLinear-mode quality:"
    )

    print(
        calculate_quality_metrics(
            state_linear
        )
    )

    print(
        "\nNonlinear-mode quality:"
    )

    print(
        calculate_quality_metrics(
            state_nonlinear
        )
    )

    zero_strength_difference = np.max(
        np.abs(
            correction_profile_nonlinear(
                previous_state=initial_state,
                z_adj_mm=-0.10,
                theta_adj_deg=-0.02,
                locator_offset_mm=0.00,
                nonlinearity_strength=0.0,
            )
            -
            correction_profile_linear(
                z_adj_mm=-0.10,
                theta_adj_deg=-0.02,
                locator_offset_mm=0.00,
            )
        )
    )

    print(
        "\nZero-nonlinearity difference from linear model:"
    )

    print(
        zero_strength_difference
    )

    if zero_strength_difference <= 1e-12:

        print(
            "\nPASS: zero nonlinearity reproduces linear correction."
        )

    else:

        print(
            "\nFAIL: zero nonlinearity does not reproduce linear correction."
        )

    print(
        "\n"
        "============================================================"
    )