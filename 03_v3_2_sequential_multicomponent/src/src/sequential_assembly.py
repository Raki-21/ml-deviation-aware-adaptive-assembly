import numpy as np


# ============================================================
# VERSION 3.2 - SEQUENTIAL MULTI-COMPONENT ASSEMBLY BASELINE
# ============================================================
#
# Purpose:
# Represent a partially assembled product as a geometric profile.
#
# For every incoming component:
#     previous assembly state
#           +
#     component deviation
#           +
#     assembly correction
#           =
#     new assembly state
#
# This is the first V3.2 baseline.
# ML and Bayesian Optimization will be added only after this
# sequential physical logic has been verified.
# ============================================================


# ------------------------------------------------------------
# 1. GLOBAL MODEL SETTINGS
# ------------------------------------------------------------

PROFILE_LENGTH_MM = 1000.0
N_PROFILE_POINTS = 101

# Normalized position along the assembly interface
s = np.linspace(0.0, 1.0, N_PROFILE_POINTS)


# ------------------------------------------------------------
# 2. CREATE EMPTY INITIAL ASSEMBLY STATE
# ------------------------------------------------------------

def create_initial_state():
    """
    Initial nominal assembly state.

    S0(s) = 0

    This means that before any varying components are assembled,
    no accumulated geometric deviation exists.
    """
    return np.zeros_like(s)


# ------------------------------------------------------------
# 3. BASIC COMPONENT DEVIATION MODELS
# ------------------------------------------------------------

def deviation_offset(offset_mm):
    """
    Constant vertical offset over the complete profile.
    """
    return np.full_like(s, offset_mm)


def deviation_tilt(angle_deg):
    """
    Linear angular deviation across the profile.
    """
    angle_rad = np.deg2rad(angle_deg)

    return (
        np.tan(angle_rad)
        * PROFILE_LENGTH_MM
        * (s - 0.5)
    )


def deviation_bend(amplitude_mm):
    """
    Simple smooth bending deviation.

    Maximum deviation occurs near the centre.
    """
    return amplitude_mm * np.sin(np.pi * s)


def deviation_waviness(amplitude_mm, waves=3):
    """
    Periodic surface waviness.
    """
    return amplitude_mm * np.sin(
        2.0 * np.pi * waves * s
    )


def deviation_twist(amplitude_mm):
    """
    Simplified twist-like profile.

    This is intentionally a controlled analytical representation,
    not a full 3D torsional model.
    """
    return amplitude_mm * np.sin(
        2.0 * np.pi * s
    ) * (s - 0.5)


# ------------------------------------------------------------
# 4. EXISTING V3.1 CORRECTION MODEL
# ------------------------------------------------------------

def correction_profile(
    z_adj_mm=0.0,
    theta_adj_deg=0.0,
    locator_offset_mm=0.0,
    locator_sigma=0.12,
):
    """
    Existing V3.1-style correction concept:

    1. z_adj
       global vertical adjustment

    2. theta_adj
       global angular adjustment

    3. locator_offset
       localized locator correction near the profile centre
    """

    theta_rad = np.deg2rad(theta_adj_deg)

    vertical = z_adj_mm

    angular = (
        np.tan(theta_rad)
        * PROFILE_LENGTH_MM
        * (s - 0.5)
    )

    locator = (
        locator_offset_mm
        * np.exp(
            -((s - 0.5) ** 2)
            / (2.0 * locator_sigma ** 2)
        )
    )

    return vertical + angular + locator


# ------------------------------------------------------------
# 5. SEQUENTIAL STATE UPDATE
# ------------------------------------------------------------

def update_assembly_state(
    previous_state,
    component_deviation,
    correction=None,
    fixture_drift=None,
):
    """
    Core Version 3.2 state-transition equation.

    S_k =
        S_(k-1)
        + D_k
        + F_k
        + C_k

    where:

    S_(k-1) = previous assembly state
    D_k     = incoming component deviation
    F_k     = fixture/process disturbance
    C_k     = applied assembly correction
    """

    if correction is None:
        correction = np.zeros_like(previous_state)

    if fixture_drift is None:
        fixture_drift = np.zeros_like(previous_state)

    new_state = (
        previous_state
        + component_deviation
        + fixture_drift
        + correction
    )

    return new_state


# ------------------------------------------------------------
# 6. QUALITY METRICS
# ------------------------------------------------------------

def calculate_quality_metrics(state):
    """
    Calculate geometric quality metrics from the current
    accumulated assembly state.
    """

    gap = np.abs(state)

    mean_gap = np.mean(gap)
    max_gap = np.max(gap)

    rms_deviation = np.sqrt(
        np.mean(state ** 2)
    )

    # End-to-end parallelism indicator
    parallelism_error = abs(
        state[-1] - state[0]
    )

    # Keep the established V3-style weighted quality logic.
    quality_score = (
        0.30 * mean_gap
        + 0.30 * max_gap
        + 0.30 * parallelism_error
        + 0.10 * rms_deviation
    )

    return {
        "mean_gap": mean_gap,
        "max_gap": max_gap,
        "parallelism_error": parallelism_error,
        "rms_deviation": rms_deviation,
        "quality_score": quality_score,
    }


# ------------------------------------------------------------
# 7. SIMPLE FIRST SANITY TEST
# ------------------------------------------------------------

if __name__ == "__main__":

    print("\n==============================================")
    print("VERSION 3.2 - SEQUENTIAL ASSEMBLY TEST")
    print("==============================================\n")

    # Initial nominal state
    state = create_initial_state()

    print("Initial state:")
    print(calculate_quality_metrics(state))

    # --------------------------------------------------------
    # COMPONENT 1
    # Simple +0.5 mm offset
    # --------------------------------------------------------

    component_1 = deviation_offset(
        offset_mm=0.5
    )

    state = update_assembly_state(
        previous_state=state,
        component_deviation=component_1,
    )

    print("\nAfter Component 1 (+0.5 mm offset):")
    print(calculate_quality_metrics(state))

    # --------------------------------------------------------
    # COMPONENT 2
    # Small positive tilt
    # --------------------------------------------------------

    component_2 = deviation_tilt(
        angle_deg=0.02
    )

    state = update_assembly_state(
        previous_state=state,
        component_deviation=component_2,
    )

    print("\nAfter Component 2 (+0.02 deg tilt):")
    print(calculate_quality_metrics(state))

    # --------------------------------------------------------
    # COMPONENT 3
    # Bend
    # --------------------------------------------------------

    component_3 = deviation_bend(
        amplitude_mm=0.30
    )

    state = update_assembly_state(
        previous_state=state,
        component_deviation=component_3,
    )

    print("\nAfter Component 3 (bend):")
    print(calculate_quality_metrics(state))

    # --------------------------------------------------------
    # COMPONENT 4
    # Waviness
    # --------------------------------------------------------

    component_4 = deviation_waviness(
        amplitude_mm=0.15,
        waves=3,
    )

    state = update_assembly_state(
        previous_state=state,
        component_deviation=component_4,
    )

    print("\nAfter Component 4 (waviness):")
    print(calculate_quality_metrics(state))

    # --------------------------------------------------------
    # COMPONENT 5
    # Twist
    # --------------------------------------------------------

    component_5 = deviation_twist(
        amplitude_mm=0.20
    )

    state = update_assembly_state(
        previous_state=state,
        component_deviation=component_5,
    )

    print("\nAfter Component 5 (twist):")
    print(calculate_quality_metrics(state))

    print("\n==============================================")
    print("SEQUENTIAL BASELINE TEST COMPLETED")
    print("==============================================")