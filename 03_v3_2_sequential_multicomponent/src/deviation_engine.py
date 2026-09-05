import numpy as np

from sequential_assembly import (
    s,
    deviation_offset,
    deviation_tilt,
    deviation_bend,
    deviation_waviness,
    deviation_twist,
    deviation_local_bump,
)


# ============================================================
# VERSION 3.2 - DEVIATION SCENARIO ENGINE
# ============================================================
#
# Purpose:
# Generate physically interpretable component-level
# geometric deviations for the sequential multi-component
# assembly framework.
#
# Each incoming component may contain:
#     - offset
#     - tilt
#     - bend
#     - waviness
#     - twist
#     - local bump
#     - mixed combinations
#
# Severity is controlled independently.
#
# IMPORTANT:
# Fixture/process drift is NOT handled here.
# That belongs to batch_disturbance.py because it represents
# a process-level condition rather than an individual
# component deviation.
# ============================================================


# ------------------------------------------------------------
# RANDOM NUMBER GENERATOR
# ------------------------------------------------------------

RNG = np.random.default_rng(42)


# ------------------------------------------------------------
# SEVERITY LEVELS
# ------------------------------------------------------------

SEVERITY_SCALE = {
    "low": 0.60,
    "medium": 1.00,
    "high": 1.50,
    "extreme": 2.00,
}


# ------------------------------------------------------------
# SUPPORTED COMPONENT DEVIATION MODES
# ------------------------------------------------------------

SUPPORTED_MODES = [
    "offset",
    "tilt",
    "bend",
    "waviness",
    "twist",
]


# ------------------------------------------------------------
# MAIN DEVIATION GENERATOR
# ------------------------------------------------------------

def generate_deviation_profile(
    severity="medium",
    mixed_probability=0.45,
    local_bump_probability=0.20,
    rng=None,
):
    """
    Generate one component-level geometric deviation profile.

    Parameters
    ----------
    severity : str
        One of:
            low
            medium
            high
            extreme

    mixed_probability : float
        Probability that the component contains more than
        one active deviation mode.

    local_bump_probability : float
        Probability that an additional localized bump
        is present.

    rng : numpy random generator or None
        Allows external reproducible random generators.
        If None, the module-level RNG is used.

    Returns
    -------
    profile : numpy.ndarray
        Total geometric deviation profile along the
        normalized interface coordinate s.

    features : dict
        Interpretable deviation features used later by
        the ML surrogate, optimizer and failure analysis.
    """

    if rng is None:
        rng = RNG

    # --------------------------------------------------------
    # VALIDATE SEVERITY
    # --------------------------------------------------------

    if severity not in SEVERITY_SCALE:
        raise ValueError(
            f"Unsupported severity level: {severity}"
        )

    scale = SEVERITY_SCALE[severity]


    # --------------------------------------------------------
    # SAMPLE POTENTIAL DEVIATION AMPLITUDES
    # --------------------------------------------------------
    #
    # These are controlled prototype assumptions.
    # They are NOT claimed to be universal industrial
    # tolerance distributions.
    #
    # Later these ranges can be justified through:
    #     - literature
    #     - sensitivity analysis
    #     - simulation
    #     - measured CMM / 3D scan data
    # --------------------------------------------------------

    offset_mm = rng.normal(
        loc=0.0,
        scale=0.30 * scale,
    )

    tilt_deg = rng.normal(
        loc=0.0,
        scale=0.020 * scale,
    )

    bend_mm = abs(
        rng.normal(
            loc=0.15 * scale,
            scale=0.06 * scale,
        )
    )

    waviness_mm = abs(
        rng.normal(
            loc=0.08 * scale,
            scale=0.03 * scale,
        )
    )

    twist_mm = rng.normal(
        loc=0.0,
        scale=0.10 * scale,
    )

    local_bump_mm = 0.0


    # --------------------------------------------------------
    # OPTIONAL LOCALIZED DEVIATION
    # --------------------------------------------------------

    if rng.random() < local_bump_probability:

        local_bump_mm = rng.normal(
            loc=0.0,
            scale=0.10 * scale,
        )


    # --------------------------------------------------------
    # DECIDE SINGLE-MODE OR MIXED-MODE COMPONENT
    # --------------------------------------------------------

    if rng.random() < mixed_probability:

        # Mixed case:
        # randomly activate 2 to 4 deviation mechanisms.

        n_modes = int(
            rng.integers(
                low=2,
                high=5,
            )
        )

        active_modes = list(
            rng.choice(
                SUPPORTED_MODES,
                size=n_modes,
                replace=False,
            )
        )

        scenario_type = "mixed"

    else:

        # Single-mode case:
        # activate exactly one deviation mechanism.

        selected_mode = str(
            rng.choice(
                SUPPORTED_MODES
            )
        )

        active_modes = [
            selected_mode
        ]

        scenario_type = selected_mode


    # --------------------------------------------------------
    # INITIAL EMPTY PROFILE
    # --------------------------------------------------------

    profile = np.zeros_like(s)


    # --------------------------------------------------------
    # ADD ACTIVE DEVIATION MODES
    # --------------------------------------------------------

    if "offset" in active_modes:

        profile += deviation_offset(
            offset_mm=offset_mm
        )


    if "tilt" in active_modes:

        profile += deviation_tilt(
            angle_deg=tilt_deg
        )


    if "bend" in active_modes:

        profile += deviation_bend(
            amplitude_mm=bend_mm
        )


    if "waviness" in active_modes:

        profile += deviation_waviness(
            amplitude_mm=waviness_mm,
            waves=3,
        )


    if "twist" in active_modes:

        profile += deviation_twist(
            amplitude_mm=twist_mm
        )


    # --------------------------------------------------------
    # ADD OPTIONAL LOCAL BUMP
    # --------------------------------------------------------

    if local_bump_mm != 0.0:

        profile += deviation_local_bump(
            amplitude_mm=local_bump_mm,
            sigma=0.12,
        )


    # --------------------------------------------------------
    # BUILD INTERPRETABLE FEATURE DICTIONARY
    # --------------------------------------------------------

    features = {

        # Basic classification
        "scenario_type": scenario_type,
        "severity": severity,

        # Human-readable list of active deviations
        "active_modes": "+".join(
            active_modes
        ),

        # Number of active base deviation mechanisms
        "n_active_modes": len(
            active_modes
        ),

        # Individual amplitudes
        "offset_mm": (
            float(offset_mm)
            if "offset" in active_modes
            else 0.0
        ),

        "tilt_deg": (
            float(tilt_deg)
            if "tilt" in active_modes
            else 0.0
        ),

        "bend_mm": (
            float(bend_mm)
            if "bend" in active_modes
            else 0.0
        ),

        "waviness_mm": (
            float(waviness_mm)
            if "waviness" in active_modes
            else 0.0
        ),

        "twist_mm": (
            float(twist_mm)
            if "twist" in active_modes
            else 0.0
        ),

        "local_bump_mm": float(
            local_bump_mm
        ),

        # Simple profile statistics useful later
        "profile_mean_mm": float(
            np.mean(profile)
        ),

        "profile_min_mm": float(
            np.min(profile)
        ),

        "profile_max_mm": float(
            np.max(profile)
        ),

        "profile_rms_mm": float(
            np.sqrt(
                np.mean(
                    profile ** 2
                )
            )
        ),

        "profile_parallelism_mm": float(
            abs(
                profile[-1]
                - profile[0]
            )
        ),
    }


    return profile, features


# ------------------------------------------------------------
# SEVERITY SAMPLER
# ------------------------------------------------------------

def sample_severity(rng=None):
    """
    Sample a severity level.

    The current probabilities deliberately create
    mostly normal/medium components while still producing
    difficult high and extreme cases.

    These probabilities are controlled prototype assumptions.
    """

    if rng is None:
        rng = RNG

    severity = rng.choice(
        [
            "low",
            "medium",
            "high",
            "extreme",
        ],
        p=[
            0.25,
            0.45,
            0.25,
            0.05,
        ],
    )

    return str(severity)


# ------------------------------------------------------------
# GENERATE MULTIPLE COMPONENTS
# ------------------------------------------------------------

def generate_component_set(
    n_components=5,
    rng=None,
):
    """
    Generate a complete set of component deviations
    for one sequential assembly.

    Each component receives its own independent
    deviation profile and severity.
    """

    if rng is None:
        rng = RNG

    component_set = []

    for component_index in range(
        1,
        n_components + 1,
    ):

        severity = sample_severity(
            rng=rng
        )

        profile, features = (
            generate_deviation_profile(
                severity=severity,
                rng=rng,
            )
        )

        features[
            "component_index"
        ] = component_index

        component_set.append(
            {
                "profile": profile,
                "features": features,
            }
        )

    return component_set


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\n"
        "=============================================="
    )

    print(
        "V3.2 DEVIATION ENGINE TEST"
    )

    print(
        "==============================================\n"
    )


    # --------------------------------------------------------
    # TEST 1:
    # Generate ten independent components
    # --------------------------------------------------------

    print(
        "TEST 1 - TEN RANDOM COMPONENTS\n"
    )

    for i in range(10):

        severity = sample_severity()

        profile, features = (
            generate_deviation_profile(
                severity=severity
            )
        )

        print(
            f"Component {i + 1}"
        )

        print(
            f"Scenario       : "
            f"{features['scenario_type']}"
        )

        print(
            f"Severity       : "
            f"{features['severity']}"
        )

        print(
            f"Active modes   : "
            f"{features['active_modes']}"
        )

        print(
            f"Offset         : "
            f"{features['offset_mm']:.4f} mm"
        )

        print(
            f"Tilt           : "
            f"{features['tilt_deg']:.5f} deg"
        )

        print(
            f"Bend           : "
            f"{features['bend_mm']:.4f} mm"
        )

        print(
            f"Waviness       : "
            f"{features['waviness_mm']:.4f} mm"
        )

        print(
            f"Twist          : "
            f"{features['twist_mm']:.4f} mm"
        )

        print(
            f"Local bump     : "
            f"{features['local_bump_mm']:.4f} mm"
        )

        print(
            f"Profile min    : "
            f"{features['profile_min_mm']:.4f} mm"
        )

        print(
            f"Profile max    : "
            f"{features['profile_max_mm']:.4f} mm"
        )

        print(
            f"Profile RMS    : "
            f"{features['profile_rms_mm']:.4f} mm"
        )

        print(
            "----------------------------------------------"
        )


    # --------------------------------------------------------
    # TEST 2:
    # Generate one complete five-component assembly set
    # --------------------------------------------------------

    print(
        "\n"
        "TEST 2 - FIVE COMPONENT ASSEMBLY SET\n"
    )

    component_set = generate_component_set(
        n_components=5
    )

    for component in component_set:

        features = component[
            "features"
        ]

        print(
            f"Component "
            f"{features['component_index']} | "
            f"{features['severity']} | "
            f"{features['active_modes']}"
        )


    print(
        "\n"
        "=============================================="
    )

    print(
        "DEVIATION ENGINE TEST COMPLETED"
    )

    print(
        "=============================================="
    )