import numpy as np

from sequential_assembly import (
    s,
    PROFILE_LENGTH_MM,
)


# ============================================================
# VERSION 3.2 - BATCH / PROCESS DISTURBANCE ENGINE
# ============================================================
#
# Purpose:
# Represent shared manufacturing-process conditions that
# affect multiple components/assemblies in the same batch.
#
# IMPORTANT:
# These are NOT individual component deviations.
#
# Component-level variation is handled in:
#     deviation_engine.py
#
# Batch/process-level variation is handled here.
# ============================================================


# ------------------------------------------------------------
# RANDOM GENERATOR
# ------------------------------------------------------------

RNG = np.random.default_rng(123)


# ------------------------------------------------------------
# SUPPORTED BATCH CONDITIONS
# ------------------------------------------------------------

SUPPORTED_BATCH_CONDITIONS = [
    "normal",
    "offset_drift",
    "angular_drift",
    "fixture_drift",
    "high_variation",
    "disturbed",
]


# ------------------------------------------------------------
# CREATE ONE BATCH CONDITION
# ------------------------------------------------------------

def create_batch_condition(
    condition="normal",
    rng=None,
):
    """
    Create one batch/process-level disturbance.

    Parameters
    ----------
    condition : str
        One of:
            normal
            offset_drift
            angular_drift
            fixture_drift
            high_variation
            disturbed

    rng : numpy random generator or None

    Returns
    -------
    disturbance_profile : numpy.ndarray
        Shared geometric disturbance profile.

    batch_features : dict
        Interpretable process/batch descriptors.
    """

    if rng is None:
        rng = RNG

    if condition not in SUPPORTED_BATCH_CONDITIONS:
        raise ValueError(
            f"Unsupported batch condition: {condition}"
        )

    # --------------------------------------------------------
    # DEFAULT VALUES
    # --------------------------------------------------------

    offset_bias_mm = 0.0
    angular_bias_deg = 0.0
    fixture_drift_mm = 0.0
    variation_multiplier = 1.0


    # --------------------------------------------------------
    # BATCH CONDITION DEFINITIONS
    # --------------------------------------------------------

    if condition == "normal":

        pass


    elif condition == "offset_drift":

        offset_bias_mm = rng.normal(
            loc=0.20,
            scale=0.05,
        )


    elif condition == "angular_drift":

        angular_bias_deg = rng.normal(
            loc=0.015,
            scale=0.005,
        )


    elif condition == "fixture_drift":

        fixture_drift_mm = rng.normal(
            loc=0.20,
            scale=0.05,
        )


    elif condition == "high_variation":

        variation_multiplier = 1.50


    elif condition == "disturbed":

        offset_bias_mm = rng.normal(
            loc=0.15,
            scale=0.05,
        )

        angular_bias_deg = rng.normal(
            loc=0.010,
            scale=0.004,
        )

        fixture_drift_mm = rng.normal(
            loc=0.15,
            scale=0.04,
        )

        variation_multiplier = 1.50


    # --------------------------------------------------------
    # BUILD OFFSET DISTURBANCE
    # --------------------------------------------------------

    offset_profile = np.full_like(
        s,
        offset_bias_mm,
    )


    # --------------------------------------------------------
    # BUILD ANGULAR DISTURBANCE
    # --------------------------------------------------------

    angular_profile = (
        np.tan(
            np.deg2rad(
                angular_bias_deg
            )
        )
        * PROFILE_LENGTH_MM
        * (s - 0.5)
    )


    # --------------------------------------------------------
    # BUILD FIXTURE-LOCAL DISTURBANCE
    # --------------------------------------------------------

    fixture_sigma = 0.18

    fixture_profile = (
        fixture_drift_mm
        * np.exp(
            -((s - 0.5) ** 2)
            / (2.0 * fixture_sigma ** 2)
        )
    )


    # --------------------------------------------------------
    # COMBINE PROCESS EFFECTS
    # --------------------------------------------------------

    disturbance_profile = (
        offset_profile
        + angular_profile
        + fixture_profile
    )


    # --------------------------------------------------------
    # BATCH FEATURE DICTIONARY
    # --------------------------------------------------------

    batch_features = {

        "batch_condition":
            condition,

        "batch_offset_bias_mm":
            float(offset_bias_mm),

        "batch_angular_bias_deg":
            float(angular_bias_deg),

        "fixture_drift_mm":
            float(fixture_drift_mm),

        "variation_multiplier":
            float(variation_multiplier),

        "disturbance_mean_mm":
            float(
                np.mean(
                    disturbance_profile
                )
            ),

        "disturbance_min_mm":
            float(
                np.min(
                    disturbance_profile
                )
            ),

        "disturbance_max_mm":
            float(
                np.max(
                    disturbance_profile
                )
            ),

        "disturbance_rms_mm":
            float(
                np.sqrt(
                    np.mean(
                        disturbance_profile ** 2
                    )
                )
            ),

        "disturbance_parallelism_mm":
            float(
                abs(
                    disturbance_profile[-1]
                    - disturbance_profile[0]
                )
            ),
    }

    return (
        disturbance_profile,
        batch_features,
    )


# ------------------------------------------------------------
# RANDOM BATCH CONDITION SAMPLER
# ------------------------------------------------------------

def sample_batch_condition(
    rng=None,
):
    """
    Sample a batch condition.

    Most batches remain normal, while a smaller fraction
    contain systematic disturbances.
    """

    if rng is None:
        rng = RNG

    condition = rng.choice(
        SUPPORTED_BATCH_CONDITIONS,
        p=[
            0.40,  # normal
            0.15,  # offset drift
            0.10,  # angular drift
            0.15,  # fixture drift
            0.10,  # high variation
            0.10,  # disturbed
        ],
    )

    return str(condition)


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\n"
        "=============================================="
    )

    print(
        "V3.2 BATCH DISTURBANCE ENGINE TEST"
    )

    print(
        "==============================================\n"
    )


    # --------------------------------------------------------
    # TEST ALL DEFINED CONDITIONS
    # --------------------------------------------------------

    for condition in SUPPORTED_BATCH_CONDITIONS:

        profile, features = (
            create_batch_condition(
                condition=condition
            )
        )

        print(
            f"Condition: "
            f"{features['batch_condition']}"
        )

        print(
            f"Offset bias       : "
            f"{features['batch_offset_bias_mm']:.4f} mm"
        )

        print(
            f"Angular bias      : "
            f"{features['batch_angular_bias_deg']:.5f} deg"
        )

        print(
            f"Fixture drift     : "
            f"{features['fixture_drift_mm']:.4f} mm"
        )

        print(
            f"Variation factor  : "
            f"{features['variation_multiplier']:.2f}"
        )

        print(
            f"Disturbance RMS   : "
            f"{features['disturbance_rms_mm']:.4f} mm"
        )

        print(
            f"Parallelism effect: "
            f"{features['disturbance_parallelism_mm']:.4f} mm"
        )

        print(
            "----------------------------------------------"
        )


    # --------------------------------------------------------
    # RANDOM BATCH TEST
    # --------------------------------------------------------

    print(
        "\nRANDOMLY SAMPLED BATCHES\n"
    )

    for batch_id in range(
        1,
        6,
    ):

        condition = sample_batch_condition()

        profile, features = (
            create_batch_condition(
                condition=condition
            )
        )

        print(
            f"Batch {batch_id} | "
            f"{condition} | "
            f"RMS disturbance = "
            f"{features['disturbance_rms_mm']:.4f} mm"
        )


    print(
        "\n"
        "=============================================="
    )

    print(
        "BATCH DISTURBANCE ENGINE TEST COMPLETED"
    )

    print(
        "=============================================="
    )