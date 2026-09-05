"""
V3.3 configuration.

The frozen V3.2 system remains unchanged.

V3.3 extends the sequential assembly framework to support
configurable sequence length and future nonlinear/hybrid
experiments.
"""


# ============================================================
# SEQUENCE SETTINGS
# ============================================================

N_COMPONENTS = 10

REFERENCE_N_COMPONENTS = 5


# ============================================================
# PROFILE SETTINGS
# ============================================================

PROFILE_LENGTH_MM = 1000.0

N_PROFILE_POINTS = 101


# ============================================================
# CORRECTION LIMITS
# ============================================================

Z_ADJ_LIMIT_MM = 2.5

THETA_ADJ_LIMIT_DEG = 1.2

LOCATOR_OFFSET_LIMIT_MM = 1.0


# ============================================================
# QUALITY WEIGHTS
# ============================================================

QUALITY_WEIGHT_MEAN_GAP = 0.30

QUALITY_WEIGHT_MAX_GAP = 0.30

QUALITY_WEIGHT_PARALLELISM = 0.30

QUALITY_WEIGHT_RMS = 0.10


# ============================================================
# NONLINEARITY SETTINGS
# ============================================================

NONLINEARITY_STRENGTH = 0.0


NONLINEARITY_LEVELS = [
    0.0,
    0.25,
    0.50,
    0.75,
    1.00,
]


# ============================================================
# RANDOM SEEDS
# ============================================================

BASE_SEED = 20260904

VALIDATION_SEED = 20260905


# ============================================================
# STRUCTURED CANDIDATE SETTINGS
# ============================================================

N_STRUCTURED_CANDIDATES = 20


# ============================================================
# PRINT CONFIGURATION
# ============================================================

if __name__ == "__main__":

    print(
        "\n"
        "============================================================"
    )

    print(
        "V3.3 CONFIGURATION"
    )

    print(
        "============================================================"
    )

    print(
        f"\nPrimary sequence length : {N_COMPONENTS}"
    )

    print(
        f"Reference sequence length: {REFERENCE_N_COMPONENTS}"
    )

    print(
        f"Profile length           : {PROFILE_LENGTH_MM} mm"
    )

    print(
        f"Profile points           : {N_PROFILE_POINTS}"
    )

    print(
        f"Structured candidates    : {N_STRUCTURED_CANDIDATES}"
    )

    print(
        f"Nonlinearity strength    : {NONLINEARITY_STRENGTH}"
    )

    print(
        "\n"
        "============================================================"
    )