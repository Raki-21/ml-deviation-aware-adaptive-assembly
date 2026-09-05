"""
V3.3 linear reproduction validation.

Purpose
-------
Verify that the new V3.3 assembly model reproduces the frozen
V3.2 physical model when nonlinearity_strength = 0.0.

This is a mandatory provenance check before nonlinear physics
is activated.

The frozen V3.2 source is loaded read-only.
No V3.2 files or results are modified.
"""

from pathlib import Path
import importlib.util

import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

SCRIPT_DIR = Path(
    __file__
).resolve().parent

PROJECT_ROOT = (
    SCRIPT_DIR
    .parents[1]
)

V32_MODEL_FILE = (
    PROJECT_ROOT
    / "03_v3_2_sequential_multicomponent"
    / "src"
    / "sequential_assembly.py"
)


# ============================================================
# LOAD FROZEN V3.2 MODEL
# ============================================================

if not V32_MODEL_FILE.exists():

    raise FileNotFoundError(
        f"Frozen V3.2 model not found:\n{V32_MODEL_FILE}"
    )


spec = importlib.util.spec_from_file_location(
    "v32_sequential_assembly",
    V32_MODEL_FILE,
)

v32 = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    v32
)


# ============================================================
# LOAD V3.3 MODEL
# ============================================================

from nonlinear_assembly_model import (
    create_initial_state,
    deviation_offset,
    deviation_tilt,
    deviation_bend,
    deviation_waviness,
    deviation_twist,
    correction_profile_linear,
    correction_profile_nonlinear,
    update_assembly_state,
    calculate_quality_metrics,
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 20260904

N_RANDOM_CASES = 200

N_SEQUENTIAL_ASSEMBLIES = 100

REFERENCE_SEQUENCE_LENGTH = 5

ABS_TOLERANCE = 1e-12


rng = np.random.default_rng(
    RANDOM_SEED
)


# ============================================================
# STORAGE
# ============================================================

checks = []


def record_check(
    name,
    difference,
    tolerance=ABS_TOLERANCE,
):

    passed = bool(
        difference
        <=
        tolerance
    )

    checks.append(
        {
            "check":
                name,

            "difference":
                float(
                    difference
                ),

            "tolerance":
                float(
                    tolerance
                ),

            "passed":
                passed,
        }
    )


def max_abs_difference(
    a,
    b,
):

    return float(
        np.max(
            np.abs(
                np.asarray(
                    a,
                    dtype=float,
                )
                -
                np.asarray(
                    b,
                    dtype=float,
                )
            )
        )
    )


# ============================================================
# START
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "V3.3 LINEAR REPRODUCTION VALIDATION"
)

print(
    "============================================================"
)

print(
    f"\nFrozen V3.2 source:\n{V32_MODEL_FILE}"
)

print(
    f"\nRandom primitive cases : {N_RANDOM_CASES}"
)

print(
    f"Sequential assemblies  : {N_SEQUENTIAL_ASSEMBLIES}"
)

print(
    f"Reference sequence K   : {REFERENCE_SEQUENCE_LENGTH}"
)

print(
    f"Absolute tolerance     : {ABS_TOLERANCE:.1e}"
)


# ============================================================
# 1. INITIAL STATE
# ============================================================

v32_initial = (
    v32.create_initial_state()
)

v33_initial = (
    create_initial_state()
)


initial_difference = (
    max_abs_difference(
        v32_initial,
        v33_initial,
    )
)


record_check(
    "initial_state",
    initial_difference,
)


# ============================================================
# 2. RANDOM PRIMITIVE FUNCTION COMPARISONS
# ============================================================

max_offset_difference = 0.0

max_tilt_difference = 0.0

max_bend_difference = 0.0

max_waviness_difference = 0.0

max_twist_difference = 0.0

max_correction_difference = 0.0

max_zero_nonlinearity_difference = 0.0

max_state_update_difference = 0.0

max_quality_difference = 0.0


for case_index in range(
    N_RANDOM_CASES
):

    # --------------------------------------------------------
    # Random deviation parameters
    # --------------------------------------------------------

    offset_mm = rng.uniform(
        -1.5,
        1.5,
    )

    tilt_deg = rng.uniform(
        -0.20,
        0.20,
    )

    bend_mm = rng.uniform(
        -1.0,
        1.0,
    )

    waviness_mm = rng.uniform(
        -0.60,
        0.60,
    )

    twist_mm = rng.uniform(
        -0.60,
        0.60,
    )

    waves = int(
        rng.integers(
            1,
            5,
        )
    )

    # --------------------------------------------------------
    # Random correction parameters
    # --------------------------------------------------------

    z_adj_mm = rng.uniform(
        -2.5,
        2.5,
    )

    theta_adj_deg = rng.uniform(
        -1.2,
        1.2,
    )

    locator_offset_mm = rng.uniform(
        -1.0,
        1.0,
    )

    # --------------------------------------------------------
    # Compare deviation functions
    # --------------------------------------------------------

    v32_offset = (
        v32.deviation_offset(
            offset_mm
        )
    )

    v33_offset = (
        deviation_offset(
            offset_mm
        )
    )

    max_offset_difference = max(
        max_offset_difference,
        max_abs_difference(
            v32_offset,
            v33_offset,
        ),
    )


    v32_tilt = (
        v32.deviation_tilt(
            tilt_deg
        )
    )

    v33_tilt = (
        deviation_tilt(
            tilt_deg
        )
    )

    max_tilt_difference = max(
        max_tilt_difference,
        max_abs_difference(
            v32_tilt,
            v33_tilt,
        ),
    )


    v32_bend = (
        v32.deviation_bend(
            bend_mm
        )
    )

    v33_bend = (
        deviation_bend(
            bend_mm
        )
    )

    max_bend_difference = max(
        max_bend_difference,
        max_abs_difference(
            v32_bend,
            v33_bend,
        ),
    )


    v32_waviness = (
        v32.deviation_waviness(
            waviness_mm,
            waves=waves,
        )
    )

    v33_waviness = (
        deviation_waviness(
            waviness_mm,
            waves=waves,
        )
    )

    max_waviness_difference = max(
        max_waviness_difference,
        max_abs_difference(
            v32_waviness,
            v33_waviness,
        ),
    )


    v32_twist = (
        v32.deviation_twist(
            twist_mm
        )
    )

    v33_twist = (
        deviation_twist(
            twist_mm
        )
    )

    max_twist_difference = max(
        max_twist_difference,
        max_abs_difference(
            v32_twist,
            v33_twist,
        ),
    )

    # --------------------------------------------------------
    # Compare linear correction
    # --------------------------------------------------------

    v32_correction = (
        v32.correction_profile(
            z_adj_mm=z_adj_mm,
            theta_adj_deg=theta_adj_deg,
            locator_offset_mm=locator_offset_mm,
        )
    )

    v33_correction = (
        correction_profile_linear(
            z_adj_mm=z_adj_mm,
            theta_adj_deg=theta_adj_deg,
            locator_offset_mm=locator_offset_mm,
        )
    )

    correction_difference = (
        max_abs_difference(
            v32_correction,
            v33_correction,
        )
    )

    max_correction_difference = max(
        max_correction_difference,
        correction_difference,
    )

    # --------------------------------------------------------
    # Compare zero-nonlinearity correction
    # --------------------------------------------------------

    random_previous_state = rng.normal(
        loc=0.0,
        scale=0.50,
        size=len(
            v32_initial
        ),
    )

    v33_zero_nonlinear = (
        correction_profile_nonlinear(
            previous_state=random_previous_state,
            z_adj_mm=z_adj_mm,
            theta_adj_deg=theta_adj_deg,
            locator_offset_mm=locator_offset_mm,
            nonlinearity_strength=0.0,
        )
    )

    zero_nonlinearity_difference = (
        max_abs_difference(
            v32_correction,
            v33_zero_nonlinear,
        )
    )

    max_zero_nonlinearity_difference = max(
        max_zero_nonlinearity_difference,
        zero_nonlinearity_difference,
    )

    # --------------------------------------------------------
    # Compare state update
    # --------------------------------------------------------

    component_deviation = (
        v32_offset
        +
        v32_tilt
        +
        v32_bend
        +
        v32_waviness
        +
        v32_twist
    )

    fixture_drift = rng.normal(
        loc=0.0,
        scale=0.02,
        size=len(
            v32_initial
        ),
    )

    v32_state = (
        v32.update_assembly_state(
            previous_state=random_previous_state,
            component_deviation=component_deviation,
            correction=v32_correction,
            fixture_drift=fixture_drift,
        )
    )

    v33_state = (
        update_assembly_state(
            previous_state=random_previous_state,
            component_deviation=component_deviation,
            correction=v33_correction,
            fixture_drift=fixture_drift,
        )
    )

    state_difference = (
        max_abs_difference(
            v32_state,
            v33_state,
        )
    )

    max_state_update_difference = max(
        max_state_update_difference,
        state_difference,
    )

    # --------------------------------------------------------
    # Compare quality metrics
    # --------------------------------------------------------

    v32_quality = (
        v32.calculate_quality_metrics(
            v32_state
        )
    )

    v33_quality = (
        calculate_quality_metrics(
            v33_state
        )
    )

    quality_keys = [
        "mean_gap",
        "max_gap",
        "parallelism_error",
        "rms_deviation",
        "quality_score",
    ]

    case_quality_difference = max(
        abs(
            float(
                v32_quality[
                    key
                ]
            )
            -
            float(
                v33_quality[
                    key
                ]
            )
        )
        for key in quality_keys
    )

    max_quality_difference = max(
        max_quality_difference,
        case_quality_difference,
    )


record_check(
    "deviation_offset",
    max_offset_difference,
)

record_check(
    "deviation_tilt",
    max_tilt_difference,
)

record_check(
    "deviation_bend",
    max_bend_difference,
)

record_check(
    "deviation_waviness",
    max_waviness_difference,
)

record_check(
    "deviation_twist",
    max_twist_difference,
)

record_check(
    "linear_correction_profile",
    max_correction_difference,
)

record_check(
    "zero_nonlinearity_correction",
    max_zero_nonlinearity_difference,
)

record_check(
    "state_update",
    max_state_update_difference,
)

record_check(
    "quality_metrics",
    max_quality_difference,
)


# ============================================================
# 3. FIVE-STAGE SEQUENTIAL REPRODUCTION
# ============================================================

max_sequential_state_difference = 0.0

max_sequential_quality_difference = 0.0


for assembly_index in range(
    N_SEQUENTIAL_ASSEMBLIES
):

    v32_state = (
        v32.create_initial_state()
    )

    v33_state = (
        create_initial_state()
    )

    for component_index in range(
        REFERENCE_SEQUENCE_LENGTH
    ):

        # ----------------------------------------------------
        # Generate identical component geometry
        # ----------------------------------------------------

        offset_mm = rng.uniform(
            -0.60,
            0.60,
        )

        tilt_deg = rng.uniform(
            -0.10,
            0.10,
        )

        bend_mm = rng.uniform(
            -0.40,
            0.40,
        )

        waviness_mm = rng.uniform(
            -0.25,
            0.25,
        )

        twist_mm = rng.uniform(
            -0.20,
            0.20,
        )

        component_v32 = (
            v32.deviation_offset(
                offset_mm
            )
            +
            v32.deviation_tilt(
                tilt_deg
            )
            +
            v32.deviation_bend(
                bend_mm
            )
            +
            v32.deviation_waviness(
                waviness_mm,
                waves=3,
            )
            +
            v32.deviation_twist(
                twist_mm
            )
        )

        component_v33 = (
            deviation_offset(
                offset_mm
            )
            +
            deviation_tilt(
                tilt_deg
            )
            +
            deviation_bend(
                bend_mm
            )
            +
            deviation_waviness(
                waviness_mm,
                waves=3,
            )
            +
            deviation_twist(
                twist_mm
            )
        )

        # ----------------------------------------------------
        # Identical fixture/process disturbance
        # ----------------------------------------------------

        fixture_drift = rng.normal(
            loc=0.0,
            scale=0.01,
            size=len(
                v32_state
            ),
        )

        # ----------------------------------------------------
        # Identical correction parameters
        # ----------------------------------------------------

        z_adj_mm = rng.uniform(
            -1.0,
            1.0,
        )

        theta_adj_deg = rng.uniform(
            -0.30,
            0.30,
        )

        locator_offset_mm = rng.uniform(
            -0.50,
            0.50,
        )

        v32_correction = (
            v32.correction_profile(
                z_adj_mm=z_adj_mm,
                theta_adj_deg=theta_adj_deg,
                locator_offset_mm=locator_offset_mm,
            )
        )

        v33_correction = (
            correction_profile_nonlinear(
                previous_state=v33_state,
                z_adj_mm=z_adj_mm,
                theta_adj_deg=theta_adj_deg,
                locator_offset_mm=locator_offset_mm,
                nonlinearity_strength=0.0,
            )
        )

        # ----------------------------------------------------
        # Update states
        # ----------------------------------------------------

        v32_state = (
            v32.update_assembly_state(
                previous_state=v32_state,
                component_deviation=component_v32,
                correction=v32_correction,
                fixture_drift=fixture_drift,
            )
        )

        v33_state = (
            update_assembly_state(
                previous_state=v33_state,
                component_deviation=component_v33,
                correction=v33_correction,
                fixture_drift=fixture_drift,
            )
        )

        sequential_state_difference = (
            max_abs_difference(
                v32_state,
                v33_state,
            )
        )

        max_sequential_state_difference = max(
            max_sequential_state_difference,
            sequential_state_difference,
        )

        # ----------------------------------------------------
        # Compare stage quality
        # ----------------------------------------------------

        q32 = (
            v32.calculate_quality_metrics(
                v32_state
            )
        )

        q33 = (
            calculate_quality_metrics(
                v33_state
            )
        )

        stage_quality_difference = max(
            abs(
                float(
                    q32[
                        key
                    ]
                )
                -
                float(
                    q33[
                        key
                    ]
                )
            )
            for key in [
                "mean_gap",
                "max_gap",
                "parallelism_error",
                "rms_deviation",
                "quality_score",
            ]
        )

        max_sequential_quality_difference = max(
            max_sequential_quality_difference,
            stage_quality_difference,
        )


record_check(
    "five_stage_sequential_state",
    max_sequential_state_difference,
)

record_check(
    "five_stage_sequential_quality",
    max_sequential_quality_difference,
)


# ============================================================
# 4. PRINT CHECK RESULTS
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "REPRODUCTION CHECK RESULTS"
)

print(
    "============================================================"
)


for check in checks:

    status = (
        "PASS"
        if check[
            "passed"
        ]
        else
        "FAIL"
    )

    print(
        f"\n[{status}] "
        f"{check['check']}"
    )

    print(
        f"  max difference : "
        f"{check['difference']:.16e}"
    )

    print(
        f"  tolerance      : "
        f"{check['tolerance']:.1e}"
    )


# ============================================================
# 5. FINAL RESULT
# ============================================================

passed = sum(
    int(
        check[
            "passed"
        ]
    )
    for check in checks
)

total = len(
    checks
)

failed = (
    total
    -
    passed
)


print(
    "\n"
    "============================================================"
)

print(
    f"Total checks : {total}"
)

print(
    f"Passed       : {passed}"
)

print(
    f"Failed       : {failed}"
)


if failed == 0:

    print(
        "\n"
        "V3.3 LINEAR REPRODUCTION: PASS"
    )

    print(
        "\n"
        "The V3.3 model reproduces the frozen V3.2 "
        "physical model at zero nonlinearity."
    )

else:

    print(
        "\n"
        "V3.3 LINEAR REPRODUCTION: FAIL"
    )

    print(
        "\n"
        "Do not activate nonlinear physics until "
        "the reproduction mismatch is resolved."
    )


print(
    "\n"
    "============================================================"
)