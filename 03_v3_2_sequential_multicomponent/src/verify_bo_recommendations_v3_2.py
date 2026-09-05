import os
import numpy as np
import pandas as pd

from sequential_assembly import (
    s,
    PROFILE_LENGTH_MM,
    create_initial_state,
    deviation_offset,
    deviation_tilt,
    deviation_bend,
    deviation_waviness,
    deviation_twist,
    deviation_local_bump,
    correction_profile,
    update_assembly_state,
    calculate_quality_metrics,
)


# ============================================================
# VERSION 3.2
# BAYESIAN OPTIMIZATION RECOMMENDATION VERIFICATION
# ============================================================
#
# PURPOSE
#
# Bayesian Optimization previously optimized the ML surrogate.
#
# That alone is NOT sufficient evidence.
#
# Here we independently:
#
#   1. reconstruct the real sequential assembly state
#   2. reconstruct the incoming component
#   3. apply zero correction to the actual simulator
#   4. apply BO-recommended correction to the actual simulator
#   5. calculate actual resulting quality
#   6. compare ML prediction vs simulator result
#
#
# This answers:
#
# "Did the ML+BO recommendation actually work when applied
#  to the underlying assembly model?"
# ============================================================


# ============================================================
# FILES
# ============================================================

FULL_COMPONENT_FILE = (
    "data/processed/"
    "v3_2_full_component_level_dataset.csv"
)

BO_FILE = (
    "results/tables/"
    "v3_2_bo_smoke_test.csv"
)

OUTPUT_FILE = (
    "results/tables/"
    "v3_2_bo_simulator_verification.csv"
)


required_files = [
    FULL_COMPONENT_FILE,
    BO_FILE,
]


missing_files = [
    file
    for file in required_files
    if not os.path.exists(file)
]


if missing_files:

    print(
        "\nERROR - Missing required files:"
    )

    for file in missing_files:
        print(file)

    raise SystemExit(
        "\nComplete the BO smoke test first."
    )


# ============================================================
# LOAD DATA
# ============================================================

component_df = pd.read_csv(
    FULL_COMPONENT_FILE
)

bo_df = pd.read_csv(
    BO_FILE
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.2 BO RECOMMENDATION SIMULATOR VERIFICATION"
)

print(
    "============================================================"
)


print(
    f"\nBO recommendations to verify : "
    f"{len(bo_df)}"
)


# ============================================================
# RECONSTRUCT COMPONENT PROFILE
# ============================================================

def reconstruct_component_profile(
    row,
):

    profile = np.zeros_like(
        s
    )


    if row["offset_mm"] != 0.0:

        profile += deviation_offset(
            offset_mm=row["offset_mm"]
        )


    if row["tilt_deg"] != 0.0:

        profile += deviation_tilt(
            angle_deg=row["tilt_deg"]
        )


    if row["bend_mm"] != 0.0:

        profile += deviation_bend(
            amplitude_mm=row["bend_mm"]
        )


    if row["waviness_mm"] != 0.0:

        profile += deviation_waviness(
            amplitude_mm=row["waviness_mm"],
            waves=3,
        )


    if row["twist_mm"] != 0.0:

        profile += deviation_twist(
            amplitude_mm=row["twist_mm"]
        )


    if row["local_bump_mm"] != 0.0:

        profile += deviation_local_bump(
            amplitude_mm=row["local_bump_mm"],
            sigma=0.12,
        )


    return profile


# ============================================================
# RECONSTRUCT BATCH / PROCESS DISTURBANCE
# ============================================================

def reconstruct_batch_disturbance(
    row,
):

    offset_profile = np.full_like(
        s,
        row[
            "batch_offset_bias_mm"
        ],
    )


    angular_profile = (
        np.tan(
            np.deg2rad(
                row[
                    "batch_angular_bias_deg"
                ]
            )
        )
        * PROFILE_LENGTH_MM
        * (s - 0.5)
    )


    fixture_sigma = 0.18


    fixture_profile = (
        row[
            "fixture_drift_mm"
        ]
        * np.exp(
            -(
                (s - 0.5) ** 2
            )
            /
            (
                2.0
                * fixture_sigma ** 2
            )
        )
    )


    return (
        offset_profile
        + angular_profile
        + fixture_profile
    )


# ============================================================
# RECONSTRUCT STATE BEFORE TARGET COMPONENT
# ============================================================

def reconstruct_previous_state(
    assembly_id,
    target_component_index,
):
    """
    Recreate the same uncorrected sequential state that was
    used when the ML training dataset was generated.

    Components before the target component are replayed
    without correction.
    """

    assembly_rows = (
        component_df[
            component_df[
                "assembly_id"
            ]
            == assembly_id
        ]
        .sort_values(
            "component_index"
        )
    )


    state = (
        create_initial_state()
    )


    for _, previous_row in (
        assembly_rows.iterrows()
    ):

        previous_component_index = int(
            previous_row[
                "component_index"
            ]
        )


        if (
            previous_component_index
            >= target_component_index
        ):

            break


        previous_component = (
            reconstruct_component_profile(
                previous_row
            )
        )


        previous_batch_disturbance = (
            reconstruct_batch_disturbance(
                previous_row
            )
        )


        state = (
            update_assembly_state(
                previous_state=state,
                component_deviation=(
                    previous_component
                ),
                fixture_drift=(
                    previous_batch_disturbance
                ),
            )
        )


    return state


# ============================================================
# VERIFICATION LOOP
# ============================================================

verification_results = []


for _, bo_row in bo_df.iterrows():

    test_case = int(
        bo_row[
            "test_case"
        ]
    )


    assembly_id = int(
        bo_row[
            "assembly_id"
        ]
    )


    component_index = int(
        bo_row[
            "component_index"
        ]
    )


    # --------------------------------------------------------
    # FIND ORIGINAL COMPONENT
    # --------------------------------------------------------

    target_rows = component_df[
        (
            component_df[
                "assembly_id"
            ]
            == assembly_id
        )
        &
        (
            component_df[
                "component_index"
            ]
            == component_index
        )
    ]


    if len(target_rows) != 1:

        raise ValueError(
            f"\nCould not uniquely identify "
            f"assembly {assembly_id}, "
            f"component {component_index}."
        )


    target_row = (
        target_rows.iloc[0]
    )


    # --------------------------------------------------------
    # RECONSTRUCT PREVIOUS SEQUENTIAL STATE
    # --------------------------------------------------------

    previous_state = (
        reconstruct_previous_state(
            assembly_id=assembly_id,
            target_component_index=(
                component_index
            ),
        )
    )


    # --------------------------------------------------------
    # RECONSTRUCT CURRENT COMPONENT
    # --------------------------------------------------------

    component_profile = (
        reconstruct_component_profile(
            target_row
        )
    )


    batch_disturbance = (
        reconstruct_batch_disturbance(
            target_row
        )
    )


    # ========================================================
    # 1. ACTUAL ZERO-CORRECTION RESULT
    # ========================================================

    zero_state = (
        update_assembly_state(
            previous_state=(
                previous_state
            ),
            component_deviation=(
                component_profile
            ),
            fixture_drift=(
                batch_disturbance
            ),
        )
    )


    zero_metrics = (
        calculate_quality_metrics(
            zero_state
        )
    )


    actual_zero_quality = float(
        zero_metrics[
            "quality_score"
        ]
    )


    # ========================================================
    # 2. APPLY BO RECOMMENDATION
    # ========================================================

    z_adj = float(
        bo_row[
            "z_adj"
        ]
    )


    theta_adj = float(
        bo_row[
            "theta_adj"
        ]
    )


    locator_offset = float(
        bo_row[
            "locator_offset"
        ]
    )


    recommended_correction = (
        correction_profile(
            z_adj_mm=z_adj,
            theta_adj_deg=(
                theta_adj
            ),
            locator_offset_mm=(
                locator_offset
            ),
        )
    )


    bo_state = (
        update_assembly_state(
            previous_state=(
                previous_state
            ),
            component_deviation=(
                component_profile
            ),
            fixture_drift=(
                batch_disturbance
            ),
            correction=(
                recommended_correction
            ),
        )
    )


    bo_metrics = (
        calculate_quality_metrics(
            bo_state
        )
    )


    actual_bo_quality = float(
        bo_metrics[
            "quality_score"
        ]
    )


    # ========================================================
    # COMPARISON
    # ========================================================

    predicted_bo_quality = float(
        bo_row[
            "bo_predicted_quality"
        ]
    )


    surrogate_error = (
        actual_bo_quality
        - predicted_bo_quality
    )


    surrogate_abs_error = abs(
        surrogate_error
    )


    actual_improvement_percent = (
        (
            actual_zero_quality
            - actual_bo_quality
        )
        /
        max(
            abs(
                actual_zero_quality
            ),
            1e-9,
        )
        * 100.0
    )


    recommendation_improved = (
        actual_bo_quality
        <
        actual_zero_quality
    )


    # ========================================================
    # PRINT TEST CASE
    # ========================================================

    print(
        "\n"
        "------------------------------------------------------------"
    )


    print(
        f"TEST CASE {test_case}"
    )


    print(
        "------------------------------------------------------------"
    )


    print(
        f"Assembly ID          : "
        f"{assembly_id}"
    )


    print(
        f"Component index      : "
        f"{component_index}"
    )


    print(
        f"Actual zero Q        : "
        f"{actual_zero_quality:.4f}"
    )


    print(
        f"RF predicted BO Q    : "
        f"{predicted_bo_quality:.4f}"
    )


    print(
        f"Actual BO Q          : "
        f"{actual_bo_quality:.4f}"
    )


    print(
        f"Prediction abs error : "
        f"{surrogate_abs_error:.4f}"
    )


    print(
        f"Actual improvement   : "
        f"{actual_improvement_percent:.2f}%"
    )


    print(
        f"Improved simulator?  : "
        f"{recommendation_improved}"
    )


    # ========================================================
    # STORE
    # ========================================================

    verification_results.append(
        {
            "test_case":
                test_case,

            "assembly_id":
                assembly_id,

            "component_index":
                component_index,

            "actual_zero_quality":
                actual_zero_quality,

            "rf_predicted_bo_quality":
                predicted_bo_quality,

            "actual_bo_quality":
                actual_bo_quality,

            "surrogate_error":
                surrogate_error,

            "surrogate_absolute_error":
                surrogate_abs_error,

            "actual_improvement_percent":
                actual_improvement_percent,

            "recommendation_improved_simulator":
                recommendation_improved,

            "z_adj":
                z_adj,

            "theta_adj":
                theta_adj,

            "locator_offset":
                locator_offset,

            "correction_utilization":
                bo_row[
                    "correction_utilization"
                ],

            "actual_mean_gap":
                bo_metrics[
                    "mean_gap"
                ],

            "actual_max_gap":
                bo_metrics[
                    "max_gap"
                ],

            "actual_parallelism":
                bo_metrics[
                    "parallelism_error"
                ],

            "actual_rms":
                bo_metrics[
                    "rms_deviation"
                ],
        }
    )


# ============================================================
# CREATE RESULT TABLE
# ============================================================

verification_df = pd.DataFrame(
    verification_results
)


os.makedirs(
    "results/tables",
    exist_ok=True,
)


verification_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# OVERALL SUMMARY
# ============================================================

success_rate = (
    verification_df[
        "recommendation_improved_simulator"
    ].mean()
    * 100.0
)


mean_actual_improvement = (
    verification_df[
        "actual_improvement_percent"
    ].mean()
)


mean_prediction_error = (
    verification_df[
        "surrogate_absolute_error"
    ].mean()
)


max_prediction_error = (
    verification_df[
        "surrogate_absolute_error"
    ].max()
)


print(
    "\n"
    "============================================================"
)

print(
    "VERIFICATION SUMMARY"
)

print(
    "============================================================"
)


print(
    f"\nRecommendations tested      : "
    f"{len(verification_df)}"
)


print(
    f"Simulator improvement rate  : "
    f"{success_rate:.2f}%"
)


print(
    f"Mean actual improvement      : "
    f"{mean_actual_improvement:.2f}%"
)


print(
    f"Mean surrogate abs. error    : "
    f"{mean_prediction_error:.4f}"
)


print(
    f"Max surrogate abs. error     : "
    f"{max_prediction_error:.4f}"
)


print(
    "\nSaved:"
)

print(
    OUTPUT_FILE
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.2 BO SIMULATOR VERIFICATION COMPLETED"
)

print(
    "============================================================"
)