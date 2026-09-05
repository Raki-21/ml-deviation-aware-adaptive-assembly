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
# VERSION 3.2 - STRATEGY COMPARISON
# ============================================================
#
# Purpose:
#
# Compare three adaptation levels on the SAME controlled
# 1000-assembly / 5000-component V3.2 dataset:
#
#   1. GLOBAL
#      One common correction policy for the entire dataset.
#
#   2. BATCH-ADAPTIVE
#      One common correction policy for each batch condition.
#
#   3. INDIVIDUAL-SEQUENTIAL
#      Correction calculated for every incoming component,
#      while also considering the current accumulated
#      assembly state.
#
# IMPORTANT:
#
# This is still a RULE-BASED strategy benchmark.
#
# Random Forest + Bayesian Optimization are NOT used yet.
#
# The purpose is to verify the new V3.2 adaptation hierarchy
# before introducing the intelligent recommendation layer.
# ============================================================


# ------------------------------------------------------------
# INPUT DATASET
# ------------------------------------------------------------

INPUT_FILE = (
    "data/processed/"
    "v3_2_full_component_level_dataset.csv"
)


if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        "\nFull V3.2 component dataset not found.\n"
        "Run generate_full_v3_2_dataset.py first."
    )


df = pd.read_csv(
    INPUT_FILE
)


# ------------------------------------------------------------
# CORRECTION LIMITS
# ------------------------------------------------------------
#
# Preserve current V3.1-style limits.
# ------------------------------------------------------------

Z_MIN = -2.5
Z_MAX = 2.5

THETA_MIN = -1.2
THETA_MAX = 1.2

LOCATOR_MIN = -1.0
LOCATOR_MAX = 1.0


# ------------------------------------------------------------
# HELPER: LIMIT CORRECTION TO CAPABILITY
# ------------------------------------------------------------

def clip_correction(
    z_adj,
    theta_adj,
    locator_offset,
):

    z_adj = float(
        np.clip(
            z_adj,
            Z_MIN,
            Z_MAX,
        )
    )

    theta_adj = float(
        np.clip(
            theta_adj,
            THETA_MIN,
            THETA_MAX,
        )
    )

    locator_offset = float(
        np.clip(
            locator_offset,
            LOCATOR_MIN,
            LOCATOR_MAX,
        )
    )

    return (
        z_adj,
        theta_adj,
        locator_offset,
    )


# ------------------------------------------------------------
# RECONSTRUCT COMPONENT PROFILE
# ------------------------------------------------------------

def reconstruct_component_profile(
    row,
):
    """
    Reconstruct the component deviation profile from the
    interpretable features stored in the full dataset.
    """

    profile = np.zeros_like(s)

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


# ------------------------------------------------------------
# RECONSTRUCT BATCH DISTURBANCE
# ------------------------------------------------------------

def reconstruct_batch_disturbance(
    row,
):
    """
    Reconstruct the shared process/batch disturbance from
    the stored batch features.
    """

    offset_profile = np.full_like(
        s,
        row["batch_offset_bias_mm"],
    )


    angular_profile = (
        np.tan(
            np.deg2rad(
                row["batch_angular_bias_deg"]
            )
        )
        * PROFILE_LENGTH_MM
        * (s - 0.5)
    )


    fixture_sigma = 0.18

    fixture_profile = (
        row["fixture_drift_mm"]
        * np.exp(
            -((s - 0.5) ** 2)
            / (2.0 * fixture_sigma ** 2)
        )
    )


    return (
        offset_profile
        + angular_profile
        + fixture_profile
    )


# ============================================================
# GLOBAL CORRECTION POLICY
# ============================================================
#
# One correction derived from the mean behaviour of the
# complete production population.
# ============================================================

GLOBAL_Z = -float(
    df["offset_mm"].mean()
    + df["batch_offset_bias_mm"].mean()
)

GLOBAL_THETA = -float(
    df["tilt_deg"].mean()
    + df["batch_angular_bias_deg"].mean()
)

GLOBAL_LOCATOR = -float(
    df["local_bump_mm"].mean()
    + df["fixture_drift_mm"].mean()
)


(
    GLOBAL_Z,
    GLOBAL_THETA,
    GLOBAL_LOCATOR,
) = clip_correction(
    GLOBAL_Z,
    GLOBAL_THETA,
    GLOBAL_LOCATOR,
)


print(
    "\nGLOBAL CORRECTION POLICY"
)

print(
    f"z_adj          = {GLOBAL_Z:.4f} mm"
)

print(
    f"theta_adj      = {GLOBAL_THETA:.5f} deg"
)

print(
    f"locator_offset = {GLOBAL_LOCATOR:.4f} mm"
)


# ============================================================
# BATCH-ADAPTIVE CORRECTION POLICIES
# ============================================================

batch_policy = (
    df
    .groupby(
        "batch_condition"
    )
    .agg(
        mean_offset=(
            "offset_mm",
            "mean",
        ),

        mean_tilt=(
            "tilt_deg",
            "mean",
        ),

        mean_local_bump=(
            "local_bump_mm",
            "mean",
        ),

        batch_offset=(
            "batch_offset_bias_mm",
            "mean",
        ),

        batch_angle=(
            "batch_angular_bias_deg",
            "mean",
        ),

        fixture_drift=(
            "fixture_drift_mm",
            "mean",
        ),
    )
)


# ------------------------------------------------------------
# STORAGE
# ------------------------------------------------------------

component_results = []
assembly_results = []


# ============================================================
# RUN EVERY ASSEMBLY
# ============================================================

assembly_ids = sorted(
    df[
        "assembly_id"
    ].unique()
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.2 STRATEGY COMPARISON"
)

print(
    "============================================================"
)

print(
    f"\nAssemblies to evaluate: "
    f"{len(assembly_ids)}"
)


for assembly_id in assembly_ids:

    assembly_rows = (
        df[
            df["assembly_id"]
            == assembly_id
        ]
        .sort_values(
            "component_index"
        )
    )


    # --------------------------------------------------------
    # INITIAL STATE FOR ALL THREE STRATEGIES
    # --------------------------------------------------------

    global_state = (
        create_initial_state()
    )

    batch_state = (
        create_initial_state()
    )

    individual_state = (
        create_initial_state()
    )


    batch_condition = (
        assembly_rows.iloc[0][
            "batch_condition"
        ]
    )


    # --------------------------------------------------------
    # GET BATCH POLICY
    # --------------------------------------------------------

    policy = batch_policy.loc[
        batch_condition
    ]


    batch_z = -float(
        policy[
            "mean_offset"
        ]
        + policy[
            "batch_offset"
        ]
    )

    batch_theta = -float(
        policy[
            "mean_tilt"
        ]
        + policy[
            "batch_angle"
        ]
    )

    batch_locator = -float(
        policy[
            "mean_local_bump"
        ]
        + policy[
            "fixture_drift"
        ]
    )


    (
        batch_z,
        batch_theta,
        batch_locator,
    ) = clip_correction(
        batch_z,
        batch_theta,
        batch_locator,
    )


    # ========================================================
    # COMPONENT-BY-COMPONENT ASSEMBLY
    # ========================================================

    for _, row in (
        assembly_rows.iterrows()
    ):

        component_index = int(
            row[
                "component_index"
            ]
        )


        component_profile = (
            reconstruct_component_profile(
                row
            )
        )


        batch_disturbance = (
            reconstruct_batch_disturbance(
                row
            )
        )


        # ====================================================
        # 1. GLOBAL STRATEGY
        # ====================================================

        global_correction = (
            correction_profile(
                z_adj_mm=GLOBAL_Z,
                theta_adj_deg=GLOBAL_THETA,
                locator_offset_mm=GLOBAL_LOCATOR,
            )
        )


        global_state = (
            update_assembly_state(
                previous_state=global_state,
                component_deviation=component_profile,
                fixture_drift=batch_disturbance,
                correction=global_correction,
            )
        )


        global_metrics = (
            calculate_quality_metrics(
                global_state
            )
        )


        # ====================================================
        # 2. BATCH-ADAPTIVE STRATEGY
        # ====================================================

        batch_correction = (
            correction_profile(
                z_adj_mm=batch_z,
                theta_adj_deg=batch_theta,
                locator_offset_mm=batch_locator,
            )
        )


        batch_state = (
            update_assembly_state(
                previous_state=batch_state,
                component_deviation=component_profile,
                fixture_drift=batch_disturbance,
                correction=batch_correction,
            )
        )


        batch_metrics = (
            calculate_quality_metrics(
                batch_state
            )
        )


        # ====================================================
        # 3. INDIVIDUAL-SEQUENTIAL STRATEGY
        # ====================================================
        #
        # Temporary intelligent-looking rule:
        #
        #   - compensate component offset
        #   - compensate component tilt
        #   - compensate local bump
        #   - compensate shared process disturbance
        #   - include current-state feedback
        #
        # This is NOT yet ML/BO.
        # ====================================================


        # --------------------------------------------
        # CURRENT STATE FEEDBACK
        # --------------------------------------------

        state_mean = float(
            np.mean(
                individual_state
            )
        )


        state_angle_effect_mm = float(
            individual_state[-1]
            - individual_state[0]
        )


        state_angle_deg = float(
            np.rad2deg(
                np.arctan(
                    state_angle_effect_mm
                    / PROFILE_LENGTH_MM
                )
            )
        )


        # --------------------------------------------
        # INDIVIDUAL CORRECTION REQUEST
        # --------------------------------------------

        individual_z = -(
            row["offset_mm"]
            + row["batch_offset_bias_mm"]
            + 0.50 * state_mean
        )


        individual_theta = -(
            row["tilt_deg"]
            + row["batch_angular_bias_deg"]
            + 0.50 * state_angle_deg
        )


        individual_locator = -(
            row["local_bump_mm"]
            + row["fixture_drift_mm"]
        )


        (
            individual_z,
            individual_theta,
            individual_locator,
        ) = clip_correction(
            individual_z,
            individual_theta,
            individual_locator,
        )


        individual_correction = (
            correction_profile(
                z_adj_mm=individual_z,
                theta_adj_deg=individual_theta,
                locator_offset_mm=individual_locator,
            )
        )


        individual_state = (
            update_assembly_state(
                previous_state=individual_state,
                component_deviation=component_profile,
                fixture_drift=batch_disturbance,
                correction=individual_correction,
            )
        )


        individual_metrics = (
            calculate_quality_metrics(
                individual_state
            )
        )


        # ====================================================
        # CORRECTION UTILIZATION
        # ====================================================

        z_util = abs(
            individual_z
        ) / Z_MAX


        theta_util = abs(
            individual_theta
        ) / THETA_MAX


        locator_util = abs(
            individual_locator
        ) / LOCATOR_MAX


        max_util = max(
            z_util,
            theta_util,
            locator_util,
        )


        # ====================================================
        # COMPONENT-LEVEL OUTPUT
        # ====================================================

        component_results.append(
            {
                "assembly_id":
                    assembly_id,

                "batch_condition":
                    batch_condition,

                "component_index":
                    component_index,

                "scenario_type":
                    row[
                        "scenario_type"
                    ],

                "severity":
                    row[
                        "severity"
                    ],

                # Global
                "global_quality":
                    global_metrics[
                        "quality_score"
                    ],

                "global_mean_gap":
                    global_metrics[
                        "mean_gap"
                    ],

                "global_parallelism":
                    global_metrics[
                        "parallelism_error"
                    ],

                # Batch
                "batch_quality":
                    batch_metrics[
                        "quality_score"
                    ],

                "batch_mean_gap":
                    batch_metrics[
                        "mean_gap"
                    ],

                "batch_parallelism":
                    batch_metrics[
                        "parallelism_error"
                    ],

                # Individual
                "individual_quality":
                    individual_metrics[
                        "quality_score"
                    ],

                "individual_mean_gap":
                    individual_metrics[
                        "mean_gap"
                    ],

                "individual_parallelism":
                    individual_metrics[
                        "parallelism_error"
                    ],

                # Individual corrections
                "individual_z_adj":
                    individual_z,

                "individual_theta_adj":
                    individual_theta,

                "individual_locator_offset":
                    individual_locator,

                "correction_utilization":
                    max_util,
            }
        )


    # ========================================================
    # FINAL ASSEMBLY METRICS
    # ========================================================

    final_global = (
        calculate_quality_metrics(
            global_state
        )
    )


    final_batch = (
        calculate_quality_metrics(
            batch_state
        )
    )


    final_individual = (
        calculate_quality_metrics(
            individual_state
        )
    )


    assembly_results.append(
        {
            "assembly_id":
                assembly_id,

            "batch_condition":
                batch_condition,

            # Final global
            "global_final_quality":
                final_global[
                    "quality_score"
                ],

            "global_final_mean_gap":
                final_global[
                    "mean_gap"
                ],

            "global_final_max_gap":
                final_global[
                    "max_gap"
                ],

            "global_final_parallelism":
                final_global[
                    "parallelism_error"
                ],

            # Final batch
            "batch_final_quality":
                final_batch[
                    "quality_score"
                ],

            "batch_final_mean_gap":
                final_batch[
                    "mean_gap"
                ],

            "batch_final_max_gap":
                final_batch[
                    "max_gap"
                ],

            "batch_final_parallelism":
                final_batch[
                    "parallelism_error"
                ],

            # Final individual
            "individual_final_quality":
                final_individual[
                    "quality_score"
                ],

            "individual_final_mean_gap":
                final_individual[
                    "mean_gap"
                ],

            "individual_final_max_gap":
                final_individual[
                    "max_gap"
                ],

            "individual_final_parallelism":
                final_individual[
                    "parallelism_error"
                ],
        }
    )


# ============================================================
# CREATE DATAFRAMES
# ============================================================

component_result_df = (
    pd.DataFrame(
        component_results
    )
)


assembly_result_df = (
    pd.DataFrame(
        assembly_results
    )
)


# ============================================================
# STRATEGY SUMMARY
# ============================================================

strategy_summary = pd.DataFrame(
    {
        "strategy": [
            "global",
            "batch_adaptive",
            "individual_sequential",
        ],

        "mean_final_quality": [
            assembly_result_df[
                "global_final_quality"
            ].mean(),

            assembly_result_df[
                "batch_final_quality"
            ].mean(),

            assembly_result_df[
                "individual_final_quality"
            ].mean(),
        ],

        "median_final_quality": [
            assembly_result_df[
                "global_final_quality"
            ].median(),

            assembly_result_df[
                "batch_final_quality"
            ].median(),

            assembly_result_df[
                "individual_final_quality"
            ].median(),
        ],

        "std_final_quality": [
            assembly_result_df[
                "global_final_quality"
            ].std(),

            assembly_result_df[
                "batch_final_quality"
            ].std(),

            assembly_result_df[
                "individual_final_quality"
            ].std(),
        ],

        "mean_parallelism": [
            assembly_result_df[
                "global_final_parallelism"
            ].mean(),

            assembly_result_df[
                "batch_final_parallelism"
            ].mean(),

            assembly_result_df[
                "individual_final_parallelism"
            ].mean(),
        ],
    }
)


# ============================================================
# IMPROVEMENT CALCULATIONS
# ============================================================

global_mean = float(
    strategy_summary.loc[
        strategy_summary[
            "strategy"
        ] == "global",
        "mean_final_quality",
    ].iloc[0]
)


batch_mean = float(
    strategy_summary.loc[
        strategy_summary[
            "strategy"
        ] == "batch_adaptive",
        "mean_final_quality",
    ].iloc[0]
)


individual_mean = float(
    strategy_summary.loc[
        strategy_summary[
            "strategy"
        ] == "individual_sequential",
        "mean_final_quality",
    ].iloc[0]
)


batch_improvement = (
    (
        global_mean
        - batch_mean
    )
    / global_mean
    * 100.0
)


individual_improvement = (
    (
        global_mean
        - individual_mean
    )
    / global_mean
    * 100.0
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

os.makedirs(
    "results/tables",
    exist_ok=True,
)


component_result_df.to_csv(
    "data/processed/"
    "v3_2_strategy_component_results.csv",
    index=False,
)


assembly_result_df.to_csv(
    "data/processed/"
    "v3_2_strategy_assembly_results.csv",
    index=False,
)


strategy_summary.to_csv(
    "results/tables/"
    "v3_2_strategy_summary.csv",
    index=False,
)


# ============================================================
# TERMINAL OUTPUT
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "STRATEGY COMPARISON RESULTS"
)

print(
    "============================================================\n"
)


print(
    strategy_summary.round(4)
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "IMPROVEMENT VS GLOBAL"
)

print(
    "------------------------------------------------------------"
)


print(
    f"Batch-adaptive improvement      : "
    f"{batch_improvement:.2f}%"
)


print(
    f"Individual-sequential improvement: "
    f"{individual_improvement:.2f}%"
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "INDIVIDUAL CORRECTION UTILIZATION"
)

print(
    "------------------------------------------------------------"
)


print(
    component_result_df[
        "correction_utilization"
    ]
    .describe(
        percentiles=[
            0.50,
            0.90,
            0.95,
            0.99,
        ]
    )
    .round(4)
)


print(
    "\nFiles saved:"
)

print(
    "data/processed/"
    "v3_2_strategy_component_results.csv"
)

print(
    "data/processed/"
    "v3_2_strategy_assembly_results.csv"
)

print(
    "results/tables/"
    "v3_2_strategy_summary.csv"
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.2 STRATEGY COMPARISON COMPLETED"
)

print(
    "============================================================"
)