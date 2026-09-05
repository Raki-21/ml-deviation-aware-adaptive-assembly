import os
import numpy as np
import pandas as pd

from sequential_assembly import (
    create_initial_state,
    update_assembly_state,
    calculate_quality_metrics,
)

from deviation_engine import (
    generate_deviation_profile,
)

from batch_disturbance import (
    create_batch_condition,
    SUPPORTED_BATCH_CONDITIONS,
)


# ============================================================
# VERSION 3.2 - FULL CONTROLLED PRODUCTION DATASET GENERATOR
# ============================================================
#
# Purpose:
# Generate the main controlled V3.2 dataset with guaranteed
# coverage of all batch/process conditions.
#
# Dataset structure:
#
#     6 batch/process conditions
#     ~1000 completed assemblies total
#     5 sequential components per assembly
#     ~5000 component-level decisions
#
# This remains the UNCORRECTED / BASELINE production dataset.
# Adaptive strategies, Random Forest and Bayesian Optimization
# are added after this dataset is verified.
# ============================================================


RNG = np.random.default_rng(20260824)


# ------------------------------------------------------------
# CORE EXPERIMENT SETTINGS
# ------------------------------------------------------------

N_COMPONENTS = 5

TARGET_TOTAL_ASSEMBLIES = 1000

BATCH_CONDITIONS = [
    "normal",
    "offset_drift",
    "angular_drift",
    "fixture_drift",
    "high_variation",
    "disturbed",
]


# ------------------------------------------------------------
# DISTRIBUTE APPROX. 1000 ASSEMBLIES ACROSS CONDITIONS
# ------------------------------------------------------------

base_count = (
    TARGET_TOTAL_ASSEMBLIES
    // len(BATCH_CONDITIONS)
)

remainder = (
    TARGET_TOTAL_ASSEMBLIES
    % len(BATCH_CONDITIONS)
)

assemblies_per_condition = {}

for i, condition in enumerate(BATCH_CONDITIONS):

    assemblies_per_condition[
        condition
    ] = (
        base_count
        + (1 if i < remainder else 0)
    )


# ------------------------------------------------------------
# CONTROLLED SEVERITY PATTERN
# ------------------------------------------------------------
#
# We deliberately ensure all severity levels are present.
#
# The pattern is repeated instead of relying entirely
# on random sampling.
# ------------------------------------------------------------

SEVERITY_PATTERN = [
    "low",
    "medium",
    "medium",
    "high",
    "low",
    "medium",
    "high",
    "medium",
    "high",
    "extreme",
]


def choose_controlled_severity(
    global_component_counter,
    variation_multiplier,
):
    """
    Deterministically cycle through severity levels.

    For high-variation / disturbed batches, the severity
    is shifted upward more often.
    """

    base_severity = SEVERITY_PATTERN[
        global_component_counter
        % len(SEVERITY_PATTERN)
    ]

    if variation_multiplier <= 1.0:
        return base_severity

    # Increase difficulty for high-variation batches.
    if base_severity == "low":
        return "medium"

    if base_severity == "medium":
        return "high"

    if base_severity == "high":
        # Occasionally push high cases to extreme.
        if (
            global_component_counter
            % 3
            == 0
        ):
            return "extreme"

        return "high"

    return "extreme"


# ------------------------------------------------------------
# STORAGE
# ------------------------------------------------------------

component_records = []
assembly_records = []

global_component_counter = 0
global_assembly_id = 0


print(
    "\n"
    "============================================================"
)

print(
    "V3.2 FULL CONTROLLED DATASET GENERATOR"
)

print(
    "============================================================\n"
)


# ============================================================
# LOOP THROUGH EVERY CONTROLLED BATCH CONDITION
# ============================================================

for batch_id, condition in enumerate(
    BATCH_CONDITIONS,
    start=1,
):

    n_assemblies = (
        assemblies_per_condition[
            condition
        ]
    )

    (
        batch_disturbance_profile,
        batch_features,
    ) = create_batch_condition(
        condition=condition,
        rng=RNG,
    )


    print(
        f"Batch condition {batch_id}/"
        f"{len(BATCH_CONDITIONS)} | "
        f"{condition} | "
        f"Assemblies = {n_assemblies}"
    )


    # ========================================================
    # LOOP THROUGH FINISHED ASSEMBLIES
    # ========================================================

    for local_assembly_number in range(
        1,
        n_assemblies + 1,
    ):

        global_assembly_id += 1

        state = create_initial_state()

        severity_sequence = []
        scenario_sequence = []


        # ====================================================
        # SEQUENTIAL COMPONENT LOOP
        # ====================================================

        for component_index in range(
            1,
            N_COMPONENTS + 1,
        ):

            global_component_counter += 1


            # ------------------------------------------------
            # CONTROLLED SEVERITY
            # ------------------------------------------------

            severity = choose_controlled_severity(
                global_component_counter=(
                    global_component_counter
                ),
                variation_multiplier=(
                    batch_features[
                        "variation_multiplier"
                    ]
                ),
            )


            # ------------------------------------------------
            # UNIQUE COMPONENT DEVIATION
            # ------------------------------------------------

            (
                component_profile,
                component_features,
            ) = generate_deviation_profile(
                severity=severity,
                rng=RNG,
            )


            severity_sequence.append(
                severity
            )

            scenario_sequence.append(
                component_features[
                    "scenario_type"
                ]
            )


            # ------------------------------------------------
            # STATE BEFORE COMPONENT
            # ------------------------------------------------

            metrics_before = (
                calculate_quality_metrics(
                    state
                )
            )


            # ------------------------------------------------
            # APPLY COMPONENT + SHARED BATCH DISTURBANCE
            # ------------------------------------------------

            state = update_assembly_state(
                previous_state=state,
                component_deviation=(
                    component_profile
                ),
                fixture_drift=(
                    batch_disturbance_profile
                ),
            )


            # ------------------------------------------------
            # STATE AFTER COMPONENT
            # ------------------------------------------------

            metrics_after = (
                calculate_quality_metrics(
                    state
                )
            )


            # ------------------------------------------------
            # COMPONENT-LEVEL RECORD
            # ------------------------------------------------

            component_records.append(
                {
                    # Identification
                    "batch_id":
                        batch_id,

                    "batch_condition":
                        condition,

                    "assembly_id":
                        global_assembly_id,

                    "component_index":
                        component_index,

                    # Batch/process variables
                    "batch_offset_bias_mm":
                        batch_features[
                            "batch_offset_bias_mm"
                        ],

                    "batch_angular_bias_deg":
                        batch_features[
                            "batch_angular_bias_deg"
                        ],

                    "fixture_drift_mm":
                        batch_features[
                            "fixture_drift_mm"
                        ],

                    "variation_multiplier":
                        batch_features[
                            "variation_multiplier"
                        ],

                    # Component deviation class
                    "scenario_type":
                        component_features[
                            "scenario_type"
                        ],

                    "severity":
                        severity,

                    "active_modes":
                        component_features[
                            "active_modes"
                        ],

                    "n_active_modes":
                        component_features[
                            "n_active_modes"
                        ],

                    # Component deviation values
                    "offset_mm":
                        component_features[
                            "offset_mm"
                        ],

                    "tilt_deg":
                        component_features[
                            "tilt_deg"
                        ],

                    "bend_mm":
                        component_features[
                            "bend_mm"
                        ],

                    "waviness_mm":
                        component_features[
                            "waviness_mm"
                        ],

                    "twist_mm":
                        component_features[
                            "twist_mm"
                        ],

                    "local_bump_mm":
                        component_features[
                            "local_bump_mm"
                        ],

                    # Component-profile descriptors
                    "component_profile_mean_mm":
                        component_features[
                            "profile_mean_mm"
                        ],

                    "component_profile_min_mm":
                        component_features[
                            "profile_min_mm"
                        ],

                    "component_profile_max_mm":
                        component_features[
                            "profile_max_mm"
                        ],

                    "component_profile_rms_mm":
                        component_features[
                            "profile_rms_mm"
                        ],

                    "component_parallelism_mm":
                        component_features[
                            "profile_parallelism_mm"
                        ],

                    # Previous assembly state
                    "state_before_mean_gap":
                        metrics_before[
                            "mean_gap"
                        ],

                    "state_before_max_gap":
                        metrics_before[
                            "max_gap"
                        ],

                    "state_before_parallelism":
                        metrics_before[
                            "parallelism_error"
                        ],

                    "state_before_rms":
                        metrics_before[
                            "rms_deviation"
                        ],

                    "state_before_quality":
                        metrics_before[
                            "quality_score"
                        ],

                    # Updated assembly state
                    "state_after_mean_gap":
                        metrics_after[
                            "mean_gap"
                        ],

                    "state_after_max_gap":
                        metrics_after[
                            "max_gap"
                        ],

                    "state_after_parallelism":
                        metrics_after[
                            "parallelism_error"
                        ],

                    "state_after_rms":
                        metrics_after[
                            "rms_deviation"
                        ],

                    "state_after_quality":
                        metrics_after[
                            "quality_score"
                        ],

                    # Incremental effect
                    "quality_change":
                        (
                            metrics_after[
                                "quality_score"
                            ]
                            -
                            metrics_before[
                                "quality_score"
                            ]
                        ),
                }
            )


        # ====================================================
        # FINAL ASSEMBLY RECORD
        # ====================================================

        final_metrics = (
            calculate_quality_metrics(
                state
            )
        )


        assembly_records.append(
            {
                "batch_id":
                    batch_id,

                "batch_condition":
                    condition,

                "assembly_id":
                    global_assembly_id,

                "n_components":
                    N_COMPONENTS,

                "variation_multiplier":
                    batch_features[
                        "variation_multiplier"
                    ],

                "batch_offset_bias_mm":
                    batch_features[
                        "batch_offset_bias_mm"
                    ],

                "batch_angular_bias_deg":
                    batch_features[
                        "batch_angular_bias_deg"
                    ],

                "fixture_drift_mm":
                    batch_features[
                        "fixture_drift_mm"
                    ],

                "severity_sequence":
                    "+".join(
                        severity_sequence
                    ),

                "scenario_sequence":
                    "+".join(
                        scenario_sequence
                    ),

                "final_mean_gap":
                    final_metrics[
                        "mean_gap"
                    ],

                "final_max_gap":
                    final_metrics[
                        "max_gap"
                    ],

                "final_parallelism":
                    final_metrics[
                        "parallelism_error"
                    ],

                "final_rms":
                    final_metrics[
                        "rms_deviation"
                    ],

                "final_quality_score":
                    final_metrics[
                        "quality_score"
                    ],
            }
        )


# ============================================================
# CREATE DATAFRAMES
# ============================================================

component_df = pd.DataFrame(
    component_records
)

assembly_df = pd.DataFrame(
    assembly_records
)


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

os.makedirs(
    "data/processed",
    exist_ok=True,
)

os.makedirs(
    "results/tables",
    exist_ok=True,
)


# ============================================================
# SAVE MAIN DATASETS
# ============================================================

COMPONENT_OUTPUT = (
    "data/processed/"
    "v3_2_full_component_level_dataset.csv"
)

ASSEMBLY_OUTPUT = (
    "data/processed/"
    "v3_2_full_assembly_level_dataset.csv"
)


component_df.to_csv(
    COMPONENT_OUTPUT,
    index=False,
)

assembly_df.to_csv(
    ASSEMBLY_OUTPUT,
    index=False,
)


# ============================================================
# CREATE SUMMARY TABLES
# ============================================================

batch_summary = (
    assembly_df
    .groupby(
        "batch_condition"
    )
    .agg(
        assemblies=(
            "assembly_id",
            "count",
        ),

        mean_quality=(
            "final_quality_score",
            "mean",
        ),

        median_quality=(
            "final_quality_score",
            "median",
        ),

        std_quality=(
            "final_quality_score",
            "std",
        ),

        mean_gap=(
            "final_mean_gap",
            "mean",
        ),

        mean_parallelism=(
            "final_parallelism",
            "mean",
        ),
    )
)


severity_summary = (
    component_df[
        "severity"
    ]
    .value_counts()
    .rename_axis(
        "severity"
    )
    .to_frame(
        "count"
    )
)


severity_summary[
    "percentage"
] = (
    severity_summary[
        "count"
    ]
    / len(component_df)
    * 100.0
)


scenario_summary = (
    component_df[
        "scenario_type"
    ]
    .value_counts()
    .rename_axis(
        "scenario_type"
    )
    .to_frame(
        "count"
    )
)


scenario_summary[
    "percentage"
] = (
    scenario_summary[
        "count"
    ]
    / len(component_df)
    * 100.0
)


stage_summary = (
    component_df
    .groupby(
        "component_index"
    )[
        "state_after_quality"
    ]
    .agg(
        [
            "mean",
            "median",
            "std",
        ]
    )
)


batch_summary.to_csv(
    "results/tables/"
    "v3_2_full_batch_summary.csv"
)

severity_summary.to_csv(
    "results/tables/"
    "v3_2_full_severity_summary.csv"
)

scenario_summary.to_csv(
    "results/tables/"
    "v3_2_full_scenario_summary.csv"
)

stage_summary.to_csv(
    "results/tables/"
    "v3_2_full_stage_summary.csv"
)


# ============================================================
# TERMINAL SUMMARY
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "FULL V3.2 DATASET GENERATED"
)

print(
    "============================================================"
)


print(
    f"\nTotal completed assemblies : "
    f"{len(assembly_df)}"
)

print(
    f"Components per assembly    : "
    f"{N_COMPONENTS}"
)

print(
    f"Total component decisions  : "
    f"{len(component_df)}"
)


print(
    "\nASSEMBLIES BY BATCH CONDITION"
)

print(
    assembly_df[
        "batch_condition"
    ]
    .value_counts()
)


print(
    "\nSEVERITY DISTRIBUTION"
)

print(
    severity_summary.round(2)
)


print(
    "\nSCENARIO DISTRIBUTION"
)

print(
    scenario_summary.round(2)
)


print(
    "\nFINAL QUALITY BY BATCH CONDITION"
)

print(
    batch_summary.round(4)
)


print(
    "\nQUALITY EVOLUTION BY COMPONENT STAGE"
)

print(
    stage_summary.round(4)
)


print(
    "\nFiles saved:"
)

print(
    COMPONENT_OUTPUT
)

print(
    ASSEMBLY_OUTPUT
)

print(
    "results/tables/"
)


print(
    "\n"
    "============================================================"
)

print(
    "FULL CONTROLLED DATASET GENERATION COMPLETED"
)

print(
    "============================================================"
)