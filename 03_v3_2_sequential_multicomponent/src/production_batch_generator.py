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
    sample_batch_condition,
)


# ============================================================
# VERSION 3.2 - PRODUCTION BATCH GENERATOR
# ============================================================
#
# Purpose:
#
# Combine:
#
#   1. Individual component variation
#   2. Shared batch/process disturbance
#   3. Sequential multi-component assembly
#
# Structure:
#
#   Production Batch
#       |
#       |-- Assembly 1
#       |      |-- Component 1
#       |      |-- Component 2
#       |      |-- ...
#       |      |-- Component 5
#       |
#       |-- Assembly 2
#       |      |-- Component 1
#       |      |-- ...
#       |
#       |-- ...
#
# This script is still a DATA GENERATION / BASELINE stage.
#
# Machine Learning and Bayesian Optimization are NOT added yet.
# ============================================================


# ------------------------------------------------------------
# RANDOM GENERATOR
# ------------------------------------------------------------

RNG = np.random.default_rng(2026)


# ------------------------------------------------------------
# PILOT SIMULATION SIZE
# ------------------------------------------------------------
#
# DO NOT change this to 1000 yet.
#
# First verify the combined model using a small dataset.
# ------------------------------------------------------------

N_BATCHES = 5
ASSEMBLIES_PER_BATCH = 10
N_COMPONENTS = 5


# ------------------------------------------------------------
# SEVERITY SAMPLING BASED ON BATCH VARIATION
# ------------------------------------------------------------

def sample_component_severity(
    variation_multiplier,
    rng,
):
    """
    Select component severity based partly on the
    current batch variability.

    Normal batches:
        mostly low / medium cases.

    High-variation or disturbed batches:
        increased probability of high / extreme cases.

    This gives the batch variation multiplier an actual
    physical/statistical effect.
    """

    if variation_multiplier <= 1.0:

        probabilities = [
            0.30,  # low
            0.45,  # medium
            0.20,  # high
            0.05,  # extreme
        ]

    else:

        probabilities = [
            0.15,  # low
            0.30,  # medium
            0.40,  # high
            0.15,  # extreme
        ]

    severity = rng.choice(
        [
            "low",
            "medium",
            "high",
            "extreme",
        ],
        p=probabilities,
    )

    return str(severity)


# ------------------------------------------------------------
# STORAGE
# ------------------------------------------------------------

component_records = []
assembly_records = []


# ============================================================
# MAIN PRODUCTION BATCH LOOP
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "V3.2 COMBINED PRODUCTION BATCH GENERATOR"
)

print(
    "============================================================\n"
)


for batch_id in range(
    1,
    N_BATCHES + 1,
):

    # --------------------------------------------------------
    # SELECT ONE SHARED PROCESS CONDITION FOR THIS BATCH
    # --------------------------------------------------------

    batch_condition = sample_batch_condition(
        rng=RNG
    )

    (
        batch_disturbance_profile,
        batch_features,
    ) = create_batch_condition(
        condition=batch_condition,
        rng=RNG,
    )


    print(
        f"Batch {batch_id} | "
        f"Condition = {batch_condition} | "
        f"Variation factor = "
        f"{batch_features['variation_multiplier']:.2f}"
    )


    # ========================================================
    # ASSEMBLIES WITHIN THIS BATCH
    # ========================================================

    for assembly_number in range(
        1,
        ASSEMBLIES_PER_BATCH + 1,
    ):

        # Unique global assembly identifier
        assembly_id = (
            (batch_id - 1)
            * ASSEMBLIES_PER_BATCH
            + assembly_number
        )


        # ----------------------------------------------------
        # INITIAL EMPTY ASSEMBLY STATE
        # ----------------------------------------------------

        state = create_initial_state()


        # Track assembly-level information
        severity_history = []
        scenario_history = []


        # ====================================================
        # SEQUENTIAL COMPONENT LOOP
        # ====================================================

        for component_index in range(
            1,
            N_COMPONENTS + 1,
        ):

            # ------------------------------------------------
            # DETERMINE COMPONENT SEVERITY
            # ------------------------------------------------

            severity = sample_component_severity(
                variation_multiplier=(
                    batch_features[
                        "variation_multiplier"
                    ]
                ),
                rng=RNG,
            )


            # ------------------------------------------------
            # GENERATE UNIQUE COMPONENT DEVIATION
            # ------------------------------------------------

            (
                component_profile,
                component_features,
            ) = generate_deviation_profile(
                severity=severity,
                rng=RNG,
            )


            severity_history.append(
                severity
            )

            scenario_history.append(
                component_features[
                    "scenario_type"
                ]
            )


            # ------------------------------------------------
            # QUALITY BEFORE ADDING THIS COMPONENT
            # ------------------------------------------------

            metrics_before = (
                calculate_quality_metrics(
                    state
                )
            )


            # ------------------------------------------------
            # SEQUENTIAL STATE UPDATE
            # ------------------------------------------------
            #
            # Current modelling assumption:
            #
            # Each component experiences:
            #
            #   its own individual deviation
            #
            # PLUS
            #
            #   the shared batch/process disturbance.
            #
            # The resulting effect is accumulated into the
            # current assembly state.
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
            # QUALITY AFTER COMPONENT ASSEMBLY
            # ------------------------------------------------

            metrics_after = (
                calculate_quality_metrics(
                    state
                )
            )


            # ------------------------------------------------
            # STORE COMPONENT-LEVEL RECORD
            # ------------------------------------------------

            component_records.append(
                {
                    # IDs
                    "batch_id":
                        batch_id,

                    "assembly_id":
                        assembly_id,

                    "component_index":
                        component_index,

                    # Batch/process condition
                    "batch_condition":
                        batch_features[
                            "batch_condition"
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

                    "variation_multiplier":
                        batch_features[
                            "variation_multiplier"
                        ],

                    # Component deviation information
                    "scenario_type":
                        component_features[
                            "scenario_type"
                        ],

                    "severity":
                        component_features[
                            "severity"
                        ],

                    "active_modes":
                        component_features[
                            "active_modes"
                        ],

                    "n_active_modes":
                        component_features[
                            "n_active_modes"
                        ],

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
                    "component_profile_rms_mm":
                        component_features[
                            "profile_rms_mm"
                        ],

                    "component_parallelism_mm":
                        component_features[
                            "profile_parallelism_mm"
                        ],

                    # State BEFORE component
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

                    # State AFTER component
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
                }
            )


        # ====================================================
        # FINAL ASSEMBLY RESULT
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

                "assembly_id":
                    assembly_id,

                "batch_condition":
                    batch_features[
                        "batch_condition"
                    ],

                "variation_multiplier":
                    batch_features[
                        "variation_multiplier"
                    ],

                "n_components":
                    N_COMPONENTS,

                "severity_sequence":
                    "+".join(
                        severity_history
                    ),

                "scenario_sequence":
                    "+".join(
                        scenario_history
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
# SAVE RESULTS
# ============================================================

component_output = (
    "data/processed/"
    "v3_2_component_level_dataset.csv"
)

assembly_output = (
    "data/processed/"
    "v3_2_assembly_level_dataset.csv"
)


component_df.to_csv(
    component_output,
    index=False,
)

assembly_df.to_csv(
    assembly_output,
    index=False,
)


# ============================================================
# SUMMARY OUTPUT
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "PILOT DATASET SUMMARY"
)

print(
    "============================================================"
)


print(
    f"\nNumber of batches       : "
    f"{N_BATCHES}"
)

print(
    f"Assemblies per batch    : "
    f"{ASSEMBLIES_PER_BATCH}"
)

print(
    f"Total assemblies        : "
    f"{len(assembly_df)}"
)

print(
    f"Components per assembly : "
    f"{N_COMPONENTS}"
)

print(
    f"Total component steps   : "
    f"{len(component_df)}"
)


# ------------------------------------------------------------
# BATCH CONDITION DISTRIBUTION
# ------------------------------------------------------------

print(
    "\nBATCH CONDITION COUNTS"
)

print(
    assembly_df[
        "batch_condition"
    ].value_counts()
)


# ------------------------------------------------------------
# COMPONENT SEVERITY DISTRIBUTION
# ------------------------------------------------------------

print(
    "\nCOMPONENT SEVERITY COUNTS"
)

print(
    component_df[
        "severity"
    ].value_counts()
)


# ------------------------------------------------------------
# COMPONENT SCENARIO DISTRIBUTION
# ------------------------------------------------------------

print(
    "\nCOMPONENT SCENARIO COUNTS"
)

print(
    component_df[
        "scenario_type"
    ].value_counts()
)


# ------------------------------------------------------------
# FINAL QUALITY SUMMARY
# ------------------------------------------------------------

print(
    "\nFINAL ASSEMBLY QUALITY"
)

print(
    assembly_df[
        "final_quality_score"
    ].describe()
)


# ------------------------------------------------------------
# SAVE LOCATIONS
# ------------------------------------------------------------

print(
    "\nSaved component-level dataset:"
)

print(
    component_output
)

print(
    "\nSaved assembly-level dataset:"
)

print(
    assembly_output
)


print(
    "\n"
    "============================================================"
)

print(
    "COMBINED PRODUCTION BATCH GENERATION COMPLETED"
)

print(
    "============================================================"
)