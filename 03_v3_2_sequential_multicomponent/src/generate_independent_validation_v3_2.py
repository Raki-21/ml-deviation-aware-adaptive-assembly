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
# VERSION 3.2
# INDEPENDENT FINAL VALIDATION DATASET GENERATOR
# ============================================================
#
# PURPOSE
#
# Generate a completely new synthetic validation population
# AFTER model training, controller development and trigger
# calibration have been completed.
#
# THIS DATASET MUST NOT BE USED FOR:
#
#   - RF training
#   - model selection
#   - trigger threshold calibration
#   - optimizer tuning
#   - correction-bound tuning
#
#
# It is reserved for final independent validation.
#
#
# DESIGN
#
#   6 batch/process conditions
#   5 independent physical batch realizations / condition
#   10 finished assemblies / batch
#   5 sequential components / assembly
#
#
# Therefore:
#
#   6 x 5 x 10 = 300 finished assemblies
#
#   300 x 5 = 1500 sequential component decisions
#
#
# IMPORTANT
#
# The physical/synthetic mechanisms are intentionally kept
# consistent with the existing full V3.2 generator.
#
# The main differences are:
#
#   1. NEW random seed
#
#   2. NEW independent batch disturbance realizations
#
#   3. separate validation-only output files
#
#   4. multiple batches per condition instead of one single
#      disturbance realization per condition
#
#
# This gives a stronger test of generalization across:
#
#   - unseen assemblies
#   - unseen component deviations
#   - unseen batch disturbances
#
# ============================================================


# ============================================================
# RANDOM SEED
# ============================================================
#
# Training/development generator used:
#
#     20260824
#
# Other controller experiments used:
#
#     20260825
#
# Final independent validation uses:
#
#     20260826
#
# Do NOT change this after observing validation results.
# ============================================================

VALIDATION_SEED = 20260826

RNG = np.random.default_rng(
    VALIDATION_SEED
)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

N_COMPONENTS = 5

BATCHES_PER_CONDITION = 5

ASSEMBLIES_PER_BATCH = 10


BATCH_CONDITIONS = [
    "normal",
    "offset_drift",
    "angular_drift",
    "fixture_drift",
    "high_variation",
    "disturbed",
]


N_CONDITIONS = len(
    BATCH_CONDITIONS
)


EXPECTED_BATCHES = (
    N_CONDITIONS
    *
    BATCHES_PER_CONDITION
)


EXPECTED_ASSEMBLIES = (
    EXPECTED_BATCHES
    *
    ASSEMBLIES_PER_BATCH
)


EXPECTED_COMPONENT_ROWS = (
    EXPECTED_ASSEMBLIES
    *
    N_COMPONENTS
)


# ============================================================
# VALIDATE BATCH CONDITION NAMES
# ============================================================

unsupported_conditions = [

    condition

    for condition
    in BATCH_CONDITIONS

    if condition
    not in SUPPORTED_BATCH_CONDITIONS
]


if unsupported_conditions:

    raise ValueError(
        "\nUnsupported validation batch conditions:\n"
        +
        "\n".join(
            unsupported_conditions
        )
    )


# ============================================================
# CONTROLLED SEVERITY PATTERN
# ============================================================
#
# Preserved from the developmental full-data generator.
#
# The purpose is to guarantee representation of all severity
# classes rather than depending completely on random sampling.
# ============================================================

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
    Preserve the same controlled severity logic used in the
    existing V3.2 full production dataset generator.

    For batches whose variation multiplier exceeds 1.0, the
    severity category is shifted upward.
    """

    base_severity = SEVERITY_PATTERN[
        global_component_counter
        %
        len(
            SEVERITY_PATTERN
        )
    ]


    if variation_multiplier <= 1.0:

        return base_severity


    if base_severity == "low":

        return "medium"


    if base_severity == "medium":

        return "high"


    if base_severity == "high":

        if (
            global_component_counter
            %
            3
            ==
            0
        ):

            return "extreme"

        return "high"


    return "extreme"


# ============================================================
# STORAGE
# ============================================================

component_records = []

assembly_records = []

batch_records = []


global_component_counter = 0

global_assembly_counter = 0

global_batch_id = 0


# ============================================================
# USE A DISTINCT VALIDATION ASSEMBLY-ID RANGE
# ============================================================
#
# This is not mathematically required by the RF because
# assembly_id is not used as an input feature.
#
# It is done for provenance and leakage protection.
# ============================================================

VALIDATION_ASSEMBLY_ID_START = 100001


# ============================================================
# HEADER
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "V3.2 INDEPENDENT FINAL VALIDATION DATASET GENERATOR"
)

print(
    "============================================================"
)


print(
    f"\nValidation seed             : "
    f"{VALIDATION_SEED}"
)


print(
    f"Batch conditions            : "
    f"{N_CONDITIONS}"
)


print(
    f"Batches / condition         : "
    f"{BATCHES_PER_CONDITION}"
)


print(
    f"Assemblies / batch          : "
    f"{ASSEMBLIES_PER_BATCH}"
)


print(
    f"Expected batches            : "
    f"{EXPECTED_BATCHES}"
)


print(
    f"Expected assemblies         : "
    f"{EXPECTED_ASSEMBLIES}"
)


print(
    f"Expected component decisions: "
    f"{EXPECTED_COMPONENT_ROWS}"
)


# ============================================================
# MAIN CONDITION LOOP
# ============================================================

for condition_index, condition in enumerate(
    BATCH_CONDITIONS,
    start=1,
):


    print(
        "\n"
        "------------------------------------------------------------"
    )


    print(
        f"Condition "
        f"{condition_index}/"
        f"{N_CONDITIONS}: "
        f"{condition}"
    )


    print(
        "------------------------------------------------------------"
    )


    # ========================================================
    # MULTIPLE INDEPENDENT BATCHES PER CONDITION
    # ========================================================

    for local_batch_number in range(
        1,
        BATCHES_PER_CONDITION + 1,
    ):


        global_batch_id += 1


        (
            batch_disturbance_profile,
            batch_features,
        ) = create_batch_condition(

            condition=condition,

            rng=RNG,
        )


        print(
            f"  Batch "
            f"{local_batch_number}/"
            f"{BATCHES_PER_CONDITION}"
            f" | global batch ID "
            f"{global_batch_id}"
        )


        # ----------------------------------------------------
        # BATCH METADATA
        # ----------------------------------------------------

        batch_records.append(
            {

                "validation_seed":
                    VALIDATION_SEED,

                "batch_id":
                    global_batch_id,

                "batch_condition":
                    condition,

                "local_batch_number":
                    local_batch_number,

                "n_assemblies":
                    ASSEMBLIES_PER_BATCH,

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
            }
        )


        # ====================================================
        # FINISHED ASSEMBLIES WITHIN THIS BATCH
        # ====================================================

        for local_assembly_number in range(
            1,
            ASSEMBLIES_PER_BATCH + 1,
        ):


            global_assembly_counter += 1


            assembly_id = (
                VALIDATION_ASSEMBLY_ID_START
                +
                global_assembly_counter
                -
                1
            )


            state = create_initial_state()


            severity_sequence = []

            scenario_sequence = []


            # ================================================
            # SEQUENTIAL COMPONENT LOOP
            # ================================================

            for component_index in range(
                1,
                N_COMPONENTS + 1,
            ):


                global_component_counter += 1


                # --------------------------------------------
                # CONTROLLED SEVERITY
                # --------------------------------------------

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


                # --------------------------------------------
                # NEW INDEPENDENT COMPONENT DEVIATION
                # --------------------------------------------

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


                # --------------------------------------------
                # STATE BEFORE COMPONENT
                # --------------------------------------------

                metrics_before = (
                    calculate_quality_metrics(
                        state
                    )
                )


                # --------------------------------------------
                # APPLY COMPONENT + SHARED BATCH DISTURBANCE
                # --------------------------------------------

                state = update_assembly_state(

                    previous_state=state,

                    component_deviation=(
                        component_profile
                    ),

                    fixture_drift=(
                        batch_disturbance_profile
                    ),
                )


                # --------------------------------------------
                # STATE AFTER COMPONENT
                # --------------------------------------------

                metrics_after = (
                    calculate_quality_metrics(
                        state
                    )
                )


                # --------------------------------------------
                # COMPONENT RECORD
                # --------------------------------------------

                component_records.append(
                    {

                        # ====================================
                        # VALIDATION PROVENANCE
                        # ====================================

                        "validation_seed":
                            VALIDATION_SEED,

                        "validation_only":
                            True,


                        # ====================================
                        # IDENTIFICATION
                        # ====================================

                        "batch_id":
                            global_batch_id,

                        "local_batch_number":
                            local_batch_number,

                        "batch_condition":
                            condition,

                        "assembly_id":
                            assembly_id,

                        "local_assembly_number":
                            local_assembly_number,

                        "component_index":
                            component_index,


                        # ====================================
                        # BATCH / PROCESS VARIABLES
                        # ====================================

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


                        # ====================================
                        # COMPONENT DEVIATION CLASS
                        # ====================================

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


                        # ====================================
                        # COMPONENT DEVIATION VALUES
                        # ====================================

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


                        # ====================================
                        # COMPONENT PROFILE DESCRIPTORS
                        # ====================================

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


                        # ====================================
                        # STATE BEFORE COMPONENT
                        # ====================================

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


                        # ====================================
                        # STATE AFTER COMPONENT
                        # ====================================

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


                        # ====================================
                        # INCREMENTAL EFFECT
                        # ====================================

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


            # ================================================
            # FINAL ASSEMBLY RECORD
            # ================================================

            final_metrics = (
                calculate_quality_metrics(
                    state
                )
            )


            assembly_records.append(
                {

                    "validation_seed":
                        VALIDATION_SEED,

                    "validation_only":
                        True,

                    "batch_id":
                        global_batch_id,

                    "local_batch_number":
                        local_batch_number,

                    "batch_condition":
                        condition,

                    "assembly_id":
                        assembly_id,

                    "local_assembly_number":
                        local_assembly_number,

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
# DATAFRAMES
# ============================================================

component_df = pd.DataFrame(
    component_records
)


assembly_df = pd.DataFrame(
    assembly_records
)


batch_df = pd.DataFrame(
    batch_records
)


# ============================================================
# VALIDATION CHECKS
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "INDEPENDENT DATASET VALIDATION CHECKS"
)

print(
    "============================================================"
)


# ------------------------------------------------------------
# EXPECTED SIZE
# ------------------------------------------------------------

if len(
    batch_df
) != EXPECTED_BATCHES:

    raise ValueError(
        f"\nExpected {EXPECTED_BATCHES} batches, "
        f"found {len(batch_df)}."
    )


if len(
    assembly_df
) != EXPECTED_ASSEMBLIES:

    raise ValueError(
        f"\nExpected {EXPECTED_ASSEMBLIES} assemblies, "
        f"found {len(assembly_df)}."
    )


if len(
    component_df
) != EXPECTED_COMPONENT_ROWS:

    raise ValueError(
        f"\nExpected {EXPECTED_COMPONENT_ROWS} component rows, "
        f"found {len(component_df)}."
    )


# ------------------------------------------------------------
# MISSING VALUES
# ------------------------------------------------------------

component_missing = int(
    component_df
    .isna()
    .sum()
    .sum()
)


assembly_missing = int(
    assembly_df
    .isna()
    .sum()
    .sum()
)


batch_missing = int(
    batch_df
    .isna()
    .sum()
    .sum()
)


if (
    component_missing != 0
    or
    assembly_missing != 0
    or
    batch_missing != 0
):

    raise ValueError(
        "\nMissing values detected in independent "
        "validation dataset."
    )


# ------------------------------------------------------------
# FIVE COMPONENTS PER ASSEMBLY
# ------------------------------------------------------------

components_per_assembly = (
    component_df
    .groupby(
        "assembly_id"
    )[
        "component_index"
    ]
    .count()
)


if not (
    components_per_assembly
    ==
    N_COMPONENTS
).all():

    raise ValueError(
        "\nNot every validation assembly contains exactly "
        f"{N_COMPONENTS} sequential components."
    )


# ------------------------------------------------------------
# EXACT BALANCE BY CONDITION
# ------------------------------------------------------------

condition_counts = (
    assembly_df[
        "batch_condition"
    ]
    .value_counts()
)


expected_per_condition = (
    BATCHES_PER_CONDITION
    *
    ASSEMBLIES_PER_BATCH
)


for condition in BATCH_CONDITIONS:

    actual_count = int(
        condition_counts.get(
            condition,
            0,
        )
    )


    if actual_count != expected_per_condition:

        raise ValueError(
            f"\nCondition {condition} expected "
            f"{expected_per_condition} assemblies, "
            f"found {actual_count}."
        )


# ------------------------------------------------------------
# UNIQUE VALIDATION IDS
# ------------------------------------------------------------

if (
    assembly_df[
        "assembly_id"
    ]
    .nunique()
    !=
    EXPECTED_ASSEMBLIES
):

    raise ValueError(
        "\nDuplicate validation assembly IDs detected."
    )


# ============================================================
# OUTPUT PATHS
# ============================================================

os.makedirs(
    "data/validation",
    exist_ok=True,
)


os.makedirs(
    "results/validation",
    exist_ok=True,
)


COMPONENT_OUTPUT = (
    "data/validation/"
    "v3_2_independent_validation_component_dataset.csv"
)


ASSEMBLY_OUTPUT = (
    "data/validation/"
    "v3_2_independent_validation_assembly_dataset.csv"
)


BATCH_OUTPUT = (
    "data/validation/"
    "v3_2_independent_validation_batch_metadata.csv"
)


# ============================================================
# SAVE DATA
# ============================================================

component_df.to_csv(
    COMPONENT_OUTPUT,
    index=False,
)


assembly_df.to_csv(
    ASSEMBLY_OUTPUT,
    index=False,
)


batch_df.to_csv(
    BATCH_OUTPUT,
    index=False,
)


# ============================================================
# SUMMARY TABLES
# ============================================================

condition_summary = (
    assembly_df
    .groupby(
        "batch_condition"
    )
    .agg(

        assemblies=(
            "assembly_id",
            "count",
        ),

        independent_batches=(
            "batch_id",
            "nunique",
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

        p95_quality=(
            "final_quality_score",
            lambda x:
                x.quantile(
                    0.95
                ),
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


batch_summary = (
    assembly_df
    .groupby(
        [
            "batch_condition",
            "batch_id",
        ]
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
    /
    len(
        component_df
    )
    *
    100.0
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
    /
    len(
        component_df
    )
    *
    100.0
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


stage_summary[
    "p95"
] = (
    component_df
    .groupby(
        "component_index"
    )[
        "state_after_quality"
    ]
    .quantile(
        0.95
    )
)


# ============================================================
# SAVE SUMMARIES
# ============================================================

condition_summary.to_csv(
    "results/validation/"
    "v3_2_independent_validation_condition_summary.csv"
)


batch_summary.to_csv(
    "results/validation/"
    "v3_2_independent_validation_batch_summary.csv"
)


severity_summary.to_csv(
    "results/validation/"
    "v3_2_independent_validation_severity_summary.csv"
)


scenario_summary.to_csv(
    "results/validation/"
    "v3_2_independent_validation_scenario_summary.csv"
)


stage_summary.to_csv(
    "results/validation/"
    "v3_2_independent_validation_stage_summary.csv"
)


# ============================================================
# VALIDATION MANIFEST
# ============================================================

manifest_df = pd.DataFrame(
    {

        "setting": [

            "dataset_role",

            "validation_seed",

            "n_batch_conditions",

            "batches_per_condition",

            "assemblies_per_batch",

            "total_batches",

            "total_assemblies",

            "components_per_assembly",

            "total_component_decisions",

            "assembly_id_start",

            "used_for_training",

            "used_for_model_selection",

            "used_for_trigger_calibration",
        ],

        "value": [

            "FINAL_INDEPENDENT_VALIDATION_ONLY",

            VALIDATION_SEED,

            N_CONDITIONS,

            BATCHES_PER_CONDITION,

            ASSEMBLIES_PER_BATCH,

            EXPECTED_BATCHES,

            EXPECTED_ASSEMBLIES,

            N_COMPONENTS,

            EXPECTED_COMPONENT_ROWS,

            VALIDATION_ASSEMBLY_ID_START,

            False,

            False,

            False,
        ],
    }
)


manifest_df.to_csv(
    "data/validation/"
    "v3_2_independent_validation_manifest.csv",
    index=False,
)


# ============================================================
# TERMINAL SUMMARY
# ============================================================

print(
    "\nPASS - Dataset size is correct."
)


print(
    "PASS - No missing values."
)


print(
    "PASS - Every assembly contains exactly "
    f"{N_COMPONENTS} components."
)


print(
    "PASS - All six batch conditions are balanced."
)


print(
    "PASS - Validation assembly IDs are unique."
)


print(
    "\n"
    "============================================================"
)

print(
    "INDEPENDENT VALIDATION DATASET GENERATED"
)

print(
    "============================================================"
)


print(
    f"\nIndependent batches         : "
    f"{len(batch_df)}"
)


print(
    f"Completed assemblies        : "
    f"{len(assembly_df)}"
)


print(
    f"Sequential decisions        : "
    f"{len(component_df)}"
)


print(
    "\nASSEMBLIES BY CONDITION"
)


print(
    condition_counts
)


print(
    "\nSEVERITY DISTRIBUTION"
)


print(
    severity_summary.round(
        2
    )
)


print(
    "\nSCENARIO DISTRIBUTION"
)


print(
    scenario_summary.round(
        2
    )
)


print(
    "\nFINAL QUALITY BY CONDITION"
)


print(
    condition_summary.round(
        4
    )
)


print(
    "\nQUALITY EVOLUTION BY COMPONENT STAGE"
)


print(
    stage_summary.round(
        4
    )
)


print(
    "\nSaved:"
)


print(
    COMPONENT_OUTPUT
)


print(
    ASSEMBLY_OUTPUT
)


print(
    BATCH_OUTPUT
)


print(
    "data/validation/"
    "v3_2_independent_validation_manifest.csv"
)


print(
    "results/validation/"
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.2 INDEPENDENT VALIDATION GENERATION COMPLETED"
)

print(
    "============================================================"
)