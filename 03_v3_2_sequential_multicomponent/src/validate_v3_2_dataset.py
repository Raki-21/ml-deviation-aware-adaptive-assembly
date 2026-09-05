import os
import numpy as np
import pandas as pd


# ============================================================
# VERSION 3.2 - DATASET VALIDATION
# ============================================================
#
# Purpose:
# Check whether the generated sequential multi-component
# production-batch dataset is complete, diverse, internally
# consistent and scientifically usable before ML/optimization.
#
# This validation checks:
#   1. required files
#   2. missing values
#   3. batch-condition coverage
#   4. severity coverage
#   5. deviation-scenario coverage
#   6. mixed-mode coverage
#   7. component-stage coverage
#   8. final quality statistics
#   9. quality evolution through assembly
#  10. disturbed vs normal batch behaviour
# ============================================================


# ------------------------------------------------------------
# FILE LOCATIONS
# ------------------------------------------------------------

COMPONENT_FILE = (
    "data/processed/"
    "v3_2_component_level_dataset.csv"
)

ASSEMBLY_FILE = (
    "data/processed/"
    "v3_2_assembly_level_dataset.csv"
)


# ------------------------------------------------------------
# BASIC FILE CHECK
# ------------------------------------------------------------

print(
    "\n"
    "============================================================"
)

print(
    "V3.2 DATASET VALIDATION"
)

print(
    "============================================================\n"
)


required_files = [
    COMPONENT_FILE,
    ASSEMBLY_FILE,
]


missing_files = [
    file
    for file in required_files
    if not os.path.exists(file)
]


if missing_files:

    print(
        "ERROR - REQUIRED DATASET FILES ARE MISSING:"
    )

    for file in missing_files:
        print(file)

    raise SystemExit(
        "\nRun production_batch_generator.py first."
    )


print(
    "PASS - Required dataset files found."
)


# ============================================================
# LOAD DATA
# ============================================================

component_df = pd.read_csv(
    COMPONENT_FILE
)

assembly_df = pd.read_csv(
    ASSEMBLY_FILE
)


print(
    f"\nComponent-level rows : "
    f"{len(component_df)}"
)

print(
    f"Assembly-level rows  : "
    f"{len(assembly_df)}"
)


# ============================================================
# 1. MISSING-VALUE CHECK
# ============================================================

print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "1. MISSING VALUE CHECK"
)

print(
    "------------------------------------------------------------"
)


component_missing = int(
    component_df.isna().sum().sum()
)

assembly_missing = int(
    assembly_df.isna().sum().sum()
)


print(
    f"Component dataset missing values : "
    f"{component_missing}"
)

print(
    f"Assembly dataset missing values  : "
    f"{assembly_missing}"
)


if (
    component_missing == 0
    and assembly_missing == 0
):
    print(
        "PASS - No missing values detected."
    )
else:
    print(
        "WARNING - Missing values require inspection."
    )


# ============================================================
# 2. BATCH CONDITION COVERAGE
# ============================================================

print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "2. BATCH CONDITION COVERAGE"
)

print(
    "------------------------------------------------------------"
)


batch_counts = (
    assembly_df[
        "batch_condition"
    ]
    .value_counts()
    .sort_index()
)


print(
    batch_counts
)


unique_batch_conditions = (
    assembly_df[
        "batch_condition"
    ]
    .nunique()
)


print(
    f"\nUnique batch conditions represented: "
    f"{unique_batch_conditions}"
)


# ============================================================
# 3. SEVERITY COVERAGE
# ============================================================

print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "3. COMPONENT SEVERITY COVERAGE"
)

print(
    "------------------------------------------------------------"
)


severity_counts = (
    component_df[
        "severity"
    ]
    .value_counts()
    .reindex(
        [
            "low",
            "medium",
            "high",
            "extreme",
        ],
        fill_value=0,
    )
)


severity_percent = (
    severity_counts
    / len(component_df)
    * 100.0
)


severity_summary = pd.DataFrame(
    {
        "count":
            severity_counts,

        "percent":
            severity_percent,
    }
)


print(
    severity_summary.round(2)
)


if (
    severity_counts > 0
).all():

    print(
        "\nPASS - All four severity levels are represented."
    )

else:

    print(
        "\nWARNING - One or more severity levels are missing."
    )


# ============================================================
# 4. SCENARIO COVERAGE
# ============================================================

print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "4. COMPONENT SCENARIO COVERAGE"
)

print(
    "------------------------------------------------------------"
)


scenario_counts = (
    component_df[
        "scenario_type"
    ]
    .value_counts()
)


print(
    scenario_counts
)


expected_scenarios = {
    "offset",
    "tilt",
    "bend",
    "waviness",
    "twist",
    "mixed",
}


present_scenarios = set(
    component_df[
        "scenario_type"
    ].unique()
)


missing_scenarios = (
    expected_scenarios
    - present_scenarios
)


if not missing_scenarios:

    print(
        "\nPASS - All main component scenarios are represented."
    )

else:

    print(
        "\nWARNING - Missing scenarios:"
    )

    print(
        missing_scenarios
    )


# ============================================================
# 5. MIXED-MODE CHECK
# ============================================================

print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "5. MIXED-MODE DEVIATION CHECK"
)

print(
    "------------------------------------------------------------"
)


mixed_df = component_df[
    component_df[
        "scenario_type"
    ] == "mixed"
]


mixed_count = len(
    mixed_df
)


mixed_percent = (
    mixed_count
    / len(component_df)
    * 100.0
)


print(
    f"Mixed-mode components : "
    f"{mixed_count}"
)

print(
    f"Mixed-mode percentage : "
    f"{mixed_percent:.2f}%"
)


if mixed_count > 0:

    print(
        "\nExample mixed deviation combinations:"
    )

    print(
        mixed_df[
            "active_modes"
        ]
        .value_counts()
        .head(10)
    )

    print(
        "\nPASS - Mixed-mode cases are present."
    )

else:

    print(
        "\nWARNING - No mixed-mode cases generated."
    )


# ============================================================
# 6. COMPONENT-STAGE COVERAGE
# ============================================================

print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "6. COMPONENT-STAGE COVERAGE"
)

print(
    "------------------------------------------------------------"
)


stage_counts = (
    component_df[
        "component_index"
    ]
    .value_counts()
    .sort_index()
)


print(
    stage_counts
)


if (
    stage_counts.nunique()
    == 1
):

    print(
        "\nPASS - All component stages contain equal sample counts."
    )

else:

    print(
        "\nWARNING - Component-stage sample counts are uneven."
    )


# ============================================================
# 7. FINAL QUALITY STATISTICS
# ============================================================

print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "7. FINAL ASSEMBLY QUALITY STATISTICS"
)

print(
    "------------------------------------------------------------"
)


quality_stats = (
    assembly_df[
        "final_quality_score"
    ]
    .describe(
        percentiles=[
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
        ]
    )
)


print(
    quality_stats.round(4)
)


# ============================================================
# 8. QUALITY EVOLUTION BY COMPONENT STAGE
# ============================================================

print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "8. QUALITY EVOLUTION THROUGH ASSEMBLY"
)

print(
    "------------------------------------------------------------"
)


stage_quality = (
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


print(
    stage_quality.round(4)
)


first_stage_mean = float(
    stage_quality.loc[
        stage_quality.index.min(),
        "mean",
    ]
)

last_stage_mean = float(
    stage_quality.loc[
        stage_quality.index.max(),
        "mean",
    ]
)


print(
    f"\nMean quality after first component : "
    f"{first_stage_mean:.4f}"
)

print(
    f"Mean quality after final component : "
    f"{last_stage_mean:.4f}"
)


if last_stage_mean > first_stage_mean:

    print(
        "\nPASS - Uncorrected geometric error generally "
        "accumulates through sequential assembly."
    )

else:

    print(
        "\nNOTE - Mean quality did not monotonically increase. "
        "Inspect cancellation effects between deviations."
    )


# ============================================================
# 9. QUALITY INCREMENT PER COMPONENT
# ============================================================

print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "9. QUALITY CHANGE CAUSED BY EACH COMPONENT"
)

print(
    "------------------------------------------------------------"
)


component_df[
    "quality_change"
] = (
    component_df[
        "state_after_quality"
    ]
    -
    component_df[
        "state_before_quality"
    ]
)


quality_change_by_stage = (
    component_df
    .groupby(
        "component_index"
    )[
        "quality_change"
    ]
    .agg(
        [
            "mean",
            "median",
            "std",
        ]
    )
)


print(
    quality_change_by_stage.round(4)
)


# ============================================================
# 10. NORMAL VS DISTURBED BATCH COMPARISON
# ============================================================

print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "10. BATCH-CONDITION QUALITY COMPARISON"
)

print(
    "------------------------------------------------------------"
)


batch_quality_summary = (
    assembly_df
    .groupby(
        "batch_condition"
    )[
        "final_quality_score"
    ]
    .agg(
        [
            "count",
            "mean",
            "median",
            "std",
        ]
    )
    .sort_values(
        "mean"
    )
)


print(
    batch_quality_summary.round(4)
)


if (
    "normal"
    in batch_quality_summary.index
):

    normal_mean = float(
        batch_quality_summary.loc[
            "normal",
            "mean",
        ]
    )

    print(
        f"\nNormal-batch mean quality: "
        f"{normal_mean:.4f}"
    )


    other_conditions = (
        batch_quality_summary
        .drop(
            index="normal",
            errors="ignore",
        )
    )


    if len(
        other_conditions
    ) > 0:

        worst_condition = (
            other_conditions[
                "mean"
            ]
            .idxmax()
        )

        worst_mean = float(
            other_conditions.loc[
                worst_condition,
                "mean",
            ]
        )


        print(
            f"Worst observed batch condition: "
            f"{worst_condition}"
        )

        print(
            f"Worst-condition mean quality: "
            f"{worst_mean:.4f}"
        )


# ============================================================
# 11. SEVERITY VS FINAL COMPONENT-STEP QUALITY
# ============================================================

print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "11. COMPONENT SEVERITY EFFECT"
)

print(
    "------------------------------------------------------------"
)


severity_quality = (
    component_df
    .groupby(
        "severity"
    )[
        "state_after_quality"
    ]
    .agg(
        [
            "count",
            "mean",
            "median",
            "std",
        ]
    )
    .reindex(
        [
            "low",
            "medium",
            "high",
            "extreme",
        ]
    )
)


print(
    severity_quality.round(4)
)


# ============================================================
# 12. PROFILE MAGNITUDE CHECK
# ============================================================

print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "12. COMPONENT PROFILE MAGNITUDE CHECK"
)

print(
    "------------------------------------------------------------"
)


profile_rms_stats = (
    component_df[
        "component_profile_rms_mm"
    ]
    .describe(
        percentiles=[
            0.50,
            0.90,
            0.95,
            0.99,
        ]
    )
)


print(
    profile_rms_stats.round(4)
)


# ============================================================
# SAVE VALIDATION TABLES
# ============================================================

os.makedirs(
    "results/tables",
    exist_ok=True,
)


severity_summary.to_csv(
    "results/tables/"
    "v3_2_severity_distribution.csv"
)


stage_quality.to_csv(
    "results/tables/"
    "v3_2_quality_by_component_stage.csv"
)


batch_quality_summary.to_csv(
    "results/tables/"
    "v3_2_quality_by_batch_condition.csv"
)


quality_change_by_stage.to_csv(
    "results/tables/"
    "v3_2_quality_change_by_stage.csv"
)


severity_quality.to_csv(
    "results/tables/"
    "v3_2_quality_by_severity.csv"
)


# ============================================================
# FINAL VALIDATION MESSAGE
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "V3.2 DATASET VALIDATION COMPLETED"
)

print(
    "============================================================"
)

print(
    "\nValidation tables saved to:"
)

print(
    "results/tables/"
)