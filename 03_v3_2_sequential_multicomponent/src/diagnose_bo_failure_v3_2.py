import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# VERSION 3.2 - BO CLOSED-LOOP FAILURE DIAGNOSIS
# ============================================================
#
# Purpose:
#
# The 100-assembly pilot showed:
#
#   - BO > Random Search
#   - but BO did not clearly outperform zero correction
#
# We now determine WHY.
#
# Checks:
#
# 1. At which sequential component does BO lose advantage?
# 2. Which batch conditions cause the problem?
# 3. Which deviation severities cause the problem?
# 4. Which scenarios cause the problem?
# 5. Does RF become optimistic at BO-selected corrections?
# 6. Are corrections near capability limits?
# 7. Does BO help early but degrade later?
#
# No model or optimizer is changed in this script.
# This is purely diagnostic.
# ============================================================


COMPONENT_FILE = (
    "data/processed/"
    "v3_2_ml_bo_pilot_component_results.csv"
)

ASSEMBLY_FILE = (
    "data/processed/"
    "v3_2_ml_bo_pilot_assembly_results.csv"
)


required_files = [
    COMPONENT_FILE,
    ASSEMBLY_FILE,
]


missing = [
    path
    for path in required_files
    if not os.path.exists(path)
]


if missing:

    print("\nERROR - Missing files:")

    for path in missing:
        print(path)

    raise SystemExit(
        "\nRun the ML+BO pilot first."
    )


component_df = pd.read_csv(
    COMPONENT_FILE
)

assembly_df = pd.read_csv(
    ASSEMBLY_FILE
)


os.makedirs(
    "results/tables",
    exist_ok=True,
)

os.makedirs(
    "results/plots",
    exist_ok=True,
)


print(
    "\n============================================================"
)

print(
    "V3.2 BO CLOSED-LOOP FAILURE DIAGNOSIS"
)

print(
    "============================================================"
)


# ============================================================
# 1. COMPONENT-STAGE ANALYSIS
# ============================================================

stage = (
    component_df
    .groupby("component_index")
    .agg(
        samples=(
            "assembly_id",
            "count",
        ),

        zero_mean=(
            "zero_quality",
            "mean",
        ),

        random_mean=(
            "random_actual_quality",
            "mean",
        ),

        bo_mean=(
            "bo_actual_quality",
            "mean",
        ),

        predicted_bo_mean=(
            "bo_predicted_quality",
            "mean",
        ),

        prediction_abs_error=(
            "bo_prediction_absolute_error",
            "mean",
        ),

        prediction_signed_error=(
            "bo_prediction_error",
            "mean",
        ),

        utilization=(
            "bo_correction_utilization",
            "mean",
        ),
    )
)


stage[
    "bo_minus_zero"
] = (
    stage["bo_mean"]
    -
    stage["zero_mean"]
)


stage[
    "bo_improvement_vs_zero_percent"
] = (
    (
        stage["zero_mean"]
        -
        stage["bo_mean"]
    )
    /
    stage["zero_mean"]
    * 100.0
)


stage[
    "bo_improvement_vs_random_percent"
] = (
    (
        stage["random_mean"]
        -
        stage["bo_mean"]
    )
    /
    stage["random_mean"]
    * 100.0
)


print(
    "\n1. PERFORMANCE BY SEQUENTIAL COMPONENT"
)

print(
    "------------------------------------------------------------"
)

print(
    stage.round(5)
)


# ============================================================
# 2. INDIVIDUAL COMPONENT WIN RATE
# ============================================================

component_df[
    "bo_beats_zero"
] = (
    component_df[
        "bo_actual_quality"
    ]
    <
    component_df[
        "zero_quality"
    ]
)


component_df[
    "bo_beats_random"
] = (
    component_df[
        "bo_actual_quality"
    ]
    <
    component_df[
        "random_actual_quality"
    ]
)


stage_win = (
    component_df
    .groupby(
        "component_index"
    )
    .agg(
        bo_vs_zero_win_rate=(
            "bo_beats_zero",
            "mean",
        ),

        bo_vs_random_win_rate=(
            "bo_beats_random",
            "mean",
        ),
    )
    * 100.0
)


print(
    "\n2. COMPONENT-LEVEL WIN RATES"
)

print(
    "------------------------------------------------------------"
)

print(
    stage_win.round(2)
)


# ============================================================
# 3. BATCH CONDITION DIAGNOSIS
# ============================================================

batch = (
    component_df
    .groupby(
        "batch_condition"
    )
    .agg(
        samples=(
            "assembly_id",
            "count",
        ),

        zero_mean=(
            "zero_quality",
            "mean",
        ),

        random_mean=(
            "random_actual_quality",
            "mean",
        ),

        bo_mean=(
            "bo_actual_quality",
            "mean",
        ),

        bo_prediction_error=(
            "bo_prediction_absolute_error",
            "mean",
        ),

        signed_prediction_error=(
            "bo_prediction_error",
            "mean",
        ),

        utilization=(
            "bo_correction_utilization",
            "mean",
        ),

        bo_zero_win_rate=(
            "bo_beats_zero",
            "mean",
        ),
    )
)


batch[
    "bo_zero_win_rate"
] *= 100.0


batch[
    "bo_improvement_vs_zero_percent"
] = (
    (
        batch["zero_mean"]
        -
        batch["bo_mean"]
    )
    /
    batch["zero_mean"]
    * 100.0
)


print(
    "\n3. PERFORMANCE BY BATCH CONDITION"
)

print(
    "------------------------------------------------------------"
)

print(
    batch.round(4)
)


# ============================================================
# 4. SEVERITY DIAGNOSIS
# ============================================================

severity = (
    component_df
    .groupby(
        "severity"
    )
    .agg(
        samples=(
            "assembly_id",
            "count",
        ),

        zero_mean=(
            "zero_quality",
            "mean",
        ),

        random_mean=(
            "random_actual_quality",
            "mean",
        ),

        bo_mean=(
            "bo_actual_quality",
            "mean",
        ),

        prediction_error=(
            "bo_prediction_absolute_error",
            "mean",
        ),

        signed_prediction_error=(
            "bo_prediction_error",
            "mean",
        ),

        utilization=(
            "bo_correction_utilization",
            "mean",
        ),

        bo_zero_win_rate=(
            "bo_beats_zero",
            "mean",
        ),
    )
)


severity[
    "bo_zero_win_rate"
] *= 100.0


severity[
    "bo_improvement_vs_zero_percent"
] = (
    (
        severity["zero_mean"]
        -
        severity["bo_mean"]
    )
    /
    severity["zero_mean"]
    * 100.0
)


severity = severity.reindex(
    [
        "low",
        "medium",
        "high",
        "extreme",
    ]
)


print(
    "\n4. PERFORMANCE BY SEVERITY"
)

print(
    "------------------------------------------------------------"
)

print(
    severity.round(4)
)


# ============================================================
# 5. SCENARIO DIAGNOSIS
# ============================================================

scenario = (
    component_df
    .groupby(
        "scenario_type"
    )
    .agg(
        samples=(
            "assembly_id",
            "count",
        ),

        zero_mean=(
            "zero_quality",
            "mean",
        ),

        random_mean=(
            "random_actual_quality",
            "mean",
        ),

        bo_mean=(
            "bo_actual_quality",
            "mean",
        ),

        prediction_error=(
            "bo_prediction_absolute_error",
            "mean",
        ),

        utilization=(
            "bo_correction_utilization",
            "mean",
        ),

        bo_zero_win_rate=(
            "bo_beats_zero",
            "mean",
        ),
    )
)


scenario[
    "bo_zero_win_rate"
] *= 100.0


scenario[
    "bo_improvement_vs_zero_percent"
] = (
    (
        scenario["zero_mean"]
        -
        scenario["bo_mean"]
    )
    /
    scenario["zero_mean"]
    * 100.0
)


scenario = scenario.sort_values(
    "bo_improvement_vs_zero_percent"
)


print(
    "\n5. PERFORMANCE BY DEVIATION SCENARIO"
)

print(
    "------------------------------------------------------------"
)

print(
    scenario.round(4)
)


# ============================================================
# 6. SURROGATE OPTIMISM CHECK
# ============================================================
#
# bo_prediction_error =
#
# actual BO quality - RF predicted BO quality
#
# Positive value:
# RF predicted a result that was too optimistic.
# ============================================================

component_df[
    "surrogate_optimistic"
] = (
    component_df[
        "bo_prediction_error"
    ]
    > 0.0
)


optimism_rate = (
    component_df[
        "surrogate_optimistic"
    ]
    .mean()
    * 100.0
)


mean_signed_error = (
    component_df[
        "bo_prediction_error"
    ]
    .mean()
)


median_signed_error = (
    component_df[
        "bo_prediction_error"
    ]
    .median()
)


print(
    "\n6. SURROGATE OPTIMISM AT BO-SELECTED POINTS"
)

print(
    "------------------------------------------------------------"
)


print(
    f"RF optimistic cases : "
    f"{optimism_rate:.2f}%"
)

print(
    f"Mean actual-predicted error : "
    f"{mean_signed_error:.5f}"
)

print(
    f"Median actual-predicted error : "
    f"{median_signed_error:.5f}"
)


# ============================================================
# 7. CORRECTION SATURATION CHECK
# ============================================================

util = (
    component_df[
        "bo_correction_utilization"
    ]
)


near_limit = (
    util >= 0.90
).mean() * 100.0


at_limit = (
    util >= 0.999
).mean() * 100.0


print(
    "\n7. CAPABILITY SATURATION"
)

print(
    "------------------------------------------------------------"
)


print(
    util
    .describe(
        percentiles=[
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )
    .round(5)
)


print(
    f"\n>= 90% capability : "
    f"{near_limit:.2f}%"
)

print(
    f"At limit          : "
    f"{at_limit:.2f}%"
)


# ============================================================
# 8. ASSEMBLY FINAL FAILURE MAGNITUDE
# ============================================================

assembly_df[
    "bo_minus_zero"
] = (
    assembly_df[
        "bo_final_quality"
    ]
    -
    assembly_df[
        "zero_final_quality"
    ]
)


worst_bo_cases = (
    assembly_df
    .sort_values(
        "bo_minus_zero",
        ascending=False,
    )
    .head(10)
)


best_bo_cases = (
    assembly_df
    .sort_values(
        "bo_minus_zero",
        ascending=True,
    )
    .head(10)
)


print(
    "\n8. WORST 10 BO CASES VS ZERO"
)

print(
    "------------------------------------------------------------"
)


print(
    worst_bo_cases[
        [
            "assembly_id",
            "batch_condition",
            "zero_final_quality",
            "bo_final_quality",
            "bo_minus_zero",
        ]
    ].round(4)
)


# ============================================================
# SAVE TABLES
# ============================================================

stage.to_csv(
    "results/tables/"
    "v3_2_bo_diagnosis_by_stage.csv"
)


stage_win.to_csv(
    "results/tables/"
    "v3_2_bo_diagnosis_stage_win_rates.csv"
)


batch.to_csv(
    "results/tables/"
    "v3_2_bo_diagnosis_by_batch.csv"
)


severity.to_csv(
    "results/tables/"
    "v3_2_bo_diagnosis_by_severity.csv"
)


scenario.to_csv(
    "results/tables/"
    "v3_2_bo_diagnosis_by_scenario.csv"
)


worst_bo_cases.to_csv(
    "results/tables/"
    "v3_2_bo_worst_assemblies.csv",
    index=False,
)


best_bo_cases.to_csv(
    "results/tables/"
    "v3_2_bo_best_assemblies.csv",
    index=False,
)


# ============================================================
# PLOT 1 - BO ADVANTAGE THROUGH SEQUENCE
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.axhline(
    0.0,
    linestyle="--",
)


plt.plot(
    stage.index,
    stage[
        "bo_improvement_vs_zero_percent"
    ],
    marker="o",
)


plt.xlabel(
    "Component Index"
)

plt.ylabel(
    "BO Improvement vs Zero (%)"
)

plt.title(
    "BO Advantage Across Sequential Assembly"
)

plt.xticks(
    stage.index
)

plt.tight_layout()


plt.savefig(
    "results/plots/"
    "v3_2_bo_advantage_by_stage.png",
    dpi=300,
)


plt.close()


# ============================================================
# PLOT 2 - PREDICTION ERROR THROUGH SEQUENCE
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    stage.index,
    stage[
        "prediction_abs_error"
    ],
    marker="o",
)


plt.xlabel(
    "Component Index"
)

plt.ylabel(
    "Mean Absolute Prediction Error"
)

plt.title(
    "Surrogate Error During Closed-Loop Sequential Operation"
)

plt.xticks(
    stage.index
)

plt.tight_layout()


plt.savefig(
    "results/plots/"
    "v3_2_closed_loop_surrogate_error_by_stage.png",
    dpi=300,
)


plt.close()


# ============================================================
# FINAL AUTOMATIC INTERPRETATION
# ============================================================

print(
    "\n============================================================"
)

print(
    "DIAGNOSTIC INTERPRETATION"
)

print(
    "============================================================"
)


first_stage = float(
    stage.loc[
        stage.index.min(),
        "bo_improvement_vs_zero_percent",
    ]
)


last_stage = float(
    stage.loc[
        stage.index.max(),
        "bo_improvement_vs_zero_percent",
    ]
)


first_error = float(
    stage.loc[
        stage.index.min(),
        "prediction_abs_error",
    ]
)


last_error = float(
    stage.loc[
        stage.index.max(),
        "prediction_abs_error",
    ]
)


if (
    first_stage > 0
    and last_stage < first_stage
):

    print(
        "\nSIGNAL: BO advantage decreases as assembly progresses."
    )


if last_error > first_error:

    print(
        "SIGNAL: Surrogate prediction error increases "
        "through the sequential trajectory."
    )


if mean_signed_error > 0:

    print(
        "SIGNAL: RF is optimistic on average at "
        "BO-selected corrections."
    )


if near_limit > 20.0:

    print(
        "SIGNAL: Many recommendations operate close "
        "to correction capability boundaries."
    )


print(
    "\nDo NOT scale the experiment until these signals "
    "are interpreted."
)


print(
    "\n============================================================"
)

print(
    "V3.2 BO FAILURE DIAGNOSIS COMPLETED"
)

print(
    "============================================================"
)