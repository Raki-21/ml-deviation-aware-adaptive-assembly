import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# VERSION 3.2 - ML+BO PILOT ANALYSIS
# ============================================================
#
# PURPOSE
#
# Analyze the completed 100-assembly sequential ML+BO pilot.
#
# Questions:
#
# 1. Does BO improve over zero correction?
# 2. Does BO beat Random Search under the same budget?
# 3. Is the advantage consistent assembly-by-assembly?
# 4. Which batch conditions are hardest?
# 5. Does performance degrade through Components 1 -> 5?
# 6. How accurate is the RF surrogate at BO-selected points?
# 7. How often does correction approach capability limits?
#
# This analysis determines whether V3.2 is ready for a
# larger final evaluation.
# ============================================================


# ------------------------------------------------------------
# FILE PATHS
# ------------------------------------------------------------

COMPONENT_FILE = (
    "data/processed/"
    "v3_2_ml_bo_pilot_component_results.csv"
)

ASSEMBLY_FILE = (
    "data/processed/"
    "v3_2_ml_bo_pilot_assembly_results.csv"
)

SUMMARY_FILE = (
    "results/tables/"
    "v3_2_ml_bo_pilot_summary.csv"
)


required_files = [
    COMPONENT_FILE,
    ASSEMBLY_FILE,
    SUMMARY_FILE,
]


missing = [
    file
    for file in required_files
    if not os.path.exists(file)
]


if missing:

    print(
        "\nERROR - Missing required pilot files:"
    )

    for file in missing:
        print(file)

    raise SystemExit(
        "\nRun evaluate_sequential_ml_bo_pilot_v3_2.py first."
    )


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

component_df = pd.read_csv(
    COMPONENT_FILE
)

assembly_df = pd.read_csv(
    ASSEMBLY_FILE
)

summary_df = pd.read_csv(
    SUMMARY_FILE
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
    "\n"
    "============================================================"
)

print(
    "V3.2 ML+BO PILOT ANALYSIS"
)

print(
    "============================================================"
)


print(
    f"\nAssemblies evaluated : "
    f"{len(assembly_df)}"
)

print(
    f"Sequential decisions : "
    f"{len(component_df)}"
)


# ============================================================
# 1. BASIC DATA INTEGRITY
# ============================================================

print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "1. DATA INTEGRITY"
)

print(
    "------------------------------------------------------------"
)


missing_values = (
    assembly_df
    .isna()
    .sum()
    .sum()
    +
    component_df
    .isna()
    .sum()
    .sum()
)


print(
    f"Missing values : "
    f"{missing_values}"
)


if missing_values == 0:

    print(
        "PASS - No missing pilot results."
    )

else:

    print(
        "WARNING - Missing values detected."
    )


# ============================================================
# 2. MAIN STRATEGY SUMMARY
# ============================================================

print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "2. MAIN STRATEGY SUMMARY"
)

print(
    "------------------------------------------------------------"
)


print(
    summary_df.round(5)
)


zero_mean = float(
    assembly_df[
        "zero_final_quality"
    ].mean()
)

random_mean = float(
    assembly_df[
        "random_final_quality"
    ].mean()
)

bo_mean = float(
    assembly_df[
        "bo_final_quality"
    ].mean()
)


bo_improvement_zero = (
    (
        zero_mean
        - bo_mean
    )
    / max(
        abs(zero_mean),
        1e-9,
    )
    * 100.0
)


bo_improvement_random = (
    (
        random_mean
        - bo_mean
    )
    / max(
        abs(random_mean),
        1e-9,
    )
    * 100.0
)


print(
    f"\nMean zero quality   : "
    f"{zero_mean:.5f}"
)

print(
    f"Mean random quality : "
    f"{random_mean:.5f}"
)

print(
    f"Mean BO quality     : "
    f"{bo_mean:.5f}"
)

print(
    f"\nBO improvement vs zero   : "
    f"{bo_improvement_zero:.2f}%"
)

print(
    f"BO improvement vs random : "
    f"{bo_improvement_random:.2f}%"
)


# ============================================================
# 3. ASSEMBLY-BY-ASSEMBLY WIN RATES
# ============================================================

print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "3. ASSEMBLY-BY-ASSEMBLY WIN RATES"
)

print(
    "------------------------------------------------------------"
)


assembly_df[
    "bo_beats_zero"
] = (
    assembly_df[
        "bo_final_quality"
    ]
    <
    assembly_df[
        "zero_final_quality"
    ]
)


assembly_df[
    "bo_beats_random"
] = (
    assembly_df[
        "bo_final_quality"
    ]
    <
    assembly_df[
        "random_final_quality"
    ]
)


assembly_df[
    "random_beats_zero"
] = (
    assembly_df[
        "random_final_quality"
    ]
    <
    assembly_df[
        "zero_final_quality"
    ]
)


bo_zero_win_rate = (
    assembly_df[
        "bo_beats_zero"
    ].mean()
    * 100.0
)


bo_random_win_rate = (
    assembly_df[
        "bo_beats_random"
    ].mean()
    * 100.0
)


random_zero_win_rate = (
    assembly_df[
        "random_beats_zero"
    ].mean()
    * 100.0
)


print(
    f"BO beats zero baseline : "
    f"{bo_zero_win_rate:.2f}%"
)

print(
    f"BO beats Random Search : "
    f"{bo_random_win_rate:.2f}%"
)

print(
    f"Random beats zero      : "
    f"{random_zero_win_rate:.2f}%"
)


# ============================================================
# 4. RESULT BY BATCH CONDITION
# ============================================================

print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "4. PERFORMANCE BY BATCH CONDITION"
)

print(
    "------------------------------------------------------------"
)


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

        zero_mean=(
            "zero_final_quality",
            "mean",
        ),

        random_mean=(
            "random_final_quality",
            "mean",
        ),

        bo_mean=(
            "bo_final_quality",
            "mean",
        ),

        bo_std=(
            "bo_final_quality",
            "std",
        ),

        bo_p95=(
            "bo_final_quality",
            lambda x:
                x.quantile(
                    0.95
                ),
        ),
    )
)


batch_summary[
    "bo_improvement_vs_zero_percent"
] = (
    (
        batch_summary[
            "zero_mean"
        ]
        -
        batch_summary[
            "bo_mean"
        ]
    )
    /
    batch_summary[
        "zero_mean"
    ]
    * 100.0
)


batch_summary[
    "bo_improvement_vs_random_percent"
] = (
    (
        batch_summary[
            "random_mean"
        ]
        -
        batch_summary[
            "bo_mean"
        ]
    )
    /
    batch_summary[
        "random_mean"
    ]
    * 100.0
)


print(
    batch_summary.round(4)
)


# ============================================================
# 5. QUALITY EVOLUTION THROUGH COMPONENTS
# ============================================================

print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "5. SEQUENTIAL QUALITY EVOLUTION"
)

print(
    "------------------------------------------------------------"
)


stage_summary = (
    component_df
    .groupby(
        "component_index"
    )
    .agg(
        zero_quality=(
            "zero_quality",
            "mean",
        ),

        random_quality=(
            "random_actual_quality",
            "mean",
        ),

        bo_quality=(
            "bo_actual_quality",
            "mean",
        ),

        bo_prediction_error=(
            "bo_prediction_absolute_error",
            "mean",
        ),
    )
)


print(
    stage_summary.round(5)
)


# ============================================================
# 6. SURROGATE ACCURACY AT BO-SELECTED POINTS
# ============================================================

print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "6. RF SURROGATE AT BO-SELECTED CORRECTIONS"
)

print(
    "------------------------------------------------------------"
)


bo_error = (
    component_df[
        "bo_prediction_absolute_error"
    ]
)


print(
    bo_error
    .describe(
        percentiles=[
            0.50,
            0.90,
            0.95,
            0.99,
        ]
    )
    .round(5)
)


mean_signed_error = (
    component_df[
        "bo_prediction_error"
    ].mean()
)


print(
    f"\nMean signed prediction error : "
    f"{mean_signed_error:.5f}"
)


# ============================================================
# 7. CAPABILITY UTILIZATION
# ============================================================

print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "7. CORRECTION CAPABILITY UTILIZATION"
)

print(
    "------------------------------------------------------------"
)


utilization = (
    component_df[
        "bo_correction_utilization"
    ]
)


print(
    utilization
    .describe(
        percentiles=[
            0.50,
            0.90,
            0.95,
            0.99,
        ]
    )
    .round(5)
)


near_limit_rate = (
    utilization
    >= 0.90
).mean() * 100.0


at_limit_rate = (
    utilization
    >= 0.999
).mean() * 100.0


print(
    f"\n>= 90% capability : "
    f"{near_limit_rate:.2f}%"
)

print(
    f"At capability limit : "
    f"{at_limit_rate:.2f}%"
)


# ============================================================
# 8. SEVERITY ANALYSIS
# ============================================================

print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "8. PERFORMANCE BY DEVIATION SEVERITY"
)

print(
    "------------------------------------------------------------"
)


severity_summary = (
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

        bo_prediction_error=(
            "bo_prediction_absolute_error",
            "mean",
        ),

        correction_utilization=(
            "bo_correction_utilization",
            "mean",
        ),
    )
)


severity_order = [
    "low",
    "medium",
    "high",
    "extreme",
]


severity_summary = (
    severity_summary
    .reindex(
        severity_order
    )
)


print(
    severity_summary.round(4)
)


# ============================================================
# 9. SCENARIO ANALYSIS
# ============================================================

print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "9. PERFORMANCE BY DEVIATION SCENARIO"
)

print(
    "------------------------------------------------------------"
)


scenario_summary = (
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

        correction_utilization=(
            "bo_correction_utilization",
            "mean",
        ),
    )
    .sort_values(
        "bo_mean"
    )
)


print(
    scenario_summary.round(4)
)


# ============================================================
# SAVE TABLES
# ============================================================

batch_summary.to_csv(
    "results/tables/"
    "v3_2_ml_bo_pilot_by_batch.csv"
)


stage_summary.to_csv(
    "results/tables/"
    "v3_2_ml_bo_pilot_by_stage.csv"
)


severity_summary.to_csv(
    "results/tables/"
    "v3_2_ml_bo_pilot_by_severity.csv"
)


scenario_summary.to_csv(
    "results/tables/"
    "v3_2_ml_bo_pilot_by_scenario.csv"
)


# ============================================================
# PLOT 1 - FINAL STRATEGY DISTRIBUTION
# ============================================================

plot_data = [
    assembly_df[
        "zero_final_quality"
    ],

    assembly_df[
        "random_final_quality"
    ],

    assembly_df[
        "bo_final_quality"
    ],
]


plt.figure(
    figsize=(8, 5)
)


plt.boxplot(
    plot_data,
    tick_labels=[
        "Zero",
        "Random Search",
        "Bayesian Optimization",
    ],
)


plt.ylabel(
    "Final Quality Score"
)

plt.title(
    "V3.2 Final Assembly Quality - Pilot"
)

plt.tight_layout()


plt.savefig(
    "results/plots/"
    "v3_2_ml_bo_final_quality_boxplot.png",
    dpi=300,
)


plt.close()


# ============================================================
# PLOT 2 - SEQUENTIAL QUALITY EVOLUTION
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    stage_summary.index,
    stage_summary[
        "zero_quality"
    ],
    marker="o",
    label="Zero correction",
)


plt.plot(
    stage_summary.index,
    stage_summary[
        "random_quality"
    ],
    marker="o",
    label="Random Search",
)


plt.plot(
    stage_summary.index,
    stage_summary[
        "bo_quality"
    ],
    marker="o",
    label="Bayesian Optimization",
)


plt.xlabel(
    "Component Index"
)

plt.ylabel(
    "Mean Accumulated Quality Score"
)

plt.title(
    "V3.2 Sequential Quality Evolution"
)

plt.xticks(
    stage_summary.index
)

plt.legend()

plt.tight_layout()


plt.savefig(
    "results/plots/"
    "v3_2_ml_bo_quality_evolution.png",
    dpi=300,
)


plt.close()


# ============================================================
# PLOT 3 - BO VS RANDOM PER ASSEMBLY
# ============================================================

minimum = min(
    assembly_df[
        "bo_final_quality"
    ].min(),

    assembly_df[
        "random_final_quality"
    ].min(),
)


maximum = max(
    assembly_df[
        "bo_final_quality"
    ].max(),

    assembly_df[
        "random_final_quality"
    ].max(),
)


plt.figure(
    figsize=(6, 6)
)


plt.scatter(
    assembly_df[
        "random_final_quality"
    ],

    assembly_df[
        "bo_final_quality"
    ],

    alpha=0.60,
)


plt.plot(
    [
        minimum,
        maximum,
    ],

    [
        minimum,
        maximum,
    ],

    linestyle="--",
)


plt.xlabel(
    "Random Search Final Quality"
)

plt.ylabel(
    "Bayesian Optimization Final Quality"
)

plt.title(
    "BO vs Random Search - Equal Evaluation Budget"
)

plt.tight_layout()


plt.savefig(
    "results/plots/"
    "v3_2_bo_vs_random_scatter.png",
    dpi=300,
)


plt.close()


# ============================================================
# FINAL DECISION SUPPORT
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "PILOT DECISION SUMMARY"
)

print(
    "============================================================"
)


if bo_mean < random_mean:

    print(
        "\nPASS - BO achieved lower mean final quality "
        "than Random Search."
    )

else:

    print(
        "\nIMPORTANT - BO did NOT achieve lower mean "
        "quality than Random Search."
    )


if bo_mean < zero_mean:

    print(
        "PASS - BO improved over the zero-correction baseline."
    )

else:

    print(
        "IMPORTANT - BO did NOT improve over zero correction."
    )


print(
    f"\nBO vs Random win rate : "
    f"{bo_random_win_rate:.2f}%"
)

print(
    f"BO vs Zero win rate   : "
    f"{bo_zero_win_rate:.2f}%"
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.2 ML+BO PILOT ANALYSIS COMPLETED"
)

print(
    "============================================================"
)