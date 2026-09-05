import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# VERSION 3.2
# FINAL THESIS EVIDENCE PACKAGE
# ============================================================
#
# PURPOSE
#
# Convert the already completed independent validation and
# robustness experiments into:
#
#   - final statistics
#   - confidence intervals
#   - paired comparisons
#   - thesis-ready tables
#   - thesis-ready figures
#
# IMPORTANT
#
# This script performs NO:
#
#   - model training
#   - controller tuning
#   - Bayesian Optimization
#   - trigger recalibration
#   - synthetic data generation
#
# All controller and validation experiments are already frozen.
# ============================================================


# ============================================================
# PATHS
# ============================================================

ASSEMBLY_FILE = (
    "data/validation/"
    "v3_2_final_controller_validation_assembly_results.csv"
)

COMPONENT_FILE = (
    "data/validation/"
    "v3_2_final_controller_validation_component_results.csv"
)

FINAL_SUMMARY_FILE = (
    "results/validation/"
    "v3_2_final_controller_validation_summary.csv"
)

STAGE_FILE = (
    "results/validation/"
    "v3_2_final_controller_validation_by_stage.csv"
)

BATCH_FILE = (
    "results/validation/"
    "v3_2_final_controller_validation_by_batch_condition.csv"
)

SEVERITY_FILE = (
    "results/validation/"
    "v3_2_final_controller_validation_by_severity.csv"
)

SCENARIO_FILE = (
    "results/validation/"
    "v3_2_final_controller_validation_by_scenario.csv"
)

ROBUSTNESS_SEED_FILE = (
    "results/validation/"
    "v3_2_final_robustness_by_seed.csv"
)

MAGNITUDE_FILE = (
    "results/validation/"
    "v3_2_final_deviation_magnitude_sensitivity.csv"
)

CAPABILITY_FILE = (
    "results/validation/"
    "v3_2_final_correction_capability_sensitivity.csv"
)

WEIGHT_FILE = (
    "results/validation/"
    "v3_2_final_quality_weight_sensitivity.csv"
)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

FIGURE_DIR = (
    "results/final_evidence/figures"
)

TABLE_DIR = (
    "results/final_evidence/tables"
)


os.makedirs(
    FIGURE_DIR,
    exist_ok=True,
)

os.makedirs(
    TABLE_DIR,
    exist_ok=True,
)


# ============================================================
# FILE CHECK
# ============================================================

required_files = [
    ASSEMBLY_FILE,
    COMPONENT_FILE,
    FINAL_SUMMARY_FILE,
    STAGE_FILE,
    BATCH_FILE,
    SEVERITY_FILE,
    SCENARIO_FILE,
    ROBUSTNESS_SEED_FILE,
    MAGNITUDE_FILE,
    CAPABILITY_FILE,
    WEIGHT_FILE,
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
        "\nFinal V3.2 validation outputs are incomplete."
    )


# ============================================================
# LOAD RESULTS
# ============================================================

assembly_df = pd.read_csv(
    ASSEMBLY_FILE
)

component_df = pd.read_csv(
    COMPONENT_FILE
)

final_summary_df = pd.read_csv(
    FINAL_SUMMARY_FILE
)

stage_df = pd.read_csv(
    STAGE_FILE
)

batch_df = pd.read_csv(
    BATCH_FILE
)

severity_df = pd.read_csv(
    SEVERITY_FILE
)

scenario_df = pd.read_csv(
    SCENARIO_FILE
)

seed_df = pd.read_csv(
    ROBUSTNESS_SEED_FILE
)

magnitude_df = pd.read_csv(
    MAGNITUDE_FILE
)

capability_df = pd.read_csv(
    CAPABILITY_FILE
)

weight_df = pd.read_csv(
    WEIGHT_FILE
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.2 FINAL THESIS EVIDENCE PACKAGE"
)

print(
    "============================================================"
)


print(
    f"\nIndependent assemblies : "
    f"{len(assembly_df)}"
)

print(
    f"Sequential decisions   : "
    f"{len(component_df)}"
)


# ============================================================
# HELPERS
# ============================================================

def mean_ci95(
    values,
):

    values = np.asarray(
        values,
        dtype=float,
    )


    values = values[
        np.isfinite(
            values
        )
    ]


    n = len(
        values
    )


    mean = float(
        np.mean(
            values
        )
    )


    if n <= 1:

        return (
            mean,
            np.nan,
            np.nan,
        )


    std = float(
        np.std(
            values,
            ddof=1,
        )
    )


    se = (
        std
        /
        np.sqrt(
            n
        )
    )


    margin = (
        1.96
        *
        se
    )


    return (
        mean,
        mean - margin,
        mean + margin,
    )


def paired_effect_size_dz(
    a,
    b,
):

    difference = (
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


    std_difference = float(
        np.std(
            difference,
            ddof=1,
        )
    )


    if std_difference <= 1e-12:

        return np.nan


    return float(
        np.mean(
            difference
        )
        /
        std_difference
    )


def save_figure(
    filename,
):

    path = os.path.join(
        FIGURE_DIR,
        filename,
    )


    plt.tight_layout()


    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )


    plt.close()


# ============================================================
# 1. FINAL CONTROLLER STATISTICS
# ============================================================

strategy_columns = {

    "Zero":
        "zero_final_quality",

    "Structured-20":
        "structured20_final_quality",

    "Selective BO":
        "selective_bo_final_quality",
}


strategy_records = []


for strategy_name, column in (
    strategy_columns.items()
):


    values = (
        assembly_df[
            column
        ]
        .to_numpy()
    )


    (
        mean,
        ci_low,
        ci_high,
    ) = mean_ci95(
        values
    )


    strategy_records.append(
        {

            "strategy":
                strategy_name,

            "n":
                len(
                    values
                ),

            "mean":
                mean,

            "median":
                float(
                    np.median(
                        values
                    )
                ),

            "std":
                float(
                    np.std(
                        values,
                        ddof=1,
                    )
                ),

            "p90":
                float(
                    np.quantile(
                        values,
                        0.90,
                    )
                ),

            "p95":
                float(
                    np.quantile(
                        values,
                        0.95,
                    )
                ),

            "ci95_low":
                ci_low,

            "ci95_high":
                ci_high,
        }
    )


strategy_statistics_df = pd.DataFrame(
    strategy_records
)


strategy_statistics_df.to_csv(
    os.path.join(
        TABLE_DIR,
        "v3_2_final_strategy_statistics.csv",
    ),
    index=False,
)


# ============================================================
# 2. PAIRED CONTROLLER COMPARISONS
# ============================================================

zero = (
    assembly_df[
        "zero_final_quality"
    ]
    .to_numpy()
)


structured = (
    assembly_df[
        "structured20_final_quality"
    ]
    .to_numpy()
)


selective = (
    assembly_df[
        "selective_bo_final_quality"
    ]
    .to_numpy()
)


structured_improvement = (
    zero
    -
    structured
)


selective_improvement = (
    zero
    -
    selective
)


bo_vs_structured = (
    structured
    -
    selective
)


paired_records = []


comparisons = [

    (
        "Structured-20 vs Zero",
        structured,
        zero,
    ),

    (
        "Selective BO vs Zero",
        selective,
        zero,
    ),

    (
        "Selective BO vs Structured-20",
        selective,
        structured,
    ),
]


for (
    comparison_name,
    method_a,
    method_b,
) in comparisons:


    difference = (
        method_b
        -
        method_a
    )


    (
        difference_mean,
        difference_ci_low,
        difference_ci_high,
    ) = mean_ci95(
        difference
    )


    wins = (
        method_a
        <
        method_b
    )


    ties = np.isclose(
        method_a,
        method_b,
        atol=1e-12,
    )


    losses = (
        method_a
        >
        method_b
    )


    paired_records.append(
        {

            "comparison":
                comparison_name,

            "n":
                len(
                    method_a
                ),

            "mean_quality_a":
                float(
                    np.mean(
                        method_a
                    )
                ),

            "mean_quality_b":
                float(
                    np.mean(
                        method_b
                    )
                ),

            "mean_improvement_b_minus_a":
                difference_mean,

            "ci95_improvement_low":
                difference_ci_low,

            "ci95_improvement_high":
                difference_ci_high,

            "win_percent_a":
                float(
                    np.mean(
                        wins
                    )
                    *
                    100.0
                ),

            "tie_percent":
                float(
                    np.mean(
                        ties
                    )
                    *
                    100.0
                ),

            "loss_percent_a":
                float(
                    np.mean(
                        losses
                    )
                    *
                    100.0
                ),

            "paired_effect_size_dz":
                paired_effect_size_dz(
                    method_b,
                    method_a,
                ),
        }
    )


paired_df = pd.DataFrame(
    paired_records
)


paired_df.to_csv(
    os.path.join(
        TABLE_DIR,
        "v3_2_final_paired_controller_statistics.csv",
    ),
    index=False,
)


# ============================================================
# 3. INDEPENDENT VALIDATION IMPROVEMENT DISTRIBUTION
# ============================================================

relative_improvement_percent = (

    (
        zero
        -
        structured
    )

    /

    np.maximum(
        np.abs(
            zero
        ),
        1e-9,
    )

    *
    100.0
)


improvement_summary_df = pd.DataFrame(
    {

        "metric": [

            "mean_relative_improvement_percent",

            "median_relative_improvement_percent",

            "p05_relative_improvement_percent",

            "p95_relative_improvement_percent",

            "minimum_relative_improvement_percent",

            "maximum_relative_improvement_percent",

            "structured_win_rate_percent",
        ],

        "value": [

            float(
                np.mean(
                    relative_improvement_percent
                )
            ),

            float(
                np.median(
                    relative_improvement_percent
                )
            ),

            float(
                np.quantile(
                    relative_improvement_percent,
                    0.05,
                )
            ),

            float(
                np.quantile(
                    relative_improvement_percent,
                    0.95,
                )
            ),

            float(
                np.min(
                    relative_improvement_percent
                )
            ),

            float(
                np.max(
                    relative_improvement_percent
                )
            ),

            float(
                np.mean(
                    structured
                    <
                    zero
                )
                *
                100.0
            ),
        ],
    }
)


improvement_summary_df.to_csv(
    os.path.join(
        TABLE_DIR,
        "v3_2_final_improvement_distribution_summary.csv",
    ),
    index=False,
)


# ============================================================
# 4. BO COMPUTATIONAL-VALUE TABLE
# ============================================================

bo_trigger_rate = float(
    component_df[
        "bo_triggered"
    ]
    .mean()
    *
    100.0
)


bo_assembly_rate = float(
    assembly_df[
        "n_bo_triggers"
    ]
    .gt(
        0
    )
    .mean()
    *
    100.0
)


bo_quality_difference = (
    selective
    -
    structured
)


(
    bo_difference_mean,
    bo_difference_ci_low,
    bo_difference_ci_high,
) = mean_ci95(
    bo_quality_difference
)


bo_value_df = pd.DataFrame(
    {

        "metric": [

            "bo_trigger_rate_percent",

            "assemblies_with_bo_percent",

            "mean_selective_minus_structured",

            "median_selective_minus_structured",

            "ci95_selective_minus_structured_low",

            "ci95_selective_minus_structured_high",

            "selective_better_percent",

            "structured_better_percent",

            "tie_percent",
        ],

        "value": [

            bo_trigger_rate,

            bo_assembly_rate,

            bo_difference_mean,

            float(
                np.median(
                    bo_quality_difference
                )
            ),

            bo_difference_ci_low,

            bo_difference_ci_high,

            float(
                np.mean(
                    selective
                    <
                    structured
                )
                *
                100.0
            ),

            float(
                np.mean(
                    structured
                    <
                    selective
                )
                *
                100.0
            ),

            float(
                np.mean(
                    np.isclose(
                        selective,
                        structured,
                        atol=1e-12,
                    )
                )
                *
                100.0
            ),
        ],
    }
)


bo_value_df.to_csv(
    os.path.join(
        TABLE_DIR,
        "v3_2_final_selective_bo_value.csv",
    ),
    index=False,
)


# ============================================================
# FIGURE 1
# FINAL STRATEGY COMPARISON
# ============================================================

plt.figure(
    figsize=(
        7.5,
        5.0,
    )
)


data_for_boxplot = [

    zero,

    structured,

    selective,
]


# IMPORTANT:
# Newer Matplotlib uses tick_labels rather than labels.
plt.boxplot(
    data_for_boxplot,
    tick_labels=[
        "Zero",
        "Structured-20",
        "Selective BO",
    ],
    showfliers=False,
)


plt.ylabel(
    "Final quality score"
)


plt.title(
    "Independent Validation: Final Assembly Quality"
)


plt.grid(
    axis="y",
    alpha=0.25,
)


save_figure(
    "figure_01_independent_strategy_comparison.png"
)


# ============================================================
# FIGURE 2
# STAGE-WISE QUALITY EVOLUTION
# ============================================================

plt.figure(
    figsize=(
        7.5,
        5.0,
    )
)


plt.plot(
    stage_df[
        "component_index"
    ],
    stage_df[
        "zero_mean_quality"
    ],
    marker="o",
    label="Zero correction",
)


plt.plot(
    stage_df[
        "component_index"
    ],
    stage_df[
        "structured_mean_quality"
    ],
    marker="o",
    label="Structured-20",
)


plt.plot(
    stage_df[
        "component_index"
    ],
    stage_df[
        "selective_mean_quality"
    ],
    marker="o",
    label="Selective BO",
)


plt.xlabel(
    "Sequential component stage"
)


plt.ylabel(
    "Mean quality score"
)


plt.title(
    "Quality Evolution Across Sequential Assembly Stages"
)


plt.xticks(
    stage_df[
        "component_index"
    ]
)


plt.legend()


plt.grid(
    alpha=0.25,
)


save_figure(
    "figure_02_stagewise_quality_evolution.png"
)


# ============================================================
# FIGURE 3
# BATCH-CONDITION COMPARISON
# ============================================================

batch_plot_df = (
    batch_df
    .sort_values(
        "zero_mean_final_quality"
    )
)


x = np.arange(
    len(
        batch_plot_df
    )
)


width = 0.26


plt.figure(
    figsize=(
        10.0,
        5.5,
    )
)


plt.bar(
    x - width,
    batch_plot_df[
        "zero_mean_final_quality"
    ],
    width=width,
    label="Zero",
)


plt.bar(
    x,
    batch_plot_df[
        "structured_mean_final_quality"
    ],
    width=width,
    label="Structured-20",
)


plt.bar(
    x + width,
    batch_plot_df[
        "selective_mean_final_quality"
    ],
    width=width,
    label="Selective BO",
)


plt.xticks(
    x,
    batch_plot_df[
        "batch_condition"
    ],
    rotation=25,
    ha="right",
)


plt.ylabel(
    "Mean final quality score"
)


plt.title(
    "Controller Performance Across Batch Conditions"
)


plt.legend()


plt.grid(
    axis="y",
    alpha=0.25,
)


save_figure(
    "figure_03_batch_condition_comparison.png"
)


# ============================================================
# FIGURE 4
# IMPROVEMENT DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(
        7.5,
        5.0,
    )
)


plt.hist(
    relative_improvement_percent,
    bins=25,
)


plt.axvline(
    np.mean(
        relative_improvement_percent
    ),
    linestyle="--",
    label=(
        f"Mean = "
        f"{np.mean(relative_improvement_percent):.1f}%"
    ),
)


plt.xlabel(
    "Structured-20 improvement vs Zero (%)"
)


plt.ylabel(
    "Number of assemblies"
)


plt.title(
    "Distribution of Assembly-Level Quality Improvement"
)


plt.legend()


plt.grid(
    axis="y",
    alpha=0.25,
)


save_figure(
    "figure_04_improvement_distribution.png"
)


# ============================================================
# FIGURE 5
# ROBUSTNESS ACROSS RANDOM SEEDS
# ============================================================

plt.figure(
    figsize=(
        7.5,
        5.0,
    )
)


plt.plot(
    seed_df[
        "seed"
    ]
    .astype(
        str
    ),
    seed_df[
        "structured_mean"
    ],
    marker="o",
    label="Structured-20 mean",
)


plt.plot(
    seed_df[
        "seed"
    ]
    .astype(
        str
    ),
    seed_df[
        "zero_mean"
    ],
    marker="o",
    label="Zero mean",
)


plt.xlabel(
    "Independent random seed"
)


plt.ylabel(
    "Mean final quality score"
)


plt.title(
    "Multi-Seed Robustness"
)


plt.legend()


plt.grid(
    alpha=0.25,
)


save_figure(
    "figure_05_multiseed_robustness.png"
)


# ============================================================
# FIGURE 6
# DEVIATION MAGNITUDE SENSITIVITY
# ============================================================

plt.figure(
    figsize=(
        7.5,
        5.0,
    )
)


plt.plot(
    magnitude_df[
        "deviation_factor"
    ],
    magnitude_df[
        "zero_mean"
    ],
    marker="o",
    label="Zero",
)


plt.plot(
    magnitude_df[
        "deviation_factor"
    ],
    magnitude_df[
        "structured_mean"
    ],
    marker="o",
    label="Structured-20",
)


plt.xlabel(
    "Deviation magnitude factor"
)


plt.ylabel(
    "Mean final quality score"
)


plt.title(
    "Sensitivity to Component-Deviation Magnitude"
)


plt.xticks(
    magnitude_df[
        "deviation_factor"
    ]
)


plt.legend()


plt.grid(
    alpha=0.25,
)


save_figure(
    "figure_06_deviation_magnitude_sensitivity.png"
)


# ============================================================
# FIGURE 7
# CORRECTION CAPABILITY SENSITIVITY
# ============================================================

capability_plot_df = (
    capability_df
    .sort_values(
        "capability_factor"
    )
)


plt.figure(
    figsize=(
        7.5,
        5.0,
    )
)


plt.plot(
    capability_plot_df[
        "capability_factor"
    ]
    *
    100.0,
    capability_plot_df[
        "structured_mean"
    ],
    marker="o",
)


plt.xlabel(
    "Available correction capability (%)"
)


plt.ylabel(
    "Structured-20 mean final quality"
)


plt.title(
    "Sensitivity to Available Correction Capability"
)


plt.xticks(
    capability_plot_df[
        "capability_factor"
    ]
    *
    100.0
)


plt.grid(
    alpha=0.25,
)


save_figure(
    "figure_07_correction_capability_sensitivity.png"
)


# ============================================================
# FIGURE 8
# QUALITY WEIGHT SENSITIVITY
# ============================================================

weight_plot_df = (
    weight_df
    .sort_values(
        "mean_improvement_percent"
    )
)


plt.figure(
    figsize=(
        8.5,
        5.0,
    )
)


plt.bar(
    weight_plot_df[
        "weight_set"
    ],
    weight_plot_df[
        "mean_improvement_percent"
    ],
)


plt.ylabel(
    "Mean improvement vs Zero (%)"
)


plt.xlabel(
    "Quality-score weighting"
)


plt.title(
    "Sensitivity to Alternative Quality-Score Definitions"
)


plt.xticks(
    rotation=20,
    ha="right",
)


plt.grid(
    axis="y",
    alpha=0.25,
)


save_figure(
    "figure_08_quality_weight_sensitivity.png"
)


# ============================================================
# FIGURE 9
# CAPABILITY UTILIZATION DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(
        7.5,
        5.0,
    )
)


plt.hist(
    component_df[
        "structured_utilization"
    ],
    bins=25,
)


plt.axvline(
    0.90,
    linestyle="--",
    label="90% capability",
)


plt.xlabel(
    "Correction capability utilization"
)


plt.ylabel(
    "Number of decisions"
)


plt.title(
    "Structured-20 Correction Capability Utilization"
)


plt.legend()


plt.grid(
    axis="y",
    alpha=0.25,
)


save_figure(
    "figure_09_capability_utilization.png"
)


# ============================================================
# FIGURE 10
# SELECTIVE BO TRIGGER BY STAGE
# ============================================================

plt.figure(
    figsize=(
        7.5,
        5.0,
    )
)


plt.bar(
    stage_df[
        "component_index"
    ],
    stage_df[
        "bo_trigger_rate_percent"
    ],
)


plt.xlabel(
    "Sequential component stage"
)


plt.ylabel(
    "BO trigger rate (%)"
)


plt.title(
    "Selective Bayesian Optimization Trigger Frequency"
)


plt.xticks(
    stage_df[
        "component_index"
    ]
)


plt.grid(
    axis="y",
    alpha=0.25,
)


save_figure(
    "figure_10_bo_trigger_by_stage.png"
)


# ============================================================
# FINAL EVIDENCE MASTER TABLE
# ============================================================

zero_mean = float(
    np.mean(
        zero
    )
)


structured_mean = float(
    np.mean(
        structured
    )
)


selective_mean = float(
    np.mean(
        selective
    )
)


system_improvement_percent = (

    (
        zero_mean
        -
        structured_mean
    )

    /

    zero_mean

    *
    100.0
)


minimum_seed_win = float(
    seed_df[
        "win_vs_zero_percent"
    ]
    .min()
)


minimum_magnitude_win = float(
    magnitude_df[
        "win_vs_zero_percent"
    ]
    .min()
)


minimum_weight_win = float(
    weight_df[
        "min_win_vs_zero_percent"
    ]
    .min()
)


capability_60_rows = (
    capability_df[
        np.isclose(
            capability_df[
                "capability_factor"
            ],
            0.60,
        )
    ]
)


if capability_60_rows.empty:

    raise ValueError(
        "\nCould not find the 60% capability sensitivity row."
    )


capability_60 = (
    capability_60_rows
    .iloc[
        0
    ]
)


master_evidence_df = pd.DataFrame(
    {

        "evidence_item": [

            "Independent validation assemblies",

            "Independent sequential decisions",

            "Zero mean final quality",

            "Structured-20 mean final quality",

            "Selective BO mean final quality",

            "Structured-20 mean improvement vs zero (%)",

            "Structured-20 assembly win rate vs zero (%)",

            "Structured-20 exact-best candidate rate (%)",

            "Structured-20 top-3 candidate rate (%)",

            "Mean Structured-20 capability utilization",

            "Structured-20 >=90% capability rate (%)",

            "Selective BO trigger rate (%)",

            "Minimum multi-seed win rate (%)",

            "Minimum deviation-magnitude win rate (%)",

            "60% correction capability win rate (%)",

            "Minimum quality-weight win rate (%)",
        ],

        "value": [

            len(
                assembly_df
            ),

            len(
                component_df
            ),

            zero_mean,

            structured_mean,

            selective_mean,

            system_improvement_percent,

            float(
                np.mean(
                    structured
                    <
                    zero
                )
                *
                100.0
            ),

            float(
                component_df[
                    "structured_exact_best_hit"
                ]
                .mean()
                *
                100.0
            ),

            float(
                component_df[
                    "structured_top3_hit"
                ]
                .mean()
                *
                100.0
            ),

            float(
                component_df[
                    "structured_utilization"
                ]
                .mean()
            ),

            float(
                component_df[
                    "structured_utilization"
                ]
                .ge(
                    0.90
                )
                .mean()
                *
                100.0
            ),

            bo_trigger_rate,

            minimum_seed_win,

            minimum_magnitude_win,

            float(
                capability_60[
                    "win_vs_zero_percent"
                ]
            ),

            minimum_weight_win,
        ],
    }
)


master_evidence_df.to_csv(
    os.path.join(
        TABLE_DIR,
        "v3_2_FINAL_MASTER_EVIDENCE_TABLE.csv",
    ),
    index=False,
)


# ============================================================
# SAVE COPIES OF IMPORTANT EXISTING TABLES
# ============================================================

stage_df.to_csv(
    os.path.join(
        TABLE_DIR,
        "v3_2_final_stagewise_results.csv",
    ),
    index=False,
)


batch_df.to_csv(
    os.path.join(
        TABLE_DIR,
        "v3_2_final_batch_condition_results.csv",
    ),
    index=False,
)


severity_df.to_csv(
    os.path.join(
        TABLE_DIR,
        "v3_2_final_severity_results.csv",
    ),
    index=False,
)


scenario_df.to_csv(
    os.path.join(
        TABLE_DIR,
        "v3_2_final_scenario_results.csv",
    ),
    index=False,
)


seed_df.to_csv(
    os.path.join(
        TABLE_DIR,
        "v3_2_final_multiseed_results.csv",
    ),
    index=False,
)


magnitude_df.to_csv(
    os.path.join(
        TABLE_DIR,
        "v3_2_final_magnitude_sensitivity_results.csv",
    ),
    index=False,
)


capability_df.to_csv(
    os.path.join(
        TABLE_DIR,
        "v3_2_final_capability_sensitivity_results.csv",
    ),
    index=False,
)


weight_df.to_csv(
    os.path.join(
        TABLE_DIR,
        "v3_2_final_weight_sensitivity_results.csv",
    ),
    index=False,
)


# ============================================================
# TERMINAL SUMMARY
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "FINAL EVIDENCE SUMMARY"
)

print(
    "============================================================"
)


print(
    f"\nZero mean final quality       : "
    f"{zero_mean:.4f}"
)


print(
    f"Structured-20 mean quality    : "
    f"{structured_mean:.4f}"
)


print(
    f"Selective BO mean quality     : "
    f"{selective_mean:.4f}"
)


print(
    f"\nStructured improvement        : "
    f"{system_improvement_percent:.2f}%"
)


print(
    f"Structured win vs Zero        : "
    f"{np.mean(structured < zero) * 100.0:.2f}%"
)


print(
    f"\nMinimum multi-seed win        : "
    f"{minimum_seed_win:.2f}%"
)


print(
    f"Minimum magnitude win         : "
    f"{minimum_magnitude_win:.2f}%"
)


print(
    f"60% capability win            : "
    f"{float(capability_60['win_vs_zero_percent']):.2f}%"
)


print(
    f"Minimum quality-weight win    : "
    f"{minimum_weight_win:.2f}%"
)


print(
    f"\nBO trigger rate               : "
    f"{bo_trigger_rate:.2f}%"
)


print(
    "\n------------------------------------------------------------"
)


print(
    "FILES GENERATED"
)


print(
    "------------------------------------------------------------"
)


print(
    "\nFigures generated:"
)


figure_files = sorted(
    [
        file
        for file in os.listdir(
            FIGURE_DIR
        )
        if file.endswith(
            ".png"
        )
    ]
)


for file in figure_files:

    print(
        f"  {file}"
    )


print(
    f"\nTotal figures generated       : "
    f"{len(figure_files)}"
)


print(
    "\nTables generated:"
)


table_files = sorted(
    [
        file
        for file in os.listdir(
            TABLE_DIR
        )
        if file.endswith(
            ".csv"
        )
    ]
)


for file in table_files:

    print(
        f"  {file}"
    )


print(
    f"\nTotal tables generated        : "
    f"{len(table_files)}"
)


print(
    "\nFigure folder:"
)


print(
    FIGURE_DIR
)


print(
    "\nTable folder:"
)


print(
    TABLE_DIR
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.2 FINAL EVIDENCE PACKAGE COMPLETED"
)

print(
    "============================================================"
)