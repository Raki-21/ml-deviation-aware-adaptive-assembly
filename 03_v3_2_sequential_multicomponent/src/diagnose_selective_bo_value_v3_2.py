import os
import numpy as np
import pandas as pd


# ============================================================
# V3.2
# SELECTIVE BAYESIAN OPTIMIZATION VALUE DIAGNOSIS
# ============================================================
#
# PURPOSE
#
# Equal-budget ablation showed that:
#
#   Structured-20 RF search
#   and
#   Structured warm-start BO
#
# perform almost identically at system level.
#
# However, Bayesian refinement was selected only in a small
# fraction of component-level decisions.
#
# This script asks:
#
#   1. Which assemblies actually benefit from BO?
#   2. How often is BO refinement selected?
#   3. At which component stages is it selected?
#   4. Is BO use associated with larger RF error or regret?
#   5. Are certain batch/severity/scenario conditions more
#      strongly associated with BO refinement?
#
#
# IMPORTANT
#
# This is a POST-HOC diagnostic.
#
# Prediction error and regret are outcome variables and MUST
# NOT later be used directly as real-time triggers because
# they require simulator truth.
#
# The purpose here is only to understand where BO adds value
# and determine what PRE-DECISION trigger should be tested.
# ============================================================


# ============================================================
# FILES
# ============================================================

WARM_COMPONENT_FILE = (
    "data/processed/"
    "v3_2_warmstart_bo_100_component_results.csv"
)

WARM_ASSEMBLY_FILE = (
    "data/processed/"
    "v3_2_warmstart_bo_100_assembly_results.csv"
)

STRUCTURED_COMPONENT_FILE = (
    "data/processed/"
    "v3_2_equal_budget_structured20_component_results.csv"
)

STRUCTURED_ASSEMBLY_FILE = (
    "data/processed/"
    "v3_2_equal_budget_structured20_assembly_results.csv"
)


DETAIL_OUTPUT = (
    "results/tables/"
    "v3_2_selective_bo_assembly_diagnosis.csv"
)

STAGE_OUTPUT = (
    "results/tables/"
    "v3_2_selective_bo_by_stage.csv"
)

BATCH_OUTPUT = (
    "results/tables/"
    "v3_2_selective_bo_by_batch.csv"
)

SEVERITY_OUTPUT = (
    "results/tables/"
    "v3_2_selective_bo_by_severity.csv"
)

SCENARIO_OUTPUT = (
    "results/tables/"
    "v3_2_selective_bo_by_scenario.csv"
)

SUMMARY_OUTPUT = (
    "results/tables/"
    "v3_2_selective_bo_value_summary.csv"
)


# ============================================================
# FILE CHECK
# ============================================================

required_files = [
    WARM_COMPONENT_FILE,
    WARM_ASSEMBLY_FILE,
    STRUCTURED_COMPONENT_FILE,
    STRUCTURED_ASSEMBLY_FILE,
]


missing_files = [
    path
    for path in required_files
    if not os.path.exists(path)
]


if missing_files:

    print(
        "\nERROR - Missing required files:"
    )

    for path in missing_files:

        print(path)

    raise SystemExit(
        "\nComplete the 100-assembly warm-start BO and "
        "equal-budget ablation first."
    )


# ============================================================
# LOAD
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "V3.2 SELECTIVE BO VALUE DIAGNOSIS"
)

print(
    "============================================================"
)


warm_component_df = pd.read_csv(
    WARM_COMPONENT_FILE
)


warm_assembly_df = pd.read_csv(
    WARM_ASSEMBLY_FILE
)


structured_component_df = pd.read_csv(
    STRUCTURED_COMPONENT_FILE
)


structured_assembly_df = pd.read_csv(
    STRUCTURED_ASSEMBLY_FILE
)


print(
    f"\nWarm-start component decisions : "
    f"{len(warm_component_df)}"
)

print(
    f"Warm-start assemblies          : "
    f"{len(warm_assembly_df)}"
)


# ============================================================
# VALIDATE KEY COLUMNS
# ============================================================

required_warm_component_columns = [
    "assembly_id",
    "batch_condition",
    "component_index",
    "severity",
    "scenario_type",
    "actual_quality",
    "prediction_absolute_error",
    "absolute_regret",
    "selected_beats_zero",
    "best_was_structured_initial",
    "correction_utilization",
]


missing_columns = [
    column
    for column in required_warm_component_columns
    if column not in warm_component_df.columns
]


if missing_columns:

    raise ValueError(
        "\nMissing warm-start component columns:\n"
        +
        "\n".join(
            missing_columns
        )
    )


# ============================================================
# BO REFINEMENT FLAG
# ============================================================

warm_component_df[
    "bo_refinement_selected"
] = (
    ~warm_component_df[
        "best_was_structured_initial"
    ].astype(
        bool
    )
)


# ============================================================
# ASSEMBLY-LEVEL PAIRED RESULT
# ============================================================

assembly_df = (
    structured_assembly_df[
        [
            "assembly_id",
            "batch_condition",
            "structured20_final_quality",
        ]
    ]
    .merge(
        warm_assembly_df[
            [
                "assembly_id",
                "warmstart_bo_final_quality",
            ]
        ],
        on="assembly_id",
        how="inner",
    )
)


assembly_df[
    "bo_quality_benefit"
] = (
    assembly_df[
        "structured20_final_quality"
    ]
    -
    assembly_df[
        "warmstart_bo_final_quality"
    ]
)


# Positive:
# BO better.
#
# Negative:
# Structured-20 better.

assembly_df[
    "bo_better"
] = (
    assembly_df[
        "bo_quality_benefit"
    ]
    >
    1e-9
)


assembly_df[
    "structured_better"
] = (
    assembly_df[
        "bo_quality_benefit"
    ]
    <
    -1e-9
)


assembly_df[
    "essential_tie"
] = (
    np.abs(
        assembly_df[
            "bo_quality_benefit"
        ]
    )
    <=
    1e-9
)


# ============================================================
# COMPONENT-LEVEL BO USE PER ASSEMBLY
# ============================================================

assembly_controller_stats = (
    warm_component_df
    .groupby(
        "assembly_id"
    )
    .agg(

        n_component_decisions=(
            "component_index",
            "count",
        ),

        n_bo_refinements=(
            "bo_refinement_selected",
            "sum",
        ),

        mean_prediction_abs_error=(
            "prediction_absolute_error",
            "mean",
        ),

        max_prediction_abs_error=(
            "prediction_absolute_error",
            "max",
        ),

        mean_regret=(
            "absolute_regret",
            "mean",
        ),

        max_regret=(
            "absolute_regret",
            "max",
        ),

        mean_utilization=(
            "correction_utilization",
            "mean",
        ),

        max_utilization=(
            "correction_utilization",
            "max",
        ),
    )
    .reset_index()
)


assembly_controller_stats[
    "bo_refinement_rate_percent"
] = (
    assembly_controller_stats[
        "n_bo_refinements"
    ]
    /
    assembly_controller_stats[
        "n_component_decisions"
    ]
    *
    100.0
)


assembly_df = assembly_df.merge(
    assembly_controller_stats,
    on="assembly_id",
    how="left",
)


# ============================================================
# ASSEMBLIES WHERE BO WAS EVER SELECTED
# ============================================================

assembly_df[
    "bo_used_at_least_once"
] = (
    assembly_df[
        "n_bo_refinements"
    ]
    >
    0
)


# ============================================================
# GENERAL SUMMARY
# ============================================================

n_assemblies = len(
    assembly_df
)


n_bo_better = int(
    assembly_df[
        "bo_better"
    ]
    .sum()
)


n_structured_better = int(
    assembly_df[
        "structured_better"
    ]
    .sum()
)


n_ties = int(
    assembly_df[
        "essential_tie"
    ]
    .sum()
)


n_bo_used = int(
    assembly_df[
        "bo_used_at_least_once"
    ]
    .sum()
)


decision_bo_rate = (
    warm_component_df[
        "bo_refinement_selected"
    ]
    .mean()
    *
    100.0
)


mean_benefit = (
    assembly_df[
        "bo_quality_benefit"
    ]
    .mean()
)


median_benefit = (
    assembly_df[
        "bo_quality_benefit"
    ]
    .median()
)


p10_benefit = (
    assembly_df[
        "bo_quality_benefit"
    ]
    .quantile(
        0.10
    )
)


p90_benefit = (
    assembly_df[
        "bo_quality_benefit"
    ]
    .quantile(
        0.90
    )
)


# ============================================================
# ASSEMBLIES WITH vs WITHOUT BO USE
# ============================================================

use_group_rows = []


for flag, label in [
    (
        False,
        "no_bo_refinement_selected",
    ),
    (
        True,
        "bo_refinement_selected_at_least_once",
    ),
]:

    subset = assembly_df[
        assembly_df[
            "bo_used_at_least_once"
        ]
        ==
        flag
    ]


    if len(
        subset
    ) == 0:

        continue


    use_group_rows.append(
        {

            "group":
                label,

            "n_assemblies":
                len(
                    subset
                ),

            "mean_bo_quality_benefit":
                subset[
                    "bo_quality_benefit"
                ]
                .mean(),

            "median_bo_quality_benefit":
                subset[
                    "bo_quality_benefit"
                ]
                .median(),

            "bo_better_rate_percent":
                subset[
                    "bo_better"
                ]
                .mean()
                *
                100.0,

            "structured_better_rate_percent":
                subset[
                    "structured_better"
                ]
                .mean()
                *
                100.0,

            "mean_prediction_abs_error":
                subset[
                    "mean_prediction_abs_error"
                ]
                .mean(),

            "mean_regret":
                subset[
                    "mean_regret"
                ]
                .mean(),

            "mean_utilization":
                subset[
                    "mean_utilization"
                ]
                .mean(),
        }
    )


use_group_df = pd.DataFrame(
    use_group_rows
)


# ============================================================
# STAGE-WISE BO SELECTION
# ============================================================

stage_rows = []


for stage in sorted(
    warm_component_df[
        "component_index"
    ]
    .unique()
):


    subset = warm_component_df[
        warm_component_df[
            "component_index"
        ]
        ==
        stage
    ]


    selected_subset = subset[
        subset[
            "bo_refinement_selected"
        ]
    ]


    stage_rows.append(
        {

            "component_index":
                int(
                    stage
                ),

            "n_decisions":
                len(
                    subset
                ),

            "n_bo_refinements":
                int(
                    selected_subset.shape[
                        0
                    ]
                ),

            "bo_refinement_rate_percent":
                subset[
                    "bo_refinement_selected"
                ]
                .mean()
                *
                100.0,

            "mean_prediction_error_all":
                subset[
                    "prediction_absolute_error"
                ]
                .mean(),

            "mean_prediction_error_when_bo_selected":
                (
                    selected_subset[
                        "prediction_absolute_error"
                    ]
                    .mean()
                    if len(
                        selected_subset
                    )
                    >
                    0
                    else np.nan
                ),

            "mean_regret_all":
                subset[
                    "absolute_regret"
                ]
                .mean(),

            "mean_regret_when_bo_selected":
                (
                    selected_subset[
                        "absolute_regret"
                    ]
                    .mean()
                    if len(
                        selected_subset
                    )
                    >
                    0
                    else np.nan
                ),
        }
    )


stage_df = pd.DataFrame(
    stage_rows
)


# ============================================================
# CATEGORY ANALYSIS FUNCTION
# ============================================================

def category_analysis(
    column
):

    rows = []


    for category in sorted(
        warm_component_df[
            column
        ]
        .dropna()
        .unique()
    ):


        subset = warm_component_df[
            warm_component_df[
                column
            ]
            ==
            category
        ]


        selected_subset = subset[
            subset[
                "bo_refinement_selected"
            ]
        ]


        rows.append(
            {

                column:
                    category,

                "n_decisions":
                    len(
                        subset
                    ),

                "n_bo_refinements":
                    int(
                        selected_subset.shape[
                            0
                        ]
                    ),

                "bo_refinement_rate_percent":
                    subset[
                        "bo_refinement_selected"
                    ]
                    .mean()
                    *
                    100.0,

                "mean_prediction_abs_error":
                    subset[
                        "prediction_absolute_error"
                    ]
                    .mean(),

                "mean_regret":
                    subset[
                        "absolute_regret"
                    ]
                    .mean(),

                "selected_beats_zero_percent":
                    subset[
                        "selected_beats_zero"
                    ]
                    .mean()
                    *
                    100.0,

                "mean_utilization":
                    subset[
                        "correction_utilization"
                    ]
                    .mean(),
            }
        )


    return pd.DataFrame(
        rows
    )


# ============================================================
# CATEGORY TABLES
# ============================================================

batch_df = category_analysis(
    "batch_condition"
)


severity_df = category_analysis(
    "severity"
)


scenario_df = category_analysis(
    "scenario_type"
)


# ============================================================
# TOP ASSEMBLIES WHERE BO HELPED / HURT
# ============================================================

best_bo_cases = (
    assembly_df
    .sort_values(
        "bo_quality_benefit",
        ascending=False,
    )
    .head(
        10
    )
)


worst_bo_cases = (
    assembly_df
    .sort_values(
        "bo_quality_benefit",
        ascending=True,
    )
    .head(
        10
    )
)


# ============================================================
# CORRELATION - DIAGNOSTIC ONLY
# ============================================================

correlation_columns = [
    "bo_quality_benefit",
    "n_bo_refinements",
    "bo_refinement_rate_percent",
    "mean_prediction_abs_error",
    "max_prediction_abs_error",
    "mean_regret",
    "max_regret",
    "mean_utilization",
    "max_utilization",
]


correlation_df = (
    assembly_df[
        correlation_columns
    ]
    .corr()
)


bo_benefit_correlations = (
    correlation_df[
        "bo_quality_benefit"
    ]
    .drop(
        "bo_quality_benefit"
    )
    .sort_values(
        ascending=False
    )
)


# ============================================================
# SUMMARY TABLE
# ============================================================

summary_df = pd.DataFrame(
    {

        "metric": [

            "n_assemblies",

            "bo_better_assemblies",

            "structured_better_assemblies",

            "tied_assemblies",

            "assemblies_where_bo_refinement_selected",

            "bo_refinement_decision_rate_percent",

            "mean_bo_quality_benefit",

            "median_bo_quality_benefit",

            "p10_bo_quality_benefit",

            "p90_bo_quality_benefit",
        ],

        "value": [

            n_assemblies,

            n_bo_better,

            n_structured_better,

            n_ties,

            n_bo_used,

            decision_bo_rate,

            mean_benefit,

            median_benefit,

            p10_benefit,

            p90_benefit,
        ],
    }
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "results/tables",
    exist_ok=True,
)


assembly_df.to_csv(
    DETAIL_OUTPUT,
    index=False,
)


stage_df.to_csv(
    STAGE_OUTPUT,
    index=False,
)


batch_df.to_csv(
    BATCH_OUTPUT,
    index=False,
)


severity_df.to_csv(
    SEVERITY_OUTPUT,
    index=False,
)


scenario_df.to_csv(
    SCENARIO_OUTPUT,
    index=False,
)


summary_df.to_csv(
    SUMMARY_OUTPUT,
    index=False,
)


use_group_df.to_csv(
    "results/tables/"
    "v3_2_selective_bo_use_group_comparison.csv",
    index=False,
)


best_bo_cases.to_csv(
    "results/tables/"
    "v3_2_selective_bo_top_helpful_assemblies.csv",
    index=False,
)


worst_bo_cases.to_csv(
    "results/tables/"
    "v3_2_selective_bo_top_harmful_assemblies.csv",
    index=False,
)


bo_benefit_correlations.to_csv(
    "results/tables/"
    "v3_2_selective_bo_benefit_correlations.csv"
)


# ============================================================
# PRINT
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "SELECTIVE BO DIAGNOSIS SUMMARY"
)

print(
    "============================================================"
)


print(
    f"\nAssemblies                     : "
    f"{n_assemblies}"
)


print(
    f"BO better assemblies           : "
    f"{n_bo_better}"
)


print(
    f"Structured better assemblies   : "
    f"{n_structured_better}"
)


print(
    f"Tied assemblies                : "
    f"{n_ties}"
)


print(
    f"\nAssemblies with BO refinement  : "
    f"{n_bo_used}"
)


print(
    f"BO-selected decision rate      : "
    f"{decision_bo_rate:.2f}%"
)


print(
    f"\nMean BO quality benefit        : "
    f"{mean_benefit:.6f}"
)


print(
    f"Median BO quality benefit      : "
    f"{median_benefit:.6f}"
)


print(
    f"P10 benefit                    : "
    f"{p10_benefit:.6f}"
)


print(
    f"P90 benefit                    : "
    f"{p90_benefit:.6f}"
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "ASSEMBLIES WITH vs WITHOUT BO REFINEMENT"
)

print(
    "------------------------------------------------------------\n"
)


print(
    use_group_df
    .round(
        6
    )
    .to_string(
        index=False
    )
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "BO REFINEMENT BY COMPONENT STAGE"
)

print(
    "------------------------------------------------------------\n"
)


print(
    stage_df
    .round(
        6
    )
    .to_string(
        index=False
    )
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "BO REFINEMENT BY BATCH CONDITION"
)

print(
    "------------------------------------------------------------\n"
)


print(
    batch_df
    .round(
        6
    )
    .to_string(
        index=False
    )
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "BO REFINEMENT BY SEVERITY"
)

print(
    "------------------------------------------------------------\n"
)


print(
    severity_df
    .round(
        6
    )
    .to_string(
        index=False
    )
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "CORRELATION WITH BO QUALITY BENEFIT"
)

print(
    "------------------------------------------------------------\n"
)


print(
    bo_benefit_correlations
    .round(
        4
    )
    .to_string()
)


# ============================================================
# INTERPRETATION
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "SELECTIVE BO INTERPRETATION"
)

print(
    "============================================================"
)


if (
    n_bo_used
    <
    0.25
    *
    n_assemblies
):


    print(
        "\nSIGNAL:"
    )


    print(
        "Bayesian refinement is used in only a minority "
        "of completed assemblies."
    )


    print(
        "This supports investigating an event-triggered / "
        "selective BO architecture."
    )


if (
    abs(
        mean_benefit
    )
    <
    0.01
):


    print(
        "\nSIGNAL:"
    )


    print(
        "The average system-level effect of BO is very small."
    )


    print(
        "A selective trigger must therefore justify BO through "
        "case-specific benefit rather than average benefit."
    )


print(
    "\nIMPORTANT SCIENTIFIC RULE:"
)


print(
    "Prediction error and regret shown here are POST-HOC "
    "diagnostics."
)


print(
    "They cannot be used as the final online trigger because "
    "their true values are unknown before assembly."
)


print(
    "\nNEXT STEP AFTER INTERPRETING THIS TABLE:"
)


print(
    "Construct a PRE-DECISION ambiguity signal using only "
    "information available before applying the correction, "
    "such as:"
)


print(
    "  - RF best-vs-second-best predicted-quality margin"
)


print(
    "  - disagreement across Random-Forest trees"
)


print(
    "  - correction utilization / boundary proximity"
)


print(
    "Then test whether selectively invoking BO on ambiguous "
    "cases preserves quality while reducing computational cost."
)


print(
    "\nSaved:"
)


print(
    DETAIL_OUTPUT
)

print(
    STAGE_OUTPUT
)

print(
    BATCH_OUTPUT
)

print(
    SEVERITY_OUTPUT
)

print(
    SCENARIO_OUTPUT
)

print(
    SUMMARY_OUTPUT
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.2 SELECTIVE BO VALUE DIAGNOSIS COMPLETED"
)

print(
    "============================================================"
)