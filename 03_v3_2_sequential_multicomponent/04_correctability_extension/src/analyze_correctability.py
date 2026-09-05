
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# V3.2 CORRECTABILITY ANALYSIS
# ============================================================
#
# Purpose:
#
# Explore whether the final independent validation results
# contain a meaningful correctability pattern.
#
# This script only reads existing frozen V3.2 results.
# It does not retrain models or modify the controller.
# ============================================================


# ============================================================
# INPUT FILES
# ============================================================

COMPONENT_FILE = (
    "data/validation/"
    "v3_2_final_controller_validation_component_results.csv"
)

ASSEMBLY_FILE = (
    "data/validation/"
    "v3_2_final_controller_validation_assembly_results.csv"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    "04_correctability_extension/"
    "results"
)

FIGURE_DIR = os.path.join(
    OUTPUT_DIR,
    "figures",
)

TABLE_DIR = os.path.join(
    OUTPUT_DIR,
    "tables",
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

for file in [
    COMPONENT_FILE,
    ASSEMBLY_FILE,
]:

    if not os.path.exists(
        file
    ):

        raise FileNotFoundError(
            f"\nMissing required file:\n{file}"
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
    "\n"
    "============================================================"
)

print(
    "V3.2 CORRECTABILITY ANALYSIS"
)

print(
    "============================================================"
)


print(
    f"\nAssemblies          : "
    f"{len(assembly_df)}"
)


print(
    f"Component decisions : "
    f"{len(component_df)}"
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_component_columns = [
    "assembly_id",
    "component_index",
    "batch_condition",
    "severity",
    "scenario_type",
    "zero_actual_quality",
    "structured_actual_quality",
    "structured_true_best_quality",
    "structured_absolute_regret",
    "structured_utilization",
    "structured_exact_best_hit",
    "structured_top3_hit",
]


missing_columns = [
    column
    for column in required_component_columns
    if column not in component_df.columns
]


if missing_columns:

    raise ValueError(
        "\nMissing expected columns:\n"
        +
        "\n".join(
            missing_columns
        )
    )


# ============================================================
# BASIC CORRECTABILITY DESCRIPTORS
# ============================================================

EPSILON = 1e-9


component_df[
    "quality_improvement"
] = (
    component_df[
        "zero_actual_quality"
    ]
    -
    component_df[
        "structured_actual_quality"
    ]
)


component_df[
    "quality_improvement_percent"
] = (
    component_df[
        "quality_improvement"
    ]
    /
    np.maximum(
        np.abs(
            component_df[
                "zero_actual_quality"
            ]
        ),
        EPSILON,
    )
    *
    100.0
)


component_df[
    "residual_quality_ratio"
] = (
    component_df[
        "structured_actual_quality"
    ]
    /
    np.maximum(
        np.abs(
            component_df[
                "zero_actual_quality"
            ]
        ),
        EPSILON,
    )
)


component_df[
    "controller_improved_case"
] = (
    component_df[
        "structured_actual_quality"
    ]
    <
    component_df[
        "zero_actual_quality"
    ]
)


component_df[
    "controller_worsened_case"
] = (
    component_df[
        "structured_actual_quality"
    ]
    >
    component_df[
        "zero_actual_quality"
    ]
)


# ============================================================
# IMPROVEMENT PER UTILIZATION
# ============================================================

component_df[
    "improvement_per_utilization"
] = (
    component_df[
        "quality_improvement"
    ]
    /
    np.maximum(
        component_df[
            "structured_utilization"
        ],
        0.01,
    )
)


# ============================================================
# DATA-DRIVEN REFERENCE THRESHOLDS
# ============================================================

improvement_q25 = float(
    component_df[
        "quality_improvement_percent"
    ]
    .quantile(
        0.25
    )
)


improvement_q50 = float(
    component_df[
        "quality_improvement_percent"
    ]
    .quantile(
        0.50
    )
)


improvement_q75 = float(
    component_df[
        "quality_improvement_percent"
    ]
    .quantile(
        0.75
    )
)


residual_q25 = float(
    component_df[
        "structured_actual_quality"
    ]
    .quantile(
        0.25
    )
)


residual_q50 = float(
    component_df[
        "structured_actual_quality"
    ]
    .quantile(
        0.50
    )
)


residual_q75 = float(
    component_df[
        "structured_actual_quality"
    ]
    .quantile(
        0.75
    )
)


utilization_q50 = float(
    component_df[
        "structured_utilization"
    ]
    .quantile(
        0.50
    )
)


utilization_q75 = float(
    component_df[
        "structured_utilization"
    ]
    .quantile(
        0.75
    )
)


regret_q75 = float(
    component_df[
        "structured_absolute_regret"
    ]
    .quantile(
        0.75
    )
)


regret_q90 = float(
    component_df[
        "structured_absolute_regret"
    ]
    .quantile(
        0.90
    )
)


# ============================================================
# EXPLORATORY CORRECTABILITY CLASSES
# ============================================================

def classify_correctability(
    row
):

    improvement_percent = (
        row[
            "quality_improvement_percent"
        ]
    )

    residual_quality = (
        row[
            "structured_actual_quality"
        ]
    )

    utilization = (
        row[
            "structured_utilization"
        ]
    )


    if not row[
        "controller_improved_case"
    ]:

        return "NOT_IMPROVED"


    if (
        improvement_percent
        >=
        improvement_q75
        and
        residual_quality
        <=
        residual_q25
        and
        utilization
        <=
        utilization_q75
    ):

        return "HIGHLY_CORRECTABLE"


    if (
        improvement_percent
        >=
        improvement_q25
        and
        residual_quality
        <=
        residual_q75
    ):

        return "CORRECTABLE"


    return "DIFFICULT"


component_df[
    "observed_correctability_class"
] = component_df.apply(
    classify_correctability,
    axis=1,
)


# ============================================================
# DIFFICULTY FLAGS
# ============================================================

component_df[
    "high_regret_case"
] = (
    component_df[
        "structured_absolute_regret"
    ]
    >=
    regret_q90
)


component_df[
    "higher_capability_case"
] = (
    component_df[
        "structured_utilization"
    ]
    >=
    utilization_q75
)


component_df[
    "higher_residual_case"
] = (
    component_df[
        "structured_actual_quality"
    ]
    >=
    residual_q75
)


# ============================================================
# CLASS SUMMARY
# ============================================================

class_summary = (
    component_df
    .groupby(
        "observed_correctability_class"
    )
    .agg(

        decisions=(
            "assembly_id",
            "count",
        ),

        mean_zero_quality=(
            "zero_actual_quality",
            "mean",
        ),

        mean_corrected_quality=(
            "structured_actual_quality",
            "mean",
        ),

        mean_improvement_percent=(
            "quality_improvement_percent",
            "mean",
        ),

        median_improvement_percent=(
            "quality_improvement_percent",
            "median",
        ),

        mean_utilization=(
            "structured_utilization",
            "mean",
        ),

        mean_regret=(
            "structured_absolute_regret",
            "mean",
        ),

        exact_best_percent=(
            "structured_exact_best_hit",
            lambda x:
                x.mean()
                *
                100.0
        ),

        top3_percent=(
            "structured_top3_hit",
            lambda x:
                x.mean()
                *
                100.0
        ),
    )
    .reset_index()
)


class_summary[
    "percentage"
] = (
    class_summary[
        "decisions"
    ]
    /
    len(
        component_df
    )
    *
    100.0
)


# ============================================================
# BATCH SUMMARY
# ============================================================

batch_summary = (
    component_df
    .groupby(
        "batch_condition"
    )
    .agg(

        decisions=(
            "assembly_id",
            "count",
        ),

        improvement_rate_percent=(
            "controller_improved_case",
            lambda x:
                x.mean()
                *
                100.0
        ),

        mean_improvement_percent=(
            "quality_improvement_percent",
            "mean",
        ),

        mean_residual_quality=(
            "structured_actual_quality",
            "mean",
        ),

        mean_utilization=(
            "structured_utilization",
            "mean",
        ),

        high_regret_percent=(
            "high_regret_case",
            lambda x:
                x.mean()
                *
                100.0
        ),

        difficult_percent=(
            "observed_correctability_class",
            lambda x:
                (
                    x
                    ==
                    "DIFFICULT"
                )
                .mean()
                *
                100.0
        ),

        not_improved_percent=(
            "observed_correctability_class",
            lambda x:
                (
                    x
                    ==
                    "NOT_IMPROVED"
                )
                .mean()
                *
                100.0
        ),
    )
    .reset_index()
)


# ============================================================
# SEVERITY SUMMARY
# ============================================================

severity_summary = (
    component_df
    .groupby(
        "severity"
    )
    .agg(

        decisions=(
            "assembly_id",
            "count",
        ),

        improvement_rate_percent=(
            "controller_improved_case",
            lambda x:
                x.mean()
                *
                100.0
        ),

        mean_improvement_percent=(
            "quality_improvement_percent",
            "mean",
        ),

        mean_residual_quality=(
            "structured_actual_quality",
            "mean",
        ),

        mean_utilization=(
            "structured_utilization",
            "mean",
        ),

        high_regret_percent=(
            "high_regret_case",
            lambda x:
                x.mean()
                *
                100.0
        ),

        difficult_percent=(
            "observed_correctability_class",
            lambda x:
                (
                    x
                    ==
                    "DIFFICULT"
                )
                .mean()
                *
                100.0
        ),

        not_improved_percent=(
            "observed_correctability_class",
            lambda x:
                (
                    x
                    ==
                    "NOT_IMPROVED"
                )
                .mean()
                *
                100.0
        ),
    )
    .reset_index()
)


# ============================================================
# SCENARIO SUMMARY
# ============================================================

scenario_summary = (
    component_df
    .groupby(
        "scenario_type"
    )
    .agg(

        decisions=(
            "assembly_id",
            "count",
        ),

        improvement_rate_percent=(
            "controller_improved_case",
            lambda x:
                x.mean()
                *
                100.0
        ),

        mean_improvement_percent=(
            "quality_improvement_percent",
            "mean",
        ),

        mean_residual_quality=(
            "structured_actual_quality",
            "mean",
        ),

        mean_utilization=(
            "structured_utilization",
            "mean",
        ),

        high_regret_percent=(
            "high_regret_case",
            lambda x:
                x.mean()
                *
                100.0
        ),

        difficult_percent=(
            "observed_correctability_class",
            lambda x:
                (
                    x
                    ==
                    "DIFFICULT"
                )
                .mean()
                *
                100.0
        ),

        not_improved_percent=(
            "observed_correctability_class",
            lambda x:
                (
                    x
                    ==
                    "NOT_IMPROVED"
                )
                .mean()
                *
                100.0
        ),
    )
    .reset_index()
)


# ============================================================
# STAGE SUMMARY
# ============================================================

stage_summary = (
    component_df
    .groupby(
        "component_index"
    )
    .agg(

        decisions=(
            "assembly_id",
            "count",
        ),

        improvement_rate_percent=(
            "controller_improved_case",
            lambda x:
                x.mean()
                *
                100.0
        ),

        mean_improvement_percent=(
            "quality_improvement_percent",
            "mean",
        ),

        mean_residual_quality=(
            "structured_actual_quality",
            "mean",
        ),

        mean_utilization=(
            "structured_utilization",
            "mean",
        ),

        mean_regret=(
            "structured_absolute_regret",
            "mean",
        ),

        difficult_percent=(
            "observed_correctability_class",
            lambda x:
                (
                    x
                    ==
                    "DIFFICULT"
                )
                .mean()
                *
                100.0
        ),

        not_improved_percent=(
            "observed_correctability_class",
            lambda x:
                (
                    x
                    ==
                    "NOT_IMPROVED"
                )
                .mean()
                *
                100.0
        ),
    )
    .reset_index()
)


# ============================================================
# ASSEMBLY-LEVEL ANALYSIS
# ============================================================

assembly_analysis = (
    assembly_df.copy()
)


assembly_analysis[
    "structured_improvement"
] = (
    assembly_analysis[
        "zero_final_quality"
    ]
    -
    assembly_analysis[
        "structured20_final_quality"
    ]
)


assembly_analysis[
    "structured_improvement_percent"
] = (
    assembly_analysis[
        "structured_improvement"
    ]
    /
    np.maximum(
        np.abs(
            assembly_analysis[
                "zero_final_quality"
            ]
        ),
        EPSILON,
    )
    *
    100.0
)


assembly_analysis[
    "structured_residual_ratio"
] = (
    assembly_analysis[
        "structured20_final_quality"
    ]
    /
    np.maximum(
        np.abs(
            assembly_analysis[
                "zero_final_quality"
            ]
        ),
        EPSILON,
    )
)


assembly_analysis[
    "assembly_improved"
] = (
    assembly_analysis[
        "structured20_final_quality"
    ]
    <
    assembly_analysis[
        "zero_final_quality"
    ]
)


assembly_residual_q75 = float(
    assembly_analysis[
        "structured20_final_quality"
    ]
    .quantile(
        0.75
    )
)


assembly_analysis[
    "high_residual_assembly"
] = (
    assembly_analysis[
        "structured20_final_quality"
    ]
    >=
    assembly_residual_q75
)


assembly_analysis[
    "difficult_assembly"
] = (
    (
        ~assembly_analysis[
            "assembly_improved"
        ]
    )
    |
    (
        assembly_analysis[
            "high_residual_assembly"
        ]
    )
)


# ============================================================
# THRESHOLD TABLE
# ============================================================

threshold_df = pd.DataFrame(
    {

        "descriptor": [

            "improvement_percent_q25",
            "improvement_percent_q50",
            "improvement_percent_q75",

            "corrected_quality_q25",
            "corrected_quality_q50",
            "corrected_quality_q75",

            "utilization_q50",
            "utilization_q75",

            "regret_q75",
            "regret_q90",

            "assembly_corrected_quality_q75",
        ],

        "value": [

            improvement_q25,
            improvement_q50,
            improvement_q75,

            residual_q25,
            residual_q50,
            residual_q75,

            utilization_q50,
            utilization_q75,

            regret_q75,
            regret_q90,

            assembly_residual_q75,
        ],
    }
)


# ============================================================
# SAVE TABLES
# ============================================================

component_df.to_csv(
    os.path.join(
        TABLE_DIR,
        "component_correctability_analysis.csv",
    ),
    index=False,
)


assembly_analysis.to_csv(
    os.path.join(
        TABLE_DIR,
        "assembly_correctability_analysis.csv",
    ),
    index=False,
)


class_summary.to_csv(
    os.path.join(
        TABLE_DIR,
        "correctability_class_summary.csv",
    ),
    index=False,
)


batch_summary.to_csv(
    os.path.join(
        TABLE_DIR,
        "correctability_by_batch.csv",
    ),
    index=False,
)


severity_summary.to_csv(
    os.path.join(
        TABLE_DIR,
        "correctability_by_severity.csv",
    ),
    index=False,
)


scenario_summary.to_csv(
    os.path.join(
        TABLE_DIR,
        "correctability_by_scenario.csv",
    ),
    index=False,
)


stage_summary.to_csv(
    os.path.join(
        TABLE_DIR,
        "correctability_by_stage.csv",
    ),
    index=False,
)


threshold_df.to_csv(
    os.path.join(
        TABLE_DIR,
        "correctability_reference_thresholds.csv",
    ),
    index=False,
)


# ============================================================
# FIGURE 1
# IMPROVEMENT VS UTILIZATION
# ============================================================

plt.figure(
    figsize=(
        7.5,
        5.5,
    )
)


plt.scatter(
    component_df[
        "structured_utilization"
    ],
    component_df[
        "quality_improvement_percent"
    ],
    alpha=0.35,
)


plt.axhline(
    0.0,
    linestyle="--",
)


plt.xlabel(
    "Correction capability utilization"
)


plt.ylabel(
    "Quality improvement vs zero (%)"
)


plt.title(
    "Observed Improvement vs Correction Utilization"
)


plt.grid(
    alpha=0.25,
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "improvement_vs_utilization.png",
    ),
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# ============================================================
# FIGURE 2
# ZERO VS CORRECTED QUALITY
# ============================================================

plt.figure(
    figsize=(
        6.5,
        6.0,
    )
)


plt.scatter(
    component_df[
        "zero_actual_quality"
    ],
    component_df[
        "structured_actual_quality"
    ],
    alpha=0.35,
)


maximum_axis = float(
    max(
        component_df[
            "zero_actual_quality"
        ]
        .max(),

        component_df[
            "structured_actual_quality"
        ]
        .max(),
    )
)


plt.plot(
    [
        0.0,
        maximum_axis,
    ],
    [
        0.0,
        maximum_axis,
    ],
    linestyle="--",
)


plt.xlabel(
    "Quality without correction"
)


plt.ylabel(
    "Quality after Structured-20 correction"
)


plt.title(
    "Observed Component-Level Correctability"
)


plt.grid(
    alpha=0.25,
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "zero_vs_corrected_quality.png",
    ),
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# ============================================================
# FIGURE 3
# CLASS DISTRIBUTION
# ============================================================

class_plot = (
    class_summary
    .sort_values(
        "percentage",
        ascending=False,
    )
)


plt.figure(
    figsize=(
        8.0,
        5.0,
    )
)


plt.bar(
    class_plot[
        "observed_correctability_class"
    ],
    class_plot[
        "percentage"
    ],
)


plt.ylabel(
    "Share of component decisions (%)"
)


plt.xlabel(
    "Observed correctability class"
)


plt.title(
    "Exploratory Correctability-Class Distribution"
)


plt.xticks(
    rotation=15,
)


plt.grid(
    axis="y",
    alpha=0.25,
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "correctability_class_distribution.png",
    ),
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# ============================================================
# FIGURE 4
# STAGE BEHAVIOUR
# ============================================================

plt.figure(
    figsize=(
        7.5,
        5.0,
    )
)


plt.plot(
    stage_summary[
        "component_index"
    ],
    stage_summary[
        "mean_residual_quality"
    ],
    marker="o",
)


plt.xlabel(
    "Sequential component stage"
)


plt.ylabel(
    "Mean residual quality after correction"
)


plt.title(
    "Residual Corrected Quality Across Assembly Stages"
)


plt.xticks(
    stage_summary[
        "component_index"
    ]
)


plt.grid(
    alpha=0.25,
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "correctability_by_stage.png",
    ),
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# ============================================================
# MAIN SUMMARY
# ============================================================

improved_rate = float(
    component_df[
        "controller_improved_case"
    ]
    .mean()
    *
    100.0
)


worsened_rate = float(
    component_df[
        "controller_worsened_case"
    ]
    .mean()
    *
    100.0
)


mean_improvement = float(
    component_df[
        "quality_improvement_percent"
    ]
    .mean()
)


median_improvement = float(
    component_df[
        "quality_improvement_percent"
    ]
    .median()
)


mean_utilization = float(
    component_df[
        "structured_utilization"
    ]
    .mean()
)


high_utilization_rate = float(
    component_df[
        "higher_capability_case"
    ]
    .mean()
    *
    100.0
)


high_regret_rate = float(
    component_df[
        "high_regret_case"
    ]
    .mean()
    *
    100.0
)


assembly_improved_rate = float(
    assembly_analysis[
        "assembly_improved"
    ]
    .mean()
    *
    100.0
)


difficult_assembly_rate = float(
    assembly_analysis[
        "difficult_assembly"
    ]
    .mean()
    *
    100.0
)


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "OBSERVED COMPONENT-LEVEL BEHAVIOUR"
)

print(
    "------------------------------------------------------------"
)


print(
    f"\nComponent decisions improved : "
    f"{improved_rate:.2f}%"
)


print(
    f"Component decisions worsened : "
    f"{worsened_rate:.2f}%"
)


print(
    f"Mean improvement             : "
    f"{mean_improvement:.2f}%"
)


print(
    f"Median improvement           : "
    f"{median_improvement:.2f}%"
)


print(
    f"Mean correction utilization  : "
    f"{mean_utilization:.3f}"
)


print(
    f"High-utilization cases       : "
    f"{high_utilization_rate:.2f}%"
)


print(
    f"High-regret decisions        : "
    f"{high_regret_rate:.2f}%"
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "ASSEMBLY-LEVEL BEHAVIOUR"
)

print(
    "------------------------------------------------------------"
)


print(
    f"\nAssemblies improved          : "
    f"{assembly_improved_rate:.2f}%"
)


print(
    f"Difficult/high residual      : "
    f"{difficult_assembly_rate:.2f}%"
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "EXPLORATORY CORRECTABILITY CLASSES"
)

print(
    "------------------------------------------------------------\n"
)


print(
    class_summary
    .round(
        4
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
    "BATCH CONDITION SUMMARY"
)

print(
    "------------------------------------------------------------\n"
)


print(
    batch_summary
    .round(
        4
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
    "SEVERITY SUMMARY"
)

print(
    "------------------------------------------------------------\n"
)


print(
    severity_summary
    .round(
        4
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
    "STAGE SUMMARY"
)

print(
    "------------------------------------------------------------\n"
)


print(
    stage_summary
    .round(
        4
    )
    .to_string(
        index=False
    )
)


print(
    "\n"
    "============================================================"
)

print(
    "CORRECTABILITY ANALYSIS INTERPRETATION"
)

print(
    "============================================================"
)


if (
    improved_rate
    >=
    85.0
    and
    assembly_improved_rate
    >=
    90.0
):

    print(
        "\nRESULT A:"
    )

    print(
        "The existing V3.2 results contain a strong observed "
        "correctability signal."
    )


    print(
        "\nNEXT QUESTION:"
    )

    print(
        "Can difficult/correctable cases be identified before "
        "the correction decision using pre-decision features?"
    )


else:

    print(
        "\nRESULT B:"
    )

    print(
        "Observed improvement exists, but a strong standalone "
        "correctability concept is not yet supported."
    )


    print(
        "\nDo not train a dedicated correctability model yet."
    )


print(
    "\nIMPORTANT:"
)


print(
    "The classes are exploratory descriptions of observed "
    "validation outcomes."
)


print(
    "They are not industrial pass/fail limits and are not yet "
    "a validated correctability classifier."
)


print(
    "\nSaved tables:"
)


print(
    TABLE_DIR
)


print(
    "\nSaved figures:"
)


print(
    FIGURE_DIR
)


print(
    "\n"
    "============================================================"
)

print(
    "CORRECTABILITY ANALYSIS COMPLETED"
)

print(
    "============================================================"
)