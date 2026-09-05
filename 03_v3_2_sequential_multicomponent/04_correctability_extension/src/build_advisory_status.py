import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# ADVISORY STATUS ANALYSIS
# ============================================================
#
# Purpose:
#
# Combine the pre-decision difficulty probability with the
# correction-capability information from the Structured-20
# recommendation.
#
# The objective is NOT to create an automatic routing
# controller.
#
# Instead, the analysis demonstrates an advisory
# decision-support layer with three statuses:
#
# ROUTINE
#     No strong pre-decision warning and correction remains
#     comfortably within available capability.
#
# REVIEW_ADVISED
#     Pre-decision difficulty probability exceeds the
#     calibrated advisory threshold.
#
# CAPABILITY_WARNING
#     Recommended correction approaches the defined
#     correction-capability boundary.
#
#
# IMPORTANT:
#
# The difficulty probability was generated fully out-of-fold
# by assembly.
#
# Post-correction outcomes are used only to evaluate whether
# the advisory status is meaningful.
# ============================================================


# ------------------------------------------------------------
# Input files
# ------------------------------------------------------------

RISK_FILE = (
    "04_correctability_extension/"
    "results/risk_gate/tables/"
    "oof_risk_predictions.csv"
)

RISK_SUMMARY_FILE = (
    "04_correctability_extension/"
    "results/risk_gate/tables/"
    "risk_gate_summary.csv"
)

CORRECTABILITY_FILE = (
    "04_correctability_extension/"
    "results/tables/"
    "component_correctability_analysis.csv"
)


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

OUTPUT_DIR = (
    "04_correctability_extension/"
    "results/advisory"
)

TABLE_DIR = os.path.join(
    OUTPUT_DIR,
    "tables",
)

FIGURE_DIR = os.path.join(
    OUTPUT_DIR,
    "figures",
)


os.makedirs(
    TABLE_DIR,
    exist_ok=True,
)

os.makedirs(
    FIGURE_DIR,
    exist_ok=True,
)


DECISION_FILE = os.path.join(
    TABLE_DIR,
    "advisory_decisions.csv",
)

SUMMARY_FILE = os.path.join(
    TABLE_DIR,
    "advisory_summary.csv",
)

STAGE_FILE = os.path.join(
    TABLE_DIR,
    "advisory_by_stage.csv",
)


# ------------------------------------------------------------
# Capability threshold
# ------------------------------------------------------------

CAPABILITY_WARNING_THRESHOLD = 0.90


# ============================================================
# FILE CHECK
# ============================================================

required_files = [
    RISK_FILE,
    RISK_SUMMARY_FILE,
    CORRECTABILITY_FILE,
]


for path in required_files:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nMissing required file:\n{path}"
        )


# ============================================================
# LOAD
# ============================================================

risk_df = pd.read_csv(
    RISK_FILE
)

risk_summary_df = pd.read_csv(
    RISK_SUMMARY_FILE
)

correctability_df = pd.read_csv(
    CORRECTABILITY_FILE
)


print(
    "\n"
    "============================================================"
)

print(
    "ADVISORY STATUS ANALYSIS"
)

print(
    "============================================================"
)


print(
    f"\nRisk decisions          : "
    f"{len(risk_df)}"
)


print(
    f"Correctability decisions: "
    f"{len(correctability_df)}"
)


# ============================================================
# READ FROZEN RISK THRESHOLD
# ============================================================

risk_summary_lookup = dict(
    zip(
        risk_summary_df[
            "metric"
        ],
        risk_summary_df[
            "value"
        ],
    )
)


if (
    "selected_probability_threshold"
    not in risk_summary_lookup
):

    raise ValueError(
        "\nRisk summary does not contain the selected "
        "probability threshold."
    )


RISK_THRESHOLD = float(
    risk_summary_lookup[
        "selected_probability_threshold"
    ]
)


print(
    f"\nRisk threshold          : "
    f"{RISK_THRESHOLD:.4f}"
)


print(
    f"Capability warning limit: "
    f"{CAPABILITY_WARNING_THRESHOLD:.2f}"
)


# ============================================================
# MERGE CAPABILITY INFORMATION
# ============================================================

KEY_COLUMNS = [
    "assembly_id",
    "component_index",
]


if risk_df.duplicated(
    subset=KEY_COLUMNS
).any():

    raise ValueError(
        "\nRisk data contains duplicate decision keys."
    )


if correctability_df.duplicated(
    subset=KEY_COLUMNS
).any():

    raise ValueError(
        "\nCorrectability data contains duplicate decision keys."
    )


CAPABILITY_COLUMNS = [
    "assembly_id",
    "component_index",
    "structured_utilization",
]


for column in CAPABILITY_COLUMNS:

    if column not in correctability_df.columns:

        raise ValueError(
            f"\nMissing capability column: {column}"
        )


decision_df = risk_df.merge(

    correctability_df[
        CAPABILITY_COLUMNS
    ],

    on=KEY_COLUMNS,

    how="inner",

    validate="one_to_one",
)


if len(
    decision_df
) != len(
    risk_df
):

    raise ValueError(
        "\nRisk and capability decision rows do not match."
    )


# ============================================================
# PRE-DECISION RISK FLAG
# ============================================================

decision_df[
    "review_advised"
] = (
    decision_df[
        "oof_attention_probability"
    ]
    >=
    RISK_THRESHOLD
)


# ============================================================
# POST-RECOMMENDATION CAPABILITY FLAG
# ============================================================
#
# Structured utilization becomes available once the candidate
# correction has been recommended.
#
# Therefore this is a post-recommendation but pre-execution
# capability check.
# ============================================================

decision_df[
    "capability_warning"
] = (
    decision_df[
        "structured_utilization"
    ]
    >=
    CAPABILITY_WARNING_THRESHOLD
)


# ============================================================
# FINAL ADVISORY STATUS
# ============================================================

def assign_status(
    row
):

    if row[
        "capability_warning"
    ]:

        return "CAPABILITY_WARNING"


    if row[
        "review_advised"
    ]:

        return "REVIEW_ADVISED"


    return "ROUTINE"


decision_df[
    "advisory_status"
] = decision_df.apply(
    assign_status,
    axis=1,
)


# ============================================================
# POST-HOC OUTCOME FLAGS
# ============================================================
#
# These are already contained in the OOF evaluation file and
# are used only to assess whether the advisory status
# concentrates problematic outcomes.
# ============================================================

required_evaluation_columns = [
    "needs_attention",
    "is_not_improved",
    "is_difficult",
    "high_regret_case",
    "higher_residual_case",
]


missing_evaluation_columns = [
    column
    for column in required_evaluation_columns
    if column not in decision_df.columns
]


if missing_evaluation_columns:

    raise ValueError(
        "\nMissing evaluation columns:\n"
        +
        "\n".join(
            missing_evaluation_columns
        )
    )


# ============================================================
# STATUS SUMMARY
# ============================================================

status_records = []


for status in [
    "ROUTINE",
    "REVIEW_ADVISED",
    "CAPABILITY_WARNING",
]:


    subset = (
        decision_df[
            decision_df[
                "advisory_status"
            ]
            ==
            status
        ]
    )


    n = len(
        subset
    )


    if n == 0:

        status_records.append(
            {

                "advisory_status":
                    status,

                "decisions":
                    0,

                "percentage":
                    0.0,

                "mean_risk_probability":
                    np.nan,

                "mean_utilization":
                    np.nan,

                "needs_attention_percent":
                    np.nan,

                "difficult_percent":
                    np.nan,

                "not_improved_percent":
                    np.nan,

                "high_regret_percent":
                    np.nan,

                "high_residual_percent":
                    np.nan,
            }
        )

        continue


    status_records.append(
        {

            "advisory_status":
                status,

            "decisions":
                n,

            "percentage":
                (
                    n
                    /
                    len(
                        decision_df
                    )
                    *
                    100.0
                ),

            "mean_risk_probability":
                float(
                    subset[
                        "oof_attention_probability"
                    ]
                    .mean()
                ),

            "mean_utilization":
                float(
                    subset[
                        "structured_utilization"
                    ]
                    .mean()
                ),

            "needs_attention_percent":
                float(
                    subset[
                        "needs_attention"
                    ]
                    .mean()
                    *
                    100.0
                ),

            "difficult_percent":
                float(
                    subset[
                        "is_difficult"
                    ]
                    .mean()
                    *
                    100.0
                ),

            "not_improved_percent":
                float(
                    subset[
                        "is_not_improved"
                    ]
                    .mean()
                    *
                    100.0
                ),

            "high_regret_percent":
                float(
                    subset[
                        "high_regret_case"
                    ]
                    .astype(
                        bool
                    )
                    .mean()
                    *
                    100.0
                ),

            "high_residual_percent":
                float(
                    subset[
                        "higher_residual_case"
                    ]
                    .astype(
                        bool
                    )
                    .mean()
                    *
                    100.0
                ),
        }
    )


status_summary_df = pd.DataFrame(
    status_records
)


# ============================================================
# STAGE SUMMARY
# ============================================================

stage_summary_df = (
    decision_df
    .groupby(
        "component_index"
    )
    .agg(

        decisions=(
            "assembly_id",
            "count",
        ),

        review_advised_percent=(
            "review_advised",
            lambda x:
                x.mean()
                *
                100.0
        ),

        capability_warning_percent=(
            "capability_warning",
            lambda x:
                x.mean()
                *
                100.0
        ),

        mean_risk_probability=(
            "oof_attention_probability",
            "mean",
        ),

        needs_attention_percent=(
            "needs_attention",
            lambda x:
                x.mean()
                *
                100.0
        ),

        not_improved_percent=(
            "is_not_improved",
            lambda x:
                x.mean()
                *
                100.0
        ),
    )
    .reset_index()
)


# ============================================================
# OVERALL ADVISORY METRICS
# ============================================================

review_mask = (
    decision_df[
        "review_advised"
    ]
)


routine_mask = (
    ~decision_df[
        "review_advised"
    ]
)


review_count = int(
    review_mask.sum()
)


routine_count = int(
    routine_mask.sum()
)


capability_warning_count = int(
    decision_df[
        "capability_warning"
    ]
    .sum()
)


review_rate = (
    review_count
    /
    len(
        decision_df
    )
    *
    100.0
)


capability_warning_rate = (
    capability_warning_count
    /
    len(
        decision_df
    )
    *
    100.0
)


review_attention_rate = float(
    decision_df.loc[
        review_mask,
        "needs_attention",
    ]
    .mean()
    *
    100.0
)


routine_attention_rate = float(
    decision_df.loc[
        routine_mask,
        "needs_attention",
    ]
    .mean()
    *
    100.0
)


review_not_improved_rate = float(
    decision_df.loc[
        review_mask,
        "is_not_improved",
    ]
    .mean()
    *
    100.0
)


routine_not_improved_rate = float(
    decision_df.loc[
        routine_mask,
        "is_not_improved",
    ]
    .mean()
    *
    100.0
)


review_high_residual_rate = float(
    decision_df.loc[
        review_mask,
        "higher_residual_case",
    ]
    .astype(
        bool
    )
    .mean()
    *
    100.0
)


routine_high_residual_rate = float(
    decision_df.loc[
        routine_mask,
        "higher_residual_case",
    ]
    .astype(
        bool
    )
    .mean()
    *
    100.0
)


attention_enrichment = (
    review_attention_rate
    /
    max(
        routine_attention_rate,
        1e-9,
    )
)


# ============================================================
# SUMMARY TABLE
# ============================================================

overall_summary_df = pd.DataFrame(
    {

        "metric": [

            "risk_threshold",

            "capability_warning_threshold",

            "total_decisions",

            "routine_decisions",

            "review_advised_decisions",

            "review_advised_percent",

            "capability_warning_decisions",

            "capability_warning_percent",

            "review_needs_attention_percent",

            "routine_needs_attention_percent",

            "attention_enrichment_review_vs_routine",

            "review_not_improved_percent",

            "routine_not_improved_percent",

            "review_high_residual_percent",

            "routine_high_residual_percent",
        ],

        "value": [

            RISK_THRESHOLD,

            CAPABILITY_WARNING_THRESHOLD,

            len(
                decision_df
            ),

            routine_count,

            review_count,

            review_rate,

            capability_warning_count,

            capability_warning_rate,

            review_attention_rate,

            routine_attention_rate,

            attention_enrichment,

            review_not_improved_rate,

            routine_not_improved_rate,

            review_high_residual_rate,

            routine_high_residual_rate,
        ],
    }
)


# ============================================================
# SAVE
# ============================================================

decision_df.to_csv(
    DECISION_FILE,
    index=False,
)


status_summary_df.to_csv(
    SUMMARY_FILE,
    index=False,
)


stage_summary_df.to_csv(
    STAGE_FILE,
    index=False,
)


overall_summary_df.to_csv(
    os.path.join(
        TABLE_DIR,
        "advisory_overall_metrics.csv",
    ),
    index=False,
)


# ============================================================
# FIGURE 1
# STATUS DISTRIBUTION
# ============================================================

status_plot_df = (
    status_summary_df[
        status_summary_df[
            "decisions"
        ]
        >
        0
    ]
)


plt.figure(
    figsize=(
        8.0,
        5.0,
    )
)


plt.bar(
    status_plot_df[
        "advisory_status"
    ],
    status_plot_df[
        "percentage"
    ],
)


plt.xlabel(
    "Advisory status"
)


plt.ylabel(
    "Share of decisions (%)"
)


plt.title(
    "Decision-Support Advisory Status Distribution"
)


plt.grid(
    axis="y",
    alpha=0.25,
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "advisory_status_distribution.png",
    ),
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# ============================================================
# FIGURE 2
# RISK ENRICHMENT
# ============================================================

comparison_df = pd.DataFrame(
    {

        "group": [
            "Routine",
            "Review advised",
        ],

        "needs_attention": [
            routine_attention_rate,
            review_attention_rate,
        ],

        "not_improved": [
            routine_not_improved_rate,
            review_not_improved_rate,
        ],

        "high_residual": [
            routine_high_residual_rate,
            review_high_residual_rate,
        ],
    }
)


x = np.arange(
    len(
        comparison_df
    )
)


width = 0.25


plt.figure(
    figsize=(
        8.0,
        5.5,
    )
)


plt.bar(
    x - width,
    comparison_df[
        "needs_attention"
    ],
    width=width,
    label="Needs attention",
)


plt.bar(
    x,
    comparison_df[
        "not_improved"
    ],
    width=width,
    label="Not improved",
)


plt.bar(
    x + width,
    comparison_df[
        "high_residual"
    ],
    width=width,
    label="High residual",
)


plt.xticks(
    x,
    comparison_df[
        "group"
    ],
)


plt.ylabel(
    "Observed outcome rate (%)"
)


plt.title(
    "Observed Risk Enrichment of Advisory Decisions"
)


plt.legend()


plt.grid(
    axis="y",
    alpha=0.25,
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "advisory_risk_enrichment.png",
    ),
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "ADVISORY STATUS SUMMARY"
)

print(
    "============================================================\n"
)


print(
    status_summary_df
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
    "OVERALL ADVISORY BEHAVIOUR"
)

print(
    "------------------------------------------------------------"
)


print(
    f"\nRisk threshold                  : "
    f"{RISK_THRESHOLD:.4f}"
)


print(
    f"Review-advised decisions        : "
    f"{review_count}"
)


print(
    f"Review-advised rate             : "
    f"{review_rate:.2f}%"
)


print(
    f"Capability warnings             : "
    f"{capability_warning_count}"
)


print(
    f"Capability-warning rate         : "
    f"{capability_warning_rate:.2f}%"
)


print(
    f"\nNeeds-attention in REVIEW       : "
    f"{review_attention_rate:.2f}%"
)


print(
    f"Needs-attention in ROUTINE      : "
    f"{routine_attention_rate:.2f}%"
)


print(
    f"Risk enrichment                 : "
    f"{attention_enrichment:.2f}x"
)


print(
    f"\nNot-improved in REVIEW          : "
    f"{review_not_improved_rate:.2f}%"
)


print(
    f"Not-improved in ROUTINE         : "
    f"{routine_not_improved_rate:.2f}%"
)


print(
    f"\nHigh-residual in REVIEW         : "
    f"{review_high_residual_rate:.2f}%"
)


print(
    f"High-residual in ROUTINE        : "
    f"{routine_high_residual_rate:.2f}%"
)


# ============================================================
# FINAL INTERPRETATION
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "ADVISORY-LAYER VERDICT"
)

print(
    "============================================================"
)


if (
    review_attention_rate
    >=
    75.0
    and
    attention_enrichment
    >=
    2.0
):


    print(
        "\nRESULT A:"
    )


    print(
        "The pre-decision risk signal meaningfully separates "
        "routine decisions from decisions that deserve "
        "additional attention."
    )


    print(
        "\nThe evidence supports an advisory difficulty-aware "
        "decision-support layer."
    )


    print(
        "\nStructured-20 remains the primary correction "
        "recommendation controller."
    )


    print(
        "\nThe risk signal should be interpreted as an advisory "
        "warning, not an automatic optimization-routing rule."
    )


else:


    print(
        "\nRESULT B:"
    )


    print(
        "The risk signal provides some separation, but the "
        "evidence is not strong enough to formalize an "
        "advisory layer."
    )


    print(
        "\nKeep the correctability analysis as an exploratory "
        "extension only."
    )


print(
    "\nIMPORTANT:"
)


print(
    "The advisory risk probability is pre-decision and fully "
    "out-of-fold by assembly."
)


print(
    "Correction utilization is evaluated after recommendation "
    "but before execution."
)


print(
    "Observed outcomes are used only for evaluation of the "
    "advisory status."
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
    "ADVISORY STATUS ANALYSIS COMPLETED"
)

print(
    "============================================================"
)