import os
import numpy as np
import pandas as pd


# ============================================================
# VERSION 3.2
# OPTIMIZER-RELEVANT SURROGATE RANKING DIAGNOSIS
# ============================================================
#
# PURPOSE
#
# Conventional regression metrics answer:
#
#     "How close is predicted quality to actual quality?"
#
# But an optimizer needs something stronger:
#
#     "Does the surrogate correctly identify which correction
#      is better for the SAME assembly state?"
#
#
# For every unseen assembly state we already evaluated a
# structured set of candidate corrections.
#
# This script asks:
#
#   1. Does the predicted-best candidate equal the true best?
#
#   2. Is the predicted-best candidate at least among the
#      true top-3 candidates?
#
#   3. How much actual quality is lost by selecting the
#      surrogate's best candidate instead of the true best?
#
#   4. Does the surrogate-selected correction improve over
#      the zero-correction candidate?
#
#   5. How well does predicted candidate ordering agree with
#      true candidate ordering?
#
#
# This is much more directly relevant to Bayesian Optimization
# than global R² alone.
# ============================================================


# ============================================================
# PATHS
# ============================================================

PREDICTION_FILE = (
    "data/processed/"
    "v3_2_scalar_vs_profile_test_predictions.csv"
)

DETAIL_OUTPUT = (
    "results/tables/"
    "v3_2_surrogate_ranking_diagnosis_details.csv"
)

SUMMARY_OUTPUT = (
    "results/tables/"
    "v3_2_surrogate_ranking_diagnosis_summary.csv"
)

STAGE_OUTPUT = (
    "results/tables/"
    "v3_2_surrogate_ranking_diagnosis_by_stage.csv"
)


# ============================================================
# FILE CHECK
# ============================================================

if not os.path.exists(
    PREDICTION_FILE
):

    raise FileNotFoundError(
        "\nPrediction file not found. "
        "Complete scalar-vs-profile comparison first."
    )


# ============================================================
# LOAD
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "V3.2 SURROGATE RANKING DIAGNOSIS"
)

print(
    "============================================================"
)


df = pd.read_csv(
    PREDICTION_FILE
)


TARGET = (
    "target_quality_score"
)


required_columns = [
    "assembly_id",
    "component_index",
    "trajectory_type",
    "candidate_source",
    TARGET,
    "scalar_prediction",
    "profile_prediction",
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    raise ValueError(
        "\nMissing required columns:\n"
        +
        "\n".join(
            missing_columns
        )
    )


print(
    f"\nRows             : {len(df)}"
)


# ============================================================
# DEFINE ONE DECISION STATE
# ============================================================
#
# All candidate corrections belonging to:
#
#   one assembly
#   one component index
#   one trajectory state
#
# represent one local optimization problem.
# ============================================================

GROUP_COLUMNS = [
    "assembly_id",
    "component_index",
    "trajectory_type",
]


n_states = (
    df[
        GROUP_COLUMNS
    ]
    .drop_duplicates()
    .shape[0]
)


print(
    f"Decision states  : {n_states}"
)


# ============================================================
# SAFE SPEARMAN WITHOUT SCIPY
# ============================================================

def rank_correlation(
    true_values,
    predicted_values,
):

    true_series = pd.Series(
        true_values
    )

    predicted_series = pd.Series(
        predicted_values
    )


    true_ranks = true_series.rank(
        method="average"
    )


    predicted_ranks = predicted_series.rank(
        method="average"
    )


    correlation = true_ranks.corr(
        predicted_ranks,
        method="pearson",
    )


    if pd.isna(
        correlation
    ):

        return 0.0


    return float(
        correlation
    )


# ============================================================
# ANALYZE ONE MODEL
# ============================================================

def analyze_model(
    model_name,
    prediction_column,
):

    records = []


    grouped = df.groupby(
        GROUP_COLUMNS,
        sort=False,
    )


    for (
        assembly_id,
        component_index,
        trajectory_type,
    ), group in grouped:


        group = group.reset_index(
            drop=True
        )


        true_values = (
            group[
                TARGET
            ]
            .to_numpy()
        )


        predicted_values = (
            group[
                prediction_column
            ]
            .to_numpy()
        )


        # ----------------------------------------------------
        # TRUE BEST
        # ----------------------------------------------------

        true_best_position = int(
            np.argmin(
                true_values
            )
        )


        true_best_quality = float(
            true_values[
                true_best_position
            ]
        )


        true_best_source = (
            group.iloc[
                true_best_position
            ][
                "candidate_source"
            ]
        )


        # ----------------------------------------------------
        # SURROGATE-PREDICTED BEST
        # ----------------------------------------------------

        predicted_best_position = int(
            np.argmin(
                predicted_values
            )
        )


        predicted_best_predicted_quality = float(
            predicted_values[
                predicted_best_position
            ]
        )


        predicted_best_actual_quality = float(
            true_values[
                predicted_best_position
            ]
        )


        predicted_best_source = (
            group.iloc[
                predicted_best_position
            ][
                "candidate_source"
            ]
        )


        # ----------------------------------------------------
        # TRUE RANK OF SURROGATE-SELECTION
        # ----------------------------------------------------

        true_order = np.argsort(
            true_values
        )


        true_rank_position = int(
            np.where(
                true_order
                ==
                predicted_best_position
            )[0][0]
        ) + 1


        exact_best_hit = (
            predicted_best_position
            ==
            true_best_position
        )


        top3_hit = (
            true_rank_position
            <=
            3
        )


        top5_hit = (
            true_rank_position
            <=
            5
        )


        # ----------------------------------------------------
        # REGRET
        #
        # How much actual quality is lost because the
        # surrogate selected a different candidate?
        # ----------------------------------------------------

        absolute_regret = (
            predicted_best_actual_quality
            -
            true_best_quality
        )


        relative_regret_percent = (
            absolute_regret
            /
            max(
                abs(
                    true_best_quality
                ),
                1e-9,
            )
            *
            100.0
        )


        # ----------------------------------------------------
        # ZERO-CORRECTION REFERENCE
        # ----------------------------------------------------

        zero_rows = group[
            group[
                "candidate_source"
            ]
            ==
            "zero"
        ]


        if len(
            zero_rows
        ) == 1:

            zero_actual_quality = float(
                zero_rows.iloc[
                    0
                ][
                    TARGET
                ]
            )


            selected_beats_zero = (
                predicted_best_actual_quality
                <
                zero_actual_quality
            )


            true_best_beats_zero = (
                true_best_quality
                <
                zero_actual_quality
            )


            selected_improvement_vs_zero = (
                (
                    zero_actual_quality
                    -
                    predicted_best_actual_quality
                )
                /
                max(
                    abs(
                        zero_actual_quality
                    ),
                    1e-9,
                )
                *
                100.0
            )


        else:

            zero_actual_quality = np.nan

            selected_beats_zero = False

            true_best_beats_zero = False

            selected_improvement_vs_zero = np.nan


        # ----------------------------------------------------
        # RANK CORRELATION
        # ----------------------------------------------------

        spearman_like = rank_correlation(
            true_values,
            predicted_values,
        )


        # ----------------------------------------------------
        # QUALITY SPREAD WITHIN CANDIDATE SET
        #
        # This tells us how difficult the local ranking
        # problem is relative to surrogate error.
        # ----------------------------------------------------

        candidate_quality_range = float(
            np.max(
                true_values
            )
            -
            np.min(
                true_values
            )
        )


        best_to_second_gap = np.nan


        if len(
            true_values
        ) >= 2:

            sorted_true = np.sort(
                true_values
            )

            best_to_second_gap = float(
                sorted_true[
                    1
                ]
                -
                sorted_true[
                    0
                ]
            )


        records.append(
            {

                "model":
                    model_name,

                "assembly_id":
                    assembly_id,

                "component_index":
                    component_index,

                "trajectory_type":
                    trajectory_type,

                "n_candidates":
                    len(
                        group
                    ),

                "true_best_quality":
                    true_best_quality,

                "true_best_source":
                    true_best_source,

                "predicted_best_predicted_quality":
                    predicted_best_predicted_quality,

                "predicted_best_actual_quality":
                    predicted_best_actual_quality,

                "predicted_best_source":
                    predicted_best_source,

                "true_rank_of_predicted_best":
                    true_rank_position,

                "exact_best_hit":
                    exact_best_hit,

                "top3_hit":
                    top3_hit,

                "top5_hit":
                    top5_hit,

                "absolute_regret":
                    absolute_regret,

                "relative_regret_percent":
                    relative_regret_percent,

                "zero_actual_quality":
                    zero_actual_quality,

                "selected_beats_zero":
                    selected_beats_zero,

                "true_best_beats_zero":
                    true_best_beats_zero,

                "selected_improvement_vs_zero_percent":
                    selected_improvement_vs_zero,

                "rank_correlation":
                    spearman_like,

                "candidate_quality_range":
                    candidate_quality_range,

                "best_to_second_quality_gap":
                    best_to_second_gap,
            }
        )


    return pd.DataFrame(
        records
    )


# ============================================================
# RUN BOTH MODELS
# ============================================================

scalar_result_df = analyze_model(
    model_name="scalar_rf",
    prediction_column="scalar_prediction",
)


profile_result_df = analyze_model(
    model_name="profile_aware_rf",
    prediction_column="profile_prediction",
)


detail_df = pd.concat(
    [
        scalar_result_df,
        profile_result_df,
    ],
    ignore_index=True,
)


# ============================================================
# SUMMARY FUNCTION
# ============================================================

def summarize(
    subset,
):

    return {

        "decision_states":
            len(
                subset
            ),

        "exact_best_hit_rate_percent":
            subset[
                "exact_best_hit"
            ]
            .mean()
            *
            100.0,

        "top3_hit_rate_percent":
            subset[
                "top3_hit"
            ]
            .mean()
            *
            100.0,

        "top5_hit_rate_percent":
            subset[
                "top5_hit"
            ]
            .mean()
            *
            100.0,

        "median_true_rank_of_predicted_best":
            subset[
                "true_rank_of_predicted_best"
            ]
            .median(),

        "mean_absolute_regret":
            subset[
                "absolute_regret"
            ]
            .mean(),

        "median_absolute_regret":
            subset[
                "absolute_regret"
            ]
            .median(),

        "p95_absolute_regret":
            subset[
                "absolute_regret"
            ]
            .quantile(
                0.95
            ),

        "mean_rank_correlation":
            subset[
                "rank_correlation"
            ]
            .mean(),

        "selected_beats_zero_rate_percent":
            subset[
                "selected_beats_zero"
            ]
            .mean()
            *
            100.0,

        "true_best_beats_zero_rate_percent":
            subset[
                "true_best_beats_zero"
            ]
            .mean()
            *
            100.0,

        "mean_selected_improvement_vs_zero_percent":
            subset[
                "selected_improvement_vs_zero_percent"
            ]
            .mean(),

        "median_best_to_second_quality_gap":
            subset[
                "best_to_second_quality_gap"
            ]
            .median(),
    }


# ============================================================
# OVERALL SUMMARY
# ============================================================

summary_rows = []


for model_name in [
    "scalar_rf",
    "profile_aware_rf",
]:

    subset = detail_df[
        detail_df[
            "model"
        ]
        ==
        model_name
    ]


    summary_rows.append(
        {

            "model":
                model_name,

            **summarize(
                subset
            ),
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# STAGE-WISE SUMMARY
# ============================================================

stage_rows = []


for model_name in [
    "scalar_rf",
    "profile_aware_rf",
]:

    model_subset = detail_df[
        detail_df[
            "model"
        ]
        ==
        model_name
    ]


    for component_index in range(
        1,
        6,
    ):

        subset = model_subset[
            model_subset[
                "component_index"
            ]
            ==
            component_index
        ]


        values = summarize(
            subset
        )


        stage_rows.append(
            {

                "model":
                    model_name,

                "component_index":
                    component_index,

                **values,
            }
        )


stage_df = pd.DataFrame(
    stage_rows
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "results/tables",
    exist_ok=True,
)


detail_df.to_csv(
    DETAIL_OUTPUT,
    index=False,
)


summary_df.to_csv(
    SUMMARY_OUTPUT,
    index=False,
)


stage_df.to_csv(
    STAGE_OUTPUT,
    index=False,
)


# ============================================================
# PRINT
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "OPTIMIZER-RELEVANT RANKING RESULTS"
)

print(
    "============================================================"
)


display_columns = [

    "model",

    "exact_best_hit_rate_percent",

    "top3_hit_rate_percent",

    "top5_hit_rate_percent",

    "median_true_rank_of_predicted_best",

    "mean_absolute_regret",

    "p95_absolute_regret",

    "mean_rank_correlation",

    "selected_beats_zero_rate_percent",

    "true_best_beats_zero_rate_percent",
]


print(
    "\n"
    +
    summary_df[
        display_columns
    ]
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
    "STAGE-WISE RANKING"
)

print(
    "------------------------------------------------------------\n"
)


stage_display = stage_df[
    [
        "model",
        "component_index",
        "exact_best_hit_rate_percent",
        "top3_hit_rate_percent",
        "mean_absolute_regret",
        "mean_rank_correlation",
        "selected_beats_zero_rate_percent",
    ]
]


print(
    stage_display
    .round(
        4
    )
    .to_string(
        index=False
    )
)


# ============================================================
# DIAGNOSTIC INTERPRETATION
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "LEARNING-TARGET / RANKING DIAGNOSIS"
)

print(
    "============================================================"
)


scalar_summary = summary_df[
    summary_df[
        "model"
    ]
    ==
    "scalar_rf"
].iloc[
    0
]


profile_summary = summary_df[
    summary_df[
        "model"
    ]
    ==
    "profile_aware_rf"
].iloc[
    0
]


print(
    f"\nScalar exact-best hit rate  : "
    f"{scalar_summary['exact_best_hit_rate_percent']:.2f}%"
)


print(
    f"Profile exact-best hit rate : "
    f"{profile_summary['exact_best_hit_rate_percent']:.2f}%"
)


print(
    f"\nScalar top-3 hit rate       : "
    f"{scalar_summary['top3_hit_rate_percent']:.2f}%"
)


print(
    f"Profile top-3 hit rate      : "
    f"{profile_summary['top3_hit_rate_percent']:.2f}%"
)


print(
    f"\nScalar mean regret          : "
    f"{scalar_summary['mean_absolute_regret']:.4f}"
)


print(
    f"Profile mean regret         : "
    f"{profile_summary['mean_absolute_regret']:.4f}"
)


print(
    f"\nScalar rank correlation     : "
    f"{scalar_summary['mean_rank_correlation']:.4f}"
)


print(
    f"Profile rank correlation    : "
    f"{profile_summary['mean_rank_correlation']:.4f}"
)


print(
    f"\nScalar selected beats zero  : "
    f"{scalar_summary['selected_beats_zero_rate_percent']:.2f}%"
)


print(
    f"Profile selected beats zero : "
    f"{profile_summary['selected_beats_zero_rate_percent']:.2f}%"
)


print(
    f"\nTrue candidate best beats zero: "
    f"{profile_summary['true_best_beats_zero_rate_percent']:.2f}%"
)


# ============================================================
# DECISION
# ============================================================

profile_top3 = float(
    profile_summary[
        "top3_hit_rate_percent"
    ]
)


profile_exact = float(
    profile_summary[
        "exact_best_hit_rate_percent"
    ]
)


profile_selected_beats_zero = float(
    profile_summary[
        "selected_beats_zero_rate_percent"
    ]
)


true_best_beats_zero = float(
    profile_summary[
        "true_best_beats_zero_rate_percent"
    ]
)


print(
    "\n"
    "------------------------------------------------------------"
)


if (
    true_best_beats_zero
    >
    profile_selected_beats_zero
    +
    10.0
):

    print(
        "STRONG SIGNAL:"
    )

    print(
        "Good corrective candidates exist substantially more "
        "often than the surrogate successfully selects them."
    )

    print(
        "\nThis means the main remaining problem is not "
        "correction capability."
    )

    print(
        "It is optimizer-relevant discrimination/ranking "
        "of candidate corrections."
    )


if (
    profile_exact
    <
    50.0
):

    print(
        "\nSIGNAL:"
    )

    print(
        "The surrogate identifies the exact best correction "
        "in fewer than half of decision states."
    )


if (
    profile_top3
    <
    75.0
):

    print(
        "\nSIGNAL:"
    )

    print(
        "Even top-3 candidate identification remains "
        "insufficiently reliable for a strong optimizer."
    )


print(
    "\nNEXT STEP:"
)


print(
    "Use these results to decide whether the surrogate should "
    "learn absolute quality or an optimizer-focused target such "
    "as correction benefit / quality improvement relative to "
    "the zero-correction state."
)


print(
    "\nDo NOT start another expensive BO run yet."
)


print(
    "\nSaved:"
)


print(
    DETAIL_OUTPUT
)

print(
    SUMMARY_OUTPUT
)

print(
    STAGE_OUTPUT
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.2 SURROGATE RANKING DIAGNOSIS COMPLETED"
)

print(
    "============================================================"
)