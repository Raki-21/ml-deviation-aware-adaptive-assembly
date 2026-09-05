import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


# ============================================================
# RISK-GATE EVALUATION
# ============================================================
#
# Purpose:
# Generate assembly-wise out-of-fold difficulty probabilities
# and evaluate whether a limited review workload can capture
# a large share of difficult assembly decisions.
#
# The prediction model uses only information available before
# the current correction decision.
#
# Post-correction quantities are used only to evaluate the
# usefulness of the risk gate.
# ============================================================


# ------------------------------------------------------------
# Input files
# ------------------------------------------------------------

PREDICTION_DATA_FILE = (
    "04_correctability_extension/"
    "results/prediction/tables/"
    "correctability_prediction_dataset.csv"
)

CORRECTABILITY_FILE = (
    "04_correctability_extension/"
    "results/tables/"
    "component_correctability_analysis.csv"
)


# ------------------------------------------------------------
# Output folders
# ------------------------------------------------------------

OUTPUT_DIR = (
    "04_correctability_extension/"
    "results/risk_gate"
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


OOF_FILE = os.path.join(
    TABLE_DIR,
    "oof_risk_predictions.csv",
)

FOLD_FILE = os.path.join(
    TABLE_DIR,
    "oof_fold_results.csv",
)

GATE_FILE = os.path.join(
    TABLE_DIR,
    "risk_gate_workload_curve.csv",
)

SUMMARY_FILE = os.path.join(
    TABLE_DIR,
    "risk_gate_summary.csv",
)


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

N_SPLITS = 5

RANDOM_STATE = 20260901

N_TREES = 300

N_JOBS = 2


# ============================================================
# FILE CHECK
# ============================================================

for path in [
    PREDICTION_DATA_FILE,
    CORRECTABILITY_FILE,
]:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nMissing required file:\n{path}"
        )


# ============================================================
# LOAD DATA
# ============================================================

prediction_df = pd.read_csv(
    PREDICTION_DATA_FILE
)

correctability_df = pd.read_csv(
    CORRECTABILITY_FILE
)


print(
    "\n"
    "============================================================"
)

print(
    "RISK-GATE EVALUATION"
)

print(
    "============================================================"
)


print(
    f"\nPrediction rows : "
    f"{len(prediction_df)}"
)

print(
    f"Assemblies      : "
    f"{prediction_df['assembly_id'].nunique()}"
)


# ============================================================
# DECISION KEY CHECK
# ============================================================

KEY_COLUMNS = [
    "assembly_id",
    "component_index",
]


if prediction_df.duplicated(
    subset=KEY_COLUMNS
).any():

    raise ValueError(
        "\nPrediction dataset contains duplicate decision keys."
    )


if correctability_df.duplicated(
    subset=KEY_COLUMNS
).any():

    raise ValueError(
        "\nCorrectability analysis contains duplicate decision keys."
    )


# ============================================================
# MERGE POST-HOC EVALUATION INFORMATION
# ============================================================
#
# observed_correctability_class and needs_attention already
# exist in the prediction dataset.
#
# Therefore they are NOT merged again.
#
# This avoids Pandas creating:
#
# observed_correctability_class_x
# observed_correctability_class_y
#
# which caused the previous KeyError.
# ============================================================

POSTHOC_COLUMNS = [
    "assembly_id",
    "component_index",
    "structured_absolute_regret",
    "structured_actual_quality",
    "higher_residual_case",
    "high_regret_case",
]


missing_posthoc_columns = [
    column
    for column in POSTHOC_COLUMNS
    if column not in correctability_df.columns
]


if missing_posthoc_columns:

    raise ValueError(
        "\nMissing required post-hoc columns:\n"
        +
        "\n".join(
            missing_posthoc_columns
        )
    )


analysis_df = prediction_df.merge(

    correctability_df[
        POSTHOC_COLUMNS
    ],

    on=KEY_COLUMNS,

    how="inner",

    validate="one_to_one",
)


if len(
    analysis_df
) != len(
    prediction_df
):

    raise ValueError(
        "\nPrediction and correctability row counts do not match."
    )


# ============================================================
# REQUIRED COLUMNS AFTER MERGE
# ============================================================

required_columns = [
    "observed_correctability_class",
    "needs_attention",
    "structured_absolute_regret",
    "structured_actual_quality",
    "higher_residual_case",
    "high_regret_case",
]


missing_columns = [
    column
    for column in required_columns
    if column not in analysis_df.columns
]


if missing_columns:

    print(
        "\nAvailable columns:"
    )

    for column in analysis_df.columns:

        print(
            column
        )

    raise ValueError(
        "\nMissing required analysis columns:\n"
        +
        "\n".join(
            missing_columns
        )
    )


# ============================================================
# POST-HOC EVALUATION FLAGS
# ============================================================

analysis_df[
    "is_not_improved"
] = (
    analysis_df[
        "observed_correctability_class"
    ]
    ==
    "NOT_IMPROVED"
)


analysis_df[
    "is_difficult"
] = (
    analysis_df[
        "observed_correctability_class"
    ]
    ==
    "DIFFICULT"
)


analysis_df[
    "is_attention_class"
] = (
    analysis_df[
        "observed_correctability_class"
    ]
    .isin(
        [
            "DIFFICULT",
            "NOT_IMPROVED",
        ]
    )
)


# ============================================================
# PRE-DECISION FEATURES
# ============================================================

COMPONENT_PROCESS_FEATURES = [

    "component_index",

    "offset_mm",
    "tilt_deg",
    "bend_mm",
    "waviness_mm",
    "twist_mm",
    "local_bump_mm",

    "component_profile_mean_mm",
    "component_profile_min_mm",
    "component_profile_max_mm",
    "component_profile_rms_mm",
    "component_parallelism_mm",

    "n_active_modes",

    "batch_offset_bias_mm",
    "batch_angular_bias_deg",
    "fixture_drift_mm",
    "variation_multiplier",
]


STATE_FEATURES = [

    "component_index",

    "state_mean_gap",
    "state_max_gap",
    "state_parallelism",
    "state_rms",
    "state_quality",

    "state_signed_mean",
    "state_end_difference",
    "state_estimated_angle_deg",

    "state_p00",
    "state_p10",
    "state_p20",
    "state_p30",
    "state_p40",
    "state_p50",
    "state_p60",
    "state_p70",
    "state_p80",
    "state_p90",
    "state_p100",
]


COMBINED_FEATURES = list(
    dict.fromkeys(
        COMPONENT_PROCESS_FEATURES
        +
        STATE_FEATURES
    )
)


missing_features = [
    column
    for column in COMBINED_FEATURES
    if column not in analysis_df.columns
]


if missing_features:

    raise ValueError(
        "\nMissing pre-decision features:\n"
        +
        "\n".join(
            missing_features
        )
    )


# ============================================================
# CHECK FOR MISSING FEATURE VALUES
# ============================================================

feature_missing_count = int(
    analysis_df[
        COMBINED_FEATURES
    ]
    .isna()
    .sum()
    .sum()
)


if feature_missing_count != 0:

    raise ValueError(
        f"\nPre-decision features contain "
        f"{feature_missing_count} missing values."
    )


# ============================================================
# MODEL
# ============================================================

def make_model(
    seed
):

    return RandomForestClassifier(

        n_estimators=N_TREES,

        min_samples_leaf=3,

        class_weight="balanced",

        random_state=seed,

        n_jobs=N_JOBS,
    )


# ============================================================
# OOF STORAGE
# ============================================================

analysis_df[
    "oof_attention_probability"
] = np.nan


analysis_df[
    "oof_predicted_attention"
] = np.nan


analysis_df[
    "oof_fold"
] = np.nan


# ============================================================
# GROUPED FIVE-FOLD OUT-OF-FOLD PREDICTIONS
# ============================================================

groups = (
    analysis_df[
        "assembly_id"
    ]
)


group_kfold = GroupKFold(
    n_splits=N_SPLITS
)


fold_records = []


for fold_number, (
    train_index,
    test_index,
) in enumerate(

    group_kfold.split(
        analysis_df,
        groups=groups,
    ),

    start=1,
):


    train_df = (
        analysis_df
        .iloc[
            train_index
        ]
        .copy()
    )


    test_df = (
        analysis_df
        .iloc[
            test_index
        ]
        .copy()
    )


    train_ids = set(
        train_df[
            "assembly_id"
        ]
        .unique()
    )


    test_ids = set(
        test_df[
            "assembly_id"
        ]
        .unique()
    )


    overlap = (
        train_ids
        .intersection(
            test_ids
        )
    )


    if len(
        overlap
    ) != 0:

        raise RuntimeError(
            f"\nAssembly leakage detected in fold "
            f"{fold_number}."
        )


    X_train = (
        train_df[
            COMBINED_FEATURES
        ]
    )


    X_test = (
        test_df[
            COMBINED_FEATURES
        ]
    )


    y_train = (
        train_df[
            "needs_attention"
        ]
        .astype(
            int
        )
        .to_numpy()
    )


    y_test = (
        test_df[
            "needs_attention"
        ]
        .astype(
            int
        )
        .to_numpy()
    )


    model = make_model(
        seed=(
            RANDOM_STATE
            +
            fold_number
        )
    )


    model.fit(
        X_train,
        y_train,
    )


    probabilities = (
        model.predict_proba(
            X_test
        )[
            :,
            1
        ]
    )


    predictions = (
        probabilities
        >=
        0.50
    ).astype(
        int
    )


    test_indices = (
        analysis_df.index[
            test_index
        ]
    )


    analysis_df.loc[
        test_indices,
        "oof_attention_probability",
    ] = probabilities


    analysis_df.loc[
        test_indices,
        "oof_predicted_attention",
    ] = predictions


    analysis_df.loc[
        test_indices,
        "oof_fold",
    ] = fold_number


    fold_balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            predictions,
        )
    )


    fold_recall = (
        recall_score(
            y_test,
            predictions,
            zero_division=0,
        )
    )


    fold_precision = (
        precision_score(
            y_test,
            predictions,
            zero_division=0,
        )
    )


    fold_f1 = (
        f1_score(
            y_test,
            predictions,
            zero_division=0,
        )
    )


    fold_roc_auc = (
        roc_auc_score(
            y_test,
            probabilities,
        )
    )


    fold_pr_auc = (
        average_precision_score(
            y_test,
            probabilities,
        )
    )


    fold_records.append(
        {

            "fold":
                fold_number,

            "train_assemblies":
                len(
                    train_ids
                ),

            "test_assemblies":
                len(
                    test_ids
                ),

            "assembly_overlap":
                len(
                    overlap
                ),

            "balanced_accuracy":
                fold_balanced_accuracy,

            "recall_attention":
                fold_recall,

            "precision_attention":
                fold_precision,

            "f1_attention":
                fold_f1,

            "roc_auc":
                fold_roc_auc,

            "pr_auc":
                fold_pr_auc,
        }
    )


    print(
        f"\nFold {fold_number}/{N_SPLITS}"
    )

    print(
        f"Train assemblies    : "
        f"{len(train_ids)}"
    )

    print(
        f"Test assemblies     : "
        f"{len(test_ids)}"
    )

    print(
        f"Assembly overlap    : "
        f"{len(overlap)}"
    )

    print(
        f"Balanced accuracy   : "
        f"{fold_balanced_accuracy * 100.0:.2f}%"
    )

    print(
        f"Attention recall    : "
        f"{fold_recall * 100.0:.2f}%"
    )


# ============================================================
# OOF COMPLETENESS CHECK
# ============================================================

missing_probability_count = int(
    analysis_df[
        "oof_attention_probability"
    ]
    .isna()
    .sum()
)


missing_prediction_count = int(
    analysis_df[
        "oof_predicted_attention"
    ]
    .isna()
    .sum()
)


if missing_probability_count != 0:

    raise RuntimeError(
        f"\n{missing_probability_count} decisions "
        "did not receive an OOF probability."
    )


if missing_prediction_count != 0:

    raise RuntimeError(
        f"\n{missing_prediction_count} decisions "
        "did not receive an OOF prediction."
    )


# ============================================================
# OVERALL OUT-OF-FOLD PERFORMANCE
# ============================================================

y_true = (
    analysis_df[
        "needs_attention"
    ]
    .astype(
        int
    )
    .to_numpy()
)


y_pred = (
    analysis_df[
        "oof_predicted_attention"
    ]
    .astype(
        int
    )
    .to_numpy()
)


y_prob = (
    analysis_df[
        "oof_attention_probability"
    ]
    .to_numpy()
)


overall_balanced_accuracy = (
    balanced_accuracy_score(
        y_true,
        y_pred,
    )
)


overall_recall = (
    recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )
)


overall_precision = (
    precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )
)


overall_f1 = (
    f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )
)


overall_roc_auc = (
    roc_auc_score(
        y_true,
        y_prob,
    )
)


overall_pr_auc = (
    average_precision_score(
        y_true,
        y_prob,
    )
)


cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[
        0,
        1,
    ],
)


tn = int(
    cm[
        0,
        0
    ]
)


fp = int(
    cm[
        0,
        1
    ]
)


fn = int(
    cm[
        1,
        0
    ]
)


tp = int(
    cm[
        1,
        1
    ]
)


specificity = (
    tn
    /
    max(
        tn + fp,
        1,
    )
)


# ============================================================
# PROBLEM POPULATIONS
# ============================================================

total_decisions = len(
    analysis_df
)


total_attention = int(
    analysis_df[
        "is_attention_class"
    ]
    .sum()
)


total_difficult = int(
    analysis_df[
        "is_difficult"
    ]
    .sum()
)


total_not_improved = int(
    analysis_df[
        "is_not_improved"
    ]
    .sum()
)


total_high_regret = int(
    analysis_df[
        "high_regret_case"
    ]
    .astype(
        bool
    )
    .sum()
)


total_high_residual = int(
    analysis_df[
        "higher_residual_case"
    ]
    .astype(
        bool
    )
    .sum()
)


baseline_attention_rate = (
    total_attention
    /
    total_decisions
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "PROBLEM POPULATION"
)

print(
    "------------------------------------------------------------"
)


print(
    f"\nTotal decisions      : "
    f"{total_decisions}"
)


print(
    f"Needs attention      : "
    f"{total_attention}"
)


print(
    f"Difficult            : "
    f"{total_difficult}"
)


print(
    f"Not improved         : "
    f"{total_not_improved}"
)


print(
    f"High regret          : "
    f"{total_high_regret}"
)


print(
    f"High residual        : "
    f"{total_high_residual}"
)


# ============================================================
# WORKLOAD CURVE
# ============================================================

WORKLOAD_LEVELS = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
]


sorted_indices = (
    analysis_df[
        "oof_attention_probability"
    ]
    .sort_values(
        ascending=False
    )
    .index
    .tolist()
)


gate_records = []


for workload in WORKLOAD_LEVELS:


    n_flagged = int(
        np.ceil(
            total_decisions
            *
            workload
        )
    )


    flagged_indices = set(
        sorted_indices[
            :
            n_flagged
        ]
    )


    flagged_df = (
        analysis_df.loc[
            analysis_df.index.isin(
                flagged_indices
            )
        ]
        .copy()
    )


    attention_capture = (
        flagged_df[
            "is_attention_class"
        ]
        .sum()
        /
        max(
            total_attention,
            1,
        )
        *
        100.0
    )


    difficult_capture = (
        flagged_df[
            "is_difficult"
        ]
        .sum()
        /
        max(
            total_difficult,
            1,
        )
        *
        100.0
    )


    not_improved_capture = (
        flagged_df[
            "is_not_improved"
        ]
        .sum()
        /
        max(
            total_not_improved,
            1,
        )
        *
        100.0
    )


    high_regret_capture = (
        flagged_df[
            "high_regret_case"
        ]
        .astype(
            bool
        )
        .sum()
        /
        max(
            total_high_regret,
            1,
        )
        *
        100.0
    )


    high_residual_capture = (
        flagged_df[
            "higher_residual_case"
        ]
        .astype(
            bool
        )
        .sum()
        /
        max(
            total_high_residual,
            1,
        )
        *
        100.0
    )


    attention_precision = (
        flagged_df[
            "is_attention_class"
        ]
        .mean()
        *
        100.0
    )


    attention_lift = (
        flagged_df[
            "is_attention_class"
        ]
        .mean()
        /
        max(
            baseline_attention_rate,
            1e-9,
        )
    )


    probability_threshold = float(
        flagged_df[
            "oof_attention_probability"
        ]
        .min()
    )


    mean_flagged_probability = float(
        flagged_df[
            "oof_attention_probability"
        ]
        .mean()
    )


    gate_records.append(
        {

            "workload_percent":
                workload
                *
                100.0,

            "n_flagged":
                n_flagged,

            "probability_threshold":
                probability_threshold,

            "mean_flagged_probability":
                mean_flagged_probability,

            "attention_capture_percent":
                attention_capture,

            "difficult_capture_percent":
                difficult_capture,

            "not_improved_capture_percent":
                not_improved_capture,

            "high_regret_capture_percent":
                high_regret_capture,

            "high_residual_capture_percent":
                high_residual_capture,

            "attention_precision_percent":
                attention_precision,

            "attention_lift":
                attention_lift,
        }
    )


gate_df = pd.DataFrame(
    gate_records
)


# ============================================================
# SELECT OPERATING POINT
# ============================================================
#
# Restrict the preferred workload to at most 30%.
#
# The score prioritizes:
#
# - needs-attention capture
# - not-improved capture
# - high-regret capture
# - high-residual capture
# ============================================================

candidate_gate_df = (
    gate_df[
        gate_df[
            "workload_percent"
        ]
        <=
        30.0
    ]
    .copy()
)


candidate_gate_df[
    "selection_score"
] = (

    0.45
    *
    candidate_gate_df[
        "attention_capture_percent"
    ]

    +

    0.25
    *
    candidate_gate_df[
        "not_improved_capture_percent"
    ]

    +

    0.20
    *
    candidate_gate_df[
        "high_regret_capture_percent"
    ]

    +

    0.10
    *
    candidate_gate_df[
        "high_residual_capture_percent"
    ]
)


best_gate_row = (
    candidate_gate_df
    .sort_values(
        [
            "selection_score",
            "workload_percent",
        ],
        ascending=[
            False,
            True,
        ],
    )
    .iloc[
        0
    ]
)


best_workload = float(
    best_gate_row[
        "workload_percent"
    ]
)


best_threshold = float(
    best_gate_row[
        "probability_threshold"
    ]
)


best_attention_capture = float(
    best_gate_row[
        "attention_capture_percent"
    ]
)


best_difficult_capture = float(
    best_gate_row[
        "difficult_capture_percent"
    ]
)


best_not_improved_capture = float(
    best_gate_row[
        "not_improved_capture_percent"
    ]
)


best_high_regret_capture = float(
    best_gate_row[
        "high_regret_capture_percent"
    ]
)


best_high_residual_capture = float(
    best_gate_row[
        "high_residual_capture_percent"
    ]
)


best_precision = float(
    best_gate_row[
        "attention_precision_percent"
    ]
)


best_lift = float(
    best_gate_row[
        "attention_lift"
    ]
)


# ============================================================
# SAVE TABLES
# ============================================================

fold_df = pd.DataFrame(
    fold_records
)


fold_df.to_csv(
    FOLD_FILE,
    index=False,
)


analysis_df.to_csv(
    OOF_FILE,
    index=False,
)


gate_df.to_csv(
    GATE_FILE,
    index=False,
)


summary_df = pd.DataFrame(
    {

        "metric": [

            "oof_balanced_accuracy",

            "oof_recall_attention",

            "oof_precision_attention",

            "oof_specificity_normal",

            "oof_f1_attention",

            "oof_roc_auc",

            "oof_pr_auc",

            "selected_workload_percent",

            "selected_probability_threshold",

            "selected_attention_capture_percent",

            "selected_difficult_capture_percent",

            "selected_not_improved_capture_percent",

            "selected_high_regret_capture_percent",

            "selected_high_residual_capture_percent",

            "selected_attention_precision_percent",

            "selected_attention_lift",
        ],

        "value": [

            overall_balanced_accuracy,

            overall_recall,

            overall_precision,

            specificity,

            overall_f1,

            overall_roc_auc,

            overall_pr_auc,

            best_workload,

            best_threshold,

            best_attention_capture,

            best_difficult_capture,

            best_not_improved_capture,

            best_high_regret_capture,

            best_high_residual_capture,

            best_precision,

            best_lift,
        ],
    }
)


summary_df.to_csv(
    SUMMARY_FILE,
    index=False,
)


# ============================================================
# FIGURE 1
# CAPTURE VS WORKLOAD
# ============================================================

plt.figure(
    figsize=(
        8.0,
        5.5,
    )
)


plt.plot(
    gate_df[
        "workload_percent"
    ],
    gate_df[
        "attention_capture_percent"
    ],
    marker="o",
    label="Needs attention",
)


plt.plot(
    gate_df[
        "workload_percent"
    ],
    gate_df[
        "not_improved_capture_percent"
    ],
    marker="o",
    label="Not improved",
)


plt.plot(
    gate_df[
        "workload_percent"
    ],
    gate_df[
        "high_regret_capture_percent"
    ],
    marker="o",
    label="High regret",
)


plt.plot(
    gate_df[
        "workload_percent"
    ],
    gate_df[
        "high_residual_capture_percent"
    ],
    marker="o",
    label="High residual",
)


plt.axvline(
    best_workload,
    linestyle="--",
    label=(
        f"Selected workload = "
        f"{best_workload:.0f}%"
    ),
)


plt.xlabel(
    "Decisions flagged for additional attention (%)"
)


plt.ylabel(
    "Problem-case capture (%)"
)


plt.title(
    "Risk Gate: Capture vs Review Workload"
)


plt.legend()


plt.grid(
    alpha=0.25,
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "risk_gate_capture_curve.png",
    ),
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# ============================================================
# FIGURE 2
# OOF RISK DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(
        8.0,
        5.0,
    )
)


plt.hist(
    analysis_df.loc[
        ~analysis_df[
            "is_attention_class"
        ],
        "oof_attention_probability",
    ],
    bins=25,
    alpha=0.60,
    label="Normal",
)


plt.hist(
    analysis_df.loc[
        analysis_df[
            "is_attention_class"
        ],
        "oof_attention_probability",
    ],
    bins=25,
    alpha=0.60,
    label="Needs attention",
)


plt.axvline(
    best_threshold,
    linestyle="--",
    label=(
        f"Selected threshold = "
        f"{best_threshold:.3f}"
    ),
)


plt.xlabel(
    "Out-of-fold difficulty probability"
)


plt.ylabel(
    "Number of decisions"
)


plt.title(
    "Pre-Decision Difficulty Risk Distribution"
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
        "risk_probability_distribution.png",
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
    "OUT-OF-FOLD PREDICTION QUALITY"
)

print(
    "============================================================"
)


print(
    f"\nBalanced accuracy   : "
    f"{overall_balanced_accuracy * 100.0:.2f}%"
)


print(
    f"Attention recall    : "
    f"{overall_recall * 100.0:.2f}%"
)


print(
    f"Attention precision : "
    f"{overall_precision * 100.0:.2f}%"
)


print(
    f"Normal specificity  : "
    f"{specificity * 100.0:.2f}%"
)


print(
    f"F1 attention        : "
    f"{overall_f1:.4f}"
)


print(
    f"ROC AUC             : "
    f"{overall_roc_auc:.4f}"
)


print(
    f"PR AUC              : "
    f"{overall_pr_auc:.4f}"
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "RISK-GATE WORKLOAD CURVE"
)

print(
    "------------------------------------------------------------\n"
)


print(
    gate_df
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
    "SELECTED OPERATING POINT"
)

print(
    "------------------------------------------------------------"
)


print(
    f"\nReview workload          : "
    f"{best_workload:.2f}%"
)


print(
    f"Probability threshold    : "
    f"{best_threshold:.4f}"
)


print(
    f"Needs-attention captured : "
    f"{best_attention_capture:.2f}%"
)


print(
    f"Difficult captured       : "
    f"{best_difficult_capture:.2f}%"
)


print(
    f"Not-improved captured    : "
    f"{best_not_improved_capture:.2f}%"
)


print(
    f"High-regret captured     : "
    f"{best_high_regret_capture:.2f}%"
)


print(
    f"High-residual captured   : "
    f"{best_high_residual_capture:.2f}%"
)


print(
    f"Attention precision      : "
    f"{best_precision:.2f}%"
)


print(
    f"Attention lift           : "
    f"{best_lift:.2f}x"
)


# ============================================================
# VERDICT
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "RISK-GATE VERDICT"
)

print(
    "============================================================"
)


if (
    best_workload
    <=
    30.0
    and
    best_attention_capture
    >=
    60.0
    and
    best_not_improved_capture
    >=
    60.0
    and
    best_lift
    >=
    1.50
):


    print(
        "\nRESULT A:"
    )


    print(
        "A limited review workload captures a large share "
        "of difficult assembly decisions."
    )


    print(
        "\nThe pre-decision difficulty probability provides "
        "a useful routing signal."
    )


    print(
        "\nA difficulty-aware decision-support experiment "
        "is justified."
    )


elif (
    best_workload
    <=
    35.0
    and
    best_attention_capture
    >=
    50.0
):


    print(
        "\nRESULT B:"
    )


    print(
        "The risk gate provides useful enrichment, but the "
        "workload/capture trade-off is moderate."
    )


    print(
        "\nTreat the gate as an advisory signal rather than "
        "an automatic routing rule."
    )


else:


    print(
        "\nRESULT C:"
    )


    print(
        "The out-of-fold risk ranking does not concentrate "
        "enough difficult decisions."
    )


    print(
        "\nDo not add a dedicated routing gate."
    )


print(
    "\nIMPORTANT:"
)


print(
    "All risk probabilities were generated out-of-fold "
    "using complete assembly groups."
)


print(
    "Post-correction outcomes were used only for evaluation "
    "and were not predictor inputs."
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
    "RISK-GATE EVALUATION COMPLETED"
)

print(
    "============================================================"
)