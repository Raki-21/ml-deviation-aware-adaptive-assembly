import os
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
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
# CORRECTABILITY PREDICTION VALIDATION
# ============================================================
#
# Repeated assembly-wise holdout validation of the
# pre-decision correctability predictor.
#
# No model inputs are allowed to contain post-correction
# information.
# ============================================================


# ============================================================
# INPUT
# ============================================================

DATA_FILE = (
    "04_correctability_extension/"
    "results/prediction/tables/"
    "correctability_prediction_dataset.csv"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    "04_correctability_extension/"
    "results/prediction/validation"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


SPLIT_RESULT_FILE = os.path.join(
    OUTPUT_DIR,
    "repeated_holdout_results.csv",
)


SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "repeated_holdout_summary.csv",
)


# ============================================================
# SETTINGS
# ============================================================

N_SPLITS = 10

TEST_SIZE = 0.20

BASE_RANDOM_STATE = 20260831

N_TREES = 250

N_JOBS = 2


# ============================================================
# FILE CHECK
# ============================================================

if not os.path.exists(
    DATA_FILE
):

    raise FileNotFoundError(
        f"\nMissing prediction dataset:\n{DATA_FILE}"
    )


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    DATA_FILE
)


print(
    "\n"
    "============================================================"
)

print(
    "CORRECTABILITY PREDICTION VALIDATION"
)

print(
    "============================================================"
)


print(
    f"\nRows       : {len(df)}"
)


print(
    f"Assemblies : "
    f"{df['assembly_id'].nunique()}"
)


# ============================================================
# FEATURE GROUPS
# ============================================================

STAGE_FEATURES = [
    "component_index",
]


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


FEATURE_GROUPS = {

    "STAGE_ONLY":
        STAGE_FEATURES,

    "COMPONENT_PROCESS":
        COMPONENT_PROCESS_FEATURES,

    "STATE_ONLY":
        STATE_FEATURES,

    "COMBINED":
        COMBINED_FEATURES,
}


# ============================================================
# TARGET
# ============================================================

TARGET_COLUMN = (
    "needs_attention"
)


if TARGET_COLUMN not in df.columns:

    raise ValueError(
        f"\nMissing target column: "
        f"{TARGET_COLUMN}"
    )


# ============================================================
# MODEL
# ============================================================

def make_model(
    random_state
):

    return RandomForestClassifier(

        n_estimators=N_TREES,

        min_samples_leaf=3,

        class_weight="balanced",

        random_state=random_state,

        n_jobs=N_JOBS,
    )


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    y_prob,
):

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            y_pred,
        )
    )


    precision = (
        precision_score(
            y_true,
            y_pred,
            zero_division=0,
        )
    )


    recall = (
        recall_score(
            y_true,
            y_pred,
            zero_division=0,
        )
    )


    f1 = (
        f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )
    )


    roc_auc = (
        roc_auc_score(
            y_true,
            y_prob,
        )
    )


    pr_auc = (
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


    return {

        "balanced_accuracy":
            float(
                balanced_accuracy
            ),

        "precision_attention":
            float(
                precision
            ),

        "recall_attention":
            float(
                recall
            ),

        "specificity_normal":
            float(
                specificity
            ),

        "f1_attention":
            float(
                f1
            ),

        "roc_auc":
            float(
                roc_auc
            ),

        "pr_auc":
            float(
                pr_auc
            ),

        "tn":
            tn,

        "fp":
            fp,

        "fn":
            fn,

        "tp":
            tp,
    }


# ============================================================
# REPEATED ASSEMBLY-WISE HOLDOUT
# ============================================================

records = []


groups = df[
    "assembly_id"
]


for split_number in range(
    1,
    N_SPLITS + 1,
):


    split_seed = (
        BASE_RANDOM_STATE
        +
        split_number
    )


    splitter = GroupShuffleSplit(

        n_splits=1,

        test_size=TEST_SIZE,

        random_state=split_seed,
    )


    train_index, test_index = next(

        splitter.split(
            df,
            groups=groups,
        )
    )


    train_df = (
        df
        .iloc[
            train_index
        ]
        .copy()
    )


    test_df = (
        df
        .iloc[
            test_index
        ]
        .copy()
    )


    train_assemblies = set(
        train_df[
            "assembly_id"
        ]
        .unique()
    )


    test_assemblies = set(
        test_df[
            "assembly_id"
        ]
        .unique()
    )


    overlap = (
        train_assemblies
        .intersection(
            test_assemblies
        )
    )


    if len(
        overlap
    ) != 0:

        raise RuntimeError(
            "\nAssembly leakage detected."
        )


    y_train = (
        train_df[
            TARGET_COLUMN
        ]
        .to_numpy()
    )


    y_test = (
        test_df[
            TARGET_COLUMN
        ]
        .to_numpy()
    )


    print(
        f"\nSplit "
        f"{split_number}/"
        f"{N_SPLITS}"
        f" | train assemblies="
        f"{len(train_assemblies)}"
        f" | test assemblies="
        f"{len(test_assemblies)}"
    )


    for (
        feature_group,
        feature_columns,
    ) in FEATURE_GROUPS.items():


        X_train = (
            train_df[
                feature_columns
            ]
        )


        X_test = (
            test_df[
                feature_columns
            ]
        )


        model = make_model(
            random_state=split_seed,
        )


        model.fit(
            X_train,
            y_train,
        )


        y_pred = model.predict(
            X_test
        )


        y_prob = (
            model.predict_proba(
                X_test
            )[
                :,
                1
            ]
        )


        metrics = calculate_metrics(

            y_true=y_test,

            y_pred=y_pred,

            y_prob=y_prob,
        )


        records.append(
            {

                "split":
                    split_number,

                "split_seed":
                    split_seed,

                "feature_group":
                    feature_group,

                "train_assemblies":
                    len(
                        train_assemblies
                    ),

                "test_assemblies":
                    len(
                        test_assemblies
                    ),

                "train_decisions":
                    len(
                        train_df
                    ),

                "test_decisions":
                    len(
                        test_df
                    ),

                "train_attention_percent":
                    float(
                        np.mean(
                            y_train
                        )
                        *
                        100.0
                    ),

                "test_attention_percent":
                    float(
                        np.mean(
                            y_test
                        )
                        *
                        100.0
                    ),

                **metrics,
            }
        )


result_df = pd.DataFrame(
    records
)


# ============================================================
# SAVE SPLIT RESULTS
# ============================================================

result_df.to_csv(
    SPLIT_RESULT_FILE,
    index=False,
)


# ============================================================
# SUMMARY BY FEATURE GROUP
# ============================================================

summary_records = []


for feature_group in FEATURE_GROUPS:


    subset = (
        result_df[
            result_df[
                "feature_group"
            ]
            ==
            feature_group
        ]
    )


    summary_records.append(
        {

            "feature_group":
                feature_group,

            "splits":
                len(
                    subset
                ),

            "balanced_accuracy_mean":
                float(
                    subset[
                        "balanced_accuracy"
                    ]
                    .mean()
                ),

            "balanced_accuracy_std":
                float(
                    subset[
                        "balanced_accuracy"
                    ]
                    .std()
                ),

            "balanced_accuracy_min":
                float(
                    subset[
                        "balanced_accuracy"
                    ]
                    .min()
                ),

            "balanced_accuracy_max":
                float(
                    subset[
                        "balanced_accuracy"
                    ]
                    .max()
                ),

            "recall_attention_mean":
                float(
                    subset[
                        "recall_attention"
                    ]
                    .mean()
                ),

            "recall_attention_min":
                float(
                    subset[
                        "recall_attention"
                    ]
                    .min()
                ),

            "specificity_mean":
                float(
                    subset[
                        "specificity_normal"
                    ]
                    .mean()
                ),

            "f1_mean":
                float(
                    subset[
                        "f1_attention"
                    ]
                    .mean()
                ),

            "roc_auc_mean":
                float(
                    subset[
                        "roc_auc"
                    ]
                    .mean()
                ),

            "roc_auc_std":
                float(
                    subset[
                        "roc_auc"
                    ]
                    .std()
                ),

            "pr_auc_mean":
                float(
                    subset[
                        "pr_auc"
                    ]
                    .mean()
                ),
        }
    )


summary_df = pd.DataFrame(
    summary_records
)


summary_df.to_csv(
    SUMMARY_FILE,
    index=False,
)


# ============================================================
# INFORMATION VALUE
# ============================================================

lookup = (
    summary_df
    .set_index(
        "feature_group"
    )
)


stage_mean = float(
    lookup.loc[
        "STAGE_ONLY",
        "balanced_accuracy_mean",
    ]
)


component_mean = float(
    lookup.loc[
        "COMPONENT_PROCESS",
        "balanced_accuracy_mean",
    ]
)


state_mean = float(
    lookup.loc[
        "STATE_ONLY",
        "balanced_accuracy_mean",
    ]
)


combined_mean = float(
    lookup.loc[
        "COMBINED",
        "balanced_accuracy_mean",
    ]
)


combined_std = float(
    lookup.loc[
        "COMBINED",
        "balanced_accuracy_std",
    ]
)


combined_min = float(
    lookup.loc[
        "COMBINED",
        "balanced_accuracy_min",
    ]
)


combined_recall = float(
    lookup.loc[
        "COMBINED",
        "recall_attention_mean",
    ]
)


combined_recall_min = float(
    lookup.loc[
        "COMBINED",
        "recall_attention_min",
    ]
)


combined_auc = float(
    lookup.loc[
        "COMBINED",
        "roc_auc_mean",
    ]
)


gain_vs_stage = (
    combined_mean
    -
    stage_mean
)


gain_vs_component = (
    combined_mean
    -
    component_mean
)


gain_vs_state = (
    combined_mean
    -
    state_mean
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "REPEATED HOLDOUT SUMMARY"
)

print(
    "============================================================\n"
)


print(
    summary_df
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
    "COMBINED MODEL ROBUSTNESS"
)

print(
    "------------------------------------------------------------"
)


print(
    f"\nMean balanced accuracy : "
    f"{combined_mean * 100.0:.2f}%"
)


print(
    f"Std balanced accuracy  : "
    f"{combined_std * 100.0:.2f} pp"
)


print(
    f"Minimum balanced acc.  : "
    f"{combined_min * 100.0:.2f}%"
)


print(
    f"Mean attention recall  : "
    f"{combined_recall * 100.0:.2f}%"
)


print(
    f"Minimum attention recall: "
    f"{combined_recall_min * 100.0:.2f}%"
)


print(
    f"Mean ROC AUC           : "
    f"{combined_auc:.4f}"
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "INFORMATION GAIN"
)

print(
    "------------------------------------------------------------"
)


print(
    f"\nCombined vs Stage       : "
    f"{gain_vs_stage * 100.0:+.2f} pp"
)


print(
    f"Combined vs Component   : "
    f"{gain_vs_component * 100.0:+.2f} pp"
)


print(
    f"Combined vs State       : "
    f"{gain_vs_state * 100.0:+.2f} pp"
)


# ============================================================
# VERDICT
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "CORRECTABILITY VALIDATION VERDICT"
)

print(
    "============================================================"
)


if (
    combined_mean
    >=
    0.75
    and
    combined_min
    >=
    0.70
    and
    gain_vs_stage
    >=
    0.10
    and
    combined_recall
    >=
    0.70
):


    print(
        "\nRESULT A:"
    )


    print(
        "The pre-decision correctability predictor is stable "
        "across repeated assembly-wise holdout splits."
    )


    print(
        "\nThe combined pre-decision information consistently "
        "outperforms simple stage information."
    )


    print(
        "\nA difficulty-aware routing experiment is justified."
    )


elif (
    combined_mean
    >=
    0.70
    and
    gain_vs_stage
    >=
    0.05
):


    print(
        "\nRESULT B:"
    )


    print(
        "The pre-decision predictor shows useful but moderate "
        "robustness."
    )


    print(
        "\nKeep the concept as an extension, but avoid treating "
        "it as a fully established routing layer."
    )


else:


    print(
        "\nRESULT C:"
    )


    print(
        "The predictor is not stable enough across repeated "
        "assembly-wise splits."
    )


    print(
        "\nDo not proceed to an automated difficulty-aware "
        "routing architecture."
    )


print(
    "\nSaved:"
)


print(
    SPLIT_RESULT_FILE
)


print(
    SUMMARY_FILE
)


print(
    "\n"
    "============================================================"
)

print(
    "CORRECTABILITY PREDICTION VALIDATION COMPLETED"
)

print(
    "============================================================"
)