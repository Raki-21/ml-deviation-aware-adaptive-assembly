import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# UNSEEN-BATCH DIFFICULTY VALIDATION
# ============================================================
#
# Purpose:
#
# Test whether the pre-decision difficulty predictor can
# generalize to completely unseen batch realizations.
#
# Previous correctability validation separated assemblies.
# This audit performs a stricter evaluation in which all
# decisions belonging to one batch are held out together.
#
# The frozen V3.2 controller is not modified.
# ============================================================


# ============================================================
# PROJECT PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

AUDIT_DIR = os.path.dirname(
    SCRIPT_DIR
)

PROJECT_ROOT = os.path.dirname(
    AUDIT_DIR
)


INPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "03_v3_2_sequential_multicomponent",
    "04_correctability_extension",
    "results",
    "prediction",
    "tables",
    "correctability_prediction_dataset.csv",
)


OUTPUT_DIR = os.path.join(
    AUDIT_DIR,
    "results",
    "unseen_batch_validation",
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


FOLD_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "unseen_batch_fold_results.csv",
)


SUMMARY_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "unseen_batch_summary.csv",
)


PREDICTION_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "unseen_batch_oof_predictions.csv",
)


# ============================================================
# SETTINGS
# ============================================================

N_TREES = 300

N_JOBS = 2

RANDOM_STATE = 20260827


# ============================================================
# CHECK INPUT FILE
# ============================================================

if not os.path.exists(
    INPUT_FILE
):

    raise FileNotFoundError(
        "\nCould not find the correctability prediction dataset:\n"
        f"{INPUT_FILE}"
    )


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)


print(
    "\n"
    "============================================================"
)

print(
    "UNSEEN-BATCH DIFFICULTY VALIDATION"
)

print(
    "============================================================"
)


print(
    f"\nInput file:\n{INPUT_FILE}"
)


print(
    f"\nRows       : {len(df)}"
)


print(
    f"Assemblies : {df['assembly_id'].nunique()}"
)


# ============================================================
# TARGET CHECK
# ============================================================

if (
    "needs_attention"
    not in df.columns
):

    raise ValueError(
        "\nColumn 'needs_attention' is missing from the "
        "prediction dataset."
    )


df[
    "needs_attention"
] = (
    df[
        "needs_attention"
    ]
    .astype(
        int
    )
)


# ============================================================
# IDENTIFY BATCH GROUP
# ============================================================
#
# Prefer a true batch identifier if one already exists.
#
# If not, reconstruct a batch grouping using combinations of
# batch-level process variables.
#
# We do not use assembly_id as the grouping variable here.
# ============================================================

batch_id_candidates = [
    "batch_id",
    "batch_index",
    "batch_number",
    "batch",
]


batch_column = None


for candidate in batch_id_candidates:

    if candidate in df.columns:

        batch_column = candidate
        break


if batch_column is not None:

    df[
        "audit_batch_group"
    ] = (
        df[
            batch_column
        ]
        .astype(
            str
        )
    )


    batch_group_source = (
        f"existing column '{batch_column}'"
    )


else:

    batch_signature_candidates = [
        "batch_offset_bias_mm",
        "batch_angular_bias_deg",
        "fixture_drift_mm",
        "variation_multiplier",
    ]


    available_batch_signature_columns = [
        column
        for column in batch_signature_candidates
        if column in df.columns
    ]


    if len(
        available_batch_signature_columns
    ) < 2:

        batch_related_columns = [
            column
            for column in df.columns
            if (
                "batch" in column.lower()
                or
                "fixture" in column.lower()
                or
                "variation" in column.lower()
            )
        ]


        print(
            "\nAvailable batch/process-related columns:"
        )


        for column in batch_related_columns:

            print(
                f"  {column}"
            )


        raise ValueError(
            "\nNo explicit batch ID was found and there are "
            "not enough batch-level variables to reconstruct "
            "batch groups safely."
        )


    signature_df = (
        df[
            available_batch_signature_columns
        ]
        .copy()
    )


    for column in (
        available_batch_signature_columns
    ):

        if pd.api.types.is_numeric_dtype(
            signature_df[
                column
            ]
        ):

            signature_df[
                column
            ] = (
                signature_df[
                    column
                ]
                .round(
                    10
                )
            )


    df[
        "audit_batch_group"
    ] = (
        signature_df
        .astype(
            str
        )
        .agg(
            "|".join,
            axis=1,
        )
    )


    batch_group_source = (
        "reconstructed from "
        +
        ", ".join(
            available_batch_signature_columns
        )
    )


# ============================================================
# BATCH-GROUP CHECK
# ============================================================

n_batches = (
    df[
        "audit_batch_group"
    ]
    .nunique()
)


print(
    f"\nBatch grouping : {batch_group_source}"
)


print(
    f"Unique batches : {n_batches}"
)


if n_batches < 5:

    raise ValueError(
        "\nToo few unique batch groups were detected for a "
        "meaningful unseen-batch validation."
    )


# ============================================================
# CHECK WHETHER ASSEMBLIES BELONG TO ONLY ONE BATCH
# ============================================================

assembly_batch_counts = (
    df.groupby(
        "assembly_id"
    )[
        "audit_batch_group"
    ]
    .nunique()
)


multi_batch_assemblies = (
    assembly_batch_counts[
        assembly_batch_counts
        >
        1
    ]
)


if len(
    multi_batch_assemblies
) != 0:

    raise ValueError(
        "\nSome assemblies are associated with more than one "
        "batch group. Batch grouping must be checked before "
        "continuing."
    )


# ============================================================
# FEATURES
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
    feature
    for feature in COMBINED_FEATURES
    if feature not in df.columns
]


if missing_features:

    print(
        "\nMissing required features:"
    )


    for feature in missing_features:

        print(
            f"  {feature}"
        )


    raise ValueError(
        "\nThe unseen-batch audit stopped because required "
        "pre-decision features are missing."
    )


missing_feature_values = int(
    df[
        COMBINED_FEATURES
    ]
    .isna()
    .sum()
    .sum()
)


if missing_feature_values != 0:

    raise ValueError(
        f"\nThe prediction dataset contains "
        f"{missing_feature_values} missing feature values."
    )


# ============================================================
# RANDOM FOREST
# ============================================================

def create_model(
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
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    y_prediction,
    y_probability,
):


    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            y_prediction,
        )
    )


    attention_recall = (
        recall_score(
            y_true,
            y_prediction,
            zero_division=0,
        )
    )


    attention_precision = (
        precision_score(
            y_true,
            y_prediction,
            zero_division=0,
        )
    )


    f1_attention = (
        f1_score(
            y_true,
            y_prediction,
            zero_division=0,
        )
    )


    confusion = confusion_matrix(
        y_true,
        y_prediction,
        labels=[
            0,
            1,
        ],
    )


    tn = int(
        confusion[
            0,
            0
        ]
    )


    fp = int(
        confusion[
            0,
            1
        ]
    )


    normal_specificity = (
        tn
        /
        max(
            tn + fp,
            1,
        )
    )


    if (
        len(
            np.unique(
                y_true
            )
        )
        ==
        2
    ):

        roc_auc = (
            roc_auc_score(
                y_true,
                y_probability,
            )
        )


        pr_auc = (
            average_precision_score(
                y_true,
                y_probability,
            )
        )


    else:

        roc_auc = np.nan

        pr_auc = np.nan


    return {

        "balanced_accuracy":
            balanced_accuracy,

        "attention_recall":
            attention_recall,

        "attention_precision":
            attention_precision,

        "normal_specificity":
            normal_specificity,

        "f1_attention":
            f1_attention,

        "roc_auc":
            roc_auc,

        "pr_auc":
            pr_auc,
    }


# ============================================================
# LEAVE-ONE-BATCH-OUT VALIDATION
# ============================================================

unique_batches = (
    df[
        "audit_batch_group"
    ]
    .drop_duplicates()
    .tolist()
)


df[
    "unseen_batch_probability"
] = np.nan


df[
    "unseen_batch_prediction"
] = np.nan


df[
    "unseen_batch_fold"
] = np.nan


fold_records = []


for fold_number, test_batch in enumerate(
    unique_batches,
    start=1,
):


    train_mask = (
        df[
            "audit_batch_group"
        ]
        !=
        test_batch
    )


    test_mask = (
        df[
            "audit_batch_group"
        ]
        ==
        test_batch
    )


    train_df = (
        df.loc[
            train_mask
        ]
        .copy()
    )


    test_df = (
        df.loc[
            test_mask
        ]
        .copy()
    )


    train_assembly_ids = set(
        train_df[
            "assembly_id"
        ]
        .unique()
    )


    test_assembly_ids = set(
        test_df[
            "assembly_id"
        ]
        .unique()
    )


    overlap = (
        train_assembly_ids
        .intersection(
            test_assembly_ids
        )
    )


    if len(
        overlap
    ) != 0:

        raise RuntimeError(
            f"\nAssembly leakage detected in unseen-batch "
            f"fold {fold_number}."
        )


    X_train = (
        train_df[
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


    X_test = (
        test_df[
            COMBINED_FEATURES
        ]
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


    if (
        len(
            np.unique(
                y_train
            )
        )
        <
        2
    ):

        raise RuntimeError(
            f"\nTraining fold {fold_number} contains only "
            "one target class."
        )


    model = create_model(
        RANDOM_STATE
        +
        fold_number
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


    metrics = calculate_metrics(

        y_true=y_test,

        y_prediction=predictions,

        y_probability=probabilities,
    )


    df.loc[
        test_mask,
        "unseen_batch_probability",
    ] = probabilities


    df.loc[
        test_mask,
        "unseen_batch_prediction",
    ] = predictions


    df.loc[
        test_mask,
        "unseen_batch_fold",
    ] = fold_number


    fold_record = {

        "fold":
            fold_number,

        "test_batch":
            str(
                test_batch
            ),

        "train_batches":
            n_batches
            -
            1,

        "test_decisions":
            len(
                test_df
            ),

        "test_assemblies":
            test_df[
                "assembly_id"
            ]
            .nunique(),

        "assembly_overlap":
            len(
                overlap
            ),

        "attention_prevalence":
            float(
                y_test.mean()
            ),
    }


    fold_record.update(
        metrics
    )


    fold_records.append(
        fold_record
    )


    print(
        f"\nFold "
        f"{fold_number}/{n_batches}"
    )


    print(
        f"Test assemblies      : "
        f"{test_df['assembly_id'].nunique()}"
    )


    print(
        f"Test decisions       : "
        f"{len(test_df)}"
    )


    print(
        f"Assembly overlap     : "
        f"{len(overlap)}"
    )


    print(
        f"Attention prevalence : "
        f"{y_test.mean() * 100.0:.2f}%"
    )


    print(
        f"Balanced accuracy    : "
        f"{metrics['balanced_accuracy'] * 100.0:.2f}%"
    )


    print(
        f"Attention recall     : "
        f"{metrics['attention_recall'] * 100.0:.2f}%"
    )


# ============================================================
# OOF COVERAGE CHECK
# ============================================================

missing_probabilities = int(
    df[
        "unseen_batch_probability"
    ]
    .isna()
    .sum()
)


missing_predictions = int(
    df[
        "unseen_batch_prediction"
    ]
    .isna()
    .sum()
)


if missing_probabilities != 0:

    raise RuntimeError(
        f"\n{missing_probabilities} decisions did not "
        "receive an unseen-batch probability."
    )


if missing_predictions != 0:

    raise RuntimeError(
        f"\n{missing_predictions} decisions did not "
        "receive an unseen-batch prediction."
    )


# ============================================================
# OVERALL OOF RESULTS
# ============================================================

y_true_all = (
    df[
        "needs_attention"
    ]
    .astype(
        int
    )
    .to_numpy()
)


y_prediction_all = (
    df[
        "unseen_batch_prediction"
    ]
    .astype(
        int
    )
    .to_numpy()
)


y_probability_all = (
    df[
        "unseen_batch_probability"
    ]
    .to_numpy()
)


overall_metrics = calculate_metrics(

    y_true=y_true_all,

    y_prediction=y_prediction_all,

    y_probability=y_probability_all,
)


fold_df = pd.DataFrame(
    fold_records
)


# ============================================================
# CREATE SUMMARY TABLE
# ============================================================

summary_records = []


metric_names = [

    "balanced_accuracy",
    "attention_recall",
    "attention_precision",
    "normal_specificity",
    "f1_attention",
    "roc_auc",
    "pr_auc",
]


for metric_name in metric_names:


    valid_values = (
        fold_df[
            metric_name
        ]
        .dropna()
    )


    summary_records.append(
        {

            "metric":
                metric_name,

            "overall_oof":
                overall_metrics[
                    metric_name
                ],

            "mean_batch":
                float(
                    valid_values.mean()
                ),

            "std_batch":
                float(
                    valid_values.std(
                        ddof=1
                    )
                ),

            "minimum_batch":
                float(
                    valid_values.min()
                ),

            "maximum_batch":
                float(
                    valid_values.max()
                ),
        }
    )


summary_df = pd.DataFrame(
    summary_records
)


# ============================================================
# SAVE RESULTS
# ============================================================

fold_df.to_csv(
    FOLD_OUTPUT,
    index=False,
)


summary_df.to_csv(
    SUMMARY_OUTPUT,
    index=False,
)


prediction_columns = [

    "assembly_id",
    "component_index",
    "audit_batch_group",
    "needs_attention",
    "unseen_batch_fold",
    "unseen_batch_probability",
    "unseen_batch_prediction",
]


df[
    prediction_columns
].to_csv(
    PREDICTION_OUTPUT,
    index=False,
)


# ============================================================
# PRINT FINAL SUMMARY
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "UNSEEN-BATCH VALIDATION SUMMARY"
)

print(
    "============================================================"
)


print(
    f"\nBatch groups evaluated : "
    f"{n_batches}"
)


print(
    f"Assemblies evaluated   : "
    f"{df['assembly_id'].nunique()}"
)


print(
    f"Decisions evaluated    : "
    f"{len(df)}"
)


print(
    "\nOverall out-of-batch performance:"
)


print(
    f"\nBalanced accuracy      : "
    f"{overall_metrics['balanced_accuracy'] * 100.0:.2f}%"
)


print(
    f"Attention recall       : "
    f"{overall_metrics['attention_recall'] * 100.0:.2f}%"
)


print(
    f"Attention precision    : "
    f"{overall_metrics['attention_precision'] * 100.0:.2f}%"
)


print(
    f"Normal specificity     : "
    f"{overall_metrics['normal_specificity'] * 100.0:.2f}%"
)


print(
    f"F1 attention           : "
    f"{overall_metrics['f1_attention']:.4f}"
)


print(
    f"ROC AUC                : "
    f"{overall_metrics['roc_auc']:.4f}"
)


print(
    f"PR AUC                 : "
    f"{overall_metrics['pr_auc']:.4f}"
)


print(
    "\nBatch-level stability:"
)


print(
    f"\nMean batch BA          : "
    f"{fold_df['balanced_accuracy'].mean() * 100.0:.2f}%"
)


print(
    f"Std batch BA           : "
    f"{fold_df['balanced_accuracy'].std(ddof=1) * 100.0:.2f} pp"
)


print(
    f"Minimum batch BA       : "
    f"{fold_df['balanced_accuracy'].min() * 100.0:.2f}%"
)


print(
    f"Maximum batch BA       : "
    f"{fold_df['balanced_accuracy'].max() * 100.0:.2f}%"
)


# ============================================================
# AUDIT INTERPRETATION
# ============================================================

overall_ba = (
    overall_metrics[
        "balanced_accuracy"
    ]
)


overall_recall = (
    overall_metrics[
        "attention_recall"
    ]
)


print(
    "\n"
    "============================================================"
)

print(
    "AUDIT VERDICT"
)

print(
    "============================================================"
)


if (
    overall_ba
    >=
    0.75
    and
    overall_recall
    >=
    0.70
):


    print(
        "\nRESULT A:"
    )


    print(
        "The difficulty predictor remains strong when "
        "complete batch groups are unseen during training."
    )


    print(
        "\nThis strengthens the evidence that the predictor "
        "is not dependent only on assembly-level splitting "
        "or previously observed batch realizations."
    )


elif (
    overall_ba
    >=
    0.68
):


    print(
        "\nRESULT B:"
    )


    print(
        "Meaningful predictive information remains under "
        "unseen-batch validation, but performance is weaker "
        "than under assembly-wise holdout."
    )


    print(
        "\nThe difference should be treated as a hierarchical "
        "generalization limitation."
    )


else:


    print(
        "\nRESULT C:"
    )


    print(
        "Prediction performance falls substantially when "
        "complete batch groups are held out."
    )


    print(
        "\nThe previous assembly-wise results therefore do not "
        "demonstrate strong generalization to unseen batch "
        "contexts."
    )


    print(
        "\nThe advisory layer should be described as "
        "batch-context dependent unless further evidence is "
        "obtained."
    )


print(
    "\nSaved fold results:"
)


print(
    FOLD_OUTPUT
)


print(
    "\nSaved summary:"
)


print(
    SUMMARY_OUTPUT
)


print(
    "\nSaved predictions:"
)


print(
    PREDICTION_OUTPUT
)


print(
    "\n"
    "============================================================"
)

print(
    "UNSEEN-BATCH AUDIT COMPLETED"
)

print(
    "============================================================"
)