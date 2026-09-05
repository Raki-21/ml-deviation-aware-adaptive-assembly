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
# OBSERVABLE-INFORMATION AUDIT
# ============================================================
#
# Purpose
# -------
# Determine whether pre-decision difficulty prediction depends
# on privileged synthetic-generator parameters.
#
# Three information levels are compared:
#
# 1. PRIVILEGED
#    Explicit deviation-generator parameters and process data.
#
# 2. OBSERVABLE
#    Geometric information that could in principle be obtained
#    from measured component / assembly profiles together with
#    known process context.
#
# 3. FULL
#    Combination of both.
#
# Complete batch/process-signature groups are held out during
# validation.
#
# The frozen V3.2 system is not modified.
# ============================================================


# ============================================================
# PATHS
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
    "observable_feature_audit",
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


FOLD_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "observable_feature_fold_results.csv",
)


SUMMARY_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "observable_feature_summary.csv",
)


# ============================================================
# SETTINGS
# ============================================================

N_TREES = 300

N_JOBS = 2

RANDOM_STATE = 20260827


# ============================================================
# LOAD
# ============================================================

if not os.path.exists(
    INPUT_FILE
):

    raise FileNotFoundError(
        f"\nMissing prediction dataset:\n{INPUT_FILE}"
    )


df = pd.read_csv(
    INPUT_FILE
)


print(
    "\n"
    "============================================================"
)

print(
    "OBSERVABLE-INFORMATION AUDIT"
)

print(
    "============================================================"
)


print(
    f"\nRows       : {len(df)}"
)


print(
    f"Assemblies : {df['assembly_id'].nunique()}"
)


# ============================================================
# TARGET
# ============================================================

if (
    "needs_attention"
    not in df.columns
):

    raise ValueError(
        "\nColumn 'needs_attention' is missing."
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
# RECONSTRUCT HARD BATCH/PROCESS GROUPS
# ============================================================

batch_signature_columns = [
    "batch_offset_bias_mm",
    "batch_angular_bias_deg",
    "fixture_drift_mm",
    "variation_multiplier",
]


missing_batch_columns = [
    column
    for column in batch_signature_columns
    if column not in df.columns
]


if missing_batch_columns:

    raise ValueError(
        "\nMissing batch/process grouping columns:\n"
        +
        "\n".join(
            missing_batch_columns
        )
    )


signature_df = (
    df[
        batch_signature_columns
    ]
    .copy()
)


for column in batch_signature_columns:

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


unique_groups = (
    df[
        "audit_batch_group"
    ]
    .drop_duplicates()
    .tolist()
)


print(
    f"\nBatch/process groups : "
    f"{len(unique_groups)}"
)


# ============================================================
# FEATURE DEFINITIONS
# ============================================================


# ------------------------------------------------------------
# Privileged synthetic-generator information
# ------------------------------------------------------------

PRIVILEGED_GEOMETRY_FEATURES = [

    "component_index",

    "offset_mm",

    "tilt_deg",

    "bend_mm",

    "waviness_mm",

    "twist_mm",

    "local_bump_mm",

    "n_active_modes",

    "batch_offset_bias_mm",

    "batch_angular_bias_deg",

    "fixture_drift_mm",

    "variation_multiplier",
]


# ------------------------------------------------------------
# Observable incoming-component geometry
# ------------------------------------------------------------

OBSERVABLE_COMPONENT_FEATURES = [

    "component_index",

    "component_profile_mean_mm",

    "component_profile_min_mm",

    "component_profile_max_mm",

    "component_profile_rms_mm",

    "component_parallelism_mm",
]


# ------------------------------------------------------------
# Observable accumulated assembly state
# ------------------------------------------------------------

OBSERVABLE_STATE_FEATURES = [

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


# ------------------------------------------------------------
# Known process context
# ------------------------------------------------------------
#
# These variables are treated separately because a production
# system could know current fixture/process settings even if
# it does not know the latent component deviation decomposition.
# ------------------------------------------------------------

KNOWN_PROCESS_FEATURES = [

    "batch_offset_bias_mm",

    "batch_angular_bias_deg",

    "fixture_drift_mm",

    "variation_multiplier",
]


OBSERVABLE_FEATURES = list(
    dict.fromkeys(

        OBSERVABLE_COMPONENT_FEATURES

        +

        OBSERVABLE_STATE_FEATURES

        +

        KNOWN_PROCESS_FEATURES
    )
)


FULL_FEATURES = list(
    dict.fromkeys(

        PRIVILEGED_GEOMETRY_FEATURES

        +

        OBSERVABLE_COMPONENT_FEATURES

        +

        OBSERVABLE_STATE_FEATURES
    )
)


FEATURE_SETS = {

    "PRIVILEGED":
        PRIVILEGED_GEOMETRY_FEATURES,

    "OBSERVABLE":
        OBSERVABLE_FEATURES,

    "FULL":
        FULL_FEATURES,
}


# ============================================================
# FEATURE CHECK
# ============================================================

for feature_set_name, features in (
    FEATURE_SETS.items()
):


    missing_features = [
        feature
        for feature in features
        if feature not in df.columns
    ]


    if missing_features:

        raise ValueError(
            f"\nMissing features for "
            f"{feature_set_name}:\n"
            +
            "\n".join(
                missing_features
            )
        )


    missing_values = int(
        df[
            features
        ]
        .isna()
        .sum()
        .sum()
    )


    if missing_values != 0:

        raise ValueError(
            f"\n{feature_set_name} contains "
            f"{missing_values} missing values."
        )


# ============================================================
# MODEL
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
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    y_probability,
):


    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            y_pred,
        )
    )


    recall = (
        recall_score(
            y_true,
            y_pred,
            zero_division=0,
        )
    )


    precision = (
        precision_score(
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


    specificity = (
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
            recall,

        "attention_precision":
            precision,

        "normal_specificity":
            specificity,

        "f1_attention":
            f1,

        "roc_auc":
            roc_auc,

        "pr_auc":
            pr_auc,
    }


# ============================================================
# HARD GROUP-HELD-OUT VALIDATION
# ============================================================

fold_records = []


for feature_set_number, (
    feature_set_name,
    features,
) in enumerate(
    FEATURE_SETS.items(),
    start=1,
):


    print(
        "\n"
        "------------------------------------------------------------"
    )


    print(
        f"FEATURE SET: {feature_set_name}"
    )


    print(
        "------------------------------------------------------------"
    )


    for fold_number, test_group in enumerate(
        unique_groups,
        start=1,
    ):


        train_mask = (
            df[
                "audit_batch_group"
            ]
            !=
            test_group
        )


        test_mask = (
            df[
                "audit_batch_group"
            ]
            ==
            test_group
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


        if overlap:

            raise RuntimeError(
                "\nAssembly leakage detected."
            )


        X_train = (
            train_df[
                features
            ]
        )


        X_test = (
            test_df[
                features
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


        model = create_model(

            RANDOM_STATE

            +

            feature_set_number
            *
            100

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

            y_pred=predictions,

            y_probability=probabilities,
        )


        fold_record = {

            "feature_set":
                feature_set_name,

            "fold":
                fold_number,

            "test_group":
                str(
                    test_group
                ),

            "test_assemblies":
                test_df[
                    "assembly_id"
                ]
                .nunique(),

            "test_decisions":
                len(
                    test_df
                ),

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
            f"{fold_number}/"
            f"{len(unique_groups)}"
            f" | BA "
            f"{metrics['balanced_accuracy'] * 100.0:.2f}%"
        )


# ============================================================
# SUMMARY
# ============================================================

fold_df = pd.DataFrame(
    fold_records
)


summary_records = []


for feature_set_name in (
    FEATURE_SETS.keys()
):


    subset = (
        fold_df[
            fold_df[
                "feature_set"
            ]
            ==
            feature_set_name
        ]
    )


    summary_records.append(
        {

            "feature_set":
                feature_set_name,

            "mean_balanced_accuracy":
                subset[
                    "balanced_accuracy"
                ]
                .mean(),

            "std_balanced_accuracy":
                subset[
                    "balanced_accuracy"
                ]
                .std(
                    ddof=1
                ),

            "minimum_balanced_accuracy":
                subset[
                    "balanced_accuracy"
                ]
                .min(),

            "mean_attention_recall":
                subset[
                    "attention_recall"
                ]
                .mean(),

            "mean_attention_precision":
                subset[
                    "attention_precision"
                ]
                .mean(),

            "mean_normal_specificity":
                subset[
                    "normal_specificity"
                ]
                .mean(),

            "mean_f1":
                subset[
                    "f1_attention"
                ]
                .mean(),

            "mean_roc_auc":
                subset[
                    "roc_auc"
                ]
                .mean(),

            "mean_pr_auc":
                subset[
                    "pr_auc"
                ]
                .mean(),
        }
    )


summary_df = pd.DataFrame(
    summary_records
)


# ============================================================
# INFORMATION DIFFERENCES
# ============================================================

summary_lookup = (
    summary_df
    .set_index(
        "feature_set"
    )
)


observable_ba = float(

    summary_lookup.loc[
        "OBSERVABLE",
        "mean_balanced_accuracy",
    ]
)


privileged_ba = float(

    summary_lookup.loc[
        "PRIVILEGED",
        "mean_balanced_accuracy",
    ]
)


full_ba = float(

    summary_lookup.loc[
        "FULL",
        "mean_balanced_accuracy",
    ]
)


observable_gap_to_full_pp = (

    (
        full_ba
        -
        observable_ba
    )

    *
    100.0
)


observable_gain_over_privileged_pp = (

    (
        observable_ba
        -
        privileged_ba
    )

    *
    100.0
)


# ============================================================
# SAVE
# ============================================================

fold_df.to_csv(
    FOLD_OUTPUT,
    index=False,
)


summary_df.to_csv(
    SUMMARY_OUTPUT,
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
    "OBSERVABLE-INFORMATION SUMMARY"
)

print(
    "============================================================"
)


print(
    "\n"
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
    "\nObservable gap to FULL:"
)


print(
    f"\n{observable_gap_to_full_pp:.2f} "
    "percentage points balanced accuracy"
)


print(
    "\nObservable gain over PRIVILEGED:"
)


print(
    f"\n{observable_gain_over_privileged_pp:.2f} "
    "percentage points balanced accuracy"
)


# ============================================================
# VERDICT
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "AUDIT 4 VERDICT"
)

print(
    "============================================================"
)


if (
    observable_ba
    >=
    0.75
    and
    observable_gap_to_full_pp
    <=
    5.0
):


    print(
        "\nRESULT A:"
    )


    print(
        "The difficulty predictor remains strong using "
        "observable geometric/state information."
    )


    print(
        "\nIts performance is close to the full model, "
        "indicating that explicit latent synthetic-deviation "
        "parameters are not essential to the advisory concept."
    )


elif (
    observable_ba
    >=
    0.68
):


    print(
        "\nRESULT B:"
    )


    print(
        "Observable geometry retains meaningful predictive "
        "information, but privileged simulator variables "
        "provide a measurable additional advantage."
    )


    print(
        "\nThe advisory concept remains viable, but future "
        "implementation should explicitly address how latent "
        "deviation descriptors are estimated or replaced by "
        "measured geometry."
    )


else:


    print(
        "\nRESULT C:"
    )


    print(
        "Performance falls substantially when privileged "
        "synthetic-generator parameters are removed."
    )


    print(
        "\nThe present difficulty predictor therefore depends "
        "strongly on information that may not be directly "
        "available from a measurement system."
    )


print(
    "\nSaved results:"
)


print(
    OUTPUT_DIR
)


print(
    "\n"
    "============================================================"
)

print(
    "OBSERVABLE-INFORMATION AUDIT COMPLETED"
)

print(
    "============================================================"
)