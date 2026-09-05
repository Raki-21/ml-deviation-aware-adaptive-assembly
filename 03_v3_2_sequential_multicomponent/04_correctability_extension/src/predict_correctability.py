import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

# Import physical profile functions from the frozen V3.2 code.
from sequential_assembly import (
    s,
    PROFILE_LENGTH_MM,
    create_initial_state,
    deviation_offset,
    deviation_tilt,
    deviation_bend,
    deviation_waviness,
    deviation_twist,
    deviation_local_bump,
    correction_profile,
    update_assembly_state,
    calculate_quality_metrics,
)


# ============================================================
# PRE-DECISION CORRECTABILITY PREDICTION
# ============================================================
#
# Goal:
#
# Test whether difficult assembly decisions can be identified
# BEFORE the correction is applied.
#
# Four information levels are compared:
#
#   1. Stage only
#   2. Incoming component + process information
#   3. Current accumulated assembly state
#   4. Combined component + process + state information
#
#
# Target:
#
#   NORMAL
#       HIGHLY_CORRECTABLE
#       CORRECTABLE
#
#   NEEDS_ATTENTION
#       DIFFICULT
#       NOT_IMPROVED
#
#
# Important:
#
# Post-correction quality, regret, selected correction and
# observed correctability metrics are used only to create the
# target label.
#
# They are NOT used as model inputs.
# ============================================================


# ============================================================
# INPUT FILES
# ============================================================

VALIDATION_COMPONENT_FILE = (
    "data/validation/"
    "v3_2_independent_validation_component_dataset.csv"
)

CONTROLLER_RESULT_FILE = (
    "data/validation/"
    "v3_2_final_controller_validation_component_results.csv"
)

CORRECTABILITY_FILE = (
    "04_correctability_extension/"
    "results/tables/"
    "component_correctability_analysis.csv"
)


# ============================================================
# OUTPUT PATHS
# ============================================================

OUTPUT_DIR = (
    "04_correctability_extension/"
    "results/prediction"
)

FIGURE_DIR = os.path.join(
    OUTPUT_DIR,
    "figures",
)

TABLE_DIR = os.path.join(
    OUTPUT_DIR,
    "tables",
)

MODEL_DIR = os.path.join(
    OUTPUT_DIR,
    "models",
)


os.makedirs(
    FIGURE_DIR,
    exist_ok=True,
)

os.makedirs(
    TABLE_DIR,
    exist_ok=True,
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True,
)


# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    VALIDATION_COMPONENT_FILE,
    CONTROLLER_RESULT_FILE,
    CORRECTABILITY_FILE,
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
        "\nComplete the previous correctability analysis first."
    )


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 20260830

TEST_SIZE = 0.20

N_RF_TREES = 250

RF_N_JOBS = 2


# ============================================================
# PROFILE SAMPLE LOCATIONS
# ============================================================

PROFILE_SAMPLE_INDICES = [
    0,
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
]


PROFILE_SAMPLE_NAMES = [
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


# ============================================================
# LOAD DATA
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "PRE-DECISION CORRECTABILITY PREDICTION"
)

print(
    "============================================================"
)


validation_df = pd.read_csv(
    VALIDATION_COMPONENT_FILE
)


controller_df = pd.read_csv(
    CONTROLLER_RESULT_FILE
)


label_df = pd.read_csv(
    CORRECTABILITY_FILE
)


print(
    f"\nValidation component rows : "
    f"{len(validation_df)}"
)


print(
    f"Controller result rows     : "
    f"{len(controller_df)}"
)


print(
    f"Correctability label rows  : "
    f"{len(label_df)}"
)


# ============================================================
# BASIC DATA CHECK
# ============================================================

KEY_COLUMNS = [
    "assembly_id",
    "component_index",
]


for name, frame in [
    (
        "validation",
        validation_df,
    ),
    (
        "controller",
        controller_df,
    ),
    (
        "correctability",
        label_df,
    ),
]:

    duplicates = int(
        frame.duplicated(
            subset=KEY_COLUMNS
        ).sum()
    )

    if duplicates != 0:

        raise ValueError(
            f"\n{name} data contains "
            f"{duplicates} duplicate decision keys."
        )


# ============================================================
# RECONSTRUCT COMPONENT PROFILE
# ============================================================

def reconstruct_component_profile(
    row
):

    profile = np.zeros_like(
        s
    )


    if row[
        "offset_mm"
    ] != 0.0:

        profile += deviation_offset(
            offset_mm=row[
                "offset_mm"
            ]
        )


    if row[
        "tilt_deg"
    ] != 0.0:

        profile += deviation_tilt(
            angle_deg=row[
                "tilt_deg"
            ]
        )


    if row[
        "bend_mm"
    ] != 0.0:

        profile += deviation_bend(
            amplitude_mm=row[
                "bend_mm"
            ]
        )


    if row[
        "waviness_mm"
    ] != 0.0:

        profile += deviation_waviness(
            amplitude_mm=row[
                "waviness_mm"
            ],
            waves=3,
        )


    if row[
        "twist_mm"
    ] != 0.0:

        profile += deviation_twist(
            amplitude_mm=row[
                "twist_mm"
            ]
        )


    if row[
        "local_bump_mm"
    ] != 0.0:

        profile += deviation_local_bump(
            amplitude_mm=row[
                "local_bump_mm"
            ],
            sigma=0.12,
        )


    return profile


# ============================================================
# RECONSTRUCT BATCH / PROCESS DISTURBANCE
# ============================================================

def reconstruct_batch_disturbance(
    row
):

    offset_profile = np.full_like(
        s,
        row[
            "batch_offset_bias_mm"
        ],
    )


    angular_profile = (
        np.tan(
            np.deg2rad(
                row[
                    "batch_angular_bias_deg"
                ]
            )
        )
        *
        PROFILE_LENGTH_MM
        *
        (
            s
            -
            0.5
        )
    )


    fixture_sigma = 0.18


    fixture_profile = (
        row[
            "fixture_drift_mm"
        ]
        *
        np.exp(
            -(
                (
                    s
                    -
                    0.5
                )
                ** 2
            )
            /
            (
                2.0
                *
                fixture_sigma ** 2
            )
        )
    )


    return (
        offset_profile
        +
        angular_profile
        +
        fixture_profile
    )


# ============================================================
# PRE-DECISION STATE FEATURES
# ============================================================

def extract_state_features(
    state
):

    metrics = calculate_quality_metrics(
        state
    )


    signed_mean = float(
        np.mean(
            state
        )
    )


    end_difference = float(
        state[-1]
        -
        state[0]
    )


    estimated_angle_deg = float(
        np.rad2deg(
            np.arctan(
                end_difference
                /
                PROFILE_LENGTH_MM
            )
        )
    )


    features = {

        "state_mean_gap":
            float(
                metrics[
                    "mean_gap"
                ]
            ),

        "state_max_gap":
            float(
                metrics[
                    "max_gap"
                ]
            ),

        "state_parallelism":
            float(
                metrics[
                    "parallelism_error"
                ]
            ),

        "state_rms":
            float(
                metrics[
                    "rms_deviation"
                ]
            ),

        "state_quality":
            float(
                metrics[
                    "quality_score"
                ]
            ),

        "state_signed_mean":
            signed_mean,

        "state_end_difference":
            end_difference,

        "state_estimated_angle_deg":
            estimated_angle_deg,
    }


    for (
        feature_name,
        index,
    ) in zip(
        PROFILE_SAMPLE_NAMES,
        PROFILE_SAMPLE_INDICES,
    ):

        features[
            feature_name
        ] = float(
            state[
                index
            ]
        )


    return features


# ============================================================
# RECONSTRUCT THE STRUCTURED-20 TRAJECTORY
# ============================================================
#
# This reproduces the actual accumulated state immediately
# BEFORE each saved Structured-20 correction.
#
# The correction values are used only to advance the state to
# the next component.
#
# They are NOT used as prediction features for the current
# decision.
# ============================================================

trajectory_records = []


assembly_ids = sorted(
    validation_df[
        "assembly_id"
    ]
    .unique()
)


for assembly_counter, assembly_id in enumerate(
    assembly_ids,
    start=1,
):


    assembly_validation = (
        validation_df[
            validation_df[
                "assembly_id"
            ]
            ==
            assembly_id
        ]
        .sort_values(
            "component_index"
        )
    )


    assembly_controller = (
        controller_df[
            controller_df[
                "assembly_id"
            ]
            ==
            assembly_id
        ]
        .sort_values(
            "component_index"
        )
    )


    if len(
        assembly_validation
    ) != 5:

        raise ValueError(
            f"\nAssembly {assembly_id} "
            "does not contain five validation rows."
        )


    if len(
        assembly_controller
    ) != 5:

        raise ValueError(
            f"\nAssembly {assembly_id} "
            "does not contain five controller rows."
        )


    state = create_initial_state()


    for component_index in range(
        1,
        6,
    ):


        row = (
            assembly_validation[
                assembly_validation[
                    "component_index"
                ]
                ==
                component_index
            ]
            .iloc[
                0
            ]
        )


        controller_row = (
            assembly_controller[
                assembly_controller[
                    "component_index"
                ]
                ==
                component_index
            ]
            .iloc[
                0
            ]
        )


        # ----------------------------------------------------
        # Record state BEFORE current correction.
        # ----------------------------------------------------

        state_features = extract_state_features(
            state
        )


        record = {

            "assembly_id":
                assembly_id,

            "component_index":
                component_index,

            # ------------------------------------------------
            # Incoming component information
            # ------------------------------------------------

            "offset_mm":
                float(
                    row[
                        "offset_mm"
                    ]
                ),

            "tilt_deg":
                float(
                    row[
                        "tilt_deg"
                    ]
                ),

            "bend_mm":
                float(
                    row[
                        "bend_mm"
                    ]
                ),

            "waviness_mm":
                float(
                    row[
                        "waviness_mm"
                    ]
                ),

            "twist_mm":
                float(
                    row[
                        "twist_mm"
                    ]
                ),

            "local_bump_mm":
                float(
                    row[
                        "local_bump_mm"
                    ]
                ),

            "component_profile_mean_mm":
                float(
                    row[
                        "component_profile_mean_mm"
                    ]
                ),

            "component_profile_min_mm":
                float(
                    row[
                        "component_profile_min_mm"
                    ]
                ),

            "component_profile_max_mm":
                float(
                    row[
                        "component_profile_max_mm"
                    ]
                ),

            "component_profile_rms_mm":
                float(
                    row[
                        "component_profile_rms_mm"
                    ]
                ),

            "component_parallelism_mm":
                float(
                    row[
                        "component_parallelism_mm"
                    ]
                ),

            "n_active_modes":
                int(
                    row[
                        "n_active_modes"
                    ]
                ),

            # ------------------------------------------------
            # Batch / process information
            # ------------------------------------------------

            "batch_offset_bias_mm":
                float(
                    row[
                        "batch_offset_bias_mm"
                    ]
                ),

            "batch_angular_bias_deg":
                float(
                    row[
                        "batch_angular_bias_deg"
                    ]
                ),

            "fixture_drift_mm":
                float(
                    row[
                        "fixture_drift_mm"
                    ]
                ),

            "variation_multiplier":
                float(
                    row[
                        "variation_multiplier"
                    ]
                ),

            # ------------------------------------------------
            # Current assembly state
            # ------------------------------------------------

            **state_features,
        }


        trajectory_records.append(
            record
        )


        # ----------------------------------------------------
        # Advance the physical state using the correction that
        # was actually selected in final validation.
        # ----------------------------------------------------

        component_profile = (
            reconstruct_component_profile(
                row
            )
        )


        batch_disturbance = (
            reconstruct_batch_disturbance(
                row
            )
        )


        correction = correction_profile(

            z_adj_mm=float(
                controller_row[
                    "structured_z_adj"
                ]
            ),

            theta_adj_deg=float(
                controller_row[
                    "structured_theta_adj"
                ]
            ),

            locator_offset_mm=float(
                controller_row[
                    "structured_locator_offset"
                ]
            ),
        )


        state = update_assembly_state(

            previous_state=state,

            component_deviation=(
                component_profile
            ),

            fixture_drift=(
                batch_disturbance
            ),

            correction=(
                correction
            ),
        )


predecision_df = pd.DataFrame(
    trajectory_records
)


print(
    f"\nReconstructed pre-decision states: "
    f"{len(predecision_df)}"
)


# ============================================================
# ADD TARGET LABEL
# ============================================================

label_columns = [
    "assembly_id",
    "component_index",
    "observed_correctability_class",
]


model_df = predecision_df.merge(

    label_df[
        label_columns
    ],

    on=[
        "assembly_id",
        "component_index",
    ],

    how="inner",
)


if len(
    model_df
) != len(
    predecision_df
):

    raise ValueError(
        "\nPre-decision feature rows and correctability labels "
        "do not match."
    )


model_df[
    "needs_attention"
] = (
    model_df[
        "observed_correctability_class"
    ]
    .isin(
        [
            "DIFFICULT",
            "NOT_IMPROVED",
        ]
    )
    .astype(
        int
    )
)


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

target_distribution = (
    model_df[
        "needs_attention"
    ]
    .value_counts()
    .sort_index()
)


normal_count = int(
    target_distribution.get(
        0,
        0,
    )
)


attention_count = int(
    target_distribution.get(
        1,
        0,
    )
)


attention_rate = (
    attention_count
    /
    len(
        model_df
    )
    *
    100.0
)


print(
    f"\nNORMAL decisions          : "
    f"{normal_count}"
)


print(
    f"NEEDS_ATTENTION decisions : "
    f"{attention_count}"
)


print(
    f"Attention rate            : "
    f"{attention_rate:.2f}%"
)


# ============================================================
# INFORMATION LEVELS
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
# ASSEMBLY-WISE TRAIN / TEST SPLIT
# ============================================================

groups = model_df[
    "assembly_id"
]


splitter = GroupShuffleSplit(

    n_splits=1,

    test_size=TEST_SIZE,

    random_state=RANDOM_STATE,
)


train_index, test_index = next(

    splitter.split(
        model_df,
        groups=groups,
    )
)


train_df = (
    model_df
    .iloc[
        train_index
    ]
    .copy()
)


test_df = (
    model_df
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


assembly_overlap = (
    train_assemblies
    .intersection(
        test_assemblies
    )
)


if assembly_overlap:

    raise RuntimeError(
        "\nAssembly leakage detected in train/test split."
    )


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "ASSEMBLY-WISE HOLDOUT"
)

print(
    "------------------------------------------------------------"
)


print(
    f"\nTraining assemblies : "
    f"{len(train_assemblies)}"
)


print(
    f"Test assemblies     : "
    f"{len(test_assemblies)}"
)


print(
    f"Training decisions  : "
    f"{len(train_df)}"
)


print(
    f"Test decisions      : "
    f"{len(test_df)}"
)


print(
    f"Assembly overlap    : "
    f"{len(assembly_overlap)}"
)


# ============================================================
# MODEL BUILDERS
# ============================================================

def make_logistic_model():

    return Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def make_rf_model():

    return RandomForestClassifier(

        n_estimators=N_RF_TREES,

        max_depth=None,

        min_samples_leaf=3,

        class_weight="balanced",

        random_state=RANDOM_STATE,

        n_jobs=RF_N_JOBS,
    )


# ============================================================
# METRICS
# ============================================================

def evaluate_predictions(
    model_name,
    feature_group,
    y_true,
    y_pred,
    y_probability,
):


    accuracy = accuracy_score(
        y_true,
        y_pred,
    )


    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred,
    )


    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )


    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )


    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )


    try:

        roc_auc = roc_auc_score(
            y_true,
            y_probability,
        )

    except ValueError:

        roc_auc = np.nan


    try:

        pr_auc = average_precision_score(
            y_true,
            y_probability,
        )

    except ValueError:

        pr_auc = np.nan


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

        "model":
            model_name,

        "feature_group":
            feature_group,

        "accuracy":
            accuracy,

        "balanced_accuracy":
            balanced_accuracy,

        "precision_attention":
            precision,

        "recall_attention":
            recall,

        "specificity_normal":
            specificity,

        "f1_attention":
            f1,

        "roc_auc":
            roc_auc,

        "pr_auc":
            pr_auc,

        "true_negative":
            tn,

        "false_positive":
            fp,

        "false_negative":
            fn,

        "true_positive":
            tp,
    }


# ============================================================
# TRAIN AND TEST ALL INFORMATION LEVELS
# ============================================================

result_records = []

prediction_records = []


y_train = train_df[
    "needs_attention"
].to_numpy()


y_test = test_df[
    "needs_attention"
].to_numpy()


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


    if (
        X_train.isna().any().any()
        or
        X_test.isna().any().any()
    ):

        raise ValueError(
            f"\nMissing values in feature group: "
            f"{feature_group}"
        )


    # ========================================================
    # LOGISTIC REGRESSION
    # ========================================================

    logistic = make_logistic_model()


    logistic.fit(
        X_train,
        y_train,
    )


    logistic_pred = logistic.predict(
        X_test
    )


    logistic_prob = (
        logistic.predict_proba(
            X_test
        )[
            :,
            1
        ]
    )


    result_records.append(
        evaluate_predictions(

            model_name="LOGISTIC",

            feature_group=feature_group,

            y_true=y_test,

            y_pred=logistic_pred,

            y_probability=logistic_prob,
        )
    )


    # ========================================================
    # RANDOM FOREST
    # ========================================================

    rf = make_rf_model()


    rf.fit(
        X_train,
        y_train,
    )


    rf_pred = rf.predict(
        X_test
    )


    rf_prob = (
        rf.predict_proba(
            X_test
        )[
            :,
            1
        ]
    )


    result_records.append(
        evaluate_predictions(

            model_name="RANDOM_FOREST",

            feature_group=feature_group,

            y_true=y_test,

            y_pred=rf_pred,

            y_probability=rf_prob,
        )
    )


    # --------------------------------------------------------
    # Save test predictions for the RF comparison.
    # --------------------------------------------------------

    for (
        row_index,
        actual_label,
        predicted_label,
        probability,
    ) in zip(

        test_df.index,

        y_test,

        rf_pred,

        rf_prob,
    ):

        prediction_records.append(
            {

                "row_index":
                    int(
                        row_index
                    ),

                "assembly_id":
                    int(
                        model_df.loc[
                            row_index,
                            "assembly_id",
                        ]
                    ),

                "component_index":
                    int(
                        model_df.loc[
                            row_index,
                            "component_index",
                        ]
                    ),

                "feature_group":
                    feature_group,

                "actual_needs_attention":
                    int(
                        actual_label
                    ),

                "predicted_needs_attention":
                    int(
                        predicted_label
                    ),

                "attention_probability":
                    float(
                        probability
                    ),
            }
        )


results_df = pd.DataFrame(
    result_records
)


prediction_df = pd.DataFrame(
    prediction_records
)


# ============================================================
# RESULT RANKING
# ============================================================

results_df[
    "performance_rank"
] = (
    results_df[
        "balanced_accuracy"
    ]
    .rank(
        ascending=False,
        method="min",
    )
)


results_df = (
    results_df
    .sort_values(
        [
            "balanced_accuracy",
            "f1_attention",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# RF INFORMATION-VALUE COMPARISON
# ============================================================

rf_results = (
    results_df[
        results_df[
            "model"
        ]
        ==
        "RANDOM_FOREST"
    ]
    .copy()
)


rf_lookup = (
    rf_results
    .set_index(
        "feature_group"
    )
)


stage_balanced_accuracy = float(
    rf_lookup.loc[
        "STAGE_ONLY",
        "balanced_accuracy",
    ]
)


component_balanced_accuracy = float(
    rf_lookup.loc[
        "COMPONENT_PROCESS",
        "balanced_accuracy",
    ]
)


state_balanced_accuracy = float(
    rf_lookup.loc[
        "STATE_ONLY",
        "balanced_accuracy",
    ]
)


combined_balanced_accuracy = float(
    rf_lookup.loc[
        "COMBINED",
        "balanced_accuracy",
    ]
)


combined_recall = float(
    rf_lookup.loc[
        "COMBINED",
        "recall_attention",
    ]
)


combined_specificity = float(
    rf_lookup.loc[
        "COMBINED",
        "specificity_normal",
    ]
)


combined_f1 = float(
    rf_lookup.loc[
        "COMBINED",
        "f1_attention",
    ]
)


combined_roc_auc = float(
    rf_lookup.loc[
        "COMBINED",
        "roc_auc",
    ]
)


state_gain_over_stage = (
    state_balanced_accuracy
    -
    stage_balanced_accuracy
)


combined_gain_over_stage = (
    combined_balanced_accuracy
    -
    stage_balanced_accuracy
)


combined_gain_over_component = (
    combined_balanced_accuracy
    -
    component_balanced_accuracy
)


combined_gain_over_state = (
    combined_balanced_accuracy
    -
    state_balanced_accuracy
)


information_value_df = pd.DataFrame(
    {

        "comparison": [

            "State vs Stage",

            "Combined vs Stage",

            "Combined vs Component/Process",

            "Combined vs State",
        ],

        "balanced_accuracy_gain": [

            state_gain_over_stage,

            combined_gain_over_stage,

            combined_gain_over_component,

            combined_gain_over_state,
        ],
    }
)


# ============================================================
# STAGE-WISE TEST PERFORMANCE FOR COMBINED RF
# ============================================================

combined_predictions = (
    prediction_df[
        prediction_df[
            "feature_group"
        ]
        ==
        "COMBINED"
    ]
    .copy()
)


stage_records = []


for stage in sorted(
    combined_predictions[
        "component_index"
    ]
    .unique()
):


    subset = (
        combined_predictions[
            combined_predictions[
                "component_index"
            ]
            ==
            stage
        ]
    )


    actual = subset[
        "actual_needs_attention"
    ]


    predicted = subset[
        "predicted_needs_attention"
    ]


    stage_records.append(
        {

            "component_index":
                int(
                    stage
                ),

            "n_test_decisions":
                len(
                    subset
                ),

            "actual_attention_percent":
                float(
                    actual.mean()
                    *
                    100.0
                ),

            "predicted_attention_percent":
                float(
                    predicted.mean()
                    *
                    100.0
                ),

            "accuracy":
                float(
                    accuracy_score(
                        actual,
                        predicted,
                    )
                ),

            "balanced_accuracy":
                float(
                    balanced_accuracy_score(
                        actual,
                        predicted,
                    )
                )
                if actual.nunique() > 1
                else np.nan,

            "recall_attention":
                float(
                    recall_score(
                        actual,
                        predicted,
                        zero_division=0,
                    )
                ),
        }
    )


stage_result_df = pd.DataFrame(
    stage_records
)


# ============================================================
# RANDOM FOREST FEATURE IMPORTANCE
# ============================================================
#
# Refit the combined RF so we can inspect which pre-decision
# variables carry most information.
# ============================================================

combined_rf = make_rf_model()


combined_rf.fit(

    train_df[
        COMBINED_FEATURES
    ],

    y_train,
)


feature_importance_df = pd.DataFrame(
    {

        "feature":
            COMBINED_FEATURES,

        "importance":
            combined_rf.feature_importances_,
    }
)


feature_importance_df = (
    feature_importance_df
    .sort_values(
        "importance",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# SAVE DATA
# ============================================================

predecision_df.to_csv(
    os.path.join(
        TABLE_DIR,
        "predecision_features.csv",
    ),
    index=False,
)


model_df.to_csv(
    os.path.join(
        TABLE_DIR,
        "correctability_prediction_dataset.csv",
    ),
    index=False,
)


results_df.to_csv(
    os.path.join(
        TABLE_DIR,
        "model_comparison.csv",
    ),
    index=False,
)


prediction_df.to_csv(
    os.path.join(
        TABLE_DIR,
        "rf_test_predictions.csv",
    ),
    index=False,
)


information_value_df.to_csv(
    os.path.join(
        TABLE_DIR,
        "information_value_comparison.csv",
    ),
    index=False,
)


stage_result_df.to_csv(
    os.path.join(
        TABLE_DIR,
        "combined_rf_by_stage.csv",
    ),
    index=False,
)


feature_importance_df.to_csv(
    os.path.join(
        TABLE_DIR,
        "combined_rf_feature_importance.csv",
    ),
    index=False,
)


# ============================================================
# FIGURE 1
# BALANCED ACCURACY BY INFORMATION LEVEL
# ============================================================

rf_plot_df = (
    rf_results
    .set_index(
        "feature_group"
    )
    .loc[
        [
            "STAGE_ONLY",
            "COMPONENT_PROCESS",
            "STATE_ONLY",
            "COMBINED",
        ]
    ]
    .reset_index()
)


plt.figure(
    figsize=(
        8.0,
        5.0,
    )
)


plt.bar(
    rf_plot_df[
        "feature_group"
    ],
    rf_plot_df[
        "balanced_accuracy"
    ]
    *
    100.0,
)


plt.ylabel(
    "Balanced accuracy (%)"
)


plt.xlabel(
    "Available pre-decision information"
)


plt.title(
    "Correctability Difficulty Prediction"
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
        "prediction_by_information_level.png",
    ),
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# ============================================================
# FIGURE 2
# TOP FEATURE IMPORTANCE
# ============================================================

top_features = (
    feature_importance_df
    .head(
        15
    )
    .sort_values(
        "importance",
        ascending=True,
    )
)


plt.figure(
    figsize=(
        8.0,
        6.0,
    )
)


plt.barh(
    top_features[
        "feature"
    ],
    top_features[
        "importance"
    ],
)


plt.xlabel(
    "Random Forest feature importance"
)


plt.ylabel(
    "Pre-decision feature"
)


plt.title(
    "Most Informative Correctability Features"
)


plt.grid(
    axis="x",
    alpha=0.25,
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "correctability_feature_importance.png",
    ),
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# ============================================================
# FIGURE 3
# ATTENTION RATE BY STAGE
# ============================================================

plt.figure(
    figsize=(
        7.5,
        5.0,
    )
)


plt.plot(
    stage_result_df[
        "component_index"
    ],
    stage_result_df[
        "actual_attention_percent"
    ],
    marker="o",
    label="Observed",
)


plt.plot(
    stage_result_df[
        "component_index"
    ],
    stage_result_df[
        "predicted_attention_percent"
    ],
    marker="o",
    label="Predicted",
)


plt.xlabel(
    "Sequential component stage"
)


plt.ylabel(
    "Needs-attention decisions (%)"
)


plt.title(
    "Predicted Correctability Difficulty Across Assembly Stages"
)


plt.xticks(
    stage_result_df[
        "component_index"
    ]
)


plt.legend()


plt.grid(
    alpha=0.25,
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "attention_rate_by_stage.png",
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
    "MODEL COMPARISON"
)

print(
    "============================================================\n"
)


print(
    results_df[
        [
            "model",
            "feature_group",
            "balanced_accuracy",
            "precision_attention",
            "recall_attention",
            "specificity_normal",
            "f1_attention",
            "roc_auc",
            "pr_auc",
        ]
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
    "RANDOM FOREST INFORMATION VALUE"
)

print(
    "------------------------------------------------------------"
)


print(
    f"\nStage-only balanced accuracy       : "
    f"{stage_balanced_accuracy * 100.0:.2f}%"
)


print(
    f"Component/process balanced accuracy: "
    f"{component_balanced_accuracy * 100.0:.2f}%"
)


print(
    f"State-only balanced accuracy       : "
    f"{state_balanced_accuracy * 100.0:.2f}%"
)


print(
    f"Combined balanced accuracy         : "
    f"{combined_balanced_accuracy * 100.0:.2f}%"
)


print(
    f"\nState gain over stage              : "
    f"{state_gain_over_stage * 100.0:+.2f} pp"
)


print(
    f"Combined gain over stage           : "
    f"{combined_gain_over_stage * 100.0:+.2f} pp"
)


print(
    f"Combined gain over component       : "
    f"{combined_gain_over_component * 100.0:+.2f} pp"
)


print(
    f"Combined gain over state           : "
    f"{combined_gain_over_state * 100.0:+.2f} pp"
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "COMBINED MODEL"
)

print(
    "------------------------------------------------------------"
)


print(
    f"\nBalanced accuracy     : "
    f"{combined_balanced_accuracy * 100.0:.2f}%"
)


print(
    f"Needs-attention recall: "
    f"{combined_recall * 100.0:.2f}%"
)


print(
    f"Normal specificity    : "
    f"{combined_specificity * 100.0:.2f}%"
)


print(
    f"F1 needs-attention    : "
    f"{combined_f1:.4f}"
)


print(
    f"ROC AUC               : "
    f"{combined_roc_auc:.4f}"
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "TOP PRE-DECISION FEATURES"
)

print(
    "------------------------------------------------------------\n"
)


print(
    feature_importance_df
    .head(
        15
    )
    .round(
        5
    )
    .to_string(
        index=False
    )
)


# ============================================================
# INTERPRETATION
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "PRE-DECISION CORRECTABILITY INTERPRETATION"
)

print(
    "============================================================"
)


if (
    combined_balanced_accuracy
    >=
    0.75
    and
    combined_gain_over_stage
    >=
    0.08
    and
    combined_recall
    >=
    0.70
):


    print(
        "\nRESULT A:"
    )


    print(
        "Correctability difficulty can be identified "
        "meaningfully before the correction decision."
    )


    print(
        "\nThe combined model adds substantial information "
        "beyond simply knowing the sequential component stage."
    )


    print(
        "\nThis supports a pre-decision difficulty-aware "
        "assembly recommendation layer."
    )


elif (
    combined_balanced_accuracy
    >=
    0.68
    and
    combined_gain_over_stage
    >=
    0.03
):


    print(
        "\nRESULT B:"
    )


    print(
        "Pre-decision correctability prediction shows useful "
        "signal, but the incremental value over simple stage "
        "information is moderate."
    )


    print(
        "\nTreat this as a promising extension rather than a "
        "fully validated decision layer."
    )


else:


    print(
        "\nRESULT C:"
    )


    print(
        "The pre-decision features do not yet provide enough "
        "additional information to justify a dedicated "
        "correctability predictor."
    )


    print(
        "\nDo not add another predictive layer to the final "
        "architecture without stronger evidence."
    )


print(
    "\nIMPORTANT:"
)


print(
    "Observed post-correction outcomes were used only to define "
    "the target label."
)


print(
    "All prediction inputs represent information available "
    "before the current correction decision."
)


print(
    "The train/test split was performed by complete assembly "
    "to prevent component-level leakage."
)


print(
    "\nSaved:"
)


print(
    TABLE_DIR
)


print(
    FIGURE_DIR
)


print(
    "\n"
    "============================================================"
)

print(
    "PRE-DECISION CORRECTABILITY PREDICTION COMPLETED"
)

print(
    "============================================================"
)