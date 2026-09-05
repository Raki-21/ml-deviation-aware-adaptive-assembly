import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# VERSION 3.2
# SCALAR vs PROFILE-AWARE SURROGATE COMPARISON
# ============================================================
#
# RESEARCH QUESTION
#
# Does retaining spatial information from the accumulated
# sequential geometry improve surrogate prediction?
#
#
# MODEL A
#
# Scalar state descriptors only.
#
#
# MODEL B
#
# Same scalar descriptors
# +
# 11 signed sampled points from the accumulated assembly profile.
#
#
# CONTROLLED COMPARISON
#
# Both models use:
#
#   - exactly the same 420,000 rows
#   - exactly the same assembly-wise train/test split
#   - exactly the same Random Forest configuration
#   - exactly the same target
#
#
# Therefore the principal experimental difference is:
#
#       geometric state representation
#
#
# IMPORTANT
#
# Groups are split by assembly_id.
#
# No rows belonging to a test assembly are allowed in training.
# ============================================================


# ============================================================
# PATHS
# ============================================================

DATA_FILE = (
    "data/processed/"
    "v3_2_profile_aware_ml_training_dataset.csv"
)

SCALAR_MODEL_FILE = (
    "models/"
    "v3_2_scalar_ab_test_surrogate.joblib"
)

PROFILE_MODEL_FILE = (
    "models/"
    "v3_2_profile_aware_quality_surrogate.joblib"
)

PROFILE_FEATURE_FILE = (
    "models/"
    "v3_2_profile_aware_surrogate_features.txt"
)

COMPARISON_FILE = (
    "results/tables/"
    "v3_2_scalar_vs_profile_surrogate_comparison.csv"
)

STAGE_FILE = (
    "results/tables/"
    "v3_2_scalar_vs_profile_error_by_stage.csv"
)

TRAJECTORY_FILE = (
    "results/tables/"
    "v3_2_scalar_vs_profile_error_by_trajectory.csv"
)

PREDICTION_FILE = (
    "data/processed/"
    "v3_2_scalar_vs_profile_test_predictions.csv"
)


# ============================================================
# FILE CHECK
# ============================================================

if not os.path.exists(
    DATA_FILE
):

    raise FileNotFoundError(
        "\nProfile-aware V3.2 training dataset not found."
    )


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

TEST_SIZE = 0.20

N_TREES = 180

N_JOBS = 2


TARGET_COLUMN = (
    "target_quality_score"
)


# ============================================================
# PROFILE FEATURES
# ============================================================

PROFILE_FEATURES = [
    "state_profile_p00",
    "state_profile_p10",
    "state_profile_p20",
    "state_profile_p30",
    "state_profile_p40",
    "state_profile_p50",
    "state_profile_p60",
    "state_profile_p70",
    "state_profile_p80",
    "state_profile_p90",
    "state_profile_p100",
]


# ============================================================
# SCALAR FEATURE SET
# ============================================================
#
# This preserves the state/component/process/correction feature
# formulation used before profile information was added.
# ============================================================

SCALAR_FEATURES = [

    # --------------------------------------------------------
    # Current assembly state
    # --------------------------------------------------------

    "state_mean_gap",
    "state_max_gap",
    "state_parallelism",
    "state_rms",
    "state_quality",
    "state_signed_mean",
    "state_signed_end_difference",
    "state_estimated_angle_deg",


    # --------------------------------------------------------
    # Incoming component deviation
    # --------------------------------------------------------

    "offset_mm",
    "tilt_deg",
    "bend_mm",
    "waviness_mm",
    "twist_mm",
    "local_bump_mm",

    "component_profile_rms_mm",
    "component_parallelism_mm",
    "n_active_modes",


    # --------------------------------------------------------
    # Batch / process state
    # --------------------------------------------------------

    "batch_offset_bias_mm",
    "batch_angular_bias_deg",
    "fixture_drift_mm",
    "variation_multiplier",


    # --------------------------------------------------------
    # Candidate correction
    # --------------------------------------------------------

    "z_adj",
    "theta_adj",
    "locator_offset",
    "correction_utilization",


    # --------------------------------------------------------
    # Sequential position
    # --------------------------------------------------------

    "component_index",
]


# ============================================================
# PROFILE-AWARE FEATURE SET
# ============================================================

PROFILE_AWARE_FEATURES = (
    SCALAR_FEATURES
    +
    PROFILE_FEATURES
)


# ============================================================
# LOAD DATA
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "V3.2 SCALAR vs PROFILE-AWARE SURROGATE TEST"
)

print(
    "============================================================"
)


print(
    "\nLoading dataset..."
)


df = pd.read_csv(
    DATA_FILE
)


print(
    f"Rows       : {len(df)}"
)

print(
    f"Assemblies : {df['assembly_id'].nunique()}"
)

print(
    f"Scalar features       : {len(SCALAR_FEATURES)}"
)

print(
    f"Profile-aware features: {len(PROFILE_AWARE_FEATURES)}"
)


# ============================================================
# COLUMN VALIDATION
# ============================================================

required_columns = (
    SCALAR_FEATURES
    +
    PROFILE_FEATURES
    +
    [
        TARGET_COLUMN,
        "assembly_id",
        "component_index",
        "trajectory_type",
        "candidate_source",
    ]
)


missing_columns = [

    column

    for column
    in required_columns

    if column
    not in df.columns
]


if missing_columns:

    print(
        "\nERROR - Missing columns:"
    )

    for column in missing_columns:

        print(
            column
        )

    raise SystemExit(
        "\nDataset structure does not match expected V3.2 format."
    )


if df[
    required_columns
].isna().sum().sum() != 0:

    raise ValueError(
        "\nERROR - Missing values detected in required columns."
    )


# ============================================================
# ASSEMBLY-WISE SPLIT
# ============================================================

groups = df[
    "assembly_id"
]


splitter = GroupShuffleSplit(

    n_splits=1,

    test_size=TEST_SIZE,

    random_state=RANDOM_SEED,
)


train_idx, test_idx = next(

    splitter.split(
        df,
        groups=groups,
    )
)


train_df = df.iloc[
    train_idx
].copy()


test_df = df.iloc[
    test_idx
].copy()


train_assemblies = set(
    train_df[
        "assembly_id"
    ].unique()
)


test_assemblies = set(
    test_df[
        "assembly_id"
    ].unique()
)


assembly_overlap = (
    train_assemblies
    .intersection(
        test_assemblies
    )
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "ASSEMBLY HOLDOUT"
)

print(
    "------------------------------------------------------------"
)


print(
    f"Training rows       : {len(train_df)}"
)

print(
    f"Test rows           : {len(test_df)}"
)

print(
    f"Training assemblies : {len(train_assemblies)}"
)

print(
    f"Test assemblies     : {len(test_assemblies)}"
)

print(
    f"Assembly overlap    : {len(assembly_overlap)}"
)


if len(
    assembly_overlap
) != 0:

    raise ValueError(
        "\nERROR - Assembly leakage detected."
    )


print(
    "PASS - Zero assembly overlap."
)


# ============================================================
# TARGET
# ============================================================

y_train = train_df[
    TARGET_COLUMN
].to_numpy()


y_test = test_df[
    TARGET_COLUMN
].to_numpy()


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
):

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )


    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )


    r2 = r2_score(
        y_true,
        y_pred,
    )


    absolute_error = np.abs(
        y_true
        -
        y_pred
    )


    p95_ae = np.quantile(
        absolute_error,
        0.95,
    )


    p99_ae = np.quantile(
        absolute_error,
        0.99,
    )


    return {

        "MAE":
            float(
                mae
            ),

        "RMSE":
            float(
                rmse
            ),

        "R2":
            float(
                r2
            ),

        "P95_AE":
            float(
                p95_ae
            ),

        "P99_AE":
            float(
                p99_ae
            ),
    }


# ============================================================
# TRAIN MODEL A - SCALAR
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "MODEL A - SCALAR STATE REPRESENTATION"
)

print(
    "============================================================"
)


X_scalar_train = train_df[
    SCALAR_FEATURES
]


X_scalar_test = test_df[
    SCALAR_FEATURES
]


scalar_model = RandomForestRegressor(

    n_estimators=N_TREES,

    max_depth=None,

    min_samples_leaf=1,

    max_features=1.0,

    random_state=RANDOM_SEED,

    n_jobs=N_JOBS,
)


scalar_start = time.time()


scalar_model.fit(
    X_scalar_train,
    y_train,
)


scalar_runtime = (
    time.time()
    -
    scalar_start
)


# Save immediately.

os.makedirs(
    "models",
    exist_ok=True,
)


joblib.dump(
    scalar_model,
    SCALAR_MODEL_FILE,
)


scalar_predictions = scalar_model.predict(
    X_scalar_test
)


scalar_metrics = calculate_metrics(
    y_test,
    scalar_predictions,
)


print(
    f"\nMAE    : {scalar_metrics['MAE']:.6f}"
)

print(
    f"RMSE   : {scalar_metrics['RMSE']:.6f}"
)

print(
    f"R²     : {scalar_metrics['R2']:.6f}"
)

print(
    f"P95 AE : {scalar_metrics['P95_AE']:.6f}"
)

print(
    f"P99 AE : {scalar_metrics['P99_AE']:.6f}"
)

print(
    f"Time   : {scalar_runtime:.2f} sec"
)


print(
    "\nScalar model saved immediately:"
)

print(
    SCALAR_MODEL_FILE
)


# ============================================================
# TRAIN MODEL B - PROFILE-AWARE
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "MODEL B - PROFILE-AWARE STATE REPRESENTATION"
)

print(
    "============================================================"
)


X_profile_train = train_df[
    PROFILE_AWARE_FEATURES
]


X_profile_test = test_df[
    PROFILE_AWARE_FEATURES
]


profile_model = RandomForestRegressor(

    n_estimators=N_TREES,

    max_depth=None,

    min_samples_leaf=1,

    max_features=1.0,

    random_state=RANDOM_SEED,

    n_jobs=N_JOBS,
)


profile_start = time.time()


profile_model.fit(
    X_profile_train,
    y_train,
)


profile_runtime = (
    time.time()
    -
    profile_start
)


# Save immediately after fitting.

joblib.dump(
    profile_model,
    PROFILE_MODEL_FILE,
)


with open(
    PROFILE_FEATURE_FILE,
    "w",
    encoding="utf-8",
) as file:

    for feature in PROFILE_AWARE_FEATURES:

        file.write(
            feature
            +
            "\n"
        )


profile_predictions = profile_model.predict(
    X_profile_test
)


profile_metrics = calculate_metrics(
    y_test,
    profile_predictions,
)


print(
    f"\nMAE    : {profile_metrics['MAE']:.6f}"
)

print(
    f"RMSE   : {profile_metrics['RMSE']:.6f}"
)

print(
    f"R²     : {profile_metrics['R2']:.6f}"
)

print(
    f"P95 AE : {profile_metrics['P95_AE']:.6f}"
)

print(
    f"P99 AE : {profile_metrics['P99_AE']:.6f}"
)

print(
    f"Time   : {profile_runtime:.2f} sec"
)


print(
    "\nProfile-aware model saved immediately:"
)

print(
    PROFILE_MODEL_FILE
)


print(
    "\nFeature list saved:"
)

print(
    PROFILE_FEATURE_FILE
)


# ============================================================
# OVERALL COMPARISON
# ============================================================

comparison_df = pd.DataFrame(
    [
        {
            "model":
                "scalar_rf",

            **scalar_metrics,

            "training_time_seconds":
                scalar_runtime,

            "n_features":
                len(
                    SCALAR_FEATURES
                ),
        },

        {
            "model":
                "profile_aware_rf",

            **profile_metrics,

            "training_time_seconds":
                profile_runtime,

            "n_features":
                len(
                    PROFILE_AWARE_FEATURES
                ),
        },
    ]
)


# ============================================================
# ERROR IMPROVEMENT
# ============================================================

mae_improvement_percent = (

    (
        scalar_metrics[
            "MAE"
        ]
        -
        profile_metrics[
            "MAE"
        ]
    )

    /

    max(
        scalar_metrics[
            "MAE"
        ],
        1e-12,
    )

    *

    100.0
)


rmse_improvement_percent = (

    (
        scalar_metrics[
            "RMSE"
        ]
        -
        profile_metrics[
            "RMSE"
        ]
    )

    /

    max(
        scalar_metrics[
            "RMSE"
        ],
        1e-12,
    )

    *

    100.0
)


p95_improvement_percent = (

    (
        scalar_metrics[
            "P95_AE"
        ]
        -
        profile_metrics[
            "P95_AE"
        ]
    )

    /

    max(
        scalar_metrics[
            "P95_AE"
        ],
        1e-12,
    )

    *

    100.0
)


# ============================================================
# PREDICTION TABLE
# ============================================================

prediction_df = test_df[
    [
        "assembly_id",
        "component_index",
        "trajectory_type",
        "candidate_source",
        TARGET_COLUMN,
    ]
].copy()


prediction_df[
    "scalar_prediction"
] = scalar_predictions


prediction_df[
    "profile_prediction"
] = profile_predictions


prediction_df[
    "scalar_absolute_error"
] = np.abs(
    prediction_df[
        TARGET_COLUMN
    ]
    -
    prediction_df[
        "scalar_prediction"
    ]
)


prediction_df[
    "profile_absolute_error"
] = np.abs(
    prediction_df[
        TARGET_COLUMN
    ]
    -
    prediction_df[
        "profile_prediction"
    ]
)


# ============================================================
# STAGE-WISE COMPARISON
# ============================================================

stage_rows = []


for component_index in range(
    1,
    6,
):


    subset = prediction_df[
        prediction_df[
            "component_index"
        ]
        ==
        component_index
    ]


    scalar_stage_mae = (
        subset[
            "scalar_absolute_error"
        ]
        .mean()
    )


    profile_stage_mae = (
        subset[
            "profile_absolute_error"
        ]
        .mean()
    )


    improvement = (

        (
            scalar_stage_mae
            -
            profile_stage_mae
        )

        /

        max(
            scalar_stage_mae,
            1e-12,
        )

        *

        100.0
    )


    stage_rows.append(
        {

            "component_index":
                component_index,

            "scalar_MAE":
                scalar_stage_mae,

            "profile_MAE":
                profile_stage_mae,

            "profile_MAE_improvement_percent":
                improvement,
        }
    )


stage_df = pd.DataFrame(
    stage_rows
)


# ============================================================
# TRAJECTORY-WISE COMPARISON
# ============================================================

trajectory_rows = []


for trajectory in sorted(
    prediction_df[
        "trajectory_type"
    ].unique()
):


    subset = prediction_df[
        prediction_df[
            "trajectory_type"
        ]
        ==
        trajectory
    ]


    scalar_mae = (
        subset[
            "scalar_absolute_error"
        ]
        .mean()
    )


    profile_mae = (
        subset[
            "profile_absolute_error"
        ]
        .mean()
    )


    improvement = (

        (
            scalar_mae
            -
            profile_mae
        )

        /

        max(
            scalar_mae,
            1e-12,
        )

        *

        100.0
    )


    trajectory_rows.append(
        {

            "trajectory_type":
                trajectory,

            "scalar_MAE":
                scalar_mae,

            "profile_MAE":
                profile_mae,

            "profile_MAE_improvement_percent":
                improvement,
        }
    )


trajectory_df = pd.DataFrame(
    trajectory_rows
)


# ============================================================
# COMPONENT-5 KEY METRIC
# ============================================================

component5_row = stage_df[
    stage_df[
        "component_index"
    ]
    ==
    5
].iloc[
    0
]


scalar_component5_mae = float(
    component5_row[
        "scalar_MAE"
    ]
)


profile_component5_mae = float(
    component5_row[
        "profile_MAE"
    ]
)


component5_improvement = float(
    component5_row[
        "profile_MAE_improvement_percent"
    ]
)


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    "results/tables",
    exist_ok=True,
)


os.makedirs(
    "data/processed",
    exist_ok=True,
)


comparison_df.to_csv(
    COMPARISON_FILE,
    index=False,
)


stage_df.to_csv(
    STAGE_FILE,
    index=False,
)


trajectory_df.to_csv(
    TRAJECTORY_FILE,
    index=False,
)


prediction_df.to_csv(
    PREDICTION_FILE,
    index=False,
)


# ============================================================
# PRINT COMPARISON
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "SCALAR vs PROFILE-AWARE COMPARISON"
)

print(
    "============================================================"
)


print(
    "\nOVERALL RESULTS\n"
)


print(
    comparison_df.round(
        6
    ).to_string(
        index=False
    )
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "PROFILE REPRESENTATION IMPROVEMENT"
)

print(
    "------------------------------------------------------------"
)


print(
    f"\nOverall MAE improvement  : "
    f"{mae_improvement_percent:.2f}%"
)


print(
    f"Overall RMSE improvement : "
    f"{rmse_improvement_percent:.2f}%"
)


print(
    f"P95 error improvement    : "
    f"{p95_improvement_percent:.2f}%"
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "STAGE-WISE MAE"
)

print(
    "------------------------------------------------------------\n"
)


print(
    stage_df.round(
        6
    ).to_string(
        index=False
    )
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "TRAJECTORY-WISE MAE"
)

print(
    "------------------------------------------------------------\n"
)


print(
    trajectory_df.round(
        6
    ).to_string(
        index=False
    )
)


# ============================================================
# HYPOTHESIS EVALUATION
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "PROFILE-REPRESENTATION HYPOTHESIS EVALUATION"
)

print(
    "============================================================"
)


print(
    f"\nScalar Component-5 MAE  : "
    f"{scalar_component5_mae:.6f}"
)


print(
    f"Profile Component-5 MAE : "
    f"{profile_component5_mae:.6f}"
)


print(
    f"Component-5 improvement : "
    f"{component5_improvement:.2f}%"
)


if (
    mae_improvement_percent
    >
    5.0
):

    print(
        "\nPASS - Profile-aware geometry produced a "
        "meaningful reduction in overall surrogate error."
    )

else:

    print(
        "\nWARNING - Profile-aware geometry did not produce "
        "a substantial overall MAE improvement."
    )


if (
    component5_improvement
    >
    5.0
):

    print(
        "PASS - Profile-aware representation improved "
        "late-stage sequential prediction."
    )

else:

    print(
        "WARNING - Profile-aware representation did not "
        "substantially improve Component-5 prediction."
    )


if (
    mae_improvement_percent
    >
    5.0
    and
    component5_improvement
    >
    5.0
):

    print(
        "\nNEXT STEP:"
    )

    print(
        "Integrate the profile-aware RF into the sequential "
        "Bayesian Optimization controller and verify it on "
        "the same saved pilot assemblies."
    )


else:

    print(
        "\nNEXT STEP:"
    )

    print(
        "Do NOT immediately integrate this model into BO."
    )

    print(
        "The remaining controller gap requires a deeper "
        "representation or learning-target diagnosis."
    )


# ============================================================
# SAVE PATHS
# ============================================================

print(
    "\nSaved:"
)

print(
    SCALAR_MODEL_FILE
)

print(
    PROFILE_MODEL_FILE
)

print(
    PROFILE_FEATURE_FILE
)

print(
    COMPARISON_FILE
)

print(
    STAGE_FILE
)

print(
    TRAJECTORY_FILE
)

print(
    PREDICTION_FILE
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.2 SCALAR vs PROFILE-AWARE TEST COMPLETED"
)

print(
    "============================================================"
)