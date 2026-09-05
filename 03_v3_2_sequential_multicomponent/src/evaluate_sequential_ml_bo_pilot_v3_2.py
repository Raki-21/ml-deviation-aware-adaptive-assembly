import os
import time
import joblib
import numpy as np
import pandas as pd

from skopt import gp_minimize
from skopt.space import Real

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
# VERSION 3.2
# END-TO-END SEQUENTIAL ML + BAYESIAN OPTIMIZATION PILOT
# ============================================================
#
# PURPOSE
#
# Run the complete adaptive recommendation loop:
#
# Current assembly state
#        +
# Incoming component deviation
#        +
# Batch/process condition
#        ↓
# Random Forest surrogate
#        ↓
# Bayesian Optimization
#        ↓
# Recommended correction
#        ↓
# Actual assembly simulator
#        ↓
# Updated physical assembly state
#        ↓
# Next component
#
#
# PILOT SIZE
#
# 100 finished assemblies
# x 5 sequential components
# = 500 adaptive decisions
#
#
# IMPORTANT
#
# Bayesian Optimization is compared against Random Search
# using the SAME evaluation budget.
#
# This is necessary for a defensible optimizer comparison.
# ============================================================


# ============================================================
# PATHS
# ============================================================

COMPONENT_FILE = (
    "data/processed/"
    "v3_2_full_component_level_dataset.csv"
)

MODEL_FILE = (
    "models/"
    "v3_2_best_quality_surrogate.joblib"
)

FEATURE_FILE = (
    "models/"
    "v3_2_surrogate_features.txt"
)

OUTPUT_COMPONENT_FILE = (
    "data/processed/"
    "v3_2_ml_bo_pilot_component_results.csv"
)

OUTPUT_ASSEMBLY_FILE = (
    "data/processed/"
    "v3_2_ml_bo_pilot_assembly_results.csv"
)

OUTPUT_SUMMARY_FILE = (
    "results/tables/"
    "v3_2_ml_bo_pilot_summary.csv"
)


# ============================================================
# FILE CHECK
# ============================================================

required_files = [
    COMPONENT_FILE,
    MODEL_FILE,
    FEATURE_FILE,
]

missing_files = [
    file
    for file in required_files
    if not os.path.exists(file)
]

if missing_files:

    print("\nERROR - Missing required files:")

    for file in missing_files:
        print(file)

    raise SystemExit(
        "\nComplete dataset generation and ML model screening first."
    )


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

N_PILOT_ASSEMBLIES = 100

N_COMPONENTS = 5

# Equal surrogate evaluation budget for BO and Random Search.
N_OPTIMIZER_EVALUATIONS = 20

# BO starts with 8 initial random evaluations.
N_BO_INITIAL_POINTS = 8

RANDOM_SEED = 20260824

RNG = np.random.default_rng(
    RANDOM_SEED
)


# ============================================================
# CORRECTION CAPABILITY
# ============================================================

Z_LIMIT = 2.5
THETA_LIMIT = 1.2
LOCATOR_LIMIT = 1.0


SEARCH_SPACE = [
    Real(
        -Z_LIMIT,
        Z_LIMIT,
        name="z_adj",
    ),

    Real(
        -THETA_LIMIT,
        THETA_LIMIT,
        name="theta_adj",
    ),

    Real(
        -LOCATOR_LIMIT,
        LOCATOR_LIMIT,
        name="locator_offset",
    ),
]


# ============================================================
# LOAD DATA + MODEL
# ============================================================

component_df = pd.read_csv(
    COMPONENT_FILE
)

model = joblib.load(
    MODEL_FILE
)


with open(
    FEATURE_FILE,
    "r",
    encoding="utf-8",
) as file:

    FEATURE_COLUMNS = [
        line.strip()
        for line in file
        if line.strip()
    ]


# ============================================================
# RECONSTRUCT COMPONENT PROFILE
# ============================================================

def reconstruct_component_profile(row):

    profile = np.zeros_like(
        s
    )

    if row["offset_mm"] != 0.0:

        profile += deviation_offset(
            offset_mm=row["offset_mm"]
        )

    if row["tilt_deg"] != 0.0:

        profile += deviation_tilt(
            angle_deg=row["tilt_deg"]
        )

    if row["bend_mm"] != 0.0:

        profile += deviation_bend(
            amplitude_mm=row["bend_mm"]
        )

    if row["waviness_mm"] != 0.0:

        profile += deviation_waviness(
            amplitude_mm=row["waviness_mm"],
            waves=3,
        )

    if row["twist_mm"] != 0.0:

        profile += deviation_twist(
            amplitude_mm=row["twist_mm"]
        )

    if row["local_bump_mm"] != 0.0:

        profile += deviation_local_bump(
            amplitude_mm=row["local_bump_mm"],
            sigma=0.12,
        )

    return profile


# ============================================================
# RECONSTRUCT BATCH DISTURBANCE
# ============================================================

def reconstruct_batch_disturbance(row):

    offset_profile = np.full_like(
        s,
        row["batch_offset_bias_mm"],
    )

    angular_profile = (
        np.tan(
            np.deg2rad(
                row["batch_angular_bias_deg"]
            )
        )
        * PROFILE_LENGTH_MM
        * (s - 0.5)
    )

    fixture_sigma = 0.18

    fixture_profile = (
        row["fixture_drift_mm"]
        * np.exp(
            -((s - 0.5) ** 2)
            / (2.0 * fixture_sigma ** 2)
        )
    )

    return (
        offset_profile
        + angular_profile
        + fixture_profile
    )


# ============================================================
# CURRENT STATE FEATURES
# ============================================================

def calculate_state_features(state):

    metrics = calculate_quality_metrics(
        state
    )

    signed_mean = float(
        np.mean(state)
    )

    signed_end_difference = float(
        state[-1]
        - state[0]
    )

    estimated_angle_deg = float(
        np.rad2deg(
            np.arctan(
                signed_end_difference
                / PROFILE_LENGTH_MM
            )
        )
    )

    return {
        "state_mean_gap":
            metrics["mean_gap"],

        "state_max_gap":
            metrics["max_gap"],

        "state_parallelism":
            metrics["parallelism_error"],

        "state_rms":
            metrics["rms_deviation"],

        "state_quality":
            metrics["quality_score"],

        "state_signed_mean":
            signed_mean,

        "state_signed_end_difference":
            signed_end_difference,

        "state_estimated_angle_deg":
            estimated_angle_deg,
    }


# ============================================================
# BUILD SURROGATE INPUT
# ============================================================

def build_ml_features(
    row,
    state,
    z_adj,
    theta_adj,
    locator_offset,
):

    state_features = (
        calculate_state_features(
            state
        )
    )

    z_util = (
        abs(z_adj)
        / Z_LIMIT
    )

    theta_util = (
        abs(theta_adj)
        / THETA_LIMIT
    )

    locator_util = (
        abs(locator_offset)
        / LOCATOR_LIMIT
    )

    utilization = max(
        z_util,
        theta_util,
        locator_util,
    )


    feature_values = {

        # Current state
        **state_features,

        # Incoming component
        "offset_mm":
            row["offset_mm"],

        "tilt_deg":
            row["tilt_deg"],

        "bend_mm":
            row["bend_mm"],

        "waviness_mm":
            row["waviness_mm"],

        "twist_mm":
            row["twist_mm"],

        "local_bump_mm":
            row["local_bump_mm"],

        "component_profile_rms_mm":
            row["component_profile_rms_mm"],

        "component_parallelism_mm":
            row["component_parallelism_mm"],

        "n_active_modes":
            row["n_active_modes"],

        # Process / batch
        "batch_offset_bias_mm":
            row["batch_offset_bias_mm"],

        "batch_angular_bias_deg":
            row["batch_angular_bias_deg"],

        "fixture_drift_mm":
            row["fixture_drift_mm"],

        "variation_multiplier":
            row["variation_multiplier"],

        # Candidate correction
        "z_adj":
            z_adj,

        "theta_adj":
            theta_adj,

        "locator_offset":
            locator_offset,

        "correction_utilization":
            utilization,

        # Sequential position
        "component_index":
            row["component_index"],
    }


    X = pd.DataFrame(
        [
            feature_values
        ]
    )


    X = X[
        FEATURE_COLUMNS
    ]


    return X


# ============================================================
# SURROGATE QUALITY PREDICTION
# ============================================================

def surrogate_predict(
    row,
    state,
    z_adj,
    theta_adj,
    locator_offset,
):

    X = build_ml_features(
        row=row,
        state=state,
        z_adj=z_adj,
        theta_adj=theta_adj,
        locator_offset=locator_offset,
    )

    prediction = model.predict(
        X
    )[0]

    return float(
        prediction
    )


# ============================================================
# BAYESIAN OPTIMIZATION
# ============================================================

def run_bayesian_optimization(
    row,
    state,
    seed,
):

    def objective(
        correction_values
    ):

        (
            z_adj,
            theta_adj,
            locator_offset,
        ) = correction_values

        return surrogate_predict(
            row=row,
            state=state,
            z_adj=z_adj,
            theta_adj=theta_adj,
            locator_offset=locator_offset,
        )


    result = gp_minimize(
        func=objective,
        dimensions=SEARCH_SPACE,
        n_calls=N_OPTIMIZER_EVALUATIONS,
        n_initial_points=N_BO_INITIAL_POINTS,
        acq_func="EI",
        random_state=seed,
    )


    return {
        "z_adj":
            float(result.x[0]),

        "theta_adj":
            float(result.x[1]),

        "locator_offset":
            float(result.x[2]),

        "predicted_quality":
            float(result.fun),
    }


# ============================================================
# RANDOM SEARCH
# ============================================================

def run_random_search(
    row,
    state,
    rng,
):
    """
    Random Search receives exactly the same number of
    surrogate evaluations as Bayesian Optimization.
    """

    best_quality = np.inf

    best_z = None
    best_theta = None
    best_locator = None


    for _ in range(
        N_OPTIMIZER_EVALUATIONS
    ):

        z_adj = float(
            rng.uniform(
                -Z_LIMIT,
                Z_LIMIT,
            )
        )

        theta_adj = float(
            rng.uniform(
                -THETA_LIMIT,
                THETA_LIMIT,
            )
        )

        locator_offset = float(
            rng.uniform(
                -LOCATOR_LIMIT,
                LOCATOR_LIMIT,
            )
        )


        predicted_quality = (
            surrogate_predict(
                row=row,
                state=state,
                z_adj=z_adj,
                theta_adj=theta_adj,
                locator_offset=locator_offset,
            )
        )


        if (
            predicted_quality
            < best_quality
        ):

            best_quality = (
                predicted_quality
            )

            best_z = z_adj

            best_theta = theta_adj

            best_locator = (
                locator_offset
            )


    return {
        "z_adj":
            best_z,

        "theta_adj":
            best_theta,

        "locator_offset":
            best_locator,

        "predicted_quality":
            best_quality,
    }


# ============================================================
# APPLY CORRECTION TO ACTUAL SIMULATOR
# ============================================================

def apply_actual_correction(
    state,
    component_profile,
    batch_disturbance,
    recommendation,
):

    correction = correction_profile(

        z_adj_mm=(
            recommendation[
                "z_adj"
            ]
        ),

        theta_adj_deg=(
            recommendation[
                "theta_adj"
            ]
        ),

        locator_offset_mm=(
            recommendation[
                "locator_offset"
            ]
        ),
    )


    new_state = (
        update_assembly_state(
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
    )


    return new_state


# ============================================================
# SELECT BALANCED PILOT ASSEMBLIES
# ============================================================

assembly_metadata = (
    component_df[
        [
            "assembly_id",
            "batch_condition",
        ]
    ]
    .drop_duplicates()
)


batch_conditions = sorted(
    assembly_metadata[
        "batch_condition"
    ].unique()
)


selected_assemblies = []


base_per_condition = (
    N_PILOT_ASSEMBLIES
    // len(batch_conditions)
)


remaining = (
    N_PILOT_ASSEMBLIES
    % len(batch_conditions)
)


for index, condition in enumerate(
    batch_conditions
):

    condition_ids = (
        assembly_metadata[
            assembly_metadata[
                "batch_condition"
            ]
            == condition
        ][
            "assembly_id"
        ]
        .to_numpy()
    )


    n_select = (
        base_per_condition
        + (
            1
            if index < remaining
            else 0
        )
    )


    chosen = RNG.choice(
        condition_ids,
        size=n_select,
        replace=False,
    )


    selected_assemblies.extend(
        chosen.tolist()
    )


selected_assemblies = sorted(
    selected_assemblies
)


# ============================================================
# START EXPERIMENT
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "V3.2 END-TO-END ML+BO SEQUENTIAL PILOT"
)

print(
    "============================================================"
)


print(
    f"\nPilot assemblies       : "
    f"{len(selected_assemblies)}"
)

print(
    f"Components / assembly  : "
    f"{N_COMPONENTS}"
)

print(
    f"Adaptive decisions     : "
    f"{len(selected_assemblies) * N_COMPONENTS}"
)

print(
    f"BO evaluations/decision: "
    f"{N_OPTIMIZER_EVALUATIONS}"
)

print(
    f"Random evaluations     : "
    f"{N_OPTIMIZER_EVALUATIONS}"
)


experiment_start = (
    time.time()
)


# ============================================================
# STORAGE
# ============================================================

component_results = []
assembly_results = []


# ============================================================
# ASSEMBLY LOOP
# ============================================================

for assembly_counter, assembly_id in enumerate(
    selected_assemblies,
    start=1,
):

    assembly_rows = (
        component_df[
            component_df[
                "assembly_id"
            ]
            == assembly_id
        ]
        .sort_values(
            "component_index"
        )
    )


    # --------------------------------------------------------
    # Three trajectories
    # --------------------------------------------------------
    #
    # zero_state:
    #     no adaptive correction
    #
    # random_state:
    #     sequential random-search recommendation
    #
    # bo_state:
    #     sequential Bayesian recommendation
    # --------------------------------------------------------

    zero_state = (
        create_initial_state()
    )

    random_state = (
        create_initial_state()
    )

    bo_state = (
        create_initial_state()
    )


    batch_condition = (
        assembly_rows.iloc[0][
            "batch_condition"
        ]
    )


    # ========================================================
    # COMPONENT LOOP
    # ========================================================

    for _, row in (
        assembly_rows.iterrows()
    ):

        component_index = int(
            row[
                "component_index"
            ]
        )


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


        # ====================================================
        # ZERO-CORRECTION TRAJECTORY
        # ====================================================

        zero_state = (
            update_assembly_state(
                previous_state=zero_state,
                component_deviation=(
                    component_profile
                ),
                fixture_drift=(
                    batch_disturbance
                ),
            )
        )


        zero_metrics = (
            calculate_quality_metrics(
                zero_state
            )
        )


        # ====================================================
        # RANDOM SEARCH TRAJECTORY
        # ====================================================

        random_recommendation = (
            run_random_search(
                row=row,
                state=random_state,
                rng=RNG,
            )
        )


        random_state = (
            apply_actual_correction(
                state=random_state,
                component_profile=(
                    component_profile
                ),
                batch_disturbance=(
                    batch_disturbance
                ),
                recommendation=(
                    random_recommendation
                ),
            )
        )


        random_metrics = (
            calculate_quality_metrics(
                random_state
            )
        )


        # ====================================================
        # BAYESIAN OPTIMIZATION TRAJECTORY
        # ====================================================

        bo_seed = (
            RANDOM_SEED
            + int(assembly_id) * 10
            + component_index
        )


        bo_recommendation = (
            run_bayesian_optimization(
                row=row,
                state=bo_state,
                seed=bo_seed,
            )
        )


        bo_state = (
            apply_actual_correction(
                state=bo_state,
                component_profile=(
                    component_profile
                ),
                batch_disturbance=(
                    batch_disturbance
                ),
                recommendation=(
                    bo_recommendation
                ),
            )
        )


        bo_metrics = (
            calculate_quality_metrics(
                bo_state
            )
        )


        # ====================================================
        # CAPABILITY UTILIZATION
        # ====================================================

        bo_z_util = (
            abs(
                bo_recommendation[
                    "z_adj"
                ]
            )
            / Z_LIMIT
        )

        bo_theta_util = (
            abs(
                bo_recommendation[
                    "theta_adj"
                ]
            )
            / THETA_LIMIT
        )

        bo_locator_util = (
            abs(
                bo_recommendation[
                    "locator_offset"
                ]
            )
            / LOCATOR_LIMIT
        )


        bo_utilization = max(
            bo_z_util,
            bo_theta_util,
            bo_locator_util,
        )


        # ====================================================
        # SURROGATE VERIFICATION ERROR
        # ====================================================

        bo_prediction_error = (
            bo_metrics[
                "quality_score"
            ]
            -
            bo_recommendation[
                "predicted_quality"
            ]
        )


        component_results.append(
            {
                "assembly_id":
                    assembly_id,

                "batch_condition":
                    batch_condition,

                "component_index":
                    component_index,

                "severity":
                    row["severity"],

                "scenario_type":
                    row[
                        "scenario_type"
                    ],

                # Zero baseline
                "zero_quality":
                    zero_metrics[
                        "quality_score"
                    ],

                # Random Search
                "random_predicted_quality":
                    random_recommendation[
                        "predicted_quality"
                    ],

                "random_actual_quality":
                    random_metrics[
                        "quality_score"
                    ],

                # BO
                "bo_predicted_quality":
                    bo_recommendation[
                        "predicted_quality"
                    ],

                "bo_actual_quality":
                    bo_metrics[
                        "quality_score"
                    ],

                "bo_prediction_error":
                    bo_prediction_error,

                "bo_prediction_absolute_error":
                    abs(
                        bo_prediction_error
                    ),

                "bo_z_adj":
                    bo_recommendation[
                        "z_adj"
                    ],

                "bo_theta_adj":
                    bo_recommendation[
                        "theta_adj"
                    ],

                "bo_locator_offset":
                    bo_recommendation[
                        "locator_offset"
                    ],

                "bo_correction_utilization":
                    bo_utilization,

                "bo_mean_gap":
                    bo_metrics[
                        "mean_gap"
                    ],

                "bo_max_gap":
                    bo_metrics[
                        "max_gap"
                    ],

                "bo_parallelism":
                    bo_metrics[
                        "parallelism_error"
                    ],

                "bo_rms":
                    bo_metrics[
                        "rms_deviation"
                    ],
            }
        )


    # ========================================================
    # FINAL ASSEMBLY RESULTS
    # ========================================================

    zero_final = (
        calculate_quality_metrics(
            zero_state
        )
    )

    random_final = (
        calculate_quality_metrics(
            random_state
        )
    )

    bo_final = (
        calculate_quality_metrics(
            bo_state
        )
    )


    bo_improvement_vs_zero = (
        (
            zero_final[
                "quality_score"
            ]
            -
            bo_final[
                "quality_score"
            ]
        )
        /
        max(
            abs(
                zero_final[
                    "quality_score"
                ]
            ),
            1e-9,
        )
        * 100.0
    )


    bo_improvement_vs_random = (
        (
            random_final[
                "quality_score"
            ]
            -
            bo_final[
                "quality_score"
            ]
        )
        /
        max(
            abs(
                random_final[
                    "quality_score"
                ]
            ),
            1e-9,
        )
        * 100.0
    )


    assembly_results.append(
        {
            "assembly_id":
                assembly_id,

            "batch_condition":
                batch_condition,

            "zero_final_quality":
                zero_final[
                    "quality_score"
                ],

            "random_final_quality":
                random_final[
                    "quality_score"
                ],

            "bo_final_quality":
                bo_final[
                    "quality_score"
                ],

            "bo_improvement_vs_zero_percent":
                bo_improvement_vs_zero,

            "bo_improvement_vs_random_percent":
                bo_improvement_vs_random,

            "bo_final_mean_gap":
                bo_final[
                    "mean_gap"
                ],

            "bo_final_max_gap":
                bo_final[
                    "max_gap"
                ],

            "bo_final_parallelism":
                bo_final[
                    "parallelism_error"
                ],

            "bo_final_rms":
                bo_final[
                    "rms_deviation"
                ],
        }
    )


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if (
        assembly_counter % 10
        == 0
    ):

        elapsed = (
            time.time()
            -
            experiment_start
        )

        print(
            f"Completed "
            f"{assembly_counter}/"
            f"{len(selected_assemblies)} assemblies "
            f"| elapsed = "
            f"{elapsed / 60.0:.2f} min"
        )


# ============================================================
# DATAFRAMES
# ============================================================

component_result_df = pd.DataFrame(
    component_results
)

assembly_result_df = pd.DataFrame(
    assembly_results
)


# ============================================================
# SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    {
        "strategy": [
            "zero_correction",
            "random_search",
            "bayesian_optimization",
        ],

        "mean_final_quality": [
            assembly_result_df[
                "zero_final_quality"
            ].mean(),

            assembly_result_df[
                "random_final_quality"
            ].mean(),

            assembly_result_df[
                "bo_final_quality"
            ].mean(),
        ],

        "median_final_quality": [
            assembly_result_df[
                "zero_final_quality"
            ].median(),

            assembly_result_df[
                "random_final_quality"
            ].median(),

            assembly_result_df[
                "bo_final_quality"
            ].median(),
        ],

        "std_final_quality": [
            assembly_result_df[
                "zero_final_quality"
            ].std(),

            assembly_result_df[
                "random_final_quality"
            ].std(),

            assembly_result_df[
                "bo_final_quality"
            ].std(),
        ],

        "p95_final_quality": [
            assembly_result_df[
                "zero_final_quality"
            ].quantile(
                0.95
            ),

            assembly_result_df[
                "random_final_quality"
            ].quantile(
                0.95
            ),

            assembly_result_df[
                "bo_final_quality"
            ].quantile(
                0.95
            ),
        ],
    }
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "data/processed",
    exist_ok=True,
)

os.makedirs(
    "results/tables",
    exist_ok=True,
)


component_result_df.to_csv(
    OUTPUT_COMPONENT_FILE,
    index=False,
)

assembly_result_df.to_csv(
    OUTPUT_ASSEMBLY_FILE,
    index=False,
)

summary_df.to_csv(
    OUTPUT_SUMMARY_FILE,
    index=False,
)


# ============================================================
# FINAL STATISTICS
# ============================================================

total_runtime = (
    time.time()
    -
    experiment_start
)


bo_win_vs_random = (
    assembly_result_df[
        "bo_final_quality"
    ]
    <
    assembly_result_df[
        "random_final_quality"
    ]
).mean() * 100.0


bo_win_vs_zero = (
    assembly_result_df[
        "bo_final_quality"
    ]
    <
    assembly_result_df[
        "zero_final_quality"
    ]
).mean() * 100.0


mean_bo_improvement_zero = (
    assembly_result_df[
        "bo_improvement_vs_zero_percent"
    ].mean()
)


mean_bo_improvement_random = (
    assembly_result_df[
        "bo_improvement_vs_random_percent"
    ].mean()
)


mean_surrogate_error = (
    component_result_df[
        "bo_prediction_absolute_error"
    ].mean()
)


p95_surrogate_error = (
    component_result_df[
        "bo_prediction_absolute_error"
    ].quantile(
        0.95
    )
)


near_capability = (
    component_result_df[
        "bo_correction_utilization"
    ]
    >= 0.90
).mean() * 100.0


at_capability = (
    component_result_df[
        "bo_correction_utilization"
    ]
    >= 0.999
).mean() * 100.0


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "V3.2 ML+BO PILOT RESULTS"
)

print(
    "============================================================\n"
)


print(
    summary_df.round(4)
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "OPTIMIZER COMPARISON"
)

print(
    "------------------------------------------------------------"
)


print(
    f"BO better than zero baseline : "
    f"{bo_win_vs_zero:.2f}%"
)

print(
    f"BO better than Random Search : "
    f"{bo_win_vs_random:.2f}%"
)

print(
    f"Mean BO improvement vs zero  : "
    f"{mean_bo_improvement_zero:.2f}%"
)

print(
    f"Mean BO improvement vs random: "
    f"{mean_bo_improvement_random:.2f}%"
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "SURROGATE VERIFICATION"
)

print(
    "------------------------------------------------------------"
)


print(
    f"Mean BO prediction abs error : "
    f"{mean_surrogate_error:.4f}"
)

print(
    f"P95 BO prediction abs error  : "
    f"{p95_surrogate_error:.4f}"
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "CORRECTION CAPABILITY"
)

print(
    "------------------------------------------------------------"
)


print(
    f"Corrections >= 90% capability : "
    f"{near_capability:.2f}%"
)

print(
    f"Corrections at capability limit: "
    f"{at_capability:.2f}%"
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "RUNTIME"
)

print(
    "------------------------------------------------------------"
)


print(
    f"Total runtime       : "
    f"{total_runtime / 60.0:.2f} min"
)

print(
    f"Runtime / assembly  : "
    f"{total_runtime / len(selected_assemblies):.2f} sec"
)


print(
    "\nSaved:"
)

print(
    OUTPUT_COMPONENT_FILE
)

print(
    OUTPUT_ASSEMBLY_FILE
)

print(
    OUTPUT_SUMMARY_FILE
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.2 END-TO-END ML+BO PILOT COMPLETED"
)

print(
    "============================================================"
)