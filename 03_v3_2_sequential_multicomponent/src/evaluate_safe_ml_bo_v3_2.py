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
# CONTROLLED FIX-1 + FIX-2 SEQUENTIAL VERIFICATION
# ============================================================
#
# PURPOSE
#
# Re-evaluate the EXACT SAME assemblies used in the original
# 100-assembly ML+BO pilot, but replace the original adaptive
# controller with:
#
#   FIX 1:
#       closed-loop-trained Random Forest surrogate
#
#   FIX 2:
#       P4 capability-aware Bayesian Optimization
#
#
# IMPORTANT
#
# We DO NOT rerun:
#
#   - zero correction
#   - Random Search
#   - original BO
#
# Their results already exist from the previous pilot.
#
# This script runs only the NEW controller and joins the new
# results to the original saved assembly results.
#
# Therefore the before-vs-after comparison uses the exact same
# physical assembly cases.
# ============================================================


# ============================================================
# PATHS
# ============================================================

COMPONENT_FILE = (
    "data/processed/"
    "v3_2_full_component_level_dataset.csv"
)

OLD_COMPONENT_RESULTS_FILE = (
    "data/processed/"
    "v3_2_ml_bo_pilot_component_results.csv"
)

OLD_ASSEMBLY_RESULTS_FILE = (
    "data/processed/"
    "v3_2_ml_bo_pilot_assembly_results.csv"
)

MODEL_FILE = (
    "models/"
    "v3_2_closed_loop_quality_surrogate.joblib"
)

FEATURE_FILE = (
    "models/"
    "v3_2_closed_loop_surrogate_features.txt"
)


NEW_COMPONENT_OUTPUT = (
    "data/processed/"
    "v3_2_safe_ml_bo_component_results.csv"
)

NEW_ASSEMBLY_OUTPUT = (
    "data/processed/"
    "v3_2_safe_ml_bo_assembly_results.csv"
)

COMPARISON_OUTPUT = (
    "results/tables/"
    "v3_2_old_vs_safe_ml_bo_comparison.csv"
)

SUMMARY_OUTPUT = (
    "results/tables/"
    "v3_2_safe_ml_bo_summary.csv"
)


# ============================================================
# FILE CHECK
# ============================================================

required_files = [
    COMPONENT_FILE,
    OLD_COMPONENT_RESULTS_FILE,
    OLD_ASSEMBLY_RESULTS_FILE,
    MODEL_FILE,
    FEATURE_FILE,
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
        "\nRequired original pilot or Fix 1 files are missing."
    )


# ============================================================
# SETTINGS
# ============================================================

N_COMPONENTS = 5

# Same BO surrogate evaluation budget as original pilot.
N_OPTIMIZER_EVALUATIONS = 20

N_BO_INITIAL_POINTS = 8

RANDOM_SEED = 20260824


# ============================================================
# CAPABILITY LIMITS
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
# P4 BARRIER PENALTY
# ============================================================
#
# Selected provisionally from the capability calibration.
#
# P4:
#
# effort_weight  = 0.04
# barrier_start  = 0.80
# barrier_weight = 1.25
#
#
# IMPORTANT:
#
# These remain prototype engineering settings.
#
# The purpose of this experiment is to determine whether this
# surrogate-screened configuration actually improves the
# physical sequential simulator results.
# ============================================================

EFFORT_WEIGHT = 0.04

BARRIER_START = 0.80

BARRIER_WEIGHT = 1.25


# ============================================================
# LOAD FILES
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "V3.2 FIX-1 + FIX-2 CONTROLLED SEQUENTIAL VERIFICATION"
)

print(
    "============================================================"
)


print(
    "\nLoading component dataset..."
)

component_df = pd.read_csv(
    COMPONENT_FILE
)


print(
    "Loading original pilot results..."
)

old_component_df = pd.read_csv(
    OLD_COMPONENT_RESULTS_FILE
)

old_assembly_df = pd.read_csv(
    OLD_ASSEMBLY_RESULTS_FILE
)


print(
    "Loading closed-loop RF..."
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
# EXACT SAME ASSEMBLIES AS ORIGINAL PILOT
# ============================================================

selected_assemblies = sorted(
    old_assembly_df[
        "assembly_id"
    ]
    .unique()
    .tolist()
)


print(
    f"\nOriginal pilot assemblies : "
    f"{len(selected_assemblies)}"
)


if len(selected_assemblies) != 100:

    print(
        "\nWARNING:"
    )

    print(
        "Original pilot does not contain exactly 100 assemblies."
    )

    print(
        "The script will continue using all saved original "
        "pilot assembly IDs."
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


    if row["offset_mm"] != 0.0:

        profile += deviation_offset(
            offset_mm=row[
                "offset_mm"
            ]
        )


    if row["tilt_deg"] != 0.0:

        profile += deviation_tilt(
            angle_deg=row[
                "tilt_deg"
            ]
        )


    if row["bend_mm"] != 0.0:

        profile += deviation_bend(
            amplitude_mm=row[
                "bend_mm"
            ]
        )


    if row["waviness_mm"] != 0.0:

        profile += deviation_waviness(
            amplitude_mm=row[
                "waviness_mm"
            ],
            waves=3,
        )


    if row["twist_mm"] != 0.0:

        profile += deviation_twist(
            amplitude_mm=row[
                "twist_mm"
            ]
        )


    if row["local_bump_mm"] != 0.0:

        profile += deviation_local_bump(
            amplitude_mm=row[
                "local_bump_mm"
            ],
            sigma=0.12,
        )


    return profile


# ============================================================
# RECONSTRUCT BATCH DISTURBANCE
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
        (s - 0.5)
    )


    fixture_sigma = 0.18


    fixture_profile = (
        row[
            "fixture_drift_mm"
        ]
        *
        np.exp(
            -(
                (s - 0.5) ** 2
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
# CURRENT STATE FEATURES
# ============================================================

def calculate_state_features(
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


    signed_end_difference = float(
        state[-1]
        -
        state[0]
    )


    estimated_angle_deg = float(
        np.rad2deg(
            np.arctan(
                signed_end_difference
                /
                PROFILE_LENGTH_MM
            )
        )
    )


    return {

        "state_mean_gap":
            metrics[
                "mean_gap"
            ],

        "state_max_gap":
            metrics[
                "max_gap"
            ],

        "state_parallelism":
            metrics[
                "parallelism_error"
            ],

        "state_rms":
            metrics[
                "rms_deviation"
            ],

        "state_quality":
            metrics[
                "quality_score"
            ],

        "state_signed_mean":
            signed_mean,

        "state_signed_end_difference":
            signed_end_difference,

        "state_estimated_angle_deg":
            estimated_angle_deg,
    }


# ============================================================
# CORRECTION NORMALIZATION
# ============================================================

def normalized_corrections(
    z_adj,
    theta_adj,
    locator_offset,
):

    z_norm = (
        abs(
            z_adj
        )
        /
        Z_LIMIT
    )


    theta_norm = (
        abs(
            theta_adj
        )
        /
        THETA_LIMIT
    )


    locator_norm = (
        abs(
            locator_offset
        )
        /
        LOCATOR_LIMIT
    )


    return (
        float(
            z_norm
        ),
        float(
            theta_norm
        ),
        float(
            locator_norm
        ),
    )


# ============================================================
# CAPABILITY UTILIZATION
# ============================================================

def calculate_utilization(
    z_adj,
    theta_adj,
    locator_offset,
):

    (
        z_norm,
        theta_norm,
        locator_norm,
    ) = normalized_corrections(
        z_adj,
        theta_adj,
        locator_offset,
    )


    return float(
        max(
            z_norm,
            theta_norm,
            locator_norm,
        )
    )


# ============================================================
# CORRECTION EFFORT
# ============================================================

def calculate_effort(
    z_adj,
    theta_adj,
    locator_offset,
):

    (
        z_norm,
        theta_norm,
        locator_norm,
    ) = normalized_corrections(
        z_adj,
        theta_adj,
        locator_offset,
    )


    effort = np.sqrt(
        (
            z_norm ** 2
            +
            theta_norm ** 2
            +
            locator_norm ** 2
        )
        /
        3.0
    )


    return float(
        effort
    )


# ============================================================
# P4 CAPABILITY PENALTY
# ============================================================

def capability_penalty(
    z_adj,
    theta_adj,
    locator_offset,
):

    utilization = calculate_utilization(
        z_adj,
        theta_adj,
        locator_offset,
    )


    effort = calculate_effort(
        z_adj,
        theta_adj,
        locator_offset,
    )


    effort_penalty = (
        EFFORT_WEIGHT
        *
        effort ** 2
    )


    if utilization <= BARRIER_START:

        boundary_penalty = 0.0

    else:

        normalized_excess = (
            utilization
            -
            BARRIER_START
        ) / (
            1.0
            -
            BARRIER_START
        )


        boundary_penalty = (
            BARRIER_WEIGHT
            *
            normalized_excess ** 3
        )


    return float(
        effort_penalty
        +
        boundary_penalty
    )


# ============================================================
# BUILD CLOSED-LOOP RF FEATURES
# ============================================================

def build_ml_features(
    row,
    state,
    z_adj,
    theta_adj,
    locator_offset,
):

    state_features = calculate_state_features(
        state
    )


    utilization = calculate_utilization(
        z_adj,
        theta_adj,
        locator_offset,
    )


    feature_values = {

        # ----------------------------------------------------
        # Current sequential assembly state
        # ----------------------------------------------------

        **state_features,


        # ----------------------------------------------------
        # Incoming component variation
        # ----------------------------------------------------

        "offset_mm":
            row[
                "offset_mm"
            ],

        "tilt_deg":
            row[
                "tilt_deg"
            ],

        "bend_mm":
            row[
                "bend_mm"
            ],

        "waviness_mm":
            row[
                "waviness_mm"
            ],

        "twist_mm":
            row[
                "twist_mm"
            ],

        "local_bump_mm":
            row[
                "local_bump_mm"
            ],

        "component_profile_rms_mm":
            row[
                "component_profile_rms_mm"
            ],

        "component_parallelism_mm":
            row[
                "component_parallelism_mm"
            ],

        "n_active_modes":
            row[
                "n_active_modes"
            ],


        # ----------------------------------------------------
        # Batch / process state
        # ----------------------------------------------------

        "batch_offset_bias_mm":
            row[
                "batch_offset_bias_mm"
            ],

        "batch_angular_bias_deg":
            row[
                "batch_angular_bias_deg"
            ],

        "fixture_drift_mm":
            row[
                "fixture_drift_mm"
            ],

        "variation_multiplier":
            row[
                "variation_multiplier"
            ],


        # ----------------------------------------------------
        # Candidate correction
        # ----------------------------------------------------

        "z_adj":
            z_adj,

        "theta_adj":
            theta_adj,

        "locator_offset":
            locator_offset,

        "correction_utilization":
            utilization,


        # ----------------------------------------------------
        # Sequential stage
        # ----------------------------------------------------

        "component_index":
            row[
                "component_index"
            ],
    }


    # --------------------------------------------------------
    # Validate feature availability
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature
        not in feature_values
    ]


    if missing_features:

        raise ValueError(
            "\nClosed-loop model requests features that "
            "are not available in the sequential controller:\n"
            +
            "\n".join(
                missing_features
            )
        )


    X = pd.DataFrame(
        [
            feature_values
        ]
    )


    return X[
        FEATURE_COLUMNS
    ]


# ============================================================
# SURROGATE PREDICTION
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
# SAFE P4 BAYESIAN OPTIMIZATION
# ============================================================

def run_safe_bayesian_optimization(
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


        predicted_quality = surrogate_predict(
            row=row,
            state=state,
            z_adj=z_adj,
            theta_adj=theta_adj,
            locator_offset=locator_offset,
        )


        penalty = capability_penalty(
            z_adj,
            theta_adj,
            locator_offset,
        )


        return (
            predicted_quality
            +
            penalty
        )


    result = gp_minimize(

        func=objective,

        dimensions=SEARCH_SPACE,

        n_calls=N_OPTIMIZER_EVALUATIONS,

        n_initial_points=N_BO_INITIAL_POINTS,

        acq_func="EI",

        random_state=seed,
    )


    z_adj = float(
        result.x[0]
    )

    theta_adj = float(
        result.x[1]
    )

    locator_offset = float(
        result.x[2]
    )


    predicted_quality = surrogate_predict(
        row=row,
        state=state,
        z_adj=z_adj,
        theta_adj=theta_adj,
        locator_offset=locator_offset,
    )


    utilization = calculate_utilization(
        z_adj,
        theta_adj,
        locator_offset,
    )


    effort = calculate_effort(
        z_adj,
        theta_adj,
        locator_offset,
    )


    penalty = capability_penalty(
        z_adj,
        theta_adj,
        locator_offset,
    )


    return {

        "z_adj":
            z_adj,

        "theta_adj":
            theta_adj,

        "locator_offset":
            locator_offset,

        "predicted_quality":
            predicted_quality,

        "utilization":
            utilization,

        "effort":
            effort,

        "penalty":
            penalty,

        "objective":
            (
                predicted_quality
                +
                penalty
            ),
    }


# ============================================================
# APPLY RECOMMENDATION TO ACTUAL SIMULATOR
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


    new_state = update_assembly_state(

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


    return new_state


# ============================================================
# EXPERIMENT
# ============================================================

experiment_start = time.time()


component_results = []

assembly_results = []


print(
    f"\nComponents per assembly : "
    f"{N_COMPONENTS}"
)

print(
    f"Adaptive decisions      : "
    f"{len(selected_assemblies) * N_COMPONENTS}"
)

print(
    f"BO evaluations/decision : "
    f"{N_OPTIMIZER_EVALUATIONS}"
)

print(
    "\nRunning new closed-loop RF + P4 BO..."
)


# ============================================================
# ASSEMBLY LOOP
# ============================================================

for (
    assembly_counter,
    assembly_id,
) in enumerate(
    selected_assemblies,
    start=1,
):


    assembly_rows = (
        component_df[
            component_df[
                "assembly_id"
            ]
            ==
            assembly_id
        ]
        .sort_values(
            "component_index"
        )
    )


    if len(assembly_rows) != N_COMPONENTS:

        raise ValueError(
            f"\nAssembly {assembly_id} has "
            f"{len(assembly_rows)} components, "
            f"expected {N_COMPONENTS}."
        )


    safe_state = create_initial_state()


    batch_condition = (
        assembly_rows.iloc[
            0
        ][
            "batch_condition"
        ]
    )


    # ========================================================
    # COMPONENT LOOP
    # ========================================================

    for _, row in assembly_rows.iterrows():


        component_index = int(
            row[
                "component_index"
            ]
        )


        component_profile = reconstruct_component_profile(
            row
        )


        batch_disturbance = reconstruct_batch_disturbance(
            row
        )


        # ----------------------------------------------------
        # Deterministic BO seed
        #
        # Same formula as original pilot.
        # ----------------------------------------------------

        bo_seed = (
            RANDOM_SEED
            +
            int(
                assembly_id
            )
            * 10
            +
            component_index
        )


        # ----------------------------------------------------
        # NEW recommendation
        # ----------------------------------------------------

        recommendation = run_safe_bayesian_optimization(

            row=row,

            state=safe_state,

            seed=bo_seed,
        )


        # ----------------------------------------------------
        # ACTUAL simulator update
        # ----------------------------------------------------

        safe_state = apply_actual_correction(

            state=safe_state,

            component_profile=(
                component_profile
            ),

            batch_disturbance=(
                batch_disturbance
            ),

            recommendation=(
                recommendation
            ),
        )


        actual_metrics = calculate_quality_metrics(
            safe_state
        )


        prediction_error = (
            actual_metrics[
                "quality_score"
            ]
            -
            recommendation[
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
                    row[
                        "severity"
                    ],

                "scenario_type":
                    row[
                        "scenario_type"
                    ],


                # --------------------------------------------
                # NEW surrogate recommendation
                # --------------------------------------------

                "safe_predicted_quality":
                    recommendation[
                        "predicted_quality"
                    ],

                "safe_actual_quality":
                    actual_metrics[
                        "quality_score"
                    ],

                "safe_prediction_error":
                    prediction_error,

                "safe_prediction_absolute_error":
                    abs(
                        prediction_error
                    ),


                # --------------------------------------------
                # Recommendation parameters
                # --------------------------------------------

                "safe_z_adj":
                    recommendation[
                        "z_adj"
                    ],

                "safe_theta_adj":
                    recommendation[
                        "theta_adj"
                    ],

                "safe_locator_offset":
                    recommendation[
                        "locator_offset"
                    ],


                # --------------------------------------------
                # Capability information
                # --------------------------------------------

                "safe_correction_utilization":
                    recommendation[
                        "utilization"
                    ],

                "safe_correction_effort":
                    recommendation[
                        "effort"
                    ],

                "safe_capability_penalty":
                    recommendation[
                        "penalty"
                    ],


                # --------------------------------------------
                # Actual quality metrics
                # --------------------------------------------

                "safe_mean_gap":
                    actual_metrics[
                        "mean_gap"
                    ],

                "safe_max_gap":
                    actual_metrics[
                        "max_gap"
                    ],

                "safe_parallelism":
                    actual_metrics[
                        "parallelism_error"
                    ],

                "safe_rms":
                    actual_metrics[
                        "rms_deviation"
                    ],
            }
        )


    # ========================================================
    # FINAL ASSEMBLY
    # ========================================================

    safe_final = calculate_quality_metrics(
        safe_state
    )


    old_row = old_assembly_df[
        old_assembly_df[
            "assembly_id"
        ]
        ==
        assembly_id
    ]


    if len(old_row) != 1:

        raise ValueError(
            f"\nCould not find exactly one original pilot "
            f"result for assembly {assembly_id}."
        )


    old_row = old_row.iloc[
        0
    ]


    zero_quality = float(
        old_row[
            "zero_final_quality"
        ]
    )


    random_quality = float(
        old_row[
            "random_final_quality"
        ]
    )


    old_bo_quality = float(
        old_row[
            "bo_final_quality"
        ]
    )


    safe_quality = float(
        safe_final[
            "quality_score"
        ]
    )


    safe_improvement_vs_zero = (
        (
            zero_quality
            -
            safe_quality
        )
        /
        max(
            abs(
                zero_quality
            ),
            1e-9,
        )
        *
        100.0
    )


    safe_improvement_vs_random = (
        (
            random_quality
            -
            safe_quality
        )
        /
        max(
            abs(
                random_quality
            ),
            1e-9,
        )
        *
        100.0
    )


    safe_improvement_vs_old_bo = (
        (
            old_bo_quality
            -
            safe_quality
        )
        /
        max(
            abs(
                old_bo_quality
            ),
            1e-9,
        )
        *
        100.0
    )


    assembly_results.append(
        {

            "assembly_id":
                assembly_id,

            "batch_condition":
                batch_condition,

            "zero_final_quality":
                zero_quality,

            "random_final_quality":
                random_quality,

            "old_bo_final_quality":
                old_bo_quality,

            "safe_bo_final_quality":
                safe_quality,

            "safe_improvement_vs_zero_percent":
                safe_improvement_vs_zero,

            "safe_improvement_vs_random_percent":
                safe_improvement_vs_random,

            "safe_improvement_vs_old_bo_percent":
                safe_improvement_vs_old_bo,

            "safe_final_mean_gap":
                safe_final[
                    "mean_gap"
                ],

            "safe_final_max_gap":
                safe_final[
                    "max_gap"
                ],

            "safe_final_parallelism":
                safe_final[
                    "parallelism_error"
                ],

            "safe_final_rms":
                safe_final[
                    "rms_deviation"
                ],
        }
    )


    # --------------------------------------------------------
    # PROGRESS + CHECKPOINT
    # --------------------------------------------------------

    if assembly_counter % 10 == 0:

        elapsed = (
            time.time()
            -
            experiment_start
        )


        print(
            f"Completed "
            f"{assembly_counter}/"
            f"{len(selected_assemblies)} assemblies"
            f" | elapsed = "
            f"{elapsed / 60.0:.2f} min"
        )


        # Save checkpoints every 10 assemblies.

        pd.DataFrame(
            component_results
        ).to_csv(
            NEW_COMPONENT_OUTPUT,
            index=False,
        )


        pd.DataFrame(
            assembly_results
        ).to_csv(
            NEW_ASSEMBLY_OUTPUT,
            index=False,
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
# SAVE FINAL RAW RESULTS
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
    NEW_COMPONENT_OUTPUT,
    index=False,
)


assembly_result_df.to_csv(
    NEW_ASSEMBLY_OUTPUT,
    index=False,
)


# ============================================================
# FINAL QUALITY SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    {

        "strategy": [
            "zero_correction",
            "random_search_original",
            "old_rf_standard_bo",
            "closed_loop_rf_p4_safe_bo",
        ],

        "mean_final_quality": [

            assembly_result_df[
                "zero_final_quality"
            ].mean(),

            assembly_result_df[
                "random_final_quality"
            ].mean(),

            assembly_result_df[
                "old_bo_final_quality"
            ].mean(),

            assembly_result_df[
                "safe_bo_final_quality"
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
                "old_bo_final_quality"
            ].median(),

            assembly_result_df[
                "safe_bo_final_quality"
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
                "old_bo_final_quality"
            ].std(),

            assembly_result_df[
                "safe_bo_final_quality"
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
                "old_bo_final_quality"
            ].quantile(
                0.95
            ),

            assembly_result_df[
                "safe_bo_final_quality"
            ].quantile(
                0.95
            ),
        ],
    }
)


summary_df.to_csv(
    SUMMARY_OUTPUT,
    index=False,
)


# ============================================================
# WIN RATES
# ============================================================

safe_win_vs_zero = (
    assembly_result_df[
        "safe_bo_final_quality"
    ]
    <
    assembly_result_df[
        "zero_final_quality"
    ]
).mean() * 100.0


safe_win_vs_random = (
    assembly_result_df[
        "safe_bo_final_quality"
    ]
    <
    assembly_result_df[
        "random_final_quality"
    ]
).mean() * 100.0


safe_win_vs_old_bo = (
    assembly_result_df[
        "safe_bo_final_quality"
    ]
    <
    assembly_result_df[
        "old_bo_final_quality"
    ]
).mean() * 100.0


# ============================================================
# MEAN PAIRED IMPROVEMENTS
# ============================================================

mean_improvement_zero = (
    assembly_result_df[
        "safe_improvement_vs_zero_percent"
    ]
    .mean()
)


mean_improvement_random = (
    assembly_result_df[
        "safe_improvement_vs_random_percent"
    ]
    .mean()
)


mean_improvement_old_bo = (
    assembly_result_df[
        "safe_improvement_vs_old_bo_percent"
    ]
    .mean()
)


# ============================================================
# NEW SURROGATE VERIFICATION
# ============================================================

mean_safe_prediction_error = (
    component_result_df[
        "safe_prediction_absolute_error"
    ]
    .mean()
)


p95_safe_prediction_error = (
    component_result_df[
        "safe_prediction_absolute_error"
    ]
    .quantile(
        0.95
    )
)


stage5_safe_prediction_error = (
    component_result_df[
        component_result_df[
            "component_index"
        ]
        ==
        5
    ][
        "safe_prediction_absolute_error"
    ]
    .mean()
)


# ============================================================
# OLD SURROGATE VERIFICATION
# ============================================================

old_mean_prediction_error = (
    old_component_df[
        "bo_prediction_absolute_error"
    ]
    .mean()
)


old_p95_prediction_error = (
    old_component_df[
        "bo_prediction_absolute_error"
    ]
    .quantile(
        0.95
    )
)


old_stage5_prediction_error = (
    old_component_df[
        old_component_df[
            "component_index"
        ]
        ==
        5
    ][
        "bo_prediction_absolute_error"
    ]
    .mean()
)


# ============================================================
# CAPABILITY
# ============================================================

safe_mean_utilization = (
    component_result_df[
        "safe_correction_utilization"
    ]
    .mean()
)


safe_p95_utilization = (
    component_result_df[
        "safe_correction_utilization"
    ]
    .quantile(
        0.95
    )
)


safe_near_capability = (
    component_result_df[
        "safe_correction_utilization"
    ]
    >=
    0.90
).mean() * 100.0


safe_at_capability = (
    component_result_df[
        "safe_correction_utilization"
    ]
    >=
    0.999
).mean() * 100.0


old_mean_utilization = (
    old_component_df[
        "bo_correction_utilization"
    ]
    .mean()
)


old_near_capability = (
    old_component_df[
        "bo_correction_utilization"
    ]
    >=
    0.90
).mean() * 100.0


old_at_capability = (
    old_component_df[
        "bo_correction_utilization"
    ]
    >=
    0.999
).mean() * 100.0


# ============================================================
# STAGE-WISE TABLE
# ============================================================

stage_rows = []


for component_index in range(
    1,
    6,
):


    old_stage = old_component_df[
        old_component_df[
            "component_index"
        ]
        ==
        component_index
    ]


    new_stage = component_result_df[
        component_result_df[
            "component_index"
        ]
        ==
        component_index
    ]


    stage_rows.append(
        {

            "component_index":
                component_index,

            "old_mean_actual_quality":
                old_stage[
                    "bo_actual_quality"
                ]
                .mean(),

            "safe_mean_actual_quality":
                new_stage[
                    "safe_actual_quality"
                ]
                .mean(),

            "old_mean_prediction_abs_error":
                old_stage[
                    "bo_prediction_absolute_error"
                ]
                .mean(),

            "safe_mean_prediction_abs_error":
                new_stage[
                    "safe_prediction_absolute_error"
                ]
                .mean(),

            "old_mean_utilization":
                old_stage[
                    "bo_correction_utilization"
                ]
                .mean(),

            "safe_mean_utilization":
                new_stage[
                    "safe_correction_utilization"
                ]
                .mean(),
        }
    )


stage_df = pd.DataFrame(
    stage_rows
)


STAGE_OUTPUT = (
    "results/tables/"
    "v3_2_old_vs_safe_bo_stage_comparison.csv"
)


stage_df.to_csv(
    STAGE_OUTPUT,
    index=False,
)


# ============================================================
# COMPARISON TABLE
# ============================================================

comparison_df = pd.DataFrame(
    {

        "metric": [

            "mean_final_quality",

            "median_final_quality",

            "p95_final_quality",

            "win_rate_vs_zero_percent",

            "win_rate_vs_random_percent",

            "mean_prediction_abs_error",

            "p95_prediction_abs_error",

            "stage5_prediction_abs_error",

            "mean_correction_utilization",

            "near_capability_ge_90_percent",

            "at_capability_ge_99_9_percent",
        ],

        "old_standard_bo": [

            assembly_result_df[
                "old_bo_final_quality"
            ]
            .mean(),

            assembly_result_df[
                "old_bo_final_quality"
            ]
            .median(),

            assembly_result_df[
                "old_bo_final_quality"
            ]
            .quantile(
                0.95
            ),

            (
                assembly_result_df[
                    "old_bo_final_quality"
                ]
                <
                assembly_result_df[
                    "zero_final_quality"
                ]
            )
            .mean()
            *
            100.0,

            (
                assembly_result_df[
                    "old_bo_final_quality"
                ]
                <
                assembly_result_df[
                    "random_final_quality"
                ]
            )
            .mean()
            *
            100.0,

            old_mean_prediction_error,

            old_p95_prediction_error,

            old_stage5_prediction_error,

            old_mean_utilization,

            old_near_capability,

            old_at_capability,
        ],

        "new_closed_loop_p4_bo": [

            assembly_result_df[
                "safe_bo_final_quality"
            ]
            .mean(),

            assembly_result_df[
                "safe_bo_final_quality"
            ]
            .median(),

            assembly_result_df[
                "safe_bo_final_quality"
            ]
            .quantile(
                0.95
            ),

            safe_win_vs_zero,

            safe_win_vs_random,

            mean_safe_prediction_error,

            p95_safe_prediction_error,

            stage5_safe_prediction_error,

            safe_mean_utilization,

            safe_near_capability,

            safe_at_capability,
        ],
    }
)


comparison_df[
    "difference_new_minus_old"
] = (
    comparison_df[
        "new_closed_loop_p4_bo"
    ]
    -
    comparison_df[
        "old_standard_bo"
    ]
)


comparison_df.to_csv(
    COMPARISON_OUTPUT,
    index=False,
)


# ============================================================
# RUNTIME
# ============================================================

total_runtime = (
    time.time()
    -
    experiment_start
)


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "FIX 1 + FIX 2 FINAL PILOT RESULTS"
)

print(
    "============================================================"
)


print(
    "\nFINAL QUALITY SUMMARY\n"
)


print(
    summary_df.round(
        4
    ).to_string(
        index=False
    )
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "NEW SAFE BO - PAIRED WIN RATES"
)

print(
    "------------------------------------------------------------"
)


print(
    f"Safe BO better than ZERO       : "
    f"{safe_win_vs_zero:.2f}%"
)

print(
    f"Safe BO better than RANDOM     : "
    f"{safe_win_vs_random:.2f}%"
)

print(
    f"Safe BO better than OLD BO     : "
    f"{safe_win_vs_old_bo:.2f}%"
)


print(
    f"\nMean improvement vs ZERO       : "
    f"{mean_improvement_zero:.2f}%"
)

print(
    f"Mean improvement vs RANDOM     : "
    f"{mean_improvement_random:.2f}%"
)

print(
    f"Mean improvement vs OLD BO     : "
    f"{mean_improvement_old_bo:.2f}%"
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "SURROGATE BEFORE vs AFTER"
)

print(
    "------------------------------------------------------------"
)


print(
    f"Old mean abs error       : "
    f"{old_mean_prediction_error:.4f}"
)

print(
    f"New mean abs error       : "
    f"{mean_safe_prediction_error:.4f}"
)


print(
    f"\nOld P95 abs error        : "
    f"{old_p95_prediction_error:.4f}"
)

print(
    f"New P95 abs error        : "
    f"{p95_safe_prediction_error:.4f}"
)


print(
    f"\nOld Component-5 MAE      : "
    f"{old_stage5_prediction_error:.4f}"
)

print(
    f"New Component-5 MAE      : "
    f"{stage5_safe_prediction_error:.4f}"
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "CAPABILITY BEFORE vs AFTER"
)

print(
    "------------------------------------------------------------"
)


print(
    f"Old mean utilization     : "
    f"{old_mean_utilization:.3f}"
)

print(
    f"New mean utilization     : "
    f"{safe_mean_utilization:.3f}"
)


print(
    f"\nOld >=90% capability     : "
    f"{old_near_capability:.2f}%"
)

print(
    f"New >=90% capability     : "
    f"{safe_near_capability:.2f}%"
)


print(
    f"\nOld >=99.9% capability   : "
    f"{old_at_capability:.2f}%"
)

print(
    f"New >=99.9% capability   : "
    f"{safe_at_capability:.2f}%"
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "STAGE-WISE COMPARISON"
)

print(
    "------------------------------------------------------------\n"
)


print(
    stage_df.round(
        4
    ).to_string(
        index=False
    )
)


# ============================================================
# EVIDENCE-BASED VERDICT
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "FIX 1 + FIX 2 EVALUATION"
)

print(
    "============================================================"
)


if (
    mean_safe_prediction_error
    <
    old_mean_prediction_error
):

    print(
        "PASS - Closed-loop RF reduced mean "
        "surrogate error during actual adaptive operation."
    )

else:

    print(
        "WARNING - Closed-loop RF did not reduce mean "
        "surrogate error during actual adaptive operation."
    )


if (
    stage5_safe_prediction_error
    <
    old_stage5_prediction_error
):

    print(
        "PASS - Late-stage Component-5 surrogate error improved."
    )

else:

    print(
        "WARNING - Late-stage Component-5 surrogate "
        "error did not improve."
    )


if (
    safe_near_capability
    <
    old_near_capability
):

    print(
        "PASS - P4 capability-aware BO reduced "
        "near-limit recommendations."
    )

else:

    print(
        "WARNING - Near-limit recommendation rate "
        "did not improve."
    )


if (
    safe_at_capability
    <
    old_at_capability
):

    print(
        "PASS - P4 capability-aware BO reduced "
        "full capability saturation."
    )

else:

    print(
        "WARNING - Full capability saturation "
        "did not improve."
    )


old_win_vs_zero = (
    assembly_result_df[
        "old_bo_final_quality"
    ]
    <
    assembly_result_df[
        "zero_final_quality"
    ]
).mean() * 100.0


if (
    safe_win_vs_zero
    >
    old_win_vs_zero
):

    print(
        "PASS - New controller increased the "
        "assembly win rate versus zero correction."
    )

else:

    print(
        "WARNING - New controller did not increase "
        "the win rate versus zero correction."
    )


if (
    assembly_result_df[
        "safe_bo_final_quality"
    ]
    .mean()
    <
    assembly_result_df[
        "zero_final_quality"
    ]
    .mean()
):

    print(
        "PASS - New controller achieved lower "
        "mean final quality score than zero correction."
    )

else:

    print(
        "IMPORTANT - New controller still does NOT "
        "beat zero correction in mean final quality."
    )


print(
    "\nIMPORTANT:"
)

print(
    "No final scientific conclusion should be made solely "
    "from this 100-assembly pilot."
)

print(
    "This experiment decides whether the two diagnosed fixes "
    "are promising enough for larger robustness validation."
)


# ============================================================
# SAVED FILES
# ============================================================

print(
    "\nSaved:"
)

print(
    NEW_COMPONENT_OUTPUT
)

print(
    NEW_ASSEMBLY_OUTPUT
)

print(
    SUMMARY_OUTPUT
)

print(
    COMPARISON_OUTPUT
)

print(
    STAGE_OUTPUT
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
    f"Total runtime      : "
    f"{total_runtime / 60.0:.2f} min"
)

print(
    f"Runtime / assembly : "
    f"{total_runtime / len(selected_assemblies):.2f} sec"
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.2 CONTROLLED FIX VERIFICATION COMPLETED"
)

print(
    "============================================================"
)