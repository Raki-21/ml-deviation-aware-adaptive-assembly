import os
import time
import joblib
import numpy as np
import pandas as pd

from skopt import Optimizer
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
# FINAL INDEPENDENT CONTROLLER VALIDATION
# ============================================================
#
# PURPOSE
#
# Evaluate the frozen V3.2 adaptive assembly controller on a
# completely independent synthetic validation population.
#
#
# VALIDATION DATA:
#
#   300 unseen finished assemblies
#   30 unseen batch realizations
#   6 balanced batch/process conditions
#   5 sequential components / assembly
#   1500 sequential decisions
#
#
# STRATEGIES
#
# 1. ZERO CORRECTION
#
# 2. STRUCTURED-20 RF
#
#       20 physically structured candidates
#       selected using frozen profile-aware Random Forest
#
# 3. SELECTIVE BAYESIAN OPTIMIZATION
#
#       First evaluate Structured-20 candidates.
#
#       Compute:
#
#           margin =
#           predicted second-best quality
#           -
#           predicted best quality
#
#       If:
#
#           margin <= FROZEN_TRIGGER_MARGIN
#
#       invoke the previously validated structured warm-start
#       Bayesian Optimization controller.
#
#       Otherwise directly use Structured-20.
#
#
# IMPORTANT SCIENTIFIC RULE
#
# Everything below is FROZEN from development:
#
#   - RF model
#   - model features
#   - correction bounds
#   - Structured-20 candidate design
#   - BO trust region
#   - BO evaluation budget
#   - capability penalty
#   - selective trigger threshold
#
#
# NOTHING is recalibrated from this validation dataset.
# ============================================================


# ============================================================
# FILE PATHS
# ============================================================

VALIDATION_COMPONENT_FILE = (
    "data/validation/"
    "v3_2_independent_validation_component_dataset.csv"
)

VALIDATION_ASSEMBLY_FILE = (
    "data/validation/"
    "v3_2_independent_validation_assembly_dataset.csv"
)

VALIDATION_MANIFEST_FILE = (
    "data/validation/"
    "v3_2_independent_validation_manifest.csv"
)


MODEL_FILE = (
    "models/"
    "v3_2_profile_aware_quality_surrogate.joblib"
)

FEATURE_FILE = (
    "models/"
    "v3_2_profile_aware_surrogate_features.txt"
)


COMPONENT_OUTPUT = (
    "data/validation/"
    "v3_2_final_controller_validation_component_results.csv"
)

ASSEMBLY_OUTPUT = (
    "data/validation/"
    "v3_2_final_controller_validation_assembly_results.csv"
)


SUMMARY_OUTPUT = (
    "results/validation/"
    "v3_2_final_controller_validation_summary.csv"
)

STAGE_OUTPUT = (
    "results/validation/"
    "v3_2_final_controller_validation_by_stage.csv"
)

BATCH_OUTPUT = (
    "results/validation/"
    "v3_2_final_controller_validation_by_batch_condition.csv"
)

SEVERITY_OUTPUT = (
    "results/validation/"
    "v3_2_final_controller_validation_by_severity.csv"
)

SCENARIO_OUTPUT = (
    "results/validation/"
    "v3_2_final_controller_validation_by_scenario.csv"
)


# ============================================================
# REQUIRED FILE CHECK
# ============================================================

required_files = [
    VALIDATION_COMPONENT_FILE,
    VALIDATION_ASSEMBLY_FILE,
    VALIDATION_MANIFEST_FILE,
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
        "\nERROR - Missing required validation files:"
    )

    for path in missing_files:

        print(path)

    raise SystemExit(
        "\nIndependent validation generation must be completed "
        "before controller validation."
    )


# ============================================================
# FROZEN CONTROLLER SETTINGS
# ============================================================

N_COMPONENTS = 5


# ------------------------------------------------------------
# Controller random seed
# ------------------------------------------------------------
#
# This belongs to the controller, not the validation-data RNG.
# It remains fixed for reproducibility.
# ------------------------------------------------------------

CONTROLLER_SEED = 20260825


# ============================================================
# FROZEN SELECTIVE-BO TRIGGER
# ============================================================
#
# Derived BEFORE independent validation.
#
# Development calibration result:
#
# Trigger rate:
#       20.00%
#
# Total regret captured:
#       77.79%
#
# Exact-best misses captured:
#       75.76%
#
# This value MUST NOT be retuned from final validation data.
# ============================================================

FROZEN_TRIGGER_MARGIN = (
    0.02800373921277931
)


# ============================================================
# CORRECTION CAPABILITY
# ============================================================

Z_LIMIT = 2.5

THETA_LIMIT = 1.2

LOCATOR_LIMIT = 1.0


# ============================================================
# STRUCTURED-20 SETTINGS
# ============================================================

N_STRUCTURED_CANDIDATES = 20


# ============================================================
# WARM-START BO SETTINGS
# ============================================================

TOTAL_BO_EVALUATIONS = 20

N_STRUCTURED_INITIAL = 8

N_BO_EVALUATIONS = (
    TOTAL_BO_EVALUATIONS
    -
    N_STRUCTURED_INITIAL
)


# ============================================================
# FROZEN TRUST REGION
# ============================================================

TRUST_Z_RADIUS = 0.90

TRUST_THETA_RADIUS = 0.36

TRUST_LOCATOR_RADIUS = 0.45


# ============================================================
# FROZEN CAPABILITY SAFEGUARD
# ============================================================

BARRIER_START = 0.80

BARRIER_WEIGHT = 1.25

EFFORT_WEIGHT = 0.04


# ============================================================
# PROFILE REPRESENTATION
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
# LOAD DATA + MODEL
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "V3.2 FINAL INDEPENDENT CONTROLLER VALIDATION"
)

print(
    "============================================================"
)


component_df = pd.read_csv(
    VALIDATION_COMPONENT_FILE
)


validation_assembly_df = pd.read_csv(
    VALIDATION_ASSEMBLY_FILE
)


manifest_df = pd.read_csv(
    VALIDATION_MANIFEST_FILE
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
# VALIDATION PROVENANCE CHECK
# ============================================================

manifest_lookup = dict(
    zip(
        manifest_df[
            "setting"
        ],
        manifest_df[
            "value"
        ],
    )
)


print(
    "\nValidation dataset role:"
)

print(
    manifest_lookup.get(
        "dataset_role",
        "UNKNOWN",
    )
)


print(
    "\nValidation seed:"
)

print(
    manifest_lookup.get(
        "validation_seed",
        "UNKNOWN",
    )
)


# ============================================================
# ASSEMBLY IDS
# ============================================================

selected_assemblies = sorted(
    component_df[
        "assembly_id"
    ]
    .unique()
    .tolist()
)


n_assemblies = len(
    selected_assemblies
)


expected_decisions = (
    n_assemblies
    *
    N_COMPONENTS
)


print(
    f"\nIndependent assemblies       : "
    f"{n_assemblies}"
)

print(
    f"Components / assembly        : "
    f"{N_COMPONENTS}"
)

print(
    f"Expected decisions           : "
    f"{expected_decisions}"
)

print(
    f"Frozen trigger margin        : "
    f"{FROZEN_TRIGGER_MARGIN:.12f}"
)

print(
    f"Structured candidates        : "
    f"{N_STRUCTURED_CANDIDATES}"
)

print(
    f"BO evaluations when triggered: "
    f"{TOTAL_BO_EVALUATIONS}"
)


# ============================================================
# VALIDATE 300-ASSEMBLY DESIGN
# ============================================================

if n_assemblies != 300:

    raise ValueError(
        f"\nExpected 300 independent validation assemblies, "
        f"found {n_assemblies}."
    )


if len(
    component_df
) != expected_decisions:

    raise ValueError(
        f"\nExpected {expected_decisions} validation decisions, "
        f"found {len(component_df)}."
    )


# ============================================================
# COMPONENT PROFILE RECONSTRUCTION
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
# BATCH DISTURBANCE RECONSTRUCTION
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
# STATE FEATURES
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


    features = {

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


    for (
        name,
        index,
    ) in zip(
        PROFILE_SAMPLE_NAMES,
        PROFILE_SAMPLE_INDICES,
    ):

        features[
            name
        ] = float(
            state[
                index
            ]
        )


    return features


# ============================================================
# CAPABILITY UTILIZATION
# ============================================================

def calculate_utilization(
    z_adj,
    theta_adj,
    locator_offset,
):

    return float(
        max(

            abs(
                z_adj
            )
            /
            Z_LIMIT,

            abs(
                theta_adj
            )
            /
            THETA_LIMIT,

            abs(
                locator_offset
            )
            /
            LOCATOR_LIMIT,
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


    return float(
        np.sqrt(
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
    )


# ============================================================
# CAPABILITY PENALTY
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

        barrier_penalty = 0.0

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


        barrier_penalty = (
            BARRIER_WEIGHT
            *
            normalized_excess ** 3
        )


    return float(
        effort_penalty
        +
        barrier_penalty
    )


# ============================================================
# CLIP CORRECTION
# ============================================================

def clip_correction(
    z_adj,
    theta_adj,
    locator_offset,
):

    return (

        float(
            np.clip(
                z_adj,
                -Z_LIMIT,
                Z_LIMIT,
            )
        ),

        float(
            np.clip(
                theta_adj,
                -THETA_LIMIT,
                THETA_LIMIT,
            )
        ),

        float(
            np.clip(
                locator_offset,
                -LOCATOR_LIMIT,
                LOCATOR_LIMIT,
            )
        ),
    )


# ============================================================
# INFORMED CORRECTION
# ============================================================

def informed_correction(
    state,
    row,
):

    signed_mean = float(
        np.mean(
            state
        )
    )


    signed_difference = float(
        state[-1]
        -
        state[0]
    )


    state_angle = float(
        np.rad2deg(
            np.arctan(
                signed_difference
                /
                PROFILE_LENGTH_MM
            )
        )
    )


    z_adj = -(
        row[
            "offset_mm"
        ]
        +
        row[
            "batch_offset_bias_mm"
        ]
        +
        0.50
        *
        signed_mean
    )


    theta_adj = -(
        row[
            "tilt_deg"
        ]
        +
        row[
            "batch_angular_bias_deg"
        ]
        +
        0.50
        *
        state_angle
    )


    locator_offset = -(
        row[
            "local_bump_mm"
        ]
        +
        row[
            "fixture_drift_mm"
        ]
    )


    return clip_correction(
        z_adj,
        theta_adj,
        locator_offset,
    )


# ============================================================
# STRUCTURED-20 CANDIDATES
# ============================================================

def generate_structured_candidates(
    state,
    row,
    seed,
):

    rng = np.random.default_rng(
        seed
    )


    candidates = []


    # --------------------------------------------------------
    # ZERO
    # --------------------------------------------------------

    candidates.append(
        {
            "z_adj": 0.0,
            "theta_adj": 0.0,
            "locator_offset": 0.0,
            "source": "zero",
        }
    )


    # --------------------------------------------------------
    # INFORMED
    # --------------------------------------------------------

    (
        informed_z,
        informed_theta,
        informed_locator,
    ) = informed_correction(
        state,
        row,
    )


    candidates.append(
        {
            "z_adj":
                informed_z,

            "theta_adj":
                informed_theta,

            "locator_offset":
                informed_locator,

            "source":
                "informed",
        }
    )


    # --------------------------------------------------------
    # LOCAL INFORMED - 8
    # --------------------------------------------------------

    for _ in range(
        8
    ):


        (
            z_adj,
            theta_adj,
            locator_offset,
        ) = clip_correction(

            informed_z
            +
            rng.normal(
                0.0,
                0.30,
            ),

            informed_theta
            +
            rng.normal(
                0.0,
                0.12,
            ),

            informed_locator
            +
            rng.normal(
                0.0,
                0.15,
            ),
        )


        candidates.append(
            {
                "z_adj":
                    z_adj,

                "theta_adj":
                    theta_adj,

                "locator_offset":
                    locator_offset,

                "source":
                    "local_informed",
            }
        )


    # --------------------------------------------------------
    # MODERATE RANDOM - 6
    # --------------------------------------------------------

    for _ in range(
        6
    ):


        candidates.append(
            {
                "z_adj":
                    float(
                        rng.uniform(
                            -0.60
                            *
                            Z_LIMIT,

                            0.60
                            *
                            Z_LIMIT,
                        )
                    ),

                "theta_adj":
                    float(
                        rng.uniform(
                            -0.60
                            *
                            THETA_LIMIT,

                            0.60
                            *
                            THETA_LIMIT,
                        )
                    ),

                "locator_offset":
                    float(
                        rng.uniform(
                            -0.60
                            *
                            LOCATOR_LIMIT,

                            0.60
                            *
                            LOCATOR_LIMIT,
                        )
                    ),

                "source":
                    "moderate_random",
            }
        )


    # --------------------------------------------------------
    # BROAD RANDOM - 4
    # --------------------------------------------------------

    for _ in range(
        4
    ):


        candidates.append(
            {
                "z_adj":
                    float(
                        rng.uniform(
                            -Z_LIMIT,
                            Z_LIMIT,
                        )
                    ),

                "theta_adj":
                    float(
                        rng.uniform(
                            -THETA_LIMIT,
                            THETA_LIMIT,
                        )
                    ),

                "locator_offset":
                    float(
                        rng.uniform(
                            -LOCATOR_LIMIT,
                            LOCATOR_LIMIT,
                        )
                    ),

                "source":
                    "broad_random",
            }
        )


    if len(
        candidates
    ) != N_STRUCTURED_CANDIDATES:

        raise RuntimeError(
            f"Expected {N_STRUCTURED_CANDIDATES} candidates, "
            f"found {len(candidates)}."
        )


    return candidates


# ============================================================
# MODEL INPUT FOR MULTIPLE CANDIDATES
# ============================================================

def build_candidate_model_input(
    row,
    state,
    candidates,
):

    state_features = calculate_state_features(
        state
    )


    records = []


    for candidate in candidates:


        z_adj = candidate[
            "z_adj"
        ]


        theta_adj = candidate[
            "theta_adj"
        ]


        locator_offset = candidate[
            "locator_offset"
        ]


        records.append(
            {

                **state_features,


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


                "z_adj":
                    z_adj,

                "theta_adj":
                    theta_adj,

                "locator_offset":
                    locator_offset,

                "correction_utilization":
                    calculate_utilization(
                        z_adj,
                        theta_adj,
                        locator_offset,
                    ),

                "component_index":
                    row[
                        "component_index"
                    ],
            }
        )


    X = pd.DataFrame(
        records
    )


    return X[
        FEATURE_COLUMNS
    ]


# ============================================================
# MODEL INPUT FOR ONE CORRECTION
# ============================================================

def build_single_model_input(
    row,
    state,
    z_adj,
    theta_adj,
    locator_offset,
):

    candidate = {
        "z_adj":
            z_adj,

        "theta_adj":
            theta_adj,

        "locator_offset":
            locator_offset,
    }


    return build_candidate_model_input(

        row=row,

        state=state,

        candidates=[
            candidate
        ],
    )


# ============================================================
# STRUCTURED RF RECOMMENDATION
# ============================================================

def structured_rf_recommendation(
    row,
    state,
    seed,
):

    candidates = generate_structured_candidates(

        state=state,

        row=row,

        seed=seed,
    )


    X = build_candidate_model_input(

        row=row,

        state=state,

        candidates=candidates,
    )


    predictions = model.predict(
        X
    )


    order = np.argsort(
        predictions
    )


    best_index = int(
        order[
            0
        ]
    )


    second_index = int(
        order[
            1
        ]
    )


    best_prediction = float(
        predictions[
            best_index
        ]
    )


    second_prediction = float(
        predictions[
            second_index
        ]
    )


    margin = float(
        second_prediction
        -
        best_prediction
    )


    selected_candidate = candidates[
        best_index
    ]


    return {

        "candidate":
            selected_candidate,

        "predicted_quality":
            best_prediction,

        "second_predicted_quality":
            second_prediction,

        "prediction_margin":
            margin,

        "all_candidates":
            candidates,

        "all_predictions":
            predictions,

        "best_index":
            best_index,
    }


# ============================================================
# SURROGATE OBJECTIVE FOR BO
# ============================================================

def evaluate_surrogate_objective(
    row,
    state,
    point,
):

    (
        z_adj,
        theta_adj,
        locator_offset,
    ) = point


    X = build_single_model_input(

        row=row,

        state=state,

        z_adj=z_adj,

        theta_adj=theta_adj,

        locator_offset=locator_offset,
    )


    predicted_quality = float(
        model.predict(
            X
        )[0]
    )


    penalty = capability_penalty(

        z_adj,

        theta_adj,

        locator_offset,
    )


    objective = (
        predicted_quality
        +
        penalty
    )


    return (
        predicted_quality,
        penalty,
        objective,
    )


# ============================================================
# BO STRUCTURED INITIAL CANDIDATES
# ============================================================

def generate_bo_initial_candidates(
    state,
    row,
    seed,
):

    rng = np.random.default_rng(
        seed
    )


    (
        informed_z,
        informed_theta,
        informed_locator,
    ) = informed_correction(
        state,
        row,
    )


    candidates = [

        (
            0.0,
            0.0,
            0.0,
        ),

        (
            informed_z,
            informed_theta,
            informed_locator,
        ),
    ]


    for _ in range(
        6
    ):


        (
            z_adj,
            theta_adj,
            locator_offset,
        ) = clip_correction(

            informed_z
            +
            rng.normal(
                0.0,
                0.30,
            ),

            informed_theta
            +
            rng.normal(
                0.0,
                0.12,
            ),

            informed_locator
            +
            rng.normal(
                0.0,
                0.15,
            ),
        )


        candidates.append(
            (
                z_adj,
                theta_adj,
                locator_offset,
            )
        )


    return candidates


# ============================================================
# LOCAL TRUST REGION
# ============================================================

def create_local_space(
    centre,
):

    (
        centre_z,
        centre_theta,
        centre_locator,
    ) = centre


    z_low = max(
        -Z_LIMIT,
        centre_z
        -
        TRUST_Z_RADIUS,
    )


    z_high = min(
        Z_LIMIT,
        centre_z
        +
        TRUST_Z_RADIUS,
    )


    theta_low = max(
        -THETA_LIMIT,
        centre_theta
        -
        TRUST_THETA_RADIUS,
    )


    theta_high = min(
        THETA_LIMIT,
        centre_theta
        +
        TRUST_THETA_RADIUS,
    )


    locator_low = max(
        -LOCATOR_LIMIT,
        centre_locator
        -
        TRUST_LOCATOR_RADIUS,
    )


    locator_high = min(
        LOCATOR_LIMIT,
        centre_locator
        +
        TRUST_LOCATOR_RADIUS,
    )


    space = [

        Real(
            z_low,
            z_high,
            name="z_adj",
        ),

        Real(
            theta_low,
            theta_high,
            name="theta_adj",
        ),

        Real(
            locator_low,
            locator_high,
            name="locator_offset",
        ),
    ]


    bounds = (

        (
            z_low,
            z_high,
        ),

        (
            theta_low,
            theta_high,
        ),

        (
            locator_low,
            locator_high,
        ),
    )


    return (
        space,
        bounds,
    )


# ============================================================
# POINT INSIDE TRUST REGION
# ============================================================

def point_inside_bounds(
    point,
    bounds,
):

    for value, (
        lower,
        upper,
    ) in zip(
        point,
        bounds,
    ):


        if (
            value
            <
            lower
            or
            value
            >
            upper
        ):

            return False


    return True


# ============================================================
# EXACT FROZEN WARM-START BO
# ============================================================

def run_warmstart_bo(
    row,
    state,
    seed,
):


    initial_candidates = (
        generate_bo_initial_candidates(

            state=state,

            row=row,

            seed=seed,
        )
    )


    evaluated_points = []

    predicted_qualities = []

    penalties = []

    objectives = []


    # --------------------------------------------------------
    # Eight structured initial evaluations
    # --------------------------------------------------------

    for point in initial_candidates:


        (
            predicted_quality,
            penalty,
            objective,
        ) = evaluate_surrogate_objective(

            row=row,

            state=state,

            point=point,
        )


        evaluated_points.append(
            tuple(
                float(
                    value
                )
                for value
                in point
            )
        )


        predicted_qualities.append(
            predicted_quality
        )


        penalties.append(
            penalty
        )


        objectives.append(
            objective
        )


    initial_best_index = int(
        np.argmin(
            objectives
        )
    )


    best_initial_point = (
        evaluated_points[
            initial_best_index
        ]
    )


    (
        local_space,
        local_bounds,
    ) = create_local_space(
        best_initial_point
    )


    optimizer = Optimizer(

        dimensions=local_space,

        base_estimator="GP",

        acq_func="EI",

        n_initial_points=0,

        random_state=seed,
    )


    # --------------------------------------------------------
    # Warm-start GP
    # --------------------------------------------------------

    told_keys = set()


    for (
        point,
        objective,
    ) in zip(
        evaluated_points,
        objectives,
    ):


        if not point_inside_bounds(
            point,
            local_bounds,
        ):

            continue


        key = tuple(
            round(
                float(
                    value
                ),
                12,
            )
            for value
            in point
        )


        if key in told_keys:

            continue


        optimizer.tell(
            list(
                point
            ),
            float(
                objective
            ),
        )


        told_keys.add(
            key
        )


    # --------------------------------------------------------
    # Twelve BO refinement evaluations
    # --------------------------------------------------------

    for _ in range(
        N_BO_EVALUATIONS
    ):


        suggested = optimizer.ask()


        point = tuple(
            float(
                value
            )
            for value
            in suggested
        )


        (
            predicted_quality,
            penalty,
            objective,
        ) = evaluate_surrogate_objective(

            row=row,

            state=state,

            point=point,
        )


        optimizer.tell(
            list(
                point
            ),
            float(
                objective
            ),
        )


        evaluated_points.append(
            point
        )


        predicted_qualities.append(
            predicted_quality
        )


        penalties.append(
            penalty
        )


        objectives.append(
            objective
        )


    if len(
        evaluated_points
    ) != TOTAL_BO_EVALUATIONS:

        raise RuntimeError(
            f"Expected {TOTAL_BO_EVALUATIONS} BO evaluations, "
            f"found {len(evaluated_points)}."
        )


    final_best_index = int(
        np.argmin(
            objectives
        )
    )


    best_point = (
        evaluated_points[
            final_best_index
        ]
    )


    return {

        "z_adj":
            best_point[
                0
            ],

        "theta_adj":
            best_point[
                1
            ],

        "locator_offset":
            best_point[
                2
            ],

        "predicted_quality":
            predicted_qualities[
                final_best_index
            ],

        "capability_penalty":
            penalties[
                final_best_index
            ],

        "objective":
            objectives[
                final_best_index
            ],

        "utilization":
            calculate_utilization(

                best_point[
                    0
                ],

                best_point[
                    1
                ],

                best_point[
                    2
                ],
            ),

        "best_was_initial":
            (
                final_best_index
                <
                N_STRUCTURED_INITIAL
            ),
    }


# ============================================================
# ACTUAL SIMULATOR
# ============================================================

def simulate_correction(
    state,
    component_profile,
    batch_disturbance,
    point,
):

    (
        z_adj,
        theta_adj,
        locator_offset,
    ) = point


    correction = correction_profile(

        z_adj_mm=z_adj,

        theta_adj_deg=theta_adj,

        locator_offset_mm=(
            locator_offset
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


    metrics = calculate_quality_metrics(
        new_state
    )


    return (
        new_state,
        metrics,
    )


# ============================================================
# ZERO UPDATE
# ============================================================

def simulate_zero(
    state,
    component_profile,
    batch_disturbance,
):


    return simulate_correction(

        state=state,

        component_profile=(
            component_profile
        ),

        batch_disturbance=(
            batch_disturbance
        ),

        point=(
            0.0,
            0.0,
            0.0,
        ),
    )


# ============================================================
# STORAGE
# ============================================================

component_records = []

assembly_records = []


# ============================================================
# RUNTIME COUNTERS
# ============================================================

experiment_start = time.time()


total_decisions = 0

total_bo_triggers = 0


structured_prediction_time = 0.0

selective_prediction_time = 0.0

bo_runtime_total = 0.0


# ============================================================
# MAIN VALIDATION LOOP
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


    if len(
        assembly_rows
    ) != N_COMPONENTS:

        raise ValueError(
            f"\nAssembly {assembly_id} does not contain exactly "
            f"{N_COMPONENTS} component rows."
        )


    # --------------------------------------------------------
    # THREE INDEPENDENT CONTROLLER TRAJECTORIES
    # --------------------------------------------------------

    zero_state = create_initial_state()

    structured_state = create_initial_state()

    selective_state = create_initial_state()


    batch_condition = (
        assembly_rows.iloc[
            0
        ][
            "batch_condition"
        ]
    )


    batch_id = int(
        assembly_rows.iloc[
            0
        ][
            "batch_id"
        ]
    )


    for _, row in assembly_rows.iterrows():


        total_decisions += 1


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


        decision_seed = (
            CONTROLLER_SEED
            +
            int(
                assembly_id
            )
            *
            10
            +
            component_index
        )


        # ====================================================
        # 1. ZERO CORRECTION
        # ====================================================

        (
            zero_state,
            zero_metrics,
        ) = simulate_zero(

            state=zero_state,

            component_profile=(
                component_profile
            ),

            batch_disturbance=(
                batch_disturbance
            ),
        )


        # ====================================================
        # 2. STRUCTURED-20 RF
        # ====================================================

        structured_start = time.time()


        structured_recommendation = (
            structured_rf_recommendation(

                row=row,

                state=structured_state,

                seed=decision_seed,
            )
        )


        structured_prediction_time += (
            time.time()
            -
            structured_start
        )


        structured_candidate = (
            structured_recommendation[
                "candidate"
            ]
        )


        structured_point = (

            structured_candidate[
                "z_adj"
            ],

            structured_candidate[
                "theta_adj"
            ],

            structured_candidate[
                "locator_offset"
            ],
        )


        (
            structured_new_state,
            structured_metrics,
        ) = simulate_correction(

            state=structured_state,

            component_profile=(
                component_profile
            ),

            batch_disturbance=(
                batch_disturbance
            ),

            point=structured_point,
        )


        # ----------------------------------------------------
        # Actual verification of all Structured-20 candidates
        #
        # Simulator verification is used for VALIDATION ONLY.
        # It does not alter the selected controller action.
        # ----------------------------------------------------

        structured_actual_qualities = []


        for candidate in (
            structured_recommendation[
                "all_candidates"
            ]
        ):


            point = (

                candidate[
                    "z_adj"
                ],

                candidate[
                    "theta_adj"
                ],

                candidate[
                    "locator_offset"
                ],
            )


            (
                _,
                candidate_metrics,
            ) = simulate_correction(

                state=structured_state,

                component_profile=(
                    component_profile
                ),

                batch_disturbance=(
                    batch_disturbance
                ),

                point=point,
            )


            structured_actual_qualities.append(
                float(
                    candidate_metrics[
                        "quality_score"
                    ]
                )
            )


        structured_actual_qualities = np.asarray(
            structured_actual_qualities
        )


        structured_true_best_index = int(
            np.argmin(
                structured_actual_qualities
            )
        )


        structured_selected_index = int(
            structured_recommendation[
                "best_index"
            ]
        )


        structured_true_order = np.argsort(
            structured_actual_qualities
        )


        structured_selected_rank = int(
            np.where(
                structured_true_order
                ==
                structured_selected_index
            )[0][0]
        ) + 1


        structured_true_best_quality = float(
            structured_actual_qualities[
                structured_true_best_index
            ]
        )


        structured_actual_quality = float(
            structured_metrics[
                "quality_score"
            ]
        )


        structured_regret = (
            structured_actual_quality
            -
            structured_true_best_quality
        )


        structured_exact_hit = (
            structured_selected_index
            ==
            structured_true_best_index
        )


        structured_top3_hit = (
            structured_selected_rank
            <=
            3
        )


        structured_prediction_error = abs(

            structured_actual_quality
            -
            structured_recommendation[
                "predicted_quality"
            ]
        )


        structured_utilization = (
            calculate_utilization(
                *structured_point
            )
        )


        # Advance structured trajectory

        structured_state = (
            structured_new_state
        )


        # ====================================================
        # 3. SELECTIVE BO CONTROLLER
        # ====================================================

        selective_start = time.time()


        selective_structured = (
            structured_rf_recommendation(

                row=row,

                state=selective_state,

                seed=decision_seed,
            )
        )


        selective_prediction_time += (
            time.time()
            -
            selective_start
        )


        selective_margin = float(
            selective_structured[
                "prediction_margin"
            ]
        )


        bo_triggered = (
            selective_margin
            <=
            FROZEN_TRIGGER_MARGIN
        )


        if bo_triggered:

            total_bo_triggers += 1


            bo_start = time.time()


            bo_recommendation = run_warmstart_bo(

                row=row,

                state=selective_state,

                seed=decision_seed,
            )


            bo_runtime_total += (
                time.time()
                -
                bo_start
            )


            selective_point = (

                bo_recommendation[
                    "z_adj"
                ],

                bo_recommendation[
                    "theta_adj"
                ],

                bo_recommendation[
                    "locator_offset"
                ],
            )


            selective_predicted_quality = float(
                bo_recommendation[
                    "predicted_quality"
                ]
            )


            selective_capability_penalty = float(
                bo_recommendation[
                    "capability_penalty"
                ]
            )


            selective_source = (
                "warmstart_bo"
            )


            bo_best_was_initial = bool(
                bo_recommendation[
                    "best_was_initial"
                ]
            )


        else:


            selective_candidate = (
                selective_structured[
                    "candidate"
                ]
            )


            selective_point = (

                selective_candidate[
                    "z_adj"
                ],

                selective_candidate[
                    "theta_adj"
                ],

                selective_candidate[
                    "locator_offset"
                ],
            )


            selective_predicted_quality = float(
                selective_structured[
                    "predicted_quality"
                ]
            )


            selective_capability_penalty = (
                0.0
            )


            selective_source = (
                "structured_direct"
            )


            bo_best_was_initial = np.nan


        (
            selective_new_state,
            selective_metrics,
        ) = simulate_correction(

            state=selective_state,

            component_profile=(
                component_profile
            ),

            batch_disturbance=(
                batch_disturbance
            ),

            point=selective_point,
        )


        selective_actual_quality = float(
            selective_metrics[
                "quality_score"
            ]
        )


        selective_prediction_error = abs(

            selective_actual_quality
            -
            selective_predicted_quality
        )


        selective_utilization = (
            calculate_utilization(
                *selective_point
            )
        )


        # ----------------------------------------------------
        # Actual zero-action reference from selective state
        # ----------------------------------------------------

        (
            _,
            selective_zero_metrics,
        ) = simulate_zero(

            state=selective_state,

            component_profile=(
                component_profile
            ),

            batch_disturbance=(
                batch_disturbance
            ),
        )


        selective_zero_quality = float(
            selective_zero_metrics[
                "quality_score"
            ]
        )


        selective_beats_zero = (
            selective_actual_quality
            <
            selective_zero_quality
        )


        # Advance selective trajectory

        selective_state = (
            selective_new_state
        )


        # ====================================================
        # COMPONENT-LEVEL RECORD
        # ====================================================

        component_records.append(
            {

                # --------------------------------------------
                # IDENTIFICATION
                # --------------------------------------------

                "assembly_id":
                    assembly_id,

                "batch_id":
                    batch_id,

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
                # ZERO
                # --------------------------------------------

                "zero_actual_quality":
                    float(
                        zero_metrics[
                            "quality_score"
                        ]
                    ),


                # --------------------------------------------
                # STRUCTURED-20
                # --------------------------------------------

                "structured_predicted_quality":
                    float(
                        structured_recommendation[
                            "predicted_quality"
                        ]
                    ),

                "structured_actual_quality":
                    structured_actual_quality,

                "structured_prediction_abs_error":
                    structured_prediction_error,

                "structured_prediction_margin":
                    float(
                        structured_recommendation[
                            "prediction_margin"
                        ]
                    ),

                "structured_true_best_quality":
                    structured_true_best_quality,

                "structured_absolute_regret":
                    structured_regret,

                "structured_exact_best_hit":
                    structured_exact_hit,

                "structured_top3_hit":
                    structured_top3_hit,

                "structured_selected_true_rank":
                    structured_selected_rank,

                "structured_z_adj":
                    structured_point[
                        0
                    ],

                "structured_theta_adj":
                    structured_point[
                        1
                    ],

                "structured_locator_offset":
                    structured_point[
                        2
                    ],

                "structured_utilization":
                    structured_utilization,

                "structured_candidate_source":
                    structured_candidate[
                        "source"
                    ],


                # --------------------------------------------
                # SELECTIVE CONTROLLER
                # --------------------------------------------

                "selective_trigger_margin":
                    selective_margin,

                "bo_triggered":
                    bo_triggered,

                "selective_controller_source":
                    selective_source,

                "selective_predicted_quality":
                    selective_predicted_quality,

                "selective_actual_quality":
                    selective_actual_quality,

                "selective_prediction_abs_error":
                    selective_prediction_error,

                "selective_beats_zero":
                    selective_beats_zero,

                "selective_z_adj":
                    selective_point[
                        0
                    ],

                "selective_theta_adj":
                    selective_point[
                        1
                    ],

                "selective_locator_offset":
                    selective_point[
                        2
                    ],

                "selective_utilization":
                    selective_utilization,

                "selective_capability_penalty":
                    selective_capability_penalty,

                "bo_best_was_structured_initial":
                    bo_best_was_initial,


                # --------------------------------------------
                # QUALITY METRICS
                # --------------------------------------------

                "selective_mean_gap":
                    selective_metrics[
                        "mean_gap"
                    ],

                "selective_max_gap":
                    selective_metrics[
                        "max_gap"
                    ],

                "selective_parallelism":
                    selective_metrics[
                        "parallelism_error"
                    ],

                "selective_rms":
                    selective_metrics[
                        "rms_deviation"
                    ],
            }
        )


    # ========================================================
    # FINAL ASSEMBLY METRICS
    # ========================================================

    zero_final_metrics = (
        calculate_quality_metrics(
            zero_state
        )
    )


    structured_final_metrics = (
        calculate_quality_metrics(
            structured_state
        )
    )


    selective_final_metrics = (
        calculate_quality_metrics(
            selective_state
        )
    )


    zero_final = float(
        zero_final_metrics[
            "quality_score"
        ]
    )


    structured_final = float(
        structured_final_metrics[
            "quality_score"
        ]
    )


    selective_final = float(
        selective_final_metrics[
            "quality_score"
        ]
    )


    assembly_component_subset = [
        record
        for record
        in component_records
        if record[
            "assembly_id"
        ]
        ==
        assembly_id
    ]


    assembly_trigger_count = int(
        sum(
            1
            for record
            in assembly_component_subset
            if record[
                "bo_triggered"
            ]
        )
    )


    assembly_records.append(
        {

            "assembly_id":
                assembly_id,

            "batch_id":
                batch_id,

            "batch_condition":
                batch_condition,

            "zero_final_quality":
                zero_final,

            "structured20_final_quality":
                structured_final,

            "selective_bo_final_quality":
                selective_final,


            "structured20_beats_zero":
                structured_final
                <
                zero_final,

            "selective_bo_beats_zero":
                selective_final
                <
                zero_final,

            "selective_bo_beats_structured20":
                selective_final
                <
                structured_final,

            "structured20_beats_selective_bo":
                structured_final
                <
                selective_final,


            "structured20_improvement_vs_zero_percent":
                (
                    (
                        zero_final
                        -
                        structured_final
                    )
                    /
                    max(
                        abs(
                            zero_final
                        ),
                        1e-9,
                    )
                    *
                    100.0
                ),

            "selective_improvement_vs_zero_percent":
                (
                    (
                        zero_final
                        -
                        selective_final
                    )
                    /
                    max(
                        abs(
                            zero_final
                        ),
                        1e-9,
                    )
                    *
                    100.0
                ),


            "selective_minus_structured_quality":
                (
                    selective_final
                    -
                    structured_final
                ),


            "n_bo_triggers":
                assembly_trigger_count,

            "bo_trigger_rate_percent":
                (
                    assembly_trigger_count
                    /
                    N_COMPONENTS
                    *
                    100.0
                ),


            "selective_final_mean_gap":
                selective_final_metrics[
                    "mean_gap"
                ],

            "selective_final_max_gap":
                selective_final_metrics[
                    "max_gap"
                ],

            "selective_final_parallelism":
                selective_final_metrics[
                    "parallelism_error"
                ],

            "selective_final_rms":
                selective_final_metrics[
                    "rms_deviation"
                ],
        }
    )


    # ========================================================
    # CHECKPOINT
    # ========================================================

    if assembly_counter % 10 == 0:


        elapsed = (
            time.time()
            -
            experiment_start
        )


        print(
            f"Completed "
            f"{assembly_counter}/"
            f"{n_assemblies}"
            f" | BO triggers so far = "
            f"{total_bo_triggers}"
            f" | elapsed = "
            f"{elapsed / 60.0:.2f} min"
        )


        os.makedirs(
            "data/validation",
            exist_ok=True,
        )


        pd.DataFrame(
            component_records
        ).to_csv(
            COMPONENT_OUTPUT,
            index=False,
        )


        pd.DataFrame(
            assembly_records
        ).to_csv(
            ASSEMBLY_OUTPUT,
            index=False,
        )


# ============================================================
# RESULT DATAFRAMES
# ============================================================

component_result_df = pd.DataFrame(
    component_records
)


assembly_result_df = pd.DataFrame(
    assembly_records
)


# ============================================================
# FINAL VALIDATION CHECK
# ============================================================

if len(
    component_result_df
) != expected_decisions:

    raise ValueError(
        "\nFinal component result count is incorrect."
    )


if len(
    assembly_result_df
) != n_assemblies:

    raise ValueError(
        "\nFinal assembly result count is incorrect."
    )


# ============================================================
# SYSTEM-LEVEL METRICS
# ============================================================

zero_mean = (
    assembly_result_df[
        "zero_final_quality"
    ]
    .mean()
)


structured_mean = (
    assembly_result_df[
        "structured20_final_quality"
    ]
    .mean()
)


selective_mean = (
    assembly_result_df[
        "selective_bo_final_quality"
    ]
    .mean()
)


zero_median = (
    assembly_result_df[
        "zero_final_quality"
    ]
    .median()
)


structured_median = (
    assembly_result_df[
        "structured20_final_quality"
    ]
    .median()
)


selective_median = (
    assembly_result_df[
        "selective_bo_final_quality"
    ]
    .median()
)


zero_std = (
    assembly_result_df[
        "zero_final_quality"
    ]
    .std()
)


structured_std = (
    assembly_result_df[
        "structured20_final_quality"
    ]
    .std()
)


selective_std = (
    assembly_result_df[
        "selective_bo_final_quality"
    ]
    .std()
)


zero_p95 = (
    assembly_result_df[
        "zero_final_quality"
    ]
    .quantile(
        0.95
    )
)


structured_p95 = (
    assembly_result_df[
        "structured20_final_quality"
    ]
    .quantile(
        0.95
    )
)


selective_p95 = (
    assembly_result_df[
        "selective_bo_final_quality"
    ]
    .quantile(
        0.95
    )
)


structured_win_zero = (
    assembly_result_df[
        "structured20_beats_zero"
    ]
    .mean()
    *
    100.0
)


selective_win_zero = (
    assembly_result_df[
        "selective_bo_beats_zero"
    ]
    .mean()
    *
    100.0
)


selective_win_structured = (
    assembly_result_df[
        "selective_bo_beats_structured20"
    ]
    .mean()
    *
    100.0
)


structured_win_selective = (
    assembly_result_df[
        "structured20_beats_selective_bo"
    ]
    .mean()
    *
    100.0
)


mean_selective_minus_structured = (
    assembly_result_df[
        "selective_minus_structured_quality"
    ]
    .mean()
)


median_selective_minus_structured = (
    assembly_result_df[
        "selective_minus_structured_quality"
    ]
    .median()
)


# ============================================================
# CONTROLLER DIAGNOSTIC METRICS
# ============================================================

trigger_rate = (
    component_result_df[
        "bo_triggered"
    ]
    .mean()
    *
    100.0
)


assemblies_with_bo = (
    assembly_result_df[
        "n_bo_triggers"
    ]
    .gt(
        0
    )
    .mean()
    *
    100.0
)


structured_prediction_mae = (
    component_result_df[
        "structured_prediction_abs_error"
    ]
    .mean()
)


selective_prediction_mae = (
    component_result_df[
        "selective_prediction_abs_error"
    ]
    .mean()
)


structured_exact_hit = (
    component_result_df[
        "structured_exact_best_hit"
    ]
    .mean()
    *
    100.0
)


structured_top3_hit = (
    component_result_df[
        "structured_top3_hit"
    ]
    .mean()
    *
    100.0
)


structured_mean_regret = (
    component_result_df[
        "structured_absolute_regret"
    ]
    .mean()
)


structured_p95_regret = (
    component_result_df[
        "structured_absolute_regret"
    ]
    .quantile(
        0.95
    )
)


structured_mean_utilization = (
    component_result_df[
        "structured_utilization"
    ]
    .mean()
)


selective_mean_utilization = (
    component_result_df[
        "selective_utilization"
    ]
    .mean()
)


structured_near_limit = (
    component_result_df[
        "structured_utilization"
    ]
    .ge(
        0.90
    )
    .mean()
    *
    100.0
)


selective_near_limit = (
    component_result_df[
        "selective_utilization"
    ]
    .ge(
        0.90
    )
    .mean()
    *
    100.0
)


selective_component_beats_zero = (
    component_result_df[
        "selective_beats_zero"
    ]
    .mean()
    *
    100.0
)


# ============================================================
# STAGE ANALYSIS
# ============================================================

stage_rows = []


for component_index in range(
    1,
    N_COMPONENTS + 1,
):


    subset = component_result_df[
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

            "n_decisions":
                len(
                    subset
                ),

            "zero_mean_quality":
                subset[
                    "zero_actual_quality"
                ]
                .mean(),

            "structured_mean_quality":
                subset[
                    "structured_actual_quality"
                ]
                .mean(),

            "selective_mean_quality":
                subset[
                    "selective_actual_quality"
                ]
                .mean(),

            "structured_prediction_mae":
                subset[
                    "structured_prediction_abs_error"
                ]
                .mean(),

            "selective_prediction_mae":
                subset[
                    "selective_prediction_abs_error"
                ]
                .mean(),

            "structured_exact_best_percent":
                subset[
                    "structured_exact_best_hit"
                ]
                .mean()
                *
                100.0,

            "structured_top3_percent":
                subset[
                    "structured_top3_hit"
                ]
                .mean()
                *
                100.0,

            "structured_mean_regret":
                subset[
                    "structured_absolute_regret"
                ]
                .mean(),

            "bo_trigger_rate_percent":
                subset[
                    "bo_triggered"
                ]
                .mean()
                *
                100.0,

            "selective_beats_zero_percent":
                subset[
                    "selective_beats_zero"
                ]
                .mean()
                *
                100.0,

            "structured_mean_utilization":
                subset[
                    "structured_utilization"
                ]
                .mean(),

            "selective_mean_utilization":
                subset[
                    "selective_utilization"
                ]
                .mean(),
        }
    )


stage_df = pd.DataFrame(
    stage_rows
)


# ============================================================
# BATCH CONDITION ANALYSIS
# ============================================================

batch_rows = []


for condition in sorted(
    assembly_result_df[
        "batch_condition"
    ]
    .unique()
):


    subset = assembly_result_df[
        assembly_result_df[
            "batch_condition"
        ]
        ==
        condition
    ]


    component_subset = component_result_df[
        component_result_df[
            "batch_condition"
        ]
        ==
        condition
    ]


    batch_rows.append(
        {

            "batch_condition":
                condition,

            "n_assemblies":
                len(
                    subset
                ),

            "zero_mean_final_quality":
                subset[
                    "zero_final_quality"
                ]
                .mean(),

            "structured_mean_final_quality":
                subset[
                    "structured20_final_quality"
                ]
                .mean(),

            "selective_mean_final_quality":
                subset[
                    "selective_bo_final_quality"
                ]
                .mean(),

            "structured_win_vs_zero_percent":
                subset[
                    "structured20_beats_zero"
                ]
                .mean()
                *
                100.0,

            "selective_win_vs_zero_percent":
                subset[
                    "selective_bo_beats_zero"
                ]
                .mean()
                *
                100.0,

            "selective_win_vs_structured_percent":
                subset[
                    "selective_bo_beats_structured20"
                ]
                .mean()
                *
                100.0,

            "bo_trigger_rate_percent":
                component_subset[
                    "bo_triggered"
                ]
                .mean()
                *
                100.0,

            "structured_mean_utilization":
                component_subset[
                    "structured_utilization"
                ]
                .mean(),

            "selective_mean_utilization":
                component_subset[
                    "selective_utilization"
                ]
                .mean(),
        }
    )


batch_df = pd.DataFrame(
    batch_rows
)


# ============================================================
# SEVERITY ANALYSIS
# ============================================================

severity_rows = []


for severity in sorted(
    component_result_df[
        "severity"
    ]
    .unique()
):


    subset = component_result_df[
        component_result_df[
            "severity"
        ]
        ==
        severity
    ]


    severity_rows.append(
        {

            "severity":
                severity,

            "n_decisions":
                len(
                    subset
                ),

            "zero_mean_quality":
                subset[
                    "zero_actual_quality"
                ]
                .mean(),

            "structured_mean_quality":
                subset[
                    "structured_actual_quality"
                ]
                .mean(),

            "selective_mean_quality":
                subset[
                    "selective_actual_quality"
                ]
                .mean(),

            "structured_exact_best_percent":
                subset[
                    "structured_exact_best_hit"
                ]
                .mean()
                *
                100.0,

            "structured_mean_regret":
                subset[
                    "structured_absolute_regret"
                ]
                .mean(),

            "bo_trigger_rate_percent":
                subset[
                    "bo_triggered"
                ]
                .mean()
                *
                100.0,

            "selective_beats_zero_percent":
                subset[
                    "selective_beats_zero"
                ]
                .mean()
                *
                100.0,

            "selective_mean_utilization":
                subset[
                    "selective_utilization"
                ]
                .mean(),
        }
    )


severity_df = pd.DataFrame(
    severity_rows
)


# ============================================================
# SCENARIO ANALYSIS
# ============================================================

scenario_rows = []


for scenario in sorted(
    component_result_df[
        "scenario_type"
    ]
    .unique()
):


    subset = component_result_df[
        component_result_df[
            "scenario_type"
        ]
        ==
        scenario
    ]


    scenario_rows.append(
        {

            "scenario_type":
                scenario,

            "n_decisions":
                len(
                    subset
                ),

            "zero_mean_quality":
                subset[
                    "zero_actual_quality"
                ]
                .mean(),

            "structured_mean_quality":
                subset[
                    "structured_actual_quality"
                ]
                .mean(),

            "selective_mean_quality":
                subset[
                    "selective_actual_quality"
                ]
                .mean(),

            "structured_exact_best_percent":
                subset[
                    "structured_exact_best_hit"
                ]
                .mean()
                *
                100.0,

            "structured_mean_regret":
                subset[
                    "structured_absolute_regret"
                ]
                .mean(),

            "bo_trigger_rate_percent":
                subset[
                    "bo_triggered"
                ]
                .mean()
                *
                100.0,

            "selective_beats_zero_percent":
                subset[
                    "selective_beats_zero"
                ]
                .mean()
                *
                100.0,

            "selective_mean_utilization":
                subset[
                    "selective_utilization"
                ]
                .mean(),
        }
    )


scenario_df = pd.DataFrame(
    scenario_rows
)


# ============================================================
# RUNTIME
# ============================================================

runtime = (
    time.time()
    -
    experiment_start
)


avg_runtime_per_assembly = (
    runtime
    /
    n_assemblies
)


bo_calls_avoided_percent = (
    100.0
    -
    trigger_rate
)


# ============================================================
# SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    {

        "metric": [

            "validation_assemblies",

            "validation_component_decisions",

            "zero_mean_final_quality",

            "structured20_mean_final_quality",

            "selective_bo_mean_final_quality",

            "zero_median_final_quality",

            "structured20_median_final_quality",

            "selective_bo_median_final_quality",

            "zero_std_final_quality",

            "structured20_std_final_quality",

            "selective_bo_std_final_quality",

            "zero_p95_final_quality",

            "structured20_p95_final_quality",

            "selective_bo_p95_final_quality",

            "structured20_win_vs_zero_percent",

            "selective_bo_win_vs_zero_percent",

            "selective_bo_win_vs_structured20_percent",

            "structured20_win_vs_selective_bo_percent",

            "mean_selective_minus_structured_quality",

            "median_selective_minus_structured_quality",

            "structured_prediction_mae",

            "selective_prediction_mae",

            "structured_exact_best_hit_percent",

            "structured_top3_hit_percent",

            "structured_mean_regret",

            "structured_p95_regret",

            "selective_component_beats_zero_percent",

            "bo_trigger_rate_percent",

            "assemblies_with_at_least_one_bo_percent",

            "bo_calls_avoided_percent",

            "structured_mean_utilization",

            "selective_mean_utilization",

            "structured_near_limit_percent",

            "selective_near_limit_percent",

            "total_runtime_seconds",

            "average_runtime_per_assembly_seconds",

            "bo_runtime_seconds",

            "frozen_trigger_margin",
        ],

        "value": [

            n_assemblies,

            len(
                component_result_df
            ),

            zero_mean,

            structured_mean,

            selective_mean,

            zero_median,

            structured_median,

            selective_median,

            zero_std,

            structured_std,

            selective_std,

            zero_p95,

            structured_p95,

            selective_p95,

            structured_win_zero,

            selective_win_zero,

            selective_win_structured,

            structured_win_selective,

            mean_selective_minus_structured,

            median_selective_minus_structured,

            structured_prediction_mae,

            selective_prediction_mae,

            structured_exact_hit,

            structured_top3_hit,

            structured_mean_regret,

            structured_p95_regret,

            selective_component_beats_zero,

            trigger_rate,

            assemblies_with_bo,

            bo_calls_avoided_percent,

            structured_mean_utilization,

            selective_mean_utilization,

            structured_near_limit,

            selective_near_limit,

            runtime,

            avg_runtime_per_assembly,

            bo_runtime_total,

            FROZEN_TRIGGER_MARGIN,
        ],
    }
)


# ============================================================
# SAVE FINAL RESULTS
# ============================================================

os.makedirs(
    "data/validation",
    exist_ok=True,
)


os.makedirs(
    "results/validation",
    exist_ok=True,
)


component_result_df.to_csv(
    COMPONENT_OUTPUT,
    index=False,
)


assembly_result_df.to_csv(
    ASSEMBLY_OUTPUT,
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


batch_df.to_csv(
    BATCH_OUTPUT,
    index=False,
)


severity_df.to_csv(
    SEVERITY_OUTPUT,
    index=False,
)


scenario_df.to_csv(
    SCENARIO_OUTPUT,
    index=False,
)


# ============================================================
# TERMINAL RESULTS
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "FINAL INDEPENDENT VALIDATION RESULTS"
)

print(
    "============================================================"
)


print(
    f"\nAssemblies                         : "
    f"{n_assemblies}"
)


print(
    f"Sequential decisions               : "
    f"{len(component_result_df)}"
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "FINAL ASSEMBLY QUALITY"
)

print(
    "------------------------------------------------------------"
)


print(
    f"\nZERO mean                         : "
    f"{zero_mean:.4f}"
)


print(
    f"Structured-20 mean                : "
    f"{structured_mean:.4f}"
)


print(
    f"Selective BO mean                 : "
    f"{selective_mean:.4f}"
)


print(
    f"\nZERO median                       : "
    f"{zero_median:.4f}"
)


print(
    f"Structured-20 median              : "
    f"{structured_median:.4f}"
)


print(
    f"Selective BO median               : "
    f"{selective_median:.4f}"
)


print(
    f"\nZERO P95                          : "
    f"{zero_p95:.4f}"
)


print(
    f"Structured-20 P95                 : "
    f"{structured_p95:.4f}"
)


print(
    f"Selective BO P95                  : "
    f"{selective_p95:.4f}"
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "PAIRED WIN RATES"
)

print(
    "------------------------------------------------------------"
)


print(
    f"\nStructured-20 beats ZERO          : "
    f"{structured_win_zero:.2f}%"
)


print(
    f"Selective BO beats ZERO           : "
    f"{selective_win_zero:.2f}%"
)


print(
    f"Selective BO beats Structured-20  : "
    f"{selective_win_structured:.2f}%"
)


print(
    f"Structured-20 beats Selective BO  : "
    f"{structured_win_selective:.2f}%"
)


print(
    f"\nMean Selective - Structured       : "
    f"{mean_selective_minus_structured:.6f}"
)


print(
    f"Median Selective - Structured     : "
    f"{median_selective_minus_structured:.6f}"
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "SURROGATE GENERALIZATION"
)

print(
    "------------------------------------------------------------"
)


print(
    f"\nStructured prediction MAE         : "
    f"{structured_prediction_mae:.4f}"
)


print(
    f"Selective prediction MAE          : "
    f"{selective_prediction_mae:.4f}"
)


print(
    f"Structured exact-best hit         : "
    f"{structured_exact_hit:.2f}%"
)


print(
    f"Structured top-3 hit              : "
    f"{structured_top3_hit:.2f}%"
)


print(
    f"Structured mean regret            : "
    f"{structured_mean_regret:.4f}"
)


print(
    f"Structured P95 regret             : "
    f"{structured_p95_regret:.4f}"
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "SELECTIVE BO BEHAVIOUR"
)

print(
    "------------------------------------------------------------"
)


print(
    f"\nFrozen trigger margin             : "
    f"{FROZEN_TRIGGER_MARGIN:.12f}"
)


print(
    f"BO trigger rate                   : "
    f"{trigger_rate:.2f}%"
)


print(
    f"Assemblies with >=1 BO call       : "
    f"{assemblies_with_bo:.2f}%"
)


print(
    f"BO calls avoided                  : "
    f"{bo_calls_avoided_percent:.2f}%"
)


print(
    f"Selective component beats zero    : "
    f"{selective_component_beats_zero:.2f}%"
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "CAPABILITY"
)

print(
    "------------------------------------------------------------"
)


print(
    f"\nStructured mean utilization       : "
    f"{structured_mean_utilization:.3f}"
)


print(
    f"Selective mean utilization        : "
    f"{selective_mean_utilization:.3f}"
)


print(
    f"Structured >=90% capability       : "
    f"{structured_near_limit:.2f}%"
)


print(
    f"Selective >=90% capability        : "
    f"{selective_near_limit:.2f}%"
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "STAGE-WISE"
)

print(
    "------------------------------------------------------------\n"
)


print(
    stage_df
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
    "BATCH CONDITION"
)

print(
    "------------------------------------------------------------\n"
)


print(
    batch_df
    .round(
        4
    )
    .to_string(
        index=False
    )
)


# ============================================================
# FINAL SCIENTIFIC VERDICT
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "FINAL CONTROLLER VALIDATION VERDICT"
)

print(
    "============================================================"
)


structured_improvement = (
    (
        zero_mean
        -
        structured_mean
    )
    /
    max(
        abs(
            zero_mean
        ),
        1e-9,
    )
    *
    100.0
)


selective_improvement = (
    (
        zero_mean
        -
        selective_mean
    )
    /
    max(
        abs(
            zero_mean
        ),
        1e-9,
    )
    *
    100.0
)


print(
    f"\nStructured mean improvement vs zero: "
    f"{structured_improvement:.2f}%"
)


print(
    f"Selective mean improvement vs zero : "
    f"{selective_improvement:.2f}%"
)


if (
    structured_win_zero
    >=
    80.0
    and
    structured_mean
    <
    zero_mean
):


    print(
        "\nPASS A:"
    )


    print(
        "Structured-20 generalizes successfully to the "
        "completely independent validation population."
    )


else:


    print(
        "\nWARNING A:"
    )


    print(
        "Structured-20 does not generalize as strongly as "
        "development results suggested."
    )


if (
    selective_mean
    <
    structured_mean
    and
    selective_win_structured
    >
    50.0
):


    print(
        "\nPASS B:"
    )


    print(
        "Selective Bayesian refinement provides additional "
        "system-level value on independent data."
    )


    print(
        "\nFINAL CONTROLLER CANDIDATE:"
    )


    print(
        "Structured RF + frozen ambiguity gate + selective BO."
    )


elif (
    abs(
        selective_mean
        -
        structured_mean
    )
    /
    max(
        abs(
            structured_mean
        ),
        1e-9,
    )
    <=
    0.02
):


    print(
        "\nRESULT B:"
    )


    print(
        "Structured-20 and Selective BO remain essentially "
        "equivalent on independent validation data."
    )


    print(
        "\nINTERPRETATION:"
    )


    print(
        "Structured RF provides the dominant performance "
        "contribution; selective BO remains an optional "
        "refinement for ambiguous decisions."
    )


else:


    print(
        "\nRESULT C:"
    )


    print(
        "Selective BO does not improve the independent "
        "validation result relative to Structured-20."
    )


    print(
        "\nFINAL CONTROLLER CANDIDATE:"
    )


    print(
        "Freeze Structured-20 as the operational controller "
        "and retain BO as an investigated refinement / "
        "benchmark."
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
    f"\nTotal runtime                     : "
    f"{runtime / 60.0:.2f} min"
)


print(
    f"Average runtime / assembly        : "
    f"{avg_runtime_per_assembly:.2f} sec"
)


print(
    f"Total BO runtime                  : "
    f"{bo_runtime_total / 60.0:.2f} min"
)


print(
    "\nIMPORTANT:"
)


print(
    "No RF retraining, trigger recalibration, trust-region "
    "tuning, penalty tuning or correction-bound tuning was "
    "performed using this independent validation population."
)


print(
    "\nSaved:"
)


print(
    COMPONENT_OUTPUT
)


print(
    ASSEMBLY_OUTPUT
)


print(
    SUMMARY_OUTPUT
)


print(
    STAGE_OUTPUT
)


print(
    BATCH_OUTPUT
)


print(
    SEVERITY_OUTPUT
)


print(
    SCENARIO_OUTPUT
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.2 FINAL INDEPENDENT CONTROLLER VALIDATION COMPLETED"
)

print(
    "============================================================"
)