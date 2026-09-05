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
# STRUCTURED WARM-START BAYESIAN OPTIMIZATION PILOT
# ============================================================
#
# PURPOSE
#
# Previous diagnostics established:
#
# 1. Simulator-direct correction is highly effective.
#
# 2. Structured profile-aware RF controller:
#       - strongly beats zero correction
#       - strongly beats Random Search
#       - strongly beats old BO
#       - achieves very high online candidate ranking accuracy
#
# Therefore:
#
# The remaining weakness is associated primarily with the
# original BO search strategy / initialization.
#
#
# THIS EXPERIMENT
#
# Reintroduces Bayesian Optimization, but with:
#
#   - physically informed initialization
#   - training-aligned local trust region
#   - profile-aware RF surrogate
#   - capability-aware safeguard
#   - EXACTLY 20 surrogate evaluations per decision
#
#
# BUDGET
#
#   8 structured initial evaluations
#   +
#   12 Bayesian evaluations
#   =
#   20 total surrogate evaluations
#
# This matches the original BO evaluation budget.
#
#
# IMPORTANT
#
# This is a 30-assembly controlled pilot.
#
# If successful, the same architecture will be evaluated on
# all 100 saved pilot assemblies.
# ============================================================


# ============================================================
# PATHS
# ============================================================

COMPONENT_FILE = (
    "data/processed/"
    "v3_2_full_component_level_dataset.csv"
)

ORIGINAL_PILOT_FILE = (
    "data/processed/"
    "v3_2_ml_bo_pilot_assembly_results.csv"
)

STRUCTURED_RF_FILE = (
    "data/processed/"
    "v3_2_structured_rf_controller_assembly_results.csv"
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
    "data/processed/"
    "v3_2_warmstart_bo_pilot_component_results.csv"
)

ASSEMBLY_OUTPUT = (
    "data/processed/"
    "v3_2_warmstart_bo_pilot_assembly_results.csv"
)

SUMMARY_OUTPUT = (
    "results/tables/"
    "v3_2_warmstart_bo_pilot_summary.csv"
)

STAGE_OUTPUT = (
    "results/tables/"
    "v3_2_warmstart_bo_pilot_by_stage.csv"
)


# ============================================================
# FILE CHECK
# ============================================================

required_files = [
    COMPONENT_FILE,
    ORIGINAL_PILOT_FILE,
    STRUCTURED_RF_FILE,
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
        "\nComplete the previous V3.2 diagnostics first."
    )


# ============================================================
# SETTINGS
# ============================================================

N_PILOT_ASSEMBLIES = 30

N_COMPONENTS = 5

TOTAL_EVALUATIONS = 20

N_STRUCTURED_INITIAL = 8

N_BO_EVALUATIONS = (
    TOTAL_EVALUATIONS
    -
    N_STRUCTURED_INITIAL
)

RANDOM_SEED = 20260825


# ============================================================
# ORIGINAL CORRECTION CAPABILITY
# ============================================================

Z_LIMIT = 2.5

THETA_LIMIT = 1.2

LOCATOR_LIMIT = 1.0


# ============================================================
# TRAINING-ALIGNED LOCAL TRUST REGION
# ============================================================
#
# Closed-loop training used local informed perturbations:
#
#   sigma_z       = 0.30 mm
#   sigma_theta   = 0.12 deg
#   sigma_locator = 0.15 mm
#
# The local BO trust region uses approximately 3 sigma:
#
#   +/- 0.90 mm
#   +/- 0.36 deg
#   +/- 0.45 mm
#
# Therefore BO searches a region that is strongly represented
# in training instead of exploring the entire correction box
# almost blindly.
#
# These radii remain prototype assumptions and will require a
# later sensitivity analysis before final freeze.
# ============================================================

TRUST_Z_RADIUS = 0.90

TRUST_THETA_RADIUS = 0.36

TRUST_LOCATOR_RADIUS = 0.45


# ============================================================
# CAPABILITY SAFEGUARD
# ============================================================
#
# Only corrections above 80% utilization are penalized.
#
# The structured controller already showed that useful
# corrections generally require very low utilization.
#
# Therefore this is a safeguard rather than the primary search
# mechanism.
# ============================================================

BARRIER_START = 0.80

BARRIER_WEIGHT = 1.25

EFFORT_WEIGHT = 0.04


# ============================================================
# PROFILE FEATURES
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
    "V3.2 STRUCTURED WARM-START BO PILOT"
)

print(
    "============================================================"
)


component_df = pd.read_csv(
    COMPONENT_FILE
)


original_pilot_df = pd.read_csv(
    ORIGINAL_PILOT_FILE
)


structured_rf_df = pd.read_csv(
    STRUCTURED_RF_FILE
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
# SELECT BALANCED 30 ASSEMBLIES
# ============================================================

metadata = (
    structured_rf_df[
        [
            "assembly_id",
            "batch_condition",
        ]
    ]
    .drop_duplicates()
)


batch_conditions = sorted(
    metadata[
        "batch_condition"
    ]
    .unique()
)


selection_rng = np.random.default_rng(
    RANDOM_SEED
)


selected_assemblies = []


base_per_condition = (
    N_PILOT_ASSEMBLIES
    //
    len(
        batch_conditions
    )
)


remaining = (
    N_PILOT_ASSEMBLIES
    %
    len(
        batch_conditions
    )
)


for condition_index, condition in enumerate(
    batch_conditions
):

    ids = (
        metadata[
            metadata[
                "batch_condition"
            ]
            ==
            condition
        ][
            "assembly_id"
        ]
        .to_numpy()
    )


    n_select = (
        base_per_condition
        +
        (
            1
            if condition_index
            <
            remaining
            else 0
        )
    )


    chosen = selection_rng.choice(
        ids,
        size=n_select,
        replace=False,
    )


    selected_assemblies.extend(
        chosen.tolist()
    )


selected_assemblies = sorted(
    selected_assemblies
)


print(
    f"\nPilot assemblies           : "
    f"{len(selected_assemblies)}"
)

print(
    f"Components / assembly      : "
    f"{N_COMPONENTS}"
)

print(
    f"Structured initial points  : "
    f"{N_STRUCTURED_INITIAL}"
)

print(
    f"Bayesian evaluations       : "
    f"{N_BO_EVALUATIONS}"
)

print(
    f"Total evaluations/decision : "
    f"{TOTAL_EVALUATIONS}"
)


# ============================================================
# COMPONENT RECONSTRUCTION
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
# BATCH DISTURBANCE
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
# CLIP
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
# MODEL INPUT FOR ONE CORRECTION
# ============================================================

def build_model_input(
    row,
    state,
    z_adj,
    theta_adj,
    locator_offset,
):

    features = calculate_state_features(
        state
    )


    features.update(
        {

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
        [
            features
        ]
    )


    return X[
        FEATURE_COLUMNS
    ]


# ============================================================
# SURROGATE OBJECTIVE
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


    X = build_model_input(
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
# STRUCTURED INITIAL CANDIDATES
# ============================================================

def generate_initial_candidates(
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


    # Six training-aligned local perturbations.

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
# CHECK POINT INSIDE LOCAL SPACE
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
# WARM-START BAYESIAN OPTIMIZATION
# ============================================================

def run_warmstart_bo(
    row,
    state,
    seed,
):

    # --------------------------------------------------------
    # 1. Structured initial candidates
    # --------------------------------------------------------

    initial_candidates = (
        generate_initial_candidates(
            state=state,
            row=row,
            seed=seed,
        )
    )


    evaluated_points = []

    predicted_qualities = []

    penalties = []

    objectives = []


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


    # --------------------------------------------------------
    # Best structured starting point
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # 2. Local trust region centred on best structured point
    # --------------------------------------------------------

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
    # Warm-start GP with structured points that fall inside
    # the selected local trust region.
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
    # Safety:
    #
    # Best structured point is guaranteed to lie inside its own
    # trust region and therefore optimizer has at least one
    # observation.
    # --------------------------------------------------------

    # --------------------------------------------------------
    # 3. Bayesian refinement
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


    # --------------------------------------------------------
    # Exactly 20 surrogate evaluations
    # --------------------------------------------------------

    if len(
        evaluated_points
    ) != TOTAL_EVALUATIONS:

        raise RuntimeError(
            f"Expected {TOTAL_EVALUATIONS} evaluations, "
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

        "all_evaluated_points":
            evaluated_points,

        "all_objectives":
            objectives,
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
# STORAGE
# ============================================================

component_records = []

assembly_records = []


# ============================================================
# RUN PILOT
# ============================================================

experiment_start = time.time()


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


    state = create_initial_state()


    original_row = (
        original_pilot_df[
            original_pilot_df[
                "assembly_id"
            ]
            ==
            assembly_id
        ]
        .iloc[
            0
        ]
    )


    structured_row = (
        structured_rf_df[
            structured_rf_df[
                "assembly_id"
            ]
            ==
            assembly_id
        ]
        .iloc[
            0
        ]
    )


    batch_condition = (
        assembly_rows.iloc[
            0
        ][
            "batch_condition"
        ]
    )


    for _, row in assembly_rows.iterrows():


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
            RANDOM_SEED
            +
            int(
                assembly_id
            )
            *
            10
            +
            component_index
        )


        recommendation = run_warmstart_bo(

            row=row,

            state=state,

            seed=decision_seed,
        )


        selected_point = (

            recommendation[
                "z_adj"
            ],

            recommendation[
                "theta_adj"
            ],

            recommendation[
                "locator_offset"
            ],
        )


        # ----------------------------------------------------
        # Actual selected result
        # ----------------------------------------------------

        (
            selected_state,
            selected_metrics,
        ) = simulate_correction(

            state=state,

            component_profile=(
                component_profile
            ),

            batch_disturbance=(
                batch_disturbance
            ),

            point=selected_point,
        )


        # ----------------------------------------------------
        # Verify all 20 evaluated candidates with simulator
        # ----------------------------------------------------

        actual_qualities = []


        for point in recommendation[
            "all_evaluated_points"
        ]:


            (
                _,
                metrics,
            ) = simulate_correction(

                state=state,

                component_profile=(
                    component_profile
                ),

                batch_disturbance=(
                    batch_disturbance
                ),

                point=point,
            )


            actual_qualities.append(
                float(
                    metrics[
                        "quality_score"
                    ]
                )
            )


        actual_qualities = np.asarray(
            actual_qualities
        )


        selected_index = int(
            np.argmin(
                recommendation[
                    "all_objectives"
                ]
            )
        )


        true_best_index = int(
            np.argmin(
                actual_qualities
            )
        )


        true_order = np.argsort(
            actual_qualities
        )


        selected_true_rank = int(
            np.where(
                true_order
                ==
                selected_index
            )[0][0]
        ) + 1


        exact_best_hit = (
            selected_index
            ==
            true_best_index
        )


        top3_hit = (
            selected_true_rank
            <=
            3
        )


        true_best_quality = float(
            actual_qualities[
                true_best_index
            ]
        )


        selected_actual_quality = float(
            selected_metrics[
                "quality_score"
            ]
        )


        regret = (
            selected_actual_quality
            -
            true_best_quality
        )


        # ----------------------------------------------------
        # Zero candidate actual reference
        # ----------------------------------------------------

        (
            _,
            zero_metrics,
        ) = simulate_correction(

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


        zero_quality = float(
            zero_metrics[
                "quality_score"
            ]
        )


        component_records.append(
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

                "predicted_quality":
                    recommendation[
                        "predicted_quality"
                    ],

                "actual_quality":
                    selected_actual_quality,

                "prediction_absolute_error":
                    abs(
                        selected_actual_quality
                        -
                        recommendation[
                            "predicted_quality"
                        ]
                    ),

                "true_best_of_20_quality":
                    true_best_quality,

                "absolute_regret":
                    regret,

                "exact_best_hit":
                    exact_best_hit,

                "top3_hit":
                    top3_hit,

                "selected_true_rank":
                    selected_true_rank,

                "selected_beats_zero":
                    selected_actual_quality
                    <
                    zero_quality,

                "best_was_structured_initial":
                    recommendation[
                        "best_was_initial"
                    ],

                "z_adj":
                    recommendation[
                        "z_adj"
                    ],

                "theta_adj":
                    recommendation[
                        "theta_adj"
                    ],

                "locator_offset":
                    recommendation[
                        "locator_offset"
                    ],

                "correction_utilization":
                    recommendation[
                        "utilization"
                    ],

                "capability_penalty":
                    recommendation[
                        "capability_penalty"
                    ],

                "mean_gap":
                    selected_metrics[
                        "mean_gap"
                    ],

                "max_gap":
                    selected_metrics[
                        "max_gap"
                    ],

                "parallelism":
                    selected_metrics[
                        "parallelism_error"
                    ],

                "rms":
                    selected_metrics[
                        "rms_deviation"
                    ],
            }
        )


        # ----------------------------------------------------
        # Advance real sequential state
        # ----------------------------------------------------

        state = selected_state


    # ========================================================
    # FINAL ASSEMBLY
    # ========================================================

    final_metrics = calculate_quality_metrics(
        state
    )


    warmstart_final_quality = float(
        final_metrics[
            "quality_score"
        ]
    )


    zero_final = float(
        original_row[
            "zero_final_quality"
        ]
    )


    random_final = float(
        original_row[
            "random_final_quality"
        ]
    )


    old_bo_final = float(
        original_row[
            "bo_final_quality"
        ]
    )


    structured_final = float(
        structured_row[
            "structured_rf_final_quality"
        ]
    )


    assembly_records.append(
        {

            "assembly_id":
                assembly_id,

            "batch_condition":
                batch_condition,

            "zero_final_quality":
                zero_final,

            "random_final_quality":
                random_final,

            "old_bo_final_quality":
                old_bo_final,

            "structured_rf_final_quality":
                structured_final,

            "warmstart_bo_final_quality":
                warmstart_final_quality,

            "warmstart_beats_zero":
                warmstart_final_quality
                <
                zero_final,

            "warmstart_beats_random":
                warmstart_final_quality
                <
                random_final,

            "warmstart_beats_old_bo":
                warmstart_final_quality
                <
                old_bo_final,

            "warmstart_beats_structured_rf":
                warmstart_final_quality
                <
                structured_final,
        }
    )


    # --------------------------------------------------------
    # PROGRESS + CHECKPOINT
    # --------------------------------------------------------

    if assembly_counter % 5 == 0:


        elapsed = (
            time.time()
            -
            experiment_start
        )


        print(
            f"Completed "
            f"{assembly_counter}/"
            f"{len(selected_assemblies)}"
            f" | elapsed = "
            f"{elapsed / 60.0:.2f} min"
        )


        os.makedirs(
            "data/processed",
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
# DATAFRAMES
# ============================================================

component_result_df = pd.DataFrame(
    component_records
)


assembly_result_df = pd.DataFrame(
    assembly_records
)


# ============================================================
# FINAL METRICS
# ============================================================

zero_mean = (
    assembly_result_df[
        "zero_final_quality"
    ]
    .mean()
)


random_mean = (
    assembly_result_df[
        "random_final_quality"
    ]
    .mean()
)


old_bo_mean = (
    assembly_result_df[
        "old_bo_final_quality"
    ]
    .mean()
)


structured_mean = (
    assembly_result_df[
        "structured_rf_final_quality"
    ]
    .mean()
)


warmstart_mean = (
    assembly_result_df[
        "warmstart_bo_final_quality"
    ]
    .mean()
)


win_zero = (
    assembly_result_df[
        "warmstart_beats_zero"
    ]
    .mean()
    *
    100.0
)


win_random = (
    assembly_result_df[
        "warmstart_beats_random"
    ]
    .mean()
    *
    100.0
)


win_old_bo = (
    assembly_result_df[
        "warmstart_beats_old_bo"
    ]
    .mean()
    *
    100.0
)


win_structured = (
    assembly_result_df[
        "warmstart_beats_structured_rf"
    ]
    .mean()
    *
    100.0
)


mean_prediction_error = (
    component_result_df[
        "prediction_absolute_error"
    ]
    .mean()
)


exact_hit = (
    component_result_df[
        "exact_best_hit"
    ]
    .mean()
    *
    100.0
)


top3_hit = (
    component_result_df[
        "top3_hit"
    ]
    .mean()
    *
    100.0
)


mean_regret = (
    component_result_df[
        "absolute_regret"
    ]
    .mean()
)


p95_regret = (
    component_result_df[
        "absolute_regret"
    ]
    .quantile(
        0.95
    )
)


selected_beats_zero = (
    component_result_df[
        "selected_beats_zero"
    ]
    .mean()
    *
    100.0
)


bo_refinement_selected_rate = (
    ~component_result_df[
        "best_was_structured_initial"
    ]
).mean() * 100.0


mean_utilization = (
    component_result_df[
        "correction_utilization"
    ]
    .mean()
)


near_limit = (
    component_result_df[
        "correction_utilization"
    ]
    .ge(
        0.90
    )
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
    6,
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

            "mean_actual_quality":
                subset[
                    "actual_quality"
                ]
                .mean(),

            "mean_prediction_abs_error":
                subset[
                    "prediction_absolute_error"
                ]
                .mean(),

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

            "mean_regret":
                subset[
                    "absolute_regret"
                ]
                .mean(),

            "selected_beats_zero_percent":
                subset[
                    "selected_beats_zero"
                ]
                .mean()
                *
                100.0,

            "bo_refinement_selected_percent":
                (
                    ~subset[
                        "best_was_structured_initial"
                    ]
                )
                .mean()
                *
                100.0,

            "mean_utilization":
                subset[
                    "correction_utilization"
                ]
                .mean(),
        }
    )


stage_df = pd.DataFrame(
    stage_rows
)


# ============================================================
# SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    {

        "metric": [

            "zero_mean_final_quality",

            "random_mean_final_quality",

            "old_bo_mean_final_quality",

            "structured_rf_mean_final_quality",

            "warmstart_bo_mean_final_quality",

            "warmstart_win_vs_zero_percent",

            "warmstart_win_vs_random_percent",

            "warmstart_win_vs_old_bo_percent",

            "warmstart_win_vs_structured_rf_percent",

            "mean_prediction_absolute_error",

            "exact_best_hit_rate_percent",

            "top3_hit_rate_percent",

            "mean_regret",

            "p95_regret",

            "selected_beats_zero_rate_percent",

            "bo_refinement_selected_rate_percent",

            "mean_correction_utilization",

            "near_limit_rate_percent",
        ],

        "value": [

            zero_mean,

            random_mean,

            old_bo_mean,

            structured_mean,

            warmstart_mean,

            win_zero,

            win_random,

            win_old_bo,

            win_structured,

            mean_prediction_error,

            exact_hit,

            top3_hit,

            mean_regret,

            p95_regret,

            selected_beats_zero,

            bo_refinement_selected_rate,

            mean_utilization,

            near_limit,
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


# ============================================================
# PRINT RESULTS
# ============================================================

runtime = (
    time.time()
    -
    experiment_start
)


print(
    "\n"
    "============================================================"
)

print(
    "WARM-START BO PILOT RESULTS"
)

print(
    "============================================================"
)


print(
    f"\nZero mean final quality          : "
    f"{zero_mean:.4f}"
)


print(
    f"Random mean final quality        : "
    f"{random_mean:.4f}"
)


print(
    f"Old BO mean final quality        : "
    f"{old_bo_mean:.4f}"
)


print(
    f"Structured RF mean final quality : "
    f"{structured_mean:.4f}"
)


print(
    f"Warm-start BO mean final quality : "
    f"{warmstart_mean:.4f}"
)


print(
    f"\nWarm-start BO beats ZERO         : "
    f"{win_zero:.2f}%"
)


print(
    f"Warm-start BO beats RANDOM       : "
    f"{win_random:.2f}%"
)


print(
    f"Warm-start BO beats OLD BO       : "
    f"{win_old_bo:.2f}%"
)


print(
    f"Warm-start BO beats structured RF: "
    f"{win_structured:.2f}%"
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "ONLINE OPTIMIZER QUALITY"
)

print(
    "------------------------------------------------------------"
)


print(
    f"Mean RF prediction abs error     : "
    f"{mean_prediction_error:.4f}"
)


print(
    f"Exact-best-of-20 hit rate        : "
    f"{exact_hit:.2f}%"
)


print(
    f"Top-3-of-20 hit rate             : "
    f"{top3_hit:.2f}%"
)


print(
    f"Mean regret                      : "
    f"{mean_regret:.4f}"
)


print(
    f"P95 regret                       : "
    f"{p95_regret:.4f}"
)


print(
    f"Selected correction beats zero   : "
    f"{selected_beats_zero:.2f}%"
)


print(
    f"BO refinement selected           : "
    f"{bo_refinement_selected_rate:.2f}%"
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
    f"Mean utilization                 : "
    f"{mean_utilization:.3f}"
)


print(
    f">=90% capability                 : "
    f"{near_limit:.2f}%"
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "STAGE-WISE RESULTS"
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


# ============================================================
# SCIENTIFIC DECISION
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "WARM-START BO DIAGNOSIS"
)

print(
    "============================================================"
)


if (
    warmstart_mean
    <
    zero_mean
    and
    win_zero
    >=
    80.0
    and
    warmstart_mean
    <=
    old_bo_mean
):


    print(
        "\nRESULT A:"
    )


    print(
        "Structured warm-start Bayesian Optimization "
        "successfully restores strong sequential performance."
    )


    print(
        "\nINTERPRETATION:"
    )


    print(
        "The previous BO weakness was primarily caused by "
        "search-space mismatch / uninformed initialization."
    )


    print(
        "Engineering-informed initialization and a "
        "training-aligned local trust region allow BO to "
        "operate where the ML surrogate is reliable."
    )


    print(
        "\nNEXT STEP:"
    )


    print(
        "Run the same controller on the complete 100-assembly "
        "pilot and then begin robustness validation."
    )


elif (
    warmstart_mean
    <
    zero_mean
    and
    win_zero
    >=
    60.0
):


    print(
        "\nRESULT B:"
    )


    print(
        "Warm-start BO improves over zero correction, "
        "but performance remains less consistent than desired."
    )


    print(
        "\nNEXT STEP:"
    )


    print(
        "Calibrate the trust-region design using controlled "
        "sensitivity testing before full validation."
    )


else:


    print(
        "\nRESULT C:"
    )


    print(
        "Warm-start BO still does not provide reliable "
        "sequential improvement."
    )


    print(
        "\nNEXT STEP:"
    )


    print(
        "Do not scale this optimizer."
    )


    print(
        "Investigate whether the GP refinement itself adds "
        "value over the structured RF candidate controller."
    )


print(
    "\nIMPORTANT:"
)

print(
    "This remains a controlled 30-assembly pilot."
)


print(
    "The trust-region radii and capability penalty are "
    "prototype settings and require later sensitivity analysis."
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
    f"\nRuntime: "
    f"{runtime / 60.0:.2f} min"
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.2 WARM-START BO PILOT COMPLETED"
)

print(
    "============================================================"
)