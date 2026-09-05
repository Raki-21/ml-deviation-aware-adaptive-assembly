import os
import time
import joblib
import numpy as np
import pandas as pd

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

from deviation_engine import (
    generate_deviation_profile,
)

from batch_disturbance import (
    create_batch_condition,
)


# ============================================================
# VERSION 3.2
# FINAL SENSITIVITY + ROBUSTNESS VALIDATION
# ============================================================
#
# PURPOSE
#
# Stress-test the FROZEN Structured-20 RF controller after
# independent validation.
#
# NO architecture development occurs here.
#
#
# QUESTIONS
#
# 1. MULTI-SEED ROBUSTNESS
#
#    Does the controller remain effective across newly
#    generated independent synthetic populations?
#
#
# 2. DEVIATION-MAGNITUDE SENSITIVITY
#
#    Does the conclusion remain stable when component
#    deviations are:
#
#        0.8x nominal
#        1.0x nominal
#        1.2x nominal
#
#
# 3. CORRECTION-CAPABILITY SENSITIVITY
#
#    Does performance deteriorate gracefully when available
#    correction capability is reduced to:
#
#        100%
#         80%
#         60%
#
#
# 4. QUALITY-WEIGHT SENSITIVITY
#
#    Does the corrected assembly remain better than zero
#    under reasonable alternative quality-score weights?
#
#
# IMPORTANT
#
# The following remain frozen:
#
#   - profile-aware RF model
#   - ML feature definition
#   - Structured-20 architecture
#   - physical deviation mechanisms
#   - sequential state model
#
# ============================================================


# ============================================================
# PATHS
# ============================================================

MODEL_FILE = (
    "models/"
    "v3_2_profile_aware_quality_surrogate.joblib"
)

FEATURE_FILE = (
    "models/"
    "v3_2_profile_aware_surrogate_features.txt"
)


RESULT_OUTPUT = (
    "results/validation/"
    "v3_2_final_sensitivity_robustness_results.csv"
)

SEED_OUTPUT = (
    "results/validation/"
    "v3_2_final_robustness_by_seed.csv"
)

MAGNITUDE_OUTPUT = (
    "results/validation/"
    "v3_2_final_deviation_magnitude_sensitivity.csv"
)

CAPABILITY_OUTPUT = (
    "results/validation/"
    "v3_2_final_correction_capability_sensitivity.csv"
)

WEIGHT_OUTPUT = (
    "results/validation/"
    "v3_2_final_quality_weight_sensitivity.csv"
)

DETAIL_OUTPUT = (
    "data/validation/"
    "v3_2_final_sensitivity_robustness_assembly_results.csv"
)


# ============================================================
# FILE CHECK
# ============================================================

for path in [
    MODEL_FILE,
    FEATURE_FILE,
]:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nMissing required file:\n{path}"
        )


# ============================================================
# LOAD FROZEN RF
# ============================================================

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
# GLOBAL SETTINGS
# ============================================================

N_COMPONENTS = 5

N_STRUCTURED_CANDIDATES = 20


BATCH_CONDITIONS = [
    "normal",
    "offset_drift",
    "angular_drift",
    "fixture_drift",
    "high_variation",
    "disturbed",
]


# ============================================================
# ROBUSTNESS DESIGN
# ============================================================
#
# Three completely new populations.
#
# These seeds are different from:
#
#   training / development
#   20260824
#
#   controller development
#   20260825
#
#   independent validation
#   20260826
# ============================================================

ROBUSTNESS_SEEDS = [
    20260827,
    20260828,
    20260829,
]


# ============================================================
# EXPERIMENT SIZE
# ============================================================
#
# 5 independent batches / condition
# 5 assemblies / batch
#
# Therefore:
#
# 6 conditions x 5 batches x 5 assemblies
#
# = 150 assemblies per experimental configuration
#
# This keeps the final robustness experiment substantial while
# remaining computationally practical.
# ============================================================

BATCHES_PER_CONDITION = 5

ASSEMBLIES_PER_BATCH = 5


# ============================================================
# DEVIATION-MAGNITUDE FACTORS
# ============================================================

DEVIATION_FACTORS = [
    0.80,
    1.00,
    1.20,
]


# ============================================================
# CORRECTION-CAPABILITY FACTORS
# ============================================================

CAPABILITY_FACTORS = [
    1.00,
    0.80,
    0.60,
]


# ============================================================
# ORIGINAL FROZEN CORRECTION CAPABILITY
# ============================================================

ORIGINAL_Z_LIMIT = 2.5

ORIGINAL_THETA_LIMIT = 1.2

ORIGINAL_LOCATOR_LIMIT = 1.0


# ============================================================
# QUALITY-WEIGHT SETS
# ============================================================
#
# All sets sum to 1.0.
#
# The controller itself remains trained on the BASE quality
# definition.
#
# Alternative weights are used as an evaluation sensitivity
# check only.
# ============================================================

QUALITY_WEIGHT_SETS = {

    "BASE": {
        "mean_gap": 0.30,
        "max_gap": 0.30,
        "parallelism": 0.30,
        "rms": 0.10,
    },

    "GAP_FOCUSED": {
        "mean_gap": 0.40,
        "max_gap": 0.35,
        "parallelism": 0.15,
        "rms": 0.10,
    },

    "PARALLELISM_FOCUSED": {
        "mean_gap": 0.20,
        "max_gap": 0.20,
        "parallelism": 0.50,
        "rms": 0.10,
    },

    "BALANCED_METRICS": {
        "mean_gap": 0.25,
        "max_gap": 0.25,
        "parallelism": 0.25,
        "rms": 0.25,
    },
}


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
# CONTROLLED SEVERITY PATTERN
# ============================================================

SEVERITY_PATTERN = [
    "low",
    "medium",
    "medium",
    "high",
    "low",
    "medium",
    "high",
    "medium",
    "high",
    "extreme",
]


def choose_controlled_severity(
    component_counter,
    variation_multiplier,
):

    severity = SEVERITY_PATTERN[
        component_counter
        %
        len(
            SEVERITY_PATTERN
        )
    ]


    if variation_multiplier <= 1.0:

        return severity


    if severity == "low":

        return "medium"


    if severity == "medium":

        return "high"


    if severity == "high":

        if component_counter % 3 == 0:

            return "extreme"

        return "high"


    return "extreme"


# ============================================================
# QUALITY CALCULATION WITH ALTERNATIVE WEIGHTS
# ============================================================

def calculate_weighted_quality(
    state,
    weights,
):

    gap = np.abs(
        state
    )


    mean_gap = float(
        np.mean(
            gap
        )
    )


    max_gap = float(
        np.max(
            gap
        )
    )


    rms = float(
        np.sqrt(
            np.mean(
                state ** 2
            )
        )
    )


    parallelism = float(
        abs(
            state[-1]
            -
            state[0]
        )
    )


    score = (

        weights[
            "mean_gap"
        ]
        *
        mean_gap

        +

        weights[
            "max_gap"
        ]
        *
        max_gap

        +

        weights[
            "parallelism"
        ]
        *
        parallelism

        +

        weights[
            "rms"
        ]
        *
        rms
    )


    return float(
        score
    )


# ============================================================
# SCALE COMPONENT DEVIATION
# ============================================================

def scale_component(
    profile,
    features,
    factor,
):

    scaled_profile = (
        profile
        *
        factor
    )


    scaled_features = (
        features.copy()
    )


    scalable_columns = [
        "offset_mm",
        "tilt_deg",
        "bend_mm",
        "waviness_mm",
        "twist_mm",
        "local_bump_mm",
        "profile_mean_mm",
        "profile_min_mm",
        "profile_max_mm",
        "profile_rms_mm",
        "profile_parallelism_mm",
    ]


    for column in scalable_columns:

        scaled_features[
            column
        ] = (
            scaled_features[
                column
            ]
            *
            factor
        )


    return (
        scaled_profile,
        scaled_features,
    )


# ============================================================
# STATE FEATURES
# ============================================================

def calculate_state_features(
    state
):

    metrics = (
        calculate_quality_metrics(
            state
        )
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
# MODEL UTILIZATION
# ============================================================
#
# IMPORTANT:
#
# The RF was trained using the ORIGINAL capability
# normalization.
#
# Therefore this value must retain the original bounds even
# during tighter capability sensitivity experiments.
# ============================================================

def calculate_model_utilization(
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
            ORIGINAL_Z_LIMIT,

            abs(
                theta_adj
            )
            /
            ORIGINAL_THETA_LIMIT,

            abs(
                locator_offset
            )
            /
            ORIGINAL_LOCATOR_LIMIT,
        )
    )


# ============================================================
# ACTUAL CAPABILITY UTILIZATION
# ============================================================

def calculate_actual_utilization(
    z_adj,
    theta_adj,
    locator_offset,
    capability_factor,
):

    z_limit = (
        ORIGINAL_Z_LIMIT
        *
        capability_factor
    )


    theta_limit = (
        ORIGINAL_THETA_LIMIT
        *
        capability_factor
    )


    locator_limit = (
        ORIGINAL_LOCATOR_LIMIT
        *
        capability_factor
    )


    return float(
        max(

            abs(
                z_adj
            )
            /
            z_limit,

            abs(
                theta_adj
            )
            /
            theta_limit,

            abs(
                locator_offset
            )
            /
            locator_limit,
        )
    )


# ============================================================
# CLIP TO CURRENT CAPABILITY
# ============================================================

def clip_correction(
    z_adj,
    theta_adj,
    locator_offset,
    capability_factor,
):

    z_limit = (
        ORIGINAL_Z_LIMIT
        *
        capability_factor
    )


    theta_limit = (
        ORIGINAL_THETA_LIMIT
        *
        capability_factor
    )


    locator_limit = (
        ORIGINAL_LOCATOR_LIMIT
        *
        capability_factor
    )


    return (

        float(
            np.clip(
                z_adj,
                -z_limit,
                z_limit,
            )
        ),

        float(
            np.clip(
                theta_adj,
                -theta_limit,
                theta_limit,
            )
        ),

        float(
            np.clip(
                locator_offset,
                -locator_limit,
                locator_limit,
            )
        ),
    )


# ============================================================
# INFORMED CORRECTION
# ============================================================

def informed_correction(
    state,
    row,
    capability_factor,
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

        capability_factor,
    )


# ============================================================
# STRUCTURED-20 CANDIDATE GENERATOR
# ============================================================

def generate_structured_candidates(
    state,
    row,
    seed,
    capability_factor,
):

    rng = np.random.default_rng(
        seed
    )


    z_limit = (
        ORIGINAL_Z_LIMIT
        *
        capability_factor
    )


    theta_limit = (
        ORIGINAL_THETA_LIMIT
        *
        capability_factor
    )


    locator_limit = (
        ORIGINAL_LOCATOR_LIMIT
        *
        capability_factor
    )


    (
        informed_z,
        informed_theta,
        informed_locator,
    ) = informed_correction(

        state=state,

        row=row,

        capability_factor=(
            capability_factor
        ),
    )


    candidates = [

        {
            "z_adj":
                0.0,

            "theta_adj":
                0.0,

            "locator_offset":
                0.0,
        },

        {
            "z_adj":
                informed_z,

            "theta_adj":
                informed_theta,

            "locator_offset":
                informed_locator,
        },
    ]


    # --------------------------------------------------------
    # Eight local informed perturbations
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
                0.30
                *
                capability_factor,
            ),

            informed_theta
            +
            rng.normal(
                0.0,
                0.12
                *
                capability_factor,
            ),

            informed_locator
            +
            rng.normal(
                0.0,
                0.15
                *
                capability_factor,
            ),

            capability_factor,
        )


        candidates.append(
            {
                "z_adj":
                    z_adj,

                "theta_adj":
                    theta_adj,

                "locator_offset":
                    locator_offset,
            }
        )


    # --------------------------------------------------------
    # Six moderate random candidates
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
                            z_limit,

                            0.60
                            *
                            z_limit,
                        )
                    ),

                "theta_adj":
                    float(
                        rng.uniform(
                            -0.60
                            *
                            theta_limit,

                            0.60
                            *
                            theta_limit,
                        )
                    ),

                "locator_offset":
                    float(
                        rng.uniform(
                            -0.60
                            *
                            locator_limit,

                            0.60
                            *
                            locator_limit,
                        )
                    ),
            }
        )


    # --------------------------------------------------------
    # Four broad candidates
    # --------------------------------------------------------

    for _ in range(
        4
    ):


        candidates.append(
            {

                "z_adj":
                    float(
                        rng.uniform(
                            -z_limit,
                            z_limit,
                        )
                    ),

                "theta_adj":
                    float(
                        rng.uniform(
                            -theta_limit,
                            theta_limit,
                        )
                    ),

                "locator_offset":
                    float(
                        rng.uniform(
                            -locator_limit,
                            locator_limit,
                        )
                    ),
            }
        )


    if len(
        candidates
    ) != N_STRUCTURED_CANDIDATES:

        raise RuntimeError(
            "Structured candidate count is not 20."
        )


    return candidates


# ============================================================
# MODEL INPUT
# ============================================================

def build_model_input(
    row,
    state,
    candidates,
):

    state_features = (
        calculate_state_features(
            state
        )
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


        record = {

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
                calculate_model_utilization(

                    z_adj,

                    theta_adj,

                    locator_offset,
                ),

            "component_index":
                row[
                    "component_index"
                ],
        }


        records.append(
            record
        )


    X = pd.DataFrame(
        records
    )


    return X[
        FEATURE_COLUMNS
    ]


# ============================================================
# STRUCTURED RF RECOMMENDATION
# ============================================================

def recommend_correction(
    row,
    state,
    seed,
    capability_factor,
):


    candidates = (
        generate_structured_candidates(

            state=state,

            row=row,

            seed=seed,

            capability_factor=(
                capability_factor
            ),
        )
    )


    X = build_model_input(

        row=row,

        state=state,

        candidates=candidates,
    )


    predictions = (
        model.predict(
            X
        )
    )


    best_index = int(
        np.argmin(
            predictions
        )
    )


    selected = candidates[
        best_index
    ]


    return (

        selected[
            "z_adj"
        ],

        selected[
            "theta_adj"
        ],

        selected[
            "locator_offset"
        ],
    )


# ============================================================
# SIMULATE ONE CORRECTION
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


    return new_state


# ============================================================
# RUN ONE EXPERIMENT CONFIGURATION
# ============================================================

def run_configuration(
    experiment_seed,
    deviation_factor,
    capability_factor,
):


    rng = np.random.default_rng(
        experiment_seed
    )


    zero_final_states = []

    corrected_final_states = []


    records = []


    global_component_counter = 0

    local_assembly_id = 0


    # ========================================================
    # CONDITION LOOP
    # ========================================================

    for condition in BATCH_CONDITIONS:


        for local_batch in range(
            1,
            BATCHES_PER_CONDITION + 1,
        ):


            (
                batch_disturbance,
                batch_features,
            ) = create_batch_condition(

                condition=condition,

                rng=rng,
            )


            # =================================================
            # ASSEMBLY LOOP
            # =================================================

            for _ in range(
                ASSEMBLIES_PER_BATCH
            ):


                local_assembly_id += 1


                zero_state = (
                    create_initial_state()
                )


                corrected_state = (
                    create_initial_state()
                )


                # =============================================
                # COMPONENT LOOP
                # =============================================

                for component_index in range(
                    1,
                    N_COMPONENTS + 1,
                ):


                    global_component_counter += 1


                    severity = (
                        choose_controlled_severity(

                            global_component_counter,

                            batch_features[
                                "variation_multiplier"
                            ],
                        )
                    )


                    (
                        component_profile,
                        component_features,
                    ) = generate_deviation_profile(

                        severity=severity,

                        rng=rng,
                    )


                    (
                        component_profile,
                        component_features,
                    ) = scale_component(

                        component_profile,

                        component_features,

                        deviation_factor,
                    )


                    row = {

                        **component_features,

                        "batch_offset_bias_mm":
                            batch_features[
                                "batch_offset_bias_mm"
                            ],

                        "batch_angular_bias_deg":
                            batch_features[
                                "batch_angular_bias_deg"
                            ],

                        "fixture_drift_mm":
                            batch_features[
                                "fixture_drift_mm"
                            ],

                        "variation_multiplier":
                            batch_features[
                                "variation_multiplier"
                            ],

                        "component_profile_rms_mm":
                            component_features[
                                "profile_rms_mm"
                            ],

                        "component_parallelism_mm":
                            component_features[
                                "profile_parallelism_mm"
                            ],

                        "component_index":
                            component_index,
                    }


                    # -----------------------------------------
                    # ZERO TRAJECTORY
                    # -----------------------------------------

                    zero_state = (
                        update_assembly_state(

                            previous_state=(
                                zero_state
                            ),

                            component_deviation=(
                                component_profile
                            ),

                            fixture_drift=(
                                batch_disturbance
                            ),
                        )
                    )


                    # -----------------------------------------
                    # STRUCTURED-20 TRAJECTORY
                    # -----------------------------------------

                    decision_seed = (

                        experiment_seed

                        +

                        local_assembly_id
                        *
                        10

                        +

                        component_index
                    )


                    point = recommend_correction(

                        row=row,

                        state=corrected_state,

                        seed=decision_seed,

                        capability_factor=(
                            capability_factor
                        ),
                    )


                    corrected_state = (
                        simulate_correction(

                            state=(
                                corrected_state
                            ),

                            component_profile=(
                                component_profile
                            ),

                            batch_disturbance=(
                                batch_disturbance
                            ),

                            point=point,
                        )
                    )


                # =============================================
                # FINAL ASSEMBLY METRICS
                # =============================================

                zero_metrics = (
                    calculate_quality_metrics(
                        zero_state
                    )
                )


                corrected_metrics = (
                    calculate_quality_metrics(
                        corrected_state
                    )
                )


                zero_quality = float(
                    zero_metrics[
                        "quality_score"
                    ]
                )


                corrected_quality = float(
                    corrected_metrics[
                        "quality_score"
                    ]
                )


                records.append(
                    {

                        "seed":
                            experiment_seed,

                        "deviation_factor":
                            deviation_factor,

                        "capability_factor":
                            capability_factor,

                        "batch_condition":
                            condition,

                        "assembly_id":
                            local_assembly_id,

                        "zero_quality":
                            zero_quality,

                        "structured_quality":
                            corrected_quality,

                        "structured_beats_zero":
                            corrected_quality
                            <
                            zero_quality,

                        "improvement_percent":
                            (
                                (
                                    zero_quality
                                    -
                                    corrected_quality
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
                            ),
                    }
                )


                zero_final_states.append(
                    zero_state.copy()
                )


                corrected_final_states.append(
                    corrected_state.copy()
                )


    return (
        pd.DataFrame(
            records
        ),
        zero_final_states,
        corrected_final_states,
    )


# ============================================================
# MAIN EXPERIMENT
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "V3.2 FINAL SENSITIVITY + ROBUSTNESS"
)

print(
    "============================================================"
)


print(
    f"\nRobustness seeds     : "
    f"{ROBUSTNESS_SEEDS}"
)


print(
    f"Deviation factors    : "
    f"{DEVIATION_FACTORS}"
)


print(
    f"Capability factors   : "
    f"{CAPABILITY_FACTORS}"
)


experiment_start = time.time()


all_results = []

weight_records = []


configuration_counter = 0


# ============================================================
# A. MULTI-SEED + DEVIATION SENSITIVITY
# ============================================================
#
# Capability remains at 100%.
# ============================================================

for seed in ROBUSTNESS_SEEDS:


    for deviation_factor in (
        DEVIATION_FACTORS
    ):


        configuration_counter += 1


        print(
            f"\nRunning configuration "
            f"{configuration_counter}"
            f" | seed={seed}"
            f" | deviation={deviation_factor:.2f}"
            f" | capability=1.00"
        )


        (
            result_df,
            zero_states,
            corrected_states,
        ) = run_configuration(

            experiment_seed=seed,

            deviation_factor=(
                deviation_factor
            ),

            capability_factor=1.00,
        )


        all_results.append(
            result_df
        )


        # ----------------------------------------------------
        # Weight sensitivity is evaluated only for the
        # nominal deviation / nominal capability configuration.
        # ----------------------------------------------------

        if deviation_factor == 1.00:


            for weight_name, weights in (
                QUALITY_WEIGHT_SETS.items()
            ):


                zero_scores = []

                corrected_scores = []


                for (
                    zero_state,
                    corrected_state,
                ) in zip(
                    zero_states,
                    corrected_states,
                ):


                    zero_scores.append(
                        calculate_weighted_quality(

                            zero_state,

                            weights,
                        )
                    )


                    corrected_scores.append(
                        calculate_weighted_quality(

                            corrected_state,

                            weights,
                        )
                    )


                zero_scores = np.asarray(
                    zero_scores
                )


                corrected_scores = np.asarray(
                    corrected_scores
                )


                win_rate = float(
                    np.mean(
                        corrected_scores
                        <
                        zero_scores
                    )
                    *
                    100.0
                )


                mean_improvement = float(
                    (
                        np.mean(
                            zero_scores
                        )
                        -
                        np.mean(
                            corrected_scores
                        )
                    )
                    /
                    max(
                        abs(
                            np.mean(
                                zero_scores
                            )
                        ),
                        1e-9,
                    )
                    *
                    100.0
                )


                weight_records.append(
                    {

                        "seed":
                            seed,

                        "weight_set":
                            weight_name,

                        "zero_mean_quality":
                            float(
                                np.mean(
                                    zero_scores
                                )
                            ),

                        "structured_mean_quality":
                            float(
                                np.mean(
                                    corrected_scores
                                )
                            ),

                        "structured_win_vs_zero_percent":
                            win_rate,

                        "mean_improvement_percent":
                            mean_improvement,
                    }
                )


# ============================================================
# B. CORRECTION CAPABILITY SENSITIVITY
# ============================================================
#
# Use nominal deviation factor only.
#
# 100% has already been evaluated above.
#
# To avoid duplicate computation, only 80% and 60% are
# evaluated here.
# ============================================================

for seed in ROBUSTNESS_SEEDS:


    for capability_factor in [
        0.80,
        0.60,
    ]:


        configuration_counter += 1


        print(
            f"\nRunning configuration "
            f"{configuration_counter}"
            f" | seed={seed}"
            f" | deviation=1.00"
            f" | capability={capability_factor:.2f}"
        )


        (
            result_df,
            _,
            _,
        ) = run_configuration(

            experiment_seed=seed,

            deviation_factor=1.00,

            capability_factor=(
                capability_factor
            ),
        )


        all_results.append(
            result_df
        )


# ============================================================
# COMBINE
# ============================================================

combined_df = pd.concat(
    all_results,
    ignore_index=True,
)


weight_df = pd.DataFrame(
    weight_records
)


# ============================================================
# MULTI-SEED ROBUSTNESS SUMMARY
# ============================================================
#
# Nominal deviation + nominal capability only.
# ============================================================

nominal_df = combined_df[
    (
        combined_df[
            "deviation_factor"
        ]
        ==
        1.00
    )
    &
    (
        combined_df[
            "capability_factor"
        ]
        ==
        1.00
    )
]


seed_summary = (
    nominal_df
    .groupby(
        "seed"
    )
    .agg(

        assemblies=(
            "assembly_id",
            "count",
        ),

        zero_mean=(
            "zero_quality",
            "mean",
        ),

        structured_mean=(
            "structured_quality",
            "mean",
        ),

        structured_median=(
            "structured_quality",
            "median",
        ),

        structured_std=(
            "structured_quality",
            "std",
        ),

        structured_p95=(
            "structured_quality",
            lambda x:
                x.quantile(
                    0.95
                ),
        ),

        win_vs_zero_percent=(
            "structured_beats_zero",
            lambda x:
                x.mean()
                *
                100.0
        ),

        mean_improvement_percent=(
            "improvement_percent",
            "mean",
        ),
    )
    .reset_index()
)


# ============================================================
# DEVIATION-MAGNITUDE SUMMARY
# ============================================================

magnitude_df = combined_df[
    combined_df[
        "capability_factor"
    ]
    ==
    1.00
]


magnitude_summary = (
    magnitude_df
    .groupby(
        "deviation_factor"
    )
    .agg(

        assemblies=(
            "assembly_id",
            "count",
        ),

        zero_mean=(
            "zero_quality",
            "mean",
        ),

        structured_mean=(
            "structured_quality",
            "mean",
        ),

        structured_median=(
            "structured_quality",
            "median",
        ),

        structured_p95=(
            "structured_quality",
            lambda x:
                x.quantile(
                    0.95
                ),
        ),

        win_vs_zero_percent=(
            "structured_beats_zero",
            lambda x:
                x.mean()
                *
                100.0
        ),

        mean_improvement_percent=(
            "improvement_percent",
            "mean",
        ),
    )
    .reset_index()
)


# ============================================================
# CAPABILITY SUMMARY
# ============================================================

capability_df = combined_df[
    combined_df[
        "deviation_factor"
    ]
    ==
    1.00
]


capability_summary = (
    capability_df
    .groupby(
        "capability_factor"
    )
    .agg(

        assemblies=(
            "assembly_id",
            "count",
        ),

        zero_mean=(
            "zero_quality",
            "mean",
        ),

        structured_mean=(
            "structured_quality",
            "mean",
        ),

        structured_median=(
            "structured_quality",
            "median",
        ),

        structured_p95=(
            "structured_quality",
            lambda x:
                x.quantile(
                    0.95
                ),
        ),

        win_vs_zero_percent=(
            "structured_beats_zero",
            lambda x:
                x.mean()
                *
                100.0
        ),

        mean_improvement_percent=(
            "improvement_percent",
            "mean",
        ),
    )
    .reset_index()
)


# ============================================================
# QUALITY-WEIGHT SUMMARY
# ============================================================

weight_summary = (
    weight_df
    .groupby(
        "weight_set"
    )
    .agg(

        seeds=(
            "seed",
            "nunique",
        ),

        zero_mean_quality=(
            "zero_mean_quality",
            "mean",
        ),

        structured_mean_quality=(
            "structured_mean_quality",
            "mean",
        ),

        mean_win_vs_zero_percent=(
            "structured_win_vs_zero_percent",
            "mean",
        ),

        mean_improvement_percent=(
            "mean_improvement_percent",
            "mean",
        ),

        min_win_vs_zero_percent=(
            "structured_win_vs_zero_percent",
            "min",
        ),
    )
    .reset_index()
)


# ============================================================
# MASTER SUMMARY
# ============================================================

robustness_min_win = float(
    seed_summary[
        "win_vs_zero_percent"
    ]
    .min()
)


robustness_min_improvement = float(
    seed_summary[
        "mean_improvement_percent"
    ]
    .min()
)


magnitude_min_win = float(
    magnitude_summary[
        "win_vs_zero_percent"
    ]
    .min()
)


capability_60_row = (
    capability_summary[
        capability_summary[
            "capability_factor"
        ]
        ==
        0.60
    ]
    .iloc[
        0
    ]
)


weight_min_win = float(
    weight_summary[
        "min_win_vs_zero_percent"
    ]
    .min()
)


master_summary = pd.DataFrame(
    {

        "metric": [

            "robustness_seeds",

            "nominal_assemblies_total",

            "minimum_seed_win_vs_zero_percent",

            "minimum_seed_mean_improvement_percent",

            "minimum_deviation_sensitivity_win_percent",

            "capability_60_percent_win_vs_zero",

            "capability_60_percent_mean_improvement",

            "minimum_quality_weight_win_percent",

        ],

        "value": [

            len(
                ROBUSTNESS_SEEDS
            ),

            len(
                nominal_df
            ),

            robustness_min_win,

            robustness_min_improvement,

            magnitude_min_win,

            float(
                capability_60_row[
                    "win_vs_zero_percent"
                ]
            ),

            float(
                capability_60_row[
                    "mean_improvement_percent"
                ]
            ),

            weight_min_win,
        ],
    }
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "results/validation",
    exist_ok=True,
)


os.makedirs(
    "data/validation",
    exist_ok=True,
)


combined_df.to_csv(
    DETAIL_OUTPUT,
    index=False,
)


master_summary.to_csv(
    RESULT_OUTPUT,
    index=False,
)


seed_summary.to_csv(
    SEED_OUTPUT,
    index=False,
)


magnitude_summary.to_csv(
    MAGNITUDE_OUTPUT,
    index=False,
)


capability_summary.to_csv(
    CAPABILITY_OUTPUT,
    index=False,
)


weight_summary.to_csv(
    WEIGHT_OUTPUT,
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
    "FINAL ROBUSTNESS RESULTS"
)

print(
    "============================================================"
)


print(
    "\nMULTI-SEED ROBUSTNESS"
)


print(
    seed_summary
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
    "DEVIATION-MAGNITUDE SENSITIVITY"
)


print(
    magnitude_summary
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
    "CORRECTION-CAPABILITY SENSITIVITY"
)


print(
    capability_summary
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
    "QUALITY-WEIGHT SENSITIVITY"
)


print(
    weight_summary
    .round(
        4
    )
    .to_string(
        index=False
    )
)


print(
    "\n"
    "============================================================"
)

print(
    "FINAL SENSITIVITY / ROBUSTNESS VERDICT"
)

print(
    "============================================================"
)


# ============================================================
# SCIENTIFIC VERDICT
# ============================================================

if (
    robustness_min_win
    >=
    90.0
    and
    magnitude_min_win
    >=
    90.0
    and
    weight_min_win
    >=
    90.0
):


    print(
        "\nPASS A:"
    )


    print(
        "The Structured-20 controller remains highly robust "
        "across independent seeds, deviation-magnitude "
        "variation and alternative quality definitions."
    )


else:


    print(
        "\nRESULT B:"
    )


    print(
        "Controller performance remains beneficial but shows "
        "meaningful sensitivity in at least one tested "
        "assumption."
    )


print(
    "\nCapability reduction result:"
)


print(
    f"At 60% correction capability, win vs zero = "
    f"{float(capability_60_row['win_vs_zero_percent']):.2f}%"
)


print(
    f"At 60% correction capability, mean improvement = "
    f"{float(capability_60_row['mean_improvement_percent']):.2f}%"
)


print(
    "\nIMPORTANT:"
)


print(
    "This experiment evaluates robustness of the frozen "
    "Structured-20 architecture. No model retraining or "
    "controller recalibration was performed."
)


print(
    "\nSaved:"
)


print(
    RESULT_OUTPUT
)


print(
    SEED_OUTPUT
)


print(
    MAGNITUDE_OUTPUT
)


print(
    CAPABILITY_OUTPUT
)


print(
    WEIGHT_OUTPUT
)


print(
    DETAIL_OUTPUT
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
    "V3.2 FINAL SENSITIVITY + ROBUSTNESS COMPLETED"
)

print(
    "============================================================"
)