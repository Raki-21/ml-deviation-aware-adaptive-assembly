"""
V3.3 residual-hybrid closed-loop pilot evaluation.

Purpose
-------
Test whether the residual ML model actually improves sequential
assembly quality beyond deterministic LSQ.

Methods compared
----------------
1. ZERO
2. LSQ
3. LSQ + predicted delta_theta
4. LSQ + predicted delta_z + delta_theta
5. LSQ + predicted delta_z + delta_theta + delta_locator
6. Direct nonlinear numerical reference

Important
---------
The ML model was trained only on observable/controller-known features.

This script evaluates the already-trained pilot model without
retraining or tuning it.

Each controller evolves its own closed-loop sequential state.

All controllers receive the exact same incoming component deviations
and fixture disturbances.

The direct numerical method is a finite-budget reference and is not
claimed to be a proven global optimum.
"""

from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from scipy.optimize import (
    differential_evolution,
    lsq_linear,
)

from nonlinear_assembly_model import (
    create_initial_state,
    deviation_offset,
    deviation_tilt,
    deviation_bend,
    deviation_waviness,
    deviation_twist,
    deviation_local_bump,
    correction_profile_nonlinear,
    calculate_quality_metrics,
)

from v33_config import (
    N_COMPONENTS,
    PROFILE_LENGTH_MM,
    N_PROFILE_POINTS,
    Z_ADJ_LIMIT_MM,
    THETA_ADJ_LIMIT_DEG,
    LOCATOR_OFFSET_LIMIT_MM,
)


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(
    __file__
).resolve().parent


V33_ROOT = (
    SCRIPT_DIR
    .parent
)


MODEL_FILE = (
    V33_ROOT
    / "models"
    / "residual_learning_full"
    / "v33_residual_rf_full.joblib"
)


FEATURE_FILE = (
    V33_ROOT
    / "data"
    / "residual_learning_full"
    / "v33_residual_learning_full_observable_features.txt"
)


RESULTS_DIR = (
    V33_ROOT
    / "results"
    / "final_independent_validation_v33"
)


RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


COMPONENT_RESULT_FILE = (
    RESULTS_DIR
    / "v33_final_independent_component_results.csv"
)


ASSEMBLY_RESULT_FILE = (
    RESULTS_DIR
    / "v33_final_independent_assembly_results.csv"
)


SUMMARY_FILE = (
    RESULTS_DIR
    / "v33_final_independent_summary.csv"
)


# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTH = N_COMPONENTS

NONLINEARITY_STRENGTH = 1.0


# Use the exact untouched pilot validation population.
N_ASSEMBLIES = 300

VALIDATION_SEED = 20260912


FIXTURE_DRIFT_STD_MM = 0.015


# Same finite-budget nonlinear reference settings as dataset pilot.
REFERENCE_MAXITER = 8

REFERENCE_POPSIZE = 5

REFERENCE_TOL = 1e-4


# ============================================================
# METHODS
# ============================================================

METHOD_ZERO = "ZERO"

METHOD_LSQ = "LSQ"

METHOD_THETA = "LSQ_PLUS_ML_THETA"

METHOD_Z_THETA = "LSQ_PLUS_ML_Z_THETA"

METHOD_ALL = "LSQ_PLUS_ML_ALL"

METHOD_DIRECT = "DIRECT_NONLINEAR_REFERENCE"


METHODS = [
    METHOD_ZERO,
    METHOD_LSQ,
    METHOD_THETA,
    METHOD_Z_THETA,
    METHOD_ALL,
    METHOD_DIRECT,
]


# ============================================================
# PROFILE AXIS
# ============================================================

s = np.linspace(
    0.0,
    1.0,
    N_PROFILE_POINTS,
)


# ============================================================
# EXACT V3.2 LSQ BASIS
# ============================================================

Z_BASIS = np.ones_like(
    s,
    dtype=float,
)


THETA_BASIS = (
    PROFILE_LENGTH_MM
    *
    (
        s
        -
        0.5
    )
)


LOCATOR_SIGMA = 0.12


LOCATOR_BASIS = np.exp(
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
        LOCATOR_SIGMA
        ** 2
    )
)


CORRECTION_BASIS = np.column_stack(
    [
        Z_BASIS,
        THETA_BASIS,
        LOCATOR_BASIS,
    ]
)


THETA_TAN_LIMIT = float(
    np.tan(
        np.deg2rad(
            THETA_ADJ_LIMIT_DEG
        )
    )
)


LOWER_LSQ_BOUNDS = np.array(
    [
        -Z_ADJ_LIMIT_MM,
        -THETA_TAN_LIMIT,
        -LOCATOR_OFFSET_LIMIT_MM,
    ],
    dtype=float,
)


UPPER_LSQ_BOUNDS = np.array(
    [
        Z_ADJ_LIMIT_MM,
        THETA_TAN_LIMIT,
        LOCATOR_OFFSET_LIMIT_MM,
    ],
    dtype=float,
)


REFERENCE_BOUNDS = [
    (
        -Z_ADJ_LIMIT_MM,
        Z_ADJ_LIMIT_MM,
    ),
    (
        -THETA_ADJ_LIMIT_DEG,
        THETA_ADJ_LIMIT_DEG,
    ),
    (
        -LOCATOR_OFFSET_LIMIT_MM,
        LOCATOR_OFFSET_LIMIT_MM,
    ),
]


# ============================================================
# PROFILE FEATURE LOCATIONS
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


# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_FILE.exists():

    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_FILE}"
    )


if not FEATURE_FILE.exists():

    raise FileNotFoundError(
        f"\nFeature list not found:\n{FEATURE_FILE}"
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


if len(
    FEATURE_COLUMNS
) != 26:

    raise RuntimeError(
        f"Expected 26 features, got {len(FEATURE_COLUMNS)}."
    )


if getattr(
    model,
    "n_features_in_",
    None,
) != 26:

    raise RuntimeError(
        "Loaded model does not expect 26 input features."
    )


if getattr(
    model,
    "n_outputs_",
    None,
) != 3:

    raise RuntimeError(
        "Loaded model does not have 3 residual outputs."
    )


# ============================================================
# GEOMETRY GENERATION
# ============================================================

def generate_component_parameters(
    rng,
):

    return {
        "offset_mm":
            rng.uniform(
                -0.60,
                0.60,
            ),

        "tilt_deg":
            rng.uniform(
                -0.10,
                0.10,
            ),

        "bend_mm":
            rng.uniform(
                -0.45,
                0.45,
            ),

        "waviness_mm":
            rng.uniform(
                -0.30,
                0.30,
            ),

        "twist_mm":
            rng.uniform(
                -0.25,
                0.25,
            ),

        "local_bump_mm":
            rng.uniform(
                -0.30,
                0.30,
            ),
    }


def build_component_profile(
    parameters,
):

    return (
        deviation_offset(
            parameters[
                "offset_mm"
            ]
        )
        +
        deviation_tilt(
            parameters[
                "tilt_deg"
            ]
        )
        +
        deviation_bend(
            parameters[
                "bend_mm"
            ]
        )
        +
        deviation_waviness(
            parameters[
                "waviness_mm"
            ],
            waves=3,
        )
        +
        deviation_twist(
            parameters[
                "twist_mm"
            ]
        )
        +
        deviation_local_bump(
            parameters[
                "local_bump_mm"
            ]
        )
    )


def generate_validation_cases():
    """
    Regenerate exactly the pilot validation geometry using
    VALIDATION_SEED = 20260912.
    """

    rng = np.random.default_rng(
        VALIDATION_SEED
    )


    cases = []


    for assembly_index in range(
        1,
        N_ASSEMBLIES
        +
        1
    ):

        components = []


        for component_index in range(
            1,
            SEQUENCE_LENGTH
            +
            1
        ):

            parameters = (
                generate_component_parameters(
                    rng
                )
            )


            component_profile = (
                build_component_profile(
                    parameters
                )
            )


            fixture_drift = rng.normal(
                loc=0.0,
                scale=FIXTURE_DRIFT_STD_MM,
                size=N_PROFILE_POINTS,
            )


            components.append(
                {
                    "component_index":
                        component_index,

                    "component_profile":
                        component_profile,

                    "fixture_drift":
                        fixture_drift,
                }
            )


        cases.append(
            {
                "assembly_index":
                    assembly_index,

                "components":
                    components,
            }
        )


    return cases


# ============================================================
# EXACT LSQ
# ============================================================

def deterministic_lsq_correction(
    uncorrected_state,
):

    result = lsq_linear(
        CORRECTION_BASIS,
        -uncorrected_state,
        bounds=(
            LOWER_LSQ_BOUNDS,
            UPPER_LSQ_BOUNDS,
        ),
        method="trf",
        lsmr_tol="auto",
    )


    z_adj_mm = float(
        result.x[0]
    )


    tan_theta = float(
        result.x[1]
    )


    theta_adj_deg = float(
        np.rad2deg(
            np.arctan(
                tan_theta
            )
        )
    )


    theta_adj_deg = float(
        np.clip(
            theta_adj_deg,
            -THETA_ADJ_LIMIT_DEG,
            THETA_ADJ_LIMIT_DEG,
        )
    )


    locator_offset_mm = float(
        result.x[2]
    )


    return np.array(
        [
            z_adj_mm,
            theta_adj_deg,
            locator_offset_mm,
        ],
        dtype=float,
    )


# ============================================================
# UTILIZATION
# ============================================================

def calculate_utilization(
    point,
):

    return float(
        max(
            abs(
                point[0]
            )
            /
            Z_ADJ_LIMIT_MM,

            abs(
                point[1]
            )
            /
            THETA_ADJ_LIMIT_DEG,

            abs(
                point[2]
            )
            /
            LOCATOR_OFFSET_LIMIT_MM,
        )
    )


# ============================================================
# OBSERVABLE FEATURES
# ============================================================

def calculate_observable_features(
    pre_correction_state,
    component_profile,
    component_index,
    lsq_point,
):

    metrics = calculate_quality_metrics(
        pre_correction_state
    )


    signed_mean = float(
        np.mean(
            pre_correction_state
        )
    )


    signed_end_difference = float(
        pre_correction_state[-1]
        -
        pre_correction_state[0]
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


    component_rms = float(
        np.sqrt(
            np.mean(
                component_profile
                ** 2
            )
        )
    )


    component_parallelism = float(
        abs(
            component_profile[-1]
            -
            component_profile[0]
        )
    )


    feature_record = {
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

        "state_signed_end_difference":
            signed_end_difference,

        "state_estimated_angle_deg":
            estimated_angle_deg,

        "component_profile_rms":
            component_rms,

        "component_parallelism":
            component_parallelism,

        "lsq_z_adj_mm":
            float(
                lsq_point[0]
            ),

        "lsq_theta_adj_deg":
            float(
                lsq_point[1]
            ),

        "lsq_locator_offset_mm":
            float(
                lsq_point[2]
            ),

        "lsq_utilization":
            calculate_utilization(
                lsq_point
            ),

        "component_index":
            int(
                component_index
            ),
    }


    for index in PROFILE_SAMPLE_INDICES:

        feature_record[
            f"state_profile_p{index:03d}"
        ] = float(
            pre_correction_state[
                index
            ]
        )


    missing_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in feature_record
    ]


    if missing_features:

        raise RuntimeError(
            "Feature generation mismatch:\n"
            +
            "\n".join(
                missing_features
            )
        )


    return feature_record


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_residual(
    pre_correction_state,
    component_profile,
    component_index,
    lsq_point,
):

    feature_record = calculate_observable_features(
        pre_correction_state=pre_correction_state,
        component_profile=component_profile,
        component_index=component_index,
        lsq_point=lsq_point,
    )


    X = np.array(
        [
            [
                feature_record[
                    column
                ]
                for column in FEATURE_COLUMNS
            ]
        ],
        dtype=float,
    )


    prediction = model.predict(
        X
    )[0]


    return np.asarray(
        prediction,
        dtype=float,
    )


# ============================================================
# CLIPPING
# ============================================================

def clip_correction(
    point,
):

    point = np.asarray(
        point,
        dtype=float,
    ).copy()


    point[0] = np.clip(
        point[0],
        -Z_ADJ_LIMIT_MM,
        Z_ADJ_LIMIT_MM,
    )


    point[1] = np.clip(
        point[1],
        -THETA_ADJ_LIMIT_DEG,
        THETA_ADJ_LIMIT_DEG,
    )


    point[2] = np.clip(
        point[2],
        -LOCATOR_OFFSET_LIMIT_MM,
        LOCATOR_OFFSET_LIMIT_MM,
    )


    return point


# ============================================================
# APPLY TRUE NONLINEAR CORRECTION
# ============================================================

def apply_true_correction(
    pre_correction_state,
    point,
):

    correction = (
        correction_profile_nonlinear(
            previous_state=pre_correction_state,
            z_adj_mm=float(
                point[0]
            ),
            theta_adj_deg=float(
                point[1]
            ),
            locator_offset_mm=float(
                point[2]
            ),
            nonlinearity_strength=NONLINEARITY_STRENGTH,
        )
    )


    corrected_state = (
        pre_correction_state
        +
        correction
    )


    return corrected_state


# ============================================================
# DIRECT NONLINEAR REFERENCE
# ============================================================

def optimize_true_nonlinear_correction(
    pre_correction_state,
    seed,
):

    evaluation_counter = {
        "count":
            0
    }


    def objective(
        point,
    ):

        evaluation_counter[
            "count"
        ] += 1


        corrected_state = (
            apply_true_correction(
                pre_correction_state,
                point,
            )
        )


        return float(
            calculate_quality_metrics(
                corrected_state
            )[
                "quality_score"
            ]
        )


    result = differential_evolution(
        objective,
        bounds=REFERENCE_BOUNDS,
        seed=int(
            seed
        ),
        maxiter=REFERENCE_MAXITER,
        popsize=REFERENCE_POPSIZE,
        tol=REFERENCE_TOL,
        polish=True,
        updating="immediate",
        workers=1,
    )


    point = clip_correction(
        result.x
    )


    return (
        point,
        bool(
            result.success
        ),
        int(
            evaluation_counter[
                "count"
            ]
        ),
    )


# ============================================================
# RUN ONE METHOD
# ============================================================

def choose_correction(
    method,
    pre_correction_state,
    component_profile,
    component_index,
    reference_seed,
):

    if method == METHOD_ZERO:

        return (
            np.zeros(
                3,
                dtype=float,
            ),
            np.zeros(
                3,
                dtype=float,
            ),
            None,
            0,
        )


    lsq_point = (
        deterministic_lsq_correction(
            pre_correction_state
        )
    )


    if method == METHOD_LSQ:

        return (
            lsq_point,
            np.zeros(
                3,
                dtype=float,
            ),
            None,
            0,
        )


    if method == METHOD_DIRECT:

        (
            direct_point,
            reference_success,
            reference_evaluations,
        ) = optimize_true_nonlinear_correction(
            pre_correction_state,
            seed=reference_seed,
        )


        return (
            direct_point,
            direct_point
            -
            lsq_point,
            reference_success,
            reference_evaluations,
        )


    predicted_residual = (
        predict_residual(
            pre_correction_state=pre_correction_state,
            component_profile=component_profile,
            component_index=component_index,
            lsq_point=lsq_point,
        )
    )


    applied_residual = np.zeros(
        3,
        dtype=float,
    )


    if method == METHOD_THETA:

        applied_residual[1] = (
            predicted_residual[1]
        )


    elif method == METHOD_Z_THETA:

        applied_residual[0] = (
            predicted_residual[0]
        )

        applied_residual[1] = (
            predicted_residual[1]
        )


    elif method == METHOD_ALL:

        applied_residual[:] = (
            predicted_residual
        )


    else:

        raise ValueError(
            f"Unknown method: {method}"
        )


    final_point = clip_correction(
        lsq_point
        +
        applied_residual
    )


    actual_applied_residual = (
        final_point
        -
        lsq_point
    )


    return (
        final_point,
        actual_applied_residual,
        None,
        0,
    )


# ============================================================
# GENERATE IDENTICAL VALIDATION CASES
# ============================================================

validation_cases = (
    generate_validation_cases()
)


# ============================================================
# RUN CLOSED-LOOP EXPERIMENT
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "V3.3 FINAL INDEPENDENT VALIDATION"
)

print(
    "============================================================"
)


print(
    f"\nAssemblies           : {N_ASSEMBLIES}"
)

print(
    f"Sequence length      : {SEQUENCE_LENGTH}"
)

print(
    f"Decisions per method : "
    f"{N_ASSEMBLIES * SEQUENCE_LENGTH}"
)

print(
    f"Nonlinearity         : {NONLINEARITY_STRENGTH}"
)

print(
    f"Methods              : {len(METHODS)}"
)


component_rows = []

assembly_rows = []


experiment_start = time.time()


for assembly_case in validation_cases:

    assembly_index = (
        assembly_case[
            "assembly_index"
        ]
    )


    method_states = {
        method:
            create_initial_state()

        for method in METHODS
    }


    for component_record in (
        assembly_case[
            "components"
        ]
    ):

        component_index = (
            component_record[
                "component_index"
            ]
        )


        component_profile = (
            component_record[
                "component_profile"
            ]
        )


        fixture_drift = (
            component_record[
                "fixture_drift"
            ]
        )


        for method in METHODS:

            previous_state = (
                method_states[
                    method
                ]
            )


            pre_correction_state = (
                previous_state
                +
                component_profile
                +
                fixture_drift
            )


            pre_quality = float(
                calculate_quality_metrics(
                    pre_correction_state
                )[
                    "quality_score"
                ]
            )


            reference_seed = (
                VALIDATION_SEED
                +
                assembly_index
                *
                1000
                +
                component_index
            )


            (
                correction_point,
                applied_residual,
                reference_success,
                reference_evaluations,
            ) = choose_correction(
                method=method,
                pre_correction_state=pre_correction_state,
                component_profile=component_profile,
                component_index=component_index,
                reference_seed=reference_seed,
            )


            corrected_state = (
                apply_true_correction(
                    pre_correction_state,
                    correction_point,
                )
            )


            method_states[
                method
            ] = corrected_state


            post_metrics = (
                calculate_quality_metrics(
                    corrected_state
                )
            )


            component_rows.append(
                {
                    "assembly_index":
                        assembly_index,

                    "component_index":
                        component_index,

                    "method":
                        method,

                    "pre_quality":
                        pre_quality,

                    "post_quality":
                        float(
                            post_metrics[
                                "quality_score"
                            ]
                        ),

                    "z_adj_mm":
                        float(
                            correction_point[0]
                        ),

                    "theta_adj_deg":
                        float(
                            correction_point[1]
                        ),

                    "locator_offset_mm":
                        float(
                            correction_point[2]
                        ),

                    "applied_delta_z_mm":
                        float(
                            applied_residual[0]
                        ),

                    "applied_delta_theta_deg":
                        float(
                            applied_residual[1]
                        ),

                    "applied_delta_locator_mm":
                        float(
                            applied_residual[2]
                        ),

                    "utilization":
                        calculate_utilization(
                            correction_point
                        ),

                    "reference_success":
                        reference_success,

                    "reference_evaluations":
                        reference_evaluations,
                }
            )


    for method in METHODS:

        final_metrics = (
            calculate_quality_metrics(
                method_states[
                    method
                ]
            )
        )


        assembly_rows.append(
            {
                "assembly_index":
                    assembly_index,

                "method":
                    method,

                "final_quality":
                    float(
                        final_metrics[
                            "quality_score"
                        ]
                    ),

                "final_mean_gap":
                    float(
                        final_metrics[
                            "mean_gap"
                        ]
                    ),

                "final_max_gap":
                    float(
                        final_metrics[
                            "max_gap"
                        ]
                    ),

                "final_parallelism":
                    float(
                        final_metrics[
                            "parallelism_error"
                        ]
                    ),

                "final_rms":
                    float(
                        final_metrics[
                            "rms_deviation"
                        ]
                    ),
            }
        )


    print(
        f"Completed assembly "
        f"{assembly_index}/{N_ASSEMBLIES}"
    )


# ============================================================
# DATAFRAMES
# ============================================================

component_df = pd.DataFrame(
    component_rows
)


assembly_df = pd.DataFrame(
    assembly_rows
)


component_df.to_csv(
    COMPONENT_RESULT_FILE,
    index=False,
)


assembly_df.to_csv(
    ASSEMBLY_RESULT_FILE,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []


lsq_quality_by_assembly = (
    assembly_df[
        assembly_df[
            "method"
        ]
        ==
        METHOD_LSQ
    ]
    .set_index(
        "assembly_index"
    )[
        "final_quality"
    ]
)


for method in METHODS:

    method_df = (
        assembly_df[
            assembly_df[
                "method"
            ]
            ==
            method
        ]
        .copy()
    )


    mean_quality = float(
        method_df[
            "final_quality"
        ]
        .mean()
    )


    median_quality = float(
        method_df[
            "final_quality"
        ]
        .median()
    )


    p95_quality = float(
        method_df[
            "final_quality"
        ]
        .quantile(
            0.95
        )
    )


    if method == METHOD_LSQ:

        beats_lsq_percent = 0.0

        mean_difference_vs_lsq = 0.0

    else:

        method_quality = (
            method_df
            .set_index(
                "assembly_index"
            )[
                "final_quality"
            ]
        )


        paired_difference = (
            method_quality
            -
            lsq_quality_by_assembly
        )


        beats_lsq_percent = float(
            100.0
            *
            np.mean(
                paired_difference
                <
                -1e-12
            )
        )


        mean_difference_vs_lsq = float(
            paired_difference.mean()
        )


    method_component_df = (
        component_df[
            component_df[
                "method"
            ]
            ==
            method
        ]
    )


    mean_utilization = float(
        method_component_df[
            "utilization"
        ]
        .mean()
    )


    summary_rows.append(
        {
            "method":
                method,

            "mean_final_quality":
                mean_quality,

            "median_final_quality":
                median_quality,

            "p95_final_quality":
                p95_quality,

            "beats_lsq_percent":
                beats_lsq_percent,

            "mean_quality_difference_vs_lsq":
                mean_difference_vs_lsq,

            "mean_utilization":
                mean_utilization,
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)


summary_df.to_csv(
    SUMMARY_FILE,
    index=False,
)


# ============================================================
# PRINT RESULTS
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


for _, row in summary_df.iterrows():

    print(
        f"\n{row['method']}"
    )


    print(
        f"  Mean final quality       : "
        f"{row['mean_final_quality']:.6f}"
    )


    print(
        f"  Median final quality     : "
        f"{row['median_final_quality']:.6f}"
    )


    print(
        f"  P95 final quality        : "
        f"{row['p95_final_quality']:.6f}"
    )


    print(
        f"  Beats LSQ                : "
        f"{row['beats_lsq_percent']:.2f}%"
    )


    print(
        f"  Mean difference vs LSQ   : "
        f"{row['mean_quality_difference_vs_lsq']:.6f}"
    )


    print(
        f"  Mean utilization         : "
        f"{row['mean_utilization']:.4f}"
    )


# ============================================================
# IDENTIFY BEST ML HYBRID
# ============================================================

HYBRID_METHODS = [
    METHOD_THETA,
    METHOD_Z_THETA,
    METHOD_ALL,
]


hybrid_summary = (
    summary_df[
        summary_df[
            "method"
        ]
        .isin(
            HYBRID_METHODS
        )
    ]
    .sort_values(
        "mean_final_quality"
    )
)


best_hybrid_row = (
    hybrid_summary
    .iloc[0]
)


best_hybrid_method = (
    best_hybrid_row[
        "method"
    ]
)


best_hybrid_quality = float(
    best_hybrid_row[
        "mean_final_quality"
    ]
)


lsq_mean_quality = float(
    summary_df[
        summary_df[
            "method"
        ]
        ==
        METHOD_LSQ
    ][
        "mean_final_quality"
    ]
    .iloc[0]
)


direct_mean_quality = float(
    summary_df[
        summary_df[
            "method"
        ]
        ==
        METHOD_DIRECT
    ][
        "mean_final_quality"
    ]
    .iloc[0]
)


relative_hybrid_improvement = float(
    100.0
    *
    (
        lsq_mean_quality
        -
        best_hybrid_quality
    )
    /
    max(
        lsq_mean_quality,
        1e-12,
    )
)


# ============================================================
# GATE
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "FINAL INDEPENDENT VALIDATION GATE"
)

print(
    "============================================================"
)


print(
    f"\nLSQ mean quality           : "
    f"{lsq_mean_quality:.6f}"
)


print(
    f"Best ML hybrid             : "
    f"{best_hybrid_method}"
)


print(
    f"Best hybrid mean quality   : "
    f"{best_hybrid_quality:.6f}"
)


print(
    f"Relative improvement       : "
    f"{relative_hybrid_improvement:.2f}%"
)


print(
    f"Direct reference quality   : "
    f"{direct_mean_quality:.6f}"
)


if (
    best_hybrid_quality
    <
    lsq_mean_quality
    -
    1e-6
):

    print(
        "\nPASS:"
    )

    print(
        "At least one residual-ML configuration improves "
        "mean closed-loop quality beyond LSQ."
    )


else:

    print(
        "\nNO CLOSED-LOOP GAIN:"
    )

    print(
        "Residual prediction did not improve mean sequential "
        "quality beyond LSQ in this pilot."
    )


print(
    "\nImportant:"
)


print(
    "This is still a small pilot with only "
    f"{N_ASSEMBLIES} assemblies."
)


print(
    "No controller tuning or retraining was performed "
    "after seeing these results."
)


print(
    "A larger independent validation is required before "
    "making a final V3.3 claim."
)


elapsed = float(
    time.time()
    -
    experiment_start
)


print(
    f"\nRuntime: {elapsed:.1f} s"
)


print(
    "\nSaved component results:"
)

print(
    COMPONENT_RESULT_FILE
)


print(
    "\nSaved assembly results:"
)

print(
    ASSEMBLY_RESULT_FILE
)


print(
    "\nSaved summary:"
)

print(
    SUMMARY_FILE
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.3 FINAL INDEPENDENT VALIDATION COMPLETED"
)

print(
    "============================================================"
)

