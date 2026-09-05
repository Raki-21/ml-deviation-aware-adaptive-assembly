"""
V3.3 residual-learning pilot dataset generation.

Purpose
-------
Generate a small scientifically controlled pilot dataset for the
future hybrid controller:

    u_hybrid = u_LSQ + delta_u_ML

For every component decision:

1. Build the observable pre-correction assembly state.
2. Compute the exact V3.2-style bounded LSQ correction.
3. Apply that LSQ correction through the true nonlinear V3.3 model.
4. Numerically search the nonlinear simulator for a stronger
   reference correction.
5. Store the correction residual:

       delta_u = u_reference - u_LSQ

The ML model is NOT trained in this script.

Important
---------
- Primary sequence length is K = 10.
- Generator-privileged deviation amplitudes are NOT included in
  the ML feature table.
- The nonlinear numerical search is a finite-budget reference,
  not a proven global optimum.
- The sequential state is propagated using LSQ during this pilot.
  Closed-loop hybrid-state augmentation will be performed later
  if the residual-learning concept is validated.
"""

from pathlib import Path
import time

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
    update_assembly_state,
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

DATA_DIR = (
    V33_ROOT
    / "data"
    / "residual_learning_full"
)

RESULTS_DIR = (
    V33_ROOT
    / "results"
    / "residual_learning_full"
)

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# PILOT SETTINGS
# ============================================================

SEQUENCE_LENGTH = N_COMPONENTS

NONLINEARITY_STRENGTH = 1.0


# Small pilot first.
N_TRAIN_ASSEMBLIES = 300

N_VALIDATION_ASSEMBLIES = 60


TRAIN_SEED = 20260909

VALIDATION_SEED = 20260910


FIXTURE_DRIFT_STD_MM = 0.015


# Finite-budget nonlinear numerical reference.
#
# This is deliberately moderate for the pilot.
# It is NOT claimed to be the global optimum.

REFERENCE_MAXITER = 8

REFERENCE_POPSIZE = 5

REFERENCE_TOL = 1e-4


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
# OBSERVABLE FEATURE DEFINITIONS
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


OBSERVABLE_FEATURE_COLUMNS = [
    "state_mean_gap",
    "state_max_gap",
    "state_parallelism",
    "state_rms",
    "state_quality",
    "state_signed_mean",
    "state_signed_end_difference",
    "state_estimated_angle_deg",

    "component_profile_rms",
    "component_parallelism",

    "lsq_z_adj_mm",
    "lsq_theta_adj_deg",
    "lsq_locator_offset_mm",
    "lsq_utilization",

    "component_index",
]


for index in PROFILE_SAMPLE_INDICES:

    OBSERVABLE_FEATURE_COLUMNS.append(
        f"state_profile_p{index:03d}"
    )


# ============================================================
# COMPONENT GENERATION
# ============================================================

def generate_component_parameters(
    rng,
):
    """
    Generate one V3.3 pilot component.

    These hidden generator parameters are used only to create
    the simulated geometry.

    They are NOT included in the operational ML feature table.
    """

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
    """
    Construct component geometry from hidden generator parameters.
    """

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


# ============================================================
# EXACT DETERMINISTIC LSQ
# ============================================================

def deterministic_lsq_correction(
    uncorrected_state,
):
    """
    Exact continuation of the frozen V3.2 deterministic
    correction formulation.

    Solver variables:
        [z, tan(theta), locator_offset]
    """

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


    return (
        z_adj_mm,
        theta_adj_deg,
        locator_offset_mm,
        bool(
            result.success
        ),
    )


# ============================================================
# UTILIZATION
# ============================================================

def calculate_utilization(
    z_adj_mm,
    theta_adj_deg,
    locator_offset_mm,
):

    return float(
        max(
            abs(
                z_adj_mm
            )
            /
            Z_ADJ_LIMIT_MM,

            abs(
                theta_adj_deg
            )
            /
            THETA_ADJ_LIMIT_DEG,

            abs(
                locator_offset_mm
            )
            /
            LOCATOR_OFFSET_LIMIT_MM,
        )
    )


# ============================================================
# OBSERVABLE FEATURE EXTRACTION
# ============================================================

def calculate_observable_features(
    pre_correction_state,
    component_profile,
    component_index,
    lsq_point,
):
    """
    Extract only measurement-compatible or controller-known
    features.

    No hidden generator amplitudes are returned.
    """

    state_quality = (
        calculate_quality_metrics(
            pre_correction_state
        )
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
                state_quality[
                    "mean_gap"
                ]
            ),

        "state_max_gap":
            float(
                state_quality[
                    "max_gap"
                ]
            ),

        "state_parallelism":
            float(
                state_quality[
                    "parallelism_error"
                ]
            ),

        "state_rms":
            float(
                state_quality[
                    "rms_deviation"
                ]
            ),

        "state_quality":
            float(
                state_quality[
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
                lsq_point[0],
                lsq_point[1],
                lsq_point[2],
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


    return feature_record


# ============================================================
# TRUE NONLINEAR QUALITY FOR ONE CANDIDATE
# ============================================================

def evaluate_true_candidate_quality(
    pre_correction_state,
    point,
):
    """
    Apply one candidate correction through the true nonlinear
    V3.3 response and return resulting quality.
    """

    z_adj_mm = float(
        point[0]
    )

    theta_adj_deg = float(
        point[1]
    )

    locator_offset_mm = float(
        point[2]
    )


    true_correction = (
        correction_profile_nonlinear(
            previous_state=pre_correction_state,
            z_adj_mm=z_adj_mm,
            theta_adj_deg=theta_adj_deg,
            locator_offset_mm=locator_offset_mm,
            nonlinearity_strength=NONLINEARITY_STRENGTH,
        )
    )


    corrected_state = (
        pre_correction_state
        +
        true_correction
    )


    quality_score = float(
        calculate_quality_metrics(
            corrected_state
        )[
            "quality_score"
        ]
    )


    return (
        quality_score,
        corrected_state,
    )


# ============================================================
# NONLINEAR NUMERICAL REFERENCE
# ============================================================

def optimize_true_nonlinear_correction(
    pre_correction_state,
    seed,
):
    """
    Finite-budget differential-evolution reference.

    This is a strong numerical target for the pilot, but it is
    not described as a guaranteed global optimum.
    """

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


        quality_score, _ = (
            evaluate_true_candidate_quality(
                pre_correction_state,
                point,
            )
        )


        return quality_score


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


    point = np.asarray(
        result.x,
        dtype=float,
    )


    final_quality, final_state = (
        evaluate_true_candidate_quality(
            pre_correction_state,
            point,
        )
    )


    return {
        "z_adj_mm":
            float(
                point[0]
            ),

        "theta_adj_deg":
            float(
                point[1]
            ),

        "locator_offset_mm":
            float(
                point[2]
            ),

        "quality_score":
            final_quality,

        "corrected_state":
            final_state,

        "evaluations":
            int(
                evaluation_counter[
                    "count"
                ]
            ),

        "success":
            bool(
                result.success
            ),
    }


# ============================================================
# GENERATE ONE SPLIT
# ============================================================

def generate_split(
    split_name,
    n_assemblies,
    seed,
):
    """
    Generate one assembly-disjoint split.

    Important:
    The sequential state follows the LSQ controller during this
    pilot. This prevents the target generator from silently
    changing the state distribution while residual learning is
    still being validated.
    """

    rng = np.random.default_rng(
        seed
    )


    records = []


    print(
        "\n"
        "------------------------------------------------------------"
    )

    print(
        f"Generating split: {split_name}"
    )

    print(
        f"Assemblies      : {n_assemblies}"
    )

    print(
        f"Sequence length : {SEQUENCE_LENGTH}"
    )

    print(
        "------------------------------------------------------------"
    )


    split_start_time = time.time()


    for assembly_index in range(
        1,
        n_assemblies
        +
        1
    ):

        lsq_state = (
            create_initial_state()
        )


        for component_index in range(
            1,
            SEQUENCE_LENGTH
            +
            1
        ):

            # ------------------------------------------------
            # Generate incoming geometry
            # ------------------------------------------------

            hidden_parameters = (
                generate_component_parameters(
                    rng
                )
            )


            component_profile = (
                build_component_profile(
                    hidden_parameters
                )
            )


            fixture_drift = rng.normal(
                loc=0.0,
                scale=FIXTURE_DRIFT_STD_MM,
                size=N_PROFILE_POINTS,
            )


            # ------------------------------------------------
            # Observable pre-correction state
            # ------------------------------------------------

            pre_correction_state = (
                lsq_state
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


            # ------------------------------------------------
            # Exact analytical LSQ
            # ------------------------------------------------

            (
                lsq_z,
                lsq_theta,
                lsq_locator,
                lsq_success,
            ) = deterministic_lsq_correction(
                pre_correction_state
            )


            lsq_point = np.array(
                [
                    lsq_z,
                    lsq_theta,
                    lsq_locator,
                ],
                dtype=float,
            )


            lsq_quality, lsq_corrected_state = (
                evaluate_true_candidate_quality(
                    pre_correction_state,
                    lsq_point,
                )
            )


            # ------------------------------------------------
            # Nonlinear numerical reference
            # ------------------------------------------------

            reference_seed = (
                seed
                +
                assembly_index
                *
                1000
                +
                component_index
            )


            reference = (
                optimize_true_nonlinear_correction(
                    pre_correction_state,
                    seed=reference_seed,
                )
            )


            reference_point = np.array(
                [
                    reference[
                        "z_adj_mm"
                    ],
                    reference[
                        "theta_adj_deg"
                    ],
                    reference[
                        "locator_offset_mm"
                    ],
                ],
                dtype=float,
            )


            # ------------------------------------------------
            # Residual target
            # ------------------------------------------------

            delta_point = (
                reference_point
                -
                lsq_point
            )


            # ------------------------------------------------
            # Observable ML features
            # ------------------------------------------------

            feature_record = (
                calculate_observable_features(
                    pre_correction_state=pre_correction_state,
                    component_profile=component_profile,
                    component_index=component_index,
                    lsq_point=lsq_point,
                )
            )


            quality_gain = float(
                lsq_quality
                -
                reference[
                    "quality_score"
                ]
            )


            relative_gain_percent = float(
                100.0
                *
                quality_gain
                /
                max(
                    lsq_quality,
                    1e-12,
                )
            )


            record = {
                "split":
                    split_name,

                "assembly_id":
                    (
                        f"{split_name.upper()}_"
                        f"{assembly_index:04d}"
                    ),

                "assembly_index":
                    assembly_index,

                "component_index":
                    component_index,

                "sequence_length":
                    SEQUENCE_LENGTH,

                "nonlinearity_strength":
                    NONLINEARITY_STRENGTH,

                **feature_record,

                "pre_correction_quality":
                    pre_quality,

                "lsq_true_quality":
                    lsq_quality,

                "reference_true_quality":
                    float(
                        reference[
                            "quality_score"
                        ]
                    ),

                "reference_gain_over_lsq":
                    quality_gain,

                "reference_relative_gain_percent":
                    relative_gain_percent,

                "reference_beats_lsq":
                    bool(
                        reference[
                            "quality_score"
                        ]
                        <
                        lsq_quality
                        -
                        1e-12
                    ),

                "reference_z_adj_mm":
                    float(
                        reference_point[0]
                    ),

                "reference_theta_adj_deg":
                    float(
                        reference_point[1]
                    ),

                "reference_locator_offset_mm":
                    float(
                        reference_point[2]
                    ),

                "target_delta_z_mm":
                    float(
                        delta_point[0]
                    ),

                "target_delta_theta_deg":
                    float(
                        delta_point[1]
                    ),

                "target_delta_locator_mm":
                    float(
                        delta_point[2]
                    ),

                "reference_utilization":
                    calculate_utilization(
                        reference_point[0],
                        reference_point[1],
                        reference_point[2],
                    ),

                "lsq_solver_success":
                    lsq_success,

                "reference_optimizer_success":
                    bool(
                        reference[
                            "success"
                        ]
                    ),

                "reference_evaluations":
                    int(
                        reference[
                            "evaluations"
                        ]
                    ),
            }


            records.append(
                record
            )


            # ------------------------------------------------
            # Pilot state propagation:
            # follow LSQ path.
            # ------------------------------------------------

            lsq_state = (
                lsq_corrected_state
            )


        if (
            assembly_index
            ==
            1
            or
            assembly_index
            %
            5
            ==
            0
            or
            assembly_index
            ==
            n_assemblies
        ):

            elapsed = (
                time.time()
                -
                split_start_time
            )


            print(
                f"{split_name}: "
                f"completed assembly "
                f"{assembly_index}/{n_assemblies} "
                f"| elapsed {elapsed:.1f} s"
            )


    return pd.DataFrame(
        records
    )


# ============================================================
# GENERATE PILOT DATA
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "V3.3 RESIDUAL-LEARNING PILOT DATASET"
)

print(
    "============================================================"
)

print(
    f"\nPrimary sequence length     : {SEQUENCE_LENGTH}"
)

print(
    f"Nonlinearity strength      : {NONLINEARITY_STRENGTH}"
)

print(
    f"Training assemblies        : {N_TRAIN_ASSEMBLIES}"
)

print(
    f"Validation assemblies      : {N_VALIDATION_ASSEMBLIES}"
)

print(
    f"Expected training decisions: "
    f"{N_TRAIN_ASSEMBLIES * SEQUENCE_LENGTH}"
)

print(
    f"Expected validation decisions: "
    f"{N_VALIDATION_ASSEMBLIES * SEQUENCE_LENGTH}"
)


overall_start_time = time.time()


train_df = generate_split(
    split_name="train",
    n_assemblies=N_TRAIN_ASSEMBLIES,
    seed=TRAIN_SEED,
)


validation_df = generate_split(
    split_name="validation",
    n_assemblies=N_VALIDATION_ASSEMBLIES,
    seed=VALIDATION_SEED,
)


full_df = pd.concat(
    [
        train_df,
        validation_df,
    ],
    ignore_index=True,
)


# ============================================================
# SAVE DATA
# ============================================================

train_file = (
    DATA_DIR
    /
    "v33_residual_learning_full_train.csv"
)


validation_file = (
    DATA_DIR
    /
    "v33_residual_learning_full_development.csv"
)


full_file = (
    DATA_DIR
    /
    "v33_residual_learning_full_all.csv"
)


feature_file = (
    DATA_DIR
    /
    "v33_residual_learning_full_observable_features.txt"
)


train_df.to_csv(
    train_file,
    index=False,
)


validation_df.to_csv(
    validation_file,
    index=False,
)


full_df.to_csv(
    full_file,
    index=False,
)


with open(
    feature_file,
    "w",
    encoding="utf-8",
) as file:

    for column in OBSERVABLE_FEATURE_COLUMNS:

        file.write(
            f"{column}\n"
        )


# ============================================================
# DATASET CHECKS
# ============================================================

train_ids = set(
    train_df[
        "assembly_id"
    ]
)


validation_ids = set(
    validation_df[
        "assembly_id"
    ]
)


assembly_overlap = (
    train_ids
    &
    validation_ids
)


expected_train_rows = (
    N_TRAIN_ASSEMBLIES
    *
    SEQUENCE_LENGTH
)


expected_validation_rows = (
    N_VALIDATION_ASSEMBLIES
    *
    SEQUENCE_LENGTH
)


# ============================================================
# SUMMARY
# ============================================================

mean_lsq_quality = float(
    full_df[
        "lsq_true_quality"
    ]
    .mean()
)


mean_reference_quality = float(
    full_df[
        "reference_true_quality"
    ]
    .mean()
)


reference_win_rate = float(
    100.0
    *
    full_df[
        "reference_beats_lsq"
    ]
    .mean()
)


mean_reference_gain = float(
    full_df[
        "reference_gain_over_lsq"
    ]
    .mean()
)


mean_relative_gain = float(
    full_df[
        "reference_relative_gain_percent"
    ]
    .mean()
)


mean_abs_delta_z = float(
    full_df[
        "target_delta_z_mm"
    ]
    .abs()
    .mean()
)


mean_abs_delta_theta = float(
    full_df[
        "target_delta_theta_deg"
    ]
    .abs()
    .mean()
)


mean_abs_delta_locator = float(
    full_df[
        "target_delta_locator_mm"
    ]
    .abs()
    .mean()
)


mean_reference_evaluations = float(
    full_df[
        "reference_evaluations"
    ]
    .mean()
)


lsq_success_rate = float(
    100.0
    *
    full_df[
        "lsq_solver_success"
    ]
    .mean()
)


reference_success_rate = float(
    100.0
    *
    full_df[
        "reference_optimizer_success"
    ]
    .mean()
)


elapsed_seconds = float(
    time.time()
    -
    overall_start_time
)


summary_df = pd.DataFrame(
    {
        "metric": [
            "sequence_length",
            "nonlinearity_strength",
            "training_assemblies",
            "validation_assemblies",
            "training_decisions",
            "validation_decisions",
            "train_validation_assembly_overlap",
            "observable_feature_count",
            "mean_lsq_true_quality",
            "mean_reference_true_quality",
            "reference_beats_lsq_percent",
            "mean_reference_gain_over_lsq",
            "mean_reference_relative_gain_percent",
            "mean_abs_target_delta_z_mm",
            "mean_abs_target_delta_theta_deg",
            "mean_abs_target_delta_locator_mm",
            "mean_reference_evaluations",
            "lsq_solver_success_percent",
            "reference_optimizer_success_percent",
            "generation_runtime_seconds",
        ],

        "value": [
            SEQUENCE_LENGTH,
            NONLINEARITY_STRENGTH,
            N_TRAIN_ASSEMBLIES,
            N_VALIDATION_ASSEMBLIES,
            len(
                train_df
            ),
            len(
                validation_df
            ),
            len(
                assembly_overlap
            ),
            len(
                OBSERVABLE_FEATURE_COLUMNS
            ),
            mean_lsq_quality,
            mean_reference_quality,
            reference_win_rate,
            mean_reference_gain,
            mean_relative_gain,
            mean_abs_delta_z,
            mean_abs_delta_theta,
            mean_abs_delta_locator,
            mean_reference_evaluations,
            lsq_success_rate,
            reference_success_rate,
            elapsed_seconds,
        ],
    }
)


summary_file = (
    RESULTS_DIR
    /
    "v33_residual_learning_full_summary.csv"
)


summary_df.to_csv(
    summary_file,
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
    "PILOT DATASET RESULTS"
)

print(
    "============================================================"
)


print(
    f"\nTrain rows                 : "
    f"{len(train_df)} / {expected_train_rows}"
)


print(
    f"Validation rows            : "
    f"{len(validation_df)} / {expected_validation_rows}"
)


print(
    f"Assembly overlap           : "
    f"{len(assembly_overlap)}"
)


print(
    f"Observable ML features     : "
    f"{len(OBSERVABLE_FEATURE_COLUMNS)}"
)


print(
    f"\nMean LSQ true quality      : "
    f"{mean_lsq_quality:.6f}"
)


print(
    f"Mean reference true quality: "
    f"{mean_reference_quality:.6f}"
)


print(
    f"Reference beats LSQ        : "
    f"{reference_win_rate:.2f}%"
)


print(
    f"Mean gain over LSQ         : "
    f"{mean_reference_gain:.6f}"
)


print(
    f"Mean relative gain         : "
    f"{mean_relative_gain:.2f}%"
)


print(
    "\nMean absolute residual targets:"
)


print(
    f"delta z                    : "
    f"{mean_abs_delta_z:.6f} mm"
)


print(
    f"delta theta                : "
    f"{mean_abs_delta_theta:.6f} deg"
)


print(
    f"delta locator              : "
    f"{mean_abs_delta_locator:.6f} mm"
)


print(
    f"\nLSQ solver success         : "
    f"{lsq_success_rate:.2f}%"
)


print(
    f"Reference optimizer success: "
    f"{reference_success_rate:.2f}%"
)


print(
    f"Mean reference evaluations : "
    f"{mean_reference_evaluations:.1f}"
)


print(
    f"Runtime                     : "
    f"{elapsed_seconds:.1f} s"
)


# ============================================================
# PILOT GATE
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "RESIDUAL-LEARNING PILOT GATE"
)

print(
    "============================================================"
)


if (
    len(
        assembly_overlap
    )
    >
    0
):

    print(
        "\nFAIL: train/validation assembly overlap detected."
    )


elif (
    mean_reference_gain
    <=
    1e-4
):

    print(
        "\nWEAK TARGET:"
    )

    print(
        "The nonlinear numerical reference provides almost "
        "no average improvement over LSQ."
    )

    print(
        "Residual learning is not yet justified."
    )


else:

    print(
        "\nPASS:"
    )

    print(
        "A measurable nonlinear correction residual exists."
    )

    print(
        "The pilot provides a valid target for testing "
        "residual ML."
    )


print(
    "\nImportant:"
)

print(
    "This does NOT yet prove ML can predict the residual."
)

print(
    "It only verifies that a useful residual correction "
    "target exists beyond analytical LSQ."
)


print(
    "\nSaved training data:"
)

print(
    train_file
)


print(
    "\nSaved validation data:"
)

print(
    validation_file
)


print(
    "\nSaved feature list:"
)

print(
    feature_file
)


print(
    "\nSaved summary:"
)

print(
    summary_file
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.3 RESIDUAL-LEARNING PILOT COMPLETED"
)

print(
    "============================================================"
)


