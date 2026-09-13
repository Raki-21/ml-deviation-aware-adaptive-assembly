"""
Phase 8 dataset generation:
Objective-Aligned Residual Learning.

Purpose
-------
Generate independent training and development datasets for testing whether
machine-learning residual correction can provide additional quality
improvement on top of the objective-aligned Composite-Q controller.

For every component decision:

1. Build the observable pre-correction assembly state.
2. Compute the exact bounded LSQ correction used in V3.3.
3. Use that LSQ point as one starting point for the validated
   multi-start Composite-Q controller.
4. Compute the Composite-Q correction by directly minimizing the
   project composite quality objective.
5. Run a higher-budget finite Differential Evolution search.
6. Compare the actual quality obtained by Composite-Q and by the
   numerical reference.
7. Retain the better verified candidate.
8. Define the Phase 8 residual-learning target as:

       delta_u = u_best_reference - u_CompositeQ

If the higher-budget numerical search does not improve Composite-Q beyond
the defined numerical tolerance, Composite-Q remains the reference and the
residual target is exactly zero.

Important
---------
- Version 3.3 and post-freeze Phase 1-7 files are not modified.
- The validated V3.3 simulator is imported as a read-only library.
- The Composite-Q implementation follows the completed Phase 1 method:
  bounded multi-start L-BFGS-B using the actual composite quality score.
- The observable feature definition is kept identical to the V3.3
  residual-learning task.
- Hidden deviation-generator amplitudes are not used as ML features.
- The higher-budget numerical search is a finite numerical reference,
  not a claimed mathematical global optimum.
- Sequential training/development states follow the Composite-Q controller.
- The ML models are not trained in this script.
"""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import sys
import time

import numpy as np
import pandas as pd

from scipy.optimize import (
    differential_evolution,
    lsq_linear,
    minimize,
)


# ============================================================
# PHASE 8 CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phase8_config import (  # noqa: E402
    V33_SRC,
    TRAINING_DATA_DIR,
    DEVELOPMENT_DATA_DIR,
    TABLES_DIR,
    NONLINEARITY_STRENGTH,
    FIXTURE_DRIFT_STD_MM,
    N_TRAIN_ASSEMBLIES,
    N_DEVELOPMENT_ASSEMBLIES,
    TRAIN_SEED,
    DEVELOPMENT_SEED,
    REFERENCE_DE_MAXITER,
    REFERENCE_DE_POPSIZE,
    REFERENCE_DE_TOL,
    REFERENCE_DE_POLISH,
    REFERENCE_IMPROVEMENT_TOL,
    ensure_phase8_directories,
)


# ============================================================
# VALIDATED V3.3 SIMULATOR -- READ ONLY
# ============================================================

if str(V33_SRC) not in sys.path:
    sys.path.insert(0, str(V33_SRC))

from nonlinear_assembly_model import (  # noqa: E402
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

from v33_config import (  # noqa: E402
    N_COMPONENTS,
    PROFILE_LENGTH_MM,
    N_PROFILE_POINTS,
    Z_ADJ_LIMIT_MM,
    THETA_ADJ_LIMIT_DEG,
    LOCATOR_OFFSET_LIMIT_MM,
)


# ============================================================
# CORE SETTINGS
# ============================================================

SEQUENCE_LENGTH = N_COMPONENTS

s = np.linspace(
    0.0,
    1.0,
    N_PROFILE_POINTS,
)


# ============================================================
# EXACT V3.3 / PHASE 1 LSQ BASIS
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


# ============================================================
# CORRECTION BOUNDS
# ============================================================

CORRECTION_BOUNDS = [
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
# COMPOSITE-Q MULTI-START SET
# Exact structure used by the completed Phase 1 experiment.
# ============================================================

Zc = Z_ADJ_LIMIT_MM
Tc = THETA_ADJ_LIMIT_DEG
Lc = LOCATOR_OFFSET_LIMIT_MM

COMPOSITE_Q_GRID_STARTS = [
    (0.0, 0.0, 0.0),
    (-Zc, 0.0, 0.0),
    (Zc, 0.0, 0.0),
    (0.0, -Tc, 0.0),
    (0.0, Tc, 0.0),
    (0.0, 0.0, -Lc),
    (0.0, 0.0, Lc),
]


# ============================================================
# OBSERVABLE FEATURE DEFINITIONS
# Kept identical to the V3.3 residual-learning feature set.
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
# Same V3.3 deviation-generation ranges.
# ============================================================

def generate_component_parameters(rng):
    """
    Generate hidden geometric parameters for one simulated component.

    These quantities generate the geometry but are not supplied to the
    machine-learning models.
    """

    return {
        "offset_mm": rng.uniform(-0.60, 0.60),
        "tilt_deg": rng.uniform(-0.10, 0.10),
        "bend_mm": rng.uniform(-0.45, 0.45),
        "waviness_mm": rng.uniform(-0.30, 0.30),
        "twist_mm": rng.uniform(-0.25, 0.25),
        "local_bump_mm": rng.uniform(-0.30, 0.30),
    }


def build_component_profile(parameters):
    """Construct one component deviation profile."""

    return (
        deviation_offset(
            parameters["offset_mm"]
        )
        +
        deviation_tilt(
            parameters["tilt_deg"]
        )
        +
        deviation_bend(
            parameters["bend_mm"]
        )
        +
        deviation_waviness(
            parameters["waviness_mm"],
            waves=3,
        )
        +
        deviation_twist(
            parameters["twist_mm"]
        )
        +
        deviation_local_bump(
            parameters["local_bump_mm"]
        )
    )


# ============================================================
# ASSEMBLY CASE GENERATION
# ============================================================

def generate_cases(
    split_name,
    n_assemblies,
    seed,
):
    """
    Generate deterministic, assembly-disjoint cases for one split.

    Geometry is generated in the main process before parallel evaluation,
    so multiprocessing does not affect random-number generation.
    """

    rng = np.random.default_rng(seed)

    cases = []

    for assembly_index in range(
        1,
        n_assemblies + 1,
    ):
        components = []

        for component_index in range(
            1,
            SEQUENCE_LENGTH + 1,
        ):
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
                "split_name":
                    split_name,

                "split_seed":
                    seed,

                "assembly_index":
                    assembly_index,

                "components":
                    components,
            }
        )

    return cases


# ============================================================
# EXACT DETERMINISTIC LSQ
# ============================================================

def deterministic_lsq_correction(
    uncorrected_state,
):
    """
    Exact bounded LSQ formulation used by the validated V3.3 framework.

    Solver variables:
        [z, tan(theta), locator]
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

    return np.array(
        [
            z_adj_mm,
            theta_adj_deg,
            locator_offset_mm,
        ],
        dtype=float,
    )


# ============================================================
# CORRECTION UTILITIES
# ============================================================

def clip_correction(point):
    """Clip a correction to the validated V3.3 bounds."""

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


def calculate_utilization(point):
    """Maximum normalized correction utilization."""

    point = np.asarray(
        point,
        dtype=float,
    )

    return float(
        max(
            abs(point[0])
            /
            Z_ADJ_LIMIT_MM,

            abs(point[1])
            /
            THETA_ADJ_LIMIT_DEG,

            abs(point[2])
            /
            LOCATOR_OFFSET_LIMIT_MM,
        )
    )


# ============================================================
# TRUE V3.3 CORRECTION AND QUALITY
# ============================================================

def apply_true_correction(
    pre_correction_state,
    point,
):
    """Apply one correction through the validated nonlinear V3.3 model."""

    point = clip_correction(
        point
    )

    correction = (
        correction_profile_nonlinear(
            previous_state=pre_correction_state,
            z_adj_mm=float(point[0]),
            theta_adj_deg=float(point[1]),
            locator_offset_mm=float(point[2]),
            nonlinearity_strength=NONLINEARITY_STRENGTH,
        )
    )

    return (
        pre_correction_state
        +
        correction
    )


def evaluate_true_candidate_quality(
    pre_correction_state,
    point,
):
    """
    Evaluate one bounded correction using the same composite quality
    definition used throughout the validated V3.3 framework.
    """

    point = clip_correction(
        point
    )

    corrected_state = (
        apply_true_correction(
            pre_correction_state,
            point,
        )
    )

    metrics = (
        calculate_quality_metrics(
            corrected_state
        )
    )

    return (
        float(
            metrics["quality_score"]
        ),
        corrected_state,
    )


# ============================================================
# COMPOSITE-Q OPTIMIZER
# Exact Phase 1 architecture:
# origin + six face-centers + LSQ point, bounded L-BFGS-B.
# ============================================================

def objective_q(
    point,
    pre_correction_state,
):
    quality, _ = (
        evaluate_true_candidate_quality(
            pre_correction_state,
            point,
        )
    )

    return quality


def optimize_composite_q(
    pre_correction_state,
    lsq_point,
):
    """
    Multi-start bounded Composite-Q optimization.

    The raw LSQ point is always retained as an explicit candidate, so
    Composite-Q cannot be worse than that starting candidate solely due
    to a failed local-search run.
    """

    evaluation_counter = {
        "count": 0
    }

    def objective(point):
        evaluation_counter["count"] += 1

        return objective_q(
            point,
            pre_correction_state,
        )

    lsq_point = clip_correction(
        lsq_point
    )

    lsq_quality = objective(
        lsq_point
    )

    best_point = (
        lsq_point.copy()
    )

    best_quality = float(
        lsq_quality
    )

    best_success = True
    best_label = "LSQ_UNOPTIMIZED"

    starts = (
        list(
            COMPOSITE_Q_GRID_STARTS
        )
        +
        [
            tuple(
                lsq_point
            )
        ]
    )

    for start in starts:
        result = minimize(
            objective,
            x0=np.array(
                start,
                dtype=float,
            ),
            method="L-BFGS-B",
            bounds=CORRECTION_BOUNDS,
        )

        result_point = clip_correction(
            result.x
        )

        result_quality = objective_q(
            result_point,
            pre_correction_state,
        )

        if (
            result_quality
            <
            best_quality
        ):
            best_quality = float(
                result_quality
            )

            best_point = (
                result_point.copy()
            )

            best_success = bool(
                result.success
            )

            best_label = (
                "LSQ_REFINED"
                if np.allclose(
                    start,
                    lsq_point,
                )
                else
                "GRID"
            )

    return {
        "point":
            best_point,

        "quality_score":
            best_quality,

        "success":
            best_success,

        "evaluations":
            int(
                evaluation_counter["count"]
            ),

        "winning_start":
            best_label,
    }


# ============================================================
# HIGH-BUDGET FINITE NUMERICAL REFERENCE
# ============================================================

def optimize_high_budget_reference(
    pre_correction_state,
    seed,
):
    """
    Higher-budget finite Differential Evolution search.

    This is a numerical reference candidate only. It is not interpreted
    as a proven global optimum.
    """

    evaluation_counter = {
        "count": 0
    }

    def objective(point):
        evaluation_counter["count"] += 1

        quality, _ = (
            evaluate_true_candidate_quality(
                pre_correction_state,
                point,
            )
        )

        return quality

    result = differential_evolution(
        objective,
        bounds=CORRECTION_BOUNDS,
        seed=int(seed),
        maxiter=REFERENCE_DE_MAXITER,
        popsize=REFERENCE_DE_POPSIZE,
        tol=REFERENCE_DE_TOL,
        polish=REFERENCE_DE_POLISH,
        updating="immediate",
        workers=1,
    )

    point = clip_correction(
        result.x
    )

    final_quality, final_state = (
        evaluate_true_candidate_quality(
            pre_correction_state,
            point,
        )
    )

    return {
        "point":
            point,

        "quality_score":
            float(
                final_quality
            ),

        "corrected_state":
            final_state,

        "evaluations":
            int(
                evaluation_counter["count"]
            ),

        "success":
            bool(
                result.success
            ),
    }


# ============================================================
# OBSERVABLE FEATURE EXTRACTION
# Exact V3.3 feature definition.
# ============================================================

def calculate_observable_features(
    pre_correction_state,
    component_profile,
    component_index,
    lsq_point,
):
    """
    Extract the same observable/controller-known feature set used in
    the V3.3 residual-learning experiment.

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

    return feature_record


# ============================================================
# PROCESS ONE ASSEMBLY
# ============================================================

def process_assembly(
    case,
):
    """
    Generate Phase 8 target rows for one complete sequential assembly.

    State propagation follows Composite-Q, because Composite-Q is the
    deterministic base controller that Phase 8 residual learning is
    intended to augment.
    """

    split_name = (
        case["split_name"]
    )

    split_seed = int(
        case["split_seed"]
    )

    assembly_index = int(
        case["assembly_index"]
    )

    composite_q_state = (
        create_initial_state()
    )

    records = []

    for component in case[
        "components"
    ]:
        component_index = int(
            component[
                "component_index"
            ]
        )

        component_profile = (
            component[
                "component_profile"
            ]
        )

        fixture_drift = (
            component[
                "fixture_drift"
            ]
        )

        # ----------------------------------------------------
        # Observable pre-correction state along Composite-Q path
        # ----------------------------------------------------

        pre_correction_state = (
            composite_q_state
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

        # ----------------------------------------------------
        # LSQ point
        # Required by Composite-Q and retained as an existing
        # observable/controller-known feature.
        # ----------------------------------------------------

        lsq_point = (
            deterministic_lsq_correction(
                pre_correction_state
            )
        )

        lsq_quality, _ = (
            evaluate_true_candidate_quality(
                pre_correction_state,
                lsq_point,
            )
        )

        # ----------------------------------------------------
        # Objective-aligned Composite-Q base correction
        # ----------------------------------------------------

        composite_q = (
            optimize_composite_q(
                pre_correction_state,
                lsq_point,
            )
        )

        composite_q_point = (
            composite_q[
                "point"
            ]
        )

        composite_q_quality, composite_q_corrected_state = (
            evaluate_true_candidate_quality(
                pre_correction_state,
                composite_q_point,
            )
        )

        # ----------------------------------------------------
        # Higher-budget numerical reference candidate
        # ----------------------------------------------------

        reference_seed = (
            split_seed
            +
            assembly_index
            *
            1000
            +
            component_index
        )

        numerical_reference = (
            optimize_high_budget_reference(
                pre_correction_state,
                seed=reference_seed,
            )
        )

        numerical_point = (
            numerical_reference[
                "point"
            ]
        )

        numerical_quality = float(
            numerical_reference[
                "quality_score"
            ]
        )

        # ----------------------------------------------------
        # Best VERIFIED target
        #
        # Never assume the expensive numerical search is better.
        # It must actually beat Composite-Q.
        # ----------------------------------------------------

        numerical_beats_composite_q = bool(
            numerical_quality
            <
            composite_q_quality
            -
            REFERENCE_IMPROVEMENT_TOL
        )

        if numerical_beats_composite_q:
            best_reference_point = (
                numerical_point.copy()
            )

            best_reference_quality = (
                numerical_quality
            )

            best_reference_source = (
                "HIGH_BUDGET_DE"
            )

        else:
            best_reference_point = (
                composite_q_point.copy()
            )

            best_reference_quality = (
                composite_q_quality
            )

            best_reference_source = (
                "COMPOSITE_Q"
            )

        # ----------------------------------------------------
        # PHASE 8 RESIDUAL TARGET
        # ----------------------------------------------------

        delta_point = (
            best_reference_point
            -
            composite_q_point
        )

        # ----------------------------------------------------
        # Same observable V3.3 ML features
        # ----------------------------------------------------

        feature_record = (
            calculate_observable_features(
                pre_correction_state=
                    pre_correction_state,

                component_profile=
                    component_profile,

                component_index=
                    component_index,

                lsq_point=
                    lsq_point,
            )
        )

        # ----------------------------------------------------
        # Quality improvements
        # ----------------------------------------------------

        reference_gain_over_composite_q = float(
            composite_q_quality
            -
            best_reference_quality
        )

        reference_relative_gain_percent = float(
            100.0
            *
            reference_gain_over_composite_q
            /
            max(
                composite_q_quality,
                1e-12,
            )
        )

        # ----------------------------------------------------
        # Save one decision row
        # ----------------------------------------------------

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
                float(
                    lsq_quality
                ),

            "composite_q_true_quality":
                float(
                    composite_q_quality
                ),

            "composite_q_z_adj_mm":
                float(
                    composite_q_point[0]
                ),

            "composite_q_theta_adj_deg":
                float(
                    composite_q_point[1]
                ),

            "composite_q_locator_offset_mm":
                float(
                    composite_q_point[2]
                ),

            "composite_q_utilization":
                calculate_utilization(
                    composite_q_point
                ),

            "composite_q_optimizer_success":
                bool(
                    composite_q[
                        "success"
                    ]
                ),

            "composite_q_evaluations":
                int(
                    composite_q[
                        "evaluations"
                    ]
                ),

            "composite_q_winning_start":
                composite_q[
                    "winning_start"
                ],

            "numerical_reference_true_quality":
                numerical_quality,

            "numerical_reference_z_adj_mm":
                float(
                    numerical_point[0]
                ),

            "numerical_reference_theta_adj_deg":
                float(
                    numerical_point[1]
                ),

            "numerical_reference_locator_offset_mm":
                float(
                    numerical_point[2]
                ),

            "numerical_reference_utilization":
                calculate_utilization(
                    numerical_point
                ),

            "numerical_reference_optimizer_success":
                bool(
                    numerical_reference[
                        "success"
                    ]
                ),

            "numerical_reference_evaluations":
                int(
                    numerical_reference[
                        "evaluations"
                    ]
                ),

            "numerical_reference_beats_composite_q":
                numerical_beats_composite_q,

            "best_reference_source":
                best_reference_source,

            "best_reference_true_quality":
                float(
                    best_reference_quality
                ),

            "best_reference_z_adj_mm":
                float(
                    best_reference_point[0]
                ),

            "best_reference_theta_adj_deg":
                float(
                    best_reference_point[1]
                ),

            "best_reference_locator_offset_mm":
                float(
                    best_reference_point[2]
                ),

            "reference_gain_over_composite_q":
                reference_gain_over_composite_q,

            "reference_relative_gain_percent":
                reference_relative_gain_percent,

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
        }

        records.append(
            record
        )

        # ----------------------------------------------------
        # Sequential state propagation
        #
        # Phase 8 target-generation trajectory follows the
        # deterministic Composite-Q base controller.
        # ----------------------------------------------------

        composite_q_state = (
            composite_q_corrected_state
        )

    return records


# ============================================================
# GENERATE ONE DATA SPLIT
# ============================================================

def generate_split(
    split_name,
    n_assemblies,
    seed,
):
    """Generate one complete independent Phase 8 data split."""

    print(
        "\n"
        "------------------------------------------------------------"
    )

    print(
        f"Generating Phase 8 split: {split_name}"
    )

    print(
        f"Assemblies      : {n_assemblies}"
    )

    print(
        f"Sequence length : {SEQUENCE_LENGTH}"
    )

    print(
        f"Seed            : {seed}"
    )

    print(
        "------------------------------------------------------------"
    )

    cases = generate_cases(
        split_name=
            split_name,

        n_assemblies=
            n_assemblies,

        seed=
            seed,
    )

    records = []

    start_time = time.time()

    n_workers = min(
        mp.cpu_count(),
        4,
    )

    completed = 0

    with ProcessPoolExecutor(
        max_workers=n_workers
    ) as executor:

        for assembly_records in executor.map(
            process_assembly,
            cases,
        ):
            records.extend(
                assembly_records
            )

            completed += 1

            if (
                completed == 1
                or
                completed % 10 == 0
                or
                completed == n_assemblies
            ):
                elapsed = (
                    time.time()
                    -
                    start_time
                )

                print(
                    f"{split_name}: "
                    f"completed "
                    f"{completed}/{n_assemblies} assemblies "
                    f"| elapsed {elapsed:.1f} s"
                )

    df = pd.DataFrame(
        records
    )

    df = df.sort_values(
        [
            "assembly_index",
            "component_index",
        ]
    ).reset_index(
        drop=True
    )

    return df


# ============================================================
# DATASET VALIDATION
# ============================================================

def validate_generated_datasets(
    train_df,
    development_df,
):
    """Run immediate integrity checks before saving the datasets."""

    expected_train_rows = (
        N_TRAIN_ASSEMBLIES
        *
        SEQUENCE_LENGTH
    )

    expected_development_rows = (
        N_DEVELOPMENT_ASSEMBLIES
        *
        SEQUENCE_LENGTH
    )

    if len(train_df) != expected_train_rows:
        raise RuntimeError(
            "Training row-count check failed: "
            f"{len(train_df)} != {expected_train_rows}"
        )

    if len(development_df) != expected_development_rows:
        raise RuntimeError(
            "Development row-count check failed: "
            f"{len(development_df)} != "
            f"{expected_development_rows}"
        )

    train_ids = set(
        train_df[
            "assembly_id"
        ]
    )

    development_ids = set(
        development_df[
            "assembly_id"
        ]
    )

    overlap = (
        train_ids
        &
        development_ids
    )

    if overlap:
        raise RuntimeError(
            "Training/development assembly overlap detected."
        )

    if (
        train_df.isna().any().any()
        or
        development_df.isna().any().any()
    ):
        raise RuntimeError(
            "NaN values detected in generated Phase 8 datasets."
        )

    target_columns = [
        "target_delta_z_mm",
        "target_delta_theta_deg",
        "target_delta_locator_mm",
    ]

    for column in target_columns:
        if not np.isfinite(
            train_df[column]
        ).all():
            raise RuntimeError(
                f"Non-finite training target detected in {column}."
            )

        if not np.isfinite(
            development_df[column]
        ).all():
            raise RuntimeError(
                f"Non-finite development target detected in {column}."
            )

    # The retained reference must never intentionally be worse
    # than Composite-Q.
    train_worse = (
        train_df[
            "best_reference_true_quality"
        ]
        >
        train_df[
            "composite_q_true_quality"
        ]
        +
        1e-9
    )

    development_worse = (
        development_df[
            "best_reference_true_quality"
        ]
        >
        development_df[
            "composite_q_true_quality"
        ]
        +
        1e-9
    )

    if train_worse.any():
        raise RuntimeError(
            "Best-reference safety check failed in training data."
        )

    if development_worse.any():
        raise RuntimeError(
            "Best-reference safety check failed in development data."
        )

    return {
        "expected_train_rows":
            expected_train_rows,

        "expected_development_rows":
            expected_development_rows,

        "assembly_overlap":
            len(
                overlap
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_phase8_directories()

    print(
        "\n"
        "============================================================"
    )

    print(
        "PHASE 8: OBJECTIVE-ALIGNED RESIDUAL DATASET GENERATION"
    )

    print(
        "============================================================"
    )

    print(
        f"\nSequence length        : {SEQUENCE_LENGTH}"
    )

    print(
        f"Nonlinearity strength : {NONLINEARITY_STRENGTH}"
    )

    print(
        f"Training assemblies   : {N_TRAIN_ASSEMBLIES}"
    )

    print(
        f"Development assemblies: {N_DEVELOPMENT_ASSEMBLIES}"
    )

    print(
        f"Observable features   : "
        f"{len(OBSERVABLE_FEATURE_COLUMNS)}"
    )

    print(
        "\nComposite-Q optimizer:"
    )

    print(
        f"  Starts              : "
        f"{len(COMPOSITE_Q_GRID_STARTS) + 1}"
    )

    print(
        "  Method              : bounded multi-start L-BFGS-B"
    )

    print(
        "\nHigher-budget numerical reference:"
    )

    print(
        f"  DE maxiter          : {REFERENCE_DE_MAXITER}"
    )

    print(
        f"  DE popsize          : {REFERENCE_DE_POPSIZE}"
    )

    print(
        f"  DE tolerance        : {REFERENCE_DE_TOL}"
    )

    overall_start = (
        time.time()
    )

    train_df = generate_split(
        split_name=
            "training",

        n_assemblies=
            N_TRAIN_ASSEMBLIES,

        seed=
            TRAIN_SEED,
    )

    development_df = generate_split(
        split_name=
            "development",

        n_assemblies=
            N_DEVELOPMENT_ASSEMBLIES,

        seed=
            DEVELOPMENT_SEED,
    )

    checks = (
        validate_generated_datasets(
            train_df,
            development_df,
        )
    )

    # ========================================================
    # OUTPUT FILES
    # ========================================================

    training_file = (
        TRAINING_DATA_DIR
        /
        "phase8_residual_learning_training.csv"
    )

    development_file = (
        DEVELOPMENT_DATA_DIR
        /
        "phase8_residual_learning_development.csv"
    )

    feature_file = (
        TRAINING_DATA_DIR
        /
        "phase8_observable_features.txt"
    )

    summary_file = (
        TABLES_DIR
        /
        "phase8_target_generation_summary.csv"
    )

    train_df.to_csv(
        training_file,
        index=False,
    )

    development_df.to_csv(
        development_file,
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

    combined_df = pd.concat(
        [
            train_df,
            development_df,
        ],
        ignore_index=True,
    )

    mean_composite_q_quality = float(
        combined_df[
            "composite_q_true_quality"
        ].mean()
    )

    mean_best_reference_quality = float(
        combined_df[
            "best_reference_true_quality"
        ].mean()
    )

    numerical_reference_win_rate = float(
        100.0
        *
        combined_df[
            "numerical_reference_beats_composite_q"
        ].mean()
    )

    mean_reference_gain = float(
        combined_df[
            "reference_gain_over_composite_q"
        ].mean()
    )

    mean_relative_gain = float(
        combined_df[
            "reference_relative_gain_percent"
        ].mean()
    )

    zero_residual_mask = (
        np.isclose(
            combined_df[
                "target_delta_z_mm"
            ],
            0.0,
            atol=1e-12,
        )
        &
        np.isclose(
            combined_df[
                "target_delta_theta_deg"
            ],
            0.0,
            atol=1e-12,
        )
        &
        np.isclose(
            combined_df[
                "target_delta_locator_mm"
            ],
            0.0,
            atol=1e-12,
        )
    )

    zero_residual_percent = float(
        100.0
        *
        zero_residual_mask.mean()
    )

    mean_abs_delta_z = float(
        combined_df[
            "target_delta_z_mm"
        ].abs().mean()
    )

    mean_abs_delta_theta = float(
        combined_df[
            "target_delta_theta_deg"
        ].abs().mean()
    )

    mean_abs_delta_locator = float(
        combined_df[
            "target_delta_locator_mm"
        ].abs().mean()
    )

    elapsed_seconds = float(
        time.time()
        -
        overall_start
    )

    summary_df = pd.DataFrame(
        {
            "metric": [
                "sequence_length",
                "nonlinearity_strength",
                "training_assemblies",
                "development_assemblies",
                "training_decisions",
                "development_decisions",
                "train_development_assembly_overlap",
                "observable_feature_count",
                "mean_composite_q_true_quality",
                "mean_best_reference_true_quality",
                "high_budget_reference_beats_composite_q_percent",
                "mean_reference_gain_over_composite_q",
                "mean_reference_relative_gain_percent",
                "zero_residual_target_percent",
                "mean_abs_target_delta_z_mm",
                "mean_abs_target_delta_theta_deg",
                "mean_abs_target_delta_locator_mm",
                "generation_runtime_seconds",
            ],

            "value": [
                SEQUENCE_LENGTH,
                NONLINEARITY_STRENGTH,
                N_TRAIN_ASSEMBLIES,
                N_DEVELOPMENT_ASSEMBLIES,
                len(
                    train_df
                ),
                len(
                    development_df
                ),
                checks[
                    "assembly_overlap"
                ],
                len(
                    OBSERVABLE_FEATURE_COLUMNS
                ),
                mean_composite_q_quality,
                mean_best_reference_quality,
                numerical_reference_win_rate,
                mean_reference_gain,
                mean_relative_gain,
                zero_residual_percent,
                mean_abs_delta_z,
                mean_abs_delta_theta,
                mean_abs_delta_locator,
                elapsed_seconds,
            ],
        }
    )

    summary_df.to_csv(
        summary_file,
        index=False,
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print(
        "\n"
        "============================================================"
    )

    print(
        "PHASE 8 TARGET-GENERATION RESULTS"
    )

    print(
        "============================================================"
    )

    print(
        f"\nTraining rows             : "
        f"{len(train_df)} / "
        f"{checks['expected_train_rows']}"
    )

    print(
        f"Development rows          : "
        f"{len(development_df)} / "
        f"{checks['expected_development_rows']}"
    )

    print(
        f"Assembly overlap          : "
        f"{checks['assembly_overlap']}"
    )

    print(
        f"Observable ML features    : "
        f"{len(OBSERVABLE_FEATURE_COLUMNS)}"
    )

    print(
        f"\nMean Composite-Q quality  : "
        f"{mean_composite_q_quality:.6f}"
    )

    print(
        f"Mean best-reference quality: "
        f"{mean_best_reference_quality:.6f}"
    )

    print(
        f"High-budget DE beats CQ   : "
        f"{numerical_reference_win_rate:.2f}%"
    )

    print(
        f"Mean remaining gain       : "
        f"{mean_reference_gain:.6f}"
    )

    print(
        f"Mean relative gain        : "
        f"{mean_relative_gain:.4f}%"
    )

    print(
        f"Zero residual targets     : "
        f"{zero_residual_percent:.2f}%"
    )

    print(
        "\nMean absolute residual targets:"
    )

    print(
        f"  delta z       : "
        f"{mean_abs_delta_z:.6f} mm"
    )

    print(
        f"  delta theta   : "
        f"{mean_abs_delta_theta:.6f} deg"
    )

    print(
        f"  delta locator : "
        f"{mean_abs_delta_locator:.6f} mm"
    )

    print(
        f"\nRuntime                   : "
        f"{elapsed_seconds:.1f} s"
    )

    print(
        "\nSaved training dataset:"
    )

    print(
        training_file
    )

    print(
        "\nSaved development dataset:"
    )

    print(
        development_file
    )

    print(
        "\nSaved observable feature list:"
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
        "\nPASS: Phase 8 training/development datasets generated "
        "and validated."
    )

    print(
        "\nImportant:"
    )

    print(
        "This result only establishes whether a residual target exists "
        "beyond Composite-Q."
    )

    print(
        "It does not yet establish whether RF or Gradient Boosting can "
        "predict that residual on unseen assemblies."
    )

    print(
        "\n"
        "============================================================"
    )

    print(
        "PHASE 8 TARGET GENERATION COMPLETED"
    )

    print(
        "============================================================"
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()