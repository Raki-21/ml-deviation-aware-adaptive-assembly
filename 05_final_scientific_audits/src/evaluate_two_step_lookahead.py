import os
import sys
import time
import numpy as np
import pandas as pd

from scipy.optimize import differential_evolution


# ============================================================
# TWO-STEP LOOK-AHEAD DIAGNOSTIC
# ============================================================
#
# Purpose
# -------
#
# Determine whether the sequential assembly problem contains
# meaningful non-myopic behaviour.
#
# Two simulator-direct strategies are compared:
#
# 1. GREEDY DIRECT
#
#    At stage k, directly minimize quality after the current
#    component only.
#
#
# 2. TWO-STEP LOOK-AHEAD
#
#    For stages k < final stage, jointly optimize the current
#    and next correction so that quality after stage k+1 is
#    minimized.
#
#    Only the CURRENT correction is applied.
#
#    At the next stage the optimization is repeated using the
#    updated assembly state.
#
#
# This is a small diagnostic, not a new production controller.
#
# The purpose is to test whether greedy sequential correction
# is already sufficient under the current additive model.
#
# The frozen V3.2 implementation is not modified.
# ============================================================


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

AUDIT_DIR = os.path.dirname(
    SCRIPT_DIR
)

PROJECT_ROOT = os.path.dirname(
    AUDIT_DIR
)

V32_DIR = os.path.join(
    PROJECT_ROOT,
    "03_v3_2_sequential_multicomponent",
)

V32_SRC_DIR = os.path.join(
    V32_DIR,
    "src",
)


if V32_SRC_DIR not in sys.path:

    sys.path.insert(
        0,
        V32_SRC_DIR,
    )


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
# INPUT
# ============================================================

INPUT_FILE = os.path.join(
    V32_DIR,
    "04_correctability_extension",
    "results",
    "prediction",
    "tables",
    "correctability_prediction_dataset.csv",
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = os.path.join(
    AUDIT_DIR,
    "results",
    "two_step_lookahead",
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


ASSEMBLY_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "two_step_assembly_results.csv",
)


COMPONENT_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "two_step_component_results.csv",
)


SUMMARY_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "two_step_summary.csv",
)


# ============================================================
# SETTINGS
# ============================================================

N_ASSEMBLIES = 30

SAMPLE_SEED = 20260827

OPTIMIZER_SEED = 20260828


Z_LIMIT = 2.5

THETA_LIMIT = 1.2

LOCATOR_LIMIT = 1.0


SINGLE_STAGE_BOUNDS = [

    (-Z_LIMIT, Z_LIMIT),

    (-THETA_LIMIT, THETA_LIMIT),

    (-LOCATOR_LIMIT, LOCATOR_LIMIT),
]


TWO_STAGE_BOUNDS = [

    (-Z_LIMIT, Z_LIMIT),

    (-THETA_LIMIT, THETA_LIMIT),

    (-LOCATOR_LIMIT, LOCATOR_LIMIT),

    (-Z_LIMIT, Z_LIMIT),

    (-THETA_LIMIT, THETA_LIMIT),

    (-LOCATOR_LIMIT, LOCATOR_LIMIT),
]


# ------------------------------------------------------------
# Optimization settings
# ------------------------------------------------------------
#
# These settings are deliberately moderate because this is a
# diagnostic rather than a production optimizer.
#
# We use the same settings for greedy and horizon-aware
# evaluation as far as dimensionality allows.
# ------------------------------------------------------------

GREEDY_MAXITER = 18

GREEDY_POPSIZE = 7


LOOKAHEAD_MAXITER = 14

LOOKAHEAD_POPSIZE = 6


TOL = 1e-6

POLISH = True


# ============================================================
# LOAD DATA
# ============================================================

if not os.path.exists(
    INPUT_FILE
):

    raise FileNotFoundError(
        f"\nMissing prediction dataset:\n{INPUT_FILE}"
    )


df = pd.read_csv(
    INPUT_FILE
)


print(
    "\n"
    "============================================================"
)

print(
    "TWO-STEP LOOK-AHEAD DIAGNOSTIC"
)

print(
    "============================================================"
)


print(
    f"\nAvailable assemblies : "
    f"{df['assembly_id'].nunique()}"
)


# ============================================================
# CHECK STRUCTURE
# ============================================================

required_columns = [

    "assembly_id",

    "component_index",

    "offset_mm",

    "tilt_deg",

    "bend_mm",

    "waviness_mm",

    "twist_mm",

    "local_bump_mm",

    "batch_offset_bias_mm",

    "batch_angular_bias_deg",

    "fixture_drift_mm",
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    raise ValueError(
        "\nMissing columns:\n"
        +
        "\n".join(
            missing_columns
        )
    )


# ============================================================
# SELECT FIXED DIAGNOSTIC SAMPLE
# ============================================================

assembly_ids = np.array(
    sorted(
        df[
            "assembly_id"
        ]
        .unique()
    )
)


if len(
    assembly_ids
) < N_ASSEMBLIES:

    raise ValueError(
        "\nNot enough assemblies for diagnostic sample."
    )


rng = np.random.default_rng(
    SAMPLE_SEED
)


sampled_assembly_ids = np.sort(

    rng.choice(

        assembly_ids,

        size=N_ASSEMBLIES,

        replace=False,
    )
)


sample_df = (

    df[
        df[
            "assembly_id"
        ]
        .isin(
            sampled_assembly_ids
        )
    ]

    .copy()
)


print(
    f"\nDiagnostic assemblies : "
    f"{N_ASSEMBLIES}"
)


print(
    "\nSample assembly IDs:"
)


print(
    sampled_assembly_ids
)


# ============================================================
# PROFILE RECONSTRUCTION
# ============================================================

def reconstruct_component_profile(
    row
):

    profile = np.zeros_like(
        s,
        dtype=float,
    )


    profile += deviation_offset(
        offset_mm=float(
            row[
                "offset_mm"
            ]
        )
    )


    profile += deviation_tilt(
        angle_deg=float(
            row[
                "tilt_deg"
            ]
        )
    )


    profile += deviation_bend(
        amplitude_mm=float(
            row[
                "bend_mm"
            ]
        )
    )


    profile += deviation_waviness(

        amplitude_mm=float(
            row[
                "waviness_mm"
            ]
        ),

        waves=3,
    )


    profile += deviation_twist(
        amplitude_mm=float(
            row[
                "twist_mm"
            ]
        )
    )


    profile += deviation_local_bump(

        amplitude_mm=float(
            row[
                "local_bump_mm"
            ]
        ),

        sigma=0.12,
    )


    return profile


# ============================================================
# BATCH / PROCESS DISTURBANCE
# ============================================================

def reconstruct_batch_disturbance(
    row
):


    offset_profile = np.full_like(

        s,

        float(
            row[
                "batch_offset_bias_mm"
            ]
        ),

        dtype=float,
    )


    angular_profile = (

        np.tan(
            np.deg2rad(
                float(
                    row[
                        "batch_angular_bias_deg"
                    ]
                )
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

        float(
            row[
                "fixture_drift_mm"
            ]
        )

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
# SIMULATE ONE STEP
# ============================================================

def simulate_step(
    state,
    component_profile,
    batch_disturbance,
    correction_values,
):


    z_adj = float(
        correction_values[
            0
        ]
    )


    theta_adj = float(
        correction_values[
            1
        ]
    )


    locator_offset = float(
        correction_values[
            2
        ]
    )


    correction = correction_profile(

        z_adj_mm=z_adj,

        theta_adj_deg=theta_adj,

        locator_offset_mm=locator_offset,
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
# GREEDY DIRECT OPTIMIZATION
# ============================================================

def optimize_greedy(
    state,
    component_profile,
    disturbance,
    seed,
):


    def objective(
        point
    ):

        (
            _,
            metrics,
        ) = simulate_step(

            state=state,

            component_profile=(
                component_profile
            ),

            batch_disturbance=(
                disturbance
            ),

            correction_values=point,
        )


        return float(
            metrics[
                "quality_score"
            ]
        )


    result = differential_evolution(

        objective,

        bounds=SINGLE_STAGE_BOUNDS,

        maxiter=GREEDY_MAXITER,

        popsize=GREEDY_POPSIZE,

        tol=TOL,

        polish=POLISH,

        seed=seed,

        workers=1,

        updating="immediate",
    )


    best_point = np.array(
        result.x,
        dtype=float,
    )


    (
        best_state,
        best_metrics,
    ) = simulate_step(

        state=state,

        component_profile=(
            component_profile
        ),

        batch_disturbance=(
            disturbance
        ),

        correction_values=(
            best_point
        ),
    )


    return (

        best_point,

        best_state,

        best_metrics,
    )


# ============================================================
# TWO-STEP LOOK-AHEAD OPTIMIZATION
# ============================================================

def optimize_two_step(
    current_state,
    current_component,
    current_disturbance,
    next_component,
    next_disturbance,
    seed,
):


    def objective(
        point
    ):


        current_correction = (
            point[
                0:3
            ]
        )


        next_correction = (
            point[
                3:6
            ]
        )


        (
            state_after_current,
            _,
        ) = simulate_step(

            state=(
                current_state
            ),

            component_profile=(
                current_component
            ),

            batch_disturbance=(
                current_disturbance
            ),

            correction_values=(
                current_correction
            ),
        )


        (
            _,
            next_metrics,
        ) = simulate_step(

            state=(
                state_after_current
            ),

            component_profile=(
                next_component
            ),

            batch_disturbance=(
                next_disturbance
            ),

            correction_values=(
                next_correction
            ),
        )


        return float(
            next_metrics[
                "quality_score"
            ]
        )


    result = differential_evolution(

        objective,

        bounds=TWO_STAGE_BOUNDS,

        maxiter=LOOKAHEAD_MAXITER,

        popsize=LOOKAHEAD_POPSIZE,

        tol=TOL,

        polish=POLISH,

        seed=seed,

        workers=1,

        updating="immediate",
    )


    current_action = np.array(

        result.x[
            0:3
        ],

        dtype=float,
    )


    predicted_next_action = np.array(

        result.x[
            3:6
        ],

        dtype=float,
    )


    (
        applied_state,
        applied_metrics,
    ) = simulate_step(

        state=current_state,

        component_profile=(
            current_component
        ),

        batch_disturbance=(
            current_disturbance
        ),

        correction_values=(
            current_action
        ),
    )


    return (

        current_action,

        predicted_next_action,

        applied_state,

        applied_metrics,

        float(
            result.fun
        ),
    )


# ============================================================
# CAPABILITY
# ============================================================

def utilization(
    point
):


    return float(

        max(

            abs(
                point[
                    0
                ]
            )
            /
            Z_LIMIT,

            abs(
                point[
                    1
                ]
            )
            /
            THETA_LIMIT,

            abs(
                point[
                    2
                ]
            )
            /
            LOCATOR_LIMIT,
        )
    )


# ============================================================
# RUN DIAGNOSTIC
# ============================================================

component_records = []

assembly_records = []


start_time = time.time()


for assembly_counter, assembly_id in enumerate(

    sampled_assembly_ids,

    start=1,
):


    assembly_rows = (

        sample_df[
            sample_df[
                "assembly_id"
            ]
            ==
            assembly_id
        ]

        .sort_values(
            "component_index"
        )

        .reset_index(
            drop=True
        )
    )


    n_components = len(
        assembly_rows
    )


    greedy_state = create_initial_state()


    lookahead_state = create_initial_state()


    for position in range(
        n_components
    ):


        current_row = (
            assembly_rows.iloc[
                position
            ]
        )


        component_index = int(
            current_row[
                "component_index"
            ]
        )


        current_component = (
            reconstruct_component_profile(
                current_row
            )
        )


        current_disturbance = (
            reconstruct_batch_disturbance(
                current_row
            )
        )


        # ----------------------------------------------------
        # GREEDY DIRECT
        # ----------------------------------------------------

        greedy_seed = (

            OPTIMIZER_SEED

            +

            int(
                assembly_id
            )
            *
            100

            +

            component_index
        )


        (
            greedy_action,

            greedy_state,

            greedy_metrics,
        ) = optimize_greedy(

            state=greedy_state,

            component_profile=(
                current_component
            ),

            disturbance=(
                current_disturbance
            ),

            seed=greedy_seed,
        )


        # ----------------------------------------------------
        # TWO-STEP LOOK-AHEAD
        # ----------------------------------------------------

        if position < (
            n_components
            -
            1
        ):


            next_row = (
                assembly_rows.iloc[
                    position
                    +
                    1
                ]
            )


            next_component = (
                reconstruct_component_profile(
                    next_row
                )
            )


            next_disturbance = (
                reconstruct_batch_disturbance(
                    next_row
                )
            )


            lookahead_seed = (

                OPTIMIZER_SEED

                +

                100000

                +

                int(
                    assembly_id
                )
                *
                100

                +

                component_index
            )


            (
                lookahead_action,

                predicted_next_action,

                lookahead_state,

                lookahead_metrics,

                predicted_two_step_quality,
            ) = optimize_two_step(

                current_state=(
                    lookahead_state
                ),

                current_component=(
                    current_component
                ),

                current_disturbance=(
                    current_disturbance
                ),

                next_component=(
                    next_component
                ),

                next_disturbance=(
                    next_disturbance
                ),

                seed=lookahead_seed,
            )


        else:


            (
                lookahead_action,

                lookahead_state,

                lookahead_metrics,
            ) = optimize_greedy(

                state=lookahead_state,

                component_profile=(
                    current_component
                ),

                disturbance=(
                    current_disturbance
                ),

                seed=(
                    OPTIMIZER_SEED
                    +
                    200000
                    +
                    int(
                        assembly_id
                    )
                ),
            )


            predicted_next_action = np.array(
                [
                    np.nan,
                    np.nan,
                    np.nan,
                ]
            )


            predicted_two_step_quality = np.nan


        component_records.append(
            {

                "assembly_id":
                    int(
                        assembly_id
                    ),

                "component_index":
                    component_index,

                "greedy_quality":
                    float(
                        greedy_metrics[
                            "quality_score"
                        ]
                    ),

                "lookahead_quality":
                    float(
                        lookahead_metrics[
                            "quality_score"
                        ]
                    ),

                "greedy_utilization":
                    utilization(
                        greedy_action
                    ),

                "lookahead_utilization":
                    utilization(
                        lookahead_action
                    ),

                "predicted_two_step_quality":
                    predicted_two_step_quality,

                "greedy_z":
                    float(
                        greedy_action[
                            0
                        ]
                    ),

                "greedy_theta":
                    float(
                        greedy_action[
                            1
                        ]
                    ),

                "greedy_locator":
                    float(
                        greedy_action[
                            2
                        ]
                    ),

                "lookahead_z":
                    float(
                        lookahead_action[
                            0
                        ]
                    ),

                "lookahead_theta":
                    float(
                        lookahead_action[
                            1
                        ]
                    ),

                "lookahead_locator":
                    float(
                        lookahead_action[
                            2
                        ]
                    ),
            }
        )


    greedy_final_quality = float(

        calculate_quality_metrics(
            greedy_state
        )[
            "quality_score"
        ]
    )


    lookahead_final_quality = float(

        calculate_quality_metrics(
            lookahead_state
        )[
            "quality_score"
        ]
    )


    assembly_records.append(
        {

            "assembly_id":
                int(
                    assembly_id
                ),

            "greedy_final_quality":
                greedy_final_quality,

            "lookahead_final_quality":
                lookahead_final_quality,

            "lookahead_minus_greedy":
                (
                    lookahead_final_quality
                    -
                    greedy_final_quality
                ),
        }
    )


    elapsed_minutes = (

        time.time()
        -
        start_time

    ) / 60.0


    print(
        f"\nCompleted "
        f"{assembly_counter}/"
        f"{N_ASSEMBLIES} assemblies"
        f" | elapsed "
        f"{elapsed_minutes:.2f} min"
    )


# ============================================================
# RESULTS
# ============================================================

component_result_df = pd.DataFrame(
    component_records
)


assembly_result_df = pd.DataFrame(
    assembly_records
)


greedy_mean = float(

    assembly_result_df[
        "greedy_final_quality"
    ]
    .mean()
)


lookahead_mean = float(

    assembly_result_df[
        "lookahead_final_quality"
    ]
    .mean()
)


absolute_difference = (

    lookahead_mean
    -
    greedy_mean
)


relative_lookahead_benefit = (

    (
        greedy_mean
        -
        lookahead_mean
    )

    /

    max(
        greedy_mean,
        1e-12,
    )

    *

    100.0
)


lookahead_better_rate = float(

    (
        assembly_result_df[
            "lookahead_final_quality"
        ]

        <

        assembly_result_df[
            "greedy_final_quality"
        ]
    )

    .mean()

    *

    100.0
)


greedy_better_rate = float(

    (
        assembly_result_df[
            "greedy_final_quality"
        ]

        <

        assembly_result_df[
            "lookahead_final_quality"
        ]
    )

    .mean()

    *

    100.0
)


median_difference = float(

    assembly_result_df[
        "lookahead_minus_greedy"
    ]
    .median()
)


p95_absolute_difference = float(

    np.percentile(

        np.abs(
            assembly_result_df[
                "lookahead_minus_greedy"
            ]
        ),

        95,
    )
)


mean_greedy_utilization = float(

    component_result_df[
        "greedy_utilization"
    ]
    .mean()
)


mean_lookahead_utilization = float(

    component_result_df[
        "lookahead_utilization"
    ]
    .mean()
)


# ============================================================
# SAVE
# ============================================================

component_result_df.to_csv(
    COMPONENT_OUTPUT,
    index=False,
)


assembly_result_df.to_csv(
    ASSEMBLY_OUTPUT,
    index=False,
)


summary_df = pd.DataFrame(
    {

        "metric": [

            "assemblies",

            "greedy_mean_final_quality",

            "lookahead_mean_final_quality",

            "absolute_lookahead_minus_greedy",

            "relative_lookahead_benefit_percent",

            "lookahead_better_percent",

            "greedy_better_percent",

            "median_difference",

            "p95_absolute_difference",

            "mean_greedy_utilization",

            "mean_lookahead_utilization",
        ],

        "value": [

            len(
                assembly_result_df
            ),

            greedy_mean,

            lookahead_mean,

            absolute_difference,

            relative_lookahead_benefit,

            lookahead_better_rate,

            greedy_better_rate,

            median_difference,

            p95_absolute_difference,

            mean_greedy_utilization,

            mean_lookahead_utilization,
        ],
    }
)


summary_df.to_csv(
    SUMMARY_OUTPUT,
    index=False,
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "TWO-STEP LOOK-AHEAD SUMMARY"
)

print(
    "============================================================"
)


print(
    f"\nAssemblies                 : "
    f"{len(assembly_result_df)}"
)


print(
    "\nMean final quality:"
)


print(
    f"\nGreedy direct              : "
    f"{greedy_mean:.4f}"
)


print(
    f"Two-step look-ahead        : "
    f"{lookahead_mean:.4f}"
)


print(
    f"\nRelative look-ahead benefit: "
    f"{relative_lookahead_benefit:.2f}%"
)


print(
    f"Look-ahead better          : "
    f"{lookahead_better_rate:.2f}%"
)


print(
    f"Greedy better              : "
    f"{greedy_better_rate:.2f}%"
)


print(
    f"Median quality difference  : "
    f"{median_difference:.4f}"
)


print(
    f"P95 absolute difference    : "
    f"{p95_absolute_difference:.4f}"
)


print(
    "\nCapability:"
)


print(
    f"\nMean greedy utilization    : "
    f"{mean_greedy_utilization:.3f}"
)


print(
    f"Mean look-ahead utilization: "
    f"{mean_lookahead_utilization:.3f}"
)


# ============================================================
# VERDICT
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "AUDIT 5 VERDICT"
)

print(
    "============================================================"
)


if (
    relative_lookahead_benefit
    >=
    10.0
):


    print(
        "\nRESULT A:"
    )


    print(
        "Two-step planning produces a substantial improvement "
        "over greedy simulator-direct correction."
    )


    print(
        "\nThe sequential problem therefore contains meaningful "
        "non-myopic structure."
    )


    print(
        "\nHorizon-aware correction is a strong future research "
        "direction."
    )


elif (
    relative_lookahead_benefit
    >=
    3.0
):


    print(
        "\nRESULT B:"
    )


    print(
        "Two-step planning provides a measurable but moderate "
        "benefit over greedy correction."
    )


    print(
        "\nGreedy correction remains strong, but some sequential "
        "planning value is present."
    )


else:


    print(
        "\nRESULT C:"
    )


    print(
        "Two-step look-ahead provides little additional "
        "benefit over greedy simulator-direct correction."
    )


    print(
        "\nUnder the present additive assembly model and "
        "correction capability, greedy sequential correction "
        "is therefore a reasonable approximation."
    )


print(
    "\nIMPORTANT:"
)


print(
    "The look-ahead controller uses knowledge of the next "
    "component only as a diagnostic assumption."
)


print(
    "This does not imply that future component geometry is "
    "available in every real production configuration."
)


print(
    "\nSaved results:"
)


print(
    OUTPUT_DIR
)


print(
    "\n"
    "============================================================"
)

print(
    "TWO-STEP LOOK-AHEAD DIAGNOSTIC COMPLETED"
)

print(
    "============================================================"
)