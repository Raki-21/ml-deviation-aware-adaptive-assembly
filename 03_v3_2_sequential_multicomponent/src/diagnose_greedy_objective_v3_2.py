import os
import time
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


# ============================================================
# VERSION 3.2
# GREEDY SEQUENTIAL OBJECTIVE DIAGNOSIS
# ============================================================
#
# RESEARCH QUESTION
#
# Is poor final adaptive performance mainly caused by:
#
#   A) ML surrogate prediction error
#
# or
#
#   B) the greedy sequential objective itself?
#
#
# The current controller approximately solves:
#
#       C_k* = argmin Q(S_k)
#
# at every component stage.
#
# But the real objective of the finished product is:
#
#       minimize Q(S_5)
#
#
# Because later deviations can naturally compensate earlier
# deviations, a locally optimal correction at stage k may not
# be globally optimal for the completed assembly.
#
#
# DIAGNOSTIC METHOD
#
# REMOVE MACHINE LEARNING COMPLETELY.
#
# For every component, candidate corrections are evaluated
# directly using the ACTUAL assembly simulator.
#
# Therefore:
#
#   - no RF error
#   - no surrogate distribution shift
#   - no BO surrogate exploitation
#
#
# If simulator-direct greedy optimization still struggles
# against zero correction, the sequential objective itself
# becomes the main suspect.
#
# If simulator-direct greedy optimization performs strongly,
# the surrogate/controller remains the primary problem.
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

OUTPUT_COMPONENT_FILE = (
    "data/processed/"
    "v3_2_greedy_diagnosis_component_results.csv"
)

OUTPUT_ASSEMBLY_FILE = (
    "data/processed/"
    "v3_2_greedy_diagnosis_assembly_results.csv"
)

OUTPUT_SUMMARY_FILE = (
    "results/tables/"
    "v3_2_greedy_objective_diagnosis_summary.csv"
)

OUTPUT_STAGE_FILE = (
    "results/tables/"
    "v3_2_greedy_objective_stage_comparison.csv"
)


# ============================================================
# FILE CHECK
# ============================================================

required_files = [
    COMPONENT_FILE,
    ORIGINAL_PILOT_FILE,
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
        "\nRequired V3.2 files are unavailable."
    )


# ============================================================
# SETTINGS
# ============================================================
#
# 30 assemblies are enough for this DIAGNOSTIC experiment.
#
# This is not final validation.
#
# Simulator-direct evaluation is cheap compared with BO,
# therefore we can use many candidate corrections.
# ============================================================

N_DIAGNOSTIC_ASSEMBLIES = 30

N_COMPONENTS = 5

N_RANDOM_CANDIDATES = 500

RANDOM_SEED = 20260825


# ============================================================
# ORIGINAL CORRECTION CAPABILITY
# ============================================================

Z_LIMIT = 2.5
THETA_LIMIT = 1.2
LOCATOR_LIMIT = 1.0


# ============================================================
# LOAD DATA
# ============================================================

component_df = pd.read_csv(
    COMPONENT_FILE
)


original_pilot_df = pd.read_csv(
    ORIGINAL_PILOT_FILE
)


# ============================================================
# USE ONLY ASSEMBLIES FROM ORIGINAL PILOT
# ============================================================

pilot_ids = (
    original_pilot_df[
        "assembly_id"
    ]
    .unique()
    .tolist()
)


pilot_component_df = (
    component_df[
        component_df[
            "assembly_id"
        ]
        .isin(
            pilot_ids
        )
    ]
    .copy()
)


# ============================================================
# BALANCED SELECTION BY BATCH CONDITION
# ============================================================

metadata = (
    pilot_component_df[
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


rng_selection = np.random.default_rng(
    RANDOM_SEED
)


selected_assemblies = []


base_per_condition = (
    N_DIAGNOSTIC_ASSEMBLIES
    //
    len(
        batch_conditions
    )
)


remaining = (
    N_DIAGNOSTIC_ASSEMBLIES
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


    if n_select > len(ids):

        n_select = len(ids)


    chosen = rng_selection.choice(
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


# ============================================================
# COMPONENT PROFILE
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
# APPLY CANDIDATE TO ACTUAL SIMULATOR
# ============================================================

def simulate_candidate(
    state,
    component_profile,
    batch_disturbance,
    z_adj,
    theta_adj,
    locator_offset,
):


    correction = correction_profile(

        z_adj_mm=z_adj,

        theta_adj_deg=theta_adj,

        locator_offset_mm=locator_offset,
    )


    candidate_state = update_assembly_state(

        previous_state=state,

        component_deviation=(
            component_profile
        ),

        fixture_drift=(
            batch_disturbance
        ),

        correction=correction,
    )


    metrics = calculate_quality_metrics(
        candidate_state
    )


    return (
        candidate_state,
        metrics,
    )


# ============================================================
# DIRECT SIMULATOR SEARCH
# ============================================================
#
# This is deliberately NOT called "oracle optimization".
#
# It is a high-budget simulator-direct random search.
#
# Every candidate is evaluated on the actual assembly model.
# ============================================================

def run_direct_greedy_search(
    state,
    component_profile,
    batch_disturbance,
    seed,
):


    rng = np.random.default_rng(
        seed
    )


    best_state = None

    best_metrics = None

    best_quality = np.inf

    best_recommendation = None


    # --------------------------------------------------------
    # Candidate 0 = DO NOTHING
    #
    # This is essential.
    #
    # The optimizer is explicitly allowed to decide that no
    # correction is better than an unnecessary intervention.
    # --------------------------------------------------------

    candidate_state, candidate_metrics = (
        simulate_candidate(

            state=state,

            component_profile=(
                component_profile
            ),

            batch_disturbance=(
                batch_disturbance
            ),

            z_adj=0.0,

            theta_adj=0.0,

            locator_offset=0.0,
        )
    )


    best_state = candidate_state

    best_metrics = candidate_metrics

    best_quality = candidate_metrics[
        "quality_score"
    ]


    best_recommendation = {

        "z_adj":
            0.0,

        "theta_adj":
            0.0,

        "locator_offset":
            0.0,
    }


    # --------------------------------------------------------
    # Informed correction candidate
    #
    # Use current + incoming uncontrolled state to generate a
    # physically interpretable first-order compensation.
    # --------------------------------------------------------

    uncontrolled_state = candidate_state


    signed_mean = float(
        np.mean(
            uncontrolled_state
        )
    )


    end_difference = float(
        uncontrolled_state[-1]
        -
        uncontrolled_state[0]
    )


    estimated_angle = float(
        np.rad2deg(
            np.arctan(
                end_difference
                /
                PROFILE_LENGTH_MM
            )
        )
    )


    informed_z = float(
        np.clip(
            -signed_mean,
            -Z_LIMIT,
            Z_LIMIT,
        )
    )


    informed_theta = float(
        np.clip(
            -estimated_angle,
            -THETA_LIMIT,
            THETA_LIMIT,
        )
    )


    informed_locator = 0.0


    informed_state, informed_metrics = (
        simulate_candidate(

            state=state,

            component_profile=(
                component_profile
            ),

            batch_disturbance=(
                batch_disturbance
            ),

            z_adj=informed_z,

            theta_adj=informed_theta,

            locator_offset=(
                informed_locator
            ),
        )
    )


    if (
        informed_metrics[
            "quality_score"
        ]
        <
        best_quality
    ):

        best_state = informed_state

        best_metrics = informed_metrics

        best_quality = informed_metrics[
            "quality_score"
        ]

        best_recommendation = {

            "z_adj":
                informed_z,

            "theta_adj":
                informed_theta,

            "locator_offset":
                informed_locator,
        }


    # --------------------------------------------------------
    # RANDOM SIMULATOR-DIRECT CANDIDATES
    # --------------------------------------------------------

    for _ in range(
        N_RANDOM_CANDIDATES
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


        candidate_state, candidate_metrics = (
            simulate_candidate(

                state=state,

                component_profile=(
                    component_profile
                ),

                batch_disturbance=(
                    batch_disturbance
                ),

                z_adj=z_adj,

                theta_adj=theta_adj,

                locator_offset=(
                    locator_offset
                ),
            )
        )


        candidate_quality = (
            candidate_metrics[
                "quality_score"
            ]
        )


        if (
            candidate_quality
            <
            best_quality
        ):

            best_quality = candidate_quality

            best_state = candidate_state

            best_metrics = candidate_metrics

            best_recommendation = {

                "z_adj":
                    z_adj,

                "theta_adj":
                    theta_adj,

                "locator_offset":
                    locator_offset,
            }


    return (
        best_state,
        best_metrics,
        best_recommendation,
    )


# ============================================================
# CAPABILITY UTILIZATION
# ============================================================

def calculate_utilization(
    recommendation
):

    z_util = (
        abs(
            recommendation[
                "z_adj"
            ]
        )
        /
        Z_LIMIT
    )


    theta_util = (
        abs(
            recommendation[
                "theta_adj"
            ]
        )
        /
        THETA_LIMIT
    )


    locator_util = (
        abs(
            recommendation[
                "locator_offset"
            ]
        )
        /
        LOCATOR_LIMIT
    )


    return float(
        max(
            z_util,
            theta_util,
            locator_util,
        )
    )


# ============================================================
# RUN DIAGNOSIS
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "V3.2 GREEDY OBJECTIVE DIAGNOSIS"
)

print(
    "============================================================"
)


print(
    f"\nDiagnostic assemblies : "
    f"{len(selected_assemblies)}"
)

print(
    f"Components / assembly : "
    f"{N_COMPONENTS}"
)

print(
    f"Direct candidates/step: "
    f"{N_RANDOM_CANDIDATES + 2}"
)

print(
    "\nMachine learning is NOT used in this experiment."
)


experiment_start = time.time()


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
            ==
            assembly_id
        ]
        .sort_values(
            "component_index"
        )
    )


    zero_state = create_initial_state()

    greedy_state = create_initial_state()


    batch_condition = (
        assembly_rows.iloc[0][
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


        # ----------------------------------------------------
        # ZERO TRAJECTORY
        # ----------------------------------------------------

        zero_state = update_assembly_state(

            previous_state=zero_state,

            component_deviation=(
                component_profile
            ),

            fixture_drift=(
                batch_disturbance
            ),
        )


        zero_metrics = calculate_quality_metrics(
            zero_state
        )


        # ----------------------------------------------------
        # SIMULATOR-DIRECT GREEDY TRAJECTORY
        # ----------------------------------------------------

        direct_seed = (
            RANDOM_SEED
            +
            int(
                assembly_id
            )
            * 10
            +
            component_index
        )


        (
            greedy_state,
            greedy_metrics,
            recommendation,
        ) = run_direct_greedy_search(

            state=greedy_state,

            component_profile=(
                component_profile
            ),

            batch_disturbance=(
                batch_disturbance
            ),

            seed=direct_seed,
        )


        utilization = calculate_utilization(
            recommendation
        )


        selected_zero_correction = (
            abs(
                recommendation[
                    "z_adj"
                ]
            )
            <
            1e-12

            and

            abs(
                recommendation[
                    "theta_adj"
                ]
            )
            <
            1e-12

            and

            abs(
                recommendation[
                    "locator_offset"
                ]
            )
            <
            1e-12
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

                "zero_quality":
                    zero_metrics[
                        "quality_score"
                    ],

                "direct_greedy_quality":
                    greedy_metrics[
                        "quality_score"
                    ],

                "direct_z_adj":
                    recommendation[
                        "z_adj"
                    ],

                "direct_theta_adj":
                    recommendation[
                        "theta_adj"
                    ],

                "direct_locator_offset":
                    recommendation[
                        "locator_offset"
                    ],

                "direct_utilization":
                    utilization,

                "selected_do_nothing":
                    selected_zero_correction,

                "direct_mean_gap":
                    greedy_metrics[
                        "mean_gap"
                    ],

                "direct_max_gap":
                    greedy_metrics[
                        "max_gap"
                    ],

                "direct_parallelism":
                    greedy_metrics[
                        "parallelism_error"
                    ],

                "direct_rms":
                    greedy_metrics[
                        "rms_deviation"
                    ],
            }
        )


    # ========================================================
    # FINAL ASSEMBLY
    # ========================================================

    zero_final = calculate_quality_metrics(
        zero_state
    )


    greedy_final = calculate_quality_metrics(
        greedy_state
    )


    zero_final_quality = float(
        zero_final[
            "quality_score"
        ]
    )


    greedy_final_quality = float(
        greedy_final[
            "quality_score"
        ]
    )


    final_improvement = (
        (
            zero_final_quality
            -
            greedy_final_quality
        )
        /
        max(
            abs(
                zero_final_quality
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
                zero_final_quality,

            "direct_greedy_final_quality":
                greedy_final_quality,

            "direct_greedy_improvement_percent":
                final_improvement,

            "direct_greedy_final_mean_gap":
                greedy_final[
                    "mean_gap"
                ],

            "direct_greedy_final_max_gap":
                greedy_final[
                    "max_gap"
                ],

            "direct_greedy_final_parallelism":
                greedy_final[
                    "parallelism_error"
                ],

            "direct_greedy_final_rms":
                greedy_final[
                    "rms_deviation"
                ],
        }
    )


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
            f"{elapsed:.1f} sec"
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
# STAGE ANALYSIS
# ============================================================

stage_rows = []


for component_index in range(
    1,
    6,
):


    stage = component_result_df[
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

            "zero_mean_quality":
                stage[
                    "zero_quality"
                ]
                .mean(),

            "direct_greedy_mean_quality":
                stage[
                    "direct_greedy_quality"
                ]
                .mean(),

            "greedy_win_rate_percent":
                (
                    stage[
                        "direct_greedy_quality"
                    ]
                    <
                    stage[
                        "zero_quality"
                    ]
                )
                .mean()
                *
                100.0,

            "mean_utilization":
                stage[
                    "direct_utilization"
                ]
                .mean(),

            "do_nothing_rate_percent":
                stage[
                    "selected_do_nothing"
                ]
                .mean()
                *
                100.0,
        }
    )


stage_df = pd.DataFrame(
    stage_rows
)


# ============================================================
# FINAL DIAGNOSTIC STATISTICS
# ============================================================

final_win_rate = (
    assembly_result_df[
        "direct_greedy_final_quality"
    ]
    <
    assembly_result_df[
        "zero_final_quality"
    ]
).mean() * 100.0


zero_mean_final = (
    assembly_result_df[
        "zero_final_quality"
    ]
    .mean()
)


greedy_mean_final = (
    assembly_result_df[
        "direct_greedy_final_quality"
    ]
    .mean()
)


mean_final_improvement = (
    assembly_result_df[
        "direct_greedy_improvement_percent"
    ]
    .mean()
)


median_final_improvement = (
    assembly_result_df[
        "direct_greedy_improvement_percent"
    ]
    .median()
)


p95_zero = (
    assembly_result_df[
        "zero_final_quality"
    ]
    .quantile(
        0.95
    )
)


p95_greedy = (
    assembly_result_df[
        "direct_greedy_final_quality"
    ]
    .quantile(
        0.95
    )
)


mean_utilization = (
    component_result_df[
        "direct_utilization"
    ]
    .mean()
)


near_limit_rate = (
    component_result_df[
        "direct_utilization"
    ]
    .ge(
        0.90
    )
    .mean()
    *
    100.0
)


do_nothing_rate = (
    component_result_df[
        "selected_do_nothing"
    ]
    .mean()
    *
    100.0
)


# ============================================================
# SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    {

        "metric": [

            "zero_mean_final_quality",

            "direct_greedy_mean_final_quality",

            "direct_greedy_final_win_rate_vs_zero_percent",

            "mean_final_improvement_percent",

            "median_final_improvement_percent",

            "zero_p95_final_quality",

            "direct_greedy_p95_final_quality",

            "direct_greedy_mean_utilization",

            "direct_greedy_near_limit_rate_percent",

            "direct_greedy_do_nothing_rate_percent",
        ],

        "value": [

            zero_mean_final,

            greedy_mean_final,

            final_win_rate,

            mean_final_improvement,

            median_final_improvement,

            p95_zero,

            p95_greedy,

            mean_utilization,

            near_limit_rate,

            do_nothing_rate,
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


stage_df.to_csv(
    OUTPUT_STAGE_FILE,
    index=False,
)


# ============================================================
# PRINT RESULTS
# ============================================================

total_runtime = (
    time.time()
    -
    experiment_start
)


print(
    "\n"
    "============================================================"
)

print(
    "GREEDY OBJECTIVE DIAGNOSIS RESULTS"
)

print(
    "============================================================"
)


print(
    f"\nZero mean final quality          : "
    f"{zero_mean_final:.4f}"
)


print(
    f"Direct greedy mean final quality : "
    f"{greedy_mean_final:.4f}"
)


print(
    f"\nDirect greedy win rate vs zero   : "
    f"{final_win_rate:.2f}%"
)


print(
    f"Mean final improvement           : "
    f"{mean_final_improvement:.2f}%"
)


print(
    f"Median final improvement         : "
    f"{median_final_improvement:.2f}%"
)


print(
    f"\nZero P95 final quality           : "
    f"{p95_zero:.4f}"
)


print(
    f"Direct greedy P95 final quality  : "
    f"{p95_greedy:.4f}"
)


print(
    f"\nDirect greedy mean utilization   : "
    f"{mean_utilization:.3f}"
)


print(
    f"Direct greedy >=90% capability   : "
    f"{near_limit_rate:.2f}%"
)


print(
    f"Optimizer selected DO NOTHING    : "
    f"{do_nothing_rate:.2f}%"
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
    stage_df.round(
        4
    ).to_string(
        index=False
    )
)


# ============================================================
# DECISION LOGIC
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "SCIENTIFIC DIAGNOSIS"
)

print(
    "============================================================"
)


if (
    greedy_mean_final
    <
    zero_mean_final
    and
    final_win_rate
    >=
    70.0
):


    print(
        "\nRESULT A:"
    )


    print(
        "Simulator-direct greedy correction substantially "
        "outperforms zero correction."
    )


    print(
        "\nINTERPRETATION:"
    )


    print(
        "The current sequential correction concept is "
        "fundamentally capable of improving final quality."
    )


    print(
        "The remaining weakness is therefore primarily in "
        "the learned surrogate / adaptive recommendation "
        "controller rather than the basic greedy objective."
    )


    print(
        "\nNEXT RESEARCH STEP:"
    )


    print(
        "Improve the surrogate/controller representation "
        "before introducing look-ahead optimization."
    )


elif (
    greedy_mean_final
    <
    zero_mean_final
    and
    final_win_rate
    >=
    55.0
):


    print(
        "\nRESULT B:"
    )


    print(
        "Simulator-direct greedy correction improves mean "
        "quality, but the advantage is inconsistent."
    )


    print(
        "\nINTERPRETATION:"
    )


    print(
        "Both controller accuracy and sequential objective "
        "formulation may be contributing."
    )


    print(
        "\nNEXT RESEARCH STEP:"
    )


    print(
        "Test a limited look-ahead objective on the same "
        "assemblies before making a larger architecture change."
    )


else:


    print(
        "\nRESULT C:"
    )


    print(
        "Even simulator-direct greedy correction does not "
        "reliably outperform zero correction."
    )


    print(
        "\nINTERPRETATION:"
    )


    print(
        "The problem is deeper than surrogate prediction."
    )


    print(
        "Greedy minimization of each intermediate assembly "
        "state is likely misaligned with final assembly quality."
    )


    print(
        "\nNEXT RESEARCH STEP:"
    )


    print(
        "Introduce and test final-assembly-aware / look-ahead "
        "sequential optimization."
    )


print(
    "\n"
    "------------------------------------------------------------"
)


print(
    "IMPORTANT:"
)


print(
    "This is a diagnostic isolation experiment, not final "
    "thesis validation."
)


print(
    "Its purpose is to determine WHICH problem should be "
    "solved next."
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
    OUTPUT_STAGE_FILE
)


print(
    f"\nRuntime: "
    f"{total_runtime:.2f} sec"
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.2 GREEDY OBJECTIVE DIAGNOSIS COMPLETED"
)

print(
    "============================================================"
)