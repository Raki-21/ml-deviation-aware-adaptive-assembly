import os
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
# VERSION 3.2 - CLOSED-LOOP AUGMENTED TRAINING DATA
# ============================================================
#
# PURPOSE
#
# The first V3.2 surrogate was mainly trained on states from
# an uncorrected sequential trajectory.
#
# Closed-loop diagnosis showed that prediction error increased
# as the adaptive assembly progressed.
#
# This script expands the state distribution using several
# physically interpretable sequential trajectories:
#
#   1. ZERO trajectory
#   2. HEURISTIC adaptive trajectory
#   3. PARTIAL correction trajectory
#   4. RANDOM-FEASIBLE trajectory
#
# For every state we evaluate a structured mixture of candidate
# corrections rather than relying only on global uniform random
# sampling.
#
# The original V3.2 dataset/model are preserved.
# ============================================================


INPUT_FILE = (
    "data/processed/"
    "v3_2_full_component_level_dataset.csv"
)

OUTPUT_FILE = (
    "data/processed/"
    "v3_2_closed_loop_ml_training_dataset.csv"
)


if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        "\nFull V3.2 component dataset not found."
    )


RNG = np.random.default_rng(
    20260824
)


# ============================================================
# CORRECTION CAPABILITY
# ============================================================

Z_LIMIT = 2.5
THETA_LIMIT = 1.2
LOCATOR_LIMIT = 1.0


# ============================================================
# TRAJECTORIES
# ============================================================

TRAJECTORY_TYPES = [
    "zero",
    "heuristic",
    "partial",
    "random_feasible",
]


# ============================================================
# COMPONENT RECONSTRUCTION
# ============================================================

def reconstruct_component_profile(row):

    profile = np.zeros_like(s)

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
# BATCH DISTURBANCE
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
# STATE FEATURES
# ============================================================

def state_features(state):

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

    angle_deg = float(
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
            angle_deg,
    }


# ============================================================
# CORRECTION CLIPPING
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
# PHYSICALLY INFORMED CORRECTION CENTRE
# ============================================================

def informed_correction(
    state,
    row,
):

    signed_mean = float(
        np.mean(state)
    )

    signed_difference = float(
        state[-1]
        - state[0]
    )

    state_angle = float(
        np.rad2deg(
            np.arctan(
                signed_difference
                / PROFILE_LENGTH_MM
            )
        )
    )


    z_adj = -(
        row["offset_mm"]
        +
        row["batch_offset_bias_mm"]
        +
        0.50 * signed_mean
    )


    theta_adj = -(
        row["tilt_deg"]
        +
        row["batch_angular_bias_deg"]
        +
        0.50 * state_angle
    )


    locator_offset = -(
        row["local_bump_mm"]
        +
        row["fixture_drift_mm"]
    )


    return clip_correction(
        z_adj,
        theta_adj,
        locator_offset,
    )


# ============================================================
# STRUCTURED CANDIDATE GENERATION
# ============================================================

def generate_candidate_set(
    state,
    row,
):
    """
    Candidate set intentionally combines:
      - zero
      - informed correction
      - local perturbations
      - moderate random exploration
      - broad random exploration
      - limited near-boundary coverage
    """

    candidates = []


    # --------------------------------------------------------
    # 1. ZERO CORRECTION
    # --------------------------------------------------------

    candidates.append(
        (
            0.0,
            0.0,
            0.0,
            "zero",
        )
    )


    # --------------------------------------------------------
    # 2. INFORMED CORRECTION
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
        (
            informed_z,
            informed_theta,
            informed_locator,
            "informed",
        )
    )


    # --------------------------------------------------------
    # 3. LOCAL PERTURBATIONS AROUND INFORMED CORRECTION
    # --------------------------------------------------------

    for _ in range(8):

        z_adj = (
            informed_z
            + RNG.normal(
                0.0,
                0.30,
            )
        )

        theta_adj = (
            informed_theta
            + RNG.normal(
                0.0,
                0.12,
            )
        )

        locator_offset = (
            informed_locator
            + RNG.normal(
                0.0,
                0.15,
            )
        )


        (
            z_adj,
            theta_adj,
            locator_offset,
        ) = clip_correction(
            z_adj,
            theta_adj,
            locator_offset,
        )


        candidates.append(
            (
                z_adj,
                theta_adj,
                locator_offset,
                "local_informed",
            )
        )


    # --------------------------------------------------------
    # 4. MODERATE RANDOM CORRECTIONS
    # --------------------------------------------------------

    for _ in range(6):

        z_adj = RNG.uniform(
            -0.60 * Z_LIMIT,
            0.60 * Z_LIMIT,
        )

        theta_adj = RNG.uniform(
            -0.60 * THETA_LIMIT,
            0.60 * THETA_LIMIT,
        )

        locator_offset = RNG.uniform(
            -0.60 * LOCATOR_LIMIT,
            0.60 * LOCATOR_LIMIT,
        )


        candidates.append(
            (
                float(z_adj),
                float(theta_adj),
                float(locator_offset),
                "moderate_random",
            )
        )


    # --------------------------------------------------------
    # 5. BROAD EXPLORATION
    # --------------------------------------------------------

    for _ in range(4):

        candidates.append(
            (
                float(
                    RNG.uniform(
                        -Z_LIMIT,
                        Z_LIMIT,
                    )
                ),

                float(
                    RNG.uniform(
                        -THETA_LIMIT,
                        THETA_LIMIT,
                    )
                ),

                float(
                    RNG.uniform(
                        -LOCATOR_LIMIT,
                        LOCATOR_LIMIT,
                    )
                ),

                "broad_random",
            )
        )


    # --------------------------------------------------------
    # 6. LIMITED CAPABILITY-BOUNDARY COVERAGE
    # --------------------------------------------------------
    #
    # We still teach RF what happens near limits,
    # but these cases are deliberately a minority.
    # --------------------------------------------------------

    boundary_signs = RNG.choice(
        [
            -1.0,
            1.0,
        ],
        size=3,
    )


    candidates.append(
        (
            float(
                boundary_signs[0]
                * 0.90
                * Z_LIMIT
            ),

            float(
                boundary_signs[1]
                * 0.90
                * THETA_LIMIT
            ),

            float(
                boundary_signs[2]
                * 0.90
                * LOCATOR_LIMIT
            ),

            "near_boundary",
        )
    )


    return candidates


# ============================================================
# TRAJECTORY CORRECTION
# ============================================================

def trajectory_correction(
    trajectory_type,
    state,
    row,
):

    if trajectory_type == "zero":

        return (
            0.0,
            0.0,
            0.0,
        )


    (
        informed_z,
        informed_theta,
        informed_locator,
    ) = informed_correction(
        state,
        row,
    )


    if trajectory_type == "heuristic":

        return (
            informed_z,
            informed_theta,
            informed_locator,
        )


    if trajectory_type == "partial":

        return (
            0.50 * informed_z,
            0.50 * informed_theta,
            0.50 * informed_locator,
        )


    if trajectory_type == "random_feasible":

        # Moderate feasible correction trajectory,
        # deliberately avoiding constant limit saturation.

        return (
            float(
                RNG.uniform(
                    -0.50 * Z_LIMIT,
                    0.50 * Z_LIMIT,
                )
            ),

            float(
                RNG.uniform(
                    -0.50 * THETA_LIMIT,
                    0.50 * THETA_LIMIT,
                )
            ),

            float(
                RNG.uniform(
                    -0.50 * LOCATOR_LIMIT,
                    0.50 * LOCATOR_LIMIT,
                )
            ),
        )


    raise ValueError(
        f"Unknown trajectory type: "
        f"{trajectory_type}"
    )


# ============================================================
# LOAD DATA
# ============================================================

component_df = pd.read_csv(
    INPUT_FILE
)


assembly_ids = sorted(
    component_df[
        "assembly_id"
    ].unique()
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.2 CLOSED-LOOP TRAINING DATA GENERATOR"
)

print(
    "============================================================"
)


print(
    f"\nAssemblies : "
    f"{len(assembly_ids)}"
)

print(
    f"Trajectory types : "
    f"{len(TRAJECTORY_TYPES)}"
)


# ============================================================
# STORAGE
# ============================================================

records = []


# ============================================================
# MAIN LOOP
# ============================================================

for assembly_counter, assembly_id in enumerate(
    assembly_ids,
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
    # Independent current state for each trajectory
    # --------------------------------------------------------

    states = {

        trajectory:
            create_initial_state()

        for trajectory
        in TRAJECTORY_TYPES
    }


    for _, row in assembly_rows.iterrows():

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
        # EVALUATE EVERY TRAJECTORY STATE
        # ====================================================

        for trajectory_type in TRAJECTORY_TYPES:

            state = states[
                trajectory_type
            ]


            current_state_features = (
                state_features(
                    state
                )
            )


            candidates = (
                generate_candidate_set(
                    state,
                    row,
                )
            )


            # =================================================
            # EVALUATE STRUCTURED CANDIDATE CORRECTIONS
            # =================================================

            for (
                z_adj,
                theta_adj,
                locator_offset,
                candidate_source,
            ) in candidates:


                correction = correction_profile(

                    z_adj_mm=z_adj,

                    theta_adj_deg=theta_adj,

                    locator_offset_mm=(
                        locator_offset
                    ),
                )


                candidate_state = (
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


                metrics = (
                    calculate_quality_metrics(
                        candidate_state
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


                correction_utilization = max(
                    z_util,
                    theta_util,
                    locator_util,
                )


                records.append(
                    {
                        # Identification
                        "assembly_id":
                            int(
                                assembly_id
                            ),

                        "component_index":
                            int(
                                row[
                                    "component_index"
                                ]
                            ),

                        "trajectory_type":
                            trajectory_type,

                        "candidate_source":
                            candidate_source,

                        # State
                        **current_state_features,

                        # Component deviation
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

                        # Batch/process
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

                        # Correction
                        "z_adj":
                            z_adj,

                        "theta_adj":
                            theta_adj,

                        "locator_offset":
                            locator_offset,

                        "correction_utilization":
                            correction_utilization,

                        # Targets
                        "target_mean_gap":
                            metrics[
                                "mean_gap"
                            ],

                        "target_max_gap":
                            metrics[
                                "max_gap"
                            ],

                        "target_parallelism":
                            metrics[
                                "parallelism_error"
                            ],

                        "target_rms":
                            metrics[
                                "rms_deviation"
                            ],

                        "target_quality_score":
                            metrics[
                                "quality_score"
                            ],
                    }
                )


            # =================================================
            # ADVANCE THIS TRAJECTORY
            # =================================================

            (
                trajectory_z,
                trajectory_theta,
                trajectory_locator,
            ) = trajectory_correction(
                trajectory_type,
                state,
                row,
            )


            trajectory_profile = (
                correction_profile(

                    z_adj_mm=(
                        trajectory_z
                    ),

                    theta_adj_deg=(
                        trajectory_theta
                    ),

                    locator_offset_mm=(
                        trajectory_locator
                    ),
                )
            )


            states[
                trajectory_type
            ] = (
                update_assembly_state(

                    previous_state=state,

                    component_deviation=(
                        component_profile
                    ),

                    fixture_drift=(
                        batch_disturbance
                    ),

                    correction=(
                        trajectory_profile
                    ),
                )
            )


    if (
        assembly_counter
        % 100
        == 0
    ):

        print(
            f"Processed "
            f"{assembly_counter}/"
            f"{len(assembly_ids)} assemblies"
        )


# ============================================================
# CREATE DATAFRAME
# ============================================================

training_df = pd.DataFrame(
    records
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "data/processed",
    exist_ok=True,
)


training_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# VALIDATION SUMMARY
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "CLOSED-LOOP DATASET SUMMARY"
)

print(
    "============================================================"
)


print(
    f"\nRows generated : "
    f"{len(training_df)}"
)

print(
    f"Missing values : "
    f"{training_df.isna().sum().sum()}"
)


print(
    "\nTRAJECTORY DISTRIBUTION"
)

print(
    training_df[
        "trajectory_type"
    ]
    .value_counts()
)


print(
    "\nCANDIDATE SOURCE DISTRIBUTION"
)

print(
    training_df[
        "candidate_source"
    ]
    .value_counts()
)


print(
    "\nCORRECTION UTILIZATION"
)

print(
    training_df[
        "correction_utilization"
    ]
    .describe(
        percentiles=[
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )
    .round(4)
)


print(
    "\nTARGET QUALITY"
)

print(
    training_df[
        "target_quality_score"
    ]
    .describe(
        percentiles=[
            0.05,
            0.50,
            0.90,
            0.95,
            0.99,
        ]
    )
    .round(4)
)


print(
    "\nSaved:"
)

print(
    OUTPUT_FILE
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.2 CLOSED-LOOP TRAINING DATA GENERATION COMPLETED"
)

print(
    "============================================================"
)