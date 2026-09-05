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
# VERSION 3.2
# PROFILE-AWARE CLOSED-LOOP TRAINING DATA
# ============================================================
#
# PURPOSE
#
# Previous V3.2 closed-loop training represented the current
# sequential assembly state using scalar descriptors:
#
#   mean gap
#   max gap
#   parallelism
#   RMS
#   quality score
#   signed mean
#   signed end difference
#   estimated angle
#
# However, the actual assembly state is a 101-point geometric
# profile:
#
#       S_k(s)
#
# Different profile shapes can have similar scalar metrics but
# may require different corrections.
#
# This script therefore PRESERVES all existing closed-loop
# trajectories, candidate generation, targets and correction
# logic, while adding a compact signed geometric representation
# of the current state.
#
#
# PROFILE REPRESENTATION
#
# 11 equally spaced points are sampled from the 101-point state:
#
#   0%, 10%, 20%, ... 100%
#
# This is intentionally compact.
#
# We do NOT yet use all 101 points and do NOT yet use PCA.
#
# The purpose is to perform a controlled representation test:
#
#   Scalar features
#
#        versus
#
#   Scalar features + sampled profile geometry
#
#
# IMPORTANT
#
# The original closed-loop dataset is preserved.
# ============================================================


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = (
    "data/processed/"
    "v3_2_full_component_level_dataset.csv"
)

OUTPUT_FILE = (
    "data/processed/"
    "v3_2_profile_aware_ml_training_dataset.csv"
)


# ============================================================
# FILE CHECK
# ============================================================

if not os.path.exists(
    INPUT_FILE
):

    raise FileNotFoundError(
        "\nFull V3.2 component dataset not found."
    )


# ============================================================
# RANDOM SEED
# ============================================================

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
# PROFILE-AWARE STATE SAMPLE LOCATIONS
# ============================================================
#
# sequential_assembly.py uses 101 state points.
#
# Therefore:
#
# index 0   = 0%
# index 10  = 10%
# ...
# index 100 = 100%
#
# These are SIGNED values.
#
# Signed geometry is important because +deviation and
# -deviation may require opposite corrective action even if
# their absolute quality metrics are similar.
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
# SANITY CHECK PROFILE SIZE
# ============================================================

if len(
    s
) != 101:

    raise ValueError(
        "\nExpected sequential assembly profile to contain "
        "101 points, but found "
        f"{len(s)}."
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


    if row[
        "offset_mm"
    ] != 0.0:

        profile += deviation_offset(
            offset_mm=row[
                "offset_mm"
            ]
        )


    if row[
        "tilt_deg"
    ] != 0.0:

        profile += deviation_tilt(
            angle_deg=row[
                "tilt_deg"
            ]
        )


    if row[
        "bend_mm"
    ] != 0.0:

        profile += deviation_bend(
            amplitude_mm=row[
                "bend_mm"
            ]
        )


    if row[
        "waviness_mm"
    ] != 0.0:

        profile += deviation_waviness(
            amplitude_mm=row[
                "waviness_mm"
            ],
            waves=3,
        )


    if row[
        "twist_mm"
    ] != 0.0:

        profile += deviation_twist(
            amplitude_mm=row[
                "twist_mm"
            ]
        )


    if row[
        "local_bump_mm"
    ] != 0.0:

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
                **
                2
            )
            /
            (
                2.0
                *
                fixture_sigma
                **
                2
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
# ORIGINAL SCALAR STATE FEATURES
# ============================================================

def scalar_state_features(
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


    angle_deg = float(
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
            angle_deg,
    }


# ============================================================
# PROFILE-AWARE STATE FEATURES
# ============================================================

def profile_state_features(
    state
):
    """
    Return signed geometric samples from the current state.

    These features retain information about the spatial shape
    of the accumulated sequential assembly deviation.
    """

    features = {}


    for (
        profile_name,
        profile_index,
    ) in zip(
        PROFILE_SAMPLE_NAMES,
        PROFILE_SAMPLE_INDICES,
    ):

        features[
            profile_name
        ] = float(
            state[
                profile_index
            ]
        )


    return features


# ============================================================
# COMBINED STATE FEATURES
# ============================================================

def state_features(
    state
):

    return {

        **scalar_state_features(
            state
        ),

        **profile_state_features(
            state
        ),
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
# STRUCTURED CANDIDATE GENERATION
# ============================================================

def generate_candidate_set(
    state,
    row,
):
    """
    Preserve the exact structured candidate philosophy from
    the previous closed-loop training-data generator.

    Candidate groups:

        1. zero
        2. informed
        3. local informed perturbations
        4. moderate random
        5. broad random
        6. near-boundary
    """

    candidates = []


    # ========================================================
    # 1. ZERO
    # ========================================================

    candidates.append(
        (
            0.0,
            0.0,
            0.0,
            "zero",
        )
    )


    # ========================================================
    # 2. INFORMED
    # ========================================================

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


    # ========================================================
    # 3. LOCAL PERTURBATIONS
    # ========================================================

    for _ in range(
        8
    ):

        z_adj = (
            informed_z
            +
            RNG.normal(
                0.0,
                0.30,
            )
        )


        theta_adj = (
            informed_theta
            +
            RNG.normal(
                0.0,
                0.12,
            )
        )


        locator_offset = (
            informed_locator
            +
            RNG.normal(
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


    # ========================================================
    # 4. MODERATE RANDOM
    # ========================================================

    for _ in range(
        6
    ):

        z_adj = RNG.uniform(
            -0.60
            *
            Z_LIMIT,
            0.60
            *
            Z_LIMIT,
        )


        theta_adj = RNG.uniform(
            -0.60
            *
            THETA_LIMIT,
            0.60
            *
            THETA_LIMIT,
        )


        locator_offset = RNG.uniform(
            -0.60
            *
            LOCATOR_LIMIT,
            0.60
            *
            LOCATOR_LIMIT,
        )


        candidates.append(
            (
                float(
                    z_adj
                ),

                float(
                    theta_adj
                ),

                float(
                    locator_offset
                ),

                "moderate_random",
            )
        )


    # ========================================================
    # 5. BROAD RANDOM
    # ========================================================

    for _ in range(
        4
    ):

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


    # ========================================================
    # 6. NEAR BOUNDARY
    # ========================================================

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
                boundary_signs[
                    0
                ]
                *
                0.90
                *
                Z_LIMIT
            ),

            float(
                boundary_signs[
                    1
                ]
                *
                0.90
                *
                THETA_LIMIT
            ),

            float(
                boundary_signs[
                    2
                ]
                *
                0.90
                *
                LOCATOR_LIMIT
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
            0.50
            *
            informed_z,

            0.50
            *
            informed_theta,

            0.50
            *
            informed_locator,
        )


    if trajectory_type == "random_feasible":

        return (

            float(
                RNG.uniform(
                    -0.50
                    *
                    Z_LIMIT,
                    0.50
                    *
                    Z_LIMIT,
                )
            ),

            float(
                RNG.uniform(
                    -0.50
                    *
                    THETA_LIMIT,
                    0.50
                    *
                    THETA_LIMIT,
                )
            ),

            float(
                RNG.uniform(
                    -0.50
                    *
                    LOCATOR_LIMIT,
                    0.50
                    *
                    LOCATOR_LIMIT,
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
    ]
    .unique()
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.2 PROFILE-AWARE CLOSED-LOOP DATA GENERATOR"
)

print(
    "============================================================"
)


print(
    f"\nAssemblies        : "
    f"{len(assembly_ids)}"
)


print(
    f"Trajectory types  : "
    f"{len(TRAJECTORY_TYPES)}"
)


print(
    f"Profile features  : "
    f"{len(PROFILE_SAMPLE_NAMES)}"
)


print(
    "\nProfile sample names:"
)


for name in PROFILE_SAMPLE_NAMES:

    print(
        f"  {name}"
    )


# ============================================================
# STORAGE
# ============================================================

records = []


# ============================================================
# MAIN LOOP
# ============================================================

for (
    assembly_counter,
    assembly_id,
) in enumerate(
    assembly_ids,
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


    # --------------------------------------------------------
    # Independent state per trajectory.
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
        # EACH TRAJECTORY
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
            # EVALUATE CANDIDATES
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
                    abs(
                        z_adj
                    )
                    /
                    Z_LIMIT
                )


                theta_util = (
                    abs(
                        theta_adj
                    )
                    /
                    THETA_LIMIT
                )


                locator_util = (
                    abs(
                        locator_offset
                    )
                    /
                    LOCATOR_LIMIT
                )


                correction_utilization = max(
                    z_util,
                    theta_util,
                    locator_util,
                )


                records.append(
                    {

                        # ====================================
                        # IDENTIFICATION
                        # ====================================

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


                        # ====================================
                        # CURRENT STATE
                        #
                        # Includes:
                        #
                        # scalar features
                        # +
                        # 11 signed profile points
                        # ====================================

                        **current_state_features,


                        # ====================================
                        # COMPONENT DEVIATION
                        # ====================================

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


                        # ====================================
                        # BATCH / PROCESS
                        # ====================================

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


                        # ====================================
                        # CANDIDATE CORRECTION
                        # ====================================

                        "z_adj":
                            z_adj,

                        "theta_adj":
                            theta_adj,

                        "locator_offset":
                            locator_offset,

                        "correction_utilization":
                            correction_utilization,


                        # ====================================
                        # TARGETS
                        # ====================================

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
            # ADVANCE CURRENT TRAJECTORY
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


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if (
        assembly_counter
        %
        100
        ==
        0
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
# VALIDATION CHECKS BEFORE SAVE
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "PROFILE-AWARE DATASET VALIDATION"
)

print(
    "============================================================"
)


expected_profile_columns_missing = [

    column

    for column
    in PROFILE_SAMPLE_NAMES

    if column
    not in training_df.columns
]


if expected_profile_columns_missing:

    raise ValueError(
        "\nERROR - Missing profile features:\n"
        +
        "\n".join(
            expected_profile_columns_missing
        )
    )


missing_values = int(
    training_df
    .isna()
    .sum()
    .sum()
)


if missing_values != 0:

    raise ValueError(
        f"\nERROR - Dataset contains "
        f"{missing_values} missing values."
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
# SUMMARY
# ============================================================

print(
    "\nRows generated : "
    f"{len(training_df)}"
)


print(
    "Columns        : "
    f"{len(training_df.columns)}"
)


print(
    "Missing values : "
    f"{missing_values}"
)


print(
    f"Profile columns : "
    f"{len(PROFILE_SAMPLE_NAMES)}"
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
    "\nPROFILE FEATURE RANGE"
)


profile_summary = (
    training_df[
        PROFILE_SAMPLE_NAMES
    ]
    .agg(
        [
            "min",
            "mean",
            "std",
            "max",
        ]
    )
    .T
)


print(
    profile_summary
    .round(
        4
    )
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
    .round(
        4
    )
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
    .round(
        4
    )
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
    "V3.2 PROFILE-AWARE TRAINING DATA GENERATION COMPLETED"
)

print(
    "============================================================"
)