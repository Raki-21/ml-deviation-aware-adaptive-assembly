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
# VERSION 3.2 - ML SURROGATE TRAINING DATA GENERATOR
# ============================================================
#
# Goal:
#
# Learn the mapping:
#
# CURRENT ASSEMBLY STATE
#       +
# INCOMING COMPONENT DEVIATION
#       +
# BATCH / PROCESS CONDITION
#       +
# CANDIDATE CORRECTION
#       ↓
# RESULTING ASSEMBLY QUALITY
#
# This creates the dataset that will later be used for
# model screening:
#
#   Linear Regression
#   Ridge Regression
#   Random Forest
#   Gradient Boosting
#   SVR
#
# Random Forest is NOT assumed to win.
# We re-test model suitability in the new sequential problem.
# ============================================================


# ------------------------------------------------------------
# INPUT
# ------------------------------------------------------------

INPUT_FILE = (
    "data/processed/"
    "v3_2_full_component_level_dataset.csv"
)

OUTPUT_FILE = (
    "data/processed/"
    "v3_2_ml_training_dataset.csv"
)


if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        "\nFull V3.2 dataset not found.\n"
        "Run generate_full_v3_2_dataset.py first."
    )


# ------------------------------------------------------------
# RANDOM GENERATOR
# ------------------------------------------------------------

RNG = np.random.default_rng(20260825)


# ------------------------------------------------------------
# NUMBER OF CORRECTION CANDIDATES PER COMPONENT STEP
# ------------------------------------------------------------
#
# 5000 component states x 15 candidate corrections
# = approximately 75,000 ML samples.
#
# Large enough for model learning but still manageable.
# ------------------------------------------------------------

N_CANDIDATES_PER_STEP = 15


# ------------------------------------------------------------
# CURRENT CORRECTION CAPABILITY
# ------------------------------------------------------------

Z_MIN = -2.5
Z_MAX = 2.5

THETA_MIN = -1.2
THETA_MAX = 1.2

LOCATOR_MIN = -1.0
LOCATOR_MAX = 1.0


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def reconstruct_component_profile(row):
    """
    Rebuild one incoming component profile from the stored
    physically interpretable deviation features.
    """

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


def reconstruct_batch_disturbance(row):
    """
    Reconstruct the process-level disturbance.
    """

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


def estimate_state_features(state):
    """
    Extract interpretable descriptors of the current
    partially assembled product.
    """

    metrics = calculate_quality_metrics(
        state
    )

    signed_mean = float(
        np.mean(state)
    )

    signed_end_difference = float(
        state[-1] - state[0]
    )

    estimated_angle_deg = float(
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
            estimated_angle_deg,
    }


def sample_candidate_correction():
    """
    Sample one feasible candidate correction from the
    currently defined V3.1/V3.2 correction space.
    """

    z_adj = RNG.uniform(
        Z_MIN,
        Z_MAX,
    )

    theta_adj = RNG.uniform(
        THETA_MIN,
        THETA_MAX,
    )

    locator_offset = RNG.uniform(
        LOCATOR_MIN,
        LOCATOR_MAX,
    )

    return (
        float(z_adj),
        float(theta_adj),
        float(locator_offset),
    )


# ============================================================
# LOAD FULL CONTROLLED DATASET
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.2 ML TRAINING DATA GENERATOR"
)

print(
    "============================================================"
)

print(
    f"\nSequential component states : {len(df)}"
)

print(
    f"Candidate corrections/state : "
    f"{N_CANDIDATES_PER_STEP}"
)

print(
    f"Expected ML rows            : "
    f"{len(df) * N_CANDIDATES_PER_STEP}"
)


# ============================================================
# STORAGE
# ============================================================

training_records = []


# ============================================================
# REPLAY EACH COMPLETE ASSEMBLY
# ============================================================

assembly_ids = sorted(
    df[
        "assembly_id"
    ].unique()
)


for assembly_counter, assembly_id in enumerate(
    assembly_ids,
    start=1,
):

    assembly_rows = (
        df[
            df["assembly_id"]
            == assembly_id
        ]
        .sort_values(
            "component_index"
        )
    )

    # --------------------------------------------------------
    # CURRENT PHYSICAL ASSEMBLY STATE
    # --------------------------------------------------------
    #
    # For training-data generation, the sequential reference
    # trajectory is reconstructed from the uncorrected
    # controlled assembly state.
    #
    # Candidate corrections are evaluated from each state,
    # but do not permanently replace this reference state.
    # --------------------------------------------------------

    state = create_initial_state()


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


        # ----------------------------------------------------
        # STATE FEATURES BEFORE ADDING CURRENT COMPONENT
        # ----------------------------------------------------

        state_features = (
            estimate_state_features(
                state
            )
        )


        # ====================================================
        # SAMPLE MANY POSSIBLE CORRECTIONS
        # ====================================================

        for candidate_id in range(
            1,
            N_CANDIDATES_PER_STEP + 1,
        ):

            (
                z_adj,
                theta_adj,
                locator_offset,
            ) = sample_candidate_correction()


            candidate_correction = (
                correction_profile(
                    z_adj_mm=z_adj,
                    theta_adj_deg=theta_adj,
                    locator_offset_mm=locator_offset,
                )
            )


            # ------------------------------------------------
            # TEMPORARY NEXT STATE FOR THIS CANDIDATE
            # ------------------------------------------------

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
                        candidate_correction
                    ),
                )
            )


            candidate_metrics = (
                calculate_quality_metrics(
                    candidate_state
                )
            )


            # ------------------------------------------------
            # CORRECTION UTILIZATION
            # ------------------------------------------------

            z_util = (
                abs(z_adj)
                / Z_MAX
            )

            theta_util = (
                abs(theta_adj)
                / THETA_MAX
            )

            locator_util = (
                abs(locator_offset)
                / LOCATOR_MAX
            )

            max_util = max(
                z_util,
                theta_util,
                locator_util,
            )


            # =================================================
            # STORE ML SAMPLE
            # =================================================

            training_records.append(
                {
                    # Identification
                    "assembly_id":
                        int(assembly_id),

                    "component_index":
                        int(
                            row[
                                "component_index"
                            ]
                        ),

                    "candidate_id":
                        candidate_id,

                    # ------------------------------------------------
                    # Current assembly state
                    # ------------------------------------------------

                    **state_features,

                    # ------------------------------------------------
                    # Incoming component deviation
                    # ------------------------------------------------

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

                    # ------------------------------------------------
                    # Batch/process condition
                    # ------------------------------------------------

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

                    # ------------------------------------------------
                    # Candidate assembly correction
                    # ------------------------------------------------

                    "z_adj":
                        z_adj,

                    "theta_adj":
                        theta_adj,

                    "locator_offset":
                        locator_offset,

                    "correction_utilization":
                        max_util,

                    # ------------------------------------------------
                    # ML TARGETS
                    # ------------------------------------------------

                    "target_mean_gap":
                        candidate_metrics[
                            "mean_gap"
                        ],

                    "target_max_gap":
                        candidate_metrics[
                            "max_gap"
                        ],

                    "target_parallelism":
                        candidate_metrics[
                            "parallelism_error"
                        ],

                    "target_rms":
                        candidate_metrics[
                            "rms_deviation"
                        ],

                    "target_quality_score":
                        candidate_metrics[
                            "quality_score"
                        ],
                }
            )


        # ----------------------------------------------------
        # UPDATE REFERENCE SEQUENTIAL STATE WITHOUT CORRECTION
        # ----------------------------------------------------
        #
        # This keeps the dataset-generation state progression
        # deterministic and avoids contaminating the current
        # assembly state with one arbitrary candidate.
        # ----------------------------------------------------

        state = update_assembly_state(
            previous_state=state,
            component_deviation=(
                component_profile
            ),
            fixture_drift=(
                batch_disturbance
            ),
        )


    # --------------------------------------------------------
    # PROGRESS OUTPUT
    # --------------------------------------------------------

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
# BUILD DATAFRAME
# ============================================================

training_df = pd.DataFrame(
    training_records
)


# ============================================================
# SAVE DATASET
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
# BASIC VALIDATION
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "ML TRAINING DATA SUMMARY"
)

print(
    "============================================================"
)


print(
    f"\nGenerated rows : "
    f"{len(training_df)}"
)

print(
    f"Columns        : "
    f"{len(training_df.columns)}"
)

print(
    f"Missing values : "
    f"{training_df.isna().sum().sum()}"
)


print(
    "\nTARGET QUALITY SCORE"
)

print(
    training_df[
        "target_quality_score"
    ]
    .describe(
        percentiles=[
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
        ]
    )
    .round(4)
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
            0.90,
            0.95,
            0.99,
        ]
    )
    .round(4)
)


print(
    f"\nSaved to:\n"
    f"{OUTPUT_FILE}"
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.2 ML TRAINING DATA GENERATION COMPLETED"
)

print(
    "============================================================"
)