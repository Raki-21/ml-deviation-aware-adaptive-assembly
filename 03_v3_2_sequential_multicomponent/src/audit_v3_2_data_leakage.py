import os
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# V3.2
# DATA-LEAKAGE / HOLDOUT AUDIT
# ============================================================
#
# PURPOSE
#
# The profile-aware RF was trained using:
#
#   GroupShuffleSplit
#   test_size = 0.20
#   random_state = 42
#
# on the profile-aware training dataset.
#
# Later controller experiments used a saved set of 100 pilot
# assemblies.
#
# This script reconstructs the exact RF train/test assembly
# split and checks whether those 100 controller assemblies
# belong to:
#
#   - RF training assemblies
#   - RF holdout assemblies
#
#
# This audit MUST be completed before final validation.
# ============================================================


PROFILE_DATA_FILE = (
    "data/processed/"
    "v3_2_profile_aware_ml_training_dataset.csv"
)

PILOT_FILE = (
    "data/processed/"
    "v3_2_ml_bo_pilot_assembly_results.csv"
)

STRUCTURED_FILE = (
    "data/processed/"
    "v3_2_equal_budget_structured20_assembly_results.csv"
)


OUTPUT_FILE = (
    "results/tables/"
    "v3_2_data_leakage_holdout_audit.csv"
)

HOLDOUT_IDS_FILE = (
    "data/processed/"
    "v3_2_true_rf_holdout_assembly_ids.csv"
)

TRAIN_IDS_FILE = (
    "data/processed/"
    "v3_2_rf_training_assembly_ids.csv"
)


required_files = [
    PROFILE_DATA_FILE,
    PILOT_FILE,
    STRUCTURED_FILE,
]


missing_files = [
    path
    for path in required_files
    if not os.path.exists(path)
]


if missing_files:

    print("\nERROR - Missing files:")

    for path in missing_files:
        print(path)

    raise SystemExit(
        "\nRequired V3.2 files are missing."
    )


print(
    "\n"
    "============================================================"
)

print(
    "V3.2 DATA-LEAKAGE / HOLDOUT AUDIT"
)

print(
    "============================================================"
)


# ============================================================
# LOAD PROFILE-AWARE DATASET
# ============================================================

df = pd.read_csv(
    PROFILE_DATA_FILE,
    usecols=[
        "assembly_id",
    ],
)


groups = df[
    "assembly_id"
]


# ============================================================
# RECONSTRUCT EXACT MODEL SPLIT
# ============================================================

splitter = GroupShuffleSplit(

    n_splits=1,

    test_size=0.20,

    random_state=42,
)


train_idx, test_idx = next(

    splitter.split(
        df,
        groups=groups,
    )
)


train_ids = set(
    df.iloc[
        train_idx
    ][
        "assembly_id"
    ]
    .unique()
)


holdout_ids = set(
    df.iloc[
        test_idx
    ][
        "assembly_id"
    ]
    .unique()
)


split_overlap = (
    train_ids
    .intersection(
        holdout_ids
    )
)


print(
    f"\nRF training assemblies : "
    f"{len(train_ids)}"
)


print(
    f"RF holdout assemblies  : "
    f"{len(holdout_ids)}"
)


print(
    f"Train/test overlap     : "
    f"{len(split_overlap)}"
)


if len(
    split_overlap
) != 0:

    raise RuntimeError(
        "\nERROR - Reconstructed GroupShuffleSplit "
        "contains assembly overlap."
    )


# ============================================================
# CONTROLLER PILOT IDs
# ============================================================

pilot_df = pd.read_csv(
    PILOT_FILE
)


structured_df = pd.read_csv(
    STRUCTURED_FILE
)


pilot_ids = set(
    pilot_df[
        "assembly_id"
    ]
    .unique()
)


structured_ids = set(
    structured_df[
        "assembly_id"
    ]
    .unique()
)


print(
    f"\nOriginal pilot assemblies     : "
    f"{len(pilot_ids)}"
)


print(
    f"Structured-20 assemblies     : "
    f"{len(structured_ids)}"
)


pilot_vs_structured_difference = (
    pilot_ids
    .symmetric_difference(
        structured_ids
    )
)


print(
    f"Pilot/Structured ID mismatch : "
    f"{len(pilot_vs_structured_difference)}"
)


# ============================================================
# OVERLAP AUDIT
# ============================================================

pilot_seen_in_training = (
    pilot_ids
    .intersection(
        train_ids
    )
)


pilot_true_holdout = (
    pilot_ids
    .intersection(
        holdout_ids
    )
)


n_pilot = len(
    pilot_ids
)


n_seen = len(
    pilot_seen_in_training
)


n_holdout = len(
    pilot_true_holdout
)


seen_percent = (
    n_seen
    /
    max(
        n_pilot,
        1,
    )
    *
    100.0
)


holdout_percent = (
    n_holdout
    /
    max(
        n_pilot,
        1,
    )
    *
    100.0
)


print(
    "\n"
    "============================================================"
)

print(
    "CONTROLLER PILOT vs RF TRAINING SPLIT"
)

print(
    "============================================================"
)


print(
    f"\nPilot assemblies seen during RF training : "
    f"{n_seen}/{n_pilot}"
    f" ({seen_percent:.2f}%)"
)


print(
    f"Pilot assemblies in true RF holdout      : "
    f"{n_holdout}/{n_pilot}"
    f" ({holdout_percent:.2f}%)"
)


# ============================================================
# SAVE TRAIN / HOLDOUT IDS
# ============================================================

os.makedirs(
    "data/processed",
    exist_ok=True,
)


os.makedirs(
    "results/tables",
    exist_ok=True,
)


pd.DataFrame(
    {
        "assembly_id":
            sorted(
                train_ids
            )
    }
).to_csv(
    TRAIN_IDS_FILE,
    index=False,
)


pd.DataFrame(
    {
        "assembly_id":
            sorted(
                holdout_ids
            )
    }
).to_csv(
    HOLDOUT_IDS_FILE,
    index=False,
)


# ============================================================
# AUDIT SUMMARY
# ============================================================

audit_df = pd.DataFrame(
    {

        "metric": [

            "rf_training_assemblies",

            "rf_holdout_assemblies",

            "rf_train_test_overlap",

            "pilot_assemblies",

            "pilot_seen_during_rf_training",

            "pilot_seen_during_rf_training_percent",

            "pilot_true_holdout_assemblies",

            "pilot_true_holdout_percent",

            "pilot_structured_id_mismatch",
        ],

        "value": [

            len(
                train_ids
            ),

            len(
                holdout_ids
            ),

            len(
                split_overlap
            ),

            n_pilot,

            n_seen,

            seen_percent,

            n_holdout,

            holdout_percent,

            len(
                pilot_vs_structured_difference
            ),
        ],
    }
)


audit_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# VERDICT
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "HOLDOUT VALIDITY VERDICT"
)

print(
    "============================================================"
)


if n_seen == 0:

    print(
        "\nRESULT A:"
    )

    print(
        "PASS - All 100 controller assemblies are unseen "
        "with respect to RF model training."
    )

    print(
        "\nThe existing 100-assembly controller results may "
        "be treated as genuine model-holdout evidence."
    )


elif n_holdout >= 50:

    print(
        "\nRESULT B:"
    )

    print(
        "MIXED - The existing 100-controller pilot contains "
        "both RF-training and RF-holdout assemblies."
    )

    print(
        "\nThe current controller results remain useful for "
        "development, but they must NOT be used as the final "
        "unseen-assembly validation."
    )

    print(
        "\nNEXT STEP:"
    )

    print(
        "Run the final frozen controller only on the saved "
        "true RF holdout assemblies."
    )


else:

    print(
        "\nRESULT C:"
    )

    print(
        "WARNING - Most controller pilot assemblies were "
        "already represented during RF training."
    )

    print(
        "\nThe current controller experiments are development "
        "evidence only."
    )

    print(
        "\nNEXT STEP:"
    )

    print(
        "Use the saved 200-assembly RF holdout set for final "
        "controller validation."
    )


print(
    "\nSaved:"
)

print(
    OUTPUT_FILE
)

print(
    HOLDOUT_IDS_FILE
)

print(
    TRAIN_IDS_FILE
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.2 DATA-LEAKAGE AUDIT COMPLETED"
)

print(
    "============================================================"
)