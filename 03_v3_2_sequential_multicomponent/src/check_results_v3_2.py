import os
import numpy as np
import pandas as pd


# ============================================================
# V3.2 RESULT CHECK
# ============================================================
#
# Checks the main validation outputs before the technical
# version is frozen.
#
# This script does not train models, tune parameters or run
# optimization.
# ============================================================


# ------------------------------------------------------------
# Main result files
# ------------------------------------------------------------

ASSEMBLY_FILE = (
    "data/validation/"
    "v3_2_final_controller_validation_assembly_results.csv"
)

COMPONENT_FILE = (
    "data/validation/"
    "v3_2_final_controller_validation_component_results.csv"
)

SUMMARY_FILE = (
    "results/validation/"
    "v3_2_final_controller_validation_summary.csv"
)

ROBUSTNESS_FILE = (
    "results/validation/"
    "v3_2_final_robustness_by_seed.csv"
)

MAGNITUDE_FILE = (
    "results/validation/"
    "v3_2_final_deviation_magnitude_sensitivity.csv"
)

CAPABILITY_FILE = (
    "results/validation/"
    "v3_2_final_correction_capability_sensitivity.csv"
)

WEIGHT_FILE = (
    "results/validation/"
    "v3_2_final_quality_weight_sensitivity.csv"
)

VALIDATION_IDS_FILE = (
    "data/validation/"
    "v3_2_independent_validation_assembly_dataset.csv"
)

RF_TRAIN_IDS_FILE = (
    "data/processed/"
    "v3_2_rf_training_assembly_ids.csv"
)

RF_HOLDOUT_IDS_FILE = (
    "data/processed/"
    "v3_2_true_rf_holdout_assembly_ids.csv"
)


FIGURE_DIR = (
    "results/final_evidence/figures"
)

TABLE_DIR = (
    "results/final_evidence/tables"
)


CHECK_OUTPUT = (
    "results/final_evidence/tables/"
    "v3_2_result_check.csv"
)


# ============================================================
# EXPECTED VALUES
# ============================================================

EXPECTED_ASSEMBLIES = 300

EXPECTED_COMPONENTS_PER_ASSEMBLY = 5

EXPECTED_DECISIONS = (
    EXPECTED_ASSEMBLIES
    *
    EXPECTED_COMPONENTS_PER_ASSEMBLY
)

EXPECTED_FIGURES = 10


# ============================================================
# HELPERS
# ============================================================

check_records = []


def record_check(
    name,
    passed,
    observed,
    expected,
):

    check_records.append(
        {
            "check":
                name,

            "status":
                "PASS"
                if passed
                else "FAIL",

            "observed":
                observed,

            "expected":
                expected,
        }
    )


def close_enough(
    a,
    b,
    tolerance=1e-8,
):

    return bool(
        np.isclose(
            float(a),
            float(b),
            atol=tolerance,
            rtol=1e-6,
        )
    )


# ============================================================
# HEADER
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "V3.2 RESULT CHECK"
)

print(
    "============================================================"
)


# ============================================================
# 1. FILE EXISTENCE
# ============================================================

required_files = [
    ASSEMBLY_FILE,
    COMPONENT_FILE,
    SUMMARY_FILE,
    ROBUSTNESS_FILE,
    MAGNITUDE_FILE,
    CAPABILITY_FILE,
    WEIGHT_FILE,
    VALIDATION_IDS_FILE,
]


for file in required_files:

    exists = os.path.exists(
        file
    )

    record_check(
        name=(
            f"File exists: {file}"
        ),
        passed=exists,
        observed=exists,
        expected=True,
    )


missing_required = [
    file
    for file in required_files
    if not os.path.exists(file)
]


if missing_required:

    print(
        "\nMissing required files:"
    )

    for file in missing_required:

        print(file)

    raise SystemExit(
        "\nCannot continue result check."
    )


# ============================================================
# LOAD DATA
# ============================================================

assembly_df = pd.read_csv(
    ASSEMBLY_FILE
)

component_df = pd.read_csv(
    COMPONENT_FILE
)

summary_df = pd.read_csv(
    SUMMARY_FILE
)

robustness_df = pd.read_csv(
    ROBUSTNESS_FILE
)

magnitude_df = pd.read_csv(
    MAGNITUDE_FILE
)

capability_df = pd.read_csv(
    CAPABILITY_FILE
)

weight_df = pd.read_csv(
    WEIGHT_FILE
)

validation_df = pd.read_csv(
    VALIDATION_IDS_FILE
)


# ============================================================
# 2. DATASET SIZE
# ============================================================

record_check(
    "Independent assembly count",
    len(
        assembly_df
    )
    ==
    EXPECTED_ASSEMBLIES,
    len(
        assembly_df
    ),
    EXPECTED_ASSEMBLIES,
)


record_check(
    "Sequential decision count",
    len(
        component_df
    )
    ==
    EXPECTED_DECISIONS,
    len(
        component_df
    ),
    EXPECTED_DECISIONS,
)


# ============================================================
# 3. UNIQUE ASSEMBLY IDS
# ============================================================

assembly_unique_ids = (
    assembly_df[
        "assembly_id"
    ]
    .nunique()
)


record_check(
    "Unique assembly IDs",
    assembly_unique_ids
    ==
    EXPECTED_ASSEMBLIES,
    assembly_unique_ids,
    EXPECTED_ASSEMBLIES,
)


# ============================================================
# 4. EXACTLY FIVE COMPONENTS PER ASSEMBLY
# ============================================================

component_counts = (
    component_df
    .groupby(
        "assembly_id"
    )
    .size()
)


all_five = bool(
    (
        component_counts
        ==
        EXPECTED_COMPONENTS_PER_ASSEMBLY
    )
    .all()
)


record_check(
    "Five decisions per assembly",
    all_five,
    (
        int(
            component_counts.min()
        ),
        int(
            component_counts.max()
        ),
    ),
    (
        EXPECTED_COMPONENTS_PER_ASSEMBLY,
        EXPECTED_COMPONENTS_PER_ASSEMBLY,
    ),
)


# ============================================================
# 5. VALIDATION IDS MATCH RESULT IDS
# ============================================================

validation_ids = set(
    validation_df[
        "assembly_id"
    ]
    .unique()
)


result_ids = set(
    assembly_df[
        "assembly_id"
    ]
    .unique()
)


id_difference = (
    validation_ids
    .symmetric_difference(
        result_ids
    )
)


record_check(
    "Validation/result assembly IDs match",
    len(
        id_difference
    )
    ==
    0,
    len(
        id_difference
    ),
    0,
)


# ============================================================
# 6. MISSING VALUES
# ============================================================

critical_assembly_columns = [
    "zero_final_quality",
    "structured20_final_quality",
    "selective_bo_final_quality",
]


critical_component_columns = [
    "zero_actual_quality",
    "structured_actual_quality",
    "selective_actual_quality",
    "structured_utilization",
    "bo_triggered",
]


assembly_missing = int(
    assembly_df[
        critical_assembly_columns
    ]
    .isna()
    .sum()
    .sum()
)


component_missing = int(
    component_df[
        critical_component_columns
    ]
    .isna()
    .sum()
    .sum()
)


record_check(
    "Critical assembly values complete",
    assembly_missing == 0,
    assembly_missing,
    0,
)


record_check(
    "Critical component values complete",
    component_missing == 0,
    component_missing,
    0,
)


# ============================================================
# 7. RECOMPUTE MAIN RESULTS
# ============================================================

zero_mean = float(
    assembly_df[
        "zero_final_quality"
    ]
    .mean()
)


structured_mean = float(
    assembly_df[
        "structured20_final_quality"
    ]
    .mean()
)


selective_mean = float(
    assembly_df[
        "selective_bo_final_quality"
    ]
    .mean()
)


structured_improvement = (
    (
        zero_mean
        -
        structured_mean
    )
    /
    zero_mean
    *
    100.0
)


structured_win_rate = (
    assembly_df[
        "structured20_final_quality"
    ]
    .lt(
        assembly_df[
            "zero_final_quality"
        ]
    )
    .mean()
    *
    100.0
)


bo_trigger_rate = (
    component_df[
        "bo_triggered"
    ]
    .mean()
    *
    100.0
)


mean_utilization = float(
    component_df[
        "structured_utilization"
    ]
    .mean()
)


near_limit_rate = (
    component_df[
        "structured_utilization"
    ]
    .ge(
        0.90
    )
    .mean()
    *
    100.0
)


exact_best_rate = (
    component_df[
        "structured_exact_best_hit"
    ]
    .mean()
    *
    100.0
)


top3_rate = (
    component_df[
        "structured_top3_hit"
    ]
    .mean()
    *
    100.0
)


# ============================================================
# 8. READ SAVED SUMMARY
# ============================================================

summary_lookup = dict(
    zip(
        summary_df[
            "metric"
        ],
        summary_df[
            "value"
        ],
    )
)


summary_checks = [

    (
        "Zero mean matches summary",
        zero_mean,
        summary_lookup.get(
            "zero_mean_final_quality",
            np.nan,
        ),
    ),

    (
        "Structured mean matches summary",
        structured_mean,
        summary_lookup.get(
            "structured20_mean_final_quality",
            np.nan,
        ),
    ),

    (
        "Selective mean matches summary",
        selective_mean,
        summary_lookup.get(
            "selective_bo_mean_final_quality",
            np.nan,
        ),
    ),

    (
        "Structured win rate matches summary",
        structured_win_rate,
        summary_lookup.get(
            "structured20_win_vs_zero_percent",
            np.nan,
        ),
    ),

    (
        "BO trigger rate matches summary",
        bo_trigger_rate,
        summary_lookup.get(
            "bo_trigger_rate_percent",
            np.nan,
        ),
    ),

    (
        "Mean utilization matches summary",
        mean_utilization,
        summary_lookup.get(
            "structured_mean_utilization",
            np.nan,
        ),
    ),

    (
        "Near-limit rate matches summary",
        near_limit_rate,
        summary_lookup.get(
            "structured_near_limit_percent",
            np.nan,
        ),
    ),

    (
        "Exact-best rate matches summary",
        exact_best_rate,
        summary_lookup.get(
            "structured_exact_best_hit_percent",
            np.nan,
        ),
    ),

    (
        "Top-3 rate matches summary",
        top3_rate,
        summary_lookup.get(
            "structured_top3_hit_percent",
            np.nan,
        ),
    ),
]


for (
    check_name,
    raw_value,
    saved_value,
) in summary_checks:

    passed = (
        np.isfinite(
            saved_value
        )
        and
        close_enough(
            raw_value,
            saved_value,
        )
    )

    record_check(
        check_name,
        passed,
        raw_value,
        saved_value,
    )


# ============================================================
# 9. BASIC QUALITY LOGIC
# ============================================================

record_check(
    "Structured mean better than Zero",
    structured_mean
    <
    zero_mean,
    structured_mean,
    f"< {zero_mean}",
)


record_check(
    "Selective mean better than Zero",
    selective_mean
    <
    zero_mean,
    selective_mean,
    f"< {zero_mean}",
)


record_check(
    "Structured assembly win rate above 90%",
    structured_win_rate
    >=
    90.0,
    structured_win_rate,
    ">= 90%",
)


# ============================================================
# 10. CAPABILITY LOGIC
# ============================================================

utilization_min = float(
    component_df[
        "structured_utilization"
    ]
    .min()
)


utilization_max = float(
    component_df[
        "structured_utilization"
    ]
    .max()
)


record_check(
    "Capability utilization is non-negative",
    utilization_min
    >=
    0.0,
    utilization_min,
    ">= 0",
)


record_check(
    "Capability utilization does not exceed 1",
    utilization_max
    <=
    1.000001,
    utilization_max,
    "<= 1",
)


# ============================================================
# 11. ROBUSTNESS FILE CHECKS
# ============================================================

minimum_seed_win = float(
    robustness_df[
        "win_vs_zero_percent"
    ]
    .min()
)


minimum_magnitude_win = float(
    magnitude_df[
        "win_vs_zero_percent"
    ]
    .min()
)


minimum_weight_win = float(
    weight_df[
        "min_win_vs_zero_percent"
    ]
    .min()
)


capability_60_rows = (
    capability_df[
        np.isclose(
            capability_df[
                "capability_factor"
            ],
            0.60,
        )
    ]
)


capability_60_exists = (
    not capability_60_rows.empty
)


record_check(
    "Three robustness seeds available",
    robustness_df[
        "seed"
    ]
    .nunique()
    ==
    3,
    robustness_df[
        "seed"
    ]
    .nunique(),
    3,
)


record_check(
    "Minimum multi-seed win rate >= 90%",
    minimum_seed_win
    >=
    90.0,
    minimum_seed_win,
    ">= 90%",
)


record_check(
    "Minimum deviation-magnitude win rate >= 90%",
    minimum_magnitude_win
    >=
    90.0,
    minimum_magnitude_win,
    ">= 90%",
)


record_check(
    "Minimum quality-weight win rate >= 90%",
    minimum_weight_win
    >=
    90.0,
    minimum_weight_win,
    ">= 90%",
)


record_check(
    "60% capability result exists",
    capability_60_exists,
    capability_60_exists,
    True,
)


if capability_60_exists:

    capability_60_win = float(
        capability_60_rows
        .iloc[
            0
        ][
            "win_vs_zero_percent"
        ]
    )


    record_check(
        "60% capability win rate >= 90%",
        capability_60_win
        >=
        90.0,
        capability_60_win,
        ">= 90%",
    )


# ============================================================
# 12. TRAIN / VALIDATION ID OVERLAP
# ============================================================
#
# Independent validation IDs were intentionally placed in a
# separate ID range. This check provides an additional record.
# ============================================================

if os.path.exists(
    RF_TRAIN_IDS_FILE
):

    train_ids_df = pd.read_csv(
        RF_TRAIN_IDS_FILE
    )


    train_ids = set(
        train_ids_df[
            "assembly_id"
        ]
        .unique()
    )


    overlap_train = (
        validation_ids
        .intersection(
            train_ids
        )
    )


    record_check(
        "Independent validation IDs overlap RF training",
        len(
            overlap_train
        )
        ==
        0,
        len(
            overlap_train
        ),
        0,
    )


else:

    record_check(
        "RF training ID file available",
        False,
        "missing",
        RF_TRAIN_IDS_FILE,
    )


if os.path.exists(
    RF_HOLDOUT_IDS_FILE
):

    holdout_ids_df = pd.read_csv(
        RF_HOLDOUT_IDS_FILE
    )


    holdout_ids = set(
        holdout_ids_df[
            "assembly_id"
        ]
        .unique()
    )


    overlap_holdout = (
        validation_ids
        .intersection(
            holdout_ids
        )
    )


    record_check(
        "Independent validation IDs overlap RF holdout",
        len(
            overlap_holdout
        )
        ==
        0,
        len(
            overlap_holdout
        ),
        0,
    )


else:

    record_check(
        "RF holdout ID file available",
        False,
        "missing",
        RF_HOLDOUT_IDS_FILE,
    )


# ============================================================
# 13. EVIDENCE FIGURES
# ============================================================

if os.path.exists(
    FIGURE_DIR
):

    figure_files = [
        file
        for file in os.listdir(
            FIGURE_DIR
        )
        if file.lower().endswith(
            ".png"
        )
    ]

else:

    figure_files = []


record_check(
    "Evidence figure count",
    len(
        figure_files
    )
    >=
    EXPECTED_FIGURES,
    len(
        figure_files
    ),
    f">= {EXPECTED_FIGURES}",
)


# ============================================================
# 14. EVIDENCE TABLES
# ============================================================

if os.path.exists(
    TABLE_DIR
):

    table_files = [
        file
        for file in os.listdir(
            TABLE_DIR
        )
        if file.lower().endswith(
            ".csv"
        )
    ]

else:

    table_files = []


record_check(
    "Evidence tables available",
    len(
        table_files
    )
    >=
    10,
    len(
        table_files
    ),
    ">= 10",
)


# ============================================================
# SAVE CHECK TABLE
# ============================================================

check_df = pd.DataFrame(
    check_records
)


os.makedirs(
    TABLE_DIR,
    exist_ok=True,
)


check_df.to_csv(
    CHECK_OUTPUT,
    index=False,
)


# ============================================================
# FINAL STATUS
# ============================================================

n_checks = len(
    check_df
)


n_pass = int(
    (
        check_df[
            "status"
        ]
        ==
        "PASS"
    )
    .sum()
)


n_fail = int(
    (
        check_df[
            "status"
        ]
        ==
        "FAIL"
    )
    .sum()
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "RECOMPUTED MAIN RESULTS"
)

print(
    "------------------------------------------------------------"
)


print(
    f"\nZero mean quality          : "
    f"{zero_mean:.6f}"
)


print(
    f"Structured mean quality    : "
    f"{structured_mean:.6f}"
)


print(
    f"Selective BO mean quality  : "
    f"{selective_mean:.6f}"
)


print(
    f"Structured improvement     : "
    f"{structured_improvement:.2f}%"
)


print(
    f"Structured win rate        : "
    f"{structured_win_rate:.2f}%"
)


print(
    f"BO trigger rate            : "
    f"{bo_trigger_rate:.2f}%"
)


print(
    f"Exact-best rate            : "
    f"{exact_best_rate:.2f}%"
)


print(
    f"Top-3 rate                 : "
    f"{top3_rate:.2f}%"
)


print(
    f"Mean capability utilization: "
    f"{mean_utilization:.4f}"
)


print(
    f">=90% capability rate      : "
    f"{near_limit_rate:.2f}%"
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "CHECK SUMMARY"
)

print(
    "------------------------------------------------------------"
)


print(
    f"\nChecks performed : "
    f"{n_checks}"
)


print(
    f"Passed           : "
    f"{n_pass}"
)


print(
    f"Failed           : "
    f"{n_fail}"
)


if n_fail == 0:

    print(
        "\nRESULT: PASS"
    )

    print(
        "The main V3.2 validation outputs are internally "
        "consistent and ready for documentation/freeze."
    )


else:

    print(
        "\nRESULT: REVIEW REQUIRED"
    )

    print(
        "At least one consistency check failed."
    )


    print(
        "\nFailed checks:"
    )


    print(
        check_df[
            check_df[
                "status"
            ]
            ==
            "FAIL"
        ]
        .to_string(
            index=False
        )
    )


print(
    "\nSaved:"
)


print(
    CHECK_OUTPUT
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.2 RESULT CHECK COMPLETED"
)

print(
    "============================================================"
)