import os
import pandas as pd
import numpy as np


# ============================================================
# FINAL SCIENTIFIC AUDIT CONSISTENCY CHECK
# ============================================================
#
# Purpose:
# Verify that all final scientific-audit outputs exist and
# remain internally consistent before the technical package
# is frozen.
#
# This script does not train models or modify scientific
# result files.
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

RESULTS_DIR = os.path.join(
    AUDIT_DIR,
    "results",
)


# ============================================================
# STORAGE
# ============================================================

checks = []


def record_check(
    name,
    passed,
    detail,
):

    checks.append(
        {
            "check": name,
            "passed": bool(passed),
            "detail": str(detail),
        }
    )


# ============================================================
# HELPERS
# ============================================================

def metric_lookup(
    dataframe
):

    if (
        "metric" not in dataframe.columns
        or
        "value" not in dataframe.columns
    ):

        raise ValueError(
            "\nExpected metric/value table structure not found."
        )

    return dict(
        zip(
            dataframe[
                "metric"
            ],
            dataframe[
                "value"
            ],
        )
    )


def finite_number(
    value
):

    try:

        return bool(
            np.isfinite(
                float(
                    value
                )
            )
        )

    except (
        TypeError,
        ValueError,
    ):

        return False


# ============================================================
# FILE LOCATIONS
# ============================================================

UNSEEN_BATCH_FILE = os.path.join(
    RESULTS_DIR,
    "unseen_batch_validation",
    "unseen_batch_summary.csv",
)


DETERMINISTIC_FILE = os.path.join(
    RESULTS_DIR,
    "deterministic_baseline",
    "deterministic_baseline_summary.csv",
)


ACHIEVABLE_FILE = os.path.join(
    RESULTS_DIR,
    "achievable_correction",
    "achievable_correction_summary.csv",
)


OBSERVABLE_FILE = os.path.join(
    RESULTS_DIR,
    "observable_feature_audit",
    "observable_feature_summary.csv",
)


LOOKAHEAD_FILE = os.path.join(
    RESULTS_DIR,
    "two_step_lookahead",
    "two_step_summary.csv",
)


ML_VALUE_FILE = os.path.join(
    RESULTS_DIR,
    "ml_value_regions",
    "ml_value_summary.csv",
)


BASIS_FILE = os.path.join(
    RESULTS_DIR,
    "correction_basis_residual",
    "basis_residual_summary.csv",
)


BASIS_QUARTILE_FILE = os.path.join(
    RESULTS_DIR,
    "correction_basis_residual",
    "basis_residual_by_quartile.csv",
)


TRADEOFF_FILE = os.path.join(
    RESULTS_DIR,
    "method_tradeoffs",
    "method_tradeoff_summary.csv",
)


TRADEOFF_TABLE_FILE = os.path.join(
    RESULTS_DIR,
    "method_tradeoffs",
    "method_tradeoff_table.csv",
)


required_files = {

    "unseen_batch":
        UNSEEN_BATCH_FILE,

    "deterministic":
        DETERMINISTIC_FILE,

    "achievable":
        ACHIEVABLE_FILE,

    "observable":
        OBSERVABLE_FILE,

    "lookahead":
        LOOKAHEAD_FILE,

    "ml_value":
        ML_VALUE_FILE,

    "basis_residual":
        BASIS_FILE,

    "basis_quartile":
        BASIS_QUARTILE_FILE,

    "method_tradeoff":
        TRADEOFF_FILE,

    "method_tradeoff_table":
        TRADEOFF_TABLE_FILE,
}


# ============================================================
# START
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "FINAL SCIENTIFIC AUDIT CONSISTENCY CHECK"
)

print(
    "============================================================"
)


# ============================================================
# FILE EXISTENCE CHECKS
# ============================================================

for name, path in required_files.items():

    exists = os.path.exists(
        path
    )

    record_check(
        f"{name}_file_exists",
        exists,
        path,
    )

    print(
        f"\n{name:24s}: "
        f"{'FOUND' if exists else 'MISSING'}"
    )


if not all(
    os.path.exists(
        path
    )
    for path in required_files.values()
):

    raise FileNotFoundError(
        "\nOne or more final audit output files are missing."
    )


# ============================================================
# LOAD RESULTS
# ============================================================

unseen_df = pd.read_csv(
    UNSEEN_BATCH_FILE
)

deterministic_df = pd.read_csv(
    DETERMINISTIC_FILE
)

achievable_df = pd.read_csv(
    ACHIEVABLE_FILE
)

observable_df = pd.read_csv(
    OBSERVABLE_FILE
)

lookahead_df = pd.read_csv(
    LOOKAHEAD_FILE
)

ml_value_df = pd.read_csv(
    ML_VALUE_FILE
)

basis_df = pd.read_csv(
    BASIS_FILE
)

basis_quartile_df = pd.read_csv(
    BASIS_QUARTILE_FILE
)

tradeoff_df = pd.read_csv(
    TRADEOFF_FILE
)

tradeoff_table_df = pd.read_csv(
    TRADEOFF_TABLE_FILE
)


# ============================================================
# METRIC TABLE LOOKUPS
# ============================================================

det = metric_lookup(
    deterministic_df
)

ach = metric_lookup(
    achievable_df
)

look = metric_lookup(
    lookahead_df
)

mlv = metric_lookup(
    ml_value_df
)

basis = metric_lookup(
    basis_df
)

trade = metric_lookup(
    tradeoff_df
)


# ============================================================
# AUDIT 1
# UNSEEN BATCH / PROCESS-SIGNATURE VALIDATION
# ============================================================

if (
    "metric" not in unseen_df.columns
    or
    "overall_oof" not in unseen_df.columns
):

    raise ValueError(
        "\nUnexpected unseen-batch summary format."
    )


unseen_metric_rows = unseen_df.set_index(
    "metric"
)


unseen_ba = float(
    unseen_metric_rows.loc[
        "balanced_accuracy",
        "overall_oof",
    ]
)


unseen_recall = float(
    unseen_metric_rows.loc[
        "attention_recall",
        "overall_oof",
    ]
)


record_check(
    "audit1_balanced_accuracy_valid",
    0.0 <= unseen_ba <= 1.0,
    unseen_ba,
)


record_check(
    "audit1_recall_valid",
    0.0 <= unseen_recall <= 1.0,
    unseen_recall,
)


record_check(
    "audit1_generalization_meaningful",
    unseen_ba >= 0.68,
    unseen_ba,
)


# ============================================================
# AUDIT 2
# DETERMINISTIC BASELINE
# ============================================================

det_zero = float(
    det[
        "zero_mean_final_quality"
    ]
)


det_lsq = float(
    det[
        "deterministic_mean_final_quality"
    ]
)


det_structured = float(
    det[
        "structured_mean_final_quality"
    ]
)


record_check(
    "audit2_zero_matches_reference",
    np.isclose(
        det_zero,
        0.9505,
        atol=5e-4,
    ),
    det_zero,
)


record_check(
    "audit2_structured_matches_reference",
    np.isclose(
        det_structured,
        0.2386,
        atol=5e-4,
    ),
    det_structured,
)


record_check(
    "audit2_lsq_improves_zero",
    det_lsq < det_zero,
    det_lsq,
)


record_check(
    "audit2_structured_improves_zero",
    det_structured < det_zero,
    det_structured,
)


# ============================================================
# AUDIT 3
# ACHIEVABLE CORRECTION
# ============================================================

ach_zero = float(
    ach[
        "zero_mean_final_quality"
    ]
)


ach_direct = float(
    ach[
        "direct_mean_final_quality"
    ]
)


ach_lsq = float(
    ach[
        "deterministic_mean_final_quality"
    ]
)


ach_structured = float(
    ach[
        "structured_mean_final_quality"
    ]
)


structured_capture = float(
    ach[
        "structured_improvement_capture_percent"
    ]
)


lsq_capture = float(
    ach[
        "deterministic_improvement_capture_percent"
    ]
)


record_check(
    "audit3_zero_consistent_with_audit2",
    np.isclose(
        ach_zero,
        det_zero,
        atol=1e-8,
    ),
    f"{ach_zero} vs {det_zero}",
)


record_check(
    "audit3_lsq_consistent_with_audit2",
    np.isclose(
        ach_lsq,
        det_lsq,
        atol=1e-8,
    ),
    f"{ach_lsq} vs {det_lsq}",
)


record_check(
    "audit3_structured_consistent_with_audit2",
    np.isclose(
        ach_structured,
        det_structured,
        atol=1e-8,
    ),
    f"{ach_structured} vs {det_structured}",
)


record_check(
    "audit3_direct_better_than_structured",
    ach_direct < ach_structured,
    f"{ach_direct} < {ach_structured}",
)


record_check(
    "audit3_direct_better_than_lsq",
    ach_direct < ach_lsq,
    f"{ach_direct} < {ach_lsq}",
)


record_check(
    "audit3_structured_capture_valid",
    0.0 <= structured_capture <= 110.0,
    structured_capture,
)


record_check(
    "audit3_lsq_capture_valid",
    0.0 <= lsq_capture <= 110.0,
    lsq_capture,
)


record_check(
    "audit3_structured_captures_majority",
    structured_capture >= 80.0,
    structured_capture,
)


record_check(
    "audit3_lsq_captures_majority",
    lsq_capture >= 80.0,
    lsq_capture,
)


# ============================================================
# AUDIT 4
# OBSERVABLE INFORMATION
# ============================================================

required_feature_sets = {
    "PRIVILEGED",
    "OBSERVABLE",
    "FULL",
}


if (
    "feature_set"
    not in observable_df.columns
):

    raise ValueError(
        "\nObservable-feature summary is missing feature_set."
    )


actual_feature_sets = set(
    observable_df[
        "feature_set"
    ]
    .astype(
        str
    )
)


record_check(
    "audit4_all_feature_sets_present",
    required_feature_sets.issubset(
        actual_feature_sets
    ),
    actual_feature_sets,
)


observable_lookup = observable_df.set_index(
    "feature_set"
)


observable_ba = float(
    observable_lookup.loc[
        "OBSERVABLE",
        "mean_balanced_accuracy",
    ]
)


privileged_ba = float(
    observable_lookup.loc[
        "PRIVILEGED",
        "mean_balanced_accuracy",
    ]
)


full_ba = float(
    observable_lookup.loc[
        "FULL",
        "mean_balanced_accuracy",
    ]
)


record_check(
    "audit4_observable_meaningful",
    observable_ba >= 0.68,
    observable_ba,
)


record_check(
    "audit4_full_best",
    (
        full_ba >= observable_ba
        and
        full_ba >= privileged_ba
    ),
    (
        f"FULL={full_ba}, "
        f"OBS={observable_ba}, "
        f"PRIV={privileged_ba}"
    ),
)


record_check(
    "audit4_observable_beats_privileged",
    observable_ba >= privileged_ba,
    (
        f"OBS={observable_ba}, "
        f"PRIV={privileged_ba}"
    ),
)


# ============================================================
# AUDIT 5
# TWO-STEP LOOK-AHEAD
# ============================================================

lookahead_n = int(
    round(
        float(
            look[
                "assemblies"
            ]
        )
    )
)


greedy_quality = float(
    look[
        "greedy_mean_final_quality"
    ]
)


lookahead_quality = float(
    look[
        "lookahead_mean_final_quality"
    ]
)


lookahead_benefit = float(
    look[
        "relative_lookahead_benefit_percent"
    ]
)


greedy_utilization = float(
    look[
        "mean_greedy_utilization"
    ]
)


lookahead_utilization = float(
    look[
        "mean_lookahead_utilization"
    ]
)


record_check(
    "audit5_sample_size",
    lookahead_n == 30,
    lookahead_n,
)


record_check(
    "audit5_quality_values_positive",
    (
        greedy_quality > 0.0
        and
        lookahead_quality > 0.0
    ),
    (
        f"{greedy_quality}, "
        f"{lookahead_quality}"
    ),
)


record_check(
    "audit5_no_large_lookahead_gain",
    lookahead_benefit < 3.0,
    lookahead_benefit,
)


record_check(
    "audit5_utilization_valid",
    (
        greedy_utilization >= 0.0
        and
        lookahead_utilization >= 0.0
    ),
    (
        f"{greedy_utilization}, "
        f"{lookahead_utilization}"
    ),
)


record_check(
    "audit5_greedy_not_materially_worse",
    (
        lookahead_quality
        -
        greedy_quality
    )
    <=
    0.01,
    (
        f"greedy={greedy_quality}, "
        f"lookahead={lookahead_quality}"
    ),
)


# ============================================================
# AUDIT 6
# ML VALUE REGIONS
# ============================================================

mlv_assemblies = int(
    round(
        float(
            mlv[
                "assemblies"
            ]
        )
    )
)


mlv_structured_wins = float(
    mlv[
        "structured20_win_percent"
    ]
)


mlv_lsq_wins = float(
    mlv[
        "deterministic_lsq_win_percent"
    ]
)


mlv_advantage = float(
    mlv[
        "mean_structured_advantage"
    ]
)


mlv_ci_low = float(
    mlv[
        "bootstrap_95ci_lower"
    ]
)


mlv_ci_high = float(
    mlv[
        "bootstrap_95ci_upper"
    ]
)


record_check(
    "audit6_sample_size",
    mlv_assemblies == 300,
    mlv_assemblies,
)


record_check(
    "audit6_win_rates_valid",
    (
        0.0 <= mlv_structured_wins <= 100.0
        and
        0.0 <= mlv_lsq_wins <= 100.0
    ),
    (
        f"Structured={mlv_structured_wins}, "
        f"LSQ={mlv_lsq_wins}"
    ),
)


record_check(
    "audit6_mean_advantage_finite",
    finite_number(
        mlv_advantage
    ),
    mlv_advantage,
)


record_check(
    "audit6_bootstrap_interval_ordered",
    mlv_ci_low <= mlv_ci_high,
    (
        f"[{mlv_ci_low}, "
        f"{mlv_ci_high}]"
    ),
)


record_check(
    "audit6_lsq_mean_advantage_supported",
    mlv_ci_high < 0.0,
    (
        f"mean={mlv_advantage}, "
        f"CI=[{mlv_ci_low}, "
        f"{mlv_ci_high}]"
    ),
)


record_check(
    "audit6_consistent_with_deterministic_quality",
    np.isclose(
        float(
            mlv[
                "mean_deterministic_final_quality"
            ]
        ),
        det_lsq,
        atol=1e-8,
    ),
    (
        f"{mlv['mean_deterministic_final_quality']} "
        f"vs {det_lsq}"
    ),
)


# ============================================================
# AUDIT 7
# CORRECTION-BASIS RESIDUAL
# ============================================================

basis_assemblies = int(
    round(
        float(
            basis[
                "assemblies"
            ]
        )
    )
)


basis_decisions = int(
    round(
        float(
            basis[
                "component_decisions"
            ]
        )
    )
)


bounded_residual = float(
    basis[
        "mean_bounded_residual_ratio"
    ]
)


unbounded_residual = float(
    basis[
        "mean_unbounded_residual_ratio"
    ]
)


bounded_explained = float(
    basis[
        "mean_bounded_explained_energy"
    ]
)


unbounded_explained = float(
    basis[
        "mean_unbounded_explained_energy"
    ]
)


capability_penalty = float(
    basis[
        "mean_capability_residual_penalty"
    ]
)


basis_ml_correlation = float(
    basis[
        "correlation_bounded_residual_vs_structured_advantage"
    ]
)


replay_difference = float(
    basis[
        "max_abs_replay_difference_vs_frozen_baseline"
    ]
)


record_check(
    "audit7_sample_size",
    (
        basis_assemblies == 300
        and
        basis_decisions == 1500
    ),
    (
        f"assemblies={basis_assemblies}, "
        f"decisions={basis_decisions}"
    ),
)


record_check(
    "audit7_residual_values_valid",
    (
        0.0 <= bounded_residual <= 1.5
        and
        0.0 <= unbounded_residual <= 1.5
    ),
    (
        f"bounded={bounded_residual}, "
        f"unbounded={unbounded_residual}"
    ),
)


record_check(
    "audit7_explained_energy_valid",
    (
        0.0 <= bounded_explained <= 1.0
        and
        0.0 <= unbounded_explained <= 1.0
    ),
    (
        f"bounded={bounded_explained}, "
        f"unbounded={unbounded_explained}"
    ),
)


record_check(
    "audit7_substantial_basis_alignment",
    unbounded_explained >= 0.50,
    unbounded_explained,
)


record_check(
    "audit7_capability_penalty_small",
    abs(
        capability_penalty
    )
    <=
    1e-6,
    capability_penalty,
)


record_check(
    "audit7_replay_matches_frozen_baseline",
    replay_difference <= 1e-10,
    replay_difference,
)


record_check(
    "audit7_ml_value_relationship_positive",
    basis_ml_correlation > 0.20,
    basis_ml_correlation,
)


# ------------------------------------------------------------
# Q1 vs Q4 basis-region comparison
# ------------------------------------------------------------

required_quartile_columns = {
    "basis_residual_quartile",
    "structured_win_percent",
    "mean_structured_advantage",
}


record_check(
    "audit7_quartile_columns_present",
    required_quartile_columns.issubset(
        basis_quartile_df.columns
    ),
    list(
        basis_quartile_df.columns
    ),
)


quartile_lookup = (
    basis_quartile_df
    .set_index(
        "basis_residual_quartile"
    )
)


q1_win = float(
    quartile_lookup.loc[
        "Q1_best_basis_fit",
        "structured_win_percent",
    ]
)


q4_win = float(
    quartile_lookup.loc[
        "Q4_worst_basis_fit",
        "structured_win_percent",
    ]
)


q4_advantage = float(
    quartile_lookup.loc[
        "Q4_worst_basis_fit",
        "mean_structured_advantage",
    ]
)


record_check(
    "audit7_structured_more_competitive_in_q4",
    q4_win > q1_win,
    (
        f"Q1={q1_win}, "
        f"Q4={q4_win}"
    ),
)


record_check(
    "audit7_q4_structured_majority",
    q4_win > 50.0,
    q4_win,
)


record_check(
    "audit7_q4_mean_advantage_nonnegative",
    q4_advantage >= 0.0,
    q4_advantage,
)


# ============================================================
# AUDIT 8
# METHOD TRADE-OFF
# ============================================================

trade_zero = float(
    trade[
        "zero_mean_final_quality"
    ]
)


trade_lsq = float(
    trade[
        "lsq_mean_final_quality"
    ]
)


trade_structured = float(
    trade[
        "structured20_mean_final_quality"
    ]
)


trade_selective = float(
    trade[
        "selective_bo_mean_final_quality"
    ]
)


trade_direct = float(
    trade[
        "direct_reference_mean_final_quality"
    ]
)


direct_evaluations = float(
    trade[
        "mean_direct_simulator_evaluations_per_decision"
    ]
)


record_check(
    "audit8_quality_values_consistent",
    (
        np.isclose(
            trade_zero,
            det_zero,
            atol=1e-8,
        )
        and
        np.isclose(
            trade_lsq,
            det_lsq,
            atol=1e-8,
        )
        and
        np.isclose(
            trade_structured,
            det_structured,
            atol=1e-8,
        )
        and
        np.isclose(
            trade_direct,
            ach_direct,
            atol=1e-8,
        )
    ),
    (
        f"zero={trade_zero}, "
        f"lsq={trade_lsq}, "
        f"struct={trade_structured}, "
        f"direct={trade_direct}"
    ),
)


record_check(
    "audit8_quality_order_valid",
    (
        trade_direct
        <
        trade_lsq
        <
        trade_structured
        <
        trade_zero
    ),
    (
        f"direct={trade_direct}, "
        f"lsq={trade_lsq}, "
        f"struct={trade_structured}, "
        f"zero={trade_zero}"
    ),
)


record_check(
    "audit8_selective_close_to_structured",
    abs(
        trade_selective
        -
        trade_structured
    )
    <=
    0.005,
    (
        f"selective={trade_selective}, "
        f"structured={trade_structured}"
    ),
)


record_check(
    "audit8_direct_cost_high",
    direct_evaluations >= 100.0,
    direct_evaluations,
)


required_methods = {
    "Zero correction",
    "Deterministic LSQ",
    "Structured-20",
    "Selective BO",
    "Direct simulator numerical reference",
}


actual_methods = set(
    tradeoff_table_df[
        "method"
    ]
    .astype(
        str
    )
)


record_check(
    "audit8_all_methods_present",
    required_methods.issubset(
        actual_methods
    ),
    actual_methods,
)


# ============================================================
# CROSS-AUDIT SCIENTIFIC CONSISTENCY
# ============================================================

record_check(
    "crossaudit_zero_consistency",
    (
        np.isclose(
            det_zero,
            ach_zero,
            atol=1e-8,
        )
        and
        np.isclose(
            det_zero,
            trade_zero,
            atol=1e-8,
        )
    ),
    (
        f"det={det_zero}, "
        f"ach={ach_zero}, "
        f"trade={trade_zero}"
    ),
)


record_check(
    "crossaudit_lsq_consistency",
    (
        np.isclose(
            det_lsq,
            ach_lsq,
            atol=1e-8,
        )
        and
        np.isclose(
            det_lsq,
            trade_lsq,
            atol=1e-8,
        )
    ),
    (
        f"det={det_lsq}, "
        f"ach={ach_lsq}, "
        f"trade={trade_lsq}"
    ),
)


record_check(
    "crossaudit_structured_consistency",
    (
        np.isclose(
            det_structured,
            ach_structured,
            atol=1e-8,
        )
        and
        np.isclose(
            det_structured,
            trade_structured,
            atol=1e-8,
        )
    ),
    (
        f"det={det_structured}, "
        f"ach={ach_structured}, "
        f"trade={trade_structured}"
    ),
)


# ============================================================
# FINAL RESULT TABLE
# ============================================================

check_df = pd.DataFrame(
    checks
)


CHECK_OUTPUT = os.path.join(
    RESULTS_DIR,
    "final_audit_consistency_checks.csv",
)


check_df.to_csv(
    CHECK_OUTPUT,
    index=False,
)


passed = int(
    check_df[
        "passed"
    ]
    .sum()
)


total = len(
    check_df
)


failed = (
    total
    -
    passed
)


# ============================================================
# PRINT DETAILED RESULTS
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "CHECK RESULTS"
)

print(
    "============================================================"
)


for _, row in check_df.iterrows():

    status = (
        "PASS"
        if row[
            "passed"
        ]
        else
        "FAIL"
    )

    print(
        f"\n[{status}] "
        f"{row['check']}"
        f" | "
        f"{row['detail']}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n"
    "============================================================"
)


print(
    f"Total checks : {total}"
)


print(
    f"Passed       : {passed}"
)


print(
    f"Failed       : {failed}"
)


if failed == 0:

    print(
        "\nFINAL AUDIT CONSISTENCY: PASS"
    )

else:

    print(
        "\nFINAL AUDIT CONSISTENCY: FAIL"
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
    "FINAL SCIENTIFIC AUDIT CHECK COMPLETED"
)

print(
    "============================================================"
)