"""
Phase 4B post-freeze experiment: which quality term explains the hybrid's
losses to LSQ.

Purpose
-------
Phase 4 found that the 16 assemblies where LSQ beats the hybrid are not
the "hard" assemblies -- the hybrid's own final RMS and parallelism are
actually BETTER (lower) on these 16 than on the other 284. That is a
group-level comparison (failure-group hybrid values vs success-group
hybrid values), and it does not by itself explain why, WITHIN the failure
group, the hybrid still scores worse than LSQ overall.

This script closes that gap. The composite quality score is defined as:

    Q = 0.30 * mean_gap + 0.30 * max_gap + 0.30 * parallelism + 0.10 * rms

so it is decomposed by construction into four terms. This script computes
the hybrid-minus-LSQ difference for each of those four terms, per
assembly, and checks -- exactly, not approximately -- that the weighted
sum of the four term differences reproduces the known
hybrid-minus-LSQ quality difference. That reconciliation is the actual
proof of which term(s) are driving the failure cases, not just a
plausible-looking table.

Design rules (same as every other post-freeze script)
--------------------------------------------------------
Reads only the frozen final-validation results. Writes nothing back to
06_v3_3_nonlinear_hybrid_upgrade/. No hypothesis test is run on the 16
failure cases -- the sample is too small for one, and this is intentionally
a descriptive, not inferential, breakdown.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PHASE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PHASE_ROOT.parent

ASSEMBLY_RESULTS_FILE = (
    REPO_ROOT
    / "06_v3_3_nonlinear_hybrid_upgrade"
    / "results"
    / "final_independent_validation_v33"
    / "v33_final_independent_assembly_results.csv"
)

RESULTS_DIR = PHASE_ROOT / "results" / "phase_4_failure_analysis"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = RESULTS_DIR / "hybrid_failure_quality_terms_summary.csv"
CASE_FILE = RESULTS_DIR / "hybrid_failure_quality_terms_cases.csv"

if not ASSEMBLY_RESULTS_FILE.exists():
    raise FileNotFoundError(f"Required frozen input not found (read-only): {ASSEMBLY_RESULTS_FILE}")

METHOD_LSQ = "LSQ"
METHOD_HYBRID = "LSQ_PLUS_ML_ALL"

# Must match v33_config.py exactly -- these are the locked composite
# quality weights, not re-derived, just quoted here for the reconciliation
# check below.
WEIGHT_MEAN_GAP = 0.30
WEIGHT_MAX_GAP = 0.30
WEIGHT_PARALLELISM = 0.30
WEIGHT_RMS = 0.10


# ============================================================
# LOAD FROZEN RESULTS (READ ONLY)
# ============================================================

df = pd.read_csv(ASSEMBLY_RESULTS_FILE)
lsq = df[df["method"] == METHOD_LSQ].set_index("assembly_index")
hybrid = df[df["method"] == METHOD_HYBRID].set_index("assembly_index")

comparison = pd.DataFrame(
    {
        "lsq_final_quality": lsq["final_quality"],
        "hybrid_final_quality": hybrid["final_quality"],
        "lsq_final_mean_gap": lsq["final_mean_gap"],
        "hybrid_final_mean_gap": hybrid["final_mean_gap"],
        "lsq_final_max_gap": lsq["final_max_gap"],
        "hybrid_final_max_gap": hybrid["final_max_gap"],
        "lsq_final_parallelism": lsq["final_parallelism"],
        "hybrid_final_parallelism": hybrid["final_parallelism"],
        "lsq_final_rms": lsq["final_rms"],
        "hybrid_final_rms": hybrid["final_rms"],
    }
)
comparison["hybrid_minus_lsq_quality"] = (
    comparison["hybrid_final_quality"] - comparison["lsq_final_quality"]
)
comparison["lsq_beat_hybrid"] = comparison["hybrid_minus_lsq_quality"] > 1e-9

METRICS = ["mean_gap", "max_gap", "parallelism", "rms"]
for metric in METRICS:
    comparison[f"hybrid_minus_lsq_{metric}"] = (
        comparison[f"hybrid_final_{metric}"] - comparison[f"lsq_final_{metric}"]
    )

# ============================================================
# RECONCILIATION CHECK -- the weighted sum of the four term differences
# MUST equal the quality difference exactly (up to floating point),
# because that is literally how the quality score is defined. If this
# fails, something is wrong with the data, not with the interpretation.
# ============================================================

comparison["reconstructed_quality_diff"] = (
    WEIGHT_MEAN_GAP * comparison["hybrid_minus_lsq_mean_gap"]
    + WEIGHT_MAX_GAP * comparison["hybrid_minus_lsq_max_gap"]
    + WEIGHT_PARALLELISM * comparison["hybrid_minus_lsq_parallelism"]
    + WEIGHT_RMS * comparison["hybrid_minus_lsq_rms"]
)
comparison["reconciliation_error"] = (
    comparison["reconstructed_quality_diff"] - comparison["hybrid_minus_lsq_quality"]
).abs()

max_reconciliation_error = float(comparison["reconciliation_error"].max())

comparison = comparison.reset_index()
failure_df = comparison[comparison["lsq_beat_hybrid"]].copy()
success_df = comparison[~comparison["lsq_beat_hybrid"]].copy()
failure_df.to_csv(CASE_FILE, index=False)


# ============================================================
# DESCRIPTIVE SUMMARY, WITH EACH TERM'S SHARE OF THE QUALITY GAP
# ============================================================

summary_rows = []
for metric in METRICS:
    diff_col = f"hybrid_minus_lsq_{metric}"
    weight = {"mean_gap": WEIGHT_MEAN_GAP, "max_gap": WEIGHT_MAX_GAP,
              "parallelism": WEIGHT_PARALLELISM, "rms": WEIGHT_RMS}[metric]

    failure_weighted_contribution = float((weight * failure_df[diff_col]).mean())
    failure_quality_gap_mean = float(failure_df["hybrid_minus_lsq_quality"].mean())
    share_of_gap_percent = (
        100.0 * failure_weighted_contribution / failure_quality_gap_mean
        if abs(failure_quality_gap_mean) > 1e-12 else float("nan")
    )

    summary_rows.append(
        {
            "metric": metric,
            "weight": weight,
            "failure_hybrid_minus_lsq_mean": float(failure_df[diff_col].mean()),
            "success_hybrid_minus_lsq_mean": float(success_df[diff_col].mean()),
            "failure_weighted_contribution_to_quality_gap": failure_weighted_contribution,
            "share_of_failure_quality_gap_percent": share_of_gap_percent,
            "failure_hybrid_absolute_mean": float(failure_df[f"hybrid_final_{metric}"].mean()),
            "failure_lsq_absolute_mean": float(failure_df[f"lsq_final_{metric}"].mean()),
            "failure_n": len(failure_df),
            "success_n": len(success_df),
        }
    )

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_FILE, index=False)


# ============================================================
# PRINT
# ============================================================

print("=" * 60)
print("PHASE 4B: FAILURE-CASE QUALITY-TERM BREAKDOWN")
print("=" * 60)
print(f"\nTotal assemblies      : {len(comparison)}")
print(f"LSQ-beats-hybrid cases: {len(failure_df)}")
print(f"Hybrid-success cases  : {len(success_df)}")

print("\n" + "=" * 60)
print("RECONCILIATION CHECK")
print("=" * 60)
print(
    f"Max |weighted term sum - actual quality diff| across all 300: "
    f"{max_reconciliation_error:.3e}"
)
print("PASS: the four terms fully account for the quality difference."
      if max_reconciliation_error < 1e-9 else
      "FAIL: terms do not reconcile -- do not trust the breakdown below.")

print("\nPositive Hybrid-LSQ difference means HYBRID is worse on that term.")
print("Negative means HYBRID is better on that term.\n")

for _, row in summary_df.iterrows():
    print(f"{row['metric']}  (weight {row['weight']})")
    print(f"  failure mean Hybrid-LSQ difference      : {row['failure_hybrid_minus_lsq_mean']:.6f}")
    print(f"  success mean Hybrid-LSQ difference      : {row['success_hybrid_minus_lsq_mean']:.6f}")
    print(f"  weighted contribution to failure quality gap : {row['failure_weighted_contribution_to_quality_gap']:.6f}")
    print(f"  share of the average failure quality gap     : {row['share_of_failure_quality_gap_percent']:.1f}%")
    print(f"  failure-case Hybrid absolute mean            : {row['failure_hybrid_absolute_mean']:.6f}")
    print(f"  failure-case LSQ absolute mean               : {row['failure_lsq_absolute_mean']:.6f}")
    print()

print(f"Saved: {CASE_FILE}")
print(f"Saved: {SUMMARY_FILE}")
print("\nDone.")