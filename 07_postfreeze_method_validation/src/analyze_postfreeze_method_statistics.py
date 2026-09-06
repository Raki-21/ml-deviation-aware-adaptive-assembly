"""
Phase 3 post-freeze experiment: formal paired statistics across the five
pre-registered hypotheses, with Holm correction for multiple comparisons.

Purpose
-------
Phases 1 and 2 already showed the direction of every effect (Composite-Q
beats everything, the hybrid beats LSQ but loses to Composite-Q, the
mean-residual baselines are statistically indistinguishable from LSQ).
This phase puts a number on how confident we can be in each of those
claims, using the same paired-bootstrap / Wilcoxon / effect-size approach
already used for the original hybrid-vs-LSQ result, extended to every
comparison this project now needs, and correcting for the fact that five
hypotheses are being tested on the same 300 assemblies at once.

Pre-registered hypotheses (fixed before this script was written, not
chosen after seeing which comparisons look good)
---------------------------------------------------------------------
H1: HYBRID vs LSQ           -- the original locked-validation claim
H2: HYBRID vs COMPOSITE_Q   -- does ML add anything beyond objective alignment?
H3: HYBRID vs GLOBAL        -- is the hybrid's edge just a fixed bias?
H4: HYBRID vs STAGE         -- same question, allowing a per-stage bias
H5: COMPOSITE_Q vs LSQ      -- how much does objective alignment alone buy?

Data source
-----------
07_postfreeze_method_validation/results/phase_2_mean_residual/phase2_mean_residual_assembly_results.csv
This file already carries all six methods' final quality per assembly
(LSQ, global mean residual, stage mean residual, locked hybrid, direct
reference, Composite-Q), joined during Phase 2. Nothing is recomputed
here -- this script only reads already-produced, already-verified numbers
and runs statistics on them.

No new dependency
------------------
Holm correction is implemented directly (a five-line algorithm) instead
of pulling in statsmodels, since it is not in this project's
requirements.txt and there is no reason to add a dependency for something
this simple.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PHASE_ROOT = SCRIPT_DIR.parent

PHASE2_ASSEMBLY_FILE = (
    PHASE_ROOT / "results" / "phase_2_mean_residual" / "phase2_mean_residual_assembly_results.csv"
)

RESULTS_DIR = PHASE_ROOT / "results" / "phase_3_statistics"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STATISTICS_FILE = RESULTS_DIR / "postfreeze_pairwise_statistics.csv"
DIFFERENCES_FILE = RESULTS_DIR / "postfreeze_pairwise_differences.csv"

if not PHASE2_ASSEMBLY_FILE.exists():
    raise FileNotFoundError(
        f"Phase 2 results not found (this script only reads them): {PHASE2_ASSEMBLY_FILE}"
    )

BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260906


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(PHASE2_ASSEMBLY_FILE)

REQUIRED_COLUMNS = [
    "lsq_final_quality",
    "global_mean_residual_final_quality",
    "stage_mean_residual_final_quality",
    "hybrid_final_quality",
    "composite_q_final_quality",
]
missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
if missing:
    raise RuntimeError(f"Phase 2 results file is missing expected columns: {missing}")

n_assemblies = len(df)


# ============================================================
# PAIRED STATISTICS FOR ONE HYPOTHESIS
# ============================================================

def paired_bootstrap_ci(diff, n_resamples, seed):
    rng = np.random.default_rng(seed)
    n = len(diff)
    means = np.empty(n_resamples)
    diff_values = diff.to_numpy()
    for i in range(n_resamples):
        sample = rng.choice(diff_values, size=n, replace=True)
        means[i] = sample.mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def analyze_hypothesis(label, method_a, series_a, method_b, series_b, seed):
    """
    diff = a - b. Since lower quality is better, diff < 0 means A beat B
    on that assembly.
    """
    diff = series_a - series_b

    mean_diff = float(diff.mean())
    median_diff = float(diff.median())
    std_diff = float(diff.std(ddof=1))

    ci_lower, ci_upper = paired_bootstrap_ci(diff, BOOTSTRAP_RESAMPLES, seed)

    a_wins = int((diff < -1e-9).sum())
    b_wins = int((diff > 1e-9).sum())
    ties = int(len(diff) - a_wins - b_wins)

    # Wilcoxon signed-rank test undefined if all differences are exactly
    # zero; guard against that degenerate case even though it won't occur
    # with continuous quality scores in this dataset.
    nonzero_diff = diff[diff.abs() > 1e-12]
    if len(nonzero_diff) >= 1:
        wilcoxon_stat, wilcoxon_p = wilcoxon(nonzero_diff)
        wilcoxon_stat = float(wilcoxon_stat)
        wilcoxon_p = float(wilcoxon_p)
    else:
        wilcoxon_stat, wilcoxon_p = float("nan"), 1.0

    effect_size = float(mean_diff / std_diff) if std_diff > 0 else float("nan")

    relative_improvement_percent = float(
        100.0 * (series_b.mean() - series_a.mean()) / max(series_b.mean(), 1e-12)
    )

    return {
        "hypothesis": label,
        "comparison": f"{method_a} vs {method_b}",
        "n_assemblies": n_assemblies,
        f"mean_{method_a.lower()}": float(series_a.mean()),
        f"mean_{method_b.lower()}": float(series_b.mean()),
        "mean_difference_a_minus_b": mean_diff,
        "median_difference_a_minus_b": median_diff,
        "paired_bootstrap_95ci_lower": ci_lower,
        "paired_bootstrap_95ci_upper": ci_upper,
        "wilcoxon_statistic": wilcoxon_stat,
        "wilcoxon_p_raw": wilcoxon_p,
        f"{method_a.lower()}_win_rate_percent": 100.0 * a_wins / n_assemblies,
        f"{method_b.lower()}_win_rate_percent": 100.0 * b_wins / n_assemblies,
        "tie_rate_percent": 100.0 * ties / n_assemblies,
        "paired_standardized_effect_size": effect_size,
        "relative_improvement_percent_a_over_b": relative_improvement_percent,
    }, diff


# ============================================================
# HOLM-BONFERRONI CORRECTION (manual, no new dependency)
# ============================================================

def holm_correction(p_values):
    """
    Standard Holm step-down procedure.
    p_values: array-like of raw p-values, in the SAME ORDER they should be
    reported back in (this function handles the internal sort/unsort).
    Returns adjusted p-values in the original order.
    """
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    sorted_p = p_values[order]

    adjusted_sorted = np.empty(m)
    running_max = 0.0
    for i in range(m):
        candidate = (m - i) * sorted_p[i]
        running_max = max(running_max, candidate)
        adjusted_sorted[i] = min(running_max, 1.0)

    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    return adjusted


# ============================================================
# RUN THE FIVE PRE-REGISTERED HYPOTHESES
# ============================================================

print("=" * 60)
print("PHASE 3: FORMAL PAIRED STATISTICS (5 pre-registered hypotheses)")
print("=" * 60)
print(f"\nn assemblies: {n_assemblies}")
print(f"Bootstrap resamples: {BOOTSTRAP_RESAMPLES}")

hypotheses = [
    ("H1", "HYBRID", df["hybrid_final_quality"], "LSQ", df["lsq_final_quality"]),
    ("H2", "HYBRID", df["hybrid_final_quality"], "COMPOSITE_Q", df["composite_q_final_quality"]),
    ("H3", "HYBRID", df["hybrid_final_quality"], "GLOBAL", df["global_mean_residual_final_quality"]),
    ("H4", "HYBRID", df["hybrid_final_quality"], "STAGE", df["stage_mean_residual_final_quality"]),
    ("H5", "COMPOSITE_Q", df["composite_q_final_quality"], "LSQ", df["lsq_final_quality"]),
]

results = []
diff_columns = {}

for i, (label, method_a, series_a, method_b, series_b) in enumerate(hypotheses):
    row, diff = analyze_hypothesis(
        label, method_a, series_a, method_b, series_b, seed=BOOTSTRAP_SEED + i
    )
    results.append(row)
    diff_columns[f"{label}_{method_a}_minus_{method_b}"] = diff.to_numpy()

results_df = pd.DataFrame(results)

# Holm correction across the 5 primary p-values, as one family.
results_df["wilcoxon_p_holm_adjusted"] = holm_correction(results_df["wilcoxon_p_raw"].to_numpy())
results_df["significant_after_holm_at_0.05"] = results_df["wilcoxon_p_holm_adjusted"] < 0.05

results_df.to_csv(STATISTICS_FILE, index=False)

diff_df = pd.DataFrame(diff_columns)
diff_df.insert(0, "assembly_index", df["assembly_index"])
diff_df.to_csv(DIFFERENCES_FILE, index=False)


# ============================================================
# PRINT
# ============================================================

print("\n" + "=" * 60)
print("RESULTS (Holm-corrected across all 5 hypotheses)")
print("=" * 60)

for _, row in results_df.iterrows():
    print(f"\n{row['hypothesis']}: {row['comparison']}")
    print(f"  mean difference (a - b)         : {row['mean_difference_a_minus_b']:.6f}")
    print(f"  median difference (a - b)       : {row['median_difference_a_minus_b']:.6f}")
    print(
        f"  95% paired bootstrap CI          : "
        f"[{row['paired_bootstrap_95ci_lower']:.6f}, {row['paired_bootstrap_95ci_upper']:.6f}]"
    )
    print(f"  Wilcoxon statistic               : {row['wilcoxon_statistic']:.2f}")
    print(f"  Wilcoxon p (raw)                 : {row['wilcoxon_p_raw']:.3e}")
    print(f"  Wilcoxon p (Holm-adjusted)       : {row['wilcoxon_p_holm_adjusted']:.3e}")
    print(f"  significant at 0.05 after Holm   : {bool(row['significant_after_holm_at_0.05'])}")
    print(f"  paired standardized effect size  : {row['paired_standardized_effect_size']:.4f}")
    print(f"  relative improvement (a over b)  : {row['relative_improvement_percent_a_over_b']:.2f}%")

print(f"\nSaved: {STATISTICS_FILE}")
print(f"Saved: {DIFFERENCES_FILE}")
print("\nDone.")