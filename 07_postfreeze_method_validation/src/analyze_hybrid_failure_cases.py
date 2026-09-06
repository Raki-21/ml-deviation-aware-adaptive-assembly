"""
Phase 4 post-freeze experiment: hybrid failure-case breakdown.

Purpose
-------
The locked final validation shows the hybrid loses to LSQ on about 5.33%
of the 300 independent assemblies (roughly 16 cases). This script asks
what, if anything, those cases have in common. The analysis variables
below were fixed BEFORE looking at which assemblies fail, from the
already-agreed plan -- nothing here was chosen after seeing which
variable happens to separate the two groups.

Analysis variables (pre-registered)
------------------------------------
1. LSQ utilization (mean, max) across the assembly's 10 decisions
2. Hybrid utilization (mean, max)
3. Hybrid applied-residual magnitude (mean |delta_z|, |delta_theta|,
   |delta_locator|)
4. Distance to correction bounds (captured by utilization directly:
   utilization = max(|z|/Z_max, |theta|/theta_max, |locator|/locator_max))
5. Largest single-stage hybrid loss (max over stages of
   hybrid_post_quality - lsq_post_quality at that stage) and which stage
   it occurs at
6. Final profile RMS and parallelism (already in the frozen assembly
   results)
7. The underlying deviation-generation parameters for the assembly
   (offset_mm, tilt_deg, bend_mm, waviness_mm, twist_mm, local_bump_mm,
   averaged across the assembly's 10 components) -- regenerated with the
   exact same seed and logic used everywhere else in this project, purely
   to characterize what kind of geometry these assemblies have. This is
   used ONLY for retrospective interpretation. It is never fed back into
   any controller as an input.

Important limitation, stated up front
----------------------------------------
With only ~16 failure cases, this is not a hypothesis test -- there is no
attempt to compute a p-value for "failure cases have higher tilt" or
similar. The comparison below is descriptive: failure-case values next to
the same variables computed over the other ~284 assemblies, so a reader
can judge for themselves whether anything stands out. "No dominant
mechanism found" is an acceptable and expected possible outcome; this
script does not manufacture a pattern out of a small sample.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PHASE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PHASE_ROOT.parent

V33_SRC = REPO_ROOT / "06_v3_3_nonlinear_hybrid_upgrade" / "src"
V33_RESULTS_DIR = (
    REPO_ROOT / "06_v3_3_nonlinear_hybrid_upgrade" / "results" / "final_independent_validation_v33"
)
ASSEMBLY_RESULTS_FILE = V33_RESULTS_DIR / "v33_final_independent_assembly_results.csv"
COMPONENT_RESULTS_FILE = V33_RESULTS_DIR / "v33_final_independent_component_results.csv"

RESULTS_DIR = PHASE_ROOT / "results" / "phase_4_failure_analysis"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FAILURE_CASES_FILE = RESULTS_DIR / "hybrid_failure_cases.csv"
FAILURE_SUMMARY_FILE = RESULTS_DIR / "hybrid_failure_summary.csv"

for required in (ASSEMBLY_RESULTS_FILE, COMPONENT_RESULTS_FILE):
    if not required.exists():
        raise FileNotFoundError(f"Required frozen input not found (read-only): {required}")

sys.path.insert(0, str(V33_SRC))

from nonlinear_assembly_model import (  # noqa: E402
    deviation_offset,
    deviation_tilt,
    deviation_bend,
    deviation_waviness,
    deviation_twist,
    deviation_local_bump,
)
from v33_config import N_COMPONENTS, N_PROFILE_POINTS, Z_ADJ_LIMIT_MM, THETA_ADJ_LIMIT_DEG, LOCATOR_OFFSET_LIMIT_MM  # noqa: E402

SEQUENCE_LENGTH = N_COMPONENTS
N_ASSEMBLIES = 300
VALIDATION_SEED = 20260912
FIXTURE_DRIFT_STD_MM = 0.015

METHOD_LSQ = "LSQ"
METHOD_HYBRID = "LSQ_PLUS_ML_ALL"


# ============================================================
# REGENERATE GENERATION PARAMETERS ONLY (verbatim seed/logic, but this
# time we KEEP the parameter dict instead of discarding it after building
# the profile -- used purely for retrospective description below).
# ============================================================

def generate_component_parameters(rng):
    return {
        "offset_mm": rng.uniform(-0.60, 0.60),
        "tilt_deg": rng.uniform(-0.10, 0.10),
        "bend_mm": rng.uniform(-0.45, 0.45),
        "waviness_mm": rng.uniform(-0.30, 0.30),
        "twist_mm": rng.uniform(-0.25, 0.25),
        "local_bump_mm": rng.uniform(-0.30, 0.30),
    }


def regenerate_parameters_only():
    rng = np.random.default_rng(VALIDATION_SEED)
    rows = []
    for assembly_index in range(1, N_ASSEMBLIES + 1):
        for component_index in range(1, SEQUENCE_LENGTH + 1):
            parameters = generate_component_parameters(rng)
            # Must call the SAME rng draws in the SAME order as the real
            # generator (build_component_profile does not draw random
            # numbers itself, only generate_component_parameters and the
            # fixture_drift draw below do) to stay bit-identical.
            _ = rng.normal(loc=0.0, scale=FIXTURE_DRIFT_STD_MM, size=N_PROFILE_POINTS)
            row = {"assembly_index": assembly_index, "component_index": component_index}
            row.update(parameters)
            rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# LOAD FROZEN RESULTS (READ ONLY)
# ============================================================

assembly_df = pd.read_csv(ASSEMBLY_RESULTS_FILE)
component_df = pd.read_csv(COMPONENT_RESULTS_FILE)

lsq_assembly = assembly_df[assembly_df["method"] == METHOD_LSQ].set_index("assembly_index")
hybrid_assembly = assembly_df[assembly_df["method"] == METHOD_HYBRID].set_index("assembly_index")

comparison = pd.DataFrame(
    {
        "lsq_final_quality": lsq_assembly["final_quality"],
        "hybrid_final_quality": hybrid_assembly["final_quality"],
        "lsq_final_rms": lsq_assembly["final_rms"],
        "hybrid_final_rms": hybrid_assembly["final_rms"],
        "lsq_final_parallelism": lsq_assembly["final_parallelism"],
        "hybrid_final_parallelism": hybrid_assembly["final_parallelism"],
    }
)
comparison["hybrid_minus_lsq"] = comparison["hybrid_final_quality"] - comparison["lsq_final_quality"]
comparison["lsq_beat_hybrid"] = comparison["hybrid_minus_lsq"] > 1e-9

n_failures = int(comparison["lsq_beat_hybrid"].sum())
print("=" * 60)
print("PHASE 4: HYBRID FAILURE-CASE BREAKDOWN")
print("=" * 60)
print(f"\nTotal assemblies                 : {len(comparison)}")
print(f"Assemblies where LSQ beat hybrid  : {n_failures} ({100 * n_failures / len(comparison):.2f}%)")

failure_ids = comparison[comparison["lsq_beat_hybrid"]].index.tolist()


# ============================================================
# PER-ASSEMBLY DIAGNOSTICS FROM COMPONENT-LEVEL DATA
# ============================================================

lsq_comp = component_df[component_df["method"] == METHOD_LSQ]
hybrid_comp = component_df[component_df["method"] == METHOD_HYBRID]

diag_rows = []
for assembly_index in comparison.index:
    lsq_rows = lsq_comp[lsq_comp["assembly_index"] == assembly_index].sort_values("component_index")
    hyb_rows = hybrid_comp[hybrid_comp["assembly_index"] == assembly_index].sort_values("component_index")

    stage_diff = hyb_rows["post_quality"].to_numpy() - lsq_rows["post_quality"].to_numpy()
    worst_stage_idx = int(np.argmax(stage_diff))
    worst_stage_loss = float(stage_diff[worst_stage_idx])
    worst_stage_number = int(hyb_rows["component_index"].to_numpy()[worst_stage_idx])

    diag_rows.append(
        {
            "assembly_index": assembly_index,
            "lsq_utilization_mean": float(lsq_rows["utilization"].mean()),
            "lsq_utilization_max": float(lsq_rows["utilization"].max()),
            "hybrid_utilization_mean": float(hyb_rows["utilization"].mean()),
            "hybrid_utilization_max": float(hyb_rows["utilization"].max()),
            "hybrid_abs_delta_z_mean": float(hyb_rows["applied_delta_z_mm"].abs().mean()),
            "hybrid_abs_delta_theta_mean": float(hyb_rows["applied_delta_theta_deg"].abs().mean()),
            "hybrid_abs_delta_locator_mean": float(hyb_rows["applied_delta_locator_mm"].abs().mean()),
            "worst_stage_hybrid_minus_lsq": worst_stage_loss,
            "worst_stage_number": worst_stage_number,
        }
    )

diag_df = pd.DataFrame(diag_rows).set_index("assembly_index")
full_df = comparison.join(diag_df)


# ============================================================
# GENERATION PARAMETERS (retrospective description only)
# ============================================================

params_df = regenerate_parameters_only()
params_agg = params_df.groupby("assembly_index")[
    ["offset_mm", "tilt_deg", "bend_mm", "waviness_mm", "twist_mm", "local_bump_mm"]
].agg(lambda x: np.abs(x).mean())
params_agg.columns = [f"mean_abs_{c}" for c in params_agg.columns]

full_df = full_df.join(params_agg)
full_df = full_df.reset_index()

failure_df = full_df[full_df["lsq_beat_hybrid"]].copy()
success_df = full_df[~full_df["lsq_beat_hybrid"]].copy()

failure_df.to_csv(FAILURE_CASES_FILE, index=False)


# ============================================================
# DESCRIPTIVE COMPARISON (no hypothesis test -- n=16 is too small)
# ============================================================

DIAGNOSTIC_COLUMNS = [
    "lsq_utilization_mean",
    "lsq_utilization_max",
    "hybrid_utilization_mean",
    "hybrid_utilization_max",
    "hybrid_abs_delta_z_mean",
    "hybrid_abs_delta_theta_mean",
    "hybrid_abs_delta_locator_mean",
    "worst_stage_hybrid_minus_lsq",
    "hybrid_final_rms",
    "hybrid_final_parallelism",
    "mean_abs_offset_mm",
    "mean_abs_tilt_deg",
    "mean_abs_bend_mm",
    "mean_abs_waviness_mm",
    "mean_abs_twist_mm",
    "mean_abs_local_bump_mm",
]

summary_rows = []
for col in DIAGNOSTIC_COLUMNS:
    summary_rows.append(
        {
            "variable": col,
            "failure_mean": float(failure_df[col].mean()),
            "success_mean": float(success_df[col].mean()),
            "failure_median": float(failure_df[col].median()),
            "success_median": float(success_df[col].median()),
            "failure_n": len(failure_df),
            "success_n": len(success_df),
        }
    )

worst_stage_counts = failure_df["worst_stage_number"].value_counts().sort_index()

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(FAILURE_SUMMARY_FILE, index=False)

print("\n" + "=" * 60)
print(f"FAILURE CASES (n={len(failure_df)}) vs SUCCESS CASES (n={len(success_df)})")
print("=" * 60)
print("Descriptive comparison only -- n is too small for a formal test.\n")
for _, row in summary_df.iterrows():
    print(
        f"  {row['variable']:32s}  failure mean={row['failure_mean']:.4f}"
        f"   success mean={row['success_mean']:.4f}"
    )

print("\nWhich stage the worst single-stage hybrid loss occurred at, across failure cases:")
print(worst_stage_counts.to_string())

print(f"\nSaved: {FAILURE_CASES_FILE}")
print(f"Saved: {FAILURE_SUMMARY_FILE}")
print("\nDone.")