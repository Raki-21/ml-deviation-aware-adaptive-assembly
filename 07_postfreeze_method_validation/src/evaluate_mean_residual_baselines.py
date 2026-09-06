"""
Phase 2 post-freeze experiment: mean-residual baselines.

Purpose
-------
Phase 1 showed that a deterministic optimizer targeting the real composite
Q objective (Composite-Q) beats both LSQ and the locked RF hybrid by a wide
margin, on every single one of 300 assemblies. That raised an obvious
follow-up question: is the RF hybrid's own advantage over LSQ coming from
genuine state-dependent personalization, or mostly from a fixed correction
bias that a simple average would reproduce just as well?

This script answers that with two non-ML controllers built from the exact
same residual training targets the RF model itself was trained on
(target_delta_z_mm, target_delta_theta_deg, target_delta_locator_mm in
06_v3_3_nonlinear_hybrid_upgrade/data/residual_learning_full/v33_residual_learning_full_train.csv),
using ONLY the "train" split -- the same split the locked model used --
so this is a fair, apples-to-apples comparison, not an easier or harder
one.

Two controllers
----------------
1. Global mean residual: one fixed [dz, dtheta, dlocator] vector, the
   average residual across all 3000 training rows, applied identically
   to u_LSQ on every decision regardless of stage or state.
2. Stage-specific mean residual: ten fixed vectors, one per sequence
   position (1 through 10), each the average residual for that stage
   only. This controller knows the current stage number and the LSQ
   point, and nothing else about the current state.

Neither controller looks at profile features, deviation magnitude, or
anything else the RF model uses. If either one gets close to the hybrid's
performance, that is evidence the hybrid's advantage is largely a
population-level correction bias rather than genuine personalization.

Design rules (same as Phase 1)
--------------------------------
1. Nothing under 06_v3_3_nonlinear_hybrid_upgrade/ is modified. Training
   data and the frozen final-validation results are only ever read.
2. The final-validation assembly population is regenerated with the exact
   same seed and logic as final_independent_validation_v33.py, and checked
   against the frozen LSQ column for exact reproducibility before any
   comparison is trusted.
3. Both new controllers replay the FULL 10-stage sequence with their own
   independent state, exactly like LSQ, Composite-Q, and the hybrid did.
4. No retraining, no looking at the training data's held-out validation
   or development rows -- "train" split only, matching what the locked
   model itself was allowed to see.
"""

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from scipy.optimize import lsq_linear


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PHASE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PHASE_ROOT.parent

V33_SRC = REPO_ROOT / "06_v3_3_nonlinear_hybrid_upgrade" / "src"
V33_RESULTS = (
    REPO_ROOT
    / "06_v3_3_nonlinear_hybrid_upgrade"
    / "results"
    / "final_independent_validation_v33"
    / "v33_final_independent_assembly_results.csv"
)
TRAIN_RESIDUAL_FILE = (
    REPO_ROOT
    / "06_v3_3_nonlinear_hybrid_upgrade"
    / "data"
    / "residual_learning_full"
    / "v33_residual_learning_full_train.csv"
)
PHASE1_ASSEMBLY_FILE = (
    PHASE_ROOT / "results" / "phase_1_composite_q" / "phase1_composite_q_assembly_results.csv"
)

RESULTS_DIR = PHASE_ROOT / "results" / "phase_2_mean_residual"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ASSEMBLY_RESULT_FILE = RESULTS_DIR / "phase2_mean_residual_assembly_results.csv"
SUMMARY_FILE = RESULTS_DIR / "phase2_mean_residual_summary.csv"

for required in (V33_RESULTS, TRAIN_RESIDUAL_FILE):
    if not required.exists():
        raise FileNotFoundError(f"Required input not found (read-only): {required}")

sys.path.insert(0, str(V33_SRC))

from nonlinear_assembly_model import (  # noqa: E402
    create_initial_state,
    deviation_offset,
    deviation_tilt,
    deviation_bend,
    deviation_waviness,
    deviation_twist,
    deviation_local_bump,
    correction_profile_nonlinear,
    calculate_quality_metrics,
)

from v33_config import (  # noqa: E402
    N_COMPONENTS,
    PROFILE_LENGTH_MM,
    N_PROFILE_POINTS,
    Z_ADJ_LIMIT_MM,
    THETA_ADJ_LIMIT_DEG,
    LOCATOR_OFFSET_LIMIT_MM,
)


# ============================================================
# SETTINGS -- identical to Phase 1 / final_independent_validation_v33.py
# ============================================================

SEQUENCE_LENGTH = N_COMPONENTS
NONLINEARITY_STRENGTH = 1.0
N_ASSEMBLIES = 300
VALIDATION_SEED = 20260912
FIXTURE_DRIFT_STD_MM = 0.015

s = np.linspace(0.0, 1.0, N_PROFILE_POINTS)

Z_BASIS = np.ones_like(s, dtype=float)
THETA_BASIS = PROFILE_LENGTH_MM * (s - 0.5)
LOCATOR_SIGMA = 0.12
LOCATOR_BASIS = np.exp(-((s - 0.5) ** 2) / (2.0 * LOCATOR_SIGMA ** 2))
CORRECTION_BASIS = np.column_stack([Z_BASIS, THETA_BASIS, LOCATOR_BASIS])

THETA_TAN_LIMIT = float(np.tan(np.deg2rad(THETA_ADJ_LIMIT_DEG)))

LOWER_LSQ_BOUNDS = np.array(
    [-Z_ADJ_LIMIT_MM, -THETA_TAN_LIMIT, -LOCATOR_OFFSET_LIMIT_MM], dtype=float
)
UPPER_LSQ_BOUNDS = np.array(
    [Z_ADJ_LIMIT_MM, THETA_TAN_LIMIT, LOCATOR_OFFSET_LIMIT_MM], dtype=float
)


# ============================================================
# ASSEMBLY GENERATION -- copied verbatim from
# final_independent_validation_v33.py (same as Phase 1's copy).
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


def build_component_profile(parameters):
    return (
        deviation_offset(parameters["offset_mm"])
        + deviation_tilt(parameters["tilt_deg"])
        + deviation_bend(parameters["bend_mm"])
        + deviation_waviness(parameters["waviness_mm"], waves=3)
        + deviation_twist(parameters["twist_mm"])
        + deviation_local_bump(parameters["local_bump_mm"])
    )


def generate_validation_cases():
    rng = np.random.default_rng(VALIDATION_SEED)
    cases = []
    for assembly_index in range(1, N_ASSEMBLIES + 1):
        components = []
        for component_index in range(1, SEQUENCE_LENGTH + 1):
            parameters = generate_component_parameters(rng)
            component_profile = build_component_profile(parameters)
            fixture_drift = rng.normal(
                loc=0.0, scale=FIXTURE_DRIFT_STD_MM, size=N_PROFILE_POINTS
            )
            components.append(
                {
                    "component_index": component_index,
                    "component_profile": component_profile,
                    "fixture_drift": fixture_drift,
                }
            )
        cases.append({"assembly_index": assembly_index, "components": components})
    return cases


# ============================================================
# LSQ (copied verbatim)
# ============================================================

def deterministic_lsq_correction(uncorrected_state):
    result = lsq_linear(
        CORRECTION_BASIS,
        -uncorrected_state,
        bounds=(LOWER_LSQ_BOUNDS, UPPER_LSQ_BOUNDS),
        method="trf",
        lsmr_tol="auto",
    )
    z_adj_mm = float(result.x[0])
    tan_theta = float(result.x[1])
    theta_adj_deg = float(np.rad2deg(np.arctan(tan_theta)))
    theta_adj_deg = float(np.clip(theta_adj_deg, -THETA_ADJ_LIMIT_DEG, THETA_ADJ_LIMIT_DEG))
    locator_offset_mm = float(result.x[2])
    return np.array([z_adj_mm, theta_adj_deg, locator_offset_mm], dtype=float)


def clip_correction(point):
    point = np.asarray(point, dtype=float).copy()
    point[0] = np.clip(point[0], -Z_ADJ_LIMIT_MM, Z_ADJ_LIMIT_MM)
    point[1] = np.clip(point[1], -THETA_ADJ_LIMIT_DEG, THETA_ADJ_LIMIT_DEG)
    point[2] = np.clip(point[2], -LOCATOR_OFFSET_LIMIT_MM, LOCATOR_OFFSET_LIMIT_MM)
    return point


def apply_true_correction(pre_correction_state, point):
    correction = correction_profile_nonlinear(
        previous_state=pre_correction_state,
        z_adj_mm=float(point[0]),
        theta_adj_deg=float(point[1]),
        locator_offset_mm=float(point[2]),
        nonlinearity_strength=NONLINEARITY_STRENGTH,
    )
    return pre_correction_state + correction


# ============================================================
# LEARN THE TWO FIXED RESIDUAL VECTORS FROM THE TRAIN SPLIT ONLY
# ============================================================

train_df = pd.read_csv(TRAIN_RESIDUAL_FILE)

if not (train_df["split"] == "train").all():
    raise RuntimeError(
        "v33_residual_learning_full_train.csv contains rows not labelled "
        "'train' -- refusing to compute residual baselines from anything "
        "outside the model's own training split."
    )

RESIDUAL_COLUMNS = ["target_delta_z_mm", "target_delta_theta_deg", "target_delta_locator_mm"]

GLOBAL_MEAN_RESIDUAL = train_df[RESIDUAL_COLUMNS].mean().to_numpy()

STAGE_MEAN_RESIDUAL = {
    int(stage): row.to_numpy()
    for stage, row in train_df.groupby("component_index")[RESIDUAL_COLUMNS].mean().iterrows()
}

if sorted(STAGE_MEAN_RESIDUAL.keys()) != list(range(1, SEQUENCE_LENGTH + 1)):
    raise RuntimeError(
        f"Expected stage-mean residuals for stages 1..{SEQUENCE_LENGTH}, "
        f"got {sorted(STAGE_MEAN_RESIDUAL.keys())}"
    )


# ============================================================
# RUN: LSQ, GLOBAL MEAN RESIDUAL, STAGE MEAN RESIDUAL
# ============================================================

validation_cases = generate_validation_cases()

print("=" * 60)
print("PHASE 2: MEAN-RESIDUAL BASELINES")
print("=" * 60)
print(f"\nTraining rows used for residual means : {len(train_df)} (split == 'train' only)")
print(f"Global mean residual [dz, dtheta, dlocator] : {GLOBAL_MEAN_RESIDUAL}")
print("Stage-specific mean residuals:")
for stage in range(1, SEQUENCE_LENGTH + 1):
    print(f"  stage {stage:2d}: {STAGE_MEAN_RESIDUAL[stage]}")

assembly_rows = []
start = time.time()

for case in validation_cases:
    assembly_index = case["assembly_index"]

    state_lsq = create_initial_state()
    state_global = create_initial_state()
    state_stage = create_initial_state()

    for comp in case["components"]:
        component_index = comp["component_index"]
        component_profile = comp["component_profile"]
        fixture_drift = comp["fixture_drift"]

        # --- LSQ (recomputed here purely for the reproducibility check) ---
        pre_state_lsq = state_lsq + component_profile + fixture_drift
        lsq_point = deterministic_lsq_correction(pre_state_lsq)
        state_lsq = apply_true_correction(pre_state_lsq, lsq_point)

        # --- Global mean residual ---
        pre_state_global = state_global + component_profile + fixture_drift
        lsq_point_global = deterministic_lsq_correction(pre_state_global)
        global_point = clip_correction(lsq_point_global + GLOBAL_MEAN_RESIDUAL)
        state_global = apply_true_correction(pre_state_global, global_point)

        # --- Stage-specific mean residual ---
        pre_state_stage = state_stage + component_profile + fixture_drift
        lsq_point_stage = deterministic_lsq_correction(pre_state_stage)
        stage_point = clip_correction(lsq_point_stage + STAGE_MEAN_RESIDUAL[component_index])
        state_stage = apply_true_correction(pre_state_stage, stage_point)

    assembly_rows.append(
        {
            "assembly_index": assembly_index,
            "lsq_final_quality": calculate_quality_metrics(state_lsq)["quality_score"],
            "global_mean_residual_final_quality": calculate_quality_metrics(state_global)[
                "quality_score"
            ],
            "stage_mean_residual_final_quality": calculate_quality_metrics(state_stage)[
                "quality_score"
            ],
        }
    )

elapsed = time.time() - start
print(f"\nSequential replay of {N_ASSEMBLIES} assemblies finished in {elapsed:.1f}s")

assembly_df = pd.DataFrame(assembly_rows)


# ============================================================
# LOAD FROZEN HYBRID/DIRECT + PHASE 1 COMPOSITE-Q (READ ONLY)
# ============================================================

frozen_df = pd.read_csv(V33_RESULTS)
frozen_lsq = frozen_df[frozen_df["method"] == "LSQ"].set_index("assembly_index")["final_quality"]
frozen_hybrid = (
    frozen_df[frozen_df["method"] == "LSQ_PLUS_ML_ALL"]
    .set_index("assembly_index")["final_quality"]
)
frozen_direct = (
    frozen_df[frozen_df["method"] == "DIRECT_NONLINEAR_REFERENCE"]
    .set_index("assembly_index")["final_quality"]
)

assembly_df = assembly_df.set_index("assembly_index")
assembly_df["frozen_lsq_final_quality"] = frozen_lsq
assembly_df["hybrid_final_quality"] = frozen_hybrid
assembly_df["direct_reference_final_quality"] = frozen_direct

if PHASE1_ASSEMBLY_FILE.exists():
    phase1_df = pd.read_csv(PHASE1_ASSEMBLY_FILE).set_index("assembly_index")
    assembly_df["composite_q_final_quality"] = phase1_df["composite_q_final_quality"]
    HAVE_COMPOSITE_Q = True
else:
    print(
        f"\nWARNING: Phase 1 results not found at {PHASE1_ASSEMBLY_FILE} -- "
        "skipping comparisons against Composite-Q."
    )
    HAVE_COMPOSITE_Q = False

assembly_df = assembly_df.reset_index()
assembly_df.to_csv(ASSEMBLY_RESULT_FILE, index=False)


# ============================================================
# REPRODUCIBILITY CHECK
# ============================================================

lsq_diff = (assembly_df["lsq_final_quality"] - assembly_df["frozen_lsq_final_quality"]).abs()
max_lsq_diff = float(lsq_diff.max())

print("\n" + "=" * 60)
print("REPRODUCIBILITY CHECK")
print("=" * 60)
print(f"Max |our LSQ - frozen LSQ| across 300 assemblies: {max_lsq_diff:.3e}")
print("PASS" if max_lsq_diff < 1e-6 else "FAIL -- do not trust the comparisons below")


# ============================================================
# SUMMARY
# ============================================================

def paired_stats(a, b, label_a, label_b):
    diff = a - b
    return {
        "comparison": f"{label_a} vs {label_b}",
        f"mean_{label_a.lower()}": float(a.mean()),
        f"mean_{label_b.lower()}": float(b.mean()),
        "mean_difference": float(diff.mean()),
        "median_difference": float(diff.median()),
        f"{label_a.lower()}_win_rate_percent": float(100.0 * (diff < -1e-9).mean()),
        f"{label_b.lower()}_win_rate_percent": float(100.0 * (diff > 1e-9).mean()),
        "relative_improvement_percent": float(
            100.0 * (b.mean() - a.mean()) / max(b.mean(), 1e-12)
        ),
    }


comparisons = [
    ("GLOBAL", assembly_df["global_mean_residual_final_quality"], "LSQ", assembly_df["lsq_final_quality"]),
    ("STAGE", assembly_df["stage_mean_residual_final_quality"], "LSQ", assembly_df["lsq_final_quality"]),
    ("HYBRID", assembly_df["hybrid_final_quality"], "GLOBAL", assembly_df["global_mean_residual_final_quality"]),
    ("HYBRID", assembly_df["hybrid_final_quality"], "STAGE", assembly_df["stage_mean_residual_final_quality"]),
    ("GLOBAL", assembly_df["global_mean_residual_final_quality"], "STAGE", assembly_df["stage_mean_residual_final_quality"]),
]

if HAVE_COMPOSITE_Q:
    comparisons.extend(
        [
            ("COMPOSITE_Q", assembly_df["composite_q_final_quality"], "GLOBAL", assembly_df["global_mean_residual_final_quality"]),
            ("COMPOSITE_Q", assembly_df["composite_q_final_quality"], "STAGE", assembly_df["stage_mean_residual_final_quality"]),
        ]
    )

summary_rows = [paired_stats(a, b, la, lb) for la, a, lb, b in comparisons]
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_FILE, index=False)

print("\n" + "=" * 60)
print("FINAL MEAN QUALITY (lower is better)")
print("=" * 60)
print(f"LSQ                          : {assembly_df['lsq_final_quality'].mean():.6f}")
print(f"Global mean residual         : {assembly_df['global_mean_residual_final_quality'].mean():.6f}")
print(f"Stage mean residual          : {assembly_df['stage_mean_residual_final_quality'].mean():.6f}")
print(f"Locked hybrid (LSQ+RF ALL)   : {assembly_df['hybrid_final_quality'].mean():.6f}")
if HAVE_COMPOSITE_Q:
    print(f"Composite-Q (Phase 1)        : {assembly_df['composite_q_final_quality'].mean():.6f}")
print(f"Direct nonlinear reference   : {assembly_df['direct_reference_final_quality'].mean():.6f}")

print("\n" + "=" * 60)
print("PAIRED COMPARISONS")
print("=" * 60)
for row in summary_rows:
    print(f"\n{row['comparison']}")
    for k, v in row.items():
        if k != "comparison":
            print(f"  {k:40s}: {v:.4f}")

print(f"\nSaved: {ASSEMBLY_RESULT_FILE}")
print(f"Saved: {SUMMARY_FILE}")
print("\nDone.")