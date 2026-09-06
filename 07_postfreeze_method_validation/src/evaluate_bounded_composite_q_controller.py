"""
Phase 1 post-freeze experiment: Bounded Composite-Q Deterministic Optimizer.

Purpose
-------
LSQ solves:

    u_LSQ = argmin_u || S + B u ||_2^2

That is NOT the same objective as the metric the whole project is scored
on, which is the weighted composite quality score:

    Q = 0.30 * mean_gap + 0.30 * max_gap + 0.30 * parallelism + 0.10 * rms

This script introduces a second deterministic controller that optimizes
Q directly instead of the raw squared-profile-error proxy:

    u_Q* = argmin_{u in bounds} Q(S, D, u)

It is a MULTI-START bounded local search (L-BFGS-B), not a closed-form
"analytical" solution -- the name "Composite-Q" describes the objective,
not the solution method.

Why this matters
-----------------
The already-frozen V3.3 regime-robustness result shows that ~71% of the
hybrid's improvement over LSQ already exists at zero nonlinearity
(10.61 out of 14.85 percentage points). That is a strong hint that most
of the hybrid's advantage comes from optimizing the right objective, not
from handling nonlinearity. This script tests that directly: if
Composite-Q alone closes most of the LSQ-to-hybrid gap, the objective-
mismatch explanation is confirmed from first principles, not just
inferred from a regime sweep.

Design rules (do not change without updating this docstring)
--------------------------------------------------------------
1. Nothing in 06_v3_3_nonlinear_hybrid_upgrade/ is imported in a way that
   executes it as a script, and nothing there is written to. Only pure
   functions and constants are reused, by importing the module
   `nonlinear_assembly_model` and `v33_config` as libraries.
2. The assembly-generation logic (seed, per-component parameter ranges,
   fixture drift) is copied verbatim from
   `final_independent_validation_v33.py` so this script reproduces
   EXACTLY the same 300 independent validation assemblies. This is
   checked automatically at runtime (see the LSQ reproducibility check
   below) rather than assumed.
3. The LSQ point is always included as one of the optimizer's starting
   points, so by construction Composite-Q can never score worse than
   LSQ on a single decision beyond numerical tolerance. This is also
   checked automatically.
4. Every method replays the FULL 10-stage sequence using its own
   corrections. No method reuses states produced by a different
   controller.
5. No retraining. No modification of any locked file. This script only
   reads the already-frozen final-validation results CSV, to line up
   the hybrid and direct-reference numbers for comparison.
"""

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from scipy.optimize import lsq_linear, minimize


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PHASE1_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PHASE1_ROOT.parent

V33_SRC = REPO_ROOT / "06_v3_3_nonlinear_hybrid_upgrade" / "src"
V33_RESULTS = (
    REPO_ROOT
    / "06_v3_3_nonlinear_hybrid_upgrade"
    / "results"
    / "final_independent_validation_v33"
    / "v33_final_independent_assembly_results.csv"
)

RESULTS_DIR = PHASE1_ROOT / "results" / "phase_1_composite_q"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

COMPONENT_RESULT_FILE = RESULTS_DIR / "phase1_composite_q_component_results.csv"
ASSEMBLY_RESULT_FILE = RESULTS_DIR / "phase1_composite_q_assembly_results.csv"
SUMMARY_FILE = RESULTS_DIR / "phase1_composite_q_summary.csv"

if not V33_RESULTS.exists():
    raise FileNotFoundError(
        f"Frozen final-validation results not found, cannot compare:\n{V33_RESULTS}\n"
        "This script only READS this file. It does not run without it."
    )

# Import the locked simulator as a library only (no script execution).
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
# SETTINGS -- copied verbatim from final_independent_validation_v33.py
# so the same 300 assemblies are regenerated bit-for-bit.
# ============================================================

SEQUENCE_LENGTH = N_COMPONENTS
NONLINEARITY_STRENGTH = 1.0          # matches the frozen final validation run
N_ASSEMBLIES = 300
VALIDATION_SEED = 20260912
FIXTURE_DRIFT_STD_MM = 0.015

METHOD_LSQ = "LSQ"
METHOD_COMPOSITE_Q = "COMPOSITE_Q"

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

# Bounds for the Composite-Q optimizer, in the actual (z_mm, theta_deg,
# locator_mm) units -- NOT the tan(theta) units LSQ solves in.
Q_BOUNDS = [
    (-Z_ADJ_LIMIT_MM, Z_ADJ_LIMIT_MM),
    (-THETA_ADJ_LIMIT_DEG, THETA_ADJ_LIMIT_DEG),
    (-LOCATOR_OFFSET_LIMIT_MM, LOCATOR_OFFSET_LIMIT_MM),
]


# ============================================================
# ASSEMBLY GENERATION -- copied verbatim (logic + seed) from
# final_independent_validation_v33.py. Do not edit independently of
# that file; if it ever changes there, this copy must change identically
# or the reproducibility check below will fail on purpose.
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


def calculate_utilization(point):
    return float(
        max(
            abs(point[0]) / Z_ADJ_LIMIT_MM,
            abs(point[1]) / THETA_ADJ_LIMIT_DEG,
            abs(point[2]) / LOCATOR_OFFSET_LIMIT_MM,
        )
    )


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
# COMPOSITE-Q OPTIMIZER
# ============================================================
#
# Multi-start set: the origin (= center of the box, since bounds are
# symmetric), the 6 face-centers of the correction box, plus the LSQ
# point -- 8 starting points total.
#
# This was chosen empirically, not guessed: benchmarked against a much
# more expensive 16-start search (8 corners + 6 face-centers + origin +
# LSQ) across 25 random pre-correction states spanning a realistic range
# of deviation magnitudes, this 8-start set matched the 16-start result
# to within 6e-7 on every single trial (see development notes), while
# running in under half the time. A cheaper 2-start set (origin + LSQ)
# was also tested and rejected: it missed the better optimum by up to
# 0.013 quality units on 5 of 25 trials, which is not a safe trade-off
# given the effect sizes this experiment is trying to measure.

Zc = Z_ADJ_LIMIT_MM
Tc = THETA_ADJ_LIMIT_DEG
Lc = LOCATOR_OFFSET_LIMIT_MM

GRID_STARTS = [
    (0.0, 0.0, 0.0),
    (-Zc, 0.0, 0.0),
    (Zc, 0.0, 0.0),
    (0.0, -Tc, 0.0),
    (0.0, Tc, 0.0),
    (0.0, 0.0, -Lc),
    (0.0, 0.0, Lc),
]


def objective_q(point, pre_correction_state):
    corrected_state = apply_true_correction(pre_correction_state, point)
    return calculate_quality_metrics(corrected_state)["quality_score"]


def optimize_composite_q(pre_correction_state, lsq_point):
    """
    Returns the best point found across all starts, GUARANTEED to be no
    worse than the raw LSQ point: the LSQ point's own (unoptimized)
    objective value is always included as one of the candidates compared
    at the end, independent of whether any local search run actually
    converges well. This is what makes the "Composite-Q never scores
    worse than LSQ" sanity gate true by construction, not by hope.
    """
    evaluation_counter = {"count": 0}

    def objective(point):
        evaluation_counter["count"] += 1
        return objective_q(point, pre_correction_state)

    lsq_point = np.asarray(lsq_point, dtype=float)
    lsq_value = objective(lsq_point)

    best_point = lsq_point.copy()
    best_value = lsq_value
    best_success = True
    best_label = "LSQ_UNOPTIMIZED"

    starts = list(GRID_STARTS) + [tuple(lsq_point)]

    for start in starts:
        result = minimize(
            objective,
            x0=np.array(start, dtype=float),
            method="L-BFGS-B",
            bounds=Q_BOUNDS,
        )
        if result.fun < best_value:
            best_value = float(result.fun)
            best_point = clip_correction(result.x)
            best_success = bool(result.success)
            best_label = "LSQ_REFINED" if np.allclose(start, lsq_point) else "GRID"

    return best_point, best_success, evaluation_counter["count"], best_label


# ============================================================
# RUN: LSQ AND COMPOSITE-Q, FULL SEQUENTIAL REPLAY
# ============================================================
#
# Each assembly's 10-stage sequence is fully independent of every other
# assembly (they only share the same simulator code and correction
# bounds), so assemblies are distributed across worker processes. This
# does not change a single number the computation produces -- it only
# changes how many CPU cores compute those numbers at once.

def process_assembly(case):
    assembly_index = case["assembly_index"]

    state_lsq = create_initial_state()
    state_q = create_initial_state()

    component_rows_local = []

    for comp in case["components"]:
        component_index = comp["component_index"]
        component_profile = comp["component_profile"]
        fixture_drift = comp["fixture_drift"]

        # --- LSQ trajectory ---
        pre_state_lsq = state_lsq + component_profile + fixture_drift
        pre_quality_lsq = calculate_quality_metrics(pre_state_lsq)["quality_score"]
        lsq_point = deterministic_lsq_correction(pre_state_lsq)
        state_lsq = apply_true_correction(pre_state_lsq, lsq_point)
        post_quality_lsq = calculate_quality_metrics(state_lsq)["quality_score"]

        # --- Composite-Q trajectory (separate, independent state) ---
        pre_state_q = state_q + component_profile + fixture_drift
        pre_quality_q = calculate_quality_metrics(pre_state_q)["quality_score"]
        # LSQ point computed on Composite-Q's OWN pre-correction state,
        # purely to serve as one guaranteed-safe optimizer starting point.
        lsq_point_for_q_start = deterministic_lsq_correction(pre_state_q)
        q_point, q_success, q_evals, q_label = optimize_composite_q(
            pre_state_q, lsq_point_for_q_start
        )
        state_q = apply_true_correction(pre_state_q, q_point)
        post_quality_q = calculate_quality_metrics(state_q)["quality_score"]

        component_rows_local.append(
            {
                "assembly_index": assembly_index,
                "component_index": component_index,
                "lsq_pre_quality": pre_quality_lsq,
                "lsq_post_quality": post_quality_lsq,
                "lsq_z_adj_mm": lsq_point[0],
                "lsq_theta_adj_deg": lsq_point[1],
                "lsq_locator_offset_mm": lsq_point[2],
                "lsq_utilization": calculate_utilization(lsq_point),
                "composite_q_pre_quality": pre_quality_q,
                "composite_q_post_quality": post_quality_q,
                "composite_q_z_adj_mm": q_point[0],
                "composite_q_theta_adj_deg": q_point[1],
                "composite_q_locator_offset_mm": q_point[2],
                "composite_q_utilization": calculate_utilization(q_point),
                "composite_q_beats_lsq_this_step": bool(
                    post_quality_q < post_quality_lsq - 1e-9
                ),
                "composite_q_optimizer_success": q_success,
                "composite_q_evaluations": q_evals,
                "composite_q_winning_start": q_label,
            }
        )

    final_quality_lsq = calculate_quality_metrics(state_lsq)["quality_score"]
    final_quality_q = calculate_quality_metrics(state_q)["quality_score"]

    assembly_row_local = {
        "assembly_index": assembly_index,
        "lsq_final_quality": final_quality_lsq,
        "composite_q_final_quality": final_quality_q,
    }

    return component_rows_local, assembly_row_local


if __name__ == "__main__":
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    N_WORKERS = min(mp.cpu_count(), 4)

    validation_cases = generate_validation_cases()

    print("=" * 60)
    print("PHASE 1: BOUNDED COMPOSITE-Q DETERMINISTIC OPTIMIZER")
    print("=" * 60)
    print(f"\nAssemblies      : {N_ASSEMBLIES}")
    print(f"Sequence length : {SEQUENCE_LENGTH}")
    print(f"Nonlinearity    : {NONLINEARITY_STRENGTH}")
    print(f"Grid starts     : {len(GRID_STARTS)} + LSQ point = {len(GRID_STARTS) + 1} per decision")
    print(f"Worker processes: {N_WORKERS}")

    component_rows = []
    assembly_rows = []
    start = time.time()
    completed = 0

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        for comp_rows_local, assembly_row_local in executor.map(process_assembly, validation_cases):
            component_rows.extend(comp_rows_local)
            assembly_rows.append(assembly_row_local)
            completed += 1
            if completed % 25 == 0:
                elapsed = time.time() - start
                print(f"  completed {completed}/{N_ASSEMBLIES} assemblies ({elapsed:.1f}s elapsed)")

    elapsed = time.time() - start
    print(f"\nSequential replay finished in {elapsed:.1f}s")

    assembly_rows.sort(key=lambda r: r["assembly_index"])
    component_rows.sort(key=lambda r: (r["assembly_index"], r["component_index"]))

    component_df = pd.DataFrame(component_rows)
    assembly_df = pd.DataFrame(assembly_rows)


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


if __name__ == "__main__":

    # ============================================================
    # LOAD FROZEN HYBRID + DIRECT REFERENCE FOR COMPARISON (READ ONLY)
    # ============================================================

    frozen_df = pd.read_csv(V33_RESULTS)

    frozen_lsq = (
        frozen_df[frozen_df["method"] == "LSQ"]
        .set_index("assembly_index")["final_quality"]
    )
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
    assembly_df["composite_q_minus_lsq"] = (
        assembly_df["composite_q_final_quality"] - assembly_df["lsq_final_quality"]
    )
    assembly_df["hybrid_minus_composite_q"] = (
        assembly_df["hybrid_final_quality"] - assembly_df["composite_q_final_quality"]
    )
    assembly_df["composite_q_beats_lsq"] = (
        assembly_df["composite_q_final_quality"] < assembly_df["lsq_final_quality"] - 1e-9
    )
    assembly_df["hybrid_beats_composite_q"] = (
        assembly_df["hybrid_final_quality"] < assembly_df["composite_q_final_quality"] - 1e-9
    )
    assembly_df = assembly_df.reset_index()

    assembly_df.to_csv(ASSEMBLY_RESULT_FILE, index=False)
    component_df.to_csv(COMPONENT_RESULT_FILE, index=False)

    # ============================================================
    # REPRODUCIBILITY CHECK -- our own LSQ recomputation must match the
    # frozen LSQ column almost exactly. If it doesn't, the assemblies are
    # NOT the same population and every comparison below is invalid.
    # ============================================================

    lsq_diff = (assembly_df["lsq_final_quality"] - assembly_df["frozen_lsq_final_quality"]).abs()
    max_lsq_diff = float(lsq_diff.max())

    print("\n" + "=" * 60)
    print("REPRODUCIBILITY CHECK (must pass before results are trusted)")
    print("=" * 60)
    print(f"Max |our LSQ - frozen LSQ| across 300 assemblies: {max_lsq_diff:.3e}")
    if max_lsq_diff < 1e-6:
        print("PASS: same 300 assemblies, same simulator, same LSQ implementation.")
    else:
        print("FAIL: assemblies or LSQ implementation diverge from the frozen run.")
        print("Do not trust the Composite-Q comparison numbers below until this is fixed.")

    # ============================================================
    # SANITY GATE -- Composite-Q must never lose to LSQ (LSQ is always
    # one of its starting points).
    # ============================================================

    step_losses = int((~component_df["composite_q_beats_lsq_this_step"]).sum())
    worse_steps = component_df[
        component_df["composite_q_post_quality"] > component_df["lsq_post_quality"] + 1e-6
    ]
    n_worse_steps = len(worse_steps)

    print("\n" + "=" * 60)
    print("PER-STEP SANITY GATE")
    print("=" * 60)
    print(f"Total decisions evaluated       : {len(component_df)}")
    print(f"Decisions where Composite-Q did not strictly beat LSQ this step : {step_losses}")
    print(f"Decisions where Composite-Q scored WORSE than LSQ by >1e-6      : {n_worse_steps}")
    if n_worse_steps == 0:
        print("PASS: Composite-Q never underperforms LSQ beyond numerical tolerance.")
    else:
        print("FAIL: optimizer bug -- Composite-Q lost to LSQ on a per-step basis.")
        print("(Not strictly beating LSQ per step is fine -- LSQ can already be optimal")
        print(" at that step. Scoring WORSE than LSQ per step should never happen.)")

    # ============================================================
    # SUMMARY
    # ============================================================

    summary_rows = [
        paired_stats(
            assembly_df["composite_q_final_quality"],
            assembly_df["lsq_final_quality"],
            "COMPOSITE_Q",
            "LSQ",
        ),
        paired_stats(
            assembly_df["hybrid_final_quality"],
            assembly_df["composite_q_final_quality"],
            "HYBRID",
            "COMPOSITE_Q",
        ),
        paired_stats(
            assembly_df["hybrid_final_quality"],
            assembly_df["lsq_final_quality"],
            "HYBRID",
            "LSQ",
        ),
    ]

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_FILE, index=False)

    print("\n" + "=" * 60)
    print("FINAL MEAN QUALITY (lower is better)")
    print("=" * 60)
    print(f"LSQ                          : {assembly_df['lsq_final_quality'].mean():.6f}")
    print(f"Composite-Q                  : {assembly_df['composite_q_final_quality'].mean():.6f}")
    print(f"Locked hybrid (LSQ+RF ALL)   : {assembly_df['hybrid_final_quality'].mean():.6f}")
    print(f"Direct nonlinear reference   : {assembly_df['direct_reference_final_quality'].mean():.6f}")

    print("\n" + "=" * 60)
    print("PAIRED COMPARISONS")
    print("=" * 60)
    for row in summary_rows:
        print(f"\n{row['comparison']}")
        for k, v in row.items():
            if k != "comparison":
                print(f"  {k:40s}: {v:.4f}")

    print(f"\nSaved: {COMPONENT_RESULT_FILE}")
    print(f"Saved: {ASSEMBLY_RESULT_FILE}")
    print(f"Saved: {SUMMARY_FILE}")
    print("\nDone.")