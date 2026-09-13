"""
Configuration for Phase 8:
Objective-Aligned Residual Learning.

All new Phase 8 outputs are written only inside
08_objective_aligned_residual_learning.

The validated V3.3 and post-freeze Phase 1-7 folders are treated as
read-only sources.
"""

from pathlib import Path


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PHASE8_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PHASE8_ROOT.parent

V33_ROOT = REPO_ROOT / "06_v3_3_nonlinear_hybrid_upgrade"
V33_SRC = V33_ROOT / "src"

POSTFREEZE_ROOT = REPO_ROOT / "07_postfreeze_method_validation"

DATA_ROOT = PHASE8_ROOT / "data"
TRAINING_DATA_DIR = DATA_ROOT / "training"
DEVELOPMENT_DATA_DIR = DATA_ROOT / "development"
FINAL_DATA_DIR = DATA_ROOT / "final"

MODEL_DIR = PHASE8_ROOT / "models"

RESULTS_ROOT = PHASE8_ROOT / "results"
TABLES_DIR = RESULTS_ROOT / "tables"
FIGURES_DIR = RESULTS_ROOT / "figures"
LOGS_DIR = RESULTS_ROOT / "logs"


# ============================================================
# ASSEMBLY MODEL
# ============================================================

NONLINEARITY_STRENGTH = 1.0
FIXTURE_DRIFT_STD_MM = 0.015


# ============================================================
# PHASE 8 DATASETS
# ============================================================

N_TRAIN_ASSEMBLIES = 300
N_DEVELOPMENT_ASSEMBLIES = 100
N_FINAL_ASSEMBLIES = 300

TRAIN_SEED = 810801
DEVELOPMENT_SEED = 810802
FINAL_SEED = 810803


# ============================================================
# MACHINE-LEARNING RANDOM STATES
# ============================================================

RF_RANDOM_STATE = 810804
GB_RANDOM_STATE = 810805
BOOTSTRAP_RANDOM_STATE = 810806


# ============================================================
# HIGH-BUDGET FINITE REFERENCE
# ============================================================

REFERENCE_DE_MAXITER = 30
REFERENCE_DE_POPSIZE = 10
REFERENCE_DE_TOL = 1e-7
REFERENCE_DE_POLISH = True

REFERENCE_IMPROVEMENT_TOL = 1e-9


# ============================================================
# RANDOM FOREST
# ============================================================

RF_N_ESTIMATORS = 500
RF_MIN_SAMPLES_LEAF = 2
RF_MAX_FEATURES = "sqrt"
RF_N_JOBS = 2


# ============================================================
# GRADIENT BOOSTING
# ============================================================

GB_N_ESTIMATORS = 300
GB_LEARNING_RATE = 0.03
GB_MAX_DEPTH = 3
GB_MIN_SAMPLES_LEAF = 2


# ============================================================
# STATISTICS
# ============================================================

SIGNIFICANCE_LEVEL = 0.05
BOOTSTRAP_SAMPLES = 10000


# ============================================================
# NUMERICAL SAFETY
# ============================================================

QUALITY_COMPARISON_TOL = 1e-9
BOUND_CHECK_TOL = 1e-9


def ensure_phase8_directories():
    """Create Phase 8 output directories only."""
    directories = [
        TRAINING_DATA_DIR,
        DEVELOPMENT_DATA_DIR,
        FINAL_DATA_DIR,
        MODEL_DIR,
        TABLES_DIR,
        FIGURES_DIR,
        LOGS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_phase8_directories()

    print("=" * 68)
    print("PHASE 8 CONFIGURATION CHECK")
    print("=" * 68)

    print(f"Phase 8 root          : {PHASE8_ROOT}")
    print(f"V3.3 root (read only) : {V33_ROOT}")
    print(f"Post-freeze root      : {POSTFREEZE_ROOT}")

    print("\nDataset sizes")
    print(f"  Training assemblies    : {N_TRAIN_ASSEMBLIES}")
    print(f"  Development assemblies : {N_DEVELOPMENT_ASSEMBLIES}")
    print(f"  Final assemblies       : {N_FINAL_ASSEMBLIES}")

    print("\nSeeds")
    print(f"  Training    : {TRAIN_SEED}")
    print(f"  Development : {DEVELOPMENT_SEED}")
    print(f"  Final       : {FINAL_SEED}")

    print("\nPASS: Phase 8 configuration loaded successfully.")