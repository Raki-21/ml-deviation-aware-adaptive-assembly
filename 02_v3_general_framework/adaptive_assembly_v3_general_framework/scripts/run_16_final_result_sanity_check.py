from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES = PROJECT_ROOT / "results" / "tables"

print("Final V3.1 Result Sanity Check")
print(f"Project root: {PROJECT_ROOT}")
print("")

# ------------------------------------------------------------
# 1. Adaptation-level comparison
# ------------------------------------------------------------
adaptation_file = TABLES / "adaptation_level_comparison.csv"

if adaptation_file.exists():
    df = pd.read_csv(adaptation_file)
    print("1. Adaptation-level comparison")
    print(df)
    print("")
else:
    print("[MISSING] adaptation_level_comparison.csv")

# ------------------------------------------------------------
# 2. Recommendation status summary
# ------------------------------------------------------------
status_file = TABLES / "recommendation_status_summary.csv"

if status_file.exists():
    df = pd.read_csv(status_file)
    print("2. Recommendation status summary")
    print(df)
    print("")
else:
    print("[MISSING] recommendation_status_summary.csv")

# ------------------------------------------------------------
# 3. Failure metric analysis
# ------------------------------------------------------------
failure_metric_file = TABLES / "failure_analysis_by_metric.csv"

if failure_metric_file.exists():
    df = pd.read_csv(failure_metric_file)
    print("3. Failure analysis by metric")
    print(df)
    print("")
else:
    print("[MISSING] failure_analysis_by_metric.csv")

# ------------------------------------------------------------
# 4. Model screening summary
# ------------------------------------------------------------
model_file = TABLES / "model_screening_summary.csv"

if model_file.exists():
    df = pd.read_csv(model_file)
    print("4. Model screening summary")
    print(df)
    print("")
else:
    print("[MISSING] model_screening_summary.csv")

# ------------------------------------------------------------
# 5. Optimization efficiency summary
# ------------------------------------------------------------
opt_file = TABLES / "optimization_efficiency_summary.csv"

if opt_file.exists():
    df = pd.read_csv(opt_file)
    print("5. Optimization efficiency summary")
    print(df)
    print("")
else:
    print("[MISSING] optimization_efficiency_summary.csv")

# ------------------------------------------------------------
# 6. Capability summary
# ------------------------------------------------------------
capability_file = TABLES / "correction_capability_summary.csv"

if capability_file.exists():
    df = pd.read_csv(capability_file)
    print("6. Correction capability summary")
    print(df)
    print("")
else:
    print("[MISSING] correction_capability_summary.csv")

print("Sanity check complete.")
print("Review the printed numbers and confirm they match the frozen interpretation.")