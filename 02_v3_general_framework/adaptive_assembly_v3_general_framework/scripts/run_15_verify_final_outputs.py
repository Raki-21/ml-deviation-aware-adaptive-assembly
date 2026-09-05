from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "results/tables/batch_aware_dataset.csv",
    "results/tables/batch_aware_dataset_summary.csv",
    "results/tables/adaptation_level_comparison.csv",
    "results/tables/adaptation_selected_cases.csv",
    "results/tables/recommendation_status_cases.csv",
    "results/tables/recommendation_status_summary.csv",
    "results/tables/recommendation_status_by_scenario.csv",
    "results/tables/failure_analysis_by_strategy.csv",
    "results/tables/failure_analysis_by_scenario.csv",
    "results/tables/failure_analysis_by_metric.csv",
    "results/tables/individual_adaptive_failed_cases.csv",
    "results/tables/model_screening_results.csv",
    "results/tables/model_screening_summary.csv",
    "results/tables/optimization_efficiency_summary.csv",
    "results/tables/optimization_efficiency_checkpoints.csv",
    "results/tables/optimization_efficiency_curve.csv",
    "results/tables/correction_capability_map.csv",
    "results/tables/correction_capability_summary.csv",

    "results/plots/v3/adaptation_strategy_quality_comparison.png",
    "results/plots/v3/adaptation_strategy_pass_rate_comparison.png",
    "results/plots/v3/recommendation_status_distribution.png",
    "results/plots/v3/failure_metric_comparison.png",
    "results/plots/v3/optimization_efficiency_curve.png",
    "results/plots/v3/model_screening_rmse_comparison.png",
    "results/plots/v3/model_screening_r2_comparison.png",
    "results/plots/v3/correction_capability_map.png",
    "results/plots/v3/correction_difficulty_ranking.png",

    "README_PROJECT_OVERVIEW.md",
    "RESULTS_FREEZE_V3_1.md",
    "V3_RESULTS_INTERPRETATION.md",
    "TECHNICAL_FREEZE_V3_1_COMPLETE.md",
    "docs/thesis_notes/VALID_SCRIPT_EXECUTION_CHECKLIST.md",
    "docs/supervisor_updates/SUPERVISOR_UPDATE_V3_1.md",
]

missing = []

print("Final Version 3.1 output verification")
print(f"Project root: {PROJECT_ROOT}")
print("")

for file in REQUIRED_FILES:
    path = PROJECT_ROOT / file
    if path.exists():
        print(f"[OK]      {file}")
    else:
        print(f"[MISSING] {file}")
        missing.append(file)

print("")
print("Verification summary")
print(f"Missing files: {len(missing)}")

if missing:
    print("")
    print("Missing file list:")
    for file in missing:
        print(f"- {file}")
    print("")
    print("Status: CHECK REQUIRED")
else:
    print("Status: ALL REQUIRED VERSION 3.1 OUTPUTS FOUND")
