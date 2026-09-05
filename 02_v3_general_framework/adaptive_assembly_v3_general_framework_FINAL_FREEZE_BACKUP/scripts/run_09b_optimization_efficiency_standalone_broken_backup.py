"""
Standalone Version 3 optimization efficiency comparison.

This script compares:
1. Random Search
2. Surrogate-guided sequential search

It does not import src/optimization_efficiency.py.
It is intentionally self-contained to avoid module errors.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_COLUMNS = [
    "dev_offset",
    "dev_tilt",
    "dev_bend",
    "dev_waviness",
    "dev_noise",
    "dev_twist",
    "fixture_drift",
    "z_adj",
    "theta_adj",
    "locator_offset",
]

DEVIATION_COLUMNS = [
    "dev_offset",
    "dev_tilt",
    "dev_bend",
    "dev_waviness",
    "dev_noise",
    "dev_twist",
    "fixture_drift",
]

PARAMETER_COLUMNS = [
    "z_adj",
    "theta_adj",
    "locator_offset",
]

PARAMETER_BOUNDS = {
    "z_adj": (-2.5, 2.5),
    "theta_adj": (-1.2, 1.2),
    "locator_offset": (-1.0, 1.0),
}


def train_surrogate(df: pd.DataFrame) -> RandomForestRegressor:
    X = df[FEATURE_COLUMNS]
    y = df["quality_score"]

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=250,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    return model


def get_unique_parts(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "batch_id",
        "part_id",
        "scenario_type",
        "severity_level",
        "disturbance_flag",
        *DEVIATION_COLUMNS,
    ]

    parts = (
        df[cols]
        .drop_duplicates(subset=["batch_id", "part_id"])
        .reset_index(drop=True)
    )

    parts["part_case_id"] = np.arange(1, len(parts) + 1)
    return parts


def sample_parameters(rng: np.random.Generator, n: int) -> pd.DataFrame:
    rows = []

    for _ in range(n):
        rows.append(
            {
                name: float(rng.uniform(low, high))
                for name, (low, high) in PARAMETER_BOUNDS.items()
            }
        )

    return pd.DataFrame(rows)


def make_features(part: pd.Series, params: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, p in params.iterrows():
        row = {}

        for col in DEVIATION_COLUMNS:
            row[col] = part[col]

        for col in PARAMETER_COLUMNS:
            row[col] = p[col]

        rows.append(row)

    return pd.DataFrame(rows)


def predict_quality(model: RandomForestRegressor, part: pd.Series, params: pd.DataFrame):
    X = make_features(part, params)
    return model.predict(X[FEATURE_COLUMNS])


def run_random_search(
    model: RandomForestRegressor,
    part: pd.Series,
    n_evaluations: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    params = sample_parameters(rng, n_evaluations)
    preds = predict_quality(model, part, params)

    rows = []
    best_so_far = np.inf

    for i, q in enumerate(preds, start=1):
        q = float(q)
        best_so_far = min(best_so_far, q)

        rows.append(
            {
                "method": "random_search",
                "evaluation": i,
                "predicted_quality": q,
                "best_quality_so_far": best_so_far,
            }
        )

    return pd.DataFrame(rows)


def run_surrogate_guided_search(
    model: RandomForestRegressor,
    part: pd.Series,
    n_evaluations: int,
    rng: np.random.Generator,
    candidate_pool_size: int = 500,
) -> pd.DataFrame:
    """
    Sequential surrogate-guided search.

    At each step, many candidate parameters are generated.
    The trained surrogate predicts quality for all candidates.
    The best predicted candidate is selected.

    This represents intelligent model-based search and is used here as the
    practical optimization-efficiency comparison against random search.
    """

    rows = []
    best_so_far = np.inf

    already_used = []

    for i in range(1, n_evaluations + 1):
        pool = sample_parameters(rng, candidate_pool_size)
        preds = predict_quality(model, part, pool)

        order = np.argsort(preds)
        chosen_index = int(order[0])

        chosen_params = pool.iloc[chosen_index]
        chosen_quality = float(preds[chosen_index])

        already_used.append(chosen_params.to_dict())
        best_so_far = min(best_so_far, chosen_quality)

        rows.append(
            {
                "method": "surrogate_guided_search",
                "evaluation": i,
                "predicted_quality": chosen_quality,
                "best_quality_so_far": best_so_far,
            }
        )

    return pd.DataFrame(rows)


def select_test_cases(parts: pd.DataFrame, n_cases: int = 30) -> pd.DataFrame:
    scenario_types = sorted(parts["scenario_type"].dropna().unique())
    cases_per_scenario = max(1, n_cases // len(scenario_types))

    selected = []

    for scenario in scenario_types:
        scenario_df = parts[parts["scenario_type"] == scenario]
        selected.append(
            scenario_df.sample(
                n=min(cases_per_scenario, len(scenario_df)),
                random_state=42,
            )
        )

    selected_df = pd.concat(selected, ignore_index=True)

    if len(selected_df) < n_cases:
        remaining = parts.drop(selected_df.index, errors="ignore")
        extra = remaining.sample(
            n=n_cases - len(selected_df),
            random_state=42,
        )
        selected_df = pd.concat([selected_df, extra], ignore_index=True)

    selected_df = selected_df.head(n_cases).reset_index(drop=True)
    selected_df["optimization_case_id"] = np.arange(1, len(selected_df) + 1)

    return selected_df


def summarize(curves: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    final_rows = (
        curves.sort_values("evaluation")
        .groupby(["optimization_case_id", "method"], as_index=False)
        .tail(1)
    )

    summary = (
        final_rows.groupby("method", as_index=False)
        .agg(
            cases=("optimization_case_id", "nunique"),
            final_mean_best_quality=("best_quality_so_far", "mean"),
            final_median_best_quality=("best_quality_so_far", "median"),
            final_std_best_quality=("best_quality_so_far", "std"),
            final_worst_best_quality=("best_quality_so_far", "max"),
            final_best_best_quality=("best_quality_so_far", "min"),
        )
    )

    random_mean = summary.loc[
        summary["method"] == "random_search",
        "final_mean_best_quality",
    ].iloc[0]

    summary["quality_improvement_vs_random_percent"] = (
        (random_mean - summary["final_mean_best_quality"])
        / random_mean
        * 100.0
    )

    checkpoints = []

    for evaluation in [5, 10, 15, 20, 25, 30]:
        sub = curves[curves["evaluation"] == evaluation]

        grouped = (
            sub.groupby("method", as_index=False)
            .agg(
                mean_best_quality=("best_quality_so_far", "mean"),
                median_best_quality=("best_quality_so_far", "median"),
                std_best_quality=("best_quality_so_far", "std"),
                cases=("optimization_case_id", "nunique"),
            )
        )

        grouped["evaluation_budget"] = evaluation
        checkpoints.append(grouped)

    checkpoint_df = pd.concat(checkpoints, ignore_index=True)

    checkpoint_df = checkpoint_df[
        [
            "evaluation_budget",
            "method",
            "mean_best_quality",
            "median_best_quality",
            "std_best_quality",
            "cases",
        ]
    ]

    return summary, checkpoint_df


def main() -> None:
    dataset_path = PROJECT_ROOT / "results" / "tables" / "batch_aware_dataset.csv"
    output_dir = PROJECT_ROOT / "results" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Running standalone optimization efficiency comparison...")
    print(f"Input dataset: {dataset_path}")

    df = pd.read_csv(dataset_path)
    model = train_surrogate(df)
    parts = get_unique_parts(df)

    test_cases = select_test_cases(parts, n_cases=30)

    all_curves = []

    for _, part in test_cases.iterrows():
        case_id = int(part["optimization_case_id"])
        seed = 42 + case_id * 100

        random_curve = run_random_search(
            model=model,
            part=part,
            n_evaluations=30,
            rng=np.random.default_rng(seed),
        )

        guided_curve = run_surrogate_guided_search(
            model=model,
            part=part,
            n_evaluations=30,
            rng=np.random.default_rng(seed),
        )

        for curve in [random_curve, guided_curve]:
            curve["optimization_case_id"] = case_id
            curve["part_case_id"] = part["part_case_id"]
            curve["scenario_type"] = part["scenario_type"]
            curve["severity_level"] = part["severity_level"]
            curve["disturbance_flag"] = part["disturbance_flag"]

        all_curves.append(random_curve)
        all_curves.append(guided_curve)

    curves = pd.concat(all_curves, ignore_index=True)

    summary, checkpoints = summarize(curves)

    summary_path = output_dir / "optimization_efficiency_summary.csv"
    checkpoints_path = output_dir / "optimization_efficiency_checkpoints.csv"
    curve_path = output_dir / "optimization_efficiency_curve.csv"

    summary.to_csv(summary_path, index=False)
    checkpoints.to_csv(checkpoints_path, index=False)
    curves.to_csv(curve_path, index=False)

    print("")
    print("Optimization efficiency summary:")
    print(summary.to_string(index=False))

    print("")
    print("Optimization efficiency checkpoints:")
    print(checkpoints.to_string(index=False))

    print("")
    print("Curve preview:")
    print(curves.head(30).to_string(index=False))

    print("")
    print("Saved files:")
    print(summary_path)
    print(checkpoints_path)
    print(curve_path)

    print("")
    print("Done.")


if __name__ == "__main__":
    main()