from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

from scipy.stats import norm

from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore", category=ConvergenceWarning)


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

QUALITY_COLUMN = "quality_score"


def load_batch_dataset(dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. "
            "Run scripts/run_05_generate_batch_dataset.py first."
        )

    return pd.read_csv(dataset_path)


def train_surrogate_model(df: pd.DataFrame, random_seed: int = 42) -> RandomForestRegressor:
    missing = [c for c in FEATURE_COLUMNS + [QUALITY_COLUMN] if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    X = df[FEATURE_COLUMNS]
    y = df[QUALITY_COLUMN]

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_seed,
    )

    model = RandomForestRegressor(
        n_estimators=250,
        min_samples_leaf=2,
        random_state=random_seed,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    return model


def get_unique_part_cases(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
        "batch_id",
        "part_id",
        "scenario_type",
        "severity_level",
        "disturbance_flag",
        *DEVIATION_COLUMNS,
    ]

    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    unique_parts = (
        df[required_columns]
        .drop_duplicates(subset=["batch_id", "part_id"])
        .reset_index(drop=True)
    )

    unique_parts["part_case_id"] = np.arange(1, len(unique_parts) + 1)

    return unique_parts


def sample_random_parameters(rng: np.random.Generator, n_samples: int) -> pd.DataFrame:
    rows = []

    for _ in range(n_samples):
        rows.append(
            {
                name: float(rng.uniform(low, high))
                for name, (low, high) in PARAMETER_BOUNDS.items()
            }
        )

    return pd.DataFrame(rows)


def build_candidate_feature_matrix(
    part_case: pd.Series,
    parameter_candidates: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, params in parameter_candidates.iterrows():
        row = {}

        for col in DEVIATION_COLUMNS:
            row[col] = part_case[col]

        for col in PARAMETER_COLUMNS:
            row[col] = params[col]

        rows.append(row)

    return pd.DataFrame(rows)


def predict_quality(
    surrogate_model: RandomForestRegressor,
    part_case: pd.Series,
    parameter_candidates: pd.DataFrame,
) -> np.ndarray:
    X_candidates = build_candidate_feature_matrix(part_case, parameter_candidates)
    return surrogate_model.predict(X_candidates[FEATURE_COLUMNS])


def run_random_search(
    surrogate_model: RandomForestRegressor,
    part_case: pd.Series,
    n_evaluations: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    candidates = sample_random_parameters(rng, n_evaluations)
    predictions = predict_quality(surrogate_model, part_case, candidates)

    rows = []
    best_quality = np.inf
    best_params = None

    for i in range(n_evaluations):
        quality = float(predictions[i])

        if quality < best_quality:
            best_quality = quality
            best_params = candidates.iloc[i].to_dict()

        rows.append(
            {
                "method": "random_search",
                "evaluation": i + 1,
                "predicted_quality": quality,
                "best_quality_so_far": best_quality,
                "best_z_adj": best_params["z_adj"],
                "best_theta_adj": best_params["theta_adj"],
                "best_locator_offset": best_params["locator_offset"],
            }
        )

    return pd.DataFrame(rows)


def expected_improvement(
    mu: np.ndarray,
    sigma: np.ndarray,
    best_observed: float,
    xi: float = 0.01,
) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-9)
    improvement = best_observed - mu - xi
    z = improvement / sigma
    return improvement * norm.cdf(z) + sigma * norm.pdf(z)


def run_bayesian_optimization(
    surrogate_model: RandomForestRegressor,
    part_case: pd.Series,
    n_evaluations: int,
    rng: np.random.Generator,
    n_initial: int = 5,
    candidate_pool_size: int = 300,
) -> pd.DataFrame:
    evaluated_params = []
    evaluated_quality = []

    initial_candidates = sample_random_parameters(rng, n_initial)
    initial_predictions = predict_quality(
        surrogate_model,
        part_case,
        initial_candidates,
    )

    for i in range(n_initial):
        evaluated_params.append(initial_candidates.iloc[i][PARAMETER_COLUMNS].to_dict())
        evaluated_quality.append(float(initial_predictions[i]))

    rows = []
    best_quality = np.inf
    best_params = None

    for i, quality in enumerate(evaluated_quality):
        if quality < best_quality:
            best_quality = quality
            best_params = evaluated_params[i]

        rows.append(
            {
                "method": "bayesian_optimization",
                "evaluation": i + 1,
                "predicted_quality": quality,
                "best_quality_so_far": best_quality,
                "best_z_adj": best_params["z_adj"],
                "best_theta_adj": best_params["theta_adj"],
                "best_locator_offset": best_params["locator_offset"],
            }
        )

    for eval_id in range(n_initial + 1, n_evaluations + 1):
        X_observed = pd.DataFrame(evaluated_params)[PARAMETER_COLUMNS].values
        y_observed = np.array(evaluated_quality)

        kernel = (
            ConstantKernel(1.0)
            * Matern(length_scale=1.0, nu=2.5)
            + WhiteKernel(noise_level=1e-5)
        )

        gp_model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "gp",
                    GaussianProcessRegressor(
                        kernel=kernel,
                        normalize_y=True,
                        random_state=int(rng.integers(0, 1_000_000)),
                        n_restarts_optimizer=1,
                    ),
                ),
            ]
        )

        gp_model.fit(X_observed, y_observed)

        candidate_pool = sample_random_parameters(rng, candidate_pool_size)
        X_pool = candidate_pool[PARAMETER_COLUMNS].values

        mu, sigma = gp_model.predict(X_pool, return_std=True)

        acquisition = expected_improvement(
            mu=mu,
            sigma=sigma,
            best_observed=float(np.min(y_observed)),
        )

        next_index = int(np.argmax(acquisition))
        next_params = candidate_pool.iloc[next_index][PARAMETER_COLUMNS].to_dict()

        next_prediction = predict_quality(
            surrogate_model,
            part_case,
            pd.DataFrame([next_params]),
        )[0]

        evaluated_params.append(next_params)
        evaluated_quality.append(float(next_prediction))

        if float(next_prediction) < best_quality:
            best_quality = float(next_prediction)
            best_params = next_params

        rows.append(
            {
                "method": "bayesian_optimization",
                "evaluation": eval_id,
                "predicted_quality": float(next_prediction),
                "best_quality_so_far": best_quality,
                "best_z_adj": best_params["z_adj"],
                "best_theta_adj": best_params["theta_adj"],
                "best_locator_offset": best_params["locator_offset"],
            }
        )

    return pd.DataFrame(rows)


def select_test_cases(
    unique_parts: pd.DataFrame,
    n_cases: int = 30,
    random_seed: int = 42,
) -> pd.DataFrame:
    scenario_types = sorted(unique_parts["scenario_type"].dropna().unique())
    cases_per_scenario = max(1, n_cases // len(scenario_types))

    selected_parts = []

    for scenario in scenario_types:
        scenario_df = unique_parts[unique_parts["scenario_type"] == scenario]
        n_select = min(cases_per_scenario, len(scenario_df))

        selected_parts.append(
            scenario_df.sample(
                n=n_select,
                random_state=random_seed,
            )
        )

    selected = pd.concat(selected_parts, ignore_index=True)

    if len(selected) < n_cases:
        selected_keys = set(zip(selected["batch_id"], selected["part_id"]))

        remaining = unique_parts[
            ~unique_parts.apply(
                lambda row: (row["batch_id"], row["part_id"]) in selected_keys,
                axis=1,
            )
        ]

        additional = remaining.sample(
            n=min(n_cases - len(selected), len(remaining)),
            random_state=random_seed,
        )

        selected = pd.concat([selected, additional], ignore_index=True)

    if len(selected) > n_cases:
        selected = selected.sample(
            n=n_cases,
            random_state=random_seed,
        )

    selected = selected.reset_index(drop=True)
    selected["optimization_case_id"] = np.arange(1, len(selected) + 1)

    return selected


def create_efficiency_summary(curve_df: pd.DataFrame) -> pd.DataFrame:
    final_rows = (
        curve_df.sort_values("evaluation")
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

    random_quality = summary.loc[
        summary["method"] == "random_search",
        "final_mean_best_quality",
    ].iloc[0]

    summary["quality_improvement_vs_random_percent"] = (
        (random_quality - summary["final_mean_best_quality"])
        / random_quality
        * 100.0
    )

    return summary


def create_checkpoint_summary(curve_df: pd.DataFrame) -> pd.DataFrame:
    checkpoints = [5, 10, 15, 20, 25, 30]
    rows = []

    for checkpoint in checkpoints:
        checkpoint_data = curve_df[curve_df["evaluation"] == checkpoint]

        grouped = (
            checkpoint_data.groupby("method", as_index=False)
            .agg(
                mean_best_quality=("best_quality_so_far", "mean"),
                median_best_quality=("best_quality_so_far", "median"),
                std_best_quality=("best_quality_so_far", "std"),
                cases=("optimization_case_id", "nunique"),
            )
        )

        for _, row in grouped.iterrows():
            rows.append(
                {
                    "evaluation_budget": checkpoint,
                    "method": row["method"],
                    "mean_best_quality": row["mean_best_quality"],
                    "median_best_quality": row["median_best_quality"],
                    "std_best_quality": row["std_best_quality"],
                    "cases": row["cases"],
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    dataset_path = PROJECT_ROOT / "results" / "tables" / "batch_aware_dataset.csv"
    output_dir = PROJECT_ROOT / "results" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_cases = 30
    n_evaluations = 30
    random_seed = 42

    print("Running Version 3 optimization efficiency comparison...")
    print(f"Input dataset: {dataset_path}")
    print("This may take a few minutes.")

    df = load_batch_dataset(dataset_path)
    surrogate_model = train_surrogate_model(df, random_seed=random_seed)
    unique_parts = get_unique_part_cases(df)

    test_cases = select_test_cases(
        unique_parts=unique_parts,
        n_cases=n_cases,
        random_seed=random_seed,
    )

    all_curves = []

    for _, part_case in test_cases.iterrows():
        case_id = int(part_case["optimization_case_id"])
        case_seed = random_seed + case_id * 100

        rng_random = np.random.default_rng(case_seed)
        rng_bo = np.random.default_rng(case_seed)

        random_curve = run_random_search(
            surrogate_model=surrogate_model,
            part_case=part_case,
            n_evaluations=n_evaluations,
            rng=rng_random,
        )

        bo_curve = run_bayesian_optimization(
            surrogate_model=surrogate_model,
            part_case=part_case,
            n_evaluations=n_evaluations,
            rng=rng_bo,
        )

        for curve in [random_curve, bo_curve]:
            curve["optimization_case_id"] = case_id
            curve["part_case_id"] = part_case["part_case_id"]
            curve["scenario_type"] = part_case["scenario_type"]
            curve["severity_level"] = part_case["severity_level"]
            curve["disturbance_flag"] = part_case["disturbance_flag"]

        all_curves.append(random_curve)
        all_curves.append(bo_curve)

    curve_df = pd.concat(all_curves, ignore_index=True)

    summary = create_efficiency_summary(curve_df)
    checkpoint_summary = create_checkpoint_summary(curve_df)

    summary_path = output_dir / "optimization_efficiency_summary.csv"
    checkpoint_path = output_dir / "optimization_efficiency_checkpoints.csv"
    curve_path = output_dir / "optimization_efficiency_curve.csv"

    summary.to_csv(summary_path, index=False)
    checkpoint_summary.to_csv(checkpoint_path, index=False)
    curve_df.to_csv(curve_path, index=False)

    print("")
    print("Optimization efficiency summary:")
    print(summary.to_string(index=False))

    print("")
    print("Optimization efficiency checkpoints:")
    print(checkpoint_summary.to_string(index=False))

    print("")
    print("Curve data preview:")
    print(curve_df.head(30).to_string(index=False))

    print("")
    print("Saved files:")
    print(summary_path)
    print(checkpoint_path)
    print(curve_path)

    print("")
    print("Done.")


if __name__ == "__main__":
    main()