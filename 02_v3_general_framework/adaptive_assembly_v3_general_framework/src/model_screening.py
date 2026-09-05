"""
ML model screening for Version 3.

Purpose:
Compare different surrogate models for predicting assembly quality_score.

Models compared:
- Linear Regression
- Ridge Regression
- Random Forest
- Gradient Boosting
- Support Vector Regression

Metrics:
- MAE
- RMSE
- R2
- training time
- prediction time

This answers the thesis question:
Why was Random Forest selected as the surrogate model?
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


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

TARGET_COLUMN = "quality_score"


def load_dataset(dataset_path: str | Path) -> pd.DataFrame:
    """
    Load the Version 3 batch-aware dataset.
    """

    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. "
            "Run scripts/run_05_generate_batch_dataset.py first."
        )

    df = pd.read_csv(dataset_path)

    missing_columns = [
        col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return df


def build_models(random_seed: int = 42) -> Dict[str, object]:
    """
    Build candidate surrogate models.

    Scaling is used for SVR and Ridge because they are sensitive to feature scale.
    Tree-based models do not require scaling.
    """

    models = {
        "Linear Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Ridge Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=2,
            random_state=random_seed,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_seed,
        ),
        "Support Vector Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVR(
                        kernel="rbf",
                        C=20.0,
                        epsilon=0.05,
                    ),
                ),
            ]
        ),
    }

    return models


def evaluate_single_model(
    model_name: str,
    model: object,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Dict[str, float | str]:
    """
    Train and evaluate one surrogate model.
    """

    start_train = time.perf_counter()
    model.fit(X_train, y_train)
    end_train = time.perf_counter()

    start_predict = time.perf_counter()
    y_pred = model.predict(X_test)
    end_predict = time.perf_counter()

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    return {
        "model": model_name,
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "training_time_seconds": float(end_train - start_train),
        "prediction_time_seconds": float(end_predict - start_predict),
        "test_samples": int(len(y_test)),
    }


def run_repeated_model_screening(
    df: pd.DataFrame,
    random_seeds: List[int] | None = None,
) -> pd.DataFrame:
    """
    Run model screening over multiple train/test splits.

    Repeated splits are used to check whether model performance is stable,
    not only good for one random split.
    """

    if random_seeds is None:
        random_seeds = [11, 22, 33, 42, 55]

    rows = []

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    for seed in random_seeds:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=seed,
        )

        models = build_models(random_seed=seed)

        for model_name, model in models.items():
            result = evaluate_single_model(
                model_name=model_name,
                model=model,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
            )

            result["split_seed"] = seed
            rows.append(result)

    return pd.DataFrame(rows)


def create_model_screening_summary(results: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate repeated-split model results.
    """

    summary = (
        results.groupby("model", as_index=False)
        .agg(
            mean_MAE=("MAE", "mean"),
            std_MAE=("MAE", "std"),
            mean_RMSE=("RMSE", "mean"),
            std_RMSE=("RMSE", "std"),
            mean_R2=("R2", "mean"),
            std_R2=("R2", "std"),
            mean_training_time_seconds=("training_time_seconds", "mean"),
            mean_prediction_time_seconds=("prediction_time_seconds", "mean"),
        )
    )

    # Ranking: lower RMSE is better, higher R2 is better.
    summary["rank_by_RMSE"] = summary["mean_RMSE"].rank(method="dense", ascending=True)
    summary["rank_by_R2"] = summary["mean_R2"].rank(method="dense", ascending=False)

    summary = summary.sort_values(
        ["rank_by_RMSE", "rank_by_R2"],
        ascending=[True, True],
    ).reset_index(drop=True)

    return summary


def save_model_screening_results(
    dataset_path: str | Path,
    output_dir: str | Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run model screening and save detailed and summary tables.
    """

    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(dataset_path)

    results = run_repeated_model_screening(df)
    summary = create_model_screening_summary(results)

    results_path = output_dir / "model_screening_results.csv"
    summary_path = output_dir / "model_screening_summary.csv"

    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)

    return results, summary