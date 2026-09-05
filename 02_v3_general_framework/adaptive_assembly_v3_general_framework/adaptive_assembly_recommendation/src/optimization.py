import numpy as np
import pandas as pd

from skopt import gp_minimize
from skopt.space import Real

from src.config import DECISION_RANGES
from src.surrogate_models import FEATURE_COLUMNS


def create_feature_row(
    deviation_case: dict,
    z_adj: float,
    theta_adj: float,
    locator_offset: float
) -> pd.DataFrame:
    """
    Create one input row for the trained surrogate model.

    The deviation case stays fixed.
    The assembly decision variables are changed by the optimizer.
    """
    row = {
        "dev_offset": deviation_case["dev_offset"],
        "dev_tilt": deviation_case["dev_tilt"],
        "dev_bend": deviation_case["dev_bend"],
        "dev_waviness": deviation_case["dev_waviness"],
        "dev_noise": deviation_case["dev_noise"],
        "z_adj": z_adj,
        "theta_adj": theta_adj,
        "locator_offset": locator_offset,
    }

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict_quality(
    model,
    deviation_case: dict,
    z_adj: float,
    theta_adj: float,
    locator_offset: float
) -> float:
    """
    Predict quality score using the trained ML surrogate model.
    """
    X_candidate = create_feature_row(
        deviation_case=deviation_case,
        z_adj=z_adj,
        theta_adj=theta_adj,
        locator_offset=locator_offset,
    )

    prediction = model.predict(X_candidate)[0]
    return float(prediction)


def fixed_setting_prediction(model, deviation_case: dict) -> dict:
    """
    Evaluate the fixed baseline setting.

    This represents a non-adaptive assembly setup:
    no vertical correction,
    no angular correction,
    no local locator correction.
    """
    score = predict_quality(
        model=model,
        deviation_case=deviation_case,
        z_adj=0.0,
        theta_adj=0.0,
        locator_offset=0.0,
    )

    return {
        "method": "fixed_setting",
        "z_adj": 0.0,
        "theta_adj": 0.0,
        "locator_offset": 0.0,
        "predicted_quality_score": score,
    }


def random_search_optimization(
    model,
    deviation_case: dict,
    n_trials: int = 80,
    random_state: int = 42
) -> dict:
    """
    Simple random-search optimization baseline.

    It randomly tests assembly parameter combinations and keeps the best one.
    """
    rng = np.random.default_rng(random_state)

    best_score = np.inf
    best_params = None
    history = []

    for i in range(n_trials):
        z_adj = rng.uniform(*DECISION_RANGES["z_adj"])
        theta_adj = rng.uniform(*DECISION_RANGES["theta_adj"])
        locator_offset = rng.uniform(*DECISION_RANGES["locator_offset"])

        score = predict_quality(
            model=model,
            deviation_case=deviation_case,
            z_adj=z_adj,
            theta_adj=theta_adj,
            locator_offset=locator_offset,
        )

        if score < best_score:
            best_score = score
            best_params = (z_adj, theta_adj, locator_offset)

        history.append({
            "iteration": i + 1,
            "current_score": score,
            "best_score": best_score,
            "method": "random_search",
        })

    return {
        "method": "random_search",
        "z_adj": best_params[0],
        "theta_adj": best_params[1],
        "locator_offset": best_params[2],
        "predicted_quality_score": best_score,
        "history": pd.DataFrame(history),
    }


def bayesian_optimization(
    model,
    deviation_case: dict,
    n_calls: int = 40,
    random_state: int = 42
) -> dict:
    """
    Bayesian optimization recommendation.

    The optimizer searches for the assembly setting that minimizes
    the predicted quality score.
    """
    search_space = [
        Real(*DECISION_RANGES["z_adj"], name="z_adj"),
        Real(*DECISION_RANGES["theta_adj"], name="theta_adj"),
        Real(*DECISION_RANGES["locator_offset"], name="locator_offset"),
    ]

    def objective(params):
        z_adj, theta_adj, locator_offset = params

        score = predict_quality(
            model=model,
            deviation_case=deviation_case,
            z_adj=z_adj,
            theta_adj=theta_adj,
            locator_offset=locator_offset,
        )

        return score

    result = gp_minimize(
        func=objective,
        dimensions=search_space,
        n_calls=n_calls,
        n_initial_points=10,
        random_state=random_state,
        acq_func="EI",
    )

    history = []
    best_score_so_far = np.inf

    for i, score in enumerate(result.func_vals):
        best_score_so_far = min(best_score_so_far, float(score))

        history.append({
            "iteration": i + 1,
            "current_score": float(score),
            "best_score": best_score_so_far,
            "method": "bayesian_optimization",
        })

    best_z_adj, best_theta_adj, best_locator_offset = result.x

    return {
        "method": "bayesian_optimization",
        "z_adj": best_z_adj,
        "theta_adj": best_theta_adj,
        "locator_offset": best_locator_offset,
        "predicted_quality_score": float(result.fun),
        "history": pd.DataFrame(history),
    }


def compare_recommendation_methods(
    model,
    deviation_case: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare fixed setting, random search, and Bayesian optimization.
    """
    fixed = fixed_setting_prediction(model, deviation_case)
    random_result = random_search_optimization(model, deviation_case)
    bayes_result = bayesian_optimization(model, deviation_case)

    fixed_score = fixed["predicted_quality_score"]

    rows = []

    for result in [fixed, random_result, bayes_result]:
        if fixed_score != 0:
            improvement_percent = (
                (fixed_score - result["predicted_quality_score"])
                / fixed_score
                * 100.0
            )
        else:
            improvement_percent = 0.0

        rows.append({
            "method": result["method"],
            "z_adj": result["z_adj"],
            "theta_adj": result["theta_adj"],
            "locator_offset": result["locator_offset"],
            "predicted_quality_score": result["predicted_quality_score"],
            "improvement_percent_vs_fixed": improvement_percent,
        })

    comparison_df = pd.DataFrame(rows)

    convergence_df = pd.concat(
        [
            random_result["history"],
            bayes_result["history"],
        ],
        ignore_index=True,
    )

    return comparison_df, convergence_df