import joblib
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FEATURE_COLUMNS = [
    "dev_offset",
    "dev_tilt",
    "dev_bend",
    "dev_waviness",
    "dev_noise",
    "z_adj",
    "theta_adj",
    "locator_offset",
]

TARGET_COLUMN = "quality_score"


def split_features_target(df: pd.DataFrame):
    """
    Split dataframe into input features X and target output y.
    """
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y


def train_models(df: pd.DataFrame, random_state: int = 42):
    """
    Train baseline and machine-learning surrogate models.

    Linear Regression:
        simple baseline model

    Random Forest:
        main nonlinear surrogate model
    """
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=random_state,
    )

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    results = []
    trained_models = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2 = r2_score(y_test, y_pred)

        results.append({
            "model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        })

        trained_models[model_name] = {
            "model": model,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
        }

    metrics_df = pd.DataFrame(results)

    return trained_models, metrics_df


def save_model(model, path: str):
    """
    Save trained model.
    """
    joblib.dump(model, path)


def load_model(path: str):
    """
    Load trained model.
    """
    return joblib.load(path)