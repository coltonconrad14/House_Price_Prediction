from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from .config import Settings
from .data import load_dataset, make_train_test_split, split_features_target
from .features import build_preprocessor


def train_and_save_model(settings: Settings) -> dict[str, float]:
    df = load_dataset(settings.raw_data_path)
    x, y = split_features_target(df, settings.target_column)
    x_train, x_test, y_train, y_test = make_train_test_split(
        x, y, test_size=settings.test_size, random_state=settings.random_state
    )

    preprocessor = build_preprocessor(x_train)
    regressor = RandomForestRegressor(
        n_estimators=300, random_state=settings.random_state)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
    }

    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, settings.model_path)
    return metrics


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Train first using scripts/train.py."
        )
    return joblib.load(model_path)
