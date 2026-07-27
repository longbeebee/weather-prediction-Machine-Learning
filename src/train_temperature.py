import time

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import N_SPLITS, RANDOM_STATE, SEARCH_ITER

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - used only when dependency is unavailable.
    XGBRegressor = None


def train_temperature_models(X_train, y_train, X_test, y_test) -> tuple[pd.DataFrame, dict]:
    models = {
        "LinearRegression": Pipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression())]
        ),
        "KNNRegressor": Pipeline(
            [("scaler", StandardScaler()), ("model", KNeighborsRegressor())]
        ),
    }
    searches = {
        "KNNRegressor": {
            "model__n_neighbors": randint(3, 26),
            "model__weights": ["uniform", "distance"],
        }
    }
    if XGBRegressor is not None:
        models["XGBoostRegressor"] = XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbosity=0,
        )
        searches["XGBoostRegressor"] = {
            "n_estimators": randint(100, 501),
            "max_depth": randint(3, 11),
            "learning_rate": uniform(0.01, 0.29),
        }

    cv = TimeSeriesSplit(n_splits=min(N_SPLITS, max(2, len(X_train) // 120)))
    rows = []
    fitted = {}
    for name, estimator in models.items():
        start = time.perf_counter()
        if name in searches:
            estimator = RandomizedSearchCV(
                estimator,
                searches[name],
                n_iter=SEARCH_ITER,
                cv=cv,
                scoring="neg_root_mean_squared_error",
                random_state=RANDOM_STATE,
                n_jobs=1,
            )
        estimator.fit(X_train, y_train)
        training_time = time.perf_counter() - start
        y_pred = estimator.predict(X_test)
        rows.append(
            {
                "model": name,
                "mae": mean_absolute_error(y_test, y_pred),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "r2": r2_score(y_test, y_pred),
                "training_time": training_time,
            }
        )
        fitted[name] = {"model": estimator, "predictions": y_pred}
    return pd.DataFrame(rows), fitted
