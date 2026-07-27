from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = None
    XGBRegressor = None

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import FALLBACK_RAW_DATA_PATH, RAW_DATA_PATH, REPORTS_DIR, ensure_directories
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.preprocessing import Preprocessor
from src.split import chronological_split
from seven_day_pipeline import build_horizon_targets, merge_horizon_targets, select_feature_columns


RANDOM_STATE = 42
N_SPLITS = 5
OUT_DIR = REPORTS_DIR / "complexity_runtime"
TABLES_DIR = OUT_DIR / "tables"
FIGURES_DIR = OUT_DIR / "figures"


def search_space_size(param_distributions) -> int | None:
    if any(not isinstance(values, (list, tuple)) for values in param_distributions.values()):
        return None
    return len(list(ParameterGrid(param_distributions)))


def effective_n_iter(param_distributions, requested_iter: int) -> int:
    size = search_space_size(param_distributions)
    return requested_iter if size is None else min(requested_iter, size)


def theoretical_fit_count(model_name: str, requested_iter: int, param_distributions=None) -> int:
    if model_name == "LinearRegression":
        return 1
    n_iter = effective_n_iter(param_distributions, requested_iter)
    return n_iter * N_SPLITS + 1


def train_temperature_model(name: str, X_train, y_train, requested_iter: int):
    if name == "LinearRegression":
        model = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
        params = None
    elif name == "KNNRegressor":
        model = Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor())])
        params = {"model__n_neighbors": randint(3, 26), "model__weights": ["uniform", "distance"]}
    elif name == "XGBoostRegressor" and XGBRegressor is not None:
        model = XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbosity=0,
        )
        params = {
            "n_estimators": randint(100, 501),
            "max_depth": randint(3, 11),
            "learning_rate": uniform(0.01, 0.29),
        }
    else:
        raise ValueError(f"Unsupported model: {name}")

    if params is not None:
        model = RandomizedSearchCV(
            model,
            params,
            n_iter=effective_n_iter(params, requested_iter),
            cv=TimeSeriesSplit(n_splits=N_SPLITS),
            scoring="neg_root_mean_squared_error",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
    start = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    return elapsed, params, getattr(model, "best_params_", {})


def train_rain_model(name: str, X_train, y_train, requested_iter: int):
    y_train = y_train.astype(int)
    if name == "LogisticRegression":
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        params = {"model__C": [0.01, 0.1, 1.0, 10.0, 100.0]}
    elif name == "SVM":
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVC(class_weight="balanced", probability=True, random_state=RANDOM_STATE)),
            ]
        )
        params = {"model__C": [0.1, 1.0, 10.0, 100.0], "model__kernel": ["linear", "rbf"]}
    elif name == "XGBoostClassifier" and XGBClassifier is not None:
        neg = max(1, int((y_train == 0).sum()))
        pos = max(1, int((y_train == 1).sum()))
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=neg / pos,
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbosity=0,
        )
        params = {
            "n_estimators": randint(100, 501),
            "max_depth": randint(3, 11),
            "learning_rate": uniform(0.01, 0.29),
        }
    else:
        raise ValueError(f"Unsupported model: {name}")

    model = RandomizedSearchCV(
        model,
        params,
        n_iter=effective_n_iter(params, requested_iter),
        cv=TimeSeriesSplit(n_splits=N_SPLITS),
        scoring="f1",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    start = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    return elapsed, params, getattr(model, "best_params_", {})


def measure_runtime() -> pd.DataFrame:
    ensure_directories()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = RAW_DATA_PATH if RAW_DATA_PATH.exists() else FALLBACK_RAW_DATA_PATH
    raw_df = DataLoader(csv_path).load()
    noon_df = Preprocessor().preprocess(raw_df, save=False)
    feature_df = FeatureEngineer().transform(noon_df)
    feature_cols = select_feature_columns(feature_df)

    rows = []
    for requested_iter in [4, 20]:
        for horizon in range(1, 8):
            horizon_df = build_horizon_targets(noon_df, horizon)
            horizon_feature_df = merge_horizon_targets(feature_df, horizon_df)
            train_df, _, _ = chronological_split(horizon_feature_df)
            X_train = train_df[feature_cols]
            y_temp_train = train_df["horizon_temperature"]
            y_rain_train = train_df["horizon_rain"].astype(int)

            for model_name in ["LinearRegression", "KNNRegressor", "XGBoostRegressor"]:
                if model_name == "XGBoostRegressor" and XGBRegressor is None:
                    continue
                elapsed, params, best_params = train_temperature_model(
                    model_name,
                    X_train,
                    y_temp_train,
                    requested_iter,
                )
                rows.append(
                    {
                        "n_iter": requested_iter,
                        "horizon_day": horizon,
                        "task": "temperature_regression",
                        "model": model_name,
                        "training_time_seconds": elapsed,
                        "effective_n_iter": 0 if params is None else effective_n_iter(params, requested_iter),
                        "time_series_splits": N_SPLITS if params is not None else 0,
                        "estimated_fit_count": theoretical_fit_count(model_name, requested_iter, params),
                        "best_params": best_params,
                    }
                )

            for model_name in ["LogisticRegression", "SVM", "XGBoostClassifier"]:
                if model_name == "XGBoostClassifier" and XGBClassifier is None:
                    continue
                elapsed, params, best_params = train_rain_model(
                    model_name,
                    X_train,
                    y_rain_train,
                    requested_iter,
                )
                rows.append(
                    {
                        "n_iter": requested_iter,
                        "horizon_day": horizon,
                        "task": "rain_classification",
                        "model": model_name,
                        "training_time_seconds": elapsed,
                        "effective_n_iter": effective_n_iter(params, requested_iter),
                        "time_series_splits": N_SPLITS,
                        "estimated_fit_count": theoretical_fit_count(model_name, requested_iter, params),
                        "best_params": best_params,
                    }
                )

    runtime_df = pd.DataFrame(rows)
    runtime_df["best_params"] = runtime_df["best_params"].astype(str)
    runtime_df.to_csv(TABLES_DIR / "seven_day_training_runtime_by_model.csv", index=False)

    summary = (
        runtime_df.groupby(["n_iter", "task", "model"], as_index=False)
        .agg(
            total_training_time_seconds=("training_time_seconds", "sum"),
            mean_training_time_seconds=("training_time_seconds", "mean"),
            total_estimated_fit_count=("estimated_fit_count", "sum"),
        )
        .sort_values(["task", "model", "n_iter"])
    )
    summary.to_csv(TABLES_DIR / "seven_day_training_runtime_summary.csv", index=False)

    version_summary = (
        runtime_df.groupby(["n_iter"], as_index=False)
        .agg(
            total_training_time_seconds=("training_time_seconds", "sum"),
            total_estimated_fit_count=("estimated_fit_count", "sum"),
        )
        .sort_values("n_iter")
    )
    version_summary.to_csv(TABLES_DIR / "seven_day_training_runtime_version_summary.csv", index=False)
    return runtime_df


def plot_runtime(runtime_df: pd.DataFrame) -> None:
    summary = (
        runtime_df.groupby(["n_iter", "task", "model"], as_index=False)
        .agg(total_training_time_seconds=("training_time_seconds", "sum"))
    )
    for task, title, filename in [
        (
            "temperature_regression",
            "7-day temperature model training time by n_iter",
            "temperature_training_time_n_iter_comparison.png",
        ),
        (
            "rain_classification",
            "7-day rain model training time by n_iter",
            "rain_training_time_n_iter_comparison.png",
        ),
    ]:
        plt.figure(figsize=(10, 5))
        task_df = summary[summary["task"] == task]
        for model in task_df["model"].unique():
            model_df = task_df[task_df["model"] == model].sort_values("n_iter")
            plt.plot(
                model_df["n_iter"],
                model_df["total_training_time_seconds"],
                marker="o",
                label=model,
            )
        plt.title(title)
        plt.xlabel("RandomizedSearchCV n_iter")
        plt.ylabel("Total training time across 7 horizons (seconds)")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / filename, dpi=160)
        plt.close()

    version_summary = (
        runtime_df.groupby(["n_iter"], as_index=False)
        .agg(total_training_time_seconds=("training_time_seconds", "sum"))
        .sort_values("n_iter")
    )
    plt.figure(figsize=(7, 5))
    plt.plot(
        version_summary["n_iter"],
        version_summary["total_training_time_seconds"],
        marker="o",
    )
    plt.title("Total 7-day pipeline training time: n_iter=4 vs n_iter=20")
    plt.xlabel("RandomizedSearchCV n_iter")
    plt.ylabel("Total measured model training time (seconds)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "total_training_time_n_iter_comparison.png", dpi=160)
    plt.close()

    fit_summary = (
        runtime_df.groupby(["n_iter"], as_index=False)
        .agg(total_estimated_fit_count=("estimated_fit_count", "sum"))
        .sort_values("n_iter")
    )
    plt.figure(figsize=(7, 5))
    plt.plot(
        fit_summary["n_iter"],
        fit_summary["total_estimated_fit_count"],
        marker="o",
        color="tab:purple",
    )
    plt.title("Estimated model fit count: n_iter=4 vs n_iter=20")
    plt.xlabel("RandomizedSearchCV n_iter")
    plt.ylabel("Estimated fit count")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "estimated_fit_count_n_iter_comparison.png", dpi=160)
    plt.close()


def main() -> None:
    runtime_df = measure_runtime()
    plot_runtime(runtime_df)
    print("Runtime and complexity measurement completed.")
    print(runtime_df.groupby("n_iter")["training_time_seconds"].sum())


if __name__ == "__main__":
    main()
