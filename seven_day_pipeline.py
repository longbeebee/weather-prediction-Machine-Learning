import numpy as np
import os
import pandas as pd
import json
from matplotlib import pyplot as plt
from scipy.stats import randint, uniform
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = None
    XGBRegressor = None

from src.config import FALLBACK_RAW_DATA_PATH, RAW_DATA_PATH, REPORTS_DIR, TABLES_DIR, ensure_directories
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.feature_engineering import FeatureEngineer
from src.split import chronological_split
from src.utils import weather_level


SEVEN_DAY_DIR = REPORTS_DIR / os.environ.get("SEVEN_DAY_OUTPUT_DIR", "seven_day_forecast")
SEVEN_DAY_TABLES = SEVEN_DAY_DIR / "tables"
SEVEN_DAY_FIGURES = SEVEN_DAY_DIR / "figures"
SEARCH_ITER = int(os.environ.get("SEVEN_DAY_SEARCH_ITER", "4"))
RANDOM_STATE = 42


def main() -> None:
    ensure_directories()
    SEVEN_DAY_TABLES.mkdir(parents=True, exist_ok=True)
    SEVEN_DAY_FIGURES.mkdir(parents=True, exist_ok=True)

    csv_path = RAW_DATA_PATH if RAW_DATA_PATH.exists() else FALLBACK_RAW_DATA_PATH
    raw_df = DataLoader(csv_path).load()
    noon_df = Preprocessor().preprocess(raw_df, save=False)
    base_feature_df = FeatureEngineer().transform(noon_df)
    feature_cols = select_feature_columns(base_feature_df)

    train_df, val_df, test_df = chronological_split(base_feature_df)
    horizon_metrics = []
    selected_rows = []
    forecast_rows = []
    test_prediction_rows = []
    threshold_curve_rows = []
    hyperparameter_tuning_rows = []

    for horizon in range(1, 8):
        horizon_df = build_horizon_targets(noon_df, horizon)
        horizon_feature_df = merge_horizon_targets(base_feature_df, horizon_df)
        train_h, val_h, test_h = chronological_split(horizon_feature_df)

        X_train = train_h[feature_cols]
        y_temp_train = train_h["horizon_temperature"]
        y_rain_train = train_h["horizon_rain"].astype(int)
        X_val = val_h[feature_cols]
        X_test = test_h[feature_cols]

        temp_models, temp_tuning_rows = train_temperature_candidates(horizon, X_train, y_temp_train)
        rain_models, rain_tuning_rows = train_rain_candidates(horizon, X_train, y_rain_train)
        hyperparameter_tuning_rows.extend(temp_tuning_rows + rain_tuning_rows)
        temp_val_metrics = evaluate_temperature_candidates(horizon, "validation", val_h, X_val, temp_models)
        temp_test_metrics = evaluate_temperature_candidates(horizon, "test", test_h, X_test, temp_models)
        rain_thresholds = tune_rain_thresholds(horizon, val_h, X_val, rain_models)
        for threshold_info in rain_thresholds.values():
            threshold_curve_rows.extend(threshold_info["curve"])
        rain_val_metrics = evaluate_rain_candidates(horizon, "validation", val_h, X_val, rain_models, rain_thresholds)
        rain_test_metrics = evaluate_rain_candidates(horizon, "test", test_h, X_test, rain_models, rain_thresholds)
        horizon_metrics.extend(temp_val_metrics + temp_test_metrics + rain_val_metrics + rain_test_metrics)

        temp_val_df = pd.DataFrame(temp_val_metrics)
        rain_val_df = pd.DataFrame(rain_val_metrics)
        best_temp_model = temp_val_df.sort_values("rmse").iloc[0]["model"]
        best_rain_model = rain_val_df.sort_values("f1", ascending=False).iloc[0]["model"]
        best_temp_test_pred = temp_models[best_temp_model].predict(X_test)
        best_rain_test_prob = rain_probabilities(rain_models[best_rain_model], X_test)
        best_rain_threshold = rain_thresholds[best_rain_model]["threshold"]
        best_rain_test_pred = (best_rain_test_prob >= best_rain_threshold).astype(int)
        test_prediction_rows.extend(
            build_test_prediction_rows(
                horizon,
                test_h,
                best_temp_model,
                best_rain_model,
                best_temp_test_pred,
                best_rain_test_pred,
            )
        )
        selected_rows.append(
            {
                "horizon_day": horizon,
                "best_temperature_model": best_temp_model,
                "validation_rmse": temp_val_df[temp_val_df["model"] == best_temp_model]["rmse"].iloc[0],
                "test_rmse": pd.DataFrame(temp_test_metrics)[
                    pd.DataFrame(temp_test_metrics)["model"] == best_temp_model
                ]["rmse"].iloc[0],
                "best_rain_model": best_rain_model,
                "validation_f1": rain_val_df[rain_val_df["model"] == best_rain_model]["f1"].iloc[0],
                "test_f1": pd.DataFrame(rain_test_metrics)[
                    pd.DataFrame(rain_test_metrics)["model"] == best_rain_model
                ]["f1"].iloc[0],
                "rain_threshold": best_rain_threshold,
            }
        )

        latest_features = base_feature_df.sort_values("time").iloc[[-1]][feature_cols]
        pred_temp = float(temp_models[best_temp_model].predict(latest_features)[0])
        pred_rain_prob = float(rain_probabilities(rain_models[best_rain_model], latest_features)[0])
        pred_rain = int(pred_rain_prob >= best_rain_threshold)
        pred_date = base_feature_df["time"].max() + pd.Timedelta(days=horizon)
        forecast_rows.append(
            {
                "horizon_day": horizon,
                "forecast_time": pred_date,
                "temperature_model": best_temp_model,
                "rain_model": best_rain_model,
                "predicted_temperature": pred_temp,
                "predicted_rain": pred_rain,
                "predicted_rain_probability": pred_rain_prob,
                "rain_threshold": best_rain_threshold,
                "predicted_rain_label": "rain" if pred_rain == 1 else "no_rain",
                "predicted_weather_level": weather_level(pred_temp),
            }
        )

    metrics_df = pd.DataFrame(horizon_metrics)
    selected_df = pd.DataFrame(selected_rows)
    forecast_df = pd.DataFrame(forecast_rows)
    test_predictions_df = pd.DataFrame(test_prediction_rows)
    threshold_df = pd.DataFrame(threshold_curve_rows)
    tuning_df = pd.DataFrame(hyperparameter_tuning_rows)
    best_hyperparameters_df = build_best_hyperparameters_table(tuning_df, selected_df)
    metrics_df.to_csv(SEVEN_DAY_TABLES / "seven_day_horizon_metrics.csv", index=False)
    selected_df.to_csv(SEVEN_DAY_TABLES / "seven_day_selected_models.csv", index=False)
    forecast_df.to_csv(SEVEN_DAY_TABLES / "seven_day_forecast_predictions.csv", index=False)
    test_predictions_df.to_csv(SEVEN_DAY_TABLES / "seven_day_test_predictions.csv", index=False)
    threshold_df.to_csv(SEVEN_DAY_TABLES / "seven_day_threshold_tuning.csv", index=False)
    tuning_df.to_csv(SEVEN_DAY_TABLES / "seven_day_hyperparameter_tuning.csv", index=False)
    best_hyperparameters_df.to_csv(SEVEN_DAY_TABLES / "seven_day_best_hyperparameters.csv", index=False)

    plot_metrics(metrics_df)
    plot_model_comparisons(metrics_df)
    plot_test_actual_vs_predicted(test_predictions_df)
    plot_threshold_tuning(threshold_df, selected_df)
    plot_hyperparameter_tuning(tuning_df, selected_df)
    plot_forecast(forecast_df)
    write_report(raw_df, noon_df, base_feature_df, metrics_df, selected_df, forecast_df, best_hyperparameters_df)
    print("7-day forecast pipeline completed successfully.")


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    blocked = {
        "time",
        "target_temperature",
        "target_precipitation",
        "target_rain",
        "target_weather_level",
    }
    return [c for c in df.select_dtypes(include="number").columns if c not in blocked]


def build_horizon_targets(noon_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = noon_df[["time", "temperature", "precipitation"]].copy().sort_values("time")
    df["horizon_temperature"] = df["temperature"].shift(-horizon)
    df["horizon_precipitation"] = df["precipitation"].shift(-horizon)
    df["horizon_rain"] = (df["horizon_precipitation"] > 0).astype(float)
    return df[["time", "horizon_temperature", "horizon_precipitation", "horizon_rain"]]


def merge_horizon_targets(base_feature_df: pd.DataFrame, horizon_df: pd.DataFrame) -> pd.DataFrame:
    df = base_feature_df.drop(
        columns=["target_temperature", "target_precipitation", "target_rain", "target_weather_level"],
        errors="ignore",
    ).merge(horizon_df, on="time", how="inner")
    df = df.dropna(subset=["horizon_temperature", "horizon_precipitation", "horizon_rain"])
    return df.reset_index(drop=True)


def train_temperature_candidates(horizon, X_train, y_train) -> tuple[dict, list[dict]]:
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

    cv = TimeSeriesSplit(n_splits=5)
    fitted = {}
    tuning_rows = []
    for name, model in models.items():
        if name in searches:
            model = RandomizedSearchCV(
                model,
                searches[name],
                n_iter=SEARCH_ITER,
                cv=cv,
                scoring="neg_root_mean_squared_error",
                random_state=RANDOM_STATE,
                n_jobs=1,
            )
        model.fit(X_train, y_train)
        fitted[name] = model
        tuning_rows.extend(extract_tuning_rows(horizon, "temperature_regression", name, model))
    return fitted, tuning_rows


def train_rain_candidates(horizon, X_train, y_train) -> tuple[dict, list[dict]]:
    y_train = y_train.astype(int)
    models = {
        "LogisticRegression": Pipeline(
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
        ),
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(class_weight="balanced", probability=True, random_state=RANDOM_STATE),
                ),
            ]
        ),
    }
    searches = {
        "LogisticRegression": {"model__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
        "SVM": {"model__C": [0.1, 1.0, 10.0, 100.0], "model__kernel": ["linear", "rbf"]},
    }
    if XGBClassifier is not None:
        neg = max(1, int((y_train == 0).sum()))
        pos = max(1, int((y_train == 1).sum()))
        models["XGBoostClassifier"] = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=neg / pos,
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbosity=0,
        )
        searches["XGBoostClassifier"] = {
            "n_estimators": randint(100, 501),
            "max_depth": randint(3, 11),
            "learning_rate": uniform(0.01, 0.29),
        }
    cv = TimeSeriesSplit(n_splits=5)
    fitted = {}
    tuning_rows = []
    for name, model in models.items():
        n_iter = min(SEARCH_ITER, search_space_size(searches[name]))
        model = RandomizedSearchCV(
            model,
            searches[name],
            n_iter=n_iter,
            cv=cv,
            scoring="f1",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        model.fit(X_train, y_train)
        fitted[name] = model
        tuning_rows.extend(extract_tuning_rows(horizon, "rain_classification", name, model))
    return fitted, tuning_rows


def search_space_size(param_distributions) -> int:
    for values in param_distributions.values():
        if not isinstance(values, (list, tuple)):
            return SEARCH_ITER
    return max(1, len(list(ParameterGrid(param_distributions))))


def extract_tuning_rows(horizon: int, task: str, model_name: str, fitted_model) -> list[dict]:
    if not hasattr(fitted_model, "cv_results_"):
        return [
            {
                "horizon_day": horizon,
                "task": task,
                "model": model_name,
                "candidate_index": 1,
                "rank_test_score": 1,
                "mean_test_score": np.nan,
                "std_test_score": np.nan,
                "mean_cv_rmse": np.nan,
                "mean_cv_f1": np.nan,
                "is_best": True,
                "best_params": describe_best_params(fitted_model),
                "params": describe_best_params(fitted_model),
                "search_iter_budget": SEARCH_ITER,
            }
        ]

    results = fitted_model.cv_results_
    best_params = describe_best_params(fitted_model)
    rows = []
    for idx, params in enumerate(results["params"]):
        score = float(results["mean_test_score"][idx])
        row = {
            "horizon_day": horizon,
            "task": task,
            "model": model_name,
            "candidate_index": idx + 1,
            "rank_test_score": int(results["rank_test_score"][idx]),
            "mean_test_score": score,
            "std_test_score": float(results["std_test_score"][idx]),
            "mean_cv_rmse": -score if task == "temperature_regression" else np.nan,
            "mean_cv_f1": score if task == "rain_classification" else np.nan,
            "is_best": int(results["rank_test_score"][idx]) == 1,
            "best_params": best_params,
            "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
            "search_iter_budget": SEARCH_ITER,
        }
        rows.append(row)
    return rows


def describe_best_params(model) -> str:
    if hasattr(model, "best_params_"):
        return json.dumps(model.best_params_, ensure_ascii=False, sort_keys=True)
    if isinstance(model, Pipeline):
        return "StandardScaler + LinearRegression"
    return model.__class__.__name__


def build_best_hyperparameters_table(tuning_df: pd.DataFrame, selected_df: pd.DataFrame) -> pd.DataFrame:
    if tuning_df.empty:
        return pd.DataFrame()
    best = tuning_df[tuning_df["is_best"] == True].copy()
    best = best[
        [
            "horizon_day",
            "task",
            "model",
            "rank_test_score",
            "mean_cv_rmse",
            "mean_cv_f1",
            "best_params",
            "search_iter_budget",
        ]
    ].drop_duplicates()

    selected_records = []
    for _, row in selected_df.iterrows():
        selected_records.append(
            {
                "horizon_day": row["horizon_day"],
                "task": "temperature_regression",
                "selected_model": row["best_temperature_model"],
                "selected_metric": "validation_rmse",
                "selected_metric_value": row["validation_rmse"],
            }
        )
        selected_records.append(
            {
                "horizon_day": row["horizon_day"],
                "task": "rain_classification",
                "selected_model": row["best_rain_model"],
                "selected_metric": "validation_f1",
                "selected_metric_value": row["validation_f1"],
            }
        )
    selected = pd.DataFrame(selected_records)
    return best.merge(
        selected,
        left_on=["horizon_day", "task", "model"],
        right_on=["horizon_day", "task", "selected_model"],
        how="left",
    )


def evaluate_temperature_candidates(horizon, split, df, X, models) -> list[dict]:
    rows = []
    for name, model in models.items():
        pred = model.predict(X)
        rows.append(evaluate_temperature(horizon, split, name, df, pred, model))
    return rows


def tune_rain_thresholds(horizon, val_df, X_val, models) -> dict:
    y_true = val_df["horizon_rain"].astype(int).to_numpy()
    thresholds = np.round(np.arange(0.10, 0.91, 0.05), 2)
    tuned = {}
    for model_name, model in models.items():
        probs = rain_probabilities(model, X_val)
        rows = []
        best = {"threshold": 0.5, "f1": -1.0}
        for threshold in thresholds:
            pred = (probs >= threshold).astype(int)
            row = {
                "horizon_day": horizon,
                "model": model_name,
                "threshold": threshold,
                "accuracy": accuracy_score(y_true, pred),
                "precision": precision_score(y_true, pred, zero_division=0),
                "recall": recall_score(y_true, pred, zero_division=0),
                "f1": f1_score(y_true, pred, zero_division=0),
            }
            rows.append(row)
            if row["f1"] > best["f1"]:
                best = {"threshold": threshold, "f1": row["f1"]}
        tuned[model_name] = {"threshold": best["threshold"], "validation_f1": best["f1"], "curve": rows}
    return tuned


def rain_probabilities(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1 / (1 + np.exp(-scores))
    return model.predict(X)


def evaluate_rain_candidates(horizon, split, df, X, models, thresholds) -> list[dict]:
    rows = []
    for name, model in models.items():
        probs = rain_probabilities(model, X)
        threshold = thresholds[name]["threshold"]
        pred = (probs >= threshold).astype(int)
        rows.append(evaluate_rain(horizon, split, name, df, pred, threshold, model))
    return rows


def evaluate_temperature(horizon, split, model_name, df, temp_pred, model) -> dict:
    y_temp = df["horizon_temperature"]
    abs_error = np.abs(y_temp - temp_pred)
    return {
        "horizon_day": horizon,
        "split": split,
        "task": "temperature_regression",
        "model": model_name,
        "best_params": describe_best_params(model),
        "search_space": model_params(model_name),
        "mae": mean_absolute_error(y_temp, temp_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_temp, temp_pred))),
        "r2": r2_score(y_temp, temp_pred),
        "within_1c": float((abs_error <= 1).mean()),
        "within_2c": float((abs_error <= 2).mean()),
        "within_3c": float((abs_error <= 3).mean()),
    }


def evaluate_rain(horizon, split, model_name, df, rain_pred, threshold, model) -> dict:
    y_rain = df["horizon_rain"].astype(int)
    return {
        "horizon_day": horizon,
        "split": split,
        "task": "rain_classification",
        "model": model_name,
        "best_params": describe_best_params(model),
        "search_space": model_params(model_name),
        "threshold": threshold,
        "accuracy": accuracy_score(y_rain, rain_pred),
        "precision": precision_score(y_rain, rain_pred, zero_division=0),
        "recall": recall_score(y_rain, rain_pred, zero_division=0),
        "f1": f1_score(y_rain, rain_pred, zero_division=0),
    }


def build_test_prediction_rows(
    horizon: int,
    test_df: pd.DataFrame,
    temp_model_name: str,
    rain_model_name: str,
    temp_pred,
    rain_pred,
) -> list[dict]:
    rows = []
    for i, (_, row) in enumerate(test_df.reset_index(drop=True).iterrows()):
        actual_temp = row["horizon_temperature"]
        predicted_temp = float(temp_pred[i])
        actual_rain = int(row["horizon_rain"])
        predicted_rain = int(rain_pred[i])
        rows.append(
            {
                "horizon_day": horizon,
                "time": row["time"],
                "temperature_model": temp_model_name,
                "rain_model": rain_model_name,
                "actual_temperature": actual_temp,
                "predicted_temperature": predicted_temp,
                "temperature_residual": actual_temp - predicted_temp,
                "actual_rain": actual_rain,
                "predicted_rain": predicted_rain,
                "rain_correct": actual_rain == predicted_rain,
            }
        )
    return rows


def model_params(model_name: str) -> str:
    if model_name == "LinearRegression":
        return "StandardScaler + LinearRegression"
    if model_name == "KNNRegressor":
        return "RandomizedSearchCV: n_neighbors 3-25, weights uniform/distance"
    if model_name == "XGBoostRegressor":
        return "RandomizedSearchCV: n_estimators 100-500, max_depth 3-10, learning_rate 0.01-0.3"
    if model_name == "LogisticRegression":
        return "RandomizedSearchCV: C in [0.01, 0.1, 1, 10, 100], class_weight balanced"
    if model_name == "SVM":
        return "RandomizedSearchCV: C in [0.1, 1, 10, 100], kernel linear/rbf, class_weight balanced"
    if model_name == "XGBoostClassifier":
        return "RandomizedSearchCV: n_estimators 100-500, max_depth 3-10, learning_rate 0.01-0.3, scale_pos_weight"
    return ""


def plot_metrics(metrics_df: pd.DataFrame) -> None:
    temp = metrics_df[(metrics_df["task"] == "temperature_regression") & (metrics_df["split"] == "test")]
    rain = metrics_df[(metrics_df["task"] == "rain_classification") & (metrics_df["split"] == "test")]

    plt.figure(figsize=(8, 5))
    for model in temp["model"].unique():
        model_df = temp[temp["model"] == model]
        plt.plot(model_df["horizon_day"], model_df["rmse"], marker="o", label=model)
    plt.title("7-day temperature RMSE by horizon and model")
    plt.xlabel("Forecast horizon day")
    plt.ylabel("RMSE (degC)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(SEVEN_DAY_FIGURES / "seven_day_temperature_error_by_horizon.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    for model in rain["model"].unique():
        model_df = rain[rain["model"] == model]
        plt.plot(model_df["horizon_day"], model_df["f1"], marker="o", label=model)
    plt.title("7-day rain F1 by horizon and model")
    plt.xlabel("Forecast horizon day")
    plt.ylabel("F1")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(SEVEN_DAY_FIGURES / "seven_day_rain_performance_by_horizon.png", dpi=160)
    plt.close()


def plot_model_comparisons(metrics_df: pd.DataFrame) -> None:
    test = metrics_df[metrics_df["split"] == "test"]
    temp = test[test["task"] == "temperature_regression"]
    rain = test[test["task"] == "rain_classification"]

    plt.figure(figsize=(11, 5))
    temp_pivot = temp.pivot(index="horizon_day", columns="model", values="rmse")
    temp_pivot.plot(kind="line", marker="o", ax=plt.gca())
    plt.title("Test RMSE comparison for 7-day temperature models")
    plt.xlabel("Forecast horizon day")
    plt.ylabel("RMSE (degC)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(SEVEN_DAY_FIGURES / "seven_day_temperature_model_comparison.png", dpi=160)
    plt.close()

    plt.figure(figsize=(11, 5))
    rain_pivot = rain.pivot(index="horizon_day", columns="model", values="f1")
    rain_pivot.plot(kind="line", marker="o", ax=plt.gca())
    plt.title("Test F1 comparison for 7-day rain models")
    plt.xlabel("Forecast horizon day")
    plt.ylabel("F1")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(SEVEN_DAY_FIGURES / "seven_day_rain_model_comparison.png", dpi=160)
    plt.close()


def plot_hyperparameter_tuning(tuning_df: pd.DataFrame, selected_df: pd.DataFrame) -> None:
    if tuning_df.empty:
        return

    tuned = tuning_df.dropna(subset=["rank_test_score"]).copy()
    if tuned.empty:
        return

    for (task, model), df in tuned.groupby(["task", "model"]):
        metric_col = "mean_cv_rmse" if task == "temperature_regression" else "mean_cv_f1"
        ylabel = "Mean CV RMSE (lower is better)" if task == "temperature_regression" else "Mean CV F1 (higher is better)"
        selected_model_col = "best_temperature_model" if task == "temperature_regression" else "best_rain_model"

        fig, axes = plt.subplots(7, 1, figsize=(13, 24), sharex=False)
        for ax, horizon in zip(axes, range(1, 8)):
            h_df = df[df["horizon_day"] == horizon].sort_values("candidate_index")
            if h_df.empty:
                ax.set_visible(False)
                continue
            ax.plot(h_df["candidate_index"], h_df[metric_col], marker="o", linewidth=1.2)
            best = h_df.sort_values("rank_test_score").iloc[0]
            ax.scatter([best["candidate_index"]], [best[metric_col]], color="red", zorder=3)
            selected = selected_df[selected_df["horizon_day"] == horizon][selected_model_col].iloc[0]
            is_selected = "selected" if selected == model else "not selected"
            best_params = shorten_params(str(best["best_params"]))
            ax.set_title(
                f"Horizon {horizon}: {model} tuning ({is_selected}); best params: {best_params}",
                fontsize=9,
            )
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
        axes[-1].set_xlabel("RandomizedSearchCV candidate index")
        plt.tight_layout()
        filename = f"hyperparameter_tuning_{task}_{model}.png"
        plt.savefig(SEVEN_DAY_FIGURES / filename, dpi=160)
        plt.close()

    selected_tuned = tuned[tuned["is_best"] == True].copy()
    for task, df in selected_tuned.groupby("task"):
        metric_col = "mean_cv_rmse" if task == "temperature_regression" else "mean_cv_f1"
        ylabel = "Best mean CV RMSE" if task == "temperature_regression" else "Best mean CV F1"
        plt.figure(figsize=(11, 5))
        for model in df["model"].unique():
            model_df = df[df["model"] == model].sort_values("horizon_day")
            plt.plot(model_df["horizon_day"], model_df[metric_col], marker="o", label=model)
        title = "Best tuned CV score by horizon for temperature models" if task == "temperature_regression" else "Best tuned CV score by horizon for rain models"
        plt.title(title)
        plt.xlabel("Forecast horizon day")
        plt.ylabel(ylabel)
        if task == "rain_classification":
            plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(SEVEN_DAY_FIGURES / f"hyperparameter_tuning_best_scores_{task}.png", dpi=160)
        plt.close()


def shorten_params(params: str, max_len: int = 130) -> str:
    return params if len(params) <= max_len else params[: max_len - 3] + "..."


def plot_test_actual_vs_predicted(test_predictions_df: pd.DataFrame) -> None:
    horizons = sorted(test_predictions_df["horizon_day"].unique())
    fig, axes = plt.subplots(7, 1, figsize=(13, 22), sharex=False)
    for ax, horizon in zip(axes, horizons):
        df = test_predictions_df[test_predictions_df["horizon_day"] == horizon].copy()
        df["time"] = pd.to_datetime(df["time"])
        ax.plot(df["time"], df["actual_temperature"], label="Actual", linewidth=1.2)
        ax.plot(df["time"], df["predicted_temperature"], label="Predicted", linewidth=1.2)
        model_name = df["temperature_model"].iloc[0]
        ax.set_title(f"Horizon {horizon}: actual vs predicted temperature on test set ({model_name})")
        ax.set_ylabel("degC")
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Test date")
    plt.tight_layout()
    plt.savefig(SEVEN_DAY_FIGURES / "test_actual_vs_predicted_temperature_best_models_by_horizon.png", dpi=160)
    plt.close()

    plt.figure(figsize=(13, 6))
    h1 = test_predictions_df[test_predictions_df["horizon_day"] == 1].copy()
    h1["time"] = pd.to_datetime(h1["time"])
    plt.plot(h1["time"], h1["actual_temperature"], label="Actual", linewidth=1.3)
    plt.plot(h1["time"], h1["predicted_temperature"], label="Predicted", linewidth=1.3)
    plt.title("Horizon 1 actual vs predicted temperature on test set")
    plt.xlabel("Test date")
    plt.ylabel("Temperature (degC)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(SEVEN_DAY_FIGURES / "test_actual_vs_predicted_temperature_horizon_1.png", dpi=160)
    plt.close()

    fig, axes = plt.subplots(7, 1, figsize=(13, 22), sharex=False)
    for ax, horizon in zip(axes, horizons):
        df = test_predictions_df[test_predictions_df["horizon_day"] == horizon].copy()
        df["time"] = pd.to_datetime(df["time"])
        ax.step(df["time"], df["actual_rain"], label="Actual rain", linewidth=1.2, where="mid")
        ax.step(df["time"], df["predicted_rain"], label="Predicted rain", linewidth=1.2, where="mid")
        model_name = df["rain_model"].iloc[0]
        ax.set_title(f"Horizon {horizon}: actual vs predicted rain on test set ({model_name})")
        ax.set_ylabel("Rain label")
        ax.set_ylim(-0.1, 1.1)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Test date")
    plt.tight_layout()
    plt.savefig(SEVEN_DAY_FIGURES / "test_actual_vs_predicted_rain_best_models_by_horizon.png", dpi=160)
    plt.close()

    rows = []
    for horizon in horizons:
        df = test_predictions_df[test_predictions_df["horizon_day"] == horizon]
        rows.append(
            {
                "horizon_day": horizon,
                "correct_rate": df["rain_correct"].mean(),
                "wrong_rate": 1 - df["rain_correct"].mean(),
            }
        )
    rate_df = pd.DataFrame(rows).melt(
        id_vars="horizon_day",
        value_vars=["correct_rate", "wrong_rate"],
        var_name="result",
        value_name="rate",
    )
    plt.figure(figsize=(10, 5))
    for result in ["correct_rate", "wrong_rate"]:
        part = rate_df[rate_df["result"] == result]
        plt.plot(part["horizon_day"], part["rate"], marker="o", label=result)
    plt.title("Rain prediction correct/wrong rate on test set by horizon")
    plt.xlabel("Forecast horizon day")
    plt.ylabel("Rate")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(SEVEN_DAY_FIGURES / "test_rain_correct_wrong_rate_by_horizon.png", dpi=160)
    plt.close()


def plot_threshold_tuning(threshold_df: pd.DataFrame, selected_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(7, 1, figsize=(11, 22), sharex=True)
    for ax, (_, selected) in zip(axes, selected_df.iterrows()):
        horizon = selected["horizon_day"]
        model = selected["best_rain_model"]
        threshold = selected["rain_threshold"]
        df = threshold_df[
            (threshold_df["horizon_day"] == horizon)
            & (threshold_df["model"] == model)
        ]
        ax.plot(df["threshold"], df["f1"], marker="o", label=f"{model} validation F1")
        ax.axvline(threshold, color="red", linestyle="--", label=f"best threshold={threshold:.2f}")
        ax.set_title(f"Horizon {horizon}: threshold tuning for selected rain model")
        ax.set_ylabel("F1")
        ax.set_ylim(0, 1)
        ax.legend(loc="lower right")
    axes[-1].set_xlabel("Decision threshold")
    plt.tight_layout()
    plt.savefig(SEVEN_DAY_FIGURES / "rain_threshold_tuning_by_horizon.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(selected_df["horizon_day"], selected_df["rain_threshold"], marker="o")
    plt.title("Selected rain decision threshold by horizon")
    plt.xlabel("Forecast horizon day")
    plt.ylabel("Selected threshold")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(SEVEN_DAY_FIGURES / "selected_rain_threshold_by_horizon.png", dpi=160)
    plt.close()

def plot_forecast(forecast_df: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(forecast_df["forecast_time"], forecast_df["predicted_temperature"], marker="o")
    plt.title("Predicted noon temperature for the next 7 days")
    plt.xlabel("Forecast date")
    plt.ylabel("Predicted temperature (degC)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(SEVEN_DAY_FIGURES / "next_7_days_temperature_forecast.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(forecast_df["horizon_day"], forecast_df["predicted_rain"])
    plt.title("Predicted rain label for the next 7 days")
    plt.xlabel("Forecast horizon day")
    plt.ylabel("Rain prediction (1=rain, 0=no rain)")
    plt.ylim(0, 1.2)
    plt.tight_layout()
    plt.savefig(SEVEN_DAY_FIGURES / "next_7_days_rain_forecast.png", dpi=160)
    plt.close()


def write_report(raw_df, noon_df, feature_df, metrics_df, selected_df, forecast_df, best_hyperparameters_df) -> None:
    validation_temp = metrics_df[(metrics_df["task"] == "temperature_regression") & (metrics_df["split"] == "validation")]
    validation_rain = metrics_df[(metrics_df["task"] == "rain_classification") & (metrics_df["split"] == "validation")]
    test_temp = metrics_df[(metrics_df["task"] == "temperature_regression") & (metrics_df["split"] == "test")]
    test_rain = metrics_df[(metrics_df["task"] == "rain_classification") & (metrics_df["split"] == "test")]
    stats = noon_df.select_dtypes(include="number").describe().round(3).reset_index()
    content = f"""# Hanoi 7-Day Noon Weather Forecasting Report

## Introduction
This report presents an extended version of the original Hanoi noon weather prediction system. Instead of predicting only the next day, this version predicts noon temperature and rain for the next 7 days. The problem is formulated as a multi-horizon supervised learning task.

## Motivation
Short-term weather forecasting is useful for planning travel, outdoor work, school activities, and daily scheduling. A 7-day forecast is more practical than a single next-day prediction because users often need to plan several days ahead. However, forecasting difficulty increases with horizon length because future atmospheric conditions become less certain.

## Contribution
This version contributes a separate 7-day forecasting pipeline with the following components:

- Multi-horizon target generation for day 1 to day 7.
- Temperature regression model for each horizon.
- Rain classification model for each horizon.
- Chronological train/validation/test evaluation.
- Horizon-wise metrics and figures.
- Final 7-day forecast output from the latest available noon observation.

## Project Objective
The objective is to predict Hanoi weather at 12:00 PM for each of the next 7 days.

The two machine learning tasks are:

- Temperature regression: predict noon temperature at horizon `h`, where `h = 1..7`.
- Rain classification: predict whether noon precipitation at horizon `h` is greater than 0.

Weather level is not trained as a separate model. It is derived from predicted temperature:

```text
temperature < 22      -> cool
22 <= temperature <=30 -> normal
temperature > 30      -> hot
```

## Dataset
- Raw hourly records: {len(raw_df):,}
- Noon records: {len(noon_df):,}
- Engineered supervised samples before horizon-specific target filtering: {len(feature_df):,}
- Latest available noon record used for next-7-day inference: {feature_df['time'].max()}

The data contains historical Open-Meteo observations from 2020 to 2025. Only records at 12:00 are used to keep the forecasting target consistent with the project objective.

## Dataset Statistics

{to_markdown(stats)}

The average noon temperature is about 26.9°C, while precipitation has a median of 0.0 mm. This indicates that rainfall is sparse at noon, making rain classification more difficult than temperature regression.

## Input and Expected Output
Input:

```text
Historical 12:00 weather records with engineered lag, rolling, and time features.
```

Outputs:

```text
For each horizon h = 1..7:
- predicted noon temperature
- predicted rain / no rain label
- derived weather level
```

## Data Preprocessing
The preprocessing logic follows the original project:

1. Detect and remove Open-Meteo metadata rows.
2. Normalize weather column names.
3. Parse the `time` column as datetime.
4. Sort observations chronologically.
5. Remove duplicated timestamps.
6. Keep only records where `hour == 12`.
7. Remove unused or leakage-prone columns.
8. Fill numeric missing values with medians if needed.

## Feature Engineering
The 7-day pipeline reuses the same feature engineering design as the original system:

- Time features: `month`, `day_of_year`, `season`.
- Lag features for temperature, humidity, and precipitation at lags 1, 3, and 7.
- Rolling features using shifted rolling windows.

Rolling features are computed with `shift(1)` before `rolling()`. This is important because the model must not use information from the day being predicted.

## Methodology
For each horizon `h = 1..7`, the target is generated as:

```text
horizon_temperature = temperature.shift(-h)
horizon_rain = precipitation.shift(-h) > 0
```

This means that horizon 1 predicts tomorrow, horizon 2 predicts two days later, and so on until horizon 7.

The same input feature set is used for each horizon, but each horizon has its own target and its own trained model pair. This direct multi-horizon approach avoids recursive error accumulation.

## Train / Validation / Test Protocol
The data is split chronologically:

| Split | Period | Role |
|---|---|---|
| Training | 2020-2023 | Fit model parameters |
| Validation | 2024 | Check model behavior and compare horizons |
| Test | 2025 | Final evaluation |

The test set is not used to train or select models. This preserves a realistic evaluation on unseen future data.

## Model Training Process
This version trains and compares multiple candidate models for each horizon:

Temperature regression candidates:

- Linear Regression.
- KNN Regressor.
- XGBoost Regressor.

Rain classification candidates:

- Logistic Regression with balanced class weights.
- SVM with balanced class weights.
- XGBoost Classifier with automatic `scale_pos_weight`.

For each horizon, the temperature model uses a pipeline:

```text
StandardScaler -> LinearRegression / KNN
XGBoostRegressor without scaling
```

Linear Regression is fitted directly after scaling. KNN and XGBoost are tuned with `RandomizedSearchCV` and `TimeSeriesSplit`.

For rain classification, the model uses:

```text
StandardScaler -> LogisticRegression / SVM
XGBoostClassifier without scaling
```

The best temperature model for each horizon is selected by validation RMSE. The best rain model for each horizon is selected by validation F1. Test metrics are reported only after model selection.

Hyperparameter search configuration:

```text
RandomizedSearchCV n_iter = {SEARCH_ITER}
Cross-validation = TimeSeriesSplit(n_splits=5)
Regression scoring = negative RMSE
Classification scoring = F1
```

Hyperparameter tuning is performed only on the training period through time-series cross-validation. The validation period is then used to select the final model per horizon. This separation is methodologically important: cross-validation estimates which configuration fits the training history best, while validation evaluates whether that fitted model generalizes to the next chronological year.

## Best Hyperparameters from RandomizedSearchCV

{to_markdown(best_hyperparameters_df)}

The table above reports the best configuration found inside the search space for each horizon and model. Linear Regression has no randomized hyperparameter search because it is fitted by ordinary least squares after scaling. KNN tuning mainly controls the number of neighbors and the weighting rule. XGBoost tuning controls tree complexity and learning dynamics through `n_estimators`, `max_depth`, and `learning_rate`. Logistic Regression and SVM tune regularization strength through `C`; SVM additionally tunes the kernel. For rain classification, the model hyperparameters are selected before the decision threshold is tuned on the validation set.

## Selected Models by Horizon

{to_markdown(selected_df)}

## Validation Performance by Horizon

### Temperature Model Comparison on Validation Set

{to_markdown(validation_temp[['horizon_day', 'model', 'mae', 'rmse', 'r2', 'within_1c', 'within_2c', 'within_3c']])}

### Rain Model Comparison on Validation Set

{to_markdown(validation_rain[['horizon_day', 'model', 'accuracy', 'precision', 'recall', 'f1']])}

## Test Performance by Horizon

### Temperature Model Comparison on Test Set

{to_markdown(test_temp[['horizon_day', 'model', 'mae', 'rmse', 'r2', 'within_1c', 'within_2c', 'within_3c']])}

Temperature error increases as the horizon becomes longer. This is expected because the model predicts further into the future with the same historical input information.

### Rain Model Comparison on Test Set

{to_markdown(test_rain[['horizon_day', 'model', 'accuracy', 'precision', 'recall', 'f1']])}

Rain prediction performance is less stable than temperature prediction. Rainfall is sparse and less continuous, so F1 fluctuates across horizons.

## Next 7-Day Forecast Output

{to_markdown(forecast_df)}

The forecast starts from the latest available noon observation in the dataset. All predicted rain labels are `no_rain`, while predicted temperature gradually moves from normal to cool conditions across the first several days.

## Figures

- ![Temperature error by horizon](figures/seven_day_temperature_error_by_horizon.png)
- ![Rain performance by horizon](figures/seven_day_rain_performance_by_horizon.png)
- ![Temperature model comparison](figures/seven_day_temperature_model_comparison.png)
- ![Rain model comparison](figures/seven_day_rain_model_comparison.png)
- ![Temperature hyperparameter tuning best scores](figures/hyperparameter_tuning_best_scores_temperature_regression.png)
- ![Rain hyperparameter tuning best scores](figures/hyperparameter_tuning_best_scores_rain_classification.png)
- ![KNN hyperparameter tuning](figures/hyperparameter_tuning_temperature_regression_KNNRegressor.png)
- ![XGBoost Regressor hyperparameter tuning](figures/hyperparameter_tuning_temperature_regression_XGBoostRegressor.png)
- ![Logistic Regression hyperparameter tuning](figures/hyperparameter_tuning_rain_classification_LogisticRegression.png)
- ![SVM hyperparameter tuning](figures/hyperparameter_tuning_rain_classification_SVM.png)
- ![XGBoost Classifier hyperparameter tuning](figures/hyperparameter_tuning_rain_classification_XGBoostClassifier.png)
- ![Test actual vs predicted temperature by horizon](figures/test_actual_vs_predicted_temperature_best_models_by_horizon.png)
- ![Test actual vs predicted temperature horizon 1](figures/test_actual_vs_predicted_temperature_horizon_1.png)
- ![Test actual vs predicted rain by horizon](figures/test_actual_vs_predicted_rain_best_models_by_horizon.png)
- ![Test rain correct wrong rate by horizon](figures/test_rain_correct_wrong_rate_by_horizon.png)
- ![Rain threshold tuning by horizon](figures/rain_threshold_tuning_by_horizon.png)
- ![Selected rain threshold by horizon](figures/selected_rain_threshold_by_horizon.png)
- ![Next 7 days temperature forecast](figures/next_7_days_temperature_forecast.png)
- ![Next 7 days rain forecast](figures/next_7_days_rain_forecast.png)

## Data Visualization Interpretation
The temperature error plot compares RMSE across Linear Regression, KNN Regressor, and XGBoost Regressor for horizons 1 to 7. The best model is the one with the lowest validation RMSE for each horizon.

The rain performance plot compares F1 across Logistic Regression, SVM, and XGBoost Classifier. F1 is more informative than accuracy because the rain label is imbalanced. A decline or fluctuation in F1 indicates that the model has difficulty consistently identifying rainy days several days ahead.

The next-7-day temperature forecast plot visualizes the predicted cooling pattern from 2025-12-31 to 2026-01-06. The rain forecast plot indicates that the model predicts no rain for all seven future noon timestamps.

The test actual-vs-predicted temperature plot compares the best selected model's predicted values with true test values across the full 2025 test period. Each subplot corresponds to one forecast horizon. When the predicted curve follows the actual curve closely, the model captures the temporal temperature pattern well. Wider gaps indicate larger forecast errors.

The test actual-vs-predicted rain plot compares the binary rain labels across the full test period. Because rain is represented as 0/1, the plot should be interpreted as event matching rather than continuous magnitude matching. The correct/wrong rate plot summarizes how often the selected rain model matches the actual test label for each horizon.

The threshold tuning plots show how validation F1 changes as the rain decision threshold varies. Instead of using the default 0.5 threshold, the pipeline selects the threshold that maximizes validation F1 for each selected rain model and horizon. Lower thresholds usually increase recall but may reduce precision; higher thresholds usually improve precision but may miss more rain events.

The hyperparameter tuning plots show the cross-validation score obtained by each sampled configuration. The red marker identifies the configuration selected by `RandomizedSearchCV`. In regression plots, lower RMSE is better. In classification plots, higher F1 is better. Comparing these plots across models helps distinguish two stages of the experiment: first, finding the best configuration within each candidate model; second, comparing the best configured models against each other on the validation set.

## Evaluation and Discussion
The selected-model table shows which model is best at each horizon according to validation metrics. This is important because the model that performs best for day 1 does not necessarily remain best for day 7.

For temperature, model quality is judged by RMSE. Lower RMSE means lower average prediction error with stronger penalty for large errors. For rain, model quality is judged by F1, which balances precision and recall for the rainy class.

The 7-day output should therefore be interpreted as an academic multi-horizon forecasting result, not as a production-grade weather forecast. A real deployment would require future forecast covariates from numerical weather prediction models.

## Discussion
Temperature forecasting remains relatively stable across the first several horizons because temperature has strong temporal continuity and seasonal structure. However, error generally increases as the horizon becomes longer, which is expected in multi-step forecasting.

Rain forecasting is harder than temperature forecasting because rainfall is sparse and less continuous. F1 can fluctuate across horizons, indicating that rain/no-rain separation depends strongly on short-term atmospheric conditions that are not fully captured by historical noon observations alone.

## Limitations
The 7-day forecast uses historical observed features available at the latest noon record. For future days, the pipeline predicts horizon-specific targets directly rather than using real future meteorological forecast variables. A production-grade 7-day weather forecast should incorporate numerical weather prediction inputs such as future humidity, pressure, wind, cloud cover, and radiation forecasts.

Additional limitations:

- The system uses one location only.
- It does not ingest future meteorological forecasts.
- The same feature snapshot is used to infer all seven horizons.
- Rain prediction is sensitive to class imbalance and year-to-year variation.
- The models are intentionally simple for interpretability.

## Conclusion
This 7-day version provides an interpretable multi-horizon forecasting baseline for Hanoi noon temperature and rain prediction. It is suitable for academic comparison and reporting, but future work should add real forecast covariates and probabilistic rain calibration.

## Future Work
Future work should include:

- Numerical weather prediction features for future days.
- More locations across Hanoi and northern Vietnam.
- Probabilistic rain forecasts instead of only binary labels.
- Deep learning or time-series models for multi-step forecasting.
- Calibration and uncertainty intervals for temperature prediction.
"""
    (SEVEN_DAY_DIR / "final_results_7day.md").write_text(content, encoding="utf-8")


def to_markdown(df: pd.DataFrame) -> str:
    rounded = df.copy()
    for col in rounded.select_dtypes(include="number").columns:
        rounded[col] = rounded[col].round(3)
    cols = [str(c) for c in rounded.columns]
    rows = ["| " + " | ".join(cols) + " |"]
    rows.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in rounded.iterrows():
        rows.append("| " + " | ".join(str(row[c]) for c in rounded.columns) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    main()
