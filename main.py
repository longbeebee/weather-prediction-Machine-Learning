import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from src.baseline import rain_baseline, regression_baseline
from src.config import (
    FALLBACK_RAW_DATA_PATH,
    RAW_DATA_PATH,
    TABLES_DIR,
    TARGET_COLUMNS,
    ensure_directories,
)
from src.data_loader import DataLoader
from src.evaluate import save_results
from src.explainability import generate_model_explanations
from src.feature_engineering import FeatureEngineer
from src.preprocessing import Preprocessor
from src.reporting import generate_report
from src.split import chronological_split
from src.train_rain import train_rain_models
from src.train_temperature import train_temperature_models
from src.visualization import (
    save_feature_importance,
    visualize_eda,
    visualize_results,
    visualize_validation_test_predictions,
)


def main() -> None:
    ensure_directories()
    csv_path = RAW_DATA_PATH if RAW_DATA_PATH.exists() else FALLBACK_RAW_DATA_PATH

    raw_df = DataLoader(csv_path).load()
    noon_df = Preprocessor().preprocess(raw_df)
    feature_df = FeatureEngineer().transform(noon_df)
    train_df, val_df, test_df = chronological_split(feature_df)

    experiment_outputs = []
    primary = None
    for include_shortwave, label in [(False, "A_without_shortwave_radiation"), (True, "B_with_shortwave_radiation")]:
        feature_cols = select_feature_columns(feature_df, include_shortwave)
        X_train = train_df[feature_cols]
        y_temp_train = train_df["target_temperature"]
        y_rain_train = train_df["target_rain"]
        X_val = val_df[feature_cols]
        y_temp_val = val_df["target_temperature"]
        y_rain_val = val_df["target_rain"]
        X_test = test_df[feature_cols]
        y_temp_test = test_df["target_temperature"]
        y_rain_test = test_df["target_rain"]

        temp_val_results, fitted_temp = train_temperature_models(X_train, y_temp_train, X_val, y_temp_val)
        rain_val_results, fitted_rain = train_rain_models(X_train, y_rain_train, X_val, y_rain_val)
        temp_test_results = evaluate_temperature_models(fitted_temp, X_test, y_temp_test, temp_val_results)
        rain_test_results = evaluate_rain_models(fitted_rain, X_test, y_rain_test, rain_val_results)
        baseline_results = build_baseline_results(train_df, val_df, test_df)

        experiment_outputs.append(
            {
                "experiment": label,
                "include_shortwave_radiation": include_shortwave,
                "best_regression_model": temp_val_results.sort_values("rmse").iloc[0]["model"],
                "best_validation_regression_rmse": temp_val_results["rmse"].min(),
                "test_rmse_for_validation_best": temp_test_results[
                    temp_test_results["model"] == temp_val_results.sort_values("rmse").iloc[0]["model"]
                ]["rmse"].iloc[0],
                "best_rain_model": rain_val_results.sort_values("f1", ascending=False).iloc[0]["model"],
                "best_validation_rain_f1": rain_val_results["f1"].max(),
                "test_f1_for_validation_best": rain_test_results[
                    rain_test_results["model"] == rain_val_results.sort_values("f1", ascending=False).iloc[0]["model"]
                ]["f1"].iloc[0],
            }
        )

        if include_shortwave:
            primary = {
                "feature_cols": feature_cols,
                "X_val": X_val,
                "X_test": X_test,
                "y_temp_val": y_temp_val,
                "y_temp_test": y_temp_test,
                "y_rain_val": y_rain_val,
                "y_rain_test": y_rain_test,
                "temp_val_results": temp_val_results,
                "rain_val_results": rain_val_results,
                "temp_test_results": temp_test_results,
                "rain_test_results": rain_test_results,
                "baseline_results": baseline_results,
                "fitted_temp": fitted_temp,
                "fitted_rain": fitted_rain,
            }

    experiment_results = pd.DataFrame(experiment_outputs)
    assert primary is not None

    best_temp, best_rain = save_results(
        primary["temp_val_results"],
        primary["rain_val_results"],
        primary["temp_test_results"],
        primary["rain_test_results"],
        primary["baseline_results"],
        experiment_results,
        primary["fitted_temp"],
        primary["fitted_rain"],
    )
    best_temp_val_pred = primary["fitted_temp"][best_temp]["predictions"]
    best_rain_val_pred = primary["fitted_rain"][best_rain]["predictions"]
    best_temp_test_pred = primary["fitted_temp"][best_temp]["model"].predict(primary["X_test"])
    best_rain_test_pred = primary["fitted_rain"][best_rain]["model"].predict(primary["X_test"])
    importance_model_name = (
        "XGBoostRegressor"
        if "XGBoostRegressor" in primary["fitted_temp"]
        else best_temp
    )
    feature_importance = save_feature_importance(
        primary["fitted_temp"][importance_model_name]["model"],
        primary["feature_cols"],
    )
    generate_model_explanations(
        primary["fitted_temp"],
        primary["fitted_rain"],
        primary["X_test"],
        primary["y_temp_test"],
        primary["y_rain_test"],
        primary["feature_cols"],
    )
    prediction_tables = save_prediction_analysis(
        val_df,
        test_df,
        primary["y_temp_val"],
        best_temp_val_pred,
        primary["y_temp_test"],
        best_temp_test_pred,
        primary["y_rain_val"],
        best_rain_val_pred,
        primary["y_rain_test"],
        best_rain_test_pred,
    )
    save_worst_prediction_days(test_df, best_temp_test_pred)
    save_split_summary(train_df, val_df, test_df)

    visualize_eda(noon_df, feature_df, primary["feature_cols"])
    visualize_results(
        primary["temp_test_results"],
        primary["rain_test_results"],
        experiment_results,
        primary["y_temp_test"],
        best_temp_test_pred,
        primary["y_rain_test"],
        best_rain_test_pred,
        feature_importance,
    )
    visualize_validation_test_predictions(
        prediction_tables["temperature_validation"],
        prediction_tables["temperature_test"],
        prediction_tables["rain_validation"],
        prediction_tables["rain_test"],
    )
    generate_report(raw_df, noon_df, feature_df, best_temp, best_rain)
    print("Pipeline completed successfully.")


def select_feature_columns(df: pd.DataFrame, include_shortwave: bool) -> list[str]:
    blocked = set(TARGET_COLUMNS + ["time", "target_precipitation"])
    if not include_shortwave:
        blocked.add("shortwave_radiation")
    numeric = df.select_dtypes(include="number").columns
    return [c for c in numeric if c not in blocked]


def build_baseline_results(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    val_reg = regression_baseline(val_df)
    test_reg = regression_baseline(test_df)
    val_rain = rain_baseline(train_df, val_df)
    test_rain = rain_baseline(train_df, test_df)
    return pd.DataFrame(
        [
            {
                "split": "validation",
                "task": "temperature_regression",
                "model": val_reg["model"],
                "primary_metric": "rmse",
                "value": val_reg["rmse"],
                "training_time": val_reg["training_time"],
                **val_reg,
            },
            {
                "split": "test",
                "task": "temperature_regression",
                "model": test_reg["model"],
                "primary_metric": "rmse",
                "value": test_reg["rmse"],
                "training_time": test_reg["training_time"],
                **test_reg,
            },
            {
                "split": "validation",
                "task": "rain_classification",
                "model": val_rain["model"],
                "primary_metric": "f1",
                "value": val_rain["f1"],
                "training_time": val_rain["training_time"],
                **val_rain,
            },
            {
                "split": "test",
                "task": "rain_classification",
                "model": test_rain["model"],
                "primary_metric": "f1",
                "value": test_rain["f1"],
                "training_time": test_rain["training_time"],
                **test_rain,
            },
        ]
    )


def evaluate_temperature_models(fitted_models, X, y, validation_results):
    rows = []
    training_times = validation_results.set_index("model")["training_time"].to_dict()
    for name, data in fitted_models.items():
        pred = data["model"].predict(X)
        rows.append(
            {
                "model": name,
                "mae": mean_absolute_error(y, pred),
                "rmse": float(np.sqrt(mean_squared_error(y, pred))),
                "r2": r2_score(y, pred),
                "training_time": training_times.get(name, 0.0),
            }
        )
    return pd.DataFrame(rows)


def evaluate_rain_models(fitted_models, X, y, validation_results):
    rows = []
    y = y.astype(int)
    training_times = validation_results.set_index("model")["training_time"].to_dict()
    for name, data in fitted_models.items():
        model = data["model"]
        pred = model.predict(X).astype(int)
        score = prediction_scores(model, X, pred)
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y, pred),
                "precision": precision_score(y, pred, zero_division=0),
                "recall": recall_score(y, pred, zero_division=0),
                "f1": f1_score(y, pred, zero_division=0),
                "roc_auc": roc_auc_score(y, score) if len(set(y)) == 2 else np.nan,
                "training_time": training_times.get(name, 0.0),
            }
        )
    return pd.DataFrame(rows)


def prediction_scores(model, X, fallback):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return fallback


def save_prediction_analysis(
    val_df,
    test_df,
    y_temp_val,
    temp_val_pred,
    y_temp_test,
    temp_test_pred,
    y_rain_val,
    rain_val_pred,
    y_rain_test,
    rain_test_pred,
) -> None:
    val_temp = temperature_prediction_table(val_df, y_temp_val, temp_val_pred)
    test_temp = temperature_prediction_table(test_df, y_temp_test, temp_test_pred)
    val_rain = rain_prediction_table(val_df, y_rain_val, rain_val_pred)
    test_rain = rain_prediction_table(test_df, y_rain_test, rain_test_pred)
    val_temp.to_csv(TABLES_DIR / "temperature_validation_predictions.csv", index=False)
    test_temp.to_csv(TABLES_DIR / "temperature_test_predictions.csv", index=False)
    val_rain.to_csv(TABLES_DIR / "rain_validation_predictions.csv", index=False)
    test_rain.to_csv(TABLES_DIR / "rain_test_predictions.csv", index=False)

    summary = pd.DataFrame(
        [
            temperature_summary("validation", val_temp),
            temperature_summary("test", test_temp),
            rain_summary("validation", val_rain),
            rain_summary("test", test_rain),
        ]
    )
    summary.to_csv(TABLES_DIR / "validation_test_prediction_summary.csv", index=False)
    return {
        "temperature_validation": val_temp,
        "temperature_test": test_temp,
        "rain_validation": val_rain,
        "rain_test": test_rain,
    }


def temperature_prediction_table(df, y_true, y_pred):
    result = df[["time"]].copy()
    result["actual_temperature"] = np.asarray(y_true)
    result["predicted_temperature"] = np.asarray(y_pred)
    result["residual"] = result["actual_temperature"] - result["predicted_temperature"]
    result["absolute_error"] = result["residual"].abs()
    for tolerance in [1, 2, 3]:
        result[f"within_{tolerance}c"] = result["absolute_error"] <= tolerance
    return result


def rain_prediction_table(df, y_true, y_pred):
    result = df[["time"]].copy()
    result["actual_rain"] = np.asarray(y_true).astype(int)
    result["predicted_rain"] = np.asarray(y_pred).astype(int)
    result["correct"] = result["actual_rain"] == result["predicted_rain"]
    return result


def temperature_summary(split, table):
    return {
        "split": split,
        "task": "temperature_regression",
        "mae": table["absolute_error"].mean(),
        "rmse": float(np.sqrt(np.mean(table["residual"] ** 2))),
        "within_1c": table["within_1c"].mean(),
        "within_2c": table["within_2c"].mean(),
        "within_3c": table["within_3c"].mean(),
    }


def rain_summary(split, table):
    y_true = table["actual_rain"]
    y_pred = table["predicted_rain"]
    return {
        "split": split,
        "task": "rain_classification",
        "correct_count": int(table["correct"].sum()),
        "wrong_count": int((~table["correct"]).sum()),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def save_worst_prediction_days(test_df: pd.DataFrame, predictions) -> None:
    errors = test_df[["time", "temperature", "target_temperature", "target_rain"]].copy()
    errors["predicted_temperature"] = predictions
    errors["absolute_error"] = np.abs(errors["target_temperature"] - errors["predicted_temperature"])
    errors.sort_values("absolute_error", ascending=False).head(20).to_csv(
        TABLES_DIR / "worst_prediction_days.csv",
        index=False,
    )


def save_split_summary(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    rows = []
    for name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        rows.append(
            {
                "split": name,
                "rows": len(df),
                "start": df["time"].min(),
                "end": df["time"].max(),
                "rain_rate": df["target_rain"].mean(),
            }
        )
    pd.DataFrame(rows).to_csv(TABLES_DIR / "split_summary.csv", index=False)


if __name__ == "__main__":
    main()
