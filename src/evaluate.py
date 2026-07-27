import pandas as pd

from .config import RAIN_MODEL_DIR, TABLES_DIR, TEMP_MODEL_DIR
from .utils import save_joblib


def save_results(
    validation_regression_results: pd.DataFrame,
    validation_rain_results: pd.DataFrame,
    test_regression_results: pd.DataFrame,
    test_rain_results: pd.DataFrame,
    baseline_results: pd.DataFrame,
    experiment_results: pd.DataFrame,
    fitted_temperature: dict,
    fitted_rain: dict,
) -> tuple[str, str]:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    validation_regression_results.to_csv(TABLES_DIR / "validation_regression_results.csv", index=False)
    validation_rain_results.to_csv(TABLES_DIR / "validation_rain_classification_results.csv", index=False)
    test_regression_results.to_csv(TABLES_DIR / "regression_results.csv", index=False)
    test_rain_results.to_csv(TABLES_DIR / "rain_classification_results.csv", index=False)
    validation_combined = pd.concat(
        [
            validation_regression_results.assign(task="temperature_regression"),
            validation_rain_results.assign(task="rain_classification"),
        ],
        ignore_index=True,
        sort=False,
    )
    test_combined = pd.concat(
        [
            test_regression_results.assign(task="temperature_regression"),
            test_rain_results.assign(task="rain_classification"),
        ],
        ignore_index=True,
        sort=False,
    )
    validation_combined.to_csv(TABLES_DIR / "validation_results.csv", index=False)
    test_combined.to_csv(TABLES_DIR / "test_results.csv", index=False)
    baseline_results.to_csv(TABLES_DIR / "baseline_results.csv", index=False)
    experiment_results.to_csv(TABLES_DIR / "experiment_ab_comparison.csv", index=False)
    _baseline_vs_ml(test_regression_results, test_rain_results, baseline_results).to_csv(
        TABLES_DIR / "baseline_vs_ml_comparison.csv", index=False
    )
    _runtime(test_regression_results, test_rain_results, baseline_results).to_csv(
        TABLES_DIR / "runtime_comparison.csv", index=False
    )

    best_temp = validation_regression_results.sort_values("rmse").iloc[0]["model"]
    best_rain = validation_rain_results.sort_values("f1", ascending=False).iloc[0]["model"]
    save_joblib(fitted_temperature[best_temp]["model"], TEMP_MODEL_DIR / "best_temperature_model.joblib")
    save_joblib(fitted_rain[best_rain]["model"], RAIN_MODEL_DIR / "best_rain_model.joblib")
    return best_temp, best_rain


def _baseline_vs_ml(regression_results, rain_results, baseline_results):
    rows = []
    for _, row in baseline_results.iterrows():
        rows.append({"task": row["task"], "model": row["model"], "primary_metric": row["primary_metric"], "value": row["value"]})
    rows.append(
        {
            "task": "temperature_regression",
            "model": regression_results.sort_values("rmse").iloc[0]["model"],
            "primary_metric": "rmse",
            "value": regression_results["rmse"].min(),
        }
    )
    rows.append(
        {
            "task": "rain_classification",
            "model": rain_results.sort_values("f1", ascending=False).iloc[0]["model"],
            "primary_metric": "f1",
            "value": rain_results["f1"].max(),
        }
    )
    return pd.DataFrame(rows)


def _runtime(regression_results, rain_results, baseline_results):
    baseline = baseline_results[["task", "model", "training_time"]]
    regression = regression_results[["model", "training_time"]].assign(task="temperature_regression")
    rain = rain_results[["model", "training_time"]].assign(task="rain_classification")
    return pd.concat([baseline, regression, rain], ignore_index=True)
