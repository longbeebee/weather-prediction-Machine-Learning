import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import StandardScaler

from .config import FIGURES_DIR, TABLES_DIR

sns.set_theme(style="whitegrid")


def visualize_eda(df: pd.DataFrame, feature_df: pd.DataFrame, feature_cols: list[str]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _line(df, "time", "temperature", "Noon temperature trend", "temperature_trend.png")
    _hist(df, "temperature", "Noon temperature distribution", "temperature_histogram.png")

    monthly = df.assign(month=df["time"].dt.month)
    _bar(
        monthly.groupby("month", as_index=False)["temperature"].mean(),
        "month",
        "temperature",
        "Monthly average noon temperature",
        "monthly_temperature.png",
    )
    _bar(
        monthly.groupby("month", as_index=False)["precipitation"].sum(),
        "month",
        "precipitation",
        "Monthly rainfall at noon",
        "monthly_rainfall.png",
    )

    rain_dist = feature_df.assign(rain_label=feature_df["target_rain"].map({0.0: "No rain", 1.0: "Rain"}))
    _count(rain_dist, "rain_label", "Rain class distribution", "rain_distribution.png")

    plt.figure(figsize=(12, 9))
    corr = feature_df[feature_cols].corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.2)
    plt.title("Numeric feature correlation heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=160)
    plt.close()

    X_scaled = StandardScaler().fit_transform(feature_df[feature_cols])
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X_scaled)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=components[:, 0], y=components[:, 1], hue=feature_df["target_rain"], palette="Set1", s=24)
    plt.title("PCA visualization of engineered weather features")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pca_visualization.png", dpi=160)
    plt.close()


def visualize_results(
    regression_results: pd.DataFrame,
    rain_results: pd.DataFrame,
    experiment_results: pd.DataFrame,
    y_temp_true,
    y_temp_pred,
    y_rain_true,
    y_rain_pred,
    feature_importance: pd.DataFrame,
) -> None:
    _metric_bar(regression_results, ["mae", "rmse", "r2"], "Regression model comparison", "regression_model_comparison.png")
    _metric_bar(
        rain_results,
        ["accuracy", "precision", "recall", "f1", "roc_auc"],
        "Rain classification model comparison",
        "classification_model_comparison.png",
    )

    plt.figure(figsize=(9, 6))
    plt.scatter(y_temp_true, y_temp_pred, alpha=0.65)
    low = min(np.min(y_temp_true), np.min(y_temp_pred))
    high = max(np.max(y_temp_true), np.max(y_temp_pred))
    plt.plot([low, high], [low, high], color="red", linewidth=1)
    plt.title("Actual vs predicted next-day noon temperature")
    plt.xlabel("Actual temperature")
    plt.ylabel("Predicted temperature")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "actual_vs_predicted_temperature.png", dpi=160)
    plt.close()

    residuals = np.asarray(y_temp_true) - np.asarray(y_temp_pred)
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True)
    plt.title("Temperature prediction residuals")
    plt.xlabel("Residual")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "temperature_residuals.png", dpi=160)
    plt.close()

    cm = confusion_matrix(y_rain_true, y_rain_pred)
    ConfusionMatrixDisplay(cm, display_labels=["No rain", "Rain"]).plot(cmap="Blues")
    plt.title("Rain classification confusion matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix_rain.png", dpi=160)
    plt.close()

    if not feature_importance.empty:
        top = feature_importance.head(10)
        plt.figure(figsize=(8, 6))
        sns.barplot(data=top, x="importance", y="feature")
        plt.title("Top XGBoost feature importance")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=160)
        plt.close()
    else:
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "Feature importance unavailable", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=160)
        plt.close()

    _experiment_plot(experiment_results)


def visualize_validation_test_predictions(
    temperature_validation: pd.DataFrame,
    temperature_test: pd.DataFrame,
    rain_validation: pd.DataFrame,
    rain_test: pd.DataFrame,
) -> None:
    _actual_vs_predicted_split(
        temperature_validation,
        "Validation actual vs predicted next-day noon temperature",
        "validation_actual_vs_predicted_temperature.png",
    )
    _actual_vs_predicted_split(
        temperature_test,
        "Test actual vs predicted next-day noon temperature",
        "test_actual_vs_predicted_temperature.png",
    )
    _residual_split(
        temperature_validation,
        "Validation temperature residuals",
        "validation_temperature_residuals.png",
    )
    _residual_split(
        temperature_test,
        "Test temperature residuals",
        "test_temperature_residuals.png",
    )
    _temperature_tolerance_plot(temperature_validation, temperature_test)
    _rain_confusion_split(
        rain_validation,
        "Validation rain confusion matrix",
        "validation_confusion_matrix_rain.png",
    )
    _rain_confusion_split(
        rain_test,
        "Test rain confusion matrix",
        "test_confusion_matrix_rain.png",
    )
    _rain_correct_wrong_plot(rain_validation, rain_test)


def visualize_temperature_model_predictions(
    fitted_temperature: dict,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    val_predictions = {}
    test_predictions = {}
    val_metrics = {}
    test_metrics = {}
    for model_name, payload in fitted_temperature.items():
        safe_name = _safe_model_name(model_name)
        model = payload["model"]
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        val_predictions[model_name] = val_pred
        test_predictions[model_name] = test_pred
        val_metrics[model_name] = _regression_metric_summary(y_val, val_pred)
        test_metrics[model_name] = _regression_metric_summary(y_test, test_pred)
        _actual_vs_predicted_arrays(
            y_val,
            val_pred,
            f"Validation actual vs predicted temperature - {model_name}\n{_metric_text(val_metrics[model_name])}",
            f"validation_actual_vs_predicted_{safe_name}.png",
        )
        _actual_vs_predicted_arrays(
            y_test,
            test_pred,
            f"Test actual vs predicted temperature - {model_name}\n{_metric_text(test_metrics[model_name])}",
            f"test_actual_vs_predicted_{safe_name}.png",
        )

    _actual_vs_predicted_grid(
        y_val,
        val_predictions,
        val_metrics,
        "Validation actual vs predicted temperature by model",
        "validation_actual_vs_predicted_all_regression_models.png",
    )
    _actual_vs_predicted_grid(
        y_test,
        test_predictions,
        test_metrics,
        "Test actual vs predicted temperature by model",
        "test_actual_vs_predicted_all_regression_models.png",
    )
    _absolute_error_boxplot(
        y_val,
        val_predictions,
        "Validation absolute error distribution by regression model",
        "validation_temperature_absolute_error_by_model.png",
    )
    _absolute_error_boxplot(
        y_test,
        test_predictions,
        "Test absolute error distribution by regression model",
        "test_temperature_absolute_error_by_model.png",
    )


def save_feature_importance(model, feature_cols: list[str]) -> pd.DataFrame:
    estimator = getattr(model, "best_estimator_", model)
    if hasattr(estimator, "named_steps"):
        estimator = estimator.named_steps.get("model", estimator)
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        result = pd.DataFrame(columns=["feature", "importance"])
    else:
        result = (
            pd.DataFrame({"feature": feature_cols, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    result.head(10).to_csv(TABLES_DIR / "top_10_features.csv", index=False)
    return result


def _line(df, x, y, title, filename):
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=df, x=x, y=y, linewidth=1)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=160)
    plt.close()


def _hist(df, col, title, filename):
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col], kde=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=160)
    plt.close()


def _bar(df, x, y, title, filename):
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x=x, y=y)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=160)
    plt.close()


def _count(df, x, title, filename):
    plt.figure(figsize=(6, 5))
    sns.countplot(data=df, x=x)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=160)
    plt.close()


def _metric_bar(results, metrics, title, filename):
    melted = results.melt(id_vars="model", value_vars=metrics, var_name="metric", value_name="value")
    plt.figure(figsize=(10, 5))
    sns.barplot(data=melted, x="model", y="value", hue="metric")
    plt.title(title)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=160)
    plt.close()


def _experiment_plot(experiment_results):
    available = [
        c
        for c in [
            "best_validation_regression_rmse",
            "test_rmse_for_validation_best",
            "best_validation_rain_f1",
            "test_f1_for_validation_best",
        ]
        if c in experiment_results.columns
    ]
    melted = experiment_results.melt(id_vars="experiment", value_vars=available, var_name="metric", value_name="value")
    plt.figure(figsize=(8, 5))
    sns.barplot(data=melted, x="experiment", y="value", hue="metric")
    plt.title("Experiment A/B comparison")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "experiment_ab_comparison.png", dpi=160)
    plt.close()


def _actual_vs_predicted_split(df, title, filename):
    plt.figure(figsize=(9, 6))
    plt.scatter(df["actual_temperature"], df["predicted_temperature"], alpha=0.65)
    low = min(df["actual_temperature"].min(), df["predicted_temperature"].min())
    high = max(df["actual_temperature"].max(), df["predicted_temperature"].max())
    plt.plot([low, high], [low, high], color="red", linewidth=1)
    plt.title(title)
    plt.xlabel("Actual temperature")
    plt.ylabel("Predicted temperature")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=160)
    plt.close()


def _actual_vs_predicted_arrays(y_true, y_pred, title, filename):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.65)
    low = min(y_true.min(), y_pred.min())
    high = max(y_true.max(), y_pred.max())
    plt.plot([low, high], [low, high], color="red", linewidth=1)
    plt.title(title)
    plt.xlabel("Actual temperature")
    plt.ylabel("Predicted temperature")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=160)
    plt.close()


def _actual_vs_predicted_grid(y_true, predictions: dict, metrics: dict, title: str, filename: str):
    y_true = np.asarray(y_true)
    n_models = len(predictions)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), sharex=True, sharey=True)
    if n_models == 1:
        axes = [axes]
    all_pred = np.concatenate([np.asarray(pred) for pred in predictions.values()])
    low = min(y_true.min(), all_pred.min())
    high = max(y_true.max(), all_pred.max())
    for ax, (model_name, pred) in zip(axes, predictions.items()):
        pred = np.asarray(pred)
        ax.scatter(y_true, pred, alpha=0.65)
        ax.plot([low, high], [low, high], color="red", linewidth=1)
        ax.set_title(f"{model_name}\n{_metric_text(metrics[model_name])}")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=160)
    plt.close()


def _safe_model_name(model_name: str) -> str:
    return (
        model_name.replace("LinearRegression", "linear_regression")
        .replace("KNNRegressor", "knn_regressor")
        .replace("XGBoostRegressor", "xgboost_regressor")
        .lower()
    )


def _regression_metric_summary(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residual = y_true - y_pred
    mae = np.mean(np.abs(residual))
    rmse = np.sqrt(np.mean(residual ** 2))
    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot else np.nan
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _metric_text(metrics):
    return f"MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}, R2={metrics['r2']:.3f}"


def _absolute_error_boxplot(y_true, predictions: dict, title: str, filename: str):
    y_true = np.asarray(y_true)
    rows = []
    for model_name, pred in predictions.items():
        pred = np.asarray(pred)
        for error in np.abs(y_true - pred):
            rows.append({"model": model_name, "absolute_error": error})
    df = pd.DataFrame(rows)
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x="model", y="absolute_error")
    sns.stripplot(data=df, x="model", y="absolute_error", color="black", alpha=0.18, size=2)
    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel("Absolute error (°C)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=160)
    plt.close()


def _residual_split(df, title, filename):
    plt.figure(figsize=(8, 5))
    sns.histplot(df["residual"], kde=True)
    plt.axvline(0, color="red", linewidth=1)
    plt.title(title)
    plt.xlabel("Residual")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=160)
    plt.close()


def _temperature_tolerance_plot(validation_df, test_df):
    rows = []
    for split, df in [("validation", validation_df), ("test", test_df)]:
        for tolerance in [1, 2, 3]:
            rows.append(
                {
                    "split": split,
                    "tolerance": f"within ±{tolerance}°C",
                    "rate": df[f"within_{tolerance}c"].mean(),
                }
            )
    plt.figure(figsize=(8, 5))
    sns.barplot(data=pd.DataFrame(rows), x="tolerance", y="rate", hue="split")
    plt.ylim(0, 1)
    plt.title("Temperature prediction tolerance accuracy")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "temperature_tolerance_accuracy.png", dpi=160)
    plt.close()


def _rain_confusion_split(df, title, filename):
    cm = confusion_matrix(df["actual_rain"], df["predicted_rain"])
    ConfusionMatrixDisplay(cm, display_labels=["No rain", "Rain"]).plot(cmap="Blues")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=160)
    plt.close()


def _rain_correct_wrong_plot(validation_df, test_df):
    rows = []
    for split, df in [("validation", validation_df), ("test", test_df)]:
        rows.append({"split": split, "result": "correct", "rate": df["correct"].mean()})
        rows.append({"split": split, "result": "wrong", "rate": 1 - df["correct"].mean()})
    plt.figure(figsize=(7, 5))
    sns.barplot(data=pd.DataFrame(rows), x="split", y="rate", hue="result")
    plt.ylim(0, 1)
    plt.title("Rain prediction correct vs wrong rate")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rain_correct_wrong_rate.png", dpi=160)
    plt.close()
