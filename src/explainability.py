import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance

from .config import FIGURES_DIR, RANDOM_STATE, TABLES_DIR


def generate_model_explanations(
    fitted_temperature: dict,
    fitted_rain: dict,
    X_test: pd.DataFrame,
    y_temp_test: pd.Series,
    y_rain_test: pd.Series,
    feature_cols: list[str],
) -> dict[str, pd.DataFrame]:
    explanations = {}

    explanations["linear_regression_coefficients"] = coefficient_table(
        fitted_temperature["LinearRegression"]["model"],
        feature_cols,
        task="temperature_regression",
    )
    explanations["linear_regression_coefficients"].to_csv(
        TABLES_DIR / "linear_regression_coefficients.csv", index=False
    )

    explanations["knn_permutation_importance"] = permutation_table(
        fitted_temperature["KNNRegressor"]["model"],
        X_test,
        y_temp_test,
        feature_cols,
        scoring="neg_root_mean_squared_error",
        task="temperature_regression",
    )
    explanations["knn_permutation_importance"].to_csv(
        TABLES_DIR / "knn_permutation_importance.csv", index=False
    )

    if "XGBoostRegressor" in fitted_temperature:
        explanations["xgboost_regressor_feature_importance"] = tree_importance_table(
            fitted_temperature["XGBoostRegressor"]["model"],
            feature_cols,
            task="temperature_regression",
        )
        explanations["xgboost_regressor_feature_importance"].to_csv(
            TABLES_DIR / "xgboost_regressor_feature_importance.csv", index=False
        )

    explanations["logistic_regression_coefficients"] = coefficient_table(
        fitted_rain["LogisticRegression"]["model"],
        feature_cols,
        task="rain_classification",
    )
    explanations["logistic_regression_coefficients"].to_csv(
        TABLES_DIR / "logistic_regression_coefficients.csv", index=False
    )

    explanations["svm_permutation_importance"] = permutation_table(
        fitted_rain["SVM"]["model"],
        X_test,
        y_rain_test,
        feature_cols,
        scoring="f1",
        task="rain_classification",
    )
    explanations["svm_permutation_importance"].to_csv(
        TABLES_DIR / "svm_permutation_importance.csv", index=False
    )

    if "XGBoostClassifier" in fitted_rain:
        explanations["xgboost_classifier_feature_importance"] = tree_importance_table(
            fitted_rain["XGBoostClassifier"]["model"],
            feature_cols,
            task="rain_classification",
        )
        explanations["xgboost_classifier_feature_importance"].to_csv(
            TABLES_DIR / "xgboost_classifier_feature_importance.csv", index=False
        )

    plot_explanations(explanations)
    return explanations


def coefficient_table(model, feature_cols: list[str], task: str) -> pd.DataFrame:
    estimator = getattr(model, "best_estimator_", model)
    if hasattr(estimator, "named_steps"):
        fitted_model = estimator.named_steps["model"]
    else:
        fitted_model = estimator

    coefs = np.asarray(fitted_model.coef_).reshape(-1)
    return (
        pd.DataFrame(
            {
                "task": task,
                "feature": feature_cols,
                "coefficient": coefs,
                "absolute_coefficient": np.abs(coefs),
                "direction": np.where(coefs >= 0, "positive", "negative"),
            }
        )
        .sort_values("absolute_coefficient", ascending=False)
        .reset_index(drop=True)
    )


def permutation_table(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_cols: list[str],
    scoring: str,
    task: str,
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring=scoring,
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    return (
        pd.DataFrame(
            {
                "task": task,
                "feature": feature_cols,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def tree_importance_table(model, feature_cols: list[str], task: str) -> pd.DataFrame:
    estimator = getattr(model, "best_estimator_", model)
    if hasattr(estimator, "named_steps"):
        estimator = estimator.named_steps["model"]
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return pd.DataFrame(columns=["task", "feature", "importance"])
    return (
        pd.DataFrame({"task": task, "feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def plot_explanations(explanations: dict[str, pd.DataFrame]) -> None:
    plot_specs = [
        ("linear_regression_coefficients", "absolute_coefficient", "linear_regression_coefficients.png"),
        ("knn_permutation_importance", "importance_mean", "knn_permutation_importance.png"),
        ("xgboost_regressor_feature_importance", "importance", "xgboost_regressor_feature_importance.png"),
        ("logistic_regression_coefficients", "absolute_coefficient", "logistic_regression_coefficients.png"),
        ("svm_permutation_importance", "importance_mean", "svm_permutation_importance.png"),
        ("xgboost_classifier_feature_importance", "importance", "xgboost_classifier_feature_importance.png"),
    ]
    for key, value_col, filename in plot_specs:
        table = explanations.get(key)
        if table is None or table.empty:
            continue
        top = table.head(10)
        plt.figure(figsize=(8, 6))
        sns.barplot(data=top, x=value_col, y="feature")
        plt.title(key.replace("_", " ").title())
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / filename, dpi=160)
        plt.close()
