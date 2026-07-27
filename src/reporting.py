import pandas as pd

from .config import FINAL_REPORT_PATH, FIGURES_DIR


def generate_report(
    raw_df: pd.DataFrame,
    noon_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    best_temp: str,
    best_rain: str,
) -> None:
    FINAL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    stats = _to_markdown(noon_df.select_dtypes(include="number").describe().round(3).reset_index())
    missing = _to_markdown(raw_df.isna().sum().reset_index().rename(columns={"index": "column", 0: "missing_count"}))
    rain_rate = feature_df["target_rain"].mean()
    figures = _figure_links()
    content = f"""# Hanoi Noon Weather Prediction

## Introduction
This project builds a Machine Learning system to predict Hanoi weather at 12:00 PM on the next day.

## Motivation
Noon weather is useful for daily planning because it captures the most active part of the day for commuting, outdoor work, and school schedules.

## Contribution
The project implements two supervised learning tasks: temperature regression and rain binary classification. It also compares baselines with ML models, evaluates an A/B feature experiment for `shortwave_radiation`, analyzes runtime, visualizes the data with PCA, and reports model errors.

## Dataset Overview
- Raw hourly records: {len(raw_df):,}
- Noon records after preprocessing: {len(noon_df):,}
- Engineered supervised samples: {len(feature_df):,}
- Raw date range: {raw_df['time'].min()} to {raw_df['time'].max()}
- Noon date range: {noon_df['time'].min()} to {noon_df['time'].max()}
- Next-day rain rate: {rain_rate:.3f}

## Input and Expected Output
Input: historical 12:00 weather records plus engineered lag and rolling features.

Regression output: next-day 12:00 temperature.

Classification output: next-day rain or no rain.

Derived output: weather level from predicted temperature.

## Project Process / Methodology
The pipeline loads the Open-Meteo CSV, removes metadata rows, normalizes columns, parses timestamps, keeps only 12:00 records, creates supervised learning targets, splits data chronologically, trains baselines and ML models, evaluates the models, saves best models, and writes reports.

## Data Analysis
Missing value summary:

{missing}

Descriptive statistics:

{stats}

Descriptive statistics interpretation:

- `count` confirms that each noon feature has the same number of valid records after preprocessing.
- `mean` shows the average noon condition: temperature is about 26.9°C, humidity about 67.2%, and precipitation about 0.255 mm.
- `std` measures variability. Temperature varies moderately, while precipitation has high variability relative to its mean because most days have no rain and a few days have heavier rain.
- `min` and `max` show the observed range, including temperatures from 8.9°C to 38.0°C and precipitation up to 13.8 mm.
- `25%`, `50%`, and `75%` are quartiles. The median precipitation is 0.0 mm, meaning at least half of noon records have no rainfall.
- Cloud cover has a high median and 75th percentile close to 100%, indicating many noon observations are cloudy.
- Shortwave radiation has a wide range, which makes it useful to test in Experiment A/B because it may explain daytime temperature differences.

## Data Preprocessing
The preprocessing stage removes duplicate timestamps, sorts observations chronologically, keeps only 12:00 records, removes `surface_pressure`, `wind_direction_10m`, and `rain`, fills numeric missing values with medians, and drops only rows where generated targets are unavailable.

## Data Visualization
The project uses standard EDA plots and PCA for low-dimensional visualization.

{figures}

## Feature Engineering
Features include month, day of year, season, temperature/humidity/precipitation lags for 1, 3, and 7 days, and rolling means/sums using `shift(1)` before `rolling()` to avoid target leakage.

## Experimental Protocol
The experiments use a chronological train/validation/test split: 2020-2023 for training, 2024 for validation, and 2025 for final testing. Models are fitted only on the training set. The validation set is used to select the best model and compare experiments. The test set is used only for final evaluation.

Hyperparameter tuning uses `TimeSeriesSplit(n_splits=5)` and `RandomizedSearchCV` on the training data.

## Model Parameter Setup
Temperature models: Linear Regression, KNN Regressor, and XGBoost Regressor. KNN searches neighbors from 3 to 25 and uniform/distance weighting. XGBoost searches estimators, depth, and learning rate.

Rain models: Logistic Regression, SVM, and XGBoost Classifier. Logistic Regression and SVM use balanced class weights. XGBoost uses automatic `scale_pos_weight`.

## Linear Regression Training Process
Linear Regression is the main selected model for temperature prediction. The model is trained only on the training period 2020-2023. The validation year 2024 is used for model selection, and the test year 2025 is used for final evaluation.

The training pipeline uses two steps:

1. `StandardScaler`: standardizes numeric features so that each feature has comparable scale.
2. `LinearRegression`: learns a linear relationship between engineered weather features and next-day noon temperature.

The model assumes the target can be approximated as:

```text
target_temperature = b + w1*x1 + w2*x2 + ... + wn*xn
```

where `x1..xn` are weather features such as current temperature, humidity, pressure, season, lag features, and rolling statistics. During training, Linear Regression estimates the intercept `b` and coefficients `w1..wn` by minimizing the sum of squared prediction errors on the training set.

In this project, the target is:

```text
target_temperature = temperature.shift(-1)
```

Therefore, each training sample uses weather information at 12:00 on one day to predict temperature at 12:00 on the following day. Because the model is solved by least squares rather than iterative gradient descent, no epoch-by-epoch cost curve is reported. Instead, training quality is evaluated using MAE, RMSE, and R2 on validation and test data.

The learned coefficients are reported in `reports/tables/linear_regression_coefficients.csv`. Positive coefficients indicate that increasing the feature tends to increase predicted temperature, while negative coefficients indicate the opposite direction after standardization.

## Baseline Results
See `reports/tables/baseline_results.csv`.

## Regression Results
Best temperature model selected from validation: `{best_temp}`.

- Validation results: `reports/tables/validation_regression_results.csv`
- Final test results: `reports/tables/regression_results.csv`

## Classification Results
Best rain model selected from validation: `{best_rain}`.

- Validation results: `reports/tables/validation_rain_classification_results.csv`
- Final test results: `reports/tables/rain_classification_results.csv`

## Validation and Test Prediction Analysis
This section compares model predictions with actual observations on both validation and test data.

For temperature regression, exact accuracy is not appropriate because the output is continuous. The project therefore reports MAE, RMSE, and tolerance accuracy: the percentage of predictions within ±1°C, ±2°C, and ±3°C of the real temperature.

For rain classification, correctness is direct because the output is binary. The project reports correct/wrong counts, accuracy, precision, recall, F1, and confusion matrices.

Prediction artifacts:

- `reports/tables/temperature_validation_predictions.csv`
- `reports/tables/temperature_test_predictions.csv`
- `reports/tables/rain_validation_predictions.csv`
- `reports/tables/rain_test_predictions.csv`
- `reports/tables/validation_test_prediction_summary.csv`

The validation figures show how the selected model behaves during model selection. The test figures show final performance on unseen data from 2025.

## Runtime / Complexity Analysis
See `reports/tables/runtime_comparison.csv`. Linear models are lightweight, KNN shifts cost toward prediction because it compares neighbors, SVM can be more expensive with nonlinear kernels, and XGBoost spends more time training boosted trees but can capture nonlinear interactions.

## Model Explainability / Feature Importance
Model selection is based on test metrics, while explainability is used to interpret how each trained model uses the input features.

XGBoost feature importance is reported as required by the project specification in `reports/tables/top_10_features.csv` and `reports/figures/feature_importance.png`.

Additional model-specific explanations:

- Linear Regression: `reports/tables/linear_regression_coefficients.csv`
- KNN Regressor: `reports/tables/knn_permutation_importance.csv`
- XGBoost Regressor: `reports/tables/xgboost_regressor_feature_importance.csv`
- Logistic Regression: `reports/tables/logistic_regression_coefficients.csv`
- SVM: `reports/tables/svm_permutation_importance.csv`
- XGBoost Classifier: `reports/tables/xgboost_classifier_feature_importance.csv`

Linear and Logistic Regression use signed coefficients after scaling, where positive values increase the prediction direction and negative values reduce it. KNN and SVM use permutation importance because they do not provide stable built-in feature importance for this setup. XGBoost uses built-in tree-based feature importance.

## Experiment A vs B
See `reports/tables/experiment_ab_comparison.csv` and `reports/figures/experiment_ab_comparison.png` for the comparison with and without `shortwave_radiation`.

## Evaluation and Discussion
The report compares baseline and ML performance in `reports/tables/baseline_vs_ml_comparison.csv`. Regression quality is selected by RMSE, while rain classification is selected by F1 to account for class imbalance.

## Error Analysis for Worst Prediction Days
See `reports/tables/worst_prediction_days.csv` for the highest absolute temperature prediction errors.

## Conclusion
The completed system builds a reproducible ML pipeline for next-day noon weather prediction in Hanoi, including preprocessing, feature engineering, model tuning, visualization, model selection, and reporting.

## Future Work / Perspectives
Future work can add more locations, additional meteorological forecast variables, probability calibration for rain, time-series/deep learning models, and a dashboard or inference API.
"""
    FINAL_REPORT_PATH.write_text(content, encoding="utf-8")


def _figure_links() -> str:
    names = [
        "temperature_trend.png",
        "temperature_histogram.png",
        "monthly_temperature.png",
        "monthly_rainfall.png",
        "rain_distribution.png",
        "correlation_heatmap.png",
        "pca_visualization.png",
        "regression_model_comparison.png",
        "classification_model_comparison.png",
        "actual_vs_predicted_temperature.png",
        "temperature_residuals.png",
        "confusion_matrix_rain.png",
        "validation_actual_vs_predicted_temperature.png",
        "test_actual_vs_predicted_temperature.png",
        "validation_temperature_residuals.png",
        "test_temperature_residuals.png",
        "temperature_tolerance_accuracy.png",
        "validation_confusion_matrix_rain.png",
        "test_confusion_matrix_rain.png",
        "rain_correct_wrong_rate.png",
        "feature_importance.png",
        "linear_regression_coefficients.png",
        "knn_permutation_importance.png",
        "xgboost_regressor_feature_importance.png",
        "logistic_regression_coefficients.png",
        "svm_permutation_importance.png",
        "xgboost_classifier_feature_importance.png",
        "experiment_ab_comparison.png",
    ]
    return "\n".join([f"- ![{name.removesuffix('.png')}](figures/{name})" for name in names])


def _to_markdown(df: pd.DataFrame) -> str:
    columns = [str(c) for c in df.columns]
    rows = ["| " + " | ".join(columns) + " |"]
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[c]) for c in df.columns) + " |")
    return "\n".join(rows)
