# Tuned vs Original 7-Day Pipeline Comparison

## Setup

This report compares two 7-day forecasting pipeline versions:

- Original: `reports/seven_day_forecast`, `RandomizedSearchCV n_iter = 4`
- Tuned: `reports/seven_day_forecast_tuned`, `RandomizedSearchCV n_iter = 20`

The tuned version does not overwrite the original version. It writes all artifacts to `reports/seven_day_forecast_tuned`.

## Summary Result

Increasing the number of sampled configurations improves the average 7-day performance:

| Metric | Original | Tuned | Direction |
|---|---:|---:|---|
| Mean temperature test RMSE | 3.290 | 3.159 | Improved |
| Mean rain test F1 | 0.625 | 0.632 | Improved |

## Horizon-Level Comparison

| Horizon | Original Temp Model | Original RMSE | Tuned Temp Model | Tuned RMSE | RMSE Delta | Original Rain Model | Original F1 | Tuned Rain Model | Tuned F1 | F1 Delta |
|---:|---|---:|---|---:|---:|---|---:|---|---:|---:|
| 1 | LinearRegression | 2.319 | XGBoostRegressor | 2.358 | +0.039 | XGBoostClassifier | 0.652 | XGBoostClassifier | 0.678 | +0.026 |
| 2 | LinearRegression | 3.010 | XGBoostRegressor | 3.061 | +0.051 | LogisticRegression | 0.653 | LogisticRegression | 0.653 | +0.000 |
| 3 | XGBoostRegressor | 3.436 | XGBoostRegressor | 3.283 | -0.153 | LogisticRegression | 0.638 | LogisticRegression | 0.638 | +0.000 |
| 4 | XGBoostRegressor | 3.682 | XGBoostRegressor | 3.415 | -0.267 | LogisticRegression | 0.617 | LogisticRegression | 0.617 | +0.000 |
| 5 | XGBoostRegressor | 3.747 | XGBoostRegressor | 3.512 | -0.235 | LogisticRegression | 0.587 | LogisticRegression | 0.587 | +0.000 |
| 6 | XGBoostRegressor | 3.422 | XGBoostRegressor | 3.262 | -0.160 | LogisticRegression | 0.597 | XGBoostClassifier | 0.620 | +0.023 |
| 7 | XGBoostRegressor | 3.417 | XGBoostRegressor | 3.220 | -0.198 | SVM | 0.634 | SVM | 0.634 | +0.000 |

## Interpretation

The tuned version improves long-horizon temperature forecasting, especially horizons 3 to 7. This is expected because a larger random search explores more XGBoost configurations and can find better tree depth, learning rate, and estimator count combinations.

The tuned version slightly improves rain forecasting on average. Improvements appear mainly at horizons 1 and 6; other horizons remain unchanged because the same rain model and threshold remain best after validation selection.

The tuned version is not strictly better for every horizon. Temperature RMSE is slightly worse at horizons 1 and 2, but the average performance across all 7 horizons is better.

## Conclusion

Increasing `RandomizedSearchCV n_iter` from 4 to 20 improves the overall 7-day pipeline, especially for temperature regression. The improvement is moderate, not dramatic, but it makes the tuned report more defensible because the hyperparameter search is less shallow than the original version.
