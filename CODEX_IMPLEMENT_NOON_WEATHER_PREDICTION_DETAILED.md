
# HANOI NOON WEATHER PREDICTION - FULL CODEX IMPLEMENTATION SPECIFICATION

## 1. PROJECT OBJECTIVE

Build a complete Machine Learning system to predict Hanoi weather at 12:00 PM of the next day.

### Scope

Only two ML tasks:

1. Temperature Prediction (Regression)
2. Rain Prediction (Binary Classification)

Weather Level is NOT a separate ML task.

Weather Level is derived from predicted temperature:

```python
if predicted_temperature < 22:
    weather_level = "cool"
elif predicted_temperature <= 30:
    weather_level = "normal"
else:
    weather_level = "hot"
```

---

# 2. TECHNOLOGY STACK

Python 3.11

Libraries:

- pandas
- numpy
- matplotlib
- scikit-learn
- xgboost
- seaborn
- scipy
- joblib

requirements.txt must be generated automatically.

---

# 3. PROJECT STRUCTURE

weather-prediction/

data/
    raw/
    processed/

models/
    temperature_regression/
    rain_classification/

reports/
    figures/
    tables/
    final_results.md

notebooks/
    01_eda.ipynb
    02_feature_engineering.ipynb
    03_temperature_regression.ipynb
    04_rain_prediction.ipynb

src/
    data_loader.py
    preprocessing.py
    feature_engineering.py
    split.py
    baseline.py
    train_temperature.py
    train_rain.py
    evaluate.py
    reporting.py
    config.py
    utils.py

main.py

README.md

requirements.txt

---

# 4. DATA LOADER

Create class:

DataLoader

Methods:

load_raw_data()
normalize_columns()
parse_datetime()

Responsibilities:

- Detect Open-Meteo metadata rows
- Load actual table
- Convert time column
- Sort ascending
- Remove duplicate timestamps

Output:

pandas DataFrame

---

# 5. PREPROCESSING

Create class:

Preprocessor

Method:

filter_noon_records()

Keep only:

hour == 12

Remove columns:

- surface_pressure
- wind_direction_10m
- rain

Run experiments:

A:
without shortwave_radiation

B:
with shortwave_radiation

Handle missing values:

Numeric:
median

Drop rows only when target unavailable.

Save:

data/processed/noon_weather_cleaned.csv

---

# 6. FEATURE ENGINEERING

Create class:

FeatureEngineer

## Time Features

month

day_of_year

season

Season Mapping:

Winter
Spring
Summer
Autumn

Encode:

0..3

## Lag Features

temperature:

lag1
lag3
lag7

humidity:

lag1
lag3
lag7

precipitation:

lag1
lag3
lag7

## Rolling Features

IMPORTANT

Must use:

shift(1)

Example:

temperature.shift(1).rolling(7).mean()

Features:

temp_roll_mean_3
temp_roll_mean_7

humidity_roll_mean_3
humidity_roll_mean_7

rain_roll_sum_3
rain_roll_sum_7

---

# 7. TARGET GENERATION

target_temperature

temperature.shift(-1)

target_precipitation

precipitation.shift(-1)

target_rain

target_precipitation > 0

target_weather_level

generated only for reporting

---

# 8. EXPLORATORY DATA ANALYSIS

Generate:

temperature_trend.png

temperature_histogram.png

monthly_temperature.png

monthly_rainfall.png

rain_distribution.png

correlation_heatmap.png

pca_visualization.png

Save:

reports/figures

---

# 9. DATA SPLIT

Primary:

Train:
2020-2023

Validation:
2024

Test:
2025

Fallback:

70%
15%
15%

Must be chronological.

No shuffle allowed.

---

# 10. BASELINE MODELS

Temperature Baseline

predict:

temperature tomorrow = temperature today

Metrics:

MAE

RMSE

R2

Rain Baseline

predict majority class

Metrics:

Accuracy

Precision

Recall

F1

Save results.

---

# 11. TEMPERATURE REGRESSION

Train only:

Linear Regression

KNN Regressor

XGBoost Regressor

## Pipeline

Linear:

StandardScaler
LinearRegression

KNN:

StandardScaler
KNeighborsRegressor

XGBoost:

XGBRegressor

## Hyperparameter Search

KNN

n_neighbors:
3-25

weights:
uniform
distance

XGBoost

n_estimators:
100-500

max_depth:
3-10

learning_rate:
0.01-0.3

---

# 12. RAIN CLASSIFICATION

Train only:

Logistic Regression

SVM

XGBoost Classifier

## Imbalance Handling

Logistic:

class_weight='balanced'

SVM:

class_weight='balanced'

XGBoost:

scale_pos_weight

computed automatically

## Hyperparameter Search

Logistic

C:
0.01-100

SVM

C:
0.1-100

kernel:
linear
rbf

XGBoost

n_estimators
max_depth
learning_rate

---

# 13. MODEL TUNING

Use:

TimeSeriesSplit

n_splits=5

Use:

RandomizedSearchCV

Never use KFold.

---

# 14. EVALUATION

Regression

MAE

RMSE

R2

Output:

reports/tables/regression_results.csv

Columns:

model
mae
rmse
r2
training_time

Classification

accuracy

precision

recall

f1

roc_auc

Output:

reports/tables/rain_classification_results.csv

Columns:

model
accuracy
precision
recall
f1
roc_auc
training_time

---

# 15. FEATURE IMPORTANCE

For XGBoost only.

Generate:

feature_importance.png

top_10_features.csv

Include explanation inside final report.

---

# 16. MODEL SELECTION RULE

Temperature:

best RMSE

Rain:

best F1

Save:

best_temperature_model.joblib

best_rain_model.joblib

---

# 17. REPORT GENERATION

Generate:

reports/final_results.md

Sections:

Introduction

Dataset Overview

EDA Findings

Feature Engineering

Baseline Results

Regression Results

Classification Results

Feature Importance

Experiment A vs B

Discussion

Conclusion

Future Work

---

# 18. MAIN PIPELINE

main.py

Execution Order:

load_data()

preprocess()

feature_engineering()

eda()

split()

baseline()

train_temperature()

train_rain()

evaluate()

generate_report()

save_models()

---

# 19. ACCEPTANCE CRITERIA

Codex implementation is accepted only if:

✓ Only 12PM records used

✓ No target leakage

✓ Rolling features use shift(1)

✓ Rain feature excluded from training

✓ TimeSeriesSplit used

✓ Baselines implemented

✓ Hyperparameter tuning implemented

✓ Experiment A/B implemented

✓ Feature importance generated

✓ Best models saved

✓ Final report generated

✓ python main.py runs successfully

---

# 20. BONUS FOR HIGH SCORE

Implement:

1. Experiment comparison table:
   With vs Without shortwave_radiation

2. Baseline vs ML comparison table

3. Feature importance interpretation

4. Rain imbalance analysis

5. Error analysis for worst prediction days

These sections must appear in final_results.md.
