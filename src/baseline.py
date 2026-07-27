import time

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


def regression_baseline(test_df: pd.DataFrame) -> dict:
    start = time.perf_counter()
    y_true = test_df["target_temperature"]
    y_pred = test_df["temperature"]
    return {
        "model": "Baseline_TodayTemperature",
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score(y_true, y_pred),
        "training_time": time.perf_counter() - start,
    }


def rain_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    start = time.perf_counter()
    majority = int(train_df["target_rain"].mode().iloc[0])
    y_true = test_df["target_rain"].astype(int)
    y_pred = np.full(len(test_df), majority)
    return {
        "model": "Baseline_MajorityClass",
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_pred) if len(set(y_true)) == 2 else np.nan,
        "training_time": time.perf_counter() - start,
    }
