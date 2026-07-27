import time

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform, loguniform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import N_SPLITS, RANDOM_STATE, SEARCH_ITER

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - used only when dependency is unavailable.
    XGBClassifier = None


def train_rain_models(X_train, y_train, X_test, y_test) -> tuple[pd.DataFrame, dict]:
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
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
                    SVC(
                        class_weight="balanced",
                        probability=True,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }
    searches = {
        "LogisticRegression": {"model__C": loguniform(0.01, 100)},
        "SVM": {"model__C": loguniform(0.1, 100), "model__kernel": ["linear", "rbf"]},
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

    cv = TimeSeriesSplit(n_splits=min(N_SPLITS, max(2, len(X_train) // 120)))
    rows = []
    fitted = {}
    for name, estimator in models.items():
        start = time.perf_counter()
        estimator = RandomizedSearchCV(
            estimator,
            searches[name],
            n_iter=SEARCH_ITER,
            cv=cv,
            scoring="f1",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        estimator.fit(X_train, y_train)
        training_time = time.perf_counter() - start
        y_pred = estimator.predict(X_test)
        y_score = _prediction_scores(estimator, X_test, y_pred)
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_score)
                if len(set(y_test)) == 2
                else np.nan,
                "training_time": training_time,
            }
        )
        fitted[name] = {"model": estimator, "predictions": y_pred, "scores": y_score}
    return pd.DataFrame(rows), fitted


def _prediction_scores(estimator, X, fallback):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    return fallback
