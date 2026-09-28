"""Feature construction shared by training, evaluation, and online serving."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature_engineering import FeatureEngineer


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    blocked = {
        "time",
        "target_temperature",
        "target_precipitation",
        "target_rain",
        "target_weather_level",
        "horizon_temperature",
        "horizon_precipitation",
        "horizon_rain",
    }
    return [column for column in df.select_dtypes(include="number").columns if column not in blocked]


def build_base_features(noon_df: pd.DataFrame) -> pd.DataFrame:
    """Build historical features without making the online row depend on a target."""
    frame = noon_df.copy().sort_values("time").reset_index(drop=True)
    engineer = FeatureEngineer()
    engineer._add_time_features(frame)
    engineer._add_lag_features(frame)
    engineer._add_rolling_features(frame)
    numeric = frame.select_dtypes(include="number").columns
    for column in numeric:
        frame[column] = frame[column].fillna(frame[column].median())
    return frame


def build_horizon_targets(noon_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if not 1 <= horizon <= 7:
        raise ValueError("horizon must be between 1 and 7")
    frame = noon_df[["time", "temperature", "precipitation"]].copy().sort_values("time")
    frame["horizon_temperature"] = frame["temperature"].shift(-horizon)
    frame["horizon_precipitation"] = frame["precipitation"].shift(-horizon)
    frame["horizon_rain"] = (frame["horizon_precipitation"] > 0).astype(float)
    return frame[["time", "horizon_temperature", "horizon_precipitation", "horizon_rain"]]


def merge_horizon_targets(base_features: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    frame = base_features.merge(targets, on="time", how="inner")
    return frame.dropna(subset=["horizon_temperature", "horizon_precipitation", "horizon_rain"]).reset_index(drop=True)


def build_latest_features(noon_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Return one inference row using only observations available at request time."""
    frame = build_base_features(noon_df)
    if len(frame) < 8:
        raise ValueError("at least 8 chronological noon observations are required")
    latest = frame.sort_values("time").iloc[[-1]].copy()
    missing = [column for column in feature_columns if column not in latest.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    values = latest[feature_columns].replace([np.inf, -np.inf], np.nan)
    if values.isna().any().any():
        raise ValueError("latest observation produces missing or non-finite features")
    return values
