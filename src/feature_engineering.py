import numpy as np
import pandas as pd

from .utils import weather_level


class FeatureEngineer:
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().sort_values("time").reset_index(drop=True)
        self._add_time_features(df)
        self._add_lag_features(df)
        self._add_rolling_features(df)
        self._add_targets(df)
        df = df.dropna(subset=["target_temperature", "target_precipitation", "target_rain"])
        numeric_cols = df.select_dtypes(include="number").columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        return df.reset_index(drop=True)

    def _add_time_features(self, df: pd.DataFrame) -> None:
        df["month"] = df["time"].dt.month
        df["day_of_year"] = df["time"].dt.dayofyear
        df["season"] = np.select(
            [
                df["month"].isin([12, 1, 2]),
                df["month"].isin([3, 4, 5]),
                df["month"].isin([6, 7, 8]),
                df["month"].isin([9, 10, 11]),
            ],
            [0, 1, 2, 3],
            default=0,
        )

    def _add_lag_features(self, df: pd.DataFrame) -> None:
        for col, prefix in [
            ("temperature", "temp"),
            ("humidity", "humidity"),
            ("precipitation", "rain"),
        ]:
            if col in df.columns:
                for lag in [1, 3, 7]:
                    df[f"{prefix}_lag{lag}"] = df[col].shift(lag)

    def _add_rolling_features(self, df: pd.DataFrame) -> None:
        df["temp_roll_mean_3"] = df["temperature"].shift(1).rolling(3).mean()
        df["temp_roll_mean_7"] = df["temperature"].shift(1).rolling(7).mean()
        df["humidity_roll_mean_3"] = df["humidity"].shift(1).rolling(3).mean()
        df["humidity_roll_mean_7"] = df["humidity"].shift(1).rolling(7).mean()
        df["rain_roll_sum_3"] = df["precipitation"].shift(1).rolling(3).sum()
        df["rain_roll_sum_7"] = df["precipitation"].shift(1).rolling(7).sum()

    def _add_targets(self, df: pd.DataFrame) -> None:
        df["target_temperature"] = df["temperature"].shift(-1)
        df["target_precipitation"] = df["precipitation"].shift(-1)
        df["target_rain"] = (df["target_precipitation"] > 0).astype(float)
        df["target_weather_level"] = df["target_temperature"].apply(
            lambda value: weather_level(value) if pd.notna(value) else np.nan
        )
