import pandas as pd

from src.production.features import build_base_features, build_latest_features, select_feature_columns


def sample_noon_rows(count=10):
    return pd.DataFrame(
        {
            "time": pd.date_range("2025-01-01 12:00", periods=count, freq="D"),
            "temperature": range(20, 20 + count),
            "humidity": [60] * count,
            "precipitation": [0.0] * count,
            "cloud_cover": [50] * count,
            "pressure_msl": [1010.0] * count,
            "wind_speed_10m": [5.0] * count,
            "shortwave_radiation": [500.0] * count,
        }
    )


def test_latest_features_are_target_free_and_ordered():
    frame = build_base_features(sample_noon_rows())
    columns = select_feature_columns(frame)
    latest = build_latest_features(sample_noon_rows(), columns)
    assert list(latest.columns) == columns
    assert latest.shape == (1, len(columns))
    assert not latest.isna().any().any()
    assert "target_temperature" not in columns

