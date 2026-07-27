from pathlib import Path

import pandas as pd


class DataLoader:
    def __init__(self, csv_path: Path):
        self.csv_path = Path(csv_path)

    def load_raw_data(self) -> pd.DataFrame:
        header_row = self._find_header_row()
        return pd.read_csv(self.csv_path, skiprows=header_row)

    def normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {
            "temperature_2m (degC)": "temperature",
            "temperature_2m (°C)": "temperature",
            "relative_humidity_2m (%)": "humidity",
            "precipitation (mm)": "precipitation",
            "rain (mm)": "rain",
            "cloud_cover (%)": "cloud_cover",
            "pressure_msl (hPa)": "pressure_msl",
            "surface_pressure (hPa)": "surface_pressure",
            "wind_speed_10m (km/h)": "wind_speed_10m",
            "wind_direction_10m (°)": "wind_direction_10m",
            "shortwave_radiation (W/m²)": "shortwave_radiation",
        }
        normalized = []
        for col in df.columns:
            clean = str(col).strip()
            normalized.append(rename_map.get(clean, clean))
        df = df.copy()
        df.columns = normalized
        return df

    def parse_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"])
        df = df.sort_values("time")
        df = df.drop_duplicates(subset=["time"], keep="first")
        return df.reset_index(drop=True)

    def load(self) -> pd.DataFrame:
        df = self.load_raw_data()
        df = self.normalize_columns(df)
        return self.parse_datetime(df)

    def _find_header_row(self) -> int:
        with self.csv_path.open("r", encoding="utf-8") as handle:
            for i, line in enumerate(handle):
                if line.lower().startswith("time,"):
                    return i
        raise ValueError(f"Could not find Open-Meteo table header in {self.csv_path}")
