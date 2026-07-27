from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "open_meteo_hanoi.csv"
FALLBACK_RAW_DATA_PATH = ROOT_DIR / "open-meteo-21.05N105.81E18m - open-meteo-21.05N105.81E18m.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PROCESSED_DATA_PATH = PROCESSED_DIR / "noon_weather_cleaned.csv"
MODELS_DIR = ROOT_DIR / "models"
TEMP_MODEL_DIR = MODELS_DIR / "temperature_regression"
RAIN_MODEL_DIR = MODELS_DIR / "rain_classification"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
FINAL_REPORT_PATH = REPORTS_DIR / "final_results.md"

RANDOM_STATE = 42
N_SPLITS = 5
SEARCH_ITER = 4

DROPPED_COLUMNS = ["surface_pressure", "wind_direction_10m", "rain"]
TARGET_COLUMNS = [
    "target_temperature",
    "target_precipitation",
    "target_rain",
    "target_weather_level",
]


def ensure_directories() -> None:
    for path in [
        PROCESSED_DIR,
        TEMP_MODEL_DIR,
        RAIN_MODEL_DIR,
        FIGURES_DIR,
        TABLES_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
