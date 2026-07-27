# Hanoi Noon Weather Prediction

This project predicts Hanoi weather at 12:00 PM on the next day using Open-Meteo historical weather data.

## Tasks

- Temperature prediction: regression target `target_temperature`
- Rain prediction: binary classification target `target_rain`
- Weather level: derived from predicted temperature for reporting only

## Run

```bash
python main.py
```

The pipeline generates processed data, model artifacts, figures, result tables, and `reports/final_results.md`.

## Outputs

- `data/processed/noon_weather_cleaned.csv`
- `models/temperature_regression/best_temperature_model.joblib`
- `models/rain_classification/best_rain_model.joblib`
- `reports/figures/*.png`
- `reports/tables/*.csv`
- `reports/final_results.md`
