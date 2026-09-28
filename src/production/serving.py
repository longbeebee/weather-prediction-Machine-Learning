"""FastAPI serving for an already evaluated 7-day production manifest."""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from src.production.features import build_latest_features
from src.utils import weather_level
from src.seven_day_candidates import rain_probabilities


REQUESTS = Counter("weather_api_requests_total", "Total API requests", ["endpoint", "status"])
PREDICTION_LATENCY = Histogram("weather_api_prediction_latency_seconds", "Prediction latency in seconds")
MODEL_READY = Gauge("weather_api_model_ready", "Whether the production model bundle is ready")


class WeatherObservation(BaseModel):
    time: datetime
    temperature: float
    humidity: float = Field(ge=0, le=100)
    precipitation: float = Field(ge=0)
    cloud_cover: float = Field(ge=0, le=100)
    pressure_msl: float
    wind_speed_10m: float = Field(ge=0)
    shortwave_radiation: float = Field(ge=0)


class ForecastRequest(BaseModel):
    observations: list[WeatherObservation] = Field(min_length=8, max_length=365)


class ForecastItem(BaseModel):
    horizon_day: int
    forecast_time: datetime
    predicted_temperature: float
    predicted_rain: bool
    predicted_rain_probability: float
    weather_level: str
    temperature_model: str
    rain_model: str


class ForecastResponse(BaseModel):
    request_id: str
    contract_version: str
    model_created_at: str
    forecast: list[ForecastItem]


class ModelService:
    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path
        self.manifest: dict | None = None
        self.models: dict[int, dict] = {}
        self.error: str | None = None
        self._load()

    @property
    def ready(self) -> bool:
        return self.manifest is not None and len(self.models) == 7

    def _load(self) -> None:
        try:
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if self.manifest.get("stage") != "production":
                raise ValueError("manifest is not a production manifest")
            root = self.manifest_path.parent
            for item in self.manifest["horizons"]:
                horizon = int(item["horizon_day"])
                self.models[horizon] = {
                    "temperature": joblib.load(root / item["temperature_model"]),
                    "rain": joblib.load(root / item["rain_model"]),
                    "temperature_model_name": item["temperature_model_name"],
                    "rain_model_name": item["rain_model_name"],
                    "rain_threshold": float(item["rain_threshold"]),
                }
        except Exception as exc:  # readiness exposes the failure without crashing the process
            self.error = str(exc)
            self.manifest = None
            self.models = {}
        MODEL_READY.set(1 if self.ready else 0)

    def predict(self, observations: list[WeatherObservation]) -> list[ForecastItem]:
        if not self.ready:
            raise RuntimeError("production models are not ready")
        frame = pd.DataFrame([observation.model_dump() for observation in observations])
        frame["time"] = pd.to_datetime(frame["time"])
        frame = frame.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
        latest_time = frame["time"].iloc[-1]
        features = build_latest_features(frame, self.manifest["feature_columns"])
        result = []
        for horizon in range(1, 8):
            model = self.models[horizon]
            temperature = float(model["temperature"].predict(features)[0])
            probability = float(rain_probabilities(model["rain"], features)[0])
            result.append(ForecastItem(
                horizon_day=horizon,
                forecast_time=latest_time + timedelta(days=horizon),
                predicted_temperature=temperature,
                predicted_rain=probability >= model["rain_threshold"],
                predicted_rain_probability=probability,
                weather_level=weather_level(temperature),
                temperature_model=model["temperature_model_name"],
                rain_model=model["rain_model_name"],
            ))
        return result


def create_app(manifest_path: Path | None = None) -> FastAPI:
    selected_path = manifest_path or Path(os.getenv("SEVEN_DAY_MODEL_MANIFEST", "models/seven_day_production/production_manifest.json"))
    service = ModelService(selected_path)
    app = FastAPI(title="Hanoi 7-Day Weather Forecast API", version="1.0.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict:
        if not service.ready:
            raise HTTPException(status_code=503, detail={"status": "not_ready", "error": service.error})
        return {"status": "ready", "contract_version": service.manifest["contract_version"]}

    @app.get("/api/v1/model/info")
    def model_info() -> dict:
        if not service.ready:
            raise HTTPException(status_code=503, detail="production models are not ready")
        return {"contract_version": service.manifest["contract_version"], "created_at": service.manifest["created_at"], "horizons": 7}

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/api/v1/predict", response_model=ForecastResponse)
    def predict(request: ForecastRequest) -> ForecastResponse:
        request_id = str(uuid.uuid4())
        started = time.perf_counter()
        try:
            with PREDICTION_LATENCY.time():
                forecasts = service.predict(request.observations)
        except ValueError as exc:
            REQUESTS.labels("predict", "422").inc()
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            REQUESTS.labels("predict", "503").inc()
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        REQUESTS.labels("predict", "200").inc()
        _ = time.perf_counter() - started
        return ForecastResponse(request_id=request_id, contract_version=service.manifest["contract_version"], model_created_at=service.manifest["created_at"], forecast=forecasts)

    return app


app = create_app()
