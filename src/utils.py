import time
from contextlib import contextmanager
from pathlib import Path

import joblib


@contextmanager
def timer():
    start = time.perf_counter()
    result = {"seconds": 0.0}
    try:
        yield result
    finally:
        result["seconds"] = time.perf_counter() - start


def save_joblib(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def weather_level(temperature: float) -> str:
    if temperature < 22:
        return "cool"
    if temperature <= 30:
        return "normal"
    return "hot"
