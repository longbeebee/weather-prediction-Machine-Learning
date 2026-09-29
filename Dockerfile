FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
# The production candidate adapter reuses the estimator implementations from
# the legacy 7-day pipeline module.
COPY seven_day_pipeline.py ./seven_day_pipeline.py

RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "src.production.serving:app", "--host", "0.0.0.0", "--port", "8000"]
