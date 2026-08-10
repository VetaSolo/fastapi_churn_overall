# syntax=docker/dockerfile:1

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=0 \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=20 \
    PIP_INDEX_URL=https://pypi.org/simple

WORKDIR /app

COPY requirements.txt .

# Кэш pip между сборками + повтор при таймаутах медленной сети
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install fastapi "uvicorn[standard]" pydantic pandas scikit-learn joblib \
    && pip install pytest httpx

COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
