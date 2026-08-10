"""Сохранение и загрузка обученной churn-модели."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

from src.core.config import MODEL_PATH


def save_churn_model(
    model: Pipeline,
    metrics: dict[str, float],
    model_type: str,
    hyperparameters: dict,
    file_path: Path | None = None,
) -> dict[str, Any]:
    if file_path is None:
        file_path = MODEL_PATH

    file_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "model_type": model_type,
        "hyperparameters": hyperparameters,
    }
    joblib.dump({"model": model, **metadata}, file_path)
    return metadata


def load_churn_model(file_path: Path | None = None) -> dict[str, Any] | None:
    if file_path is None:
        file_path = MODEL_PATH

    if not file_path.exists():
        return None

    model_bundle = joblib.load(file_path)
    if not isinstance(model_bundle, dict):
        raise ValueError("Файл модели имеет неправильную структуру.")
    if "model" not in model_bundle:
        raise ValueError("В файле отсутствует модель.")

    model_bundle.setdefault("trained_at", None)
    model_bundle.setdefault("metrics", None)
    model_bundle.setdefault("model_type", None)
    model_bundle.setdefault("hyperparameters", {})
    return model_bundle
