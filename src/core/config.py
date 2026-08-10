"""Конфигурация путей и значений по умолчанию."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = PROJECT_ROOT / "data" / "churn_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
HISTORY_PATH = PROJECT_ROOT / "models" / "training_history.json"

EMPTY_MODEL_METADATA: dict = {
    "trained_at": None,
    "metrics": None,
    "model_type": None,
    "hyperparameters": {},
}
