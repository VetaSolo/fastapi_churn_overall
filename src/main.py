"""Точка входа FastAPI-приложения для предсказания churn."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from src.api.routes import router
from src.core.config import DATASET_PATH, EMPTY_MODEL_METADATA
from src.core.error_handlers import register_exception_handlers
from src.core.exceptions import ChurnServiceError
from src.core.logging import setup_logging
from src.ml.dataset import ChurnDataset
from src.ml.persistence import load_churn_model

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FastAPI Churn Overall",
    description="API для предсказания оттока клиентов",
    version="0.1.0",
)
register_exception_handlers(app)
app.include_router(router)

try:
    saved_model = load_churn_model()
except Exception as error:
    logger.exception("Не удалось загрузить сохранённую модель: %s", error)
    saved_model = None

if saved_model is not None:
    app.state.churn_model = saved_model["model"]
    app.state.churn_model_metadata = {
        "trained_at": saved_model["trained_at"],
        "metrics": saved_model["metrics"],
        "model_type": saved_model["model_type"],
        "hyperparameters": saved_model["hyperparameters"],
    }
    logger.info(
        "Модель загружена: type=%s trained_at=%s",
        saved_model["model_type"],
        saved_model["trained_at"],
    )
else:
    app.state.churn_model = None
    app.state.churn_model_metadata = dict(EMPTY_MODEL_METADATA)
    logger.warning("Сохранённая модель не найдена — сервис стартует без модели")

try:
    app.state.dataset = ChurnDataset(DATASET_PATH)
    logger.info(
        "Датасет загружен: path=%s rows=%s columns=%s",
        DATASET_PATH,
        app.state.dataset.dataframe.shape[0],
        app.state.dataset.dataframe.shape[1],
    )
except (FileNotFoundError, ValueError, ChurnServiceError) as error:
    logger.error("Не удалось загрузить датасет: %s", error)
    app.state.dataset = None
