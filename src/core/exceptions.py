"""Доменные исключения churn-сервиса."""

from __future__ import annotations

from typing import Any


class ChurnServiceError(Exception):
    """Базовая ошибка churn-сервиса."""

    code = "CHURN_SERVICE_ERROR"
    status_code = 400

    def __init__(
        self,
        message: str,
        details: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


class DatasetError(ChurnServiceError):
    code = "DATASET_ERROR"
    status_code = 400


class EmptyDatasetError(DatasetError):
    code = "EMPTY_DATASET"


class DataPreparationError(ChurnServiceError):
    code = "DATA_PREPARATION_ERROR"
    status_code = 400


class ModelNotTrainedError(ChurnServiceError):
    code = "MODEL_NOT_TRAINED"
    status_code = 503


class ModelTrainingError(ChurnServiceError):
    code = "MODEL_TRAINING_ERROR"
    status_code = 400


class PredictionError(ChurnServiceError):
    code = "PREDICTION_ERROR"
    status_code = 422
