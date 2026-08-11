"""HTTP-эндпоинты churn API."""

from __future__ import annotations

import logging
from typing import Annotated, Literal, Union

from fastapi import APIRouter, Body, HTTPException, Query, Request

from src.core.exceptions import (
    DatasetError,
    EmptyDatasetError,
    ModelNotTrainedError,
)
from src.ml.data_prep import (
    get_class_distribution,
    prepare_churn_data,
    split_churn_data,
)
from src.ml.features import (
    ALLOWED_CATEGORICAL_VALUES,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    FEATURE_TYPES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from src.ml.history import get_training_history, save_training_record
from src.ml.pipeline import evaluate_churn_model, predict_churn, train_churn_model
from src.ml.persistence import save_churn_model
from src.schemas.churn import (
    DatasetRowChurn,
    ErrorResponse,
    FeatureVectorChurn,
    HealthResponse,
    PredictionResponseChurn,
    TrainingConfigChurn,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _dataset_is_loaded(request: Request) -> bool:
    dataset = getattr(request.app.state, "dataset", None)
    return (
        dataset is not None
        and dataset.dataframe is not None
        and not dataset.dataframe.empty
    )


def _require_dataset(request: Request):
    if not _dataset_is_loaded(request):
        raise DatasetError("Датасет не загружен.")
    return request.app.state.dataset


@router.get("/")
def root() -> dict[str, str]:
    return {"message": "ml churn service is running"}


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    model_available = request.app.state.churn_model is not None
    dataset_loaded = _dataset_is_loaded(request)
    metadata = request.app.state.churn_model_metadata

    status: Literal["ok", "degraded"] = (
        "ok" if model_available and dataset_loaded else "degraded"
    )
    dataset_rows = (
        int(request.app.state.dataset.dataframe.shape[0])
        if dataset_loaded
        else None
    )

    return HealthResponse(
        status=status,
        model_available=model_available,
        dataset_loaded=dataset_loaded,
        model_type=metadata.get("model_type"),
        dataset_rows=dataset_rows,
    )


@router.post(
    "/predict",
    response_model=PredictionResponseChurn | list[PredictionResponseChurn],
    responses={
        422: {"model": ErrorResponse, "description": "Некорректные входные данные"},
        503: {"model": ErrorResponse, "description": "Модель ещё не обучена"},
    },
)
def predict(
    request: Request,
    clients: Annotated[
        Union[FeatureVectorChurn, list[FeatureVectorChurn]],
        Body(
            openapi_examples={
                "single_client": {
                    "summary": "Один клиент",
                    "value": {
                        "monthly_fee": 79.99,
                        "usage_hours": 25.5,
                        "support_requests": 4,
                        "account_age_months": 6,
                        "failed_payments": 2,
                        "region": "europe",
                        "device_type": "mobile",
                        "payment_method": "card",
                        "autopay_enabled": 0,
                    },
                },
                "client_list": {
                    "summary": "Несколько клиентов",
                    "value": [
                        {
                            "monthly_fee": 29.99,
                            "usage_hours": 150.0,
                            "support_requests": 0,
                            "account_age_months": 36,
                            "failed_payments": 0,
                            "region": "europe",
                            "device_type": "desktop",
                            "payment_method": "card",
                            "autopay_enabled": 1,
                        },
                        {
                            "monthly_fee": 99.99,
                            "usage_hours": 8.0,
                            "support_requests": 7,
                            "account_age_months": 2,
                            "failed_payments": 3,
                            "region": "america",
                            "device_type": "mobile",
                            "payment_method": "card",
                            "autopay_enabled": 0,
                        },
                    ],
                },
            }
        ),
    ],
) -> PredictionResponseChurn | list[PredictionResponseChurn]:
    model = request.app.state.churn_model
    if model is None:
        raise ModelNotTrainedError(
            "Churn модель ещё не обучена. "
            "Сначала вызовите POST /model/train."
        )

    is_single_client = isinstance(clients, FeatureVectorChurn)
    client_list = [clients] if is_single_client else clients
    logger.info("Запрос /predict: clients=%s", len(client_list))

    try:
        predictions = predict_churn(model=model, clients=client_list)
    except ValueError as error:
        logger.error("Ошибка предсказания: %s", error)
        raise HTTPException(status_code=422, detail=str(error)) from error

    logger.info(
        "Предсказание завершено: clients=%s churn_positive=%s",
        len(predictions),
        sum(item.churn for item in predictions),
    )
    return predictions[0] if is_single_client else predictions


@router.get("/dataset/preview", response_model=list[DatasetRowChurn])
def dataset_preview(
    request: Request,
    n: int = Query(default=5, ge=1, le=100),
) -> list[DatasetRowChurn]:
    return _require_dataset(request).preview(n)


@router.get("/dataset/info")
def dataset_info(request: Request) -> dict:
    return _require_dataset(request).info()


@router.get("/dataset/split-info")
def dataset_split_info(request: Request) -> dict:
    dataset = _require_dataset(request)
    X_train, X_test, y_train, y_test = prepare_churn_data(dataset.dataframe)
    return {
        "train": {
            "rows": len(X_train),
            "columns": X_train.shape[1],
            "churn_distribution": get_class_distribution(y_train),
        },
        "test": {
            "rows": len(X_test),
            "columns": X_test.shape[1],
            "churn_distribution": get_class_distribution(y_test),
        },
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }


@router.post(
    "/model/train",
    responses={
        400: {"model": ErrorResponse, "description": "Ошибка датасета или обучения"},
    },
)
def train_model(request: Request, config: TrainingConfigChurn) -> dict:
    dataset = _require_dataset(request)

    if dataset.dataframe.empty:
        raise EmptyDatasetError("Датасет пуст.")

    logger.info(
        "Старт обучения: model_type=%s hyperparameters=%s rows=%s",
        config.model_type,
        config.hyperparameters,
        len(dataset.dataframe),
    )

    train_data, test_data = split_churn_data(dataset.dataframe)
    model = train_churn_model(train_dataframe=train_data, config=config)
    metrics = evaluate_churn_model(model=model, test_dataframe=test_data)
    metadata = save_churn_model(
        model=model,
        metrics=metrics,
        model_type=config.model_type,
        hyperparameters=config.hyperparameters,
    )

    request.app.state.churn_model = model
    request.app.state.churn_model_metadata = metadata
    save_training_record(
        {
            "timestamp": metadata["trained_at"],
            "model_type": config.model_type,
            "hyperparameters": config.hyperparameters,
            "metrics": metrics,
        }
    )

    logger.info(
        "Обучение завершено: model_type=%s metrics=%s trained_at=%s",
        config.model_type,
        metrics,
        metadata["trained_at"],
    )
    return {
        "message": "Модель успешно обучена и сохранена",
        "model_type": config.model_type,
        "hyperparameters": config.hyperparameters,
        "train_rows": len(train_data),
        "test_rows": len(test_data),
        "metrics": metrics,
        "trained_at": metadata["trained_at"],
    }


@router.get("/model/status")
def model_status(request: Request) -> dict:
    metadata = request.app.state.churn_model_metadata
    return {
        "trained": request.app.state.churn_model is not None,
        "model_type": metadata["model_type"],
        "hyperparameters": metadata["hyperparameters"],
        "trained_at": metadata["trained_at"],
        "metrics": metadata["metrics"],
    }


@router.get("/model/metrics")
def model_metrics(
    model_type: Literal["logreg", "random_forest"] | None = None,
    limit: int = Query(default=5, ge=1, le=100),
) -> dict:
    history = get_training_history(model_type=model_type, limit=limit)
    return {
        "latest": history[0] if history else None,
        "history": history,
        "count": len(history),
    }


@router.get("/model/schema")
def model_schema() -> dict:
    features = []
    for feature in FEATURE_COLUMNS:
        feature_info = {
            "name": feature,
            "type": FEATURE_TYPES[feature],
            "category": (
                "numeric" if feature in NUMERIC_FEATURES else "categorical"
            ),
        }
        if feature == "autopay_enabled":
            feature_info["allowed_values"] = [0, 1]
        elif feature in ALLOWED_CATEGORICAL_VALUES:
            feature_info["allowed_values"] = list(
                ALLOWED_CATEGORICAL_VALUES[feature]
            )
        features.append(feature_info)

    return {"target": TARGET_COLUMN, "features": features}
