"""Схемы запроса и ответа churn API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.ml.features import DeviceType, PaymentMethod, Region


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: Any | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_available: bool
    dataset_loaded: bool
    model_type: str | None = None
    dataset_rows: int | None = None


class FeatureVectorChurn(BaseModel):
    """Признаки клиента для предсказания churn."""

    model_config = ConfigDict(extra="forbid")

    monthly_fee: float
    usage_hours: float
    support_requests: int
    account_age_months: int
    failed_payments: int
    region: Region
    device_type: DeviceType
    payment_method: PaymentMethod
    autopay_enabled: int = Field(..., ge=0, le=1)

    @field_validator("region", "device_type", "payment_method", mode="before")
    @classmethod
    def normalize_categorical(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().lower()
        return value


class DatasetRowChurn(FeatureVectorChurn):
    """Строка тренировочного датасета."""

    churn: int = Field(..., ge=0, le=1, description="Целевая метка оттока")


class PredictionResponseChurn(BaseModel):
    churn: int = Field(..., ge=0, le=1)
    probabilities: dict[str, float]


class TrainingConfigChurn(BaseModel):
    model_type: Literal["logreg", "random_forest"]
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
