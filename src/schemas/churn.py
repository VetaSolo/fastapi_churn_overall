"""Схемы запроса и ответа churn API."""

from __future__ import annotations

from math import isnan
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


def _normalize_categorical(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and isnan(value):
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "nan", "none", "<na>"}:
            return None
        return normalized
    return value


def _nan_to_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and isnan(value):
        return None
    try:
        import pandas as pd

        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


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
        return _normalize_categorical(value)


class DatasetRowChurn(BaseModel):
    """Строка тренировочного датасета; признаки могут содержать пропуски."""

    model_config = ConfigDict(extra="forbid")

    monthly_fee: float | None = None
    usage_hours: float | None = None
    support_requests: int | None = None
    account_age_months: int | None = None
    failed_payments: int | None = None
    region: Region | None = None
    device_type: DeviceType | None = None
    payment_method: PaymentMethod | None = None
    autopay_enabled: int | None = Field(default=None, ge=0, le=1)
    churn: int = Field(..., ge=0, le=1, description="Целевая метка оттока")

    @field_validator(
        "monthly_fee",
        "usage_hours",
        "support_requests",
        "account_age_months",
        "failed_payments",
        "autopay_enabled",
        mode="before",
    )
    @classmethod
    def coerce_numeric_nan(cls, value: Any) -> Any:
        return _nan_to_none(value)

    @field_validator("region", "device_type", "payment_method", mode="before")
    @classmethod
    def normalize_optional_categorical(cls, value: Any) -> Any:
        return _normalize_categorical(value)


class PredictionResponseChurn(BaseModel):
    churn: int = Field(..., ge=0, le=1)
    probabilities: dict[str, float]


class TrainingConfigChurn(BaseModel):
    model_type: Literal["logreg", "random_forest"]
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
