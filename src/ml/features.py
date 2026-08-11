"""Константы признаков churn-модели."""

from __future__ import annotations

from typing import Literal

NUMERIC_FEATURES = [
    "monthly_fee",
    "usage_hours",
    "support_requests",
    "account_age_months",
    "failed_payments",
    "autopay_enabled",
]

CATEGORICAL_FEATURES = [
    "region",
    "device_type",
    "payment_method",
]

ALLOWED_REGIONS = ("africa", "america", "asia", "europe")
ALLOWED_DEVICE_TYPES = ("desktop", "mobile", "tablet")
ALLOWED_PAYMENT_METHODS = ("card", "crypto", "paypal")

ALLOWED_CATEGORICAL_VALUES: dict[str, tuple[str, ...]] = {
    "region": ALLOWED_REGIONS,
    "device_type": ALLOWED_DEVICE_TYPES,
    "payment_method": ALLOWED_PAYMENT_METHODS,
}

Region = Literal["africa", "america", "asia", "europe"]
DeviceType = Literal["desktop", "mobile", "tablet"]
PaymentMethod = Literal["card", "crypto", "paypal"]

FEATURE_TYPES = {
    "monthly_fee": "float",
    "usage_hours": "float",
    "support_requests": "int",
    "account_age_months": "int",
    "failed_payments": "int",
    "autopay_enabled": "int",
    "region": "str",
    "device_type": "str",
    "payment_method": "str",
}

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "churn"
