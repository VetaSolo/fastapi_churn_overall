"""Константы признаков churn-модели."""

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
