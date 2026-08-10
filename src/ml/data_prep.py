"""Подготовка и разбиение churn-данных."""

from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from src.core.exceptions import DataPreparationError, EmptyDatasetError
from src.ml.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)


def get_class_distribution(target: pd.Series) -> dict[str, int]:
    distribution = target.value_counts().sort_index().to_dict()
    return {
        str(churn_class): int(count)
        for churn_class, count in distribution.items()
    }


def split_churn_data(
    dataframe: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Стратифицированное разбиение на train/test."""

    if dataframe.empty:
        raise EmptyDatasetError("Тренировочный датасет пуст.")

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [
        column for column in required_columns if column not in dataframe.columns
    ]
    if missing_columns:
        raise DataPreparationError(
            "Датасет имеет неправильную структуру.",
            details={"missing_columns": missing_columns},
        )

    data = dataframe[required_columns].copy()
    data = data.dropna(subset=[TARGET_COLUMN])

    if data.empty:
        raise EmptyDatasetError(
            "После удаления строк без churn датасет пуст."
        )

    if data[TARGET_COLUMN].nunique() < 2:
        raise DataPreparationError(
            "Для обучения необходимы классы churn 0 и 1."
        )

    return train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=data[TARGET_COLUMN],
    )


def prepare_churn_data(
    dataframe: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Готовит X/y с простым импутом для обзора split-info."""

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [
        column for column in required_columns if column not in dataframe.columns
    ]
    if missing_columns:
        raise DataPreparationError(
            "Датасет имеет неправильную структуру.",
            details={"missing_columns": missing_columns},
        )

    data = dataframe.dropna(subset=[TARGET_COLUMN]).copy()
    X = data[FEATURE_COLUMNS].copy()
    y = data[TARGET_COLUMN].astype(int)

    if y.nunique() < 2:
        raise DataPreparationError(
            "Для обучения необходимы оба класса churn: 0 и 1."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    numeric_imputer = SimpleImputer(strategy="median")
    categorical_imputer = SimpleImputer(strategy="most_frequent")

    X_train[NUMERIC_FEATURES] = numeric_imputer.fit_transform(
        X_train[NUMERIC_FEATURES]
    )
    X_test[NUMERIC_FEATURES] = numeric_imputer.transform(
        X_test[NUMERIC_FEATURES]
    )
    X_train[CATEGORICAL_FEATURES] = categorical_imputer.fit_transform(
        X_train[CATEGORICAL_FEATURES]
    )
    X_test[CATEGORICAL_FEATURES] = categorical_imputer.transform(
        X_test[CATEGORICAL_FEATURES]
    )

    return X_train, X_test, y_train, y_test
