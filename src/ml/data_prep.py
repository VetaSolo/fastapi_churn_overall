"""Подготовка и разбиение churn-данных."""

from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from src.core.exceptions import DataPreparationError, EmptyDatasetError
from src.ml.features import (
    ALLOWED_CATEGORICAL_VALUES,
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


def impute_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Обрабатывает пропуски до строгой валидации строк.

    - строки без целевой метки churn удаляются
    - числовые признаки: median
    - категориальные: приведение к lower + most_frequent
    """
    data = dataframe.copy()
    data = data.dropna(subset=[TARGET_COLUMN])

    if data.empty:
        raise EmptyDatasetError(
            "После удаления строк без churn датасет пуст."
        )

    for column in NUMERIC_FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        if data[column].isna().any():
            median_value = data[column].median()
            if pd.isna(median_value):
                raise DataPreparationError(
                    f"Не удалось заполнить пропуски в колонке {column}.",
                )
            data[column] = data[column].fillna(median_value)

    for column in CATEGORICAL_FEATURES:
        data[column] = (
            data[column]
            .astype("string")
            .str.strip()
            .str.lower()
            .replace({"": pd.NA, "nan": pd.NA, "none": pd.NA, "<na>": pd.NA})
        )
        if data[column].isna().any():
            mode_values = data[column].mode(dropna=True)
            if mode_values.empty:
                fill_value = ALLOWED_CATEGORICAL_VALUES[column][0]
            else:
                fill_value = mode_values.iloc[0]
            data[column] = data[column].fillna(fill_value)

        allowed = set(ALLOWED_CATEGORICAL_VALUES[column])
        invalid_mask = ~data[column].isin(allowed)
        if invalid_mask.any():
            invalid_values = sorted(data.loc[invalid_mask, column].unique().tolist())
            raise DataPreparationError(
                f"Недопустимые значения в колонке {column}.",
                details={
                    "column": column,
                    "invalid_values": invalid_values,
                    "allowed_values": list(ALLOWED_CATEGORICAL_VALUES[column]),
                },
            )

    data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    data = data.dropna(subset=[TARGET_COLUMN])
    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(int)

    return data


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
    """Готовит X/y с импутом для обзора split-info."""

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [
        column for column in required_columns if column not in dataframe.columns
    ]
    if missing_columns:
        raise DataPreparationError(
            "Датасет имеет неправильную структуру.",
            details={"missing_columns": missing_columns},
        )

    data = impute_missing_values(dataframe[required_columns])
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

    # Повторный imputer в split-info безопасен: после impute_missing_values
    # пропусков уже нет, но слой сохраняет единый preprocessing-контракт.
    numeric_imputer = SimpleImputer(strategy="median")
    categorical_imputer = SimpleImputer(strategy="most_frequent")

    X_train[NUMERIC_FEATURES] = numeric_imputer.fit_transform(
        X_train[NUMERIC_FEATURES]
    )
    X_test[NUMERIC_FEATURES] = numeric_imputer.transform(X_test[NUMERIC_FEATURES])
    X_train[CATEGORICAL_FEATURES] = categorical_imputer.fit_transform(
        X_train[CATEGORICAL_FEATURES]
    )
    X_test[CATEGORICAL_FEATURES] = categorical_imputer.transform(
        X_test[CATEGORICAL_FEATURES]
    )

    return X_train, X_test, y_train, y_test
