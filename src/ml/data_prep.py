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


def clean_raw_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Нормализует сырой CSV без импутации.

    Пропуски сохраняются: median/mode считаются только на train
    внутри ML pipeline / prepare_churn_data после split.
    """
    data = dataframe.copy()
    data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    data = data.dropna(subset=[TARGET_COLUMN])

    if data.empty:
        raise EmptyDatasetError(
            "После удаления строк без churn датасет пуст."
        )

    for column in NUMERIC_FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    for column in CATEGORICAL_FEATURES:
        data[column] = (
            data[column]
            .astype("string")
            .str.strip()
            .str.lower()
            .replace({"": pd.NA, "nan": pd.NA, "none": pd.NA, "<na>": pd.NA})
        )
        allowed = set(ALLOWED_CATEGORICAL_VALUES[column])
        non_null = data[column].notna()
        invalid_mask = non_null & ~data[column].isin(allowed)
        if invalid_mask.any():
            invalid_values = sorted(
                data.loc[invalid_mask, column].unique().tolist()
            )
            raise DataPreparationError(
                f"Недопустимые значения в колонке {column}.",
                details={
                    "column": column,
                    "invalid_values": invalid_values,
                    "allowed_values": list(ALLOWED_CATEGORICAL_VALUES[column]),
                },
            )

    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(int)
    return data


def _ensure_stratifiable_target(target: pd.Series) -> None:
    if target.nunique() < 2:
        raise DataPreparationError(
            "Для обучения необходимы классы churn 0 и 1."
        )

    class_counts = target.value_counts().sort_index()
    min_class_count = int(class_counts.min())
    if min_class_count < 2:
        raise DataPreparationError(
            "Недостаточно объектов minority-класса "
            "для стратифицированного разбиения.",
            details={
                "class_counts": {
                    str(label): int(count)
                    for label, count in class_counts.items()
                },
                "min_class_count": min_class_count,
                "required_min_per_class": 2,
            },
        )


def _stratified_frame_split(
    data: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _ensure_stratifiable_target(data[TARGET_COLUMN])
    try:
        return train_test_split(
            data,
            test_size=test_size,
            random_state=random_state,
            stratify=data[TARGET_COLUMN],
        )
    except ValueError as error:
        raise DataPreparationError(
            "Не удалось выполнить стратифицированное разбиение датасета.",
            details={"reason": str(error)},
        ) from error


def split_churn_data(
    dataframe: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Стратифицированное разбиение на train/test без импутации."""

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

    return _stratified_frame_split(
        data,
        test_size=test_size,
        random_state=random_state,
    )


def prepare_churn_data(
    dataframe: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Готовит X/y для split-info.

    Сначала split, затем imputer fit только на train.
    """
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [
        column for column in required_columns if column not in dataframe.columns
    ]
    if missing_columns:
        raise DataPreparationError(
            "Датасет имеет неправильную структуру.",
            details={"missing_columns": missing_columns},
        )

    data = clean_raw_dataframe(dataframe[required_columns])
    train_data, test_data = _stratified_frame_split(
        data,
        test_size=test_size,
        random_state=random_state,
    )

    X_train = train_data[FEATURE_COLUMNS].copy()
    X_test = test_data[FEATURE_COLUMNS].copy()
    y_train = train_data[TARGET_COLUMN].astype(int)
    y_test = test_data[TARGET_COLUMN].astype(int)

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
