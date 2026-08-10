"""Обучение, оценка и предсказание churn pipeline."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.core.exceptions import ModelTrainingError, PredictionError
from src.ml.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from src.schemas.churn import (
    FeatureVectorChurn,
    PredictionResponseChurn,
    TrainingConfigChurn,
)


def create_churn_classifier(config: TrainingConfigChurn):
    hyperparameters = config.hyperparameters.copy()

    try:
        if config.model_type == "logreg":
            hyperparameters.setdefault("max_iter", 1000)
            hyperparameters.setdefault("random_state", 42)
            return LogisticRegression(**hyperparameters)

        if config.model_type == "random_forest":
            hyperparameters.setdefault("random_state", 42)
            return RandomForestClassifier(**hyperparameters)
    except (TypeError, ValueError) as error:
        raise ModelTrainingError(
            "Некорректные гиперпараметры модели.",
            details={
                "model_type": config.model_type,
                "reason": str(error),
            },
        ) from error

    raise ModelTrainingError(
        f"Неизвестный тип модели: {config.model_type}"
    )


def build_churn_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_churn_pipeline(config: TrainingConfigChurn) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_churn_preprocessor()),
            ("classifier", create_churn_classifier(config)),
        ]
    )


def train_churn_model(
    train_dataframe: pd.DataFrame,
    config: TrainingConfigChurn,
) -> Pipeline:
    if train_dataframe.empty:
        raise ModelTrainingError("Тренировочный датасет пуст.")

    X_train = train_dataframe[FEATURE_COLUMNS].copy()
    y_train = train_dataframe[TARGET_COLUMN].astype(int)
    pipeline = build_churn_pipeline(config)

    try:
        pipeline.fit(X_train, y_train)
    except (TypeError, ValueError) as error:
        raise ModelTrainingError(
            "Некорректные гиперпараметры модели.",
            details={
                "model_type": config.model_type,
                "reason": str(error),
            },
        ) from error

    return pipeline


def evaluate_churn_model(
    model: Pipeline,
    test_dataframe: pd.DataFrame,
) -> dict[str, float]:
    X_test = test_dataframe[FEATURE_COLUMNS].copy()
    y_test = test_dataframe[TARGET_COLUMN].astype(int)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    classes = list(model.named_steps["classifier"].classes_)

    try:
        positive_class_index = classes.index(1)
    except ValueError as error:
        raise ValueError("Модель не содержит churn-класс 1.") from error

    churn_probabilities = probabilities[:, positive_class_index]
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, churn_probabilities)),
    }


def predict_churn(
    model: Pipeline,
    clients: list[FeatureVectorChurn],
) -> list[PredictionResponseChurn]:
    if not clients:
        raise PredictionError("Нужно передать хотя бы одного клиента.")

    dataframe = pd.DataFrame([client.model_dump() for client in clients])
    expected_features = list(
        getattr(model, "feature_names_in_", FEATURE_COLUMNS)
    )
    missing_features = [
        feature
        for feature in expected_features
        if feature not in dataframe.columns
    ]
    if missing_features:
        raise PredictionError(
            "Набор признаков не соответствует модели.",
            details={
                "missing_features": missing_features,
                "expected_features": expected_features,
            },
        )

    dataframe = dataframe.reindex(columns=expected_features)

    try:
        predicted_classes = model.predict(dataframe)
        predicted_probabilities = model.predict_proba(dataframe)
    except (TypeError, ValueError) as error:
        raise PredictionError(
            "Не удалось выполнить предсказание.",
            details={"reason": str(error)},
        ) from error

    classes = model.named_steps["classifier"].classes_
    results: list[PredictionResponseChurn] = []

    for predicted_class, probabilities in zip(
        predicted_classes,
        predicted_probabilities,
    ):
        results.append(
            PredictionResponseChurn(
                churn=int(predicted_class),
                probabilities={
                    str(int(class_value)): float(probability)
                    for class_value, probability in zip(classes, probabilities)
                },
            )
        )

    return results
