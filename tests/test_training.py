"""Unit-тесты обучения и предсказания модели (без FastAPI)."""

from pathlib import Path

import joblib
import pytest
from sklearn.pipeline import Pipeline

from src.core.exceptions import ModelTrainingError, PredictionError
from src.ml.data_prep import split_churn_data
from src.ml.pipeline import (
    evaluate_churn_model,
    predict_churn,
    train_churn_model,
)
from src.ml.persistence import load_churn_model, save_churn_model
from src.schemas.churn import FeatureVectorChurn, TrainingConfigChurn


def test_train_logreg_returns_fitted_pipeline(synthetic_dataframe, train_config):
    train_data, _ = split_churn_data(synthetic_dataframe, random_state=42)
    model = train_churn_model(train_data, train_config)

    assert isinstance(model, Pipeline)
    assert "preprocessor" in model.named_steps
    assert "classifier" in model.named_steps
    assert hasattr(model, "feature_names_in_")


def test_train_random_forest(synthetic_dataframe):
    train_data, _ = split_churn_data(synthetic_dataframe, random_state=42)
    config = TrainingConfigChurn(
        model_type="random_forest",
        hyperparameters={"n_estimators": 20, "random_state": 42},
    )
    model = train_churn_model(train_data, config)
    assert model.named_steps["classifier"].__class__.__name__ == (
        "RandomForestClassifier"
    )


def test_train_is_reproducible(synthetic_dataframe, train_config):
    train_data, _ = split_churn_data(synthetic_dataframe, random_state=42)
    model_a = train_churn_model(train_data, train_config)
    model_b = train_churn_model(train_data, train_config)

    client = FeatureVectorChurn(
        monthly_fee=50.0,
        usage_hours=30.0,
        support_requests=2,
        account_age_months=10,
        failed_payments=1,
        region="Europe",
        device_type="mobile",
        payment_method="card",
        autopay_enabled=0,
    )
    pred_a = predict_churn(model_a, [client])
    pred_b = predict_churn(model_b, [client])
    assert pred_a[0].churn == pred_b[0].churn
    assert pred_a[0].probabilities == pred_b[0].probabilities


def test_evaluate_churn_model_metrics(synthetic_dataframe, train_config):
    train_data, test_data = split_churn_data(synthetic_dataframe, random_state=42)
    model = train_churn_model(train_data, train_config)
    metrics = evaluate_churn_model(model, test_data)

    assert set(metrics) == {"accuracy", "f1", "roc_auc"}
    for value in metrics.values():
        assert 0.0 <= value <= 1.0


def test_save_and_load_churn_model(
    synthetic_dataframe,
    train_config,
    tmp_path: Path,
):
    train_data, test_data = split_churn_data(synthetic_dataframe, random_state=42)
    model = train_churn_model(train_data, train_config)
    metrics = evaluate_churn_model(model, test_data)
    path = tmp_path / "model.joblib"

    metadata = save_churn_model(
        model=model,
        metrics=metrics,
        model_type=train_config.model_type,
        hyperparameters=train_config.hyperparameters,
        file_path=path,
    )
    assert path.exists()
    assert metadata["model_type"] == "logreg"
    assert metadata["metrics"] == metrics

    loaded = load_churn_model(path)
    assert loaded is not None
    assert loaded["model_type"] == "logreg"
    assert isinstance(loaded["model"], Pipeline)


def test_load_missing_model_returns_none(tmp_path: Path):
    assert load_churn_model(tmp_path / "missing.joblib") is None


def test_predict_churn_single_and_batch(synthetic_dataframe, train_config):
    train_data, _ = split_churn_data(synthetic_dataframe, random_state=42)
    model = train_churn_model(train_data, train_config)
    clients = [
        FeatureVectorChurn(
            monthly_fee=25.0,
            usage_hours=100.0,
            support_requests=0,
            account_age_months=24,
            failed_payments=0,
            region="Europe",
            device_type="desktop",
            payment_method="card",
            autopay_enabled=1,
        ),
        FeatureVectorChurn(
            monthly_fee=90.0,
            usage_hours=8.0,
            support_requests=6,
            account_age_months=2,
            failed_payments=3,
            region="America",
            device_type="mobile",
            payment_method="paypal",
            autopay_enabled=0,
        ),
    ]
    results = predict_churn(model, clients)
    assert len(results) == 2
    for result in results:
        assert result.churn in (0, 1)
        assert set(result.probabilities) == {"0", "1"}
        assert abs(sum(result.probabilities.values()) - 1.0) < 1e-6


def test_predict_churn_empty_list_raises(synthetic_dataframe, train_config):
    train_data, _ = split_churn_data(synthetic_dataframe, random_state=42)
    model = train_churn_model(train_data, train_config)
    with pytest.raises(PredictionError):
        predict_churn(model, [])


def test_invalid_hyperparameters_raise(synthetic_dataframe):
    train_data, _ = split_churn_data(synthetic_dataframe, random_state=42)
    config = TrainingConfigChurn(
        model_type="logreg",
        hyperparameters={"max_iter": "not-an-int"},
    )
    with pytest.raises(ModelTrainingError):
        train_churn_model(train_data, config)


def test_saved_bundle_structure(
    synthetic_dataframe,
    train_config,
    tmp_path: Path,
):
    train_data, test_data = split_churn_data(synthetic_dataframe, random_state=42)
    model = train_churn_model(train_data, train_config)
    metrics = evaluate_churn_model(model, test_data)
    path = tmp_path / "bundle.joblib"

    save_churn_model(
        model=model,
        metrics=metrics,
        model_type="logreg",
        hyperparameters={"random_state": 42},
        file_path=path,
    )
    bundle = joblib.load(path)
    assert {"model", "trained_at", "metrics", "model_type", "hyperparameters"} <= set(
        bundle
    )
