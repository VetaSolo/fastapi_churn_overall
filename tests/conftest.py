"""Общие фикстуры для воспроизводимых churn-тестов."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.ml.dataset import ChurnDataset
from src.schemas.churn import TrainingConfigChurn


def make_synthetic_churn_dataframe(n_per_class: int = 40) -> pd.DataFrame:
    """Синтетический датасет с обоими классами churn."""
    rows: list[dict] = []
    regions = ["Europe", "America", "Asia"]
    devices = ["mobile", "desktop", "tablet"]
    payments = ["card", "paypal", "crypto"]

    for i in range(n_per_class):
        rows.append(
            {
                "monthly_fee": 20.0 + (i % 10),
                "usage_hours": 80.0 + i,
                "support_requests": i % 2,
                "account_age_months": 12 + (i % 24),
                "failed_payments": 0,
                "region": regions[i % len(regions)],
                "device_type": devices[i % len(devices)],
                "payment_method": payments[i % len(payments)],
                "autopay_enabled": 1,
                "churn": 0,
            }
        )
        rows.append(
            {
                "monthly_fee": 70.0 + (i % 15),
                "usage_hours": 5.0 + (i % 8),
                "support_requests": 3 + (i % 5),
                "account_age_months": 1 + (i % 6),
                "failed_payments": 1 + (i % 3),
                "region": regions[i % len(regions)],
                "device_type": devices[i % len(devices)],
                "payment_method": payments[i % len(payments)],
                "autopay_enabled": 0,
                "churn": 1,
            }
        )

    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_dataframe() -> pd.DataFrame:
    return make_synthetic_churn_dataframe(n_per_class=40)


@pytest.fixture
def synthetic_csv(tmp_path: Path, synthetic_dataframe: pd.DataFrame) -> Path:
    path = tmp_path / "churn_synthetic.csv"
    synthetic_dataframe.to_csv(path, index=False)
    return path


@pytest.fixture
def churn_dataset(synthetic_csv: Path) -> ChurnDataset:
    return ChurnDataset(synthetic_csv)


@pytest.fixture
def train_config() -> TrainingConfigChurn:
    return TrainingConfigChurn(
        model_type="logreg",
        hyperparameters={"max_iter": 500, "random_state": 42},
    )


@pytest.fixture
def isolated_model_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    model_path = tmp_path / "churn_model.joblib"
    history_path = tmp_path / "training_history.json"
    monkeypatch.setattr("src.core.config.MODEL_PATH", model_path)
    monkeypatch.setattr("src.ml.persistence.MODEL_PATH", model_path)
    monkeypatch.setattr("src.core.config.HISTORY_PATH", history_path)
    monkeypatch.setattr("src.ml.history.HISTORY_PATH", history_path)
    return {"model_path": model_path, "history_path": history_path}


@pytest.fixture
def api_client(
    churn_dataset: ChurnDataset,
    isolated_model_paths: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    import src.main as main_module

    main_module.app.state.dataset = churn_dataset
    main_module.app.state.churn_model = None
    main_module.app.state.churn_model_metadata = {
        "trained_at": None,
        "metrics": None,
        "model_type": None,
        "hyperparameters": {},
    }

    with TestClient(main_module.app) as client:
        yield client

    main_module.app.state.churn_model = None
    main_module.app.state.churn_model_metadata = {
        "trained_at": None,
        "metrics": None,
        "model_type": None,
        "hyperparameters": {},
    }


@pytest.fixture
def sample_client_payload() -> dict:
    return {
        "monthly_fee": 79.99,
        "usage_hours": 25.5,
        "support_requests": 4,
        "account_age_months": 6,
        "failed_payments": 2,
        "region": "Europe",
        "device_type": "mobile",
        "payment_method": "card",
        "autopay_enabled": 0,
    }
