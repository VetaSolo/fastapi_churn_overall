"""Интеграционные тесты churn pipeline через FastAPI TestClient."""


def test_health(api_client):
    response = api_client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "ml churn service is running"
    }


def test_operational_health_before_train(api_client):
    response = api_client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["dataset_loaded"] is True
    assert body["dataset_rows"] == 80
    assert body["model_available"] is False
    assert body["status"] == "degraded"


def test_operational_health_after_train(api_client):
    api_client.post(
        "/model/train",
        json={
            "model_type": "logreg",
            "hyperparameters": {"random_state": 42},
        },
    )

    response = api_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["model_available"] is True
    assert body["dataset_loaded"] is True
    assert body["status"] == "ok"
    assert body["model_type"] == "logreg"


def test_full_pipeline_train_status_predict(
    api_client,
    sample_client_payload,
    isolated_model_paths,
):
    """
    Сценарий:
    1) обучение на синтетическом CSV
    2) статус модели
    3) предсказание
    """
    train_response = api_client.post(
        "/model/train",
        json={
            "model_type": "logreg",
            "hyperparameters": {
                "max_iter": 500,
                "random_state": 42,
            },
        },
    )

    assert train_response.status_code == 200
    train_body = train_response.json()
    assert train_body["message"] == (
        "Модель успешно обучена и сохранена"
    )
    assert train_body["model_type"] == "logreg"
    assert train_body["train_rows"] + train_body["test_rows"] == 80
    assert set(train_body["metrics"]) == {
        "accuracy",
        "f1",
        "roc_auc",
    }
    assert isolated_model_paths["model_path"].exists()

    status_response = api_client.get("/model/status")
    assert status_response.status_code == 200
    status_body = status_response.json()
    assert status_body["trained"] is True
    assert status_body["model_type"] == "logreg"
    assert status_body["trained_at"] is not None
    assert status_body["metrics"] is not None

    predict_response = api_client.post(
        "/predict",
        json=sample_client_payload,
    )
    assert predict_response.status_code == 200
    prediction = predict_response.json()
    assert prediction["churn"] in (0, 1)
    assert set(prediction["probabilities"]) == {"0", "1"}
    assert abs(
        sum(prediction["probabilities"].values()) - 1.0
    ) < 1e-6


def test_predict_batch_after_train(api_client, sample_client_payload):
    api_client.post(
        "/model/train",
        json={
            "model_type": "random_forest",
            "hyperparameters": {
                "n_estimators": 25,
                "random_state": 42,
            },
        },
    )

    payload = [
        sample_client_payload,
        {
            **sample_client_payload,
            "monthly_fee": 19.99,
            "usage_hours": 140.0,
            "support_requests": 0,
            "failed_payments": 0,
            "autopay_enabled": 1,
        },
    ]

    response = api_client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert len(body) == 2


def test_dataset_endpoints_use_synthetic_data(api_client):
    info = api_client.get("/dataset/info")
    assert info.status_code == 200
    assert info.json()["rows"] == 80

    preview = api_client.get("/dataset/preview?n=5")
    assert preview.status_code == 200
    assert len(preview.json()) == 5

    split_info = api_client.get("/dataset/split-info")
    assert split_info.status_code == 200
    body = split_info.json()
    assert body["train"]["rows"] == 64
    assert body["test"]["rows"] == 16


def test_model_metrics_history_after_train(api_client):
    api_client.post(
        "/model/train",
        json={
            "model_type": "logreg",
            "hyperparameters": {"random_state": 42},
        },
    )

    response = api_client.get("/model/metrics?limit=5")
    assert response.status_code == 200
    body = response.json()
    assert body["count"] >= 1
    assert body["latest"]["model_type"] == "logreg"
    assert "metrics" in body["latest"]


def test_training_is_reproducible_via_api(api_client):
    config = {
        "model_type": "logreg",
        "hyperparameters": {
            "max_iter": 500,
            "random_state": 42,
        },
    }

    first = api_client.post("/model/train", json=config).json()
    second = api_client.post("/model/train", json=config).json()

    assert first["metrics"] == second["metrics"]
