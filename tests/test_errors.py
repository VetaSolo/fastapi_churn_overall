"""Тесты корректной обработки ошибок API."""


def test_predict_without_trained_model_returns_503(
    api_client,
    sample_client_payload,
):
    response = api_client.post(
        "/predict",
        json=sample_client_payload,
    )

    assert response.status_code == 503
    body = response.json()
    assert body["code"] == "MODEL_NOT_TRAINED"
    assert "не обучена" in body["message"].lower()


def test_predict_validation_error_returns_422(api_client):
    # Сначала обучаем, чтобы ошибка была именно валидацией, а не 503
    api_client.post(
        "/model/train",
        json={
            "model_type": "logreg",
            "hyperparameters": {"random_state": 42},
        },
    )

    response = api_client.post(
        "/predict",
        json={
            "monthly_fee": "not-a-number",
            "usage_hours": 10,
            "support_requests": 1,
            "account_age_months": 3,
            "failed_payments": 0,
            "region": "Europe",
            "device_type": "mobile",
            "payment_method": "card",
            "autopay_enabled": 1,
        },
    )

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "VALIDATION_ERROR"
    assert isinstance(body["details"], list)
    assert body["details"]


def test_predict_missing_fields_returns_422(api_client):
    api_client.post(
        "/model/train",
        json={
            "model_type": "logreg",
            "hyperparameters": {"random_state": 42},
        },
    )

    response = api_client.post(
        "/predict",
        json={"monthly_fee": 10.0},
    )

    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_ERROR"


def test_predict_extra_fields_forbidden(api_client, sample_client_payload):
    api_client.post(
        "/model/train",
        json={
            "model_type": "logreg",
            "hyperparameters": {"random_state": 42},
        },
    )

    payload = {
        **sample_client_payload,
        "unknown_feature": 123,
    }
    response = api_client.post("/predict", json=payload)

    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_ERROR"


def test_train_invalid_model_type_returns_422(api_client):
    response = api_client.post(
        "/model/train",
        json={
            "model_type": "xgboost",
            "hyperparameters": {},
        },
    )

    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_ERROR"


def test_train_invalid_hyperparameters_returns_error(api_client):
    response = api_client.post(
        "/model/train",
        json={
            "model_type": "logreg",
            "hyperparameters": {"max_iter": "bad"},
        },
    )

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "MODEL_TRAINING_ERROR"


def test_model_status_when_not_trained(api_client):
    response = api_client.get("/model/status")

    assert response.status_code == 200
    body = response.json()
    assert body["trained"] is False
    assert body["model_type"] is None
    assert body["metrics"] is None


def test_dataset_preview_limit_validation(api_client):
    response = api_client.get("/dataset/preview?n=0")

    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_ERROR"
