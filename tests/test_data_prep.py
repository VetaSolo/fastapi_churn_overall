"""Unit-тесты подготовки данных (без FastAPI)."""

from pathlib import Path

import pandas as pd
import pytest

from src.core.exceptions import DataPreparationError, EmptyDatasetError
from src.ml.data_prep import (
    get_class_distribution,
    prepare_churn_data,
    split_churn_data,
)
from src.ml.dataset import ChurnDataset
from src.ml.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)


def test_churn_dataset_loads_and_validates(synthetic_csv):
    dataset = ChurnDataset(synthetic_csv)

    assert len(dataset.dataframe) == 80
    assert set(FEATURE_COLUMNS + [TARGET_COLUMN]).issubset(
        dataset.dataframe.columns
    )
    assert set(dataset.dataframe[TARGET_COLUMN].unique()) == {0, 1}


def test_churn_dataset_preview(churn_dataset):
    preview = churn_dataset.preview(n=3)
    assert len(preview) == 3
    assert all(row.churn in (0, 1) for row in preview)


def test_churn_dataset_info(churn_dataset):
    info = churn_dataset.info()
    assert info["rows"] == 80
    assert info["columns"] == len(FEATURE_COLUMNS) + 1
    assert info["churn_distribution"]["0"] == 40
    assert info["churn_distribution"]["1"] == 40


def test_split_churn_data_is_reproducible(synthetic_dataframe):
    train_a, test_a = split_churn_data(
        synthetic_dataframe, test_size=0.2, random_state=42
    )
    train_b, test_b = split_churn_data(
        synthetic_dataframe, test_size=0.2, random_state=42
    )

    assert len(train_a) == 64
    assert len(test_a) == 16
    pd.testing.assert_frame_equal(
        train_a.reset_index(drop=True),
        train_b.reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        test_a.reset_index(drop=True),
        test_b.reset_index(drop=True),
    )


def test_split_churn_data_stratified(synthetic_dataframe):
    train_data, test_data = split_churn_data(
        synthetic_dataframe, test_size=0.2, random_state=42
    )
    train_dist = get_class_distribution(train_data[TARGET_COLUMN])
    test_dist = get_class_distribution(test_data[TARGET_COLUMN])
    assert train_dist["0"] == train_dist["1"]
    assert test_dist["0"] == test_dist["1"]


def test_split_churn_data_empty_raises():
    empty = pd.DataFrame(columns=FEATURE_COLUMNS + [TARGET_COLUMN])
    with pytest.raises(EmptyDatasetError):
        split_churn_data(empty)


def test_split_churn_data_single_class_raises(synthetic_dataframe):
    single_class = synthetic_dataframe.copy()
    single_class[TARGET_COLUMN] = 0
    with pytest.raises(DataPreparationError):
        split_churn_data(single_class)


def test_prepare_churn_data_shapes_and_features(synthetic_dataframe):
    X_train, X_test, y_train, y_test = prepare_churn_data(
        synthetic_dataframe, test_size=0.2, random_state=42
    )
    assert len(X_train) == len(y_train) == 64
    assert len(X_test) == len(y_test) == 16
    assert list(X_train.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert y_train.isna().sum() == 0
    assert X_train[NUMERIC_FEATURES].isna().sum().sum() == 0
    assert X_train[CATEGORICAL_FEATURES].isna().sum().sum() == 0


def test_prepare_churn_data_is_reproducible(synthetic_dataframe):
    first = prepare_churn_data(synthetic_dataframe, test_size=0.2, random_state=42)
    second = prepare_churn_data(synthetic_dataframe, test_size=0.2, random_state=42)
    pd.testing.assert_frame_equal(
        first[0].reset_index(drop=True),
        second[0].reset_index(drop=True),
    )
    pd.testing.assert_series_equal(
        first[2].reset_index(drop=True),
        second[2].reset_index(drop=True),
    )


def test_get_class_distribution():
    target = pd.Series([0, 0, 1, 1, 1])
    assert get_class_distribution(target) == {"0": 2, "1": 3}


def test_dataset_imputes_missing_values(tmp_path: Path, synthetic_dataframe):
    dataframe = synthetic_dataframe.copy()
    dataframe.loc[0, "support_requests"] = None
    dataframe.loc[1, "region"] = None
    dataframe.loc[2, "device_type"] = ""
    dataframe.loc[3, "payment_method"] = "NAN"
    dataframe.loc[4, "monthly_fee"] = None

    path = tmp_path / "churn_with_na.csv"
    dataframe.to_csv(path, index=False)

    dataset = ChurnDataset(path)

    assert dataset.dataframe["support_requests"].isna().sum() == 0
    assert dataset.dataframe["region"].isna().sum() == 0
    assert dataset.dataframe["device_type"].isna().sum() == 0
    assert dataset.dataframe["payment_method"].isna().sum() == 0
    assert dataset.dataframe["monthly_fee"].isna().sum() == 0
    assert set(dataset.dataframe["region"].unique()).issubset(
        {"africa", "america", "asia", "europe"}
    )


def test_dataset_rejects_invalid_category(tmp_path: Path, synthetic_dataframe):
    dataframe = synthetic_dataframe.copy()
    dataframe.loc[0, "region"] = "atlantis"
    path = tmp_path / "churn_invalid.csv"
    dataframe.to_csv(path, index=False)

    with pytest.raises(DataPreparationError):
        ChurnDataset(path)


def test_feature_vector_normalizes_case():
    from src.schemas.churn import FeatureVectorChurn

    client = FeatureVectorChurn(
        monthly_fee=10.0,
        usage_hours=5.0,
        support_requests=1,
        account_age_months=3,
        failed_payments=0,
        region="Europe",
        device_type="Mobile",
        payment_method="Card",
        autopay_enabled=1,
    )
    assert client.region == "europe"
    assert client.device_type == "mobile"
    assert client.payment_method == "card"
