"""Загрузка и описание churn-датасета."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.core.exceptions import DataPreparationError
from src.ml.data_prep import clean_raw_dataframe
from src.ml.features import FEATURE_COLUMNS, TARGET_COLUMN
from src.schemas.churn import DatasetRowChurn, FeatureVectorChurn


class ChurnDataset:
    """Загрузка и валидация CSV без импутации (пропуски обрабатывает pipeline)."""

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self.dataframe = self._load_dataset()

    def _load_dataset(self) -> pd.DataFrame:
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Файл датасета не найден: {self.file_path}"
            )

        dataframe = pd.read_csv(self.file_path)
        required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
        missing_columns = required_columns - set(dataframe.columns)

        if missing_columns:
            raise DataPreparationError(
                "Датасет имеет неправильную структуру.",
                details={"missing_columns": sorted(missing_columns)},
            )

        dataframe = clean_raw_dataframe(dataframe[list(required_columns)])

        validated_rows = [
            DatasetRowChurn.model_validate(row).model_dump()
            for row in dataframe.to_dict(orient="records")
        ]
        return pd.DataFrame(validated_rows)

    def preview(self, n: int = 5) -> list[DatasetRowChurn]:
        rows = self.dataframe.head(n).to_dict(orient="records")
        return [DatasetRowChurn.model_validate(row) for row in rows]

    def info(self) -> dict:
        churn_distribution = (
            self.dataframe["churn"].value_counts().sort_index().to_dict()
        )
        return {
            "rows": int(self.dataframe.shape[0]),
            "columns": int(self.dataframe.shape[1]),
            "features": list(FeatureVectorChurn.model_fields),
            "churn_distribution": {
                str(churn_class): int(count)
                for churn_class, count in churn_distribution.items()
            },
        }
