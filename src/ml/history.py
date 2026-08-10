"""История обучений churn-модели."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.core.config import HISTORY_PATH


def load_training_history(
    file_path: Path | None = None,
) -> list[dict[str, Any]]:
    if file_path is None:
        file_path = HISTORY_PATH

    if not file_path.exists():
        return []

    try:
        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError:
        return []

    return data if isinstance(data, list) else []


def save_training_record(
    record: dict[str, Any],
    file_path: Path | None = None,
) -> None:
    if file_path is None:
        file_path = HISTORY_PATH

    file_path.parent.mkdir(parents=True, exist_ok=True)
    history = load_training_history(file_path)
    history.append(record)

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)


def get_training_history(
    model_type: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    history = load_training_history()
    if model_type is not None:
        history = [
            record
            for record in history
            if record.get("model_type") == model_type
        ]
    return list(reversed(history[-limit:]))
