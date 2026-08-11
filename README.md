# FastAPI Churn Overall

ML-сервис на FastAPI для предсказания оттока клиентов (churn): обучение моделей, сохранение артефактов и online-inference через REST API.

## Цель сервиса

- загружать тренировочный датасет `data/churn_dataset.csv`
- обучать pipeline (`logreg` или `random_forest`) с препроцессингом
- сохранять модель в `models/churn_model.joblib` и историю в `models/training_history.json`
- отдавать предсказания класса churn и вероятностей для одного или нескольких клиентов
- предоставлять эксплуатационные эндпоинты `/health` и `/docs`

## Структура проекта

```
fastapi_churn_overall/
├── data/
│   └── churn_dataset.csv
├── models/
│   └── churn_model.joblib
├── src/
│   ├── main.py              # точка входа FastAPI
│   ├── api/                 # HTTP-роуты
│   ├── ml/                  # dataset, pipeline, persistence, history
│   ├── schemas/             # Pydantic-схемы
│   └── core/                # config, logging, exceptions
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md
```

## Формат датасета `churn_dataset.csv`

CSV с заголовком. Каждая строка — один клиент.

| Колонка | Тип | Описание |
|---------|-----|----------|
| `monthly_fee` | float | Ежемесячный платёж |
| `usage_hours` | float | Часы использования |
| `support_requests` | int | Обращения в поддержку |
| `account_age_months` | int | Возраст аккаунта в месяцах |
| `failed_payments` | int | Неуспешные платежи |
| `region` | str | `africa`, `america`, `asia`, `europe` |
| `device_type` | str | `desktop`, `mobile`, `tablet` |
| `payment_method` | str | `card`, `crypto`, `paypal` |
| `autopay_enabled` | int (`0`/`1`) | Автоплатёж |
| `churn` | int (`0`/`1`) | Целевая метка оттока |

Пример строки:

```csv
monthly_fee,usage_hours,support_requests,account_age_months,failed_payments,region,device_type,payment_method,autopay_enabled,churn
9.99,27.92,1,14,1,america,desktop,card,1,1
```

## Локальный запуск

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn src.main:app --reload
```

Сервис: http://127.0.0.1:8000  
Swagger: http://127.0.0.1:8000/docs  
Health: http://127.0.0.1:8000/health

## Запуск в Docker

```bash
docker build -t fastapi-churn .
docker run --rm -p 8000:8000 --name fastapi-churn-svc fastapi-churn
```

Проверка:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/docs
```

## Примеры запросов

### Обучение модели — `POST /model/train`

```bash
curl -X POST http://127.0.0.1:8000/model/train ^
  -H "Content-Type: application/json" ^
  -d "{\"model_type\":\"logreg\",\"hyperparameters\":{\"max_iter\":1000,\"random_state\":42}}"
```

Пример ответа:

```json
{
  "message": "Модель успешно обучена и сохранена",
  "model_type": "logreg",
  "hyperparameters": {"max_iter": 1000, "random_state": 42},
  "train_rows": 1600,
  "test_rows": 400,
  "metrics": {"accuracy": 0.91, "f1": 0.90, "roc_auc": 0.95},
  "trained_at": "2026-08-10T18:00:00+00:00"
}
```

Для Random Forest:

```json
{
  "model_type": "random_forest",
  "hyperparameters": {"n_estimators": 100, "random_state": 42}
}
```

### Предсказание — `POST /predict`

Один клиент:

```bash
curl -X POST http://127.0.0.1:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"monthly_fee\":79.99,\"usage_hours\":25.5,\"support_requests\":4,\"account_age_months\":6,\"failed_payments\":2,\"region\":\"europe\",\"device_type\":\"mobile\",\"payment_method\":\"card\",\"autopay_enabled\":0}"
```

Категории приводятся к lower-case (`Europe` → `europe`). Неизвестные значения (`atlantis` и т.п.) отклоняются с `422`.
Пропуски в тренировочном CSV заполняются при загрузке: median для числовых, most_frequent для категориальных; строки без `churn` удаляются.
Пример ответа:

```json
{
  "churn": 1,
  "probabilities": {"0": 0.22, "1": 0.78}
}
```

Несколько клиентов — передайте JSON-массив объектов с теми же полями.

### Полезные эндпоинты

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/health` | Статус модели и датасета |
| GET | `/model/status` | Активная модель и метрики |
| GET | `/model/metrics` | История обучений |
| GET | `/model/schema` | Схема признаков |
| GET | `/dataset/info` | Размер и распределение churn |
| GET | `/dataset/preview?n=5` | Превью строк датасета |

## Тесты

```bash
pytest
```

Тесты используют синтетический датасет и временные пути модели, чтобы не перезаписывать `models/` в репозитории.
