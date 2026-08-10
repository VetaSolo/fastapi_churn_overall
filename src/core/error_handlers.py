"""Глобальные обработчики ошибок FastAPI."""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.core.exceptions import ChurnServiceError

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    """Регистрирует единый JSON-формат ошибок."""

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        details = []
        for error in exc.errors():
            location = [
                str(item) for item in error["loc"] if item != "body"
            ]
            details.append(
                {
                    "field": ".".join(location),
                    "message": error["msg"],
                    "type": error["type"],
                }
            )

        logger.warning(
            "Ошибка валидации %s %s: %s",
            request.method,
            request.url.path,
            details,
        )
        return JSONResponse(
            status_code=422,
            content={
                "code": "VALIDATION_ERROR",
                "message": "Ошибка валидации входных данных.",
                "details": details,
            },
        )

    @app.exception_handler(ChurnServiceError)
    async def churn_service_exception_handler(
        request: Request,
        exc: ChurnServiceError,
    ) -> JSONResponse:
        logger.error(
            "Ошибка сервиса %s %s [%s]: %s | details=%s",
            request.method,
            request.url.path,
            exc.code,
            exc.message,
            exc.details,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request,
        exc: HTTPException,
    ) -> JSONResponse:
        logger.error(
            "HTTPException %s %s: %s",
            request.method,
            request.url.path,
            exc.detail,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "code": "HTTP_ERROR",
                "message": str(exc.detail),
                "details": None,
            },
        )

    @app.exception_handler(Exception)
    async def unexpected_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        logger.exception(
            "Неожиданная ошибка %s %s",
            request.method,
            request.url.path,
        )
        return JSONResponse(
            status_code=500,
            content={
                "code": "INTERNAL_SERVER_ERROR",
                "message": "Внутренняя ошибка сервиса.",
                "details": None,
            },
        )
