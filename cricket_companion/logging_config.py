from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
from datetime import UTC, datetime
from typing import Any


_ctx_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
_ctx_session_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("session_id", default=None)
_ctx_user_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("user_id", default=None)


def set_log_context(*, request_id: str | None = None, session_id: str | None = None, user_id: str | None = None) -> None:
    if request_id is not None:
        _ctx_request_id.set(request_id)
    if session_id is not None:
        _ctx_session_id.set(session_id)
    if user_id is not None:
        _ctx_user_id.set(user_id)


def clear_log_context() -> None:
    _ctx_request_id.set(None)
    _ctx_session_id.set(None)
    _ctx_user_id.set(None)


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover
        # Inject contextvars unless already explicitly provided.
        if not hasattr(record, "request_id"):
            record.request_id = _ctx_request_id.get()
        if not hasattr(record, "session_id"):
            record.session_id = _ctx_session_id.get()
        if not hasattr(record, "user_id"):
            record.user_id = _ctx_user_id.get()
        return True


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover
        ts = datetime.fromtimestamp(record.created, tz=UTC).isoformat().replace("+00:00", "Z")
        payload: dict[str, Any] = {
            "ts": ts,
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # Common context fields.
        for k in ("request_id", "session_id", "user_id"):
            v = getattr(record, k, None)
            if v:
                payload[k] = v

        # Structured extras (best-effort): include any non-standard LogRecord fields.
        standard = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
        }
        for k, v in record.__dict__.items():
            if k in standard or k in {"request_id", "session_id", "user_id"}:
                continue
            try:
                json.dumps(v)  # type: ignore[arg-type]
                payload[k] = v
            except Exception:
                payload[k] = str(v)

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def setup_logging(*, level: str | None = None, log_format: str | None = None) -> None:
    """
    Baseline logging setup for API/UI/MCP servers/pipelines.

    Phase 2.8.1:
    - Structured JSON logs (recommended for Docker)
    - request_id/session_id/user_id injected via contextvars

    Env defaults:
    - CC_LOG_LEVEL (default: INFO)
    - CC_LOG_FORMAT (default: json) one of: json | text
    """
    level_eff = (level or os.environ.get("CC_LOG_LEVEL") or "INFO").upper()
    fmt_eff = (log_format or os.environ.get("CC_LOG_FORMAT") or "json").lower().strip()

    numeric_level = logging.getLevelName(level_eff)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)
    handler.addFilter(_ContextFilter())

    if fmt_eff == "text":
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(request_id)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    else:
        handler.setFormatter(_JsonFormatter())
    root_logger.addHandler(handler)
