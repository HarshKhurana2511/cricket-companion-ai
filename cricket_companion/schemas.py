from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field


class ErrorCode(str, Enum):
    INVALID_INPUT = "INVALID_INPUT"
    TIMEOUT = "TIMEOUT"
    UPSTREAM_ERROR = "UPSTREAM_ERROR"
    UPSTREAM_BLOCKED = "UPSTREAM_BLOCKED"
    NOT_FOUND = "NOT_FOUND"
    INTERNAL = "INTERNAL"


class Citation(BaseModel):
    url: str
    fetched_at: datetime
    title: str | None = None


class CacheInfo(BaseModel):
    hit: bool
    key: str | None = None
    expires_at: datetime | None = None


class ToolMeta(BaseModel):
    request_id: str | None = None
    elapsed_ms: int | None = None
    cache: CacheInfo | None = None
    source_ids: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)


class ToolError(BaseModel):
    code: ErrorCode
    message: str
    details: Any | None = None


T = TypeVar("T")


class ToolResponse(BaseModel, Generic[T]):
    ok: bool
    data: T | None = None
    error: ToolError | None = None
    meta: ToolMeta = Field(default_factory=ToolMeta)

    @staticmethod
    def success(data: T, *, meta: ToolMeta | None = None) -> "ToolResponse[T]":
        return ToolResponse(ok=True, data=data, error=None, meta=meta or ToolMeta())

    @staticmethod
    def failure(
        code: ErrorCode,
        message: str,
        *,
        details: Any | None = None,
        meta: ToolMeta | None = None,
    ) -> "ToolResponse[T]":
        return ToolResponse(
            ok=False,
            data=None,
            error=ToolError(code=code, message=message, details=details),
            meta=meta or ToolMeta(),
        )


class WebFetchRequest(BaseModel):
    url: str
    mode: Literal["scorecard", "match"] = "scorecard"


class WebFetchResult(BaseModel):
    url: str
    fetched_at: datetime
    payload: dict[str, Any] = Field(default_factory=dict)
    citations: list[Citation] = Field(default_factory=list)

