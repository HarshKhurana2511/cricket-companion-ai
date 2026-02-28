from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from cricket_companion.schemas import Citation


DataType = Literal["string", "int", "float", "bool", "date", "datetime"]
ChartType = Literal["line", "bar", "scatter", "hist"]


def utc_now() -> datetime:
    return datetime.now(UTC)


class ArtifactSource(BaseModel):
    tool_name: str | None = None
    request_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    notes: str | None = None


class TableColumn(BaseModel):
    name: str
    dtype: DataType = "string"
    unit: str | None = None
    description: str | None = None


class TableArtifact(BaseModel):
    table_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str | None = None

    columns: list[TableColumn] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)

    truncated: bool = False
    row_count: int | None = None

    source: ArtifactSource = Field(default_factory=ArtifactSource)


class ChartArtifact(BaseModel):
    chart_id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    chart_type: ChartType

    table_id: str
    x: str
    y: list[str] = Field(default_factory=list)

    color_by: str | None = None
    notes: str | None = None


class AssistantOutput(BaseModel):
    """
    Structured, UI-friendly assistant output.

    This is separate from the raw tool traces; it is what we *want to show* the user.
    """

    answer_text: str
    citations: list[Citation] = Field(default_factory=list)
    tables: list[TableArtifact] = Field(default_factory=list)
    charts: list[ChartArtifact] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)

