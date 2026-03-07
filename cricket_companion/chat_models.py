from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from cricket_companion.schemas import Citation, ErrorCode, ToolError
from cricket_companion.output_models import AssistantOutput, ChartArtifact, TableArtifact


Role = Literal["user", "assistant", "system"]
Route = Literal["basic", "analyst", "sim", "fantasy", "unknown"]
EventType = Literal["chunk", "tool_start", "tool_end", "error", "done"]


def utc_now() -> datetime:
    return datetime.now(UTC)


class Message(BaseModel):
    message_id: str = Field(default_factory=lambda: str(uuid4()))
    role: Role
    content: str
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str = "local-user"
    message: Message
    debug: bool = False
    max_context_messages: int = 30


class ChatError(BaseModel):
    code: ErrorCode
    message: str
    details: Any | None = None


class ToolCallTrace(BaseModel):
    tool_name: str
    started_at: datetime = Field(default_factory=utc_now)
    ended_at: datetime | None = None
    elapsed_ms: int | None = None

    request: Any | None = None
    response: Any | None = None

    cache_hit: bool | None = None
    cache_key: str | None = None

    citations: list[Citation] = Field(default_factory=list)
    error: ToolError | None = None


class StreamEvent(BaseModel):
    type: EventType
    timestamp: datetime = Field(default_factory=utc_now)
    data: dict[str, Any] = Field(default_factory=dict)


class ChatState(BaseModel):
    """
    LangGraph state for a single `/chat` request.

    Keep this JSON-serializable so it can be logged/persisted as needed.
    """

    session_id: str
    request_id: str
    user_id: str = "local-user"

    user_message: Message
    messages: list[Message] = Field(default_factory=list)
    summary: str | None = None
    prefs: dict[str, Any] = Field(default_factory=dict)

    route: Route = "unknown"
    route_reason: str | None = None
    clarifying_question: str | None = None

    tool_plan: dict[str, Any] | None = None

    tool_traces: list[ToolCallTrace] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    tables: list[TableArtifact] = Field(default_factory=list)
    charts: list[ChartArtifact] = Field(default_factory=list)
    assistant_output: AssistantOutput | None = None
    events: list[StreamEvent] = Field(default_factory=list)

    draft_answer: str | None = None
    final_answer: str | None = None

    errors: list[ChatError] = Field(default_factory=list)


class ChatResponse(BaseModel):
    session_id: str
    request_id: str
    assistant_message: Message
    citations: list[Citation] = Field(default_factory=list)
    tables: list[TableArtifact] = Field(default_factory=list)
    charts: list[ChartArtifact] = Field(default_factory=list)

    route: Route = "unknown"
    tool_traces: list[ToolCallTrace] = Field(default_factory=list)
    errors: list[ChatError] = Field(default_factory=list)
