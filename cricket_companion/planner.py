from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from cricket_companion.chat_models import ChatState, Route
from cricket_companion.config import Settings, get_settings


ToolName = Literal["retrieval", "web_search", "stats", "sim", "fantasy"]


class PlannedToolCall(BaseModel):
    tool_name: ToolName
    args: dict[str, Any] = Field(default_factory=dict)
    timeout_s: int
    use_cache: bool = True
    ttl_days: int | None = None
    note: str | None = None


class ToolPlan(BaseModel):
    route: Route
    calls: list[PlannedToolCall] = Field(default_factory=list)
    reason: str
    clarifying_question: str | None = None


class _BasicExtraction(BaseModel):
    query: str


class _AnalystExtraction(BaseModel):
    question: str
    format: str | None = None
    player: str | None = None
    team: str | None = None
    since_year: int | None = None
    until_year: int | None = None
    limit: int = Field(default=20, ge=1, le=200)


def _needs_web(text: str) -> bool:
    return bool(
        re.search(
            r"\b(latest|today|yesterday|current|news|injury|availability|squad|playing\s*xi|update)\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def _extract_with_llm(state: ChatState, settings: Settings, *, kind: Literal["basic", "analyst"]) -> dict[str, Any] | None:
    """
    LLM-assisted parameter extraction only (not tool selection).

    Returns a dict matching the expected schema for the given `kind`, or None on failure.
    """
    if not settings.openai_api_key:
        return None

    try:
        from openai import OpenAI
    except Exception:
        return None

    if kind == "basic":
        system = (
            "Extract a concise search query for a cricket assistant.\n"
            "Return ONLY JSON with key: query.\n"
            "The query should be short and focused (no extra text)."
        )
        schema = _BasicExtraction
    else:
        system = (
            "Extract analyst query parameters for a cricket stats assistant.\n"
            "Return ONLY JSON with keys: question, format, player, team, since_year, until_year, limit.\n"
            "- question: restate the stats question clearly\n"
            "- format: optional (e.g., T20/ODI/Test/IPL)\n"
            "- since_year/until_year: optional ints\n"
            "- limit: number of rows desired (1-200)\n"
        )
        schema = _AnalystExtraction

    payload = {
        "session_summary": state.summary,
        # Avoid sending datetimes to the LLM (json.dumps can't serialize them by default).
        "recent_messages": [m.model_dump(exclude={"created_at"}) for m in state.messages[-6:]],
        "current_message": state.user_message.model_dump(exclude={"created_at"}),
    }

    client = OpenAI(api_key=settings.openai_api_key)
    try:
        resp = client.chat.completions.create(
            model=settings.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or "{}"
        parsed = schema.model_validate_json(content)
        return parsed.model_dump()
    except (ValidationError, Exception):
        return None


def plan_tools(state: ChatState, *, settings: Settings | None = None) -> ToolPlan:
    """
    Deterministic tool selection by route, with optional LLM parameter extraction.

    This planner ONLY decides "which tools" and "with what args". It does not execute tools.
    """
    effective_settings = settings or get_settings()

    if state.route == "unknown":
        return ToolPlan(
            route="unknown",
            calls=[],
            reason=state.route_reason or "No route selected.",
            clarifying_question=state.clarifying_question
            or "Is your question about cricket rules/explanations, stats/analysis, a match scenario, or fantasy team selection?",
        )

    if state.route == "basic":
        extracted = _extract_with_llm(state, effective_settings, kind="basic")
        query = (extracted or {}).get("query") or state.user_message.content

        calls: list[PlannedToolCall] = [
            PlannedToolCall(
                tool_name="retrieval",
                args={"query": query, "top_k": 5},
                timeout_s=effective_settings.timeout_retrieval_s,
                use_cache=True,
                note="RAG over curated cricket knowledge",
            )
        ]

        if _needs_web(state.user_message.content):
            calls.append(
                PlannedToolCall(
                    tool_name="web_search",
                    args={"query": query, "top_k": 5},
                    timeout_s=effective_settings.timeout_web_s,
                    use_cache=True,
                    ttl_days=effective_settings.web_cache_ttl_days,
                    note="Tavily/ESPN match info (cached + cited)",
                )
            )

        return ToolPlan(route="basic", calls=calls, reason="Basic route: retrieval (and optional web if freshness implied).")

    if state.route == "analyst":
        extracted = _extract_with_llm(state, effective_settings, kind="analyst")
        spec = extracted or {"question": state.user_message.content, "limit": 20}

        return ToolPlan(
            route="analyst",
            calls=[
                PlannedToolCall(
                    tool_name="stats",
                    args=spec,
                    timeout_s=effective_settings.timeout_stats_s,
                    use_cache=False,
                    note="DuckDB-backed stats query (grounded)",
                )
            ],
            reason="Analyst route: always plan stats tool call.",
        )

    if state.route == "sim":
        return ToolPlan(
            route="sim",
            calls=[],
            reason="Simulator is planned for Phase 3; no tools executed yet.",
            clarifying_question="Simulator mode will be implemented later. Share the match situation (target/required, overs/balls left, wickets in hand) so we can use it once sim is added.",
        )

    if state.route == "fantasy":
        return ToolPlan(
            route="fantasy",
            calls=[],
            reason="Fantasy assistant is planned for Phase 3; no tools executed yet.",
            clarifying_question="Fantasy mode will be implemented later. Which match/teams is this for, and what fantasy rules do you want (platform, credits/budget, roles, team limits)?",
        )

    return ToolPlan(route="unknown", calls=[], reason="Unhandled route.", clarifying_question=state.clarifying_question)
