from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from cricket_companion.chat_models import ChatState, ToolCallTrace
from cricket_companion.output_models import TableArtifact
from cricket_companion.planner import PlannedToolCall, ToolPlan
from cricket_companion.schemas import Citation, ErrorCode, ToolError
from cricket_companion.tools import RetrievalMcpClient, StatsMcpClient
from cricket_companion.logging_config import get_logger


log = get_logger("executor")


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _extract_citations(obj: Any) -> list[Citation]:
    try:
        if isinstance(obj, dict):
            meta = obj.get("meta") or {}
            citations = meta.get("citations") or []
            return [Citation.model_validate(c) for c in citations]
    except Exception:
        return []
    return []


def execute_tool_plan_iter(state: ChatState) -> Any:
    """
    Executes `state.tool_plan` and yields timeline events suitable for streaming.

    Yields dicts shaped like: {"event": str, "data": dict}.
    """
    plan_dict = state.tool_plan
    if not plan_dict:
        return iter(())

    plan = ToolPlan.model_validate(plan_dict)
    stats_client: StatsMcpClient | None = None
    retrieval_client: RetrievalMcpClient | None = None

    def _iter():
        nonlocal stats_client, retrieval_client
        try:
            for call in plan.calls:
                started_at = _utc_now()
                started_perf = time.perf_counter()

                yield {
                    "event": "tool_start",
                    "data": {
                        "tool_name": call.tool_name,
                        "request": call.model_dump(mode="json"),
                        "started_at": started_at.isoformat(),
                    },
                }
                log.info(
                    "tool.call_start",
                    extra={
                        "tool_name": call.tool_name,
                        "request_id": state.request_id,
                        "session_id": state.session_id,
                    },
                )

                response_any: Any = None
                tool_error: ToolError | None = None
                citations: list[Citation] = []

                try:
                    if call.tool_name == "stats":
                        stats_client = stats_client or StatsMcpClient()
                        resp = stats_client.query(call.args)
                        response_any = resp.model_dump(mode="json")
                    elif call.tool_name == "retrieval":
                        retrieval_client = retrieval_client or RetrievalMcpClient()
                        resp = retrieval_client.retrieve(
                            query=str(call.args.get("query", "")),
                            top_k=int(call.args.get("top_k", 5)),
                        )
                        response_any = resp.model_dump(mode="json")
                    elif call.tool_name == "web_search":
                        tool_error = ToolError(code=ErrorCode.NOT_FOUND, message="web_search not implemented yet.")
                    elif call.tool_name in {"sim", "fantasy"}:
                        tool_error = ToolError(code=ErrorCode.NOT_FOUND, message=f"{call.tool_name} not implemented yet.")
                    else:
                        tool_error = ToolError(code=ErrorCode.NOT_FOUND, message=f"Unknown planned tool: {call.tool_name}")
                except Exception as exc:
                    tool_error = ToolError(code=ErrorCode.INTERNAL, message=str(exc))

                elapsed_ms = int((time.perf_counter() - started_perf) * 1000)
                ended_at = _utc_now()

                if response_any is not None:
                    citations = _extract_citations(response_any)

                trace = ToolCallTrace(
                    tool_name=call.tool_name,
                    started_at=started_at,
                    ended_at=ended_at,
                    elapsed_ms=elapsed_ms,
                    request=call.model_dump(mode="json"),
                    response=response_any,
                    cache_hit=None,
                    cache_key=None,
                    citations=citations,
                    error=tool_error,
                )
                state.tool_traces.append(trace)

                # Merge citations (web tools later will populate these)
                if citations:
                    state.citations.extend(citations)
                    yield {
                        "event": "artifact_citations",
                        "data": {"count": len(citations)},
                    }

                # Merge tables from stats if present and shaped like a TableArtifact
                if call.tool_name == "stats" and response_any is not None:
                    try:
                        data = (response_any.get("data") if isinstance(response_any, dict) else None) or {}
                        if isinstance(data, dict) and data.get("name") and "rows" in data:
                            table = TableArtifact.model_validate(data)
                            state.tables.append(table)
                            yield {
                                "event": "artifact_table",
                                "data": {
                                    "table_id": table.table_id,
                                    "name": table.name,
                                    "columns": [c.name for c in table.columns],
                                    "rows_preview": len(table.rows[:3]),
                                },
                            }
                    except Exception:
                        pass

                ok = None
                if isinstance(response_any, dict):
                    ok = response_any.get("ok")
                yield {
                    "event": "tool_end",
                    "data": {
                        "tool_name": call.tool_name,
                        "ok": ok,
                        "error": tool_error.model_dump(mode="json") if tool_error else None,
                        "elapsed_ms": elapsed_ms,
                        "ended_at": ended_at.isoformat(),
                    },
                }
                log.info(
                    "tool.call_end",
                    extra={
                        "tool_name": call.tool_name,
                        "ok": ok,
                        "elapsed_ms": elapsed_ms,
                        "error_code": tool_error.code if tool_error else None,
                        "request_id": state.request_id,
                        "session_id": state.session_id,
                    },
                )
        finally:
            if stats_client is not None:
                stats_client.close()
            if retrieval_client is not None:
                retrieval_client.close()

    return _iter()


def execute_tool_plan(state: ChatState) -> ChatState:
    """
    Executes `state.tool_plan` (if present) and appends ToolCallTrace entries.

    This is intentionally simple:
    - sequential execution
    - populates traces and merges citations/tables where possible
    """
    for _ in execute_tool_plan_iter(state):
        pass

    return state
