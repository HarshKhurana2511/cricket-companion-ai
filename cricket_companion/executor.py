from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from cricket_companion.chat_models import ChatState, ToolCallTrace
from cricket_companion.output_models import TableArtifact
from cricket_companion.planner import PlannedToolCall, ToolPlan
from cricket_companion.schemas import Citation, ErrorCode, ToolError
from cricket_companion.tools import RetrievalMcpClient, SimMcpClient, StatsMcpClient, WebMcpClient
from cricket_companion.logging_config import get_logger
from cricket_companion.config import get_settings


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


def _extract_cache_info(obj: Any) -> tuple[bool | None, str | None]:
    try:
        if isinstance(obj, dict):
            meta = obj.get("meta") or {}
            cache = meta.get("cache") or {}
            if not isinstance(cache, dict):
                return (None, None)
            hit = cache.get("hit")
            key = cache.get("key")
            return (hit if isinstance(hit, bool) else None, key if isinstance(key, str) else None)
    except Exception:
        return (None, None)
    return (None, None)


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
    web_client: WebMcpClient | None = None
    sim_client: SimMcpClient | None = None
    settings = get_settings()

    def _iter():
        nonlocal stats_client, retrieval_client, web_client, sim_client
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
                secondary_calls: list[tuple[str, dict[str, Any]]] = []

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
                        web_client = web_client or WebMcpClient()
                        resp = web_client.search(
                            {
                                "query": str(call.args.get("query", "")),
                                "top_k": int(call.args.get("top_k", 5)),
                                "topic": call.args.get("topic"),
                                "search_depth": call.args.get("search_depth"),
                                "time_range": call.args.get("time_range"),
                                "days": call.args.get("days"),
                                "country": call.args.get("country"),
                                "include_domains": call.args.get("include_domains"),
                                "exclude_domains": call.args.get("exclude_domains"),
                                "include_answer": bool(call.args.get("include_answer", False)),
                                "include_raw_content": bool(call.args.get("include_raw_content", False)),
                            }
                        )
                        response_any = resp.model_dump(mode="json")
                        # Best-effort enrichment: if search returns ESPNcricinfo links, ingest the top one.
                        try:
                            data = (response_any.get("data") if isinstance(response_any, dict) else None) or {}
                            results = data.get("results") if isinstance(data, dict) else None
                            if isinstance(results, list):
                                espn_url = None
                                first_url = None
                                for r in results[:5]:
                                    if not isinstance(r, dict):
                                        continue
                                    u = r.get("url")
                                    if not isinstance(u, str) or not u:
                                        continue
                                    if first_url is None:
                                        first_url = u
                                    if "espncricinfo.com" in u:
                                        espn_url = u
                                        break
                                if espn_url:
                                    secondary_calls.append(("espn_ingest", {"url": espn_url, "mode": "scorecard"}))
                                elif first_url:
                                    secondary_calls.append(("web_fetch", {"url": first_url, "mode": "article", "max_chars": 8000}))
                        except Exception:
                            pass
                    elif call.tool_name == "sim":
                        sim_client = sim_client or SimMcpClient()
                        resp = sim_client.run(call.args)
                        response_any = resp.model_dump(mode="json")
                    elif call.tool_name == "fantasy":
                        tool_error = ToolError(code=ErrorCode.NOT_FOUND, message="fantasy not implemented yet.")
                    else:
                        tool_error = ToolError(code=ErrorCode.NOT_FOUND, message=f"Unknown planned tool: {call.tool_name}")
                except Exception as exc:
                    tool_error = ToolError(code=ErrorCode.INTERNAL, message=str(exc))

                elapsed_ms = int((time.perf_counter() - started_perf) * 1000)
                ended_at = _utc_now()

                if response_any is not None:
                    citations = _extract_citations(response_any)

                cache_hit, cache_key = _extract_cache_info(response_any)
                trace = ToolCallTrace(
                    tool_name=call.tool_name,
                    started_at=started_at,
                    ended_at=ended_at,
                    elapsed_ms=elapsed_ms,
                    request=call.model_dump(mode="json"),
                    response=response_any,
                    cache_hit=cache_hit,
                    cache_key=cache_key,
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

                # Run secondary web enrichment calls (espn_ingest/web_fetch) and rebuild web index best-effort.
                if secondary_calls and web_client is not None:
                    for tool_name, tool_args in secondary_calls[:2]:
                        started2_at = _utc_now()
                        started2_perf = time.perf_counter()
                        yield {
                            "event": "tool_start",
                            "data": {
                                "tool_name": tool_name,
                                "request": {"tool_name": tool_name, "args": tool_args},
                                "started_at": started2_at.isoformat(),
                            },
                        }
                        resp2_any: Any = None
                        err2: ToolError | None = None
                        try:
                            if tool_name == "espn_ingest":
                                resp2 = web_client.espn_ingest(tool_args)
                                resp2_any = resp2.model_dump(mode="json")
                            elif tool_name == "web_fetch":
                                resp2 = web_client.fetch(tool_args)
                                resp2_any = resp2.model_dump(mode="json")
                            else:
                                err2 = ToolError(code=ErrorCode.NOT_FOUND, message=f"Unknown secondary tool: {tool_name}")
                        except Exception as exc:
                            err2 = ToolError(code=ErrorCode.INTERNAL, message=str(exc))

                        elapsed2_ms = int((time.perf_counter() - started2_perf) * 1000)
                        ended2_at = _utc_now()
                        citations2 = _extract_citations(resp2_any)
                        cache2_hit, cache2_key = _extract_cache_info(resp2_any)
                        trace2 = ToolCallTrace(
                            tool_name=tool_name,
                            started_at=started2_at,
                            ended_at=ended2_at,
                            elapsed_ms=elapsed2_ms,
                            request={"tool_name": tool_name, "args": tool_args},
                            response=resp2_any,
                            cache_hit=cache2_hit,
                            cache_key=cache2_key,
                            citations=citations2,
                            error=err2,
                        )
                        state.tool_traces.append(trace2)
                        if citations2:
                            state.citations.extend(citations2)
                            yield {"event": "artifact_citations", "data": {"count": len(citations2)}}

                        yield {
                            "event": "tool_end",
                            "data": {
                                "tool_name": tool_name,
                                "ok": (resp2_any.get("ok") if isinstance(resp2_any, dict) else None),
                                "error": err2.model_dump(mode="json") if err2 else None,
                                "elapsed_ms": elapsed2_ms,
                                "ended_at": ended2_at.isoformat(),
                            },
                        }

                    # Index cached web outputs into retrieval chunks (3.1.4) so subsequent turns can retrieve them.
                    try:
                        from pathlib import Path

                        from pipelines.build_web_index import build_web_index

                        build_web_index(cache_dir=Path(settings.cache_dir), out_dir=Path("data/retrieval"))
                    except Exception:
                        pass

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
            if web_client is not None:
                web_client.close()
            if sim_client is not None:
                sim_client.close()

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
