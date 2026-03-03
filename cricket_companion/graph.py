from __future__ import annotations

from typing import Any
from typing_extensions import TypedDict


class GraphState(TypedDict, total=False):
    session_id: str
    request_id: str

    user_message: dict[str, Any]
    messages: list[dict[str, Any]]
    summary: str | None

    route: str
    route_reason: str | None
    clarifying_question: str | None

    tool_plan: dict[str, Any] | None

    draft_answer: str | None
    final_answer: str | None

    citations: list[dict[str, Any]]
    tables: list[dict[str, Any]]
    charts: list[dict[str, Any]]
    assistant_output: dict[str, Any] | None

    tool_traces: list[dict[str, Any]]
    events: list[dict[str, Any]]
    errors: list[dict[str, Any]]


def build_graph():
    """
    Build and compile the LangGraph for the agent core.

    Nodes (Phase 2.1):
    - route: route intent (heuristics-first, LLM fallback)
    - plan: deterministic tool selection + optional LLM parameter extraction
    - respond: produce a simple structured response (placeholder until tool execution exists)
    """
    try:
        from langgraph.graph import END, StateGraph
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "LangGraph is not installed. Install it with: uv add langgraph"
        ) from exc

    from cricket_companion.chat_models import ChatState
    from cricket_companion.output_models import AssistantOutput
    from cricket_companion.executor import execute_tool_plan
    from cricket_companion.planner import plan_tools
    from cricket_companion.router import route_intent

    def route_node(state: GraphState) -> GraphState:
        s = ChatState.model_validate(state)
        s = route_intent(s)
        return s.model_dump()

    def plan_node(state: GraphState) -> GraphState:
        s = ChatState.model_validate(state)
        plan = plan_tools(s)
        data = s.model_dump()
        data["tool_plan"] = plan.model_dump()
        return data

    def execute_tools_node(state: GraphState) -> GraphState:
        s = ChatState.model_validate(state)
        s = execute_tool_plan(s)
        return s.model_dump()

    def respond_node(state: GraphState) -> GraphState:
        s = ChatState.model_validate(state)
        plan = (state.get("tool_plan") or {})
        calls = plan.get("calls") or []

        if s.route == "unknown":
            answer = s.clarifying_question or "Can you clarify what you want?"
            output = AssistantOutput(answer_text=answer, citations=[], tables=[], charts=[])
        elif s.route == "basic":
            from cricket_companion.basic_response import build_basic_output

            retrieval_trace = next((t for t in reversed(s.tool_traces) if t.tool_name == "retrieval"), None)
            if retrieval_trace is None:
                output = AssistantOutput(
                    answer_text="I couldn't run retrieval for this question. Please try again.",
                    citations=[],
                    tables=[],
                    charts=[],
                    warnings=["Missing retrieval tool trace."],
                )
            elif retrieval_trace.error is not None:
                output = AssistantOutput(
                    answer_text=f"Retrieval tool error: {retrieval_trace.error.code}: {retrieval_trace.error.message}",
                    citations=[],
                    tables=[],
                    charts=[],
                    warnings=["Retrieval tool execution failed before returning hits."],
                )
            else:
                resp = retrieval_trace.response if isinstance(retrieval_trace.response, dict) else None
                ok = bool(resp.get("ok")) if resp is not None else False
                if not ok:
                    err = (resp or {}).get("error") or {}
                    code = err.get("code") or "UNKNOWN"
                    msg = err.get("message") or "Retrieval failed."
                    output = AssistantOutput(
                        answer_text=f"Couldn't retrieve supporting docs ({code}): {msg}",
                        citations=[],
                        tables=[],
                        charts=[],
                        warnings=["No basic answer was composed because retrieval returned ok=false."],
                    )
                else:
                    output = build_basic_output(question=s.user_message.content, retrieval_tool_response=resp or {})
        elif s.route == "analyst":
            from cricket_companion.analyst_response import build_analyst_output

            stats_trace = next((t for t in reversed(s.tool_traces) if t.tool_name == "stats"), None)
            if stats_trace is None:
                output = AssistantOutput(
                    answer_text="I couldn't run the stats tool for this question. Please try again.",
                    citations=[],
                    tables=[],
                    charts=[],
                    warnings=["Missing stats tool trace."],
                )
            elif stats_trace.error is not None:
                output = AssistantOutput(
                    answer_text=f"Stats tool error: {stats_trace.error.code}: {stats_trace.error.message}",
                    citations=[],
                    tables=[],
                    charts=[],
                    warnings=["Stats tool execution failed before a SQL result was produced."],
                )
            else:
                resp = stats_trace.response if isinstance(stats_trace.response, dict) else None
                ok = bool(resp.get("ok")) if resp is not None else False
                if not ok:
                    err = (resp or {}).get("error") or {}
                    code = err.get("code") or "UNKNOWN"
                    msg = err.get("message") or "Stats query failed."
                    details = err.get("details") or {}
                    clarifying = details.get("clarifying_question") if isinstance(details, dict) else None

                    answer = clarifying or f"Couldn't compute stats ({code}): {msg}"
                    output = AssistantOutput(
                        answer_text=answer,
                        citations=[],
                        tables=[],
                        charts=[],
                        warnings=["No stats claims were made because the SQL tool did not return ok=true."],
                        assumptions=["Dataset is IPL men (Cricsheet) only."],
                    )
                elif not s.tables:
                    output = AssistantOutput(
                        answer_text="Stats query succeeded but returned no table output to ground an answer.",
                        citations=[],
                        tables=[],
                        charts=[],
                        warnings=["No stats claims were made because there was no SQL table output."],
                        assumptions=["Dataset is IPL men (Cricsheet) only."],
                    )
                else:
                    output = build_analyst_output(
                        table=s.tables[0],
                        citations=s.citations,
                        warnings=[],
                        assumptions=[
                            "Dataset is IPL men (Cricsheet) only.",
                            "Economy/run rates use legal balls (wides/no-balls excluded from ball counts).",
                            "Bowler wickets exclude non-bowler dismissals (e.g., run outs) where applicable.",
                        ],
                    )
        else:
            call_lines = []
            for call in calls:
                call_lines.append(f"- {call.get('tool_name')} args={call.get('args')}")
            planned = "\n".join(call_lines) if call_lines else "(no tools planned)"
            answer = (
                f"Route: {s.route}\n"
                f"Reason: {s.route_reason or '(none)'}\n\n"
                f"Planned tool calls:\n{planned}\n\n"
                "Next: richer grounded response composition will be added in later tasks."
            )
            output = AssistantOutput(answer_text=answer, citations=s.citations, tables=s.tables, charts=s.charts)

        s.final_answer = output.answer_text
        s.assistant_output = output
        s.citations = output.citations
        s.tables = output.tables
        s.charts = output.charts
        return s.model_dump()

    graph = StateGraph(GraphState)
    graph.add_node("route", route_node)
    graph.add_node("plan", plan_node)
    graph.add_node("execute_tools", execute_tools_node)
    graph.add_node("respond", respond_node)

    graph.set_entry_point("route")
    graph.add_edge("route", "plan")
    graph.add_edge("plan", "execute_tools")
    graph.add_edge("execute_tools", "respond")
    graph.add_edge("respond", END)

    return graph.compile()
