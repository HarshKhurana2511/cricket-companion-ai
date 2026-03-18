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


def compose_assistant_output(*, state: "ChatState", tool_plan: dict[str, Any] | None) -> "AssistantOutput":
    """
    Deterministic response composition (used by the graph and streaming API).

    This function intentionally does not do LLM composing; that is layered on top (2.7.0/2.7.2).
    """
    from cricket_companion.output_models import AssistantOutput

    plan = tool_plan or {}
    calls = plan.get("calls") or []

    if state.route == "unknown":
        answer = state.clarifying_question or "Can you clarify what you want?"
        return AssistantOutput(answer_text=answer, citations=[], tables=[], charts=[])

    if state.route == "basic":
        from cricket_companion.basic_response import build_basic_output

        retrieval_trace = next((t for t in reversed(state.tool_traces) if t.tool_name == "retrieval"), None)
        if retrieval_trace is None:
            return AssistantOutput(
                answer_text="I couldn't run retrieval for this question. Please try again.",
                citations=[],
                tables=[],
                charts=[],
                warnings=["Missing retrieval tool trace."],
            )
        if retrieval_trace.error is not None:
            return AssistantOutput(
                answer_text=f"Retrieval tool error: {retrieval_trace.error.code}: {retrieval_trace.error.message}",
                citations=[],
                tables=[],
                charts=[],
                warnings=["Retrieval tool execution failed before returning hits."],
            )

        resp = retrieval_trace.response if isinstance(retrieval_trace.response, dict) else None
        ok = bool(resp.get("ok")) if resp is not None else False
        if not ok:
            err = (resp or {}).get("error") or {}
            code = err.get("code") or "UNKNOWN"
            msg = err.get("message") or "Retrieval failed."
            return AssistantOutput(
                answer_text=f"Couldn't retrieve supporting docs ({code}): {msg}",
                citations=[],
                tables=[],
                charts=[],
                warnings=["No basic answer was composed because retrieval returned ok=false."],
            )

        output = build_basic_output(question=state.user_message.content, retrieval_tool_response=resp or {})
        # Merge in any web citations collected during tool execution (web_search/web_fetch/espn_ingest).
        # Keep the list small and de-duplicated by URL.
        if state.citations:
            seen: set[str] = {c.url for c in output.citations}
            for c in state.citations:
                if c.url in seen:
                    continue
                output.citations.append(c)
                seen.add(c.url)
                if len(output.citations) >= 6:
                    break
        return output

    if state.route == "analyst":
        from cricket_companion.analyst_response import build_analyst_output

        stats_trace = next((t for t in reversed(state.tool_traces) if t.tool_name == "stats"), None)
        if stats_trace is None:
            return AssistantOutput(
                answer_text="I couldn't run the stats tool for this question. Please try again.",
                citations=[],
                tables=[],
                charts=[],
                warnings=["Missing stats tool trace."],
            )
        if stats_trace.error is not None:
            return AssistantOutput(
                answer_text=f"Stats tool error: {stats_trace.error.code}: {stats_trace.error.message}",
                citations=[],
                tables=[],
                charts=[],
                warnings=["Stats tool execution failed before a SQL result was produced."],
            )

        resp = stats_trace.response if isinstance(stats_trace.response, dict) else None
        ok = bool(resp.get("ok")) if resp is not None else False
        if not ok:
            err = (resp or {}).get("error") or {}
            code = err.get("code") or "UNKNOWN"
            msg = err.get("message") or "Stats query failed."
            details = err.get("details") or {}
            clarifying = details.get("clarifying_question") if isinstance(details, dict) else None

            answer = clarifying or f"Couldn't compute stats ({code}): {msg}"
            return AssistantOutput(
                answer_text=answer,
                citations=[],
                tables=[],
                charts=[],
                warnings=["No stats claims were made because the SQL tool did not return ok=true."],
                assumptions=["Dataset is IPL men (Cricsheet) only."],
            )
        if not state.tables:
            return AssistantOutput(
                answer_text="Stats query succeeded but returned no table output to ground an answer.",
                citations=[],
                tables=[],
                charts=[],
                warnings=["No stats claims were made because there was no SQL table output."],
                assumptions=["Dataset is IPL men (Cricsheet) only."],
            )

        return build_analyst_output(
            table=state.tables[0],
            citations=state.citations,
            warnings=[],
            assumptions=[
                "Dataset is IPL men (Cricsheet) only.",
                "Economy/run rates use legal balls (wides/no-balls excluded from ball counts).",
                "Bowler wickets exclude non-bowler dismissals (e.g., run outs) where applicable.",
            ],
        )

    if state.route == "sim":
        from cricket_companion.sim_response import build_sim_output

        sim_trace = next((t for t in reversed(state.tool_traces) if t.tool_name == "sim"), None)
        if sim_trace is None:
            return AssistantOutput(
                answer_text="I couldn't run the simulator for this question. Please try again.",
                citations=[],
                tables=[],
                charts=[],
                warnings=["Missing sim tool trace."],
            )
        if sim_trace.error is not None:
            return AssistantOutput(
                answer_text=f"Sim tool error: {sim_trace.error.code}: {sim_trace.error.message}",
                citations=[],
                tables=[],
                charts=[],
                warnings=["Sim tool execution failed before returning output."],
            )

        resp = sim_trace.response if isinstance(sim_trace.response, dict) else None
        ok = bool(resp.get("ok")) if resp is not None else False
        if not ok:
            err = (resp or {}).get("error") or {}
            code = err.get("code") or "UNKNOWN"
            msg = err.get("message") or "Simulation failed."
            return AssistantOutput(
                answer_text=f"Couldn't simulate this scenario ({code}): {msg}",
                citations=[],
                tables=[],
                charts=[],
                warnings=["No sim answer was composed because sim returned ok=false."],
            )

        return build_sim_output(question=state.user_message.content, sim_tool_response=resp or {}, request_id=state.request_id)

    call_lines = []
    for call in calls:
        call_lines.append(f"- {call.get('tool_name')} args={call.get('args')}")
    planned = "\n".join(call_lines) if call_lines else "(no tools planned)"
    answer = (
        f"Route: {state.route}\n"
        f"Reason: {state.route_reason or '(none)'}\n\n"
        f"Planned tool calls:\n{planned}\n\n"
        "Next: richer grounded response composition will be added in later tasks."
    )
    return AssistantOutput(answer_text=answer, citations=state.citations, tables=state.tables, charts=state.charts)


def build_graph(*, enable_composer: bool = True):
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
    from cricket_companion.llm_composer import compose_answer_with_llm

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
        output = compose_assistant_output(state=s, tool_plan=plan if isinstance(plan, dict) else None)

        # Phase 2.7.0: LLM composer pass (best-effort). Falls back to deterministic output on failure.
        if enable_composer:
            composed = compose_answer_with_llm(
                route=s.route,
                user_message=s.user_message.content,
                draft_answer_text=output.answer_text,
                tool_plan=plan if isinstance(plan, dict) else None,
                tool_traces=[t.model_dump(mode="json") for t in s.tool_traces],
                citations=[c.model_dump(mode="json") for c in output.citations],
                tables=[t.model_dump(mode="json") for t in output.tables],
                prefs=s.prefs,
                session_summary=s.summary,
            )
            if composed is not None:
                output.answer_text = composed

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
