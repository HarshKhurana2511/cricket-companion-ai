from __future__ import annotations

import json
import re
from typing import Any

from cricket_companion.chat_models import ChatRequest, ChatResponse, ChatState, Message
from cricket_companion.config import Settings, get_settings
from cricket_companion.logging_config import get_logger, set_log_context
from cricket_companion.memory_store import MemoryStore
from cricket_companion.llm_composer import extract_numbers
from cricket_companion.llm_composer import stream_compose_answer_with_llm


log = get_logger("chat_service")


def _parse_pref_value(raw: str) -> Any:
    s = (raw or "").strip()
    if s == "":
        return ""
    if s.lower() in {"true", "false", "null"}:
        return json.loads(s.lower())
    if re.match(r"^-?\d+$", s):
        return int(s)
    if re.match(r"^-?\d+\.\d+$", s):
        return float(s)
    if s[:1] in {"{", "[", "\""}:
        try:
            return json.loads(s)
        except Exception:
            return s
    return s


def _handle_mem_command(
    store: MemoryStore,
    con: object,
    *,
    user_id: str,
    session_id: str,
    artifacts_dir: object,
    text: str,
) -> str | None:
    """
    Deterministic memory governance commands (2.6.4):
      /mem help
      /mem sessions
      /mem show <session_id> [n]
      /mem export <session_id>
      /mem export-user
      /mem clear-summary <session_id> confirm
      /mem delete <session_id> confirm
      /mem delete-last <n> confirm
      /mem purge-user confirm
    """
    t = (text or "").strip()
    if not t.lower().startswith(("/mem", "mem ")):
        return None

    parts = t.split()
    if len(parts) == 1 or parts[1].lower() in {"help", "-h", "--help"}:
        return (
            "Memory commands:\n"
            "- /mem sessions\n"
            "- /mem show <session_id> [n]\n"
            "- /mem export <session_id>\n"
            "- /mem export-user\n"
            "- /mem clear-summary <session_id> confirm\n"
            "- /mem delete <session_id> confirm\n"
            "- /mem delete-last <n> confirm   (deletes last n messages from current session)\n"
            "- /mem purge-user confirm        (deletes prefs and user-scoped sessions)\n"
        )

    cmd = parts[1].lower()
    if cmd == "sessions":
        sessions = store.list_sessions(con, user_id=user_id, limit=50)
        return "Sessions:\n" + json.dumps(sessions, indent=2, ensure_ascii=False)

    if cmd == "show" and len(parts) >= 3:
        sid = parts[2]
        n = int(parts[3]) if len(parts) >= 4 and parts[3].isdigit() else 20
        data = store.get_session(con, session_id=sid, message_limit=n)
        return "Session:\n" + json.dumps(data, indent=2, ensure_ascii=False) if data else f"No such session: {sid}"

    from pathlib import Path

    if cmd == "export" and len(parts) >= 3:
        sid = parts[2]
        out_dir = Path(str(artifacts_dir)) / "memory_exports" / sid
        paths = store.export_session_artifacts(con, session_id=sid, out_dir=out_dir)
        return "Exported session artifacts:\n" + json.dumps(paths, indent=2, ensure_ascii=False)

    if cmd in {"export-user", "export_user"}:
        out_dir = Path(str(artifacts_dir)) / "memory_exports" / "users" / user_id
        paths = store.export_user_artifacts(con, user_id=user_id, out_dir=out_dir)
        return "Exported user artifacts:\n" + json.dumps(paths, indent=2, ensure_ascii=False)

    # Destructive commands require explicit "confirm".
    if cmd == "clear-summary" and len(parts) >= 4 and parts[-1].lower() == "confirm":
        sid = parts[2]
        changed = store.clear_session_summary(con, session_id=sid)
        return f"Cleared summary for session: {sid}" if changed else f"No summary to clear for session: {sid}"

    if cmd == "delete" and len(parts) >= 4 and parts[-1].lower() == "confirm":
        sid = parts[2]
        deleted = store.delete_session(con, session_id=sid)
        return f"Deleted session: {sid}" if deleted else f"No such session: {sid}"

    if cmd in {"delete-last", "delete_last"} and len(parts) >= 4 and parts[-1].lower() == "confirm":
        n = int(parts[2]) if parts[2].isdigit() else 0
        if n <= 0:
            return "Usage: /mem delete-last <n> confirm"
        deleted = store.delete_last_messages(con, session_id=session_id, n=n)
        return f"Deleted last {deleted} messages from session {session_id}."

    if cmd in {"purge-user", "purge_user"} and len(parts) >= 3 and parts[-1].lower() == "confirm":
        stats = store.purge_user(con, user_id=user_id)
        return "Purged user data:\n" + json.dumps(stats, indent=2, ensure_ascii=False)

    return "Usage: /mem help"


def _handle_pref_command(store: MemoryStore, con: object, *, user_id: str, text: str) -> str | None:
    """
    Deterministic preference commands:
      /pref list
      /pref set key=value
      /pref del key
    """
    t = (text or "").strip()
    if not t.lower().startswith(("/pref", "pref ")):
        return None

    parts = t.split(maxsplit=2)
    if len(parts) < 2:
        return "Usage: /pref list | /pref set key=value | /pref del key"

    cmd = parts[1].lower()
    if cmd == "list":
        prefs = store.load_preferences(con, user_id=user_id)
        return "Preferences:\n" + json.dumps(prefs, indent=2, ensure_ascii=False)

    if cmd == "set" and len(parts) == 3:
        kv = parts[2].strip()
        if "=" not in kv:
            return "Usage: /pref set key=value"
        key, value_raw = kv.split("=", 1)
        key = key.strip()
        value = _parse_pref_value(value_raw)
        if not key:
            return "Preference key cannot be empty."
        store.set_preference(con, user_id=user_id, key=key, value=value)
        return f"Saved preference: {key}={json.dumps(value, ensure_ascii=False)}"

    if cmd in {"del", "delete", "rm", "remove"} and len(parts) == 3:
        key = parts[2].strip()
        if not key:
            return "Usage: /pref del key"
        existed = store.delete_preference(con, user_id=user_id, key=key)
        return f"Deleted preference: {key}" if existed else f"No such preference: {key}"

    return "Usage: /pref list | /pref set key=value | /pref del key"


def handle_chat(request: ChatRequest, *, settings: Settings | None = None) -> ChatResponse:
    """
    Library entrypoint for a single chat turn.

    Phase 2.6.1:
    - loads session memory (messages + summary) from DuckDB
    - runs the LangGraph agent
    - persists user+assistant messages
    - updates session summary using an LLM (if configured)
    """
    effective_settings = settings or get_settings(load_env_file=True)

    try:
        from cricket_companion.graph import build_graph
    except Exception as exc:
        raise RuntimeError(f"Failed to import graph: {exc}") from exc

    store = MemoryStore(db_path=effective_settings.memory_db_path)
    con = store.connect()
    try:
        user_id = request.user_id or "local-user"
        set_log_context(request_id=request.request_id, session_id=request.session_id, user_id=user_id)
        log.info(
            "chat.turn_start",
            extra={
                "message_id": request.message.message_id,
                "max_context_messages": request.max_context_messages,
                "debug": request.debug,
            },
        )

        mem_resp = _handle_mem_command(
            store,
            con,
            user_id=user_id,
            session_id=request.session_id,
            artifacts_dir=effective_settings.artifacts_dir,
            text=request.message.content,
        )
        if mem_resp is not None:
            assistant_message = Message(role="assistant", content=mem_resp)
            store.append_messages(
                con,
                session_id=request.session_id,
                user_id=user_id,
                messages=[
                    request.message.model_dump(mode="json"),
                    assistant_message.model_dump(mode="json"),
                ],
            )
            store.summarize_if_needed(
                con,
                session_id=request.session_id,
                keep_last_n=effective_settings.memory_keep_last_n,
                summarize_chunk_n=effective_settings.memory_summarize_chunk_n,
                summary_max_chars=effective_settings.memory_summary_max_chars,
                summary_model=effective_settings.summary_model,
                openai_api_key=effective_settings.openai_api_key,
            )
            return ChatResponse(
                session_id=request.session_id,
                request_id=request.request_id,
                assistant_message=assistant_message,
                citations=[],
                tables=[],
                charts=[],
                route="basic",
                tool_traces=[],
                errors=[],
            )

        # Preference commands are handled deterministically without calling the agent graph.
        pref_resp = _handle_pref_command(store, con, user_id=user_id, text=request.message.content)
        if pref_resp is not None:
            assistant_message = Message(role="assistant", content=pref_resp)
            store.append_messages(
                con,
                session_id=request.session_id,
                user_id=user_id,
                messages=[
                    request.message.model_dump(mode="json"),
                    assistant_message.model_dump(mode="json"),
                ],
            )
            store.summarize_if_needed(
                con,
                session_id=request.session_id,
                keep_last_n=effective_settings.memory_keep_last_n,
                summarize_chunk_n=effective_settings.memory_summarize_chunk_n,
                summary_max_chars=effective_settings.memory_summary_max_chars,
                summary_model=effective_settings.summary_model,
                openai_api_key=effective_settings.openai_api_key,
            )
            return ChatResponse(
                session_id=request.session_id,
                request_id=request.request_id,
                assistant_message=assistant_message,
                citations=[],
                tables=[],
                charts=[],
                route="basic",
                tool_traces=[],
                errors=[],
            )

        ctx = store.load_context(con, session_id=request.session_id, user_id=user_id, max_messages=request.max_context_messages)
        prefs = store.load_preferences(con, user_id=user_id)

        state = ChatState(
            session_id=request.session_id,
            request_id=request.request_id,
            user_id=user_id,
            user_message=request.message,
            messages=[Message.model_validate(m) for m in ctx.messages],
            summary=ctx.summary,
            prefs=prefs,
        )

        graph = build_graph()
        result = graph.invoke(state.model_dump())
        log.info(
            "chat.turn_graph_completed",
            extra={
                "route": result.get("route") or "unknown",
                "tool_calls": len(((result.get("tool_plan") or {}) or {}).get("calls") or []),
                "tool_traces": len(result.get("tool_traces") or []),
                "tables": len(result.get("tables") or []),
                "citations": len(result.get("citations") or []),
            },
        )

        assistant_text = str(result.get("final_answer") or "").strip() or "(no answer)"
        assistant_message = Message(role="assistant", content=assistant_text)

        store.append_messages(
            con,
            session_id=request.session_id,
            user_id=user_id,
            messages=[
                request.message.model_dump(mode="json"),
                assistant_message.model_dump(mode="json"),
            ],
        )

        # LLM-based session summary updates (best-effort).
        store.summarize_if_needed(
            con,
            session_id=request.session_id,
            keep_last_n=effective_settings.memory_keep_last_n,
            summarize_chunk_n=effective_settings.memory_summarize_chunk_n,
            summary_max_chars=effective_settings.memory_summary_max_chars,
            summary_model=effective_settings.summary_model,
            openai_api_key=effective_settings.openai_api_key,
        )

        return ChatResponse(
            session_id=request.session_id,
            request_id=request.request_id,
            assistant_message=assistant_message,
            citations=[c for c in (result.get("citations") or [])],
            tables=[t for t in (result.get("tables") or [])],
            charts=[c for c in (result.get("charts") or [])],
            route=result.get("route") or "unknown",
            tool_traces=[t for t in (result.get("tool_traces") or [])],
            errors=[e for e in (result.get("errors") or [])],
        )
    finally:
        try:
            con.close()
        except Exception:
            pass


def stream_chat(request: ChatRequest, *, settings: Settings | None = None) -> Any:
    """
    Phase 2.7.2: Streaming chat generator.

    Yields events shaped like: {"event": str, "data": dict}.

    Streaming scope (A): stream only the final LLM composer output. Tool execution is still synchronous.
    """
    effective_settings = settings or get_settings(load_env_file=True)
    store = MemoryStore(db_path=effective_settings.memory_db_path)
    con = store.connect()

    def emit(event: str, data: dict[str, Any]) -> dict[str, Any]:
        return {"event": event, "data": data}

    try:
        user_id = request.user_id or "local-user"
        set_log_context(request_id=request.request_id, session_id=request.session_id, user_id=user_id)
        log.info(
            "chat.stream_turn_start",
            extra={
                "message_id": request.message.message_id,
                "max_context_messages": request.max_context_messages,
                "debug": request.debug,
            },
        )

        # Handle governance and prefs deterministically (single-chunk stream).
        mem_resp = _handle_mem_command(
            store,
            con,
            user_id=user_id,
            session_id=request.session_id,
            artifacts_dir=effective_settings.artifacts_dir,
            text=request.message.content,
        )
        if mem_resp is not None:
            assistant_message = Message(role="assistant", content=mem_resp)
            store.append_messages(
                con,
                session_id=request.session_id,
                user_id=user_id,
                messages=[request.message.model_dump(mode="json"), assistant_message.model_dump(mode="json")],
            )
            store.summarize_if_needed(
                con,
                session_id=request.session_id,
                keep_last_n=effective_settings.memory_keep_last_n,
                summarize_chunk_n=effective_settings.memory_summarize_chunk_n,
                summary_max_chars=effective_settings.memory_summary_max_chars,
                summary_model=effective_settings.summary_model,
                openai_api_key=effective_settings.openai_api_key,
            )
            yield emit("chunk", {"text": mem_resp})
            yield emit("done", {"ok": True})
            return

        pref_resp = _handle_pref_command(store, con, user_id=user_id, text=request.message.content)
        if pref_resp is not None:
            assistant_message = Message(role="assistant", content=pref_resp)
            store.append_messages(
                con,
                session_id=request.session_id,
                user_id=user_id,
                messages=[request.message.model_dump(mode="json"), assistant_message.model_dump(mode="json")],
            )
            store.summarize_if_needed(
                con,
                session_id=request.session_id,
                keep_last_n=effective_settings.memory_keep_last_n,
                summarize_chunk_n=effective_settings.memory_summarize_chunk_n,
                summary_max_chars=effective_settings.memory_summary_max_chars,
                summary_model=effective_settings.summary_model,
                openai_api_key=effective_settings.openai_api_key,
            )
            yield emit("chunk", {"text": pref_resp})
            yield emit("done", {"ok": True})
            return

        # Stream timeline events by running route -> plan -> tools -> respond manually.
        from cricket_companion.executor import execute_tool_plan_iter
        from cricket_companion.graph import compose_assistant_output
        from cricket_companion.planner import plan_tools
        from cricket_companion.router import route_intent

        ctx = store.load_context(
            con,
            session_id=request.session_id,
            user_id=user_id,
            max_messages=request.max_context_messages,
        )
        prefs = store.load_preferences(con, user_id=user_id)

        state = ChatState(
            session_id=request.session_id,
            request_id=request.request_id,
            user_id=user_id,
            user_message=request.message,
            messages=[Message.model_validate(m) for m in ctx.messages],
            summary=ctx.summary,
            prefs=prefs,
        )

        state = route_intent(state, settings=effective_settings)
        yield emit(
            "route",
            {
                "route": state.route,
                "route_reason": state.route_reason,
                "clarifying_question": state.clarifying_question,
            },
        )
        log.info("router.decision", extra={"route": state.route, "has_clarifying": bool(state.clarifying_question)})

        plan = plan_tools(state, settings=effective_settings)
        state.tool_plan = plan.model_dump(mode="json")
        yield emit("plan", {"tool_plan": state.tool_plan})
        log.info("planner.plan_created", extra={"tool_calls": len(plan.calls)})

        for evt in execute_tool_plan_iter(state):
            if isinstance(evt, dict) and "event" in evt and "data" in evt:
                yield emit(str(evt["event"]), dict(evt["data"]))

        output = compose_assistant_output(state=state, tool_plan=state.tool_plan if isinstance(state.tool_plan, dict) else None)
        # This draft is what the LLM composer streams (2.7.2 A).
        draft = output.answer_text.strip() or "(no answer)"
        route = state.route

        # Stream the final composer (A). If unavailable, fall back to streaming the deterministic draft.
        deltas = stream_compose_answer_with_llm(
            route=route,
            user_message=request.message.content,
            draft_answer_text=draft,
            tool_plan=state.tool_plan if isinstance(state.tool_plan, dict) else None,
            tool_traces=[t.model_dump(mode="json") for t in state.tool_traces],
            citations=[c.model_dump(mode="json") for c in output.citations],
            tables=[t.model_dump(mode="json") for t in output.tables],
            prefs=prefs,
            session_summary=ctx.summary,
            settings=effective_settings,
        )

        buf: list[str] = []
        any_streamed = False
        for d in deltas:
            any_streamed = True
            buf.append(str(d))
            yield emit("chunk", {"text": str(d)})

        composed = "".join(buf).strip() if any_streamed else None
        final_text = composed or draft

        # Post-validate analyst numbers: if composer introduced new numbers, discard it.
        if composed and route == "analyst":
            if extract_numbers(composed) - extract_numbers(draft):
                final_text = draft

        assistant_message = Message(role="assistant", content=final_text)

        store.append_messages(
            con,
            session_id=request.session_id,
            user_id=user_id,
            messages=[request.message.model_dump(mode="json"), assistant_message.model_dump(mode="json")],
        )
        store.summarize_if_needed(
            con,
            session_id=request.session_id,
            keep_last_n=effective_settings.memory_keep_last_n,
            summarize_chunk_n=effective_settings.memory_summarize_chunk_n,
            summary_max_chars=effective_settings.memory_summary_max_chars,
            summary_model=effective_settings.summary_model,
            openai_api_key=effective_settings.openai_api_key,
        )

        resp = ChatResponse(
            session_id=request.session_id,
            request_id=request.request_id,
            assistant_message=assistant_message,
            citations=output.citations,
            tables=output.tables,
            charts=output.charts,
            route=state.route,
            tool_traces=state.tool_traces,
            errors=state.errors,
        )
        yield emit("result", resp.model_dump(mode="json"))
        yield emit("done", {"ok": True})
        log.info(
            "chat.stream_turn_completed",
            extra={
                "route": state.route,
                "tool_traces": len(state.tool_traces),
                "tables": len(output.tables),
                "citations": len(output.citations),
            },
        )
    except Exception as exc:
        log.exception("chat.stream_turn_failed")
        yield emit("error", {"message": str(exc)})
        yield emit("done", {"ok": False})
    finally:
        try:
            con.close()
        except Exception:
            pass
