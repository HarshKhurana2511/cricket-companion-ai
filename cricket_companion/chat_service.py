from __future__ import annotations

import json
import re
from typing import Any

from cricket_companion.chat_models import ChatRequest, ChatResponse, ChatState, Message
from cricket_companion.config import Settings, get_settings
from cricket_companion.memory_store import MemoryStore


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
