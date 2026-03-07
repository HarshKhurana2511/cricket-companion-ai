from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from cricket_companion.config import Settings, get_settings


class _ComposerOut(BaseModel):
    answer_text: str = Field(min_length=1)


_NUM_RE = re.compile(r"(?<![A-Za-z0-9_])(-?\d+(?:\.\d+)?)")
_SQLY_RE = re.compile(r"\b(select|insert|update|delete|create|drop|alter)\b", re.IGNORECASE)


def _numbers(text: str) -> set[str]:
    return {m.group(1) for m in _NUM_RE.finditer(text or "")}


def extract_numbers(text: str) -> set[str]:
    """
    Public helper for post-validation (e.g., streaming composer checks).
    """
    return _numbers(text)


def _build_payload(
    *,
    route: str,
    user_message: str,
    draft_answer_text: str,
    tool_plan: dict[str, Any] | None,
    tool_traces: list[dict[str, Any]],
    citations: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    prefs: dict[str, Any],
    session_summary: str | None,
) -> dict[str, Any]:
    # Keep payload small and JSON-serializable.
    small_tables: list[dict[str, Any]] = []
    for t in tables[:1]:
        if not isinstance(t, dict):
            continue
        small_tables.append(
            {
                "table_id": t.get("table_id"),
                "name": t.get("name"),
                "columns": [c.get("name") for c in (t.get("columns") or []) if isinstance(c, dict)],
                "rows_preview": (t.get("rows") or [])[:10],
            }
        )

    retrieval_hits: list[dict[str, Any]] = []
    if route == "basic":
        tr = next((x for x in reversed(tool_traces) if isinstance(x, dict) and x.get("tool_name") == "retrieval"), None)
        resp = (tr or {}).get("response") if isinstance(tr, dict) else None
        if isinstance(resp, dict) and resp.get("ok") is True:
            data = resp.get("data") or {}
            hits = data.get("hits") if isinstance(data, dict) else None
            if isinstance(hits, list):
                for h in hits[:5]:
                    if not isinstance(h, dict):
                        continue
                    meta = h.get("metadata") if isinstance(h.get("metadata"), dict) else {}
                    retrieval_hits.append(
                        {
                            "source_id": h.get("source_id"),
                            "chunk_id": h.get("chunk_id"),
                            "heading": (meta or {}).get("heading"),
                            "text": h.get("text"),
                            "score": h.get("score"),
                        }
                    )

    return {
        "route": route,
        "user_message": user_message,
        "session_summary": session_summary,
        "user_prefs": prefs,
        "tool_plan": tool_plan,
        "tool_traces_brief": [
            {
                "tool_name": tr.get("tool_name"),
                "elapsed_ms": tr.get("elapsed_ms"),
                "ok": (tr.get("response") or {}).get("ok") if isinstance(tr.get("response"), dict) else None,
                "error": tr.get("error"),
            }
            for tr in tool_traces[-5:]
            if isinstance(tr, dict)
        ],
        "draft_answer_text": draft_answer_text,
        "retrieval_hits": retrieval_hits,
        "tables_preview": small_tables,
        "citations": citations[:8],
    }


def compose_answer_with_llm(
    *,
    route: str,
    user_message: str,
    draft_answer_text: str,
    tool_plan: dict[str, Any] | None,
    tool_traces: list[dict[str, Any]],
    citations: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    prefs: dict[str, Any],
    session_summary: str | None,
    settings: Settings | None = None,
) -> str | None:
    """
    Phase 2.7.0: LLM-based answer composer (summarizer) grounded on tool outputs.

    Returns a composed answer_text if safe/valid, else None (caller should fall back to draft).
    """
    effective_settings = settings or get_settings(load_env_file=True)
    if not effective_settings.openai_api_key:
        return None

    try:
        from openai import OpenAI
    except Exception:
        return None

    payload = _build_payload(
        route=route,
        user_message=user_message,
        draft_answer_text=draft_answer_text,
        tool_plan=tool_plan,
        tool_traces=tool_traces,
        citations=citations,
        tables=tables,
        prefs=prefs,
        session_summary=session_summary,
    )

    system = (
        "You are a grounded answer composer for a cricket assistant.\n"
        "Rewrite the draft answer to be clearer and more helpful, but follow these rules strictly:\n"
        "- Do NOT generate SQL or pseudo-SQL.\n"
        "- Do NOT invent facts.\n"
        "- If the route is 'analyst', do NOT introduce new numbers: only reuse numbers that already appear in draft_answer_text.\n"
        "- If the route is 'basic', ground your explanation in retrieval_hits (use them even if the draft is off-topic).\n"
        "- Keep it concise and structured (short paragraphs + bullets).\n"
        "Return ONLY JSON: {\"answer_text\": \"...\"}.\n"
    )

    client = OpenAI(api_key=effective_settings.openai_api_key)
    try:
        resp = client.chat.completions.create(
            model=effective_settings.composer_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or "{}"
        parsed = _ComposerOut.model_validate_json(content)
    except (ValidationError, Exception):
        return None

    answer = parsed.answer_text.strip()
    if not answer:
        return None

    if _SQLY_RE.search(answer):
        return None

    if len(answer) > effective_settings.composer_max_chars:
        answer = answer[: max(0, effective_settings.composer_max_chars - 3)].rstrip() + "..."

    # Analyst safeguard: no new numbers beyond the draft.
    if route == "analyst":
        draft_nums = _numbers(draft_answer_text)
        new_nums = _numbers(answer) - draft_nums
        if new_nums:
            return None

    return answer


def stream_compose_answer_with_llm(
    *,
    route: str,
    user_message: str,
    draft_answer_text: str,
    tool_plan: dict[str, Any] | None,
    tool_traces: list[dict[str, Any]],
    citations: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    prefs: dict[str, Any],
    session_summary: str | None,
    settings: Settings | None = None,
) -> Any:
    """
    Phase 2.7.2: Stream the final LLM composer output (plain text deltas).

    Yields string chunks (deltas). Caller should buffer and validate at the end.
    """
    effective_settings = settings or get_settings(load_env_file=True)
    if not effective_settings.openai_api_key:
        return iter(())

    try:
        from openai import OpenAI
    except Exception:
        return iter(())

    payload = _build_payload(
        route=route,
        user_message=user_message,
        draft_answer_text=draft_answer_text,
        tool_plan=tool_plan,
        tool_traces=tool_traces,
        citations=citations,
        tables=tables,
        prefs=prefs,
        session_summary=session_summary,
    )

    system = (
        "You are a grounded answer composer for a cricket assistant.\n"
        "Rewrite the draft answer to be clearer and more helpful.\n"
        "Rules:\n"
        "- Output ONLY plain text (no JSON).\n"
        "- Do NOT generate SQL or pseudo-SQL.\n"
        "- Do NOT invent facts.\n"
        "- If route='analyst': do NOT introduce new numbers; only reuse numbers already present in draft_answer_text.\n"
        "- If route='basic': ground your explanation in retrieval_hits.\n"
        "- Keep concise and structured.\n"
    )

    client = OpenAI(api_key=effective_settings.openai_api_key)
    try:
        stream = client.chat.completions.create(
            model=effective_settings.composer_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            stream=True,
        )
    except Exception:
        return iter(())

    def _iter() -> Any:
        for event in stream:
            try:
                delta = event.choices[0].delta.content
            except Exception:
                delta = None
            if delta:
                yield delta

    return _iter()
