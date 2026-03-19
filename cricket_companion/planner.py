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
            # "web:" is an explicit user override to force web tooling even if freshness keywords are absent.
            r"(\bweb\s*:|\b(latest|today|yesterday|current|news|injury|availability|squad|playing\s*xi|update)\b)",
            text,
            flags=re.IGNORECASE,
        )
    )


def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```") and t.endswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _overs_text_to_balls(overs_text: str) -> int | None:
    """
    Parses overs like "10", "10.2" into balls (10*6 + 2).
    """
    s = (overs_text or "").strip()
    if not s:
        return None
    m = re.fullmatch(r"(\d+)(?:\.(\d))?", s)
    if not m:
        return None
    o = int(m.group(1))
    b = int(m.group(2) or "0")
    if b < 0 or b > 5:
        return None
    return o * 6 + b


def _default_max_overs_for_format(fmt: str) -> int:
    f = (fmt or "").strip().upper()
    if f in {"ODI"}:
        return 50
    if f in {"TEST"}:
        # For Test, a "scenario sim" without a session/innings limit isn't well defined. Keep a default,
        # but the sim tool isn't intended for Tests in baseline mode.
        return 90
    # T20/IPL default.
    return 20


def _try_extract_sim_request_from_text(text: str) -> dict[str, Any] | None:
    """
    Heuristic extraction for simulator prompts.

    Supports:
    - JSON payloads that match SimulationRequest
    - "need X off Y (balls|overs) with Z wkts"
    - "chasing T, at R/W in O overs"
    """
    raw = _strip_code_fences(text)

    # 1) Direct JSON payload.
    if raw.startswith("{") and raw.endswith("}"):
        try:
            payload = json.loads(raw)
            from cricket_companion.sim_schemas import SimulationRequest

            SimulationRequest.model_validate(payload)
            return payload
        except Exception:
            pass

    # Detect format hint.
    fmt = "T20"
    if re.search(r"\bipl\b", raw, flags=re.IGNORECASE):
        fmt = "IPL"
    elif re.search(r"\bodi\b", raw, flags=re.IGNORECASE):
        fmt = "ODI"
    elif re.search(r"\btest\b", raw, flags=re.IGNORECASE):
        fmt = "TEST"
    elif re.search(r"\bt20\b", raw, flags=re.IGNORECASE):
        fmt = "T20"

    max_overs = _default_max_overs_for_format(fmt)
    max_balls = max_overs * 6
    sim_model = "historical_blend" if re.search(r"\b(historical|history|based\s+on\s+history)\b", raw, flags=re.IGNORECASE) else "baseline"

    # 2) "need X off Y ..." style (chase mode).
    m_need = re.search(
        r"\bneed(?:s|ed)?\s+(?P<runs>\d+)\s+(?:off|from)\s+(?P<qty>\d+(?:\.\d)?)\s*(?P<unit>balls?|overs?)?\b",
        raw,
        flags=re.IGNORECASE,
    )
    if m_need:
        runs_left = int(m_need.group("runs"))
        qty = m_need.group("qty")
        unit = (m_need.group("unit") or "").lower().strip()
        if unit.startswith("over"):
            balls_left = _overs_text_to_balls(qty)
        else:
            # Default: interpret as balls when unit is omitted.
            balls_left = int(float(qty))
        if balls_left is None:
            return None

        wkts_in_hand = 6
        m_w = re.search(r"\bwith\s+(?P<w>\d+)\s+wickets?\s+(?:in\s+hand|left)\b", raw, flags=re.IGNORECASE)
        if m_w:
            wkts_in_hand = int(m_w.group("w"))
        wkts_in_hand = max(0, min(10, wkts_in_hand))

        balls_elapsed = max(0, max_balls - balls_left)
        wkts_lost = 10 - wkts_in_hand
        payload = {
            "format": fmt,
            "mode": "chase",
            "match_state": {
                "innings": 2,
                "score": {"runs": 0, "wkts": wkts_lost, "balls": balls_elapsed},
                "limits": {"max_overs": max_overs},
                "phase": "unknown",
                "chase": {"target_runs": runs_left, "revised": False},
            },
            "simulation": {"n_sims": 5000, "seed": None, "model": sim_model, "return_distributions": True},
        }
        try:
            from cricket_companion.sim_schemas import SimulationRequest

            SimulationRequest.model_validate(payload)
            return payload
        except Exception:
            return None

    # 3) "chasing T ... R/W in O overs" style.
    m_score = re.search(
        r"\b(?P<runs>\d+)\s*/\s*(?P<wkts>\d+)\s+(?:in\s+)?(?P<overs>\d+(?:\.\d)?)\s*overs?\b",
        raw,
        flags=re.IGNORECASE,
    )
    m_target = re.search(r"\b(?:chasing|target)\s+(?P<t>\d+)\b", raw, flags=re.IGNORECASE)
    if m_score and m_target:
        runs = int(m_score.group("runs"))
        wkts = int(m_score.group("wkts"))
        balls = _overs_text_to_balls(m_score.group("overs"))
        if balls is None:
            return None
        target = int(m_target.group("t"))
        payload = {
            "format": fmt,
            "mode": "chase",
            "match_state": {
                "innings": 2,
                "score": {"runs": runs, "wkts": wkts, "balls": balls},
                "limits": {"max_overs": max_overs},
                "phase": "unknown",
                "chase": {"target_runs": target, "revised": False},
            },
            "simulation": {"n_sims": 5000, "seed": None, "model": sim_model, "return_distributions": True},
        }
        try:
            from cricket_companion.sim_schemas import SimulationRequest

            SimulationRequest.model_validate(payload)
            return payload
        except Exception:
            return None

    return None


def _try_extract_fantasy_request_from_text(text: str) -> dict[str, Any] | None:
    raw = _strip_code_fences(text)
    if raw.startswith("{") and raw.endswith("}"):
        try:
            payload = json.loads(raw)
            from cricket_companion.fantasy_schemas import FantasyRequest

            FantasyRequest.model_validate(payload)
            return payload
        except Exception:
            return None
    return None


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
        "user_prefs": state.prefs,
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
        top_k_pref = state.prefs.get("retrieval_top_k")
        top_k = int(top_k_pref) if isinstance(top_k_pref, int) and 1 <= int(top_k_pref) <= 50 else 5

        calls: list[PlannedToolCall] = [
            PlannedToolCall(
                tool_name="retrieval",
                args={"query": query, "top_k": top_k},
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

        # Apply user preferences only to missing fields (never override explicit user inputs).
        fmt = spec.get("format")
        if fmt in {None, ""} and isinstance(state.prefs.get("default_format"), str):
            spec["format"] = state.prefs["default_format"]
        if spec.get("since_year") is None and isinstance(state.prefs.get("default_since_year"), int):
            spec["since_year"] = state.prefs["default_since_year"]
        if spec.get("until_year") is None and isinstance(state.prefs.get("default_until_year"), int):
            spec["until_year"] = state.prefs["default_until_year"]

        # Limit is always present due to schema defaults; only apply a default limit preference when the user
        # didn't ask for a specific top-N and didn't mention a limit.
        text = state.user_message.content or ""
        if (
            isinstance(state.prefs.get("default_limit"), int)
            and not re.search(r"\btop\s+\d+\b", text, flags=re.IGNORECASE)
            and not re.search(r"\blimit\b", text, flags=re.IGNORECASE)
        ):
            spec["limit"] = int(state.prefs["default_limit"])

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
        payload = _try_extract_sim_request_from_text(state.user_message.content or "")
        if payload is None:
            return ToolPlan(
                route="sim",
                calls=[],
                reason="Simulator route selected but scenario details could not be extracted deterministically.",
                clarifying_question=(
                    "Share a scenario like: `need 30 off 18 with 6 wickets in hand` or "
                    "`chasing 168, at 78/3 in 10 overs` (or paste JSON matching the SimulationRequest schema)."
                ),
            )
        return ToolPlan(
            route="sim",
            calls=[
                PlannedToolCall(
                    tool_name="sim",
                    args=payload,
                    timeout_s=effective_settings.timeout_sim_s,
                    use_cache=False,
                    note="Monte Carlo scenario simulation (baseline)",
                )
            ],
            reason="Sim route: run baseline Monte Carlo simulator.",
        )

    if state.route == "fantasy":
        payload = _try_extract_fantasy_request_from_text(state.user_message.content or "")
        if payload is None:
            return ToolPlan(
                route="fantasy",
                calls=[],
                reason="Fantasy route selected but rules/player pool JSON was not provided.",
                clarifying_question=(
                    "Paste a JSON payload matching the FantasyRequest schema (rules + teams + players). "
                    "You can generate an example with: `python tests/manual_fantasy_schema_test.py`."
                ),
            )
        return ToolPlan(
            route="fantasy",
            calls=[
                PlannedToolCall(
                    tool_name="fantasy",
                    args=payload,
                    timeout_s=effective_settings.timeout_fantasy_s,
                    use_cache=False,
                    note="Fantasy XI optimizer (baseline constraints)",
                )
            ],
            reason="Fantasy route: run optimizer over provided player pool.",
        )

    return ToolPlan(route="unknown", calls=[], reason="Unhandled route.", clarifying_question=state.clarifying_question)
