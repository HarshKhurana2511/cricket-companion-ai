from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from cricket_companion.chat_models import ChatState, Route
from cricket_companion.config import Settings, get_settings


class RouteDecision(BaseModel):
    route: Route
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    clarifying_question: str | None = None


@dataclass(frozen=True)
class _HeuristicSignal:
    route: Route
    weight: int
    label: str
    pattern: re.Pattern[str]


_SIGNALS: list[_HeuristicSignal] = [
    _HeuristicSignal(
        route="basic",
        weight=3,
        label="basic_keywords",
        pattern=re.compile(
            r"\b(explain|define|meaning|web|rules|difference|what\s+is|what's|whats|when\s+is|how\s+do)\b",
            re.IGNORECASE,
        ),
    ),
    _HeuristicSignal(
        route="basic",
        weight=6,
        label="basic_define_metric",
        pattern=re.compile(
            r"\b(what\s+is|define|web|meaning\s+of|explain)\s+(batting\s+)?(average|strike\s*rate|economy|run\s*rate|nrr|net\s*run\s*rate)\b",
            re.IGNORECASE,
        ),
    ),
    _HeuristicSignal(
        route="fantasy",
        weight=5,
        label="fantasy_keywords",
        pattern=re.compile(
            r"\b(fantasy|dream11|my11circle|xi|playing\s*xi|captain|vice\s*captain|vc|credits|budget|"
            r"all[-\s]*rounder|wicket[-\s]*keeper|wk)\b",
            re.IGNORECASE,
        ),
    ),
    _HeuristicSignal(
        route="sim",
        weight=5,
        label="scenario_phrase",
        pattern=re.compile(
            r"\b(need(ed)?\s+\d+\s+(off|from)\s+\d+|runs?\s+(needed|required)|target\s+\d+|"
            r"overs?\s+left|balls?\s+left|wickets?\s+(in\s+hand|left)|chase\s+plan)\b",
            re.IGNORECASE,
        ),
    ),
    _HeuristicSignal(
        route="analyst",
        weight=4,
        label="stats_keywords",
        pattern=re.compile(
            r"\b(average|strike\s*rate|economy|dot\s*ball|boundary|wickets?|runs?|since\s+\d{4}|"
            r"vs\.?|versus|compare|top\s+\d+|rank|distribution|trend|correlation)\b",
            re.IGNORECASE,
        ),
    ),
]


def _score_with_heuristics(text: str) -> tuple[dict[Route, int], list[str]]:
    scores: dict[Route, int] = {"basic": 0, "analyst": 0, "sim": 0, "fantasy": 0, "unknown": 0}
    hits: list[str] = []
    for signal in _SIGNALS:
        if signal.pattern.search(text):
            scores[signal.route] += signal.weight
            hits.append(signal.label)
    return scores, hits


def _high_confidence(scores: dict[Route, int], *, min_score: int, margin: int) -> tuple[bool, Route]:
    ordered = sorted(
        ((route, score) for route, score in scores.items() if route != "unknown"),
        key=lambda x: x[1],
        reverse=True,
    )
    best_route, best = ordered[0]
    second = ordered[1][1]

    # High confidence is configurable:
    # - `min_score` ensures at least one strong match
    # - `margin` ensures it beats the next best by a gap (set to 0 for "easy" mode)
    return (best >= min_score and (best - second) >= margin), best_route


def _clarify_from_scores(scores: dict[Route, int]) -> str:
    # Pick the most likely route and ask the minimum fields required to proceed.
    candidates = sorted(
        ((route, score) for route, score in scores.items() if route not in {"basic", "unknown"}),
        key=lambda x: x[1],
        reverse=True,
    )
    if not candidates or candidates[0][1] == 0:
        return "Is your question about cricket rules/explanations, stats/analysis, a match scenario, or fantasy team selection?"

    top_route = candidates[0][0]
    if top_route == "fantasy":
        return "Which match/teams is this for, and what fantasy rules do you want (platform, credits/budget, roles, team limits)?"
    if top_route == "sim":
        return "What’s the exact match situation (format, target/required, overs or balls left, wickets in hand, and any pitch/conditions)?"
    if top_route == "analyst":
        return "What exact stat question do you want (format, date range, teams/players, and the metric to compare)?"

    return "Can you clarify what you want to do (basic explanation vs stats analysis vs scenario simulation vs fantasy XI)?"


def heuristic_route(state: ChatState, *, min_score: int, margin: int) -> RouteDecision:
    text = state.user_message.content
    scores, hits = _score_with_heuristics(text)
    confident, best_route = _high_confidence(scores, min_score=min_score, margin=margin)

    if confident:
        confidence = min(1.0, scores[best_route] / 8.0)
        return RouteDecision(
            route=best_route,
            confidence=confidence,
            reason=f"Heuristics confident (scores={scores}; hits={hits})",
            clarifying_question=None,
        )

    return RouteDecision(
        route="unknown",
        confidence=0.0,
        reason=f"Heuristics not confident (scores={scores}; hits={hits})",
        clarifying_question=_clarify_from_scores(scores),
    )


class _LLMRouteDecision(BaseModel):
    route: Route
    confidence: Literal["low", "medium", "high"]
    reason: str
    clarifying_question: str | None = None


def llm_route(state: ChatState, settings: Settings) -> RouteDecision:
    """
    LLM-based classification. Must remain lightweight: no tool calls, only routing.
    """
    if not settings.openai_api_key:
        return RouteDecision(
            route="unknown",
            confidence=0.0,
            reason="Missing CC_OPENAI_API_KEY; cannot run LLM router.",
            clarifying_question=_clarify_from_scores(_score_with_heuristics(state.user_message.content)[0]),
        )

    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover
        return RouteDecision(
            route="unknown",
            confidence=0.0,
            reason=f"OpenAI SDK import failed: {exc}",
            clarifying_question=_clarify_from_scores(_score_with_heuristics(state.user_message.content)[0]),
        )

    system = (
        "You are an intent router for a cricket assistant. "
        "Choose exactly one route from: basic, analyst, sim, fantasy, unknown.\n"
        "- basic: cricket rules/explanations/general questions\n"
        "- analyst: stats/analysis questions requiring historical data (DuckDB)\n"
        "- sim: match scenario/strategy questions (need target/overs/wkts)\n"
        "- fantasy: fantasy XI/captain/credits/roles/constraints\n"
        "- unknown: ambiguous; ask a short clarifying question\n\n"
        "Return ONLY JSON with keys: route, confidence, reason, clarifying_question.\n"
        "confidence must be one of: low, medium, high.\n"
    )

    user = {
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
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or "{}"
        parsed = _LLMRouteDecision.model_validate_json(content)
    except ValidationError as exc:
        return RouteDecision(
            route="unknown",
            confidence=0.0,
            reason=f"LLM router JSON validation failed: {exc}",
            clarifying_question=_clarify_from_scores(_score_with_heuristics(state.user_message.content)[0]),
        )
    except Exception as exc:
        return RouteDecision(
            route="unknown",
            confidence=0.0,
            reason=f"LLM router call failed: {exc}",
            clarifying_question=_clarify_from_scores(_score_with_heuristics(state.user_message.content)[0]),
        )

    conf_map = {"low": 0.3, "medium": 0.6, "high": 0.9}
    confidence = conf_map.get(parsed.confidence, 0.0)

    # Guardrail: if model says unknown but no question provided, generate a minimal one.
    clarifying = parsed.clarifying_question
    if parsed.route == "unknown" and not clarifying:
        clarifying = _clarify_from_scores(_score_with_heuristics(state.user_message.content)[0])

    return RouteDecision(
        route=parsed.route,
        confidence=confidence,
        reason=parsed.reason,
        clarifying_question=clarifying,
    )


def route_intent(state: ChatState, *, settings: Settings | None = None) -> ChatState:
    """
    Approach 3 router:
    - If heuristics confidence is high, route immediately.
    - Else ask the LLM to classify.
    """
    effective_settings = settings or get_settings()

    # UI / caller override: allow forcing a route via message metadata.
    try:
        md = state.user_message.metadata if isinstance(state.user_message.metadata, dict) else {}
        forced = md.get("force_route")
        if isinstance(forced, str) and forced in {"basic", "analyst", "sim", "fantasy", "unknown"}:
            state.route = forced  # type: ignore[assignment]
            state.route_reason = f"Forced route via message.metadata.force_route={forced!r}"
            state.clarifying_question = None
            return state
        ui_mode = md.get("ui_mode")
        if isinstance(ui_mode, str) and ui_mode.lower() in {"sim", "simulator", "scenario"}:
            state.route = "sim"
            state.route_reason = f"Forced route via message.metadata.ui_mode={ui_mode!r}"
            state.clarifying_question = None
            return state
    except Exception:
        pass

    decision = heuristic_route(
        state,
        min_score=effective_settings.router_heuristic_min_score,
        margin=effective_settings.router_heuristic_margin,
    )
    if decision.route != "unknown":
        state.route = decision.route
        state.route_reason = decision.reason
        state.clarifying_question = None
        return state

    llm_decision = llm_route(state, effective_settings)
    state.route = llm_decision.route
    # Keep heuristic scoring visible even when we fall back to LLM.
    state.route_reason = f"{decision.reason}; LLM: {llm_decision.reason}"
    state.clarifying_question = llm_decision.clarifying_question
    return state
