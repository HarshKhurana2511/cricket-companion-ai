from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal


InjuryStatus = Literal["fit", "doubtful", "out", "unknown"]


@dataclass(frozen=True)
class NewsSignal:
    injury_status: InjuryStatus
    is_probable_xi: bool | None
    reason: str


_OUT_PATTERNS = [
    r"\bruled\s+out\b",
    r"\bout\s+of\s+(?:the\s+)?(?:match|tournament|ipl)\b",
    r"\bmiss(?:es|ing)\b",
    r"\bwithdraw(?:n|s)\b",
    r"\bwill\s+not\s+play\b",
    r"\b(unavailable|sidelined)\b",
]

_DOUBTFUL_PATTERNS = [
    r"\bdoubtful\b",
    r"\bfitness\s+test\b",
    r"\b(niggle|knock|strain|tightness)\b",
    r"\b(uncertain|touch\s+and\s+go)\b",
    r"\bmonitor(?:ing)?\b",
]

_FIT_PATTERNS = [
    r"\bcleared\b",
    r"\bavailable\b",
    r"\bfit\s+to\s+play\b",
    r"\breturns?\b",
]

_XI_PATTERNS = [
    r"\bplaying\s*xi\b",
    r"\bnamed\s+in\s+(?:the\s+)?xi\b",
    r"\bin\s+the\s+xi\b",
    r"\bin\s+the\s+line[-\s]*up\b",
]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def classify_news_text(text: str) -> NewsSignal | None:
    """
    Deterministic availability extraction from short news text (title/snippet).
    """
    t = _norm(text)
    if not t:
        return None

    for pat in _OUT_PATTERNS:
        if re.search(pat, t, flags=re.IGNORECASE):
            return NewsSignal(injury_status="out", is_probable_xi=False, reason=f"Matched OUT pattern: {pat}")

    for pat in _DOUBTFUL_PATTERNS:
        if re.search(pat, t, flags=re.IGNORECASE):
            # XI unknown when only injury is mentioned.
            return NewsSignal(injury_status="doubtful", is_probable_xi=None, reason=f"Matched DOUBTFUL pattern: {pat}")

    for pat in _XI_PATTERNS:
        if re.search(pat, t, flags=re.IGNORECASE):
            # Playing XI implies availability.
            return NewsSignal(injury_status="fit", is_probable_xi=True, reason=f"Matched XI pattern: {pat}")

    for pat in _FIT_PATTERNS:
        if re.search(pat, t, flags=re.IGNORECASE):
            return NewsSignal(injury_status="fit", is_probable_xi=None, reason=f"Matched FIT pattern: {pat}")

    return None


def apply_news_signal_to_player(player: dict[str, Any], signal: NewsSignal) -> tuple[dict[str, Any], bool]:
    """
    Returns (updated_player, changed).

    We only overwrite `injury_status` when the new status is more informative than the existing one:
      out > doubtful > fit > unknown.
    """
    updated = dict(player)
    changed = False

    def rank(s: str) -> int:
        return {"unknown": 0, "fit": 1, "doubtful": 2, "out": 3}.get(s, 0)

    current_status = str(updated.get("injury_status") or "unknown")
    if rank(signal.injury_status) > rank(current_status):
        updated["injury_status"] = signal.injury_status
        changed = True

    if signal.is_probable_xi is not None:
        cur_prob = updated.get("is_probable_xi")
        if cur_prob is None or cur_prob != signal.is_probable_xi:
            updated["is_probable_xi"] = signal.is_probable_xi
            changed = True

    md = updated.get("metadata")
    if not isinstance(md, dict):
        md = {}
    # Keep only a short reason string; citations are handled at the tool trace level.
    md["news_reason"] = signal.reason
    updated["metadata"] = md
    return updated, changed

