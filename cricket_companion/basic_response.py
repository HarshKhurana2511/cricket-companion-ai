from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from cricket_companion.output_models import AssistantOutput, TableArtifact
from cricket_companion.schemas import Citation


def _utc_now() -> datetime:
    return datetime.now(UTC)


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "does",
    "example",
    "explain",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "its",
    "me",
    "of",
    "on",
    "or",
    "quick",
    "simple",
    "simpler",
    "terms",
    "that",
    "the",
    "this",
    "to",
    "used",
    "use",
    "when",
    "what",
    "with",
    "you",
    "your",
}


def _slug(text: str) -> str:
    s = (text or "").strip().lower()
    s = re.sub(r"[^a-z0-9\s]+", "-", s)
    s = re.sub(r"\s+", "-", s).strip("-")
    return s or "section"


def _extract_hits(tool_response: dict[str, Any]) -> list[dict[str, Any]]:
    data = tool_response.get("data") or {}
    hits = data.get("hits") if isinstance(data, dict) else None
    if not isinstance(hits, list):
        return []
    return [h for h in hits if isinstance(h, dict)]


def _make_citation(hit: dict[str, Any]) -> Citation:
    source_id = str(hit.get("source_id") or "")
    heading = str(((hit.get("metadata") or {}) if isinstance(hit.get("metadata"), dict) else {}).get("heading") or "")
    anchor = _slug(heading) if heading else ""
    url = f"doc:{source_id}" + (f"#{anchor}" if anchor else "")
    title = heading or None
    return Citation(url=url, fetched_at=_utc_now(), title=title)


def _shorten(text: str, *, max_chars: int) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[: max(0, max_chars - 3)].rstrip() + "..."


def _topic_terms(question: str) -> list[str]:
    """
    Extract topic-bearing terms from the question for deterministic reranking.

    Goal: prefer chunks that mention the actual topic (e.g., "LBW") over generic
    glossary sections that match only on common words like "explain" or "simple".
    """
    q = (question or "").strip()
    if not q:
        return []

    # Keep common cricket abbreviations as-is (LBW, DLS, etc.).
    acronyms = re.findall(r"\b[A-Z]{2,6}\b", q)

    # Basic tokenization for other terms.
    tokens = re.findall(r"[A-Za-z0-9]+", q.lower())
    tokens = [t for t in tokens if len(t) >= 3 and t not in _STOPWORDS]

    terms: list[str] = []
    seen: set[str] = set()
    for t in acronyms + tokens:
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        terms.append(t)
    return terms[:8]


def _rerank_hits(question: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    terms = [t.lower() for t in _topic_terms(question)]
    if not terms or not hits:
        return hits

    def score(hit: dict[str, Any]) -> tuple[int, float]:
        meta = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
        heading = str((meta or {}).get("heading") or "")
        blob = " ".join(
            [
                str(hit.get("source_id") or ""),
                heading,
                str(hit.get("text") or ""),
            ]
        ).lower()
        term_matches = sum(1 for t in terms if t and t in blob)
        base = hit.get("score")
        base_score = float(base) if isinstance(base, (int, float)) else 0.0
        return (term_matches, base_score)

    # Stable, deterministic: sort by (term_matches, base_score) desc, keep original order as final tiebreaker.
    scored = list(enumerate(hits))
    scored.sort(key=lambda pair: (score(pair[1])[0], score(pair[1])[1], -pair[0]), reverse=True)
    return [h for _, h in scored]


def _has_acronym_topic(question: str) -> str | None:
    q = (question or "").strip()
    if not q:
        return None
    acronyms = re.findall(r"\b[A-Z]{2,6}\b", q)
    return acronyms[0] if acronyms else None


def _term_matches_in_hit(hit: dict[str, Any], terms: list[str]) -> int:
    meta = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
    heading = str((meta or {}).get("heading") or "")
    blob = " ".join([str(hit.get("source_id") or ""), heading, str(hit.get("text") or "")]).lower()
    return sum(1 for t in terms if t and t in blob)


def _low_confidence(question: str, hits: list[dict[str, Any]]) -> bool:
    """
    Conservative "don't answer" gate for basic-mode composition.

    We avoid producing a confident-looking answer if retrieval hits don't appear to
    mention the user's actual topic.
    """
    if not hits:
        return True

    # If the question contains a strong acronym topic (LBW/DLS/etc.), require that
    # the top hit mentions it.
    acronym = _has_acronym_topic(question)
    if acronym:
        top_blob = " ".join(
            [
                str(hits[0].get("source_id") or ""),
                str(((hits[0].get("metadata") or {}) if isinstance(hits[0].get("metadata"), dict) else {}).get("heading") or ""),
                str(hits[0].get("text") or ""),
            ]
        ).lower()
        return acronym.lower() not in top_blob

    terms = [t.lower() for t in _topic_terms(question)]
    if not terms:
        # No topic-bearing terms extracted; rely on ambiguity rules instead of confidence gating.
        return False

    best = max(_term_matches_in_hit(h, terms) for h in hits[:5])
    return best <= 0


def _clarifying_question_if_needed(question: str) -> str | None:
    """
    Deterministic ambiguity handling for basic cricket Q&A.

    If a question is under-specified in a way that materially changes the answer,
    ask a short clarifying question instead of guessing.
    """
    q = (question or "").strip()
    ql = q.lower()

    def has_any(*needles: str) -> bool:
        return any(n in ql for n in needles)

    # Metrics ambiguity.
    if re.search(r"\b(avg|average)\b", ql) and not has_any("batting", "bowling", "economy", "strike rate", "sr"):
        return "Do you mean **batting average** or **bowling average** (or something like economy/strike rate)?"
    if ("strike rate" in ql or re.search(r"\bsr\b", ql)) and not has_any("batting", "bowling"):
        return "Do you mean **batting** strike rate or **bowling** strike rate? (They’re different.)"
    if "economy" in ql and not has_any("bowling", "bowler"):
        return "Do you mean **bowling economy rate** (runs conceded per over based on legal balls)?"

    # Format-specific rules.
    if "powerplay" in ql and not has_any("t20", "odi", "ipl", "bbl", "hundred"):
        return "Which format are you asking about (T20 or ODI)? Powerplay rules differ by format/league."

    # Judgment calls need scenario details.
    if re.search(r"\bis this\b", ql) and ("lbw" in ql or "out" in ql):
        return (
            "To judge that, I need a bit more detail: where did it pitch, where did it hit the pad, "
            "was a shot offered, and would it have hit the stumps?"
        )

    return None


def _answer_from_hits(question: str, hits: list[dict[str, Any]]) -> str:
    if not hits:
        return (
            "I couldn't find this in my local cricket knowledge base yet. "
            "Can you clarify what format/rule scenario you mean?"
        )

    # Prefer a small number of high-signal chunks.
    use_hits = hits[:3]

    lines: list[str] = []
    lines.append("Answer (grounded on local knowledge base):")

    # Use the top hit as the main explanation; add supporting bullets from other hits.
    top = use_hits[0]
    top_text = _shorten(str(top.get("text") or ""), max_chars=900)
    lines.append("")
    lines.append(top_text)

    if len(use_hits) > 1:
        lines.append("")
        lines.append("Related notes:")
        for h in use_hits[1:]:
            meta = h.get("metadata") if isinstance(h.get("metadata"), dict) else {}
            heading = (meta or {}).get("heading")
            preview = _shorten(str(h.get("text") or ""), max_chars=220).replace("\n", " ")
            if heading:
                lines.append(f"- {heading}: {preview}")
            else:
                lines.append(f"- {preview}")

    lines.append("")
    lines.append("If you want, tell me the format (T20/ODI/Test) and I’ll tailor the explanation.")
    return "\n".join(lines)


def build_basic_output(
    *,
    question: str,
    retrieval_tool_response: dict[str, Any],
) -> AssistantOutput:
    hits = _rerank_hits(question, _extract_hits(retrieval_tool_response))

    clarifying = _clarifying_question_if_needed(question)
    if clarifying:
        return AssistantOutput(
            answer_text=clarifying,
            citations=[],
            tables=[],
            charts=[],
            warnings=["Asked a clarifying question instead of guessing."],
            assumptions=["Responses are grounded on the local docs/knowledge corpus."],
        )

    if _low_confidence(question, hits):
        return AssistantOutput(
            answer_text=(
                "I couldn't find a reliable answer for that in my local cricket knowledge base yet. "
                "Can you rephrase or tell me the exact rule/topic (and format: T20/ODI/Test)?"
            ),
            citations=[],
            tables=[],
            charts=[],
            warnings=["Low retrieval confidence; no answer composed."],
            assumptions=["Responses are grounded on the local docs/knowledge corpus."],
        )

    citations = [_make_citation(h) for h in hits[:3]]
    answer = _answer_from_hits(question, hits)

    # Basic mode usually doesn't produce tables/charts.
    return AssistantOutput(
        answer_text=answer,
        citations=citations,
        tables=[],
        charts=[],
        warnings=[],
        assumptions=["Responses are grounded on the local docs/knowledge corpus."],
    )
