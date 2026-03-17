from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import socket
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError

from cricket_companion.config import get_settings
from cricket_companion.schemas import Citation, ErrorCode, ToolMeta, ToolResponse, ToolError


TOOL_SEARCH = "web_search"
TOOL_FETCH = "web_fetch"
TOOL_ESPN = "espn_ingest"


_FRESHNESS_RE = re.compile(
    r"\b(latest|today|yesterday|current|news|injury|availability|squad|playing\s*xi|update)\b",
    flags=re.IGNORECASE,
)

_INJURY_INTENT_RE = re.compile(
    r"\b(injury|injured|availability|unavailable|ruled\s*out|doubtful|fitness|niggle|return)\b",
    flags=re.IGNORECASE,
)

_CRICKET_RE = re.compile(
    r"\b(cricket|ipl|indian\s+premier\s+league|bcci|icc|t20|odi|test|ranji|psl|bbl)\b",
    flags=re.IGNORECASE,
)


class WebSearchQuery(BaseModel):
    query: str = Field(min_length=1, max_length=400)
    top_k: int = Field(default=5, ge=1, le=20)

    # Tavily parameters (optional)
    topic: Literal["general", "news", "finance"] | None = None
    search_depth: Literal["basic", "advanced", "fast", "ultra-fast"] | None = None
    time_range: Literal["day", "week", "month", "year"] | None = None
    days: int | None = Field(default=None, ge=1, le=365)
    country: str | None = None
    include_domains: list[str] | None = None
    exclude_domains: list[str] | None = None

    include_answer: bool = False
    include_raw_content: bool = False


class WebSearchResultItem(BaseModel):
    url: str
    title: str | None = None
    snippet: str | None = None
    score: float | None = None
    raw_content: str | None = None


class WebSearchResult(BaseModel):
    query: str
    results: list[WebSearchResultItem] = Field(default_factory=list)
    answer: str | None = None


class WebFetchRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2000)
    mode: Literal["article", "match", "scorecard"] = "article"
    max_chars: int = Field(default=12000, ge=500, le=50000)
    ttl_days: int | None = Field(default=None, ge=1, le=30)
    force_refresh: bool = False


class WebFetchResponseData(BaseModel):
    url: str
    fetched_at: datetime
    payload: dict[str, Any] = Field(default_factory=dict)


class EspnIngestRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2000)
    mode: Literal["metadata", "scorecard"] = "scorecard"
    ttl_days: int | None = Field(default=None, ge=1, le=30)
    force_refresh: bool = False


class EspnIngestData(BaseModel):
    url: str
    fetched_at: datetime
    source: Literal["espncricinfo"] = "espncricinfo"
    metadata: dict[str, Any] = Field(default_factory=dict)
    scorecard: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class _JsonRpcRequest:
    jsonrpc: str
    method: str
    params: dict[str, Any]
    id: Any | None


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _read_requests() -> Any:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        yield line


def _write_response(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _ok(id_value: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id_value, "result": result}


def _err(id_value: Any, code: int, message: str, data: Any | None = None) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": id_value, "error": err}


def _tools_list() -> dict[str, Any]:
    return {
        "tools": [
            {
                "name": TOOL_SEARCH,
                "description": "Web search via Tavily (freshness-aware defaults; returns citations).",
                "inputSchema": WebSearchQuery.model_json_schema(),
            }
            ,
            {
                "name": TOOL_FETCH,
                "description": "Controlled webpage fetch (SSRF-safe, timeouts/size caps, extraction, caching).",
                "inputSchema": WebFetchRequest.model_json_schema(),
            },
            {
                "name": TOOL_ESPN,
                "description": "ESPNcricinfo ingestion (metadata-first; scorecard best-effort; cached fallback on block).",
                "inputSchema": EspnIngestRequest.model_json_schema(),
            },
        ]
    }


def _default_topic(query: str) -> Literal["general", "news"]:
    return "news" if _FRESHNESS_RE.search(query or "") else "general"


def _normalize_query(query: str) -> tuple[str, bool]:
    """
    Expand ambiguous cricket acronyms (especially "IPL") so general-purpose web search stays on-topic.
    Returns (normalized_query, rewritten?).
    """
    q = (query or "").strip()
    if not q:
        return ("", False)

    rewritten = False
    if re.search(r"\bipl\b", q, flags=re.IGNORECASE) and not re.search(r"\bcricket\b", q, flags=re.IGNORECASE):
        q = re.sub(r"\bipl\b", "Indian Premier League (IPL)", q, flags=re.IGNORECASE)
        q = f"{q} cricket"
        rewritten = True

    return (q, rewritten)


def _coerce_snippet(text: str | None, *, max_chars: int = 500) -> str | None:
    if not text:
        return None
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max(0, max_chars - 1)] + "…"


def _looks_injury_related(item: WebSearchResultItem) -> bool:
    hay = f"{item.title or ''}\n{item.snippet or ''}\n{item.url}"
    return bool(_INJURY_INTENT_RE.search(hay))


def _dedupe_by_url(items: list[WebSearchResultItem]) -> list[WebSearchResultItem]:
    seen: set[str] = set()
    out: list[WebSearchResultItem] = []
    for it in items:
        url = (it.url or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(it)
    return out


_TRACKING_PARAMS_PREFIXES = ("utm_",)
_TRACKING_PARAMS_EXACT = {
    "fbclid",
    "gclid",
    "igshid",
    "mc_cid",
    "mc_eid",
    "ref",
    "ref_src",
}


def _normalize_url(url: str) -> str:
    parsed = urlparse((url or "").strip())
    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    kept: list[tuple[str, str]] = []
    for k, v in query_pairs:
        lk = (k or "").lower()
        if any(lk.startswith(p) for p in _TRACKING_PARAMS_PREFIXES):
            continue
        if lk in _TRACKING_PARAMS_EXACT:
            continue
        kept.append((k, v))
    rebuilt = parsed._replace(fragment="", query=urlencode(kept, doseq=True))
    return urlunparse(rebuilt)


def _cache_key(*, url: str, mode: str) -> str:
    normalized = _normalize_url(url)
    raw = f"v1|{mode}|{normalized}".encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()


def _web_cache_dir(settings_cache_dir: Path) -> Path:
    return settings_cache_dir / "web"


def _read_cache(path: Path) -> dict[str, Any] | None:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_cache(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def _is_public_ip(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except Exception:
        return False
    if addr.is_loopback or addr.is_private or addr.is_link_local or addr.is_multicast or addr.is_reserved:
        return False
    return True


def _is_safe_host(host: str) -> bool:
    h = (host or "").strip().lower()
    if not h:
        return False
    if h in {"localhost"} or h.endswith(".local") or h.endswith(".internal"):
        return False

    # Host is an IP literal.
    try:
        ipaddress.ip_address(h)
        return _is_public_ip(h)
    except Exception:
        pass

    try:
        infos = socket.getaddrinfo(h, None, proto=socket.IPPROTO_TCP)
    except Exception:
        return False

    for info in infos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        ip = sockaddr[0]
        if not _is_public_ip(str(ip)):
            return False
    return True


def _extract_html_payload(*, html: str, url: str, mode: str, max_chars: int) -> dict[str, Any]:
    title: str | None = None
    published_at: str | None = None
    text: str = ""

    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        if soup.title and soup.title.string:
            title = " ".join(str(soup.title.string).split())[:300]

        for meta in soup.find_all("meta"):
            try:
                prop = (meta.get("property") or meta.get("name") or "").strip().lower()
                if prop in {"article:published_time", "og:published_time", "publish_date", "pubdate", "date"}:
                    content = (meta.get("content") or "").strip()
                    if content:
                        published_at = content[:80]
                        break
            except Exception:
                continue

        root = soup.find("article") or soup.body or soup
        text = " ".join(root.get_text(" ").split())
    except Exception:
        stripped = re.sub(r"<(script|style|noscript)[^>]*>.*?</\\1>", " ", html, flags=re.IGNORECASE | re.DOTALL)
        stripped = re.sub(r"<[^>]+>", " ", stripped)
        text = " ".join(stripped.split())

    if len(text) > max_chars:
        text = text[: max(0, max_chars - 1)] + "…"

    return {
        "normalized_url": _normalize_url(url),
        "mode": mode,
        "title": title,
        "published_at": published_at,
        "text": text,
    }


def _fetch_html(
    *,
    url: str,
    timeout_s: int,
    max_bytes: int,
    max_redirects: int,
    request_id: str,
) -> tuple[str, datetime, int, str] | ToolError:
    parsed = urlparse((url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ToolError(code=ErrorCode.INVALID_INPUT, message="Invalid URL (must be http/https).")
    if not _is_safe_host(parsed.hostname or ""):
        return ToolError(code=ErrorCode.INVALID_INPUT, message="Blocked URL host by SSRF policy.")

    connect_timeout = min(5.0, float(timeout_s))
    read_timeout = max(1.0, float(timeout_s) - connect_timeout)

    try:
        import requests  # type: ignore
        from requests.exceptions import TooManyRedirects  # type: ignore
    except Exception as exc:
        return ToolError(code=ErrorCode.INTERNAL, message=f"Missing requests dependency: {exc}")

    headers = {
        "User-Agent": "cricket-companion/0.1 (+espn-ingest)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    try:
        with requests.Session() as sess:
            sess.max_redirects = int(max_redirects)
            resp = sess.get(
                url,
                headers=headers,
                timeout=(connect_timeout, read_timeout),
                stream=True,
                allow_redirects=True,
            )
            status = int(resp.status_code)
            content_type = (resp.headers.get("Content-Type") or "").lower()
            if status >= 400:
                code = ErrorCode.UPSTREAM_BLOCKED if status in {401, 403, 429} else ErrorCode.UPSTREAM_ERROR
                return ToolError(code=code, message=f"HTTP {status}", details={"status_code": status})
            if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
                return ToolError(code=ErrorCode.UPSTREAM_ERROR, message=f"Unsupported content type: {content_type or 'unknown'}")

            chunks: list[bytes] = []
            total = 0
            for chunk in resp.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                total += len(chunk)
                if total > int(max_bytes):
                    return ToolError(code=ErrorCode.UPSTREAM_ERROR, message="Response too large.")
                chunks.append(chunk)
            body = b"".join(chunks)
            enc = resp.encoding or "utf-8"
            html = body.decode(enc, errors="replace")
            fetched_at = _utc_now()
            return (html, fetched_at, status, content_type)
    except TooManyRedirects:
        return ToolError(code=ErrorCode.UPSTREAM_ERROR, message="Too many redirects.")
    except TimeoutError as exc:
        return ToolError(code=ErrorCode.TIMEOUT, message=str(exc))
    except Exception as exc:
        return ToolError(code=ErrorCode.UPSTREAM_ERROR, message=f"Fetch failed: {exc}", details={"request_id": request_id})


def _tavily_fallback_for_espn_url(*, url: str, timeout_s: int, request_id: str) -> tuple[dict[str, Any], list[str]] | ToolError:
    """
    ESPNcricinfo can return 403 to direct HTTP clients. When that happens, we attempt a Tavily-powered fallback
    that returns at least metadata + extracted text. This is best-effort and may not include scorecard structure.
    """
    settings = get_settings()
    if not settings.tavily_api_key:
        return ToolError(code=ErrorCode.UPSTREAM_BLOCKED, message="ESPN blocked direct fetch and CC_TAVILY_API_KEY is not set for fallback.")

    try:
        from tavily import TavilyClient  # type: ignore
    except Exception as exc:
        return ToolError(code=ErrorCode.INTERNAL, message=f"Failed to import Tavily client for fallback: {exc}")

    normalized = _normalize_url(url)
    warnings: list[str] = ["Direct fetch blocked; used Tavily fallback content (may be incomplete)."]

    try:
        client = TavilyClient(api_key=settings.tavily_api_key, client_source="cricket-companion:web-mcp")
        raw = client.extract(
            urls=normalized,
            extract_depth="advanced",
            format="markdown",
            timeout=float(timeout_s),
        )
        client.close()
    except Exception as exc:
        raw = None
        try:
            # Fallback to search if extract fails for some reason.
            client = TavilyClient(api_key=settings.tavily_api_key, client_source="cricket-companion:web-mcp")
            raw = client.search(
                query=f"site:espncricinfo.com {normalized}",
                topic="general",
                search_depth="advanced",
                max_results=5,
                include_domains=["espncricinfo.com"],
                include_answer=False,
                include_raw_content="text",
                timeout=float(timeout_s),
            )
            client.close()
        except Exception as exc2:
            return ToolError(
                code=ErrorCode.UPSTREAM_ERROR,
                message=f"Tavily fallback extract+search failed: {exc2}",
                details={"request_id": request_id},
            )

    if not isinstance(raw, dict):
        return ToolError(code=ErrorCode.UPSTREAM_ERROR, message="Tavily fallback returned invalid payload.", details={"request_id": request_id})

    # `extract` returns `results` entries with extracted content.
    results = raw.get("results") or raw.get("data") or []
    picked: dict[str, Any] | None = None
    if isinstance(results, list) and results:
        for r in results:
            if not isinstance(r, dict):
                continue
            r_url = str(r.get("url") or r.get("source_url") or "").strip()
            if r_url and _normalize_url(r_url) == normalized:
                picked = r
                break
        if picked is None:
            picked = results[0] if isinstance(results[0], dict) else None

    # `search` returns `results` with content/raw_content.
    if picked is None and isinstance(raw.get("results"), list) and raw.get("results"):
        r0 = raw["results"][0]
        picked = r0 if isinstance(r0, dict) else None

    if not isinstance(picked, dict):
        return ToolError(code=ErrorCode.UPSTREAM_ERROR, message="Tavily fallback could not pick a result.", details={"request_id": request_id})

    title = str(picked.get("title") or "").strip() or None
    text = picked.get("raw_content") or picked.get("content") or picked.get("text") or ""
    # Preserve newlines if present (markdown extraction) so we can parse tables.
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    # Ensure we don't return runaway content.
    if len(text) > 20000:
        text = text[:19999] + "…"

    payload = {
        "normalized_url": normalized,
        "mode": "tavily_fallback",
        "title": title,
        "published_at": None,
        "text": text,
    }
    return (payload, warnings)


def _parse_md_table(lines: list[str], start_idx: int) -> tuple[list[str], list[list[str]], int] | None:
    """
    Parse a GitHub-flavored Markdown table beginning at start_idx.
    Returns (headers, rows, next_idx_after_table).
    """
    if start_idx < 0 or start_idx >= len(lines):
        return None
    header_line = lines[start_idx].strip()
    if "|" not in header_line:
        return None
    if start_idx + 1 >= len(lines):
        return None
    sep = lines[start_idx + 1].strip()
    if "-" not in sep or "|" not in sep:
        return None

    def split_row(row: str) -> list[str]:
        r = row.strip().strip("|")
        parts = [p.strip() for p in r.split("|")]
        return parts

    headers = split_row(header_line)
    rows: list[list[str]] = []
    i = start_idx + 2
    while i < len(lines):
        raw = lines[i].rstrip("\n")
        if raw.strip() == "":
            break
        if "|" not in raw:
            break
        cells = split_row(raw)
        # Skip separator-like rows.
        if all(re.fullmatch(r"-+", c.replace(" ", "")) for c in cells if c != ""):
            i += 1
            continue
        rows.append(cells)
        i += 1
    return (headers, rows, i)


def _parse_scorecard_from_markdown(markdown_text: str) -> dict[str, Any] | None:
    """
    Best-effort scorecard parser for Tavily markdown extraction.
    Looks for sections like:
      "<Team> Innings"
      "Batting" table
      "Bowling" table
    """
    text = (markdown_text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln != ""]
    if not lines:
        return None

    def clean_team(raw: str) -> str:
        t = (raw or "").strip()
        t = t.strip("*").strip()
        return t

    def looks_like_batting_table_header(s: str) -> bool:
        low = (s or "").strip().lower()
        return low.startswith("| batting") or "| batting" in low

    def looks_like_bowling_table_header(s: str) -> bool:
        low = (s or "").strip().lower()
        return low.startswith("| bowling") or "| bowling" in low

    def has_batting_table_soon(idx: int, *, window: int = 40) -> bool:
        for k in range(idx, min(len(lines), idx + window)):
            if looks_like_batting_table_header(lines[k]):
                return True
        return False

    def infer_table_kind(headers: list[str]) -> Literal["batting", "bowling", "unknown"]:
        hs = [(h or "").strip().lower() for h in headers]
        s = set(hs)
        # Bowling tables usually have these.
        if "econ" in s or "nb" in s or "wd" in s or ("o" in s and "w" in s):
            return "bowling"
        # Batting tables usually have these.
        if "sr" in s and ("4s" in s or "6s" in s) and ("r" in s or "runs" in s):
            return "batting"
        return "unknown"

    innings: list[dict[str, Any]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m_ovs = re.search(r"^(?P<team>.+?)\s*\(\d+\s+ovs\s+maximum\)\b", line, flags=re.IGNORECASE)
        m_inn = re.search(r"^(?P<team>.+?)\s+Innings\b", line, flags=re.IGNORECASE)
        if m_ovs:
            team = clean_team(m_ovs.group("team"))
        elif m_inn and has_batting_table_soon(i + 1):
            team = clean_team(m_inn.group("team"))
        else:
            i += 1
            continue

        inn: dict[str, Any] = {"team": team, "batting": [], "bowling": [], "notes": []}

        # Scan forward for tables.
        j = i + 1
        while j < len(lines):
            # Stop at the next innings marker.
            if (
                re.search(r"^.+\s*\(\d+\s+ovs\s+maximum\)\b", lines[j], flags=re.IGNORECASE)
                or re.search(r"^.+\s+Innings\b", lines[j], flags=re.IGNORECASE)
            ) and j != i:
                break

            # Batting table.
            if looks_like_batting_table_header(lines[j]) or lines[j].lower().startswith("batting"):
                t = _parse_md_table(lines, j) or _parse_md_table(lines, j + 1)
                if t:
                    headers, rows, next_idx = t
                    kind = infer_table_kind(headers)
                    is_bowling_table = kind == "bowling"
                    # Heuristic mapping for ESPN-like tables.
                    for r in rows:
                        if len(r) < 2:
                            continue
                        if len(r) == 1 or (len(r) >= 2 and r[0] == "" and r[1] == ""):
                            continue
                        if is_bowling_table:
                            row_obj = {"raw": r, "bowler": r[0]}
                            for h, v in zip(headers[1:], r[1:], strict=False):
                                key = (h or "").strip().lower()
                                if key == "o":
                                    row_obj["overs"] = v
                                elif key == "m":
                                    row_obj["maidens"] = v
                                elif key in {"r", "runs"}:
                                    row_obj["runs"] = v
                                elif key in {"w", "wkts", "wickets"}:
                                    row_obj["wickets"] = v
                                elif key in {"econ", "economy"}:
                                    row_obj["econ"] = v
                            inn["bowling"].append(row_obj)
                        else:
                            row_obj = {"raw": r}
                            row_obj["player"] = r[0]
                            if len(r) >= 2:
                                row_obj["dismissal"] = r[1]
                            for h, v in zip(headers[2:], r[2:], strict=False):
                                key = (h or "").strip().lower()
                                if key in {"r", "runs"}:
                                    row_obj["runs"] = v
                                elif key in {"b", "balls"}:
                                    row_obj["balls"] = v
                                elif key in {"4s", "fours"}:
                                    row_obj["fours"] = v
                                elif key in {"6s", "sixes"}:
                                    row_obj["sixes"] = v
                                elif key in {"sr", "strike rate"}:
                                    row_obj["sr"] = v
                            inn["batting"].append(row_obj)
                    j = next_idx
                    continue

            # Bowling table.
            if looks_like_bowling_table_header(lines[j]) or lines[j].lower().startswith("bowling"):
                t = _parse_md_table(lines, j) or _parse_md_table(lines, j + 1)
                if t:
                    headers, rows, next_idx = t
                    kind = infer_table_kind(headers)
                    is_batting_table = kind == "batting"
                    for r in rows:
                        if not r:
                            continue
                        if is_batting_table:
                            if len(r) < 2:
                                continue
                            row_obj = {"raw": r, "player": r[0], "dismissal": r[1]}
                            for h, v in zip(headers[2:], r[2:], strict=False):
                                key = (h or "").strip().lower()
                                if key in {"r", "runs"}:
                                    row_obj["runs"] = v
                                elif key in {"b", "balls"}:
                                    row_obj["balls"] = v
                                elif key in {"4s", "fours"}:
                                    row_obj["fours"] = v
                                elif key in {"6s", "sixes"}:
                                    row_obj["sixes"] = v
                                elif key in {"sr", "strike rate"}:
                                    row_obj["sr"] = v
                            inn["batting"].append(row_obj)
                        else:
                            row_obj = {"raw": r, "bowler": r[0]}
                            for h, v in zip(headers[1:], r[1:], strict=False):
                                key = (h or "").strip().lower()
                                if key == "o":
                                    row_obj["overs"] = v
                                elif key == "m":
                                    row_obj["maidens"] = v
                                elif key in {"r", "runs"}:
                                    row_obj["runs"] = v
                                elif key in {"w", "wkts", "wickets"}:
                                    row_obj["wickets"] = v
                                elif key in {"econ", "economy"}:
                                    row_obj["econ"] = v
                            inn["bowling"].append(row_obj)
                    j = next_idx
                    continue

            j += 1

        innings.append(inn)
        i = j

    if not innings:
        return None
    return {"innings": innings}


def _is_espncricinfo_url(url: str) -> bool:
    parsed = urlparse((url or "").strip())
    host = (parsed.hostname or "").lower()
    return host == "www.espncricinfo.com" or host.endswith(".espncricinfo.com") or host == "espncricinfo.com"


def _parse_json_ld(html: str) -> list[Any]:
    objs: list[Any] = []
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html, "lxml")
        for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
            raw = tag.string or tag.get_text() or ""
            raw = raw.strip()
            if not raw:
                continue
            try:
                parsed = json.loads(raw)
            except Exception:
                continue
            if isinstance(parsed, list):
                objs.extend(parsed)
            else:
                objs.append(parsed)
    except Exception:
        pass
    return objs


def _find_next_data(html: str) -> dict[str, Any] | None:
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html, "lxml")
        tag = soup.find("script", attrs={"id": "__NEXT_DATA__"})
        if tag is None:
            return None
        raw = tag.string or tag.get_text() or ""
        raw = raw.strip()
        if not raw:
            return None
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _extract_basic_metadata(*, url: str, html: str) -> dict[str, Any]:
    meta: dict[str, Any] = {"url": url, "normalized_url": _normalize_url(url)}

    title: str | None = None
    description: str | None = None
    published_at: str | None = None

    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html, "lxml")
        if soup.title and soup.title.string:
            title = " ".join(str(soup.title.string).split())[:300]
        for m in soup.find_all("meta"):
            prop = (m.get("property") or m.get("name") or "").strip().lower()
            content = (m.get("content") or "").strip()
            if not content:
                continue
            if prop in {"og:title", "twitter:title"} and not title:
                title = content[:300]
            if prop in {"og:description", "description", "twitter:description"} and not description:
                description = content[:500]
            if prop in {"article:published_time", "og:published_time", "publish_date", "pubdate", "date"} and not published_at:
                published_at = content[:80]
    except Exception:
        pass

    if title:
        meta["title"] = title
    if description:
        meta["description"] = description
    if published_at:
        meta["published_at"] = published_at

    # Teams heuristic from title.
    if title:
        m = re.search(r"(.+?)\s+vs\s+(.+?)(?:\s+[-|]|$)", title, flags=re.IGNORECASE)
        if m:
            meta["teams"] = [m.group(1).strip()[:60], m.group(2).strip()[:60]]

    # JSON-LD SportsEvent extraction (best-effort)
    for obj in _parse_json_ld(html):
        if not isinstance(obj, dict):
            continue
        t = str(obj.get("@type") or "")
        if t.lower() in {"sportsevent", "event"}:
            meta["jsonld"] = {
                k: obj.get(k)
                for k in ["name", "startDate", "endDate", "location", "competitor", "eventStatus", "description"]
                if k in obj
            }
            break

    return meta


def _walk_find_dict_with_keys(obj: Any, required: set[str], *, max_nodes: int = 20000) -> dict[str, Any] | None:
    """
    Generic recursive search helper to locate a nested dict that contains a set of keys.
    Keeps the ESPN parser resilient to upstream schema churn.
    """
    stack: list[Any] = [obj]
    seen = 0
    while stack:
        cur = stack.pop()
        seen += 1
        if seen > max_nodes:
            return None
        if isinstance(cur, dict):
            keys = set(cur.keys())
            if required.issubset(keys):
                return cur
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)
    return None


def _extract_scorecard_from_next_data(next_data: dict[str, Any]) -> dict[str, Any] | None:
    # Try a few common shapes.
    candidate = _walk_find_dict_with_keys(next_data, {"innings"})
    if candidate and isinstance(candidate.get("innings"), list):
        innings = candidate.get("innings")
        slim_innings: list[dict[str, Any]] = []
        for inn in innings:
            if not isinstance(inn, dict):
                continue
            slim_innings.append(
                {
                    k: inn.get(k)
                    for k in [
                        "inningNumber",
                        "team",
                        "teamName",
                        "runs",
                        "wickets",
                        "overs",
                        "runRate",
                        "isComplete",
                    ]
                    if k in inn
                }
            )
        if slim_innings:
            return {"innings": slim_innings}

    # Fallback: look for a match dict with teams and scores.
    match = _walk_find_dict_with_keys(next_data, {"teams"})
    if match:
        out: dict[str, Any] = {}
        teams = match.get("teams")
        if isinstance(teams, list):
            out["teams"] = teams[:4]
        return out or None

    return None


def _espn_cache_path(*, settings: Any, url: str, mode: str) -> Path:
    cache_key = _cache_key(url=url, mode=f"espn:{mode}")
    return _web_cache_dir(settings.cache_dir) / f"espn_{cache_key}.json"


def _espn_ingest(spec: EspnIngestRequest, *, request_id: str) -> ToolResponse[Any]:
    settings = get_settings()
    if not _is_espncricinfo_url(spec.url):
        return ToolResponse.failure(
            ErrorCode.INVALID_INPUT,
            "Only ESPNcricinfo URLs are allowed for espn_ingest.",
            meta=ToolMeta(request_id=request_id),
        )

    now = _utc_now()
    ttl_days = int(spec.ttl_days) if spec.ttl_days is not None else int(settings.web_cache_ttl_days)
    cache_path = _espn_cache_path(settings=settings, url=spec.url, mode=spec.mode)

    cached = _read_cache(cache_path)
    if cached and not spec.force_refresh:
        try:
            expires_at = datetime.fromisoformat(str(cached.get("expires_at")))
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=UTC)
            data = EspnIngestData.model_validate(cached.get("data") or {})
            if expires_at > now:
                return ToolResponse.success(
                    data=data.model_dump(mode="json"),
                    meta=ToolMeta(
                        request_id=request_id,
                        cache={"hit": True, "key": cache_path.name, "expires_at": expires_at},
                        citations=[Citation(url=data.url, fetched_at=data.fetched_at, title=str((data.metadata or {}).get("title") or "") or None)],
                        source_ids=["espn_cache:hit=true"],
                    ),
                )
        except Exception:
            pass

    fetch = _fetch_html(url=spec.url, timeout_s=int(settings.timeout_web_s), max_bytes=1_800_000, max_redirects=3, request_id=request_id)
    if isinstance(fetch, ToolError):
        # On block (403/429/401), attempt Tavily-powered fallback to still return something useful.
        if fetch.code == ErrorCode.UPSTREAM_BLOCKED:
            fb = _tavily_fallback_for_espn_url(url=spec.url, timeout_s=int(settings.timeout_web_s), request_id=request_id)
            if not isinstance(fb, ToolError):
                payload, warnings = fb
                fetched_at = _utc_now()
                scorecard = None
                if spec.mode == "scorecard":
                    scorecard = _parse_scorecard_from_markdown(str(payload.get("text") or ""))
                    if scorecard is None:
                        warnings = list(warnings) + ["Could not parse structured scorecard from fallback content."]

                metadata = {
                    "url": spec.url,
                    "normalized_url": _normalize_url(spec.url),
                    "title": payload.get("title"),
                    "fallback_text": (payload.get("text") or "")[:8000],
                }
                data = EspnIngestData(
                    url=spec.url,
                    fetched_at=fetched_at,
                    metadata={**metadata, "fallback": "tavily"},
                    scorecard=scorecard,
                    warnings=warnings,
                )
                expires_at = fetched_at + timedelta(days=ttl_days)
                _write_cache(
                    cache_path,
                    {
                        "expires_at": expires_at.isoformat(),
                        "data": data.model_dump(mode="json"),
                    },
                )
                return ToolResponse.success(
                    data=data.model_dump(mode="json"),
                    meta=ToolMeta(
                        request_id=request_id,
                        cache={"hit": False, "key": cache_path.name, "expires_at": expires_at},
                        citations=[Citation(url=spec.url, fetched_at=fetched_at, title=str(payload.get("title") or "") or None)],
                        source_ids=["espn_cache:hit=false", "espn:fallback=tavily"],
                    ),
                )

        # Fallback to stale cache when blocked/error.
        if cached:
            try:
                data = EspnIngestData.model_validate(cached.get("data") or {})
                expires_at_raw = cached.get("expires_at")
                expires_at = None
                if expires_at_raw:
                    expires_at = datetime.fromisoformat(str(expires_at_raw))
                    if expires_at.tzinfo is None:
                        expires_at = expires_at.replace(tzinfo=UTC)
                return ToolResponse.success(
                    data=data.model_dump(mode="json"),
                    meta=ToolMeta(
                        request_id=request_id,
                        cache={"hit": True, "key": cache_path.name, "expires_at": expires_at},
                        citations=[Citation(url=data.url, fetched_at=data.fetched_at, title=str((data.metadata or {}).get("title") or "") or None)],
                        source_ids=["espn_cache:stale_fallback=true"],
                    ),
                )
            except Exception:
                pass
        return ToolResponse.failure(fetch.code, fetch.message, details=fetch.details, meta=ToolMeta(request_id=request_id))

    html, fetched_at, status, content_type = fetch
    warnings: list[str] = []

    metadata = _extract_basic_metadata(url=spec.url, html=html)
    metadata.update({"status_code": status, "content_type": content_type})

    scorecard: dict[str, Any] | None = None
    next_data = _find_next_data(html)
    if next_data is None:
        warnings.append("Missing __NEXT_DATA__; scorecard parsing may be incomplete.")
    else:
        if spec.mode == "scorecard":
            scorecard = _extract_scorecard_from_next_data(next_data)
            if scorecard is None:
                warnings.append("Could not extract structured scorecard from page; returning metadata only.")

    data = EspnIngestData(
        url=spec.url,
        fetched_at=fetched_at,
        metadata=metadata,
        scorecard=scorecard if spec.mode == "scorecard" else None,
        warnings=warnings,
    )

    expires_at = fetched_at + timedelta(days=ttl_days)
    _write_cache(
        cache_path,
        {
            "expires_at": expires_at.isoformat(),
            "data": data.model_dump(mode="json"),
        },
    )

    return ToolResponse.success(
        data=data.model_dump(mode="json"),
        meta=ToolMeta(
            request_id=request_id,
            cache={"hit": False, "key": cache_path.name, "expires_at": expires_at},
            citations=[Citation(url=spec.url, fetched_at=fetched_at, title=str(metadata.get("title") or "") or None)],
            source_ids=["espn_cache:hit=false", f"espn:mode={spec.mode}"],
        ),
    )


def _controlled_fetch(spec: WebFetchRequest, *, request_id: str) -> ToolResponse[Any]:
    settings = get_settings()
    now = _utc_now()

    cache_key = _cache_key(url=spec.url, mode=spec.mode)
    cache_path = _web_cache_dir(settings.cache_dir) / f"{cache_key}.json"
    ttl_days = int(spec.ttl_days) if spec.ttl_days is not None else int(settings.web_cache_ttl_days)

    cached = _read_cache(cache_path)
    if cached and not spec.force_refresh:
        try:
            expires_at = datetime.fromisoformat(str(cached.get("expires_at")))
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=UTC)
            data = WebFetchResponseData.model_validate(cached.get("data") or {})
            if expires_at > now:
                return ToolResponse.success(
                    data=data.model_dump(mode="json"),
                    meta=ToolMeta(
                        request_id=request_id,
                        cache={"hit": True, "key": cache_key, "expires_at": expires_at},
                        citations=[
                            Citation(url=data.url, fetched_at=data.fetched_at, title=str((data.payload or {}).get("title") or "") or None)
                        ],
                        source_ids=["web_cache:hit=true"],
                    ),
                )
        except Exception:
            pass

    parsed = urlparse((spec.url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ToolResponse.failure(
            ErrorCode.INVALID_INPUT,
            "Invalid URL (must be http/https).",
            meta=ToolMeta(request_id=request_id),
        )
    if not _is_safe_host(parsed.hostname or ""):
        return ToolResponse.failure(
            ErrorCode.INVALID_INPUT,
            "Blocked URL host by SSRF policy.",
            meta=ToolMeta(request_id=request_id),
        )

    timeout_s = max(1, int(settings.timeout_web_s))
    connect_timeout = min(5.0, float(timeout_s))
    read_timeout = max(1.0, float(timeout_s) - connect_timeout)
    max_bytes = 1_500_000

    try:
        import requests  # type: ignore
        from requests.exceptions import TooManyRedirects  # type: ignore
    except Exception as exc:
        return ToolResponse.failure(
            ErrorCode.INTERNAL,
            f"Missing requests dependency: {exc}",
            meta=ToolMeta(request_id=request_id),
        )

    headers = {
        "User-Agent": "cricket-companion/0.1 (+controlled-fetch)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    started = time.perf_counter()
    try:
        with requests.Session() as sess:
            sess.max_redirects = 3
            resp = sess.get(
                spec.url,
                headers=headers,
                timeout=(connect_timeout, read_timeout),
                stream=True,
                allow_redirects=True,
            )
            status = int(resp.status_code)
            content_type = (resp.headers.get("Content-Type") or "").lower()
            if status >= 400:
                raise RuntimeError(f"HTTP {status}")
            if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
                raise RuntimeError(f"Unsupported content type: {content_type or 'unknown'}")

            chunks: list[bytes] = []
            total = 0
            for chunk in resp.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    raise RuntimeError("Response too large.")
                chunks.append(chunk)
            body = b"".join(chunks)
            enc = resp.encoding or "utf-8"
            html = body.decode(enc, errors="replace")
    except TooManyRedirects:
        return ToolResponse.failure(ErrorCode.UPSTREAM_ERROR, "Too many redirects.", meta=ToolMeta(request_id=request_id))
    except TimeoutError as exc:
        return ToolResponse.failure(ErrorCode.TIMEOUT, str(exc), meta=ToolMeta(request_id=request_id))
    except Exception as exc:
        if cached:
            try:
                data = WebFetchResponseData.model_validate(cached.get("data") or {})
                expires_at_raw = cached.get("expires_at")
                expires_at = None
                if expires_at_raw:
                    expires_at = datetime.fromisoformat(str(expires_at_raw))
                    if expires_at.tzinfo is None:
                        expires_at = expires_at.replace(tzinfo=UTC)
                return ToolResponse.success(
                    data=data.model_dump(mode="json"),
                    meta=ToolMeta(
                        request_id=request_id,
                        cache={"hit": True, "key": cache_key, "expires_at": expires_at},
                        citations=[
                            Citation(url=data.url, fetched_at=data.fetched_at, title=str((data.payload or {}).get("title") or "") or None)
                        ],
                        source_ids=["web_cache:stale_fallback=true"],
                    ),
                )
            except Exception:
                pass
        return ToolResponse.failure(
            ErrorCode.UPSTREAM_ERROR,
            f"Fetch failed: {exc}",
            meta=ToolMeta(request_id=request_id),
        )

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    fetched_at = _utc_now()
    payload = _extract_html_payload(html=html, url=spec.url, mode=spec.mode, max_chars=int(spec.max_chars))
    payload.update({"status_code": status, "content_type": content_type})
    data = WebFetchResponseData(url=spec.url, fetched_at=fetched_at, payload=payload)

    expires_at = fetched_at + timedelta(days=ttl_days)
    _write_cache(
        cache_path,
        {
            "cache_key": cache_key,
            "expires_at": expires_at.isoformat(),
            "data": data.model_dump(mode="json"),
        },
    )

    return ToolResponse.success(
        data=data.model_dump(mode="json"),
        meta=ToolMeta(
            request_id=request_id,
            elapsed_ms=elapsed_ms,
            cache={"hit": False, "key": cache_key, "expires_at": expires_at},
            citations=[Citation(url=spec.url, fetched_at=fetched_at, title=payload.get("title"))],
            source_ids=["web_cache:hit=false"],
        ),
    )


def _tools_call(params: dict[str, Any]) -> dict[str, Any]:
    name = params.get("name")
    arguments = params.get("arguments") or {}

    started = time.perf_counter()
    request_id = str(uuid4())

    if name == TOOL_FETCH:
        try:
            fetch_spec = WebFetchRequest.model_validate(arguments)
        except ValidationError as exc:
            resp = ToolResponse.failure(
                code=ErrorCode.INVALID_INPUT,
                message="Invalid web fetch request.",
                details=exc.errors(),
                meta=ToolMeta(request_id=request_id),
            )
            return {
                "content": [{"type": "text", "text": json.dumps(resp.model_dump(mode="json"), ensure_ascii=False)}],
                "isError": True,
            }

        resp = _controlled_fetch(fetch_spec, request_id=request_id)
        return {
            "content": [{"type": "text", "text": json.dumps(resp.model_dump(mode="json"), ensure_ascii=False)}],
            "isError": not bool(resp.ok),
        }

    if name == TOOL_ESPN:
        try:
            espn_spec = EspnIngestRequest.model_validate(arguments)
        except ValidationError as exc:
            resp = ToolResponse.failure(
                code=ErrorCode.INVALID_INPUT,
                message="Invalid ESPN ingestion request.",
                details=exc.errors(),
                meta=ToolMeta(request_id=request_id),
            )
            return {
                "content": [{"type": "text", "text": json.dumps(resp.model_dump(mode="json"), ensure_ascii=False)}],
                "isError": True,
            }

        resp = _espn_ingest(espn_spec, request_id=request_id)
        return {
            "content": [{"type": "text", "text": json.dumps(resp.model_dump(mode="json"), ensure_ascii=False)}],
            "isError": not bool(resp.ok),
        }

    if name != TOOL_SEARCH:
        resp = ToolResponse.failure(
            code=ErrorCode.NOT_FOUND,
            message=f"Unknown tool: {name!r}",
            meta=ToolMeta(request_id=request_id),
        )
        return {
            "content": [{"type": "text", "text": json.dumps(resp.model_dump(mode="json"), ensure_ascii=False)}],
            "isError": True,
        }

    try:
        spec = WebSearchQuery.model_validate(arguments)
    except ValidationError as exc:
        resp = ToolResponse.failure(
            code=ErrorCode.INVALID_INPUT,
            message="Invalid web search request.",
            details=exc.errors(),
            meta=ToolMeta(request_id=request_id),
        )
        return {
            "content": [{"type": "text", "text": json.dumps(resp.model_dump(mode="json"), ensure_ascii=False)}],
            "isError": True,
        }

    settings = get_settings()
    if not settings.tavily_api_key:
        resp = ToolResponse.failure(
            code=ErrorCode.INVALID_INPUT,
            message="Missing CC_TAVILY_API_KEY (required for Tavily web search).",
            meta=ToolMeta(request_id=request_id),
        )
        return {
            "content": [{"type": "text", "text": json.dumps(resp.model_dump(mode="json"), ensure_ascii=False)}],
            "isError": True,
        }

    try:
        from tavily import TavilyClient  # type: ignore
        from tavily.errors import (  # type: ignore
            BadRequestError,
            ForbiddenError,
            InvalidAPIKeyError,
            MissingAPIKeyError,
            TimeoutError as TavilyTimeoutError,
            UsageLimitExceededError,
        )
    except Exception as exc:
        resp = ToolResponse.failure(
            code=ErrorCode.INTERNAL,
            message=f"Failed to import Tavily client: {exc}",
            meta=ToolMeta(request_id=request_id),
        )
        return {
            "content": [{"type": "text", "text": json.dumps(resp.model_dump(mode="json"), ensure_ascii=False)}],
            "isError": True,
        }

    topic = spec.topic or _default_topic(spec.query)
    timeout_s = max(1, int(settings.timeout_web_s))
    fetched_at = _utc_now()

    try:
        normalized_query, rewritten = _normalize_query(spec.query)
        client = TavilyClient(api_key=settings.tavily_api_key, client_source="cricket-companion:web-mcp")
        raw = client.search(
            query=normalized_query,
            topic=topic,
            search_depth=spec.search_depth,
            time_range=spec.time_range,
            days=spec.days,
            country=spec.country,
            include_domains=spec.include_domains,
            exclude_domains=spec.exclude_domains,
            max_results=int(spec.top_k),
            include_answer=bool(spec.include_answer),
            include_raw_content=("text" if spec.include_raw_content else False),
            timeout=float(timeout_s),
        )
        retry_raw: dict[str, Any] | None = None
        if _INJURY_INTENT_RE.search(spec.query or ""):
            results1 = raw.get("results") or []
            has_injury_hit = False
            for r in results1:
                if not isinstance(r, dict):
                    continue
                hay = f"{r.get('title') or ''}\n{r.get('content') or ''}\n{r.get('url') or ''}"
                if _INJURY_INTENT_RE.search(hay):
                    has_injury_hit = True
                    break
            if not has_injury_hit:
                retry_query = normalized_query
                if not re.search(r"\binjur", retry_query, flags=re.IGNORECASE):
                    retry_query = f"{retry_query} injury update"
                retry_raw = client.search(
                    query=retry_query,
                    topic=topic,
                    search_depth=spec.search_depth,
                    time_range=spec.time_range,
                    days=spec.days,
                    country=spec.country,
                    include_domains=spec.include_domains,
                    exclude_domains=spec.exclude_domains,
                    max_results=int(spec.top_k),
                    include_answer=bool(spec.include_answer),
                    include_raw_content=("text" if spec.include_raw_content else False),
                    timeout=float(timeout_s),
                )
        client.close()
    except TavilyTimeoutError as exc:
        resp = ToolResponse.failure(
            code=ErrorCode.TIMEOUT,
            message=str(exc),
            meta=ToolMeta(request_id=request_id),
        )
        return {
            "content": [{"type": "text", "text": json.dumps(resp.model_dump(mode="json"), ensure_ascii=False)}],
            "isError": True,
        }
    except (MissingAPIKeyError, InvalidAPIKeyError) as exc:
        resp = ToolResponse.failure(
            code=ErrorCode.UPSTREAM_ERROR,
            message=str(exc),
            details={"hint": "Check CC_TAVILY_API_KEY."},
            meta=ToolMeta(request_id=request_id),
        )
        return {
            "content": [{"type": "text", "text": json.dumps(resp.model_dump(mode="json"), ensure_ascii=False)}],
            "isError": True,
        }
    except UsageLimitExceededError as exc:
        resp = ToolResponse.failure(
            code=ErrorCode.UPSTREAM_ERROR,
            message=str(exc),
            details={"hint": "Tavily usage/rate limit exceeded."},
            meta=ToolMeta(request_id=request_id),
        )
        return {
            "content": [{"type": "text", "text": json.dumps(resp.model_dump(mode="json"), ensure_ascii=False)}],
            "isError": True,
        }
    except ForbiddenError as exc:
        resp = ToolResponse.failure(
            code=ErrorCode.UPSTREAM_BLOCKED,
            message=str(exc),
            meta=ToolMeta(request_id=request_id),
        )
        return {
            "content": [{"type": "text", "text": json.dumps(resp.model_dump(mode="json"), ensure_ascii=False)}],
            "isError": True,
        }
    except BadRequestError as exc:
        resp = ToolResponse.failure(
            code=ErrorCode.INVALID_INPUT,
            message=str(exc),
            meta=ToolMeta(request_id=request_id),
        )
        return {
            "content": [{"type": "text", "text": json.dumps(resp.model_dump(mode="json"), ensure_ascii=False)}],
            "isError": True,
        }
    except Exception as exc:
        resp = ToolResponse.failure(
            code=ErrorCode.INTERNAL,
            message=f"Tavily search failed: {exc}",
            meta=ToolMeta(request_id=request_id),
        )
        return {
            "content": [{"type": "text", "text": json.dumps(resp.model_dump(mode="json"), ensure_ascii=False)}],
            "isError": True,
        }

    results_raw = raw.get("results") or []
    if "retry_raw" in locals() and isinstance(retry_raw, dict):
        results_raw = list(results_raw) + list(retry_raw.get("results") or [])
    items: list[WebSearchResultItem] = []
    citations: list[Citation] = []
    for r in results_raw:
        if not isinstance(r, dict):
            continue
        url = str(r.get("url") or "").strip()
        if not url:
            continue
        title = str(r.get("title") or "").strip() or None
        content = r.get("content")
        score = r.get("score")
        raw_content = r.get("raw_content") if spec.include_raw_content else None
        item = WebSearchResultItem(
            url=url,
            title=title,
            snippet=_coerce_snippet(str(content) if content is not None else None),
            score=float(score) if isinstance(score, (int, float)) else None,
            raw_content=str(raw_content) if raw_content is not None else None,
        )
        items.append(item)
        citations.append(Citation(url=url, fetched_at=fetched_at, title=title))

    # Post-filter for cricket intent so generic "injury updates" queries don't drift to other sports.
    # If filtering removes everything, keep the original list (better than empty).
    if _CRICKET_RE.search(spec.query or ""):
        filtered_items: list[WebSearchResultItem] = []
        filtered_citations: list[Citation] = []
        for item, citation in zip(items, citations, strict=False):
            hay = f"{item.title or ''}\n{item.snippet or ''}\n{item.url}"
            if _CRICKET_RE.search(hay):
                filtered_items.append(item)
                filtered_citations.append(citation)
        if filtered_items:
            items = filtered_items
            citations = filtered_citations

    # If the user asked for injuries/availability, prefer results that look injury-related.
    if _INJURY_INTENT_RE.search(spec.query or "") and items:
        injury_items = [it for it in items if _looks_injury_related(it)]
        non_injury_items = [it for it in items if not _looks_injury_related(it)]
        if injury_items:
            items = injury_items + non_injury_items

    # Dedupe (Tavily retry can create overlaps) and regenerate citations in the final order.
    items = _dedupe_by_url(items)
    citations = [Citation(url=it.url, fetched_at=fetched_at, title=it.title) for it in items]

    answer = raw.get("answer") if bool(spec.include_answer) else None
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    data = WebSearchResult(query=spec.query, results=items, answer=str(answer) if answer else None)
    source_ids = [f"tavily:topic={topic}", f"tavily:top_k={int(spec.top_k)}"]
    if "rewritten" in locals() and rewritten:
        source_ids.append("tavily:query_rewritten=true")
    if "retry_raw" in locals() and isinstance(retry_raw, dict):
        source_ids.append("tavily:retry_injury=true")
    resp = ToolResponse.success(
        data=data.model_dump(mode="json"),
        meta=ToolMeta(
            request_id=request_id,
            elapsed_ms=elapsed_ms,
            citations=citations,
            source_ids=source_ids,
        ),
    )
    return {
        "content": [{"type": "text", "text": json.dumps(resp.model_dump(mode="json"), ensure_ascii=False)}],
        "isError": False,
    }


def _parse_request(raw: str) -> _JsonRpcRequest:
    obj = json.loads(raw)
    return _JsonRpcRequest(
        jsonrpc=obj.get("jsonrpc", "2.0"),
        method=obj["method"],
        params=obj.get("params") or {},
        id=obj.get("id"),
    )


def main() -> None:
    """
    Minimal MCP-like JSON-RPC server over stdio.

    Supported methods:
    - tools/list
    - tools/call
    """
    for raw in _read_requests():
        try:
            req = _parse_request(raw)
        except Exception as exc:
            _write_response(_err(None, -32700, "Parse error", data=str(exc)))
            continue

        if req.id is None:
            continue

        try:
            if req.method == "tools/list":
                _write_response(_ok(req.id, _tools_list()))
            elif req.method == "tools/call":
                _write_response(_ok(req.id, _tools_call(req.params)))
            else:
                _write_response(_err(req.id, -32601, f"Method not found: {req.method}"))
        except Exception as exc:  # pragma: no cover
            _write_response(_err(req.id, -32603, "Internal error", data=str(exc)))


if __name__ == "__main__":
    main()
