from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError

from cricket_companion.schemas import ErrorCode, ToolMeta, ToolResponse


TOOL_NAME = "retrieve"

_DEFAULT_MODEL = os.environ.get("CC_RETRIEVAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_RETRIEVAL_MODE = (os.environ.get("CC_RETRIEVAL_MODE", "lexical") or "lexical").strip().lower()


def _repo_root() -> str:
    import pathlib

    return str(pathlib.Path(__file__).resolve().parents[2])


def _retrieval_dir() -> str:
    return os.environ.get("CC_RETRIEVAL_DIR", os.path.join("data", "retrieval"))


@lru_cache(maxsize=1)
def _artifact_paths() -> tuple[str, str, str]:
    retrieval_dir = _retrieval_dir()
    retrieval_dir_abs = os.path.join(_repo_root(), retrieval_dir) if not os.path.isabs(retrieval_dir) else retrieval_dir
    index_path = os.path.join(retrieval_dir_abs, "knowledge.faiss")
    chunks_path = os.path.join(retrieval_dir_abs, "chunks.jsonl")
    web_chunks_path = os.path.join(retrieval_dir_abs, "web_chunks.jsonl")
    return index_path, chunks_path, web_chunks_path


def _load_chunks_jsonl(path: str) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


@lru_cache(maxsize=1)
def _get_index() -> Any:
    index_path, _, _ = _artifact_paths()
    import faiss  # type: ignore

    return faiss.read_index(index_path)


def _get_chunks() -> list[dict[str, Any]]:
    _, chunks_path, web_chunks_path = _artifact_paths()
    chunks: list[dict[str, Any]] = []
    if os.path.exists(chunks_path):
        chunks.extend(_load_chunks_jsonl(chunks_path))
    if os.path.exists(web_chunks_path):
        chunks.extend(_load_chunks_jsonl(web_chunks_path))
    return chunks


@lru_cache(maxsize=1)
def _get_model() -> Any:
    from sentence_transformers import SentenceTransformer  # type: ignore

    return SentenceTransformer(_DEFAULT_MODEL)


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


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
    "could",
    "do",
    "does",
    "example",
    "explain",
    "explained",
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


def _tokenize(text: str) -> set[str]:
    tokens = {m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")}
    # Keep only topic-bearing tokens.
    return {t for t in tokens if len(t) >= 3 and t not in _STOPWORDS}


def _get_chunk_tokens() -> list[set[str]]:
    return [_tokenize(str(c.get("text", ""))) for c in _get_chunks()]


def _lexical_search(query: str, top_k: int) -> tuple[list[float], list[int]]:
    q_tokens = _tokenize(query)
    # Preserve common cricket acronyms explicitly (LBW, DLS, etc.).
    acronyms = [a.lower() for a in re.findall(r"\b[A-Z]{2,6}\b", query or "")]
    q_tokens |= set(acronyms)
    if not q_tokens:
        return ([], [])

    scored: list[tuple[float, int]] = []
    denom = max(1.0, float(len(q_tokens)) ** 0.5)
    for i, tokens in enumerate(_get_chunk_tokens()):
        if not tokens:
            continue
        overlap = len(q_tokens & tokens)
        if overlap == 0:
            continue
        score = overlap / denom
        # Strongly prefer chunks that contain the user's acronym topic (if any).
        if acronyms and any(a in tokens for a in acronyms):
            score += 1.0
        scored.append((score, i))

    scored.sort(key=lambda x: x[0], reverse=True)
    scored = scored[:top_k]
    return ([s for s, _ in scored], [idx for _, idx in scored])


class RetrievalQuery(BaseModel):
    """
    Retrieval query (FAISS-backed).
    """

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


class RetrievalHit(BaseModel):
    chunk_id: str
    source_id: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    query: str
    top_k: int
    hits: list[RetrievalHit] = Field(default_factory=list)


@dataclass(frozen=True)
class _JsonRpcRequest:
    jsonrpc: str
    method: str
    params: dict[str, Any]
    id: Any | None


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
                "name": TOOL_NAME,
                "description": "Retrieve relevant cricket knowledge chunks (FAISS-backed).",
                "inputSchema": RetrievalQuery.model_json_schema(),
            }
        ]
    }


def _tools_call(params: dict[str, Any]) -> dict[str, Any]:
    name = params.get("name")
    arguments = params.get("arguments") or {}

    started = time.perf_counter()
    request_id = str(uuid4())

    if name != TOOL_NAME:
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
        query = RetrievalQuery.model_validate(arguments)
    except ValidationError as exc:
        resp = ToolResponse.failure(
            code=ErrorCode.INVALID_INPUT,
            message="Invalid retrieval query.",
            details=exc.errors(),
            meta=ToolMeta(request_id=request_id),
        )
        return {
            "content": [{"type": "text", "text": json.dumps(resp.model_dump(mode="json"), ensure_ascii=False)}],
            "isError": True,
        }

    index_path, chunks_path, web_chunks_path = _artifact_paths()

    if not os.path.exists(chunks_path) and not os.path.exists(web_chunks_path):
        resp = ToolResponse.failure(
            code=ErrorCode.NOT_FOUND,
            message=(
                "Retrieval chunks not found. Build them first with:\n"
                "- uv run python pipelines/build_knowledge_index.py\n"
                "- uv run python pipelines/build_web_index.py"
            ),
            details={"expected": {"chunks": chunks_path, "web_chunks": web_chunks_path, "index": index_path}},
            meta=ToolMeta(request_id=request_id),
        )
        return {
            "content": [{"type": "text", "text": json.dumps(resp.model_dump(mode="json"), ensure_ascii=False)}],
            "isError": True,
        }

    chunks = _get_chunks()
    top_k = int(query.top_k)

    mode = _RETRIEVAL_MODE
    scores: list[float]
    ids: list[int]

    if mode == "vector":
        if not os.path.exists(index_path):
            resp = ToolResponse.failure(
                code=ErrorCode.NOT_FOUND,
                message="FAISS index not found for vector mode. Build it first with: uv run python pipelines/build_knowledge_index.py",
                details={"expected": {"index": index_path}},
                meta=ToolMeta(request_id=request_id),
            )
            return {
                "content": [{"type": "text", "text": json.dumps(resp.model_dump(mode="json"), ensure_ascii=False)}],
                "isError": True,
            }

        try:
            import numpy as np  # type: ignore

            index = _get_index()
            model = _get_model()
            q_emb = model.encode([query.query], convert_to_numpy=True, normalize_embeddings=True)
            if q_emb.dtype != np.float32:
                q_emb = q_emb.astype(np.float32)
            s2, i2 = index.search(q_emb, top_k)
            scores = [float(x) for x in s2[0].tolist()]
            ids = [int(x) for x in i2[0].tolist()]
        except Exception:
            scores, ids = _lexical_search(query.query, top_k)
            mode = "lexical_fallback"
    else:
        scores, ids = _lexical_search(query.query, top_k)
    hit_rows: list[RetrievalHit] = []
    for score, faiss_id in zip(scores, ids, strict=False):
        if faiss_id < 0 or faiss_id >= len(chunks):
            continue
        c = chunks[int(faiss_id)]
        hit_rows.append(
            RetrievalHit(
                chunk_id=str(c.get("chunk_id", "")),
                source_id=str(c.get("source_id", "")),
                text=str(c.get("text", "")),
                score=float(score),
                metadata=dict(c.get("metadata") or {}),
            )
        )

    result = RetrievalResult(query=query.query, top_k=query.top_k, hits=hit_rows)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    resp = ToolResponse.success(
        data=result.model_dump(mode="json"),
        meta=ToolMeta(request_id=request_id, elapsed_ms=elapsed_ms, source_ids=[f"retrieval_mode:{mode}"]),
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
