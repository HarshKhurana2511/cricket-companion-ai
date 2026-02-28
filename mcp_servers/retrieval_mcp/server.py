from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError

from cricket_companion.schemas import ErrorCode, ToolMeta, ToolResponse


TOOL_NAME = "retrieve"


class RetrievalQuery(BaseModel):
    """
    Initial retrieval query (skeleton).

    Real FAISS indexing/search will be implemented in 2.5.
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
                "description": "Retrieve relevant cricket knowledge chunks (FAISS-backed; skeleton).",
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

    # Skeleton behavior: return zero hits (index not built yet).
    result = RetrievalResult(query=query.query, top_k=query.top_k, hits=[])
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    resp = ToolResponse.success(
        data=result.model_dump(mode="json"),
        meta=ToolMeta(request_id=request_id, elapsed_ms=elapsed_ms, source_ids=[]),
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

