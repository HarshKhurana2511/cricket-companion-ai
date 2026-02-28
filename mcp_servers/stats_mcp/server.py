from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError

from cricket_companion.output_models import ArtifactSource, TableArtifact
from cricket_companion.schemas import ErrorCode, ToolMeta, ToolResponse


TOOL_NAME = "stats_query"


class StatsQuerySpec(BaseModel):
    """
    Initial stats query spec (skeleton).

    This intentionally mirrors what the planner currently produces, so the MCP
    boundary is stable before we implement real DuckDB querying (2.3/2.4).
    """

    question: str
    format: str | None = None
    player: str | None = None
    team: str | None = None
    since_year: int | None = None
    until_year: int | None = None
    limit: int = Field(default=20, ge=1, le=200)


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
                "description": "Run a read-only cricket stats query (DuckDB-backed; skeleton).",
                "inputSchema": StatsQuerySpec.model_json_schema(),
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
        spec = StatsQuerySpec.model_validate(arguments)
    except ValidationError as exc:
        resp = ToolResponse.failure(
            code=ErrorCode.INVALID_INPUT,
            message="Invalid stats query spec.",
            details=exc.errors(),
            meta=ToolMeta(request_id=request_id),
        )
        return {
            "content": [{"type": "text", "text": json.dumps(resp.model_dump(mode="json"), ensure_ascii=False)}],
            "isError": True,
        }

    # Skeleton response: a table artifact placeholder.
    # Real DuckDB querying and table population will be implemented in 2.3/2.4.
    table = TableArtifact(
        name="stats_result",
        description=f"Skeleton result for: {spec.question}",
        columns=[],
        rows=[],
        truncated=False,
        row_count=0,
        source=ArtifactSource(tool_name="stats-mcp", request_id=request_id, notes="skeleton"),
    )

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    resp = ToolResponse.success(
        data=table.model_dump(mode="json"),
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

    Notes:
    - This is a skeleton to establish the MCP boundary.
    - Client wrappers will be implemented in 2.2.3.
    """
    for raw in _read_requests():
        try:
            req = _parse_request(raw)
        except Exception as exc:
            _write_response(_err(None, -32700, "Parse error", data=str(exc)))
            continue

        # Notifications have no id; ignore them for now.
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

