from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import duckdb
from pydantic import BaseModel, Field, ValidationError

from cricket_companion.output_models import ArtifactSource, TableArtifact
from cricket_companion.schemas import ErrorCode, ToolMeta, ToolResponse
from cricket_companion.stats_templates import (
    TemplateBuildError,
    available_template_ids,
    build_template_plan,
    select_template_id,
)


TOOL_NAME = "stats_query"

# Allowlist: stats queries must only touch these relations (tables/views).
_ALLOWED_RELATIONS: set[str] = {
    "matches",
    "innings",
    "deliveries",
    "players",
    "match_players",
    "ingestion_manifest",
    # Derived views (2.3.3)
    "deliveries_enriched",
    "innings_summary",
    "innings_phase_summary",
}

# Disallow any non-read-only statements and obvious side-effect / file access helpers.
_BANNED_SQL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(insert|update|delete|merge|create|drop|alter|truncate|copy)\b", re.IGNORECASE),
    re.compile(r"\b(attach|detach|load|install|pragma|set)\b", re.IGNORECASE),
    re.compile(r"\b(read_csv|read_parquet|read_json|read_excel|from_csv_auto|from_parquet)\b", re.IGNORECASE),
    re.compile(r"\b(httpfs|s3|gs|azure)\b", re.IGNORECASE),
    re.compile(r"\binformation_schema\b", re.IGNORECASE),
]


def _repo_root() -> str:
    # mcp_servers/stats_mcp/server.py -> repo root
    import pathlib

    return str(pathlib.Path(__file__).resolve().parents[2])


def _default_db_path() -> str:
    # Allow overriding DB path without changing the MCP contract.
    # This should remain a local path (no remote FS).
    return os.environ.get("CC_CRICKET_DB_PATH", os.path.join("data", "duckdb", "cricket_ipl_men.duckdb"))


def _looks_like_read_only_select(sql: str) -> bool:
    stripped = sql.strip()
    if not stripped:
        return False
    lowered = stripped.lstrip().lower()
    return lowered.startswith("select") or lowered.startswith("with")


def _check_sql_is_safe(sql: str) -> None:
    """
    Enforces a conservative SQL safety policy:
    - only SELECT / WITH ... SELECT
    - rejects obvious side-effect statements and file/network helpers
    """
    if not _looks_like_read_only_select(sql):
        raise ValueError("Only SELECT queries are allowed.")
    for pat in _BANNED_SQL_PATTERNS:
        if pat.search(sql):
            raise ValueError(f"Query contains a banned keyword/pattern: {pat.pattern}")


def _extract_relation_tokens(sql: str) -> set[str]:
    """
    Best-effort extraction of relation identifiers after FROM/JOIN.
    This is not a full SQL parser; it's a guardrail to catch accidental use of non-allowlisted relations.
    """
    tokens: set[str] = set()
    # Remove quoted strings to reduce false positives.
    scrubbed = re.sub(r"'[^']*'", "''", sql)
    # Capture identifiers after FROM or JOIN (optionally schema-qualified).
    for m in re.finditer(r"\b(from|join)\s+([a-zA-Z_][\w\.]*)", scrubbed, flags=re.IGNORECASE):
        ident = m.group(2)
        # Strip schema if present.
        name = ident.split(".")[-1]
        tokens.add(name)
    return tokens


def _check_relations_allowlisted(sql: str) -> None:
    refs = _extract_relation_tokens(sql)
    if not refs:
        return
    bad = sorted([r for r in refs if r not in _ALLOWED_RELATIONS])
    if bad:
        raise ValueError(f"Query references non-allowlisted relation(s): {bad}. Allowed: {sorted(_ALLOWED_RELATIONS)}")


def _coerce_json_scalar(v: Any) -> Any:
    if v is None:
        return None
    # pandas/numpy scalars
    try:
        import numpy as np  # type: ignore

        if isinstance(v, (np.generic,)):
            v = v.item()
    except Exception:
        pass

    # DuckDB may return NaN/inf as floats; JSON can't represent them reliably.
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None

    # pandas Timestamp/date-like
    try:
        import pandas as pd  # type: ignore

        if isinstance(v, (pd.Timestamp,)):
            return v.isoformat()
    except Exception:
        pass

    return v


def _dtype_to_datatype(dtype_str: str) -> str:
    """
    Maps pandas/duckdb dtypes to our TableColumn.DataType.
    Falls back to "string".
    """
    d = (dtype_str or "").lower()
    if "bool" in d:
        return "bool"
    if d.startswith("int") or d.startswith("uint"):
        return "int"
    if d.startswith("float") or "double" in d:
        return "float"
    if "datetime" in d:
        return "datetime"
    if d == "date":
        return "date"
    return "string"


def _run_safe_select(
    *,
    db_path: str,
    sql: str,
    params: dict[str, Any] | None,
    max_rows: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    _check_sql_is_safe(sql)
    _check_relations_allowlisted(sql)

    # Enforce a hard row cap by wrapping the query.
    wrapped = f"select * from ({sql}) as q limit {int(max_rows)}"

    # Open read-only to enforce safety at the engine boundary too.
    def _named_params_to_positional(query: str, named: dict[str, Any]) -> tuple[str, list[Any]]:
        values: list[Any] = []

        def _repl(m: re.Match[str]) -> str:
            key = m.group(1)
            if key not in named:
                raise ValueError(f"Missing SQL parameter: {key!r}")
            values.append(named[key])
            return "?"

        q2 = re.sub(r":([a-zA-Z_]\w*)", _repl, query)
        return q2, values

    con = duckdb.connect(db_path, read_only=True)
    try:
        if params:
            param_sql, values = _named_params_to_positional(wrapped, params)
            df = con.execute(param_sql, values).fetchdf()
        else:
            df = con.execute(wrapped).fetchdf()
    finally:
        con.close()

    columns = [{"name": c, "dtype": _dtype_to_datatype(str(df.dtypes.get(c)))} for c in df.columns]
    records: list[dict[str, Any]] = df.to_dict(orient="records")
    rows: list[dict[str, Any]] = []
    for rec in records:
        rows.append({k: _coerce_json_scalar(v) for k, v in rec.items()})
    truncated = len(rows) >= max_rows
    return columns, rows, truncated


class StatsQuerySpec(BaseModel):
    """
    Stats query spec (Phase 2.4.1).

    Recommended policy:
    - templates-only execution (safe SQL built in code, not user-supplied)
    - read-only, allowlisted relations

    Notes:
    - We keep the original fields used by the planner to maintain boundary stability.
    - `template_id` enables Phase 2.4.2 query templates.
    """

    question: str
    format: str | None = None
    player: str | None = None
    team: str | None = None
    since_year: int | None = None
    until_year: int | None = None
    limit: int = Field(default=20, ge=1, le=200)
    template_id: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


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

    # Phase 2.4.2: templates-only execution.
    db_rel = _default_db_path()
    db_path = os.path.join(_repo_root(), db_rel) if not os.path.isabs(db_rel) else db_rel

    if not os.path.exists(db_path):
        resp = ToolResponse.failure(
            code=ErrorCode.NOT_FOUND,
            message=f"DuckDB not found: {db_path}",
            meta=ToolMeta(request_id=request_id),
        )
        return {
            "content": [{"type": "text", "text": json.dumps(resp.model_dump(mode="json"), ensure_ascii=False)}],
            "isError": True,
        }

    # Determine which template to run.
    template_id = spec.template_id
    if not template_id:
        template_id = select_template_id(spec.question, spec.model_dump(mode="json"))

    if not template_id:
        resp = ToolResponse.failure(
            code=ErrorCode.INVALID_INPUT,
            message="Could not choose an analytics template for this question.",
            details={
                "available_templates": available_template_ids(),
                "hint": "Try specifying: death overs / powerplay / strike rate, or pass template_id explicitly.",
            },
            meta=ToolMeta(request_id=request_id),
        )
        return {
            "content": [{"type": "text", "text": json.dumps(resp.model_dump(mode="json"), ensure_ascii=False)}],
            "isError": True,
        }

    try:
        plan = build_template_plan(template_id=template_id, spec=spec.model_dump(mode="json"), db_path=db_path)
    except TemplateBuildError as exc:
        resp = ToolResponse.failure(
            code=ErrorCode.INVALID_INPUT,
            message=str(exc),
            details={"clarifying_question": exc.clarifying_question, "template_id": template_id},
            meta=ToolMeta(request_id=request_id),
        )
        return {
            "content": [{"type": "text", "text": json.dumps(resp.model_dump(mode="json"), ensure_ascii=False)}],
            "isError": True,
        }
    except Exception as exc:
        resp = ToolResponse.failure(
            code=ErrorCode.INTERNAL,
            message=f"Failed to build query template: {exc}",
            meta=ToolMeta(request_id=request_id),
        )
        return {
            "content": [{"type": "text", "text": json.dumps(resp.model_dump(mode="json"), ensure_ascii=False)}],
            "isError": True,
        }

    try:
        columns, rows, truncated = _run_safe_select(
            db_path=db_path,
            sql=plan.sql,
            params=plan.params,
            max_rows=min(int(spec.limit), 200),
        )
    except Exception as exc:
        resp = ToolResponse.failure(
            code=ErrorCode.UPSTREAM_ERROR,
            message=f"Stats query failed: {exc}",
            meta=ToolMeta(request_id=request_id),
        )
        return {
            "content": [{"type": "text", "text": json.dumps(resp.model_dump(mode="json"), ensure_ascii=False)}],
            "isError": True,
        }

    table = TableArtifact(
        name="stats_result",
        description=plan.description,
        columns=columns,
        rows=rows,
        truncated=truncated,
        row_count=len(rows),
        source=ArtifactSource(tool_name="stats-mcp", request_id=request_id, notes=f"duckdb:{os.path.basename(db_path)}"),
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
