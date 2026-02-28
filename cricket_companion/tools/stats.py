from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cricket_companion.config import Settings, get_settings
from cricket_companion.schemas import ErrorCode, ToolMeta, ToolResponse
from cricket_companion.tools.base import JsonRpcError, StdioJsonRpcClient, python_cmd_for_script


class StatsMcpClient:
    def __init__(self, *, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        repo_root = Path(__file__).resolve().parents[2]
        script = repo_root / "mcp_servers" / "stats_mcp" / "server.py"
        self._rpc = StdioJsonRpcClient(python_cmd_for_script(str(script)))

    def close(self) -> None:
        self._rpc.close()

    def list_tools(self) -> dict[str, Any]:
        return self._rpc.call("tools/list", {}, timeout_s=self._settings.timeout_stats_s)

    def query(self, spec: dict[str, Any]) -> ToolResponse[Any]:
        try:
            result = self._rpc.call(
                "tools/call",
                {"name": "stats_query", "arguments": spec},
                timeout_s=self._settings.timeout_stats_s,
            )
        except TimeoutError as exc:
            return ToolResponse.failure(ErrorCode.TIMEOUT, str(exc), meta=ToolMeta())
        except JsonRpcError as exc:
            return ToolResponse.failure(ErrorCode.UPSTREAM_ERROR, str(exc), details=exc.data, meta=ToolMeta())

        content = ((result.get("content") or [{}])[0] or {}).get("text") or "{}"
        try:
            return ToolResponse[Any].model_validate_json(content)
        except Exception as exc:
            return ToolResponse.failure(ErrorCode.INTERNAL, f"Failed to parse tool response: {exc}", meta=ToolMeta())

