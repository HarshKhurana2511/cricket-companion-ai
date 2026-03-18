from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from queue import Queue
from typing import Any


@dataclass(frozen=True)
class JsonRpcError(Exception):
    code: int
    message: str
    data: Any | None = None

    def __str__(self) -> str:  # pragma: no cover
        return f"JSON-RPC error {self.code}: {self.message}"


class StdioJsonRpcClient:
    """
    Minimal JSON-RPC client over stdio.

    - Writes one JSON request per line to stdin
    - Reads one JSON response per line from stdout
    - Uses a background reader thread so calls can time out on Windows
    """

    def __init__(self, cmd: list[str]) -> None:
        self._cmd = cmd
        self._proc: subprocess.Popen[str] | None = None
        self._write_lock = threading.Lock()
        self._next_id = 1
        self._pending: dict[int, Queue[dict[str, Any]]] = {}
        self._pending_lock = threading.Lock()
        self._reader: threading.Thread | None = None

    def start(self) -> None:
        if self._proc is not None:
            return

        env = os.environ.copy()
        # Ensure all MCP child processes emit UTF-8 on stdout so our JSON-RPC reader can decode reliably on Windows.
        # Some libraries print non-ASCII characters using the system code page otherwise.
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")

        self._proc = subprocess.Popen(
            self._cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        self._reader = threading.Thread(target=self._read_loop, name="jsonrpc-stdio-reader", daemon=True)
        self._reader.start()

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            proc.terminate()
        except Exception:
            pass

    def call(self, method: str, params: dict[str, Any] | None = None, *, timeout_s: int = 30) -> dict[str, Any]:
        self.start()
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("JSON-RPC process not started.")

        request_id = self._next_id
        self._next_id += 1

        q: Queue[dict[str, Any]] = Queue(maxsize=1)
        with self._pending_lock:
            self._pending[request_id] = q

        payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}
        line = json.dumps(payload, ensure_ascii=False)
        with self._write_lock:
            self._proc.stdin.write(line + "\n")
            self._proc.stdin.flush()

        try:
            resp = q.get(timeout=timeout_s)
        except Exception as exc:
            with self._pending_lock:
                self._pending.pop(request_id, None)
            raise TimeoutError(f"JSON-RPC call timed out: method={method}") from exc

        # Some servers may include `"error": null` on success. Treat that as success.
        if resp.get("error") is not None:
            err = resp.get("error") or {}
            raise JsonRpcError(code=err.get("code", -1), message=err.get("message", "error"), data=err.get("data"))

        return resp.get("result") or {}

    def _read_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return

        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                resp = json.loads(line)
            except Exception:
                continue

            resp_id = resp.get("id")
            if not isinstance(resp_id, int):
                continue

            with self._pending_lock:
                q = self._pending.pop(resp_id, None)
            if q is not None:
                q.put(resp)


def python_cmd_for_script(script_path: str) -> list[str]:
    return [sys.executable, script_path]
