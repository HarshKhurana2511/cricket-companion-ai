from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError

from cricket_companion.fantasy_schemas import FantasyRequest, PlayerPoolEntry
from cricket_companion.schemas import ErrorCode, ToolMeta, ToolResponse


TOOL_NAME = "fantasy_optimize"


class _ToolsCallArgs(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class _JsonRpcRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: int | None = None
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class _JsonRpcResponse:
    jsonrpc: str
    id: int | None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


def _ok(req_id: int, result: dict[str, Any]) -> dict[str, Any]:
    return _JsonRpcResponse(jsonrpc="2.0", id=req_id, result=result).__dict__


def _err(req_id: int | None, code: int, message: str, *, data: Any | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        payload["data"] = data
    return _JsonRpcResponse(jsonrpc="2.0", id=req_id, error=payload).__dict__


def _read_requests() -> Any:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        yield line


def _write_response(obj: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _tools_list() -> dict[str, Any]:
    schema = FantasyRequest.model_json_schema()
    return {
        "tools": [
            {
                "name": TOOL_NAME,
                "description": "Fantasy XI optimizer (baseline constraints: roles, budget, team limits, C/VC).",
                "inputSchema": schema,
            }
        ]
    }


def _role_key(role: str) -> str:
    r = (role or "unknown").strip().lower()
    if r in {"wk", "bat", "ar", "bowl"}:
        return r
    return "unknown"


def _project_points(p: PlayerPoolEntry) -> float:
    # Deterministic fallback projection if expected_points is missing.
    if isinstance(p.expected_points, (int, float)):
        base = float(p.expected_points)
    else:
        role = _role_key(p.role)
        base = 42.0 if role == "ar" else 36.0 if role == "wk" else 34.0 if role == "bat" else 33.0 if role == "bowl" else 30.0

    if p.is_probable_xi is True:
        base += 4.0
    if p.injury_status == "doubtful":
        base -= 10.0
    elif p.injury_status == "out":
        base -= 100.0
    return max(0.0, base)


def _optimize(req: FantasyRequest) -> dict[str, Any]:
    rules = req.rules
    team_count = int(rules.team_count)
    budget = float(rules.budget)
    max_from_team = int(rules.max_from_one_team)

    role_min = {"wk": rules.roles.wk.min, "bat": rules.roles.bat.min, "ar": rules.roles.ar.min, "bowl": rules.roles.bowl.min}
    role_max = {"wk": rules.roles.wk.max, "bat": rules.roles.bat.max, "ar": rules.roles.ar.max, "bowl": rules.roles.bowl.max}

    must_include = {n.strip().lower() for n in req.preferences.must_include}
    must_exclude = {n.strip().lower() for n in req.preferences.must_exclude}

    # Preprocess and score players.
    players: list[dict[str, Any]] = []
    for p in req.players:
        name_key = p.name.strip().lower()
        if name_key in must_exclude and name_key not in must_include:
            continue
        # Baseline: exclude confirmed OUT unless explicitly forced in.
        if p.injury_status == "out" and name_key not in must_include:
            continue
        players.append(
            {
                "name": p.name,
                "name_key": name_key,
                "team": p.team,
                "role": _role_key(p.role),
                "credits": float(p.credits),
                "points": float(_project_points(p)),
                "is_probable_xi": p.is_probable_xi,
                "injury_status": p.injury_status,
                "news_reason": (p.metadata or {}).get("news_reason") if isinstance(p.metadata, dict) else None,
            }
        )

    if len(players) < team_count:
        raise ValueError("Not enough eligible players to form a team.")

    # Ensure must-include exists.
    names_in_pool = {p["name_key"] for p in players}
    missing = [n for n in must_include if n and n not in names_in_pool]
    if missing:
        raise ValueError(f"Must-include players not found/eligible: {missing[:5]}")

    # Sort by points descending (tie-break by cheaper credits).
    players.sort(key=lambda x: (-float(x["points"]), float(x["credits"]), x["name_key"]))

    cap_mult = float(rules.captain.captain_multiplier)
    vc_mult = float(rules.captain.vice_captain_multiplier)

    def best_cv_bonus(points_list: list[float]) -> float:
        if not points_list:
            return 0.0
        pts = sorted(points_list, reverse=True)
        c = pts[0]
        v = pts[1] if len(pts) > 1 else 0.0
        return (cap_mult - 1.0) * c + (vc_mult - 1.0) * v

    # Upper bound for pruning: current points + best remaining points, plus best possible captain/VC bonus.
    points_sorted = [float(p["points"]) for p in players]

    best_total = -1.0
    best_team: list[dict[str, Any]] = []

    # Greedy seed (gives an initial best_total).
    def greedy_seed() -> None:
        nonlocal best_total, best_team
        pick: list[dict[str, Any]] = []
        credits = 0.0
        role_counts = {"wk": 0, "bat": 0, "ar": 0, "bowl": 0, "unknown": 0}
        team_counts: dict[str, int] = {}
        # force must include first
        for n in must_include:
            for p in players:
                if p["name_key"] == n:
                    pick.append(p)
                    credits += float(p["credits"])
                    role_counts[p["role"]] = role_counts.get(p["role"], 0) + 1
                    team_counts[p["team"]] = team_counts.get(p["team"], 0) + 1
                    break
        for p in players:
            if len(pick) >= team_count:
                break
            if p in pick:
                continue
            if credits + float(p["credits"]) > budget:
                continue
            if team_counts.get(p["team"], 0) + 1 > max_from_team:
                continue
            if p["role"] in role_max and role_counts.get(p["role"], 0) + 1 > role_max[p["role"]]:
                continue
            pick.append(p)
            credits += float(p["credits"])
            role_counts[p["role"]] = role_counts.get(p["role"], 0) + 1
            team_counts[p["team"]] = team_counts.get(p["team"], 0) + 1

        if len(pick) != team_count:
            return
        # Basic role mins check
        for r, m in role_min.items():
            if role_counts.get(r, 0) < m:
                return
        total_points = sum(float(x["points"]) for x in pick) + best_cv_bonus([float(x["points"]) for x in pick])
        if total_points > best_total:
            best_total = total_points
            best_team = list(pick)

    greedy_seed()

    def can_still_meet_role_mins(*, role_counts: dict[str, int], slots_left: int) -> bool:
        need = 0
        for r, mn in role_min.items():
            need += max(0, int(mn) - int(role_counts.get(r, 0)))
        return need <= slots_left

    def role_max_ok(*, role_counts: dict[str, int]) -> bool:
        for r, mx in role_max.items():
            if int(role_counts.get(r, 0)) > int(mx):
                return False
        return True

    def bound_upper(*, idx: int, chosen_points: float, chosen_pts_list: list[float], chosen_n: int) -> float:
        slots_left = team_count - chosen_n
        if slots_left <= 0:
            pts = chosen_pts_list
            return chosen_points + best_cv_bonus(pts)
        # Take the next best remaining points as a loose upper bound (ignores constraints).
        remaining = points_sorted[idx : idx + slots_left]
        ub_points = chosen_points + sum(remaining)
        ub_pts_list = chosen_pts_list + remaining
        return ub_points + best_cv_bonus(ub_pts_list)

    # Branch-and-bound search.
    def dfs(
        idx: int,
        chosen: list[dict[str, Any]],
        credits: float,
        role_counts: dict[str, int],
        team_counts: dict[str, int],
        chosen_points: float,
        chosen_pts_list: list[float],
    ) -> None:
        nonlocal best_total, best_team

        chosen_n = len(chosen)
        slots_left = team_count - chosen_n

        if slots_left == 0:
            for r, mn in role_min.items():
                if role_counts.get(r, 0) < mn:
                    return
            total = chosen_points + best_cv_bonus(chosen_pts_list)
            if total > best_total:
                best_total = total
                best_team = list(chosen)
            return

        if idx >= len(players):
            return

        if not can_still_meet_role_mins(role_counts=role_counts, slots_left=slots_left):
            return

        # Prune by simple optimistic upper bound.
        if bound_upper(idx=idx, chosen_points=chosen_points, chosen_pts_list=chosen_pts_list, chosen_n=chosen_n) <= best_total:
            return

        p = players[idx]

        # Option 1: include p (if feasible)
        name_key = p["name_key"]
        must = name_key in must_include

        if credits + float(p["credits"]) <= budget:
            if team_counts.get(p["team"], 0) + 1 <= max_from_team:
                new_role_counts = dict(role_counts)
                new_team_counts = dict(team_counts)
                new_role_counts[p["role"]] = new_role_counts.get(p["role"], 0) + 1
                new_team_counts[p["team"]] = new_team_counts.get(p["team"], 0) + 1
                if role_max_ok(role_counts=new_role_counts):
                    chosen.append(p)
                    dfs(
                        idx + 1,
                        chosen,
                        credits + float(p["credits"]),
                        new_role_counts,
                        new_team_counts,
                        chosen_points + float(p["points"]),
                        chosen_pts_list + [float(p["points"])],
                    )
                    chosen.pop()

        # Option 2: skip p (only if not required)
        if not must:
            dfs(idx + 1, chosen, credits, role_counts, team_counts, chosen_points, chosen_pts_list)

    # Start with forced must-includes preselected.
    chosen0: list[dict[str, Any]] = []
    credits0 = 0.0
    role_counts0 = {"wk": 0, "bat": 0, "ar": 0, "bowl": 0, "unknown": 0}
    team_counts0: dict[str, int] = {}
    pts0 = 0.0
    pts_list0: list[float] = []
    for n in must_include:
        for p in players:
            if p["name_key"] == n:
                chosen0.append(p)
                credits0 += float(p["credits"])
                role_counts0[p["role"]] = role_counts0.get(p["role"], 0) + 1
                team_counts0[p["team"]] = team_counts0.get(p["team"], 0) + 1
                pts0 += float(p["points"])
                pts_list0.append(float(p["points"]))
                break

    if credits0 > budget:
        raise ValueError("Must-include players exceed budget.")
    if any(v > max_from_team for v in team_counts0.values()):
        raise ValueError("Must-include players violate max_from_one_team constraint.")
    if not role_max_ok(role_counts=role_counts0):
        raise ValueError("Must-include players violate role max constraints.")

    dfs(0, chosen0, credits0, role_counts0, team_counts0, pts0, pts_list0)

    if not best_team:
        raise ValueError("No feasible XI found under the given constraints.")

    # Choose captain and vice-captain from best XI.
    best_team_sorted = sorted(best_team, key=lambda x: (-float(x["points"]), float(x["credits"]), x["name_key"]))
    captain = best_team_sorted[0]["name"]
    vice = best_team_sorted[1]["name"] if len(best_team_sorted) > 1 else best_team_sorted[0]["name"]

    total_credits = sum(float(p["credits"]) for p in best_team)
    credits_remaining = round(budget - total_credits, 2)
    base_points = sum(float(p["points"]) for p in best_team)
    total_points = round(base_points + best_cv_bonus([float(p["points"]) for p in best_team]), 2)

    role_counts_out: dict[str, int] = {"wk": 0, "bat": 0, "ar": 0, "bowl": 0, "unknown": 0}
    team_counts_out: dict[str, int] = {}
    for p in best_team:
        role_counts_out[p["role"]] = role_counts_out.get(p["role"], 0) + 1
        team_counts_out[p["team"]] = team_counts_out.get(p["team"], 0) + 1

    return {
        "platform": rules.platform,
        "format": rules.format,
        "team_count": team_count,
        "budget": budget,
        "selected_team": [
            {
                "name": p["name"],
                "team": p["team"],
                "role": p["role"],
                "credits": p["credits"],
                "expected_points": p["points"],
                "is_probable_xi": p["is_probable_xi"],
                "injury_status": p["injury_status"],
                "news_reason": p.get("news_reason"),
            }
            for p in best_team_sorted
        ],
        "captain": captain,
        "vice_captain": vice,
        "total_credits": round(total_credits, 2),
        "credits_remaining": credits_remaining,
        "role_counts": role_counts_out,
        "team_counts": team_counts_out,
        "projected_points": total_points,
        "warnings": [
            "Optimizer uses expected_points when available; otherwise uses a simple deterministic fallback by role.",
            "Baseline constraints only; does not incorporate live injury/news unless players are marked out/doubtful.",
        ],
    }


def _tools_call(params: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    request_id = str(uuid4())
    meta = ToolMeta(request_id=request_id)
    try:
        call = _ToolsCallArgs.model_validate(params)
    except ValidationError as exc:
        resp = ToolResponse.failure(ErrorCode.INVALID_INPUT, "Invalid tools/call params.", details=exc.errors(), meta=meta)
        return {"content": [{"type": "text", "text": resp.model_dump_json()}]}

    if call.name != TOOL_NAME:
        resp = ToolResponse.failure(ErrorCode.NOT_FOUND, f"Unknown tool: {call.name}", meta=meta)
        return {"content": [{"type": "text", "text": resp.model_dump_json()}]}

    try:
        req = FantasyRequest.model_validate(call.arguments)
    except ValidationError as exc:
        resp = ToolResponse.failure(ErrorCode.INVALID_INPUT, "Invalid FantasyRequest.", details=exc.errors(), meta=meta)
        return {"content": [{"type": "text", "text": resp.model_dump_json()}]}

    try:
        data = _optimize(req)
        meta.elapsed_ms = int((time.perf_counter() - started) * 1000)
        meta.source_ids = [f"fantasy:platform={req.rules.platform}", "fantasy:solver=branch_and_bound"]
        resp = ToolResponse.success(data, meta=meta)
    except Exception as exc:
        meta.elapsed_ms = int((time.perf_counter() - started) * 1000)
        resp = ToolResponse.failure(ErrorCode.UPSTREAM_ERROR, f"Optimization failed: {exc}", meta=meta)

    return {"content": [{"type": "text", "text": resp.model_dump_json()}]}


def _parse_request(raw: str) -> _JsonRpcRequest:
    obj = json.loads(raw)
    return _JsonRpcRequest(
        jsonrpc=obj.get("jsonrpc", "2.0"),
        method=obj["method"],
        params=obj.get("params") or {},
        id=obj.get("id"),
    )


def main() -> None:
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

