from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError

from cricket_companion.schemas import ErrorCode, ToolMeta, ToolResponse
from cricket_companion.sim_schemas import SimulationRequest


TOOL_NAME = "sim_run"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    idx = (len(sorted_vals) - 1) * _clamp(p, 0.0, 1.0)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def _phase_from_balls(*, balls_elapsed: int, max_balls: int, explicit: str) -> str:
    if explicit and explicit != "unknown":
        return explicit
    if max_balls <= 0:
        return "unknown"
    # T20-style phases; for ODI/Test we still produce a coarse phase bucket.
    over = balls_elapsed // 6 + 1
    if over <= 6:
        return "powerplay"
    if over <= max(6, (max_balls // 6) - 5):
        return "middle"
    return "death"


Outcome = Literal["W", "0", "1", "2", "3", "4", "6"]


def _base_probs(phase: str) -> dict[Outcome, float]:
    # Simple, explainable priors. These are not calibrated; they're a baseline for 3.2.2.
    if phase == "powerplay":
        return {"W": 0.045, "0": 0.29, "1": 0.32, "2": 0.08, "3": 0.01, "4": 0.18, "6": 0.075}
    if phase == "death":
        return {"W": 0.07, "0": 0.25, "1": 0.27, "2": 0.09, "3": 0.01, "4": 0.23, "6": 0.15}
    # middle / unknown
    return {"W": 0.04, "0": 0.33, "1": 0.39, "2": 0.11, "3": 0.01, "4": 0.12, "6": 0.03}


def _normalize_probs(p: dict[Outcome, float]) -> dict[Outcome, float]:
    s = sum(max(0.0, v) for v in p.values())
    if s <= 0:
        return {"W": 0.05, "0": 0.3, "1": 0.4, "2": 0.1, "3": 0.01, "4": 0.1, "6": 0.04}
    return {k: max(0.0, v) / s for k, v in p.items()}


def _adjust_for_pressure(p: dict[Outcome, float], *, pressure_ratio: float) -> dict[Outcome, float]:
    # pressure_ratio > 1 => need to score faster than baseline
    q = dict(p)
    pr = _clamp(pressure_ratio, 0.5, 2.0)
    if pr > 1.05:
        boost = _clamp(pr, 1.0, 1.8)
        q["4"] *= boost
        q["6"] *= boost
        q["0"] *= 1.0 / _clamp(boost, 1.0, 1.6)
        q["1"] *= 1.0 / _clamp(boost, 1.0, 1.4)
        q["W"] *= 1.0 + _clamp((pr - 1.0) * 0.35, 0.0, 0.35)
    elif pr < 0.95:
        ease = _clamp(1.0 / pr, 1.0, 1.7)
        q["4"] *= 1.0 / _clamp(ease, 1.0, 1.5)
        q["6"] *= 1.0 / _clamp(ease, 1.0, 1.7)
        q["0"] *= _clamp(ease, 1.0, 1.4)
        q["1"] *= _clamp(ease, 1.0, 1.3)
        q["W"] *= 1.0 / _clamp(ease, 1.0, 1.5)
    return q


def _adjust_for_wickets(p: dict[Outcome, float], *, wkts_left: int) -> dict[Outcome, float]:
    # Fewer wickets left => be more conservative and reduce wicket risk.
    q = dict(p)
    wl = int(_clamp(float(wkts_left), 0.0, 10.0))
    if wl <= 2:
        q["W"] *= 0.75
        q["6"] *= 0.75
        q["4"] *= 0.85
        q["1"] *= 1.10
        q["0"] *= 1.05
    elif wl <= 4:
        q["W"] *= 0.90
        q["6"] *= 0.90
        q["4"] *= 0.95
        q["1"] *= 1.05
    return q


def _adjust_for_strength(p: dict[Outcome, float], *, bat: float | None, bowl: float | None) -> dict[Outcome, float]:
    if bat is None and bowl is None:
        return p
    bat_r = 0.5 if bat is None else float(bat)
    bowl_r = 0.5 if bowl is None else float(bowl)
    delta = _clamp(bat_r - bowl_r, -0.5, 0.5)  # -0.5..0.5
    q = dict(p)
    if delta > 0.01:
        q["4"] *= 1.0 + delta * 0.7
        q["6"] *= 1.0 + delta * 0.9
        q["W"] *= 1.0 - delta * 0.6
        q["0"] *= 1.0 - delta * 0.3
    elif delta < -0.01:
        d = abs(delta)
        q["4"] *= 1.0 - d * 0.6
        q["6"] *= 1.0 - d * 0.8
        q["W"] *= 1.0 + d * 0.8
        q["0"] *= 1.0 + d * 0.4
    return q


def _sample_outcome(rng: Any, probs: dict[Outcome, float]) -> Outcome:
    x = rng.random()
    acc = 0.0
    for k, v in probs.items():
        acc += v
        if x <= acc:
            return k
    return "1"


def _default_db_path() -> str:
    return os.environ.get("CC_CRICKET_DB_PATH", os.path.join("data", "duckdb", "cricket_ipl_men.duckdb"))


def _balls_left_bucket(balls_left: int) -> int:
    if balls_left <= 0:
        return 0
    return int(balls_left // 6) * 6


def _rr_bucket(runs_left: int, balls_left: int) -> float:
    if balls_left <= 0:
        return 99.0
    rr = float(runs_left) / float(balls_left)
    return float(int(rr * 4.0) / 4.0)


def _historical_lookup_model(req: SimulationRequest) -> dict[str, Any] | None:
    ms = req.match_state
    if req.mode != "chase" or ms.chase is None:
        return None

    max_balls = ms.limits.max_balls
    balls_left0 = max_balls - ms.score.balls
    wkts_left0 = 10 - ms.score.wkts
    runs_left0 = ms.chase.target_runs - ms.score.runs
    if balls_left0 <= 0 or runs_left0 <= 0:
        return None

    phase0 = _phase_from_balls(balls_elapsed=ms.score.balls, max_balls=max_balls, explicit=str(ms.phase))
    balls_bucket0 = _balls_left_bucket(balls_left0)
    rr0 = _rr_bucket(runs_left0, balls_left0)

    window_overs = int(req.simulation.historical_window_overs)
    bucket_lo = max(0, balls_bucket0 - window_overs * 6)
    bucket_hi = min(max_balls, balls_bucket0 + window_overs * 6)

    try:
        import duckdb  # type: ignore
    except Exception:
        return None

    db = _default_db_path()
    try:
        con = duckdb.connect(db, read_only=True)
    except Exception:
        return None

    try:
        sql = """
        WITH
        inn1 AS (
          SELECT match_id, CAST(runs_total AS INTEGER) AS inn1_runs
          FROM innings_summary
          WHERE innings_index = 1
        ),
        legal_inn2 AS (
          SELECT
            d.match_id,
            d.phase,
            d.ball_number_in_innings,
            d.batting_team_key,
            m.winner_team_key,
            m.overs,
            (inn1.inn1_runs + 1) AS target_runs,
            CAST(d.runs_total AS INTEGER) AS runs_this_ball,
            CASE WHEN d.is_wicket THEN 1 ELSE 0 END AS is_wicket_this_ball,
            CASE WHEN COALESCE(d.extra_wides,0) > 0 OR COALESCE(d.extra_noballs,0) > 0 THEN 0 ELSE 1 END AS is_legal
          FROM deliveries_enriched d
          JOIN matches m ON m.match_id = d.match_id
          JOIN inn1 ON inn1.match_id = d.match_id
          WHERE d.innings_index = 2
        ),
        legal_balls AS (
          SELECT * FROM legal_inn2 WHERE is_legal = 1
        ),
        states AS (
          SELECT
            match_id,
            phase,
            overs,
            target_runs,
            batting_team_key,
            winner_team_key,
            (winner_team_key = batting_team_key) AS chase_won,
            (row_number() OVER (PARTITION BY match_id ORDER BY ball_number_in_innings) - 1) AS balls_before,
            COALESCE(sum(runs_this_ball) OVER (PARTITION BY match_id ORDER BY ball_number_in_innings ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS runs_before,
            COALESCE(sum(is_wicket_this_ball) OVER (PARTITION BY match_id ORDER BY ball_number_in_innings ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS wkts_before,
            runs_this_ball,
            is_wicket_this_ball
          FROM legal_balls
        ),
        feats AS (
          SELECT
            phase,
            CAST(overs AS INTEGER) AS max_overs,
            CAST((CAST(overs AS INTEGER) * 6 - balls_before) AS INTEGER) AS balls_left,
            CAST((target_runs - runs_before) AS INTEGER) AS runs_left,
            CAST((10 - wkts_before) AS INTEGER) AS wkts_left,
            chase_won,
            runs_this_ball,
            is_wicket_this_ball
          FROM states
          WHERE (CAST(overs AS INTEGER) * 6 - balls_before) > 0
            AND (target_runs - runs_before) > 0
            AND wkts_before < 10
        ),
        buckets AS (
          SELECT
            phase,
            CAST(floor(balls_left / 6) * 6 AS INTEGER) AS balls_left_bucket,
            wkts_left,
            CAST(floor((runs_left::DOUBLE / balls_left::DOUBLE) * 4.0) / 4.0 AS DOUBLE) AS rr_bucket,
            chase_won,
            runs_this_ball,
            is_wicket_this_ball
          FROM feats
        )
        SELECT
          phase,
          balls_left_bucket,
          wkts_left,
          rr_bucket,
          count(*) AS n,
          avg(CASE WHEN chase_won THEN 1.0 ELSE 0.0 END) AS win_rate,
          sum(CASE WHEN is_wicket_this_ball = 1 THEN 1 ELSE 0 END) AS c_w,
          sum(CASE WHEN is_wicket_this_ball = 0 AND runs_this_ball = 0 THEN 1 ELSE 0 END) AS c_0,
          sum(CASE WHEN is_wicket_this_ball = 0 AND runs_this_ball = 1 THEN 1 ELSE 0 END) AS c_1,
          sum(CASE WHEN is_wicket_this_ball = 0 AND runs_this_ball = 2 THEN 1 ELSE 0 END) AS c_2,
          sum(CASE WHEN is_wicket_this_ball = 0 AND runs_this_ball = 3 THEN 1 ELSE 0 END) AS c_3,
          sum(CASE WHEN is_wicket_this_ball = 0 AND runs_this_ball >= 4 AND runs_this_ball < 6 THEN 1 ELSE 0 END) AS c_4,
          sum(CASE WHEN is_wicket_this_ball = 0 AND runs_this_ball >= 6 THEN 1 ELSE 0 END) AS c_6
        FROM buckets
        WHERE balls_left_bucket BETWEEN ? AND ?
        GROUP BY phase, balls_left_bucket, wkts_left, rr_bucket
        """
        rows = con.execute(sql, [bucket_lo, bucket_hi]).fetchall()
    except Exception:
        return None
    finally:
        try:
            con.close()
        except Exception:
            pass

    if not rows:
        return None

    buckets_map: dict[str, Any] = {}
    total_n = 0
    win_num = 0.0
    for r in rows:
        (
            phase,
            balls_left_bucket,
            wkts_left,
            rr_bucket,
            n,
            win_rate,
            c_w,
            c_0,
            c_1,
            c_2,
            c_3,
            c_4,
            c_6,
        ) = r
        n_i = int(n or 0)
        if n_i <= 0:
            continue
        probs = _normalize_probs(
            {
                "W": float(c_w or 0),
                "0": float(c_0 or 0),
                "1": float(c_1 or 0),
                "2": float(c_2 or 0),
                "3": float(c_3 or 0),
                "4": float(c_4 or 0),
                "6": float(c_6 or 0),
            }
        )
        key = f"{phase}|{int(balls_left_bucket)}|{int(wkts_left)}|{float(rr_bucket):.2f}"
        buckets_map[key] = {"n": n_i, "probs": probs, "win_rate": float(win_rate or 0.0)}
        total_n += n_i
        win_num += float(win_rate or 0.0) * float(n_i)

    if not buckets_map:
        return None

    overall_win = (win_num / float(total_n)) if total_n > 0 else 0.0
    return {
        "seed_key": f"{phase0}|{balls_bucket0}|{wkts_left0}|{rr0:.2f}",
        "buckets": buckets_map,
        "overall": {"n": total_n, "win_rate": round(overall_win, 4)},
        "window": {"balls_left_bucket_lo": bucket_lo, "balls_left_bucket_hi": bucket_hi},
    }


def _pick_empirical(
    hist: dict[str, Any],
    *,
    phase: str,
    balls_left: int,
    wkts_left: int,
    runs_left: int,
) -> tuple[int, dict[Outcome, float]] | None:
    buckets = hist.get("buckets")
    if not isinstance(buckets, dict):
        return None
    k = f"{phase}|{_balls_left_bucket(balls_left)}|{int(wkts_left)}|{_rr_bucket(runs_left, balls_left):.2f}"
    hit = buckets.get(k)
    if not isinstance(hit, dict):
        return None
    n = int(hit.get("n") or 0)
    probs = hit.get("probs")
    if n <= 0 or not isinstance(probs, dict):
        return None
    try:
        parsed = {kk: float(vv) for kk, vv in probs.items()}
    except Exception:
        return None
    return n, _normalize_probs(parsed)  # type: ignore[arg-type]


def _simulate(req: SimulationRequest) -> dict[str, Any]:
    import random

    ms = req.match_state
    max_balls = ms.limits.max_balls
    balls_left0 = max_balls - ms.score.balls
    wkts_left0 = 10 - ms.score.wkts

    if balls_left0 < 0:
        balls_left0 = 0
    if wkts_left0 < 0:
        wkts_left0 = 0

    target_runs = ms.chase.target_runs if (req.mode == "chase" and ms.chase is not None) else None

    rng = random.Random(req.simulation.seed)

    historical: dict[str, Any] | None = None
    if req.simulation.model == "historical_blend":
        historical = _historical_lookup_model(req)

    wins = 0
    final_scores: list[int] = []
    balls_used_to_finish: list[int] = []

    bat_rating = None
    bowl_rating = None
    if req.strength and req.strength.team:
        bat_rating = req.strength.team.batting_rating
        bowl_rating = req.strength.team.bowling_rating

    for _ in range(req.simulation.n_sims):
        runs = int(ms.score.runs)
        balls = int(ms.score.balls)
        wkts = int(ms.score.wkts)
        balls_left = balls_left0
        wkts_left = wkts_left0

        # For chase mode, keep a per-sim goal based on current runs/target.
        runs_left = (int(target_runs) - runs) if target_runs is not None else None
        finished_balls_used: int | None = None

        while balls_left > 0 and wkts_left > 0:
            phase = _phase_from_balls(balls_elapsed=balls, max_balls=max_balls, explicit=str(ms.phase))
            base = _base_probs(phase)

            # Pressure only makes sense in chase mode.
            pressure_ratio = 1.0
            if req.mode == "chase" and runs_left is not None and balls_left > 0:
                required_per_ball = max(0.0, float(runs_left) / float(balls_left))
                baseline_rr = 8.0 if phase == "middle" else 8.5 if phase == "powerplay" else 10.5
                baseline_per_ball = baseline_rr / 6.0
                pressure_ratio = required_per_ball / max(0.01, baseline_per_ball)

            baseline_probs = _normalize_probs(
                _adjust_for_strength(
                    _adjust_for_wickets(_adjust_for_pressure(base, pressure_ratio=pressure_ratio), wkts_left=wkts_left),
                    bat=bat_rating,
                    bowl=bowl_rating,
                )
            )
            probs = baseline_probs
            if historical is not None and req.mode == "chase" and runs_left is not None:
                picked = _pick_empirical(historical, phase=phase, balls_left=balls_left, wkts_left=wkts_left, runs_left=runs_left)
                if picked is not None:
                    n_emp, emp_probs = picked
                    min_n = int(req.simulation.historical_min_samples)
                    if n_emp >= min_n:
                        alpha = min(float(req.simulation.historical_blend_max_alpha), float(n_emp) / float(n_emp + min_n))
                        blended: dict[Outcome, float] = {}
                        for k2 in baseline_probs.keys():
                            blended[k2] = alpha * float(emp_probs.get(k2, 0.0)) + (1.0 - alpha) * float(baseline_probs.get(k2, 0.0))
                        probs = _normalize_probs(blended)

            o = _sample_outcome(rng, probs)
            balls += 1
            balls_left -= 1
            if o == "W":
                wkts += 1
                wkts_left -= 1
            else:
                runs += int(o)

            if req.mode == "chase" and target_runs is not None:
                runs_left = int(target_runs) - runs
                if runs_left <= 0:
                    wins += 1
                    finished_balls_used = max_balls - balls_left
                    break

        final_scores.append(runs)
        if finished_balls_used is not None:
            balls_used_to_finish.append(finished_balls_used)

    final_scores_sorted = sorted(final_scores)
    p10 = int(round(_percentile([float(x) for x in final_scores_sorted], 0.10)))
    p50 = int(round(_percentile([float(x) for x in final_scores_sorted], 0.50)))
    p90 = int(round(_percentile([float(x) for x in final_scores_sorted], 0.90)))
    avg = float(sum(final_scores)) / float(len(final_scores) or 1)

    out: dict[str, Any] = {
        "mode": req.mode,
        "format": req.format,
        "n_sims": req.simulation.n_sims,
        "seed": req.simulation.seed,
        "model": req.simulation.model,
        "inputs": {
            "innings": ms.innings,
            "runs": ms.score.runs,
            "wkts": ms.score.wkts,
            "balls": ms.score.balls,
            "max_overs": ms.limits.effective_overs,
            "target_runs": target_runs,
        },
        "score_percentiles": {"p10": p10, "p50": p50, "p90": p90},
        "expected_final_runs": round(avg, 2),
    }

    if req.mode == "chase" and req.simulation.n_sims > 0:
        out["win_probability"] = round(float(wins) / float(req.simulation.n_sims), 4)
        if balls_used_to_finish:
            balls_used_sorted = sorted(balls_used_to_finish)
            out["balls_to_finish_percentiles"] = {
                "p10": int(round(_percentile([float(x) for x in balls_used_sorted], 0.10))),
                "p50": int(round(_percentile([float(x) for x in balls_used_sorted], 0.50))),
                "p90": int(round(_percentile([float(x) for x in balls_used_sorted], 0.90))),
            }

    if historical is not None and isinstance(historical.get("overall"), dict):
        out["historical"] = {
            "overall": historical.get("overall"),
            "window": historical.get("window"),
            "seed_key": historical.get("seed_key"),
            "bucket_count": len(historical.get("buckets") or {}),
        }

    out["warnings"] = [
        "Baseline simulator: probabilities are heuristic (not calibrated to ball-by-ball historical data).",
        "Ignores extras and matchups; treats each ball as independent given coarse phase/pressure.",
    ]

    return out


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
    schema = SimulationRequest.model_json_schema()
    return {
        "tools": [
            {
                "name": TOOL_NAME,
                "description": "Monte Carlo cricket scenario simulator (baseline, heuristic probabilities).",
                "inputSchema": schema,
            }
        ]
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
        req = SimulationRequest.model_validate(call.arguments)
    except ValidationError as exc:
        resp = ToolResponse.failure(ErrorCode.INVALID_INPUT, "Invalid SimulationRequest.", details=exc.errors(), meta=meta)
        return {"content": [{"type": "text", "text": resp.model_dump_json()}]}

    try:
        data = _simulate(req)
        meta.elapsed_ms = int((time.perf_counter() - started) * 1000)
        meta.source_ids = [f"sim:model={req.simulation.model}", f"sim:n_sims={req.simulation.n_sims}"]
        resp = ToolResponse.success(data, meta=meta)
    except Exception as exc:
        meta.elapsed_ms = int((time.perf_counter() - started) * 1000)
        resp = ToolResponse.failure(ErrorCode.INTERNAL, f"Simulation failed: {exc}", meta=meta)

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
