from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import duckdb
import pandas as pd

from cricket_companion.config import get_settings


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _norm_key(text: str) -> str:
    """
    Normalizes a raw string into a stable-ish key:
    - lowercase
    - strip punctuation
    - collapse whitespace
    """
    s = (text or "").strip().lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Rename-only mapping (distinct franchises remain distinct).
# Keys are normalized raw names; values are canonical team keys.
_TEAM_RENAME_MAP: dict[str, str] = {
    _norm_key("Delhi Daredevils"): _norm_key("Delhi Capitals"),
    _norm_key("Royal Challengers Bangalore"): _norm_key("Royal Challengers Bengaluru"),
    _norm_key("Kings XI Punjab"): _norm_key("Punjab Kings"),
    # Spelling/alias normalization treated like rename (same franchise).
    _norm_key("Rising Pune Supergiants"): _norm_key("Rising Pune Supergiant"),
}


def team_key(raw: str) -> str:
    k = _norm_key(raw)
    return _TEAM_RENAME_MAP.get(k, k)


def venue_key(raw: str) -> str:
    return _norm_key(raw)


def player_key_from_registry_or_name(raw: str, registry_id: str | None) -> str:
    # Prefer Cricsheet registry people IDs where available (stable).
    if registry_id:
        return registry_id.strip()
    return _norm_key(raw)


def _safe_get(obj: dict[str, Any], path: list[str]) -> Any:
    cur: Any = obj
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def _infer_result_type(info: dict[str, Any]) -> tuple[str, str | None]:
    """
    Returns (result_type, outcome_result_raw).
    """
    outcome = info.get("outcome") or {}
    if not isinstance(outcome, dict):
        return ("unknown", None)

    raw = outcome.get("result")
    if isinstance(raw, str) and raw.strip():
        normalized = _norm_key(raw)
        # Keep a controlled set; map common phrases.
        if normalized in {"no result", "no"}:
            return ("no_result", raw)
        if normalized in {"abandoned"}:
            return ("abandoned", raw)
        if normalized in {"tie", "tied"}:
            return ("tie", raw)
        return (normalized.replace(" ", "_"), raw)

    if outcome.get("winner"):
        return ("normal", None)

    return ("unknown", None)


def _season_from_info_or_date(info: dict[str, Any], start_date: str | None) -> int | None:
    # Cricsheet sometimes embeds season-like fields in event/stage; keep it simple for IPL:
    # if we can't find a numeric season in info, derive from year of match date.
    event = info.get("event") or {}
    if isinstance(event, dict):
        for key in ("season", "year"):
            v = event.get(key)
            if isinstance(v, int):
                return v
            if isinstance(v, str) and v.isdigit():
                return int(v)
        name = event.get("name")
        if isinstance(name, str):
            m = re.search(r"\b(20\d{2})\b", name)
            if m:
                return int(m.group(1))

    if start_date and isinstance(start_date, str):
        m = re.match(r"^(20\d{2})-\d{2}-\d{2}$", start_date.strip())
        if m:
            return int(m.group(1))
    return None


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Missing YAML parser. Install PyYAML with: uv add pyyaml"
        ) from exc

    # Cricsheet delivery keys can look like `2.10` (i.e., over 2, 10th ball due to many extras).
    # PyYAML will otherwise parse `2.10` as the float `2.1`, losing the trailing zero and causing
    # delivery key collisions (e.g., `2.1` vs `2.10` both become `2.1`).
    class _NoFloatLoader(yaml.SafeLoader):  # type: ignore[name-defined]
        pass

    for first_char, resolvers in list(getattr(_NoFloatLoader, "yaml_implicit_resolvers", {}).items()):
        _NoFloatLoader.yaml_implicit_resolvers[first_char] = [  # type: ignore[attr-defined]
            (tag, regexp) for (tag, regexp) in resolvers if tag != "tag:yaml.org,2002:float"
        ]

    with path.open("r", encoding="utf-8") as f:
        obj = yaml.load(f, Loader=_NoFloatLoader)
    if not isinstance(obj, dict):
        raise ValueError("YAML root is not a mapping/object.")
    return obj


@dataclass(frozen=True)
class ParsedMatch:
    match_id: str
    meta: dict[str, Any]
    info: dict[str, Any]
    innings: list[dict[str, Any]]
    registry_people: dict[str, str]


def _parse_match(path: Path) -> ParsedMatch:
    match_id = path.stem
    obj = _load_yaml(path)
    meta = obj.get("meta") or {}
    info = obj.get("info") or {}
    innings = obj.get("innings") or []

    if not isinstance(meta, dict) or not isinstance(info, dict) or not isinstance(innings, list):
        raise ValueError("Invalid YAML structure: expected meta/info dicts and innings list.")

    registry_people = _safe_get(info, ["registry", "people"]) or {}
    if not isinstance(registry_people, dict):
        registry_people = {}

    return ParsedMatch(
        match_id=str(match_id),
        meta=meta,
        info=info,
        innings=innings,
        registry_people={str(k): str(v) for k, v in registry_people.items()},
    )


def _iter_innings(parsed: ParsedMatch) -> Iterable[tuple[int, str, dict[str, Any]]]:
    """
    Yields (innings_index, innings_name_raw, innings_payload).
    """
    for i, entry in enumerate(parsed.innings, start=1):
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError(f"Invalid innings entry at index {i}: expected single-key mapping.")
        innings_name_raw, payload = next(iter(entry.items()))
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid innings payload for {innings_name_raw!r}.")
        yield (i, str(innings_name_raw), payload)


def _is_super_over(innings_index: int, innings_name_raw: str) -> bool:
    # IPL super overs appear as 3rd/4th innings in Cricsheet YAML.
    if innings_index > 2:
        return True
    return "super" in innings_name_raw.lower()


def _delivery_key_parts(ball_key: Any) -> tuple[int, int]:
    """
    Cricsheet delivery keys look like 0.1, 12.6, 19.7, etc.
    Returns (over_number, ball_in_over).
    """
    if isinstance(ball_key, (int, float)):
        s = str(ball_key)
    else:
        s = str(ball_key)
    if "." not in s:
        raise ValueError(f"Unexpected delivery key: {ball_key!r}")
    over_s, ball_s = s.split(".", 1)
    return (int(over_s), int(ball_s))


def _extract_fielders(wicket: dict[str, Any]) -> list[str]:
    fielders = wicket.get("fielders") or wicket.get("fielders_involved") or []
    if isinstance(fielders, list):
        return [str(x) for x in fielders if x is not None]
    if isinstance(fielders, str):
        return [fielders]
    return []


def _extras_breakdown(extras: dict[str, Any] | None) -> dict[str, int]:
    ex = extras or {}
    def _i(name: str) -> int:
        v = ex.get(name)
        if v is None:
            return 0
        try:
            return int(v)
        except Exception:
            return 0

    return {
        "extra_wides": _i("wides"),
        "extra_noballs": _i("noballs"),
        "extra_byes": _i("byes"),
        "extra_legbyes": _i("legbyes"),
        "extra_penalty": _i("penalty"),
    }


def _legal_ball_flags(extra_wides: int, extra_noballs: int) -> tuple[bool, bool, bool]:
    """
    Standard definitions:
    - wides do not count as a legal ball for batter or bowler
    - no-balls do not count as legal balls for batter/bowler counts
      (the "free hit" is still an extra delivery)
    """
    if extra_wides > 0:
        return (False, False, False)
    if extra_noballs > 0:
        return (False, False, False)
    return (True, True, True)


def _ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        create table if not exists ingestion_manifest (
          match_id text not null,
          source_file text not null,
          file_hash text not null,
          ingested_at_utc timestamp not null,
          status text not null,
          error text,
          parser_version text not null,
          primary key (match_id, file_hash)
        )
        """
    )

    con.execute(
        """
        create table if not exists matches (
          match_id text primary key,
          competition text,
          match_type text,
          gender text,
          season integer,
          start_date text,
          balls_per_over integer,
          overs integer,

          city_raw text,
          city_key text,
          venue_raw text,
          venue_key text,

          team1_raw text,
          team1_key text,
          team2_raw text,
          team2_key text,

          toss_winner_raw text,
          toss_winner_key text,
          toss_decision text,

          winner_team_raw text,
          winner_team_key text,
          by_runs integer,
          by_wickets integer,
          method text,
          result_type text,
          outcome_result_raw text,

          player_of_match_json text,

          data_version text,
          meta_created text,
          meta_revision integer
        )
        """
    )

    con.execute(
        """
        create table if not exists innings (
          match_id text not null,
          innings_index integer not null,
          innings_name_raw text,
          innings_type text not null,
          batting_team_raw text,
          batting_team_key text,
          bowling_team_raw text,
          bowling_team_key text,
          primary key (match_id, innings_index)
        )
        """
    )

    con.execute(
        """
        create table if not exists deliveries (
          match_id text not null,
          innings_index integer not null,
          innings_type text not null,

          over_number integer not null,
          ball_in_over integer not null,
          ball_number_in_innings integer not null,

          striker_raw text,
          striker_key text,
          non_striker_raw text,
          non_striker_key text,
          bowler_raw text,
          bowler_key text,

          runs_batter integer,
          runs_extras integer,
          runs_total integer,

          extra_wides integer,
          extra_noballs integer,
          extra_byes integer,
          extra_legbyes integer,
          extra_penalty integer,
          has_extras boolean,

          is_wicket boolean,
          wicket_kind text,
          player_dismissed_raw text,
          player_dismissed_key text,
          fielders_json text,

          is_legal_ball boolean,
          counts_ball_faced boolean,
          counts_ball_bowled boolean,

          primary key (match_id, innings_index, over_number, ball_in_over)
        )
        """
    )

    con.execute(
        """
        create table if not exists players (
          player_key text primary key,
          player_name_raw text not null,
          registry_id text
        )
        """
    )

    con.execute(
        """
        create table if not exists match_players (
          match_id text not null,
          team_raw text,
          team_key text,
          player_key text not null,
          player_name_raw text not null,
          registry_id text,
          primary key (match_id, team_key, player_key)
        )
        """
    )


def _df_insert(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    view = f"__df_{table}"
    con.register(view, df)
    cols = ", ".join([f'"{c}"' for c in df.columns])
    con.execute(f'insert into "{table}" ({cols}) select {cols} from "{view}"')
    con.unregister(view)


def _already_ingested_ok(con: duckdb.DuckDBPyConnection, match_id: str, file_hash: str) -> bool:
    row = con.execute(
        """
        select 1
        from ingestion_manifest
        where match_id = ? and file_hash = ? and status = 'ok'
        limit 1
        """,
        [match_id, file_hash],
    ).fetchone()
    return bool(row)


def _delete_match_rows(con: duckdb.DuckDBPyConnection, match_id: str) -> None:
    con.execute('delete from deliveries where match_id = ?', [match_id])
    con.execute('delete from innings where match_id = ?', [match_id])
    con.execute('delete from match_players where match_id = ?', [match_id])
    con.execute('delete from matches where match_id = ?', [match_id])


def _upsert_players(con: duckdb.DuckDBPyConnection, players_rows: list[dict[str, Any]]) -> None:
    if not players_rows:
        return
    df = pd.DataFrame(players_rows).drop_duplicates(subset=["player_key"])
    view = "__players_upsert"
    con.register(view, df)
    # Prefer registry_id-backed keys; for duplicates, keep existing if present.
    con.execute(
        f"""
        insert into players (player_key, player_name_raw, registry_id)
        select player_key, player_name_raw, registry_id from {view}
        on conflict (player_key) do update set
          player_name_raw = excluded.player_name_raw,
          registry_id = coalesce(players.registry_id, excluded.registry_id)
        """
    )
    con.unregister(view)


def ingest_one(con: duckdb.DuckDBPyConnection, yaml_path: Path, *, parser_version: str) -> tuple[bool, str | None]:
    match_id = yaml_path.stem
    file_hash = _sha256(yaml_path)
    rel_path = str(yaml_path.as_posix())

    if _already_ingested_ok(con, match_id, file_hash):
        return (True, None)

    try:
        parsed = _parse_match(yaml_path)

        info = parsed.info
        meta = parsed.meta

        competition = info.get("competition")
        match_type = info.get("match_type")
        gender = info.get("gender")
        balls_per_over = info.get("balls_per_over")
        overs = info.get("overs")

        dates = info.get("dates") or []
        start_date = dates[0] if isinstance(dates, list) and dates else None
        if start_date is not None:
            start_date = str(start_date)

        season = _season_from_info_or_date(info, start_date)

        city_raw = info.get("city")
        venue_raw = info.get("venue")
        city_raw_s = str(city_raw) if city_raw is not None else None
        venue_raw_s = str(venue_raw) if venue_raw is not None else None

        teams = info.get("teams") or []
        team1_raw = str(teams[0]) if isinstance(teams, list) and len(teams) >= 1 else None
        team2_raw = str(teams[1]) if isinstance(teams, list) and len(teams) >= 2 else None

        toss = info.get("toss") or {}
        toss_winner_raw = toss.get("winner") if isinstance(toss, dict) else None
        toss_decision = toss.get("decision") if isinstance(toss, dict) else None

        outcome = info.get("outcome") or {}
        winner_team_raw = outcome.get("winner") if isinstance(outcome, dict) else None
        by = outcome.get("by") if isinstance(outcome, dict) else None
        by_runs = None
        by_wickets = None
        if isinstance(by, dict):
            if by.get("runs") is not None:
                by_runs = int(by.get("runs"))
            if by.get("wickets") is not None:
                by_wickets = int(by.get("wickets"))
        method = outcome.get("method") if isinstance(outcome, dict) else None
        result_type, outcome_result_raw = _infer_result_type(info)

        pom = info.get("player_of_match") or []
        player_of_match_json = json.dumps(pom, ensure_ascii=False) if pom else "[]"

        match_row = {
            "match_id": match_id,
            "competition": str(competition) if competition is not None else "IPL",
            "match_type": str(match_type) if match_type is not None else "T20",
            "gender": str(gender) if gender is not None else "male",
            "season": season,
            "start_date": start_date,
            "balls_per_over": int(balls_per_over) if balls_per_over is not None else 6,
            "overs": int(overs) if overs is not None else 20,
            "city_raw": city_raw_s,
            "city_key": venue_key(city_raw_s or "") if city_raw_s else None,
            "venue_raw": venue_raw_s,
            "venue_key": venue_key(venue_raw_s or "") if venue_raw_s else None,
            "team1_raw": team1_raw,
            "team1_key": team_key(team1_raw or "") if team1_raw else None,
            "team2_raw": team2_raw,
            "team2_key": team_key(team2_raw or "") if team2_raw else None,
            "toss_winner_raw": str(toss_winner_raw) if toss_winner_raw is not None else None,
            "toss_winner_key": team_key(str(toss_winner_raw)) if toss_winner_raw is not None else None,
            "toss_decision": str(toss_decision) if toss_decision is not None else None,
            "winner_team_raw": str(winner_team_raw) if winner_team_raw is not None else None,
            "winner_team_key": team_key(str(winner_team_raw)) if winner_team_raw is not None else None,
            "by_runs": by_runs,
            "by_wickets": by_wickets,
            "method": str(method) if method is not None else None,
            "result_type": result_type,
            "outcome_result_raw": outcome_result_raw,
            "player_of_match_json": player_of_match_json,
            "data_version": str(meta.get("data_version")) if meta.get("data_version") is not None else None,
            "meta_created": str(meta.get("created")) if meta.get("created") is not None else None,
            "meta_revision": int(meta.get("revision")) if meta.get("revision") is not None else None,
        }

        innings_rows: list[dict[str, Any]] = []
        deliveries_rows: list[dict[str, Any]] = []

        # Player registry for this match; prefer registry_id.
        players_rows: list[dict[str, Any]] = []
        match_players_rows: list[dict[str, Any]] = []

        info_players = info.get("players") or {}
        if isinstance(info_players, dict):
            for team_raw, plist in info_players.items():
                if not isinstance(plist, list):
                    continue
                for p_raw_any in plist:
                    p_raw = str(p_raw_any)
                    reg_id = parsed.registry_people.get(p_raw)
                    p_key = player_key_from_registry_or_name(p_raw, reg_id)
                    players_rows.append({"player_key": p_key, "player_name_raw": p_raw, "registry_id": reg_id})
                    match_players_rows.append(
                        {
                            "match_id": match_id,
                            "team_raw": str(team_raw),
                            "team_key": team_key(str(team_raw)),
                            "player_key": p_key,
                            "player_name_raw": p_raw,
                            "registry_id": reg_id,
                        }
                    )

        # Normalize innings + deliveries.
        for innings_index, innings_name_raw, payload in _iter_innings(parsed):
            batting_team_raw = str(payload.get("team")) if payload.get("team") is not None else None
            batting_team_key = team_key(batting_team_raw) if batting_team_raw else None

            innings_type = "super_over" if _is_super_over(innings_index, innings_name_raw) else "normal"

            # Best-effort bowling team inference: the other team from match teams.
            bowling_team_raw = None
            if batting_team_raw and team1_raw and team2_raw:
                if _norm_key(batting_team_raw) == _norm_key(team1_raw):
                    bowling_team_raw = team2_raw
                elif _norm_key(batting_team_raw) == _norm_key(team2_raw):
                    bowling_team_raw = team1_raw
            bowling_team_key = team_key(bowling_team_raw) if bowling_team_raw else None

            innings_rows.append(
                {
                    "match_id": match_id,
                    "innings_index": innings_index,
                    "innings_name_raw": innings_name_raw,
                    "innings_type": innings_type,
                    "batting_team_raw": batting_team_raw,
                    "batting_team_key": batting_team_key,
                    "bowling_team_raw": bowling_team_raw,
                    "bowling_team_key": bowling_team_key,
                }
            )

            deliveries = payload.get("deliveries") or []
            if not isinstance(deliveries, list):
                continue

            ball_seq = 0
            for d in deliveries:
                if not isinstance(d, dict) or len(d) != 1:
                    continue
                ball_key, detail_any = next(iter(d.items()))
                if not isinstance(detail_any, dict):
                    continue
                detail = detail_any

                over_number, ball_in_over = _delivery_key_parts(ball_key)
                ball_seq += 1

                striker_raw = str(detail.get("batsman") or detail.get("batter") or "")
                non_striker_raw = str(detail.get("non_striker") or "")
                bowler_raw = str(detail.get("bowler") or "")

                striker_reg = parsed.registry_people.get(striker_raw)
                non_striker_reg = parsed.registry_people.get(non_striker_raw)
                bowler_reg = parsed.registry_people.get(bowler_raw)

                striker_key = player_key_from_registry_or_name(striker_raw, striker_reg)
                non_striker_key = player_key_from_registry_or_name(non_striker_raw, non_striker_reg)
                bowler_key = player_key_from_registry_or_name(bowler_raw, bowler_reg)

                players_rows.extend(
                    [
                        {"player_key": striker_key, "player_name_raw": striker_raw, "registry_id": striker_reg},
                        {"player_key": non_striker_key, "player_name_raw": non_striker_raw, "registry_id": non_striker_reg},
                        {"player_key": bowler_key, "player_name_raw": bowler_raw, "registry_id": bowler_reg},
                    ]
                )

                runs = detail.get("runs") or {}
                runs_batter = int(runs.get("batsman") or runs.get("batter") or 0) if isinstance(runs, dict) else 0
                runs_extras = int(runs.get("extras") or 0) if isinstance(runs, dict) else 0
                runs_total = int(runs.get("total") or 0) if isinstance(runs, dict) else 0

                extras = detail.get("extras")
                extras_bd = _extras_breakdown(extras if isinstance(extras, dict) else None)
                has_extras = any(v > 0 for v in extras_bd.values())

                is_legal_ball, counts_ball_faced, counts_ball_bowled = _legal_ball_flags(
                    extras_bd["extra_wides"], extras_bd["extra_noballs"]
                )

                wicket = detail.get("wicket") if isinstance(detail.get("wicket"), dict) else None
                is_wicket = wicket is not None
                wicket_kind = str(wicket.get("kind")) if wicket and wicket.get("kind") is not None else None
                player_dismissed_raw = str(wicket.get("player_out")) if wicket and wicket.get("player_out") is not None else None
                dismissed_reg = parsed.registry_people.get(player_dismissed_raw) if player_dismissed_raw else None
                player_dismissed_key = (
                    player_key_from_registry_or_name(player_dismissed_raw, dismissed_reg) if player_dismissed_raw else None
                )
                if player_dismissed_raw:
                    players_rows.append(
                        {"player_key": player_dismissed_key, "player_name_raw": player_dismissed_raw, "registry_id": dismissed_reg}
                    )
                fielders_json = json.dumps(_extract_fielders(wicket or {}), ensure_ascii=False) if wicket else "[]"

                deliveries_rows.append(
                    {
                        "match_id": match_id,
                        "innings_index": innings_index,
                        "innings_type": innings_type,
                        "over_number": over_number,
                        "ball_in_over": ball_in_over,
                        "ball_number_in_innings": ball_seq,
                        "striker_raw": striker_raw,
                        "striker_key": striker_key,
                        "non_striker_raw": non_striker_raw,
                        "non_striker_key": non_striker_key,
                        "bowler_raw": bowler_raw,
                        "bowler_key": bowler_key,
                        "runs_batter": runs_batter,
                        "runs_extras": runs_extras,
                        "runs_total": runs_total,
                        **extras_bd,
                        "has_extras": bool(has_extras),
                        "is_wicket": bool(is_wicket),
                        "wicket_kind": wicket_kind,
                        "player_dismissed_raw": player_dismissed_raw,
                        "player_dismissed_key": player_dismissed_key,
                        "fielders_json": fielders_json,
                        "is_legal_ball": bool(is_legal_ball),
                        "counts_ball_faced": bool(counts_ball_faced),
                        "counts_ball_bowled": bool(counts_ball_bowled),
                    }
                )

        # Replace-by-match_id semantics.
        con.execute("begin transaction")
        _delete_match_rows(con, match_id)

        _df_insert(con, "matches", pd.DataFrame([match_row]))
        _df_insert(con, "innings", pd.DataFrame(innings_rows))
        _df_insert(con, "deliveries", pd.DataFrame(deliveries_rows))

        _upsert_players(con, players_rows)
        _df_insert(con, "match_players", pd.DataFrame(match_players_rows))

        con.execute(
            """
            insert into ingestion_manifest
              (match_id, source_file, file_hash, ingested_at_utc, status, error, parser_version)
            values (?, ?, ?, ?, 'ok', null, ?)
            on conflict (match_id, file_hash) do update set
              source_file = excluded.source_file,
              ingested_at_utc = excluded.ingested_at_utc,
              status = excluded.status,
              error = null,
              parser_version = excluded.parser_version
            """,
            [match_id, rel_path, file_hash, _utc_now(), parser_version],
        )
        con.execute("commit")

        return (True, None)
    except Exception as exc:
        try:
            con.execute("rollback")
        except Exception:
            pass
        con.execute(
            """
            insert into ingestion_manifest
              (match_id, source_file, file_hash, ingested_at_utc, status, error, parser_version)
            values (?, ?, ?, ?, 'failed', ?, ?)
            on conflict (match_id, file_hash) do update set
              source_file = excluded.source_file,
              ingested_at_utc = excluded.ingested_at_utc,
              status = excluded.status,
              error = excluded.error,
              parser_version = excluded.parser_version
            """,
            [match_id, rel_path, file_hash, _utc_now(), str(exc), parser_version],
        )
        return (False, str(exc))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest Cricsheet IPL-men YAML into DuckDB (normalized tables).")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory containing Cricsheet IPL YAML files. Default: data/cricsheet/ipl_men/yaml",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/duckdb/cricket_ipl_men.duckdb",
        help="DuckDB output path.",
    )
    parser.add_argument(
        "--max-matches",
        type=int,
        default=None,
        help="Optional limit for quick runs (for debugging).",
    )
    args = parser.parse_args(argv)

    settings = get_settings(load_env_file=True)
    default_input = settings.data_dir / "cricsheet" / "ipl_men" / "yaml"
    input_dir = Path(args.input_dir) if args.input_dir else default_input
    db_path = Path(args.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"ERROR: input dir not found: {input_dir}", file=sys.stderr)
        return 2

    yaml_paths = sorted(input_dir.glob("*.yaml"))
    if args.max_matches is not None:
        yaml_paths = yaml_paths[: max(0, int(args.max_matches))]

    print(f"Input: {input_dir}")
    print(f"DB: {db_path}")
    print(f"YAML files: {len(yaml_paths)}")

    parser_version = "ipl_yaml_to_duckdb_v1"

    con = duckdb.connect(str(db_path))
    try:
        _ensure_schema(con)
        ok = 0
        failed = 0
        for p in yaml_paths:
            success, err = ingest_one(con, p, parser_version=parser_version)
            if success:
                ok += 1
            else:
                failed += 1
                print(f"[FAILED] {p.name}: {err}", file=sys.stderr)
        print(f"Done. ok={ok} failed={failed}")
        return 0 if failed == 0 else 1
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
