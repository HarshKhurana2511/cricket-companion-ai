from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    details: str
    severity: str  # "HARD" | "WARN"


def _fetch_one_int(con: duckdb.DuckDBPyConnection, sql: str, params: list | None = None) -> int:
    row = con.execute(sql, params or []).fetchone()
    if not row:
        return 0
    return int(row[0] or 0)


def _table_exists(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    row = con.execute(
        """
        select 1
        from information_schema.tables
        where table_schema = 'main' and table_name = ?
        limit 1
        """,
        [name],
    ).fetchone()
    return bool(row)


def _view_exists(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    row = con.execute(
        """
        select 1
        from information_schema.views
        where table_schema = 'main' and table_name = ?
        limit 1
        """,
        [name],
    ).fetchone()
    return bool(row)


def _hard(name: str, ok: bool, details: str) -> CheckResult:
    return CheckResult(name=name, ok=ok, details=details, severity="HARD")


def _warn(name: str, ok: bool, details: str) -> CheckResult:
    return CheckResult(name=name, ok=ok, details=details, severity="WARN")


def _print_results(results: list[CheckResult]) -> None:
    hard_fail = [r for r in results if r.severity == "HARD" and not r.ok]
    warn_fail = [r for r in results if r.severity == "WARN" and not r.ok]

    print("\nVALIDATION RESULTS")
    print("-" * 80)
    for r in results:
        status = "OK" if r.ok else ("FAIL" if r.severity == "HARD" else "WARN")
        print(f"[{status}] ({r.severity}) {r.name}: {r.details}")

    print("-" * 80)
    print(f"Hard failures: {len(hard_fail)}")
    print(f"Warnings: {len(warn_fail)}")


def run_validations(con: duckdb.DuckDBPyConnection, *, yaml_dir: Path | None) -> list[CheckResult]:
    results: list[CheckResult] = []

    # A) Schema presence (hard)
    required_tables = ["ingestion_manifest", "matches", "innings", "deliveries", "players", "match_players"]
    for t in required_tables:
        results.append(_hard(f"table:{t}", _table_exists(con, t), "present" if _table_exists(con, t) else "missing"))

    derived_views = ["deliveries_enriched", "innings_summary", "innings_phase_summary"]
    for v in derived_views:
        exists = _view_exists(con, v)
        details = "present" if exists else "missing (run pipelines/derive_features_ipl_duckdb.py first)"
        # Missing derived views doesn't mean ingestion is corrupt; treat as warning.
        results.append(_warn(f"view:{v}", exists, details))

    # B) Manifest vs disk (warn/hard depending on mismatch)
    if yaml_dir is not None:
        yaml_count = len(list(yaml_dir.glob("*.yaml"))) if yaml_dir.exists() else 0
        ok_matches = _fetch_one_int(
            con, "select count(distinct match_id) from ingestion_manifest where status = 'ok'"
        )
        results.append(
            _warn(
                "manifest_vs_disk",
                ok_matches == yaml_count and yaml_count > 0,
                f"disk_yaml={yaml_count} manifest_ok_matches={ok_matches}",
            )
        )

    # C) Row counts consistency (hard)
    matches_count = _fetch_one_int(con, "select count(*) from matches")
    manifest_ok = _fetch_one_int(con, "select count(distinct match_id) from ingestion_manifest where status='ok'")
    results.append(
        _hard(
            "matches_vs_manifest",
            matches_count == manifest_ok and matches_count > 0,
            f"matches={matches_count} manifest_ok_matches={manifest_ok}",
        )
    )

    # D) Referential integrity (hard)
    orphan_innings = _fetch_one_int(
        con,
        """
        select count(*)
        from innings i
        left join matches m on m.match_id = i.match_id
        where m.match_id is null
        """,
    )
    results.append(_hard("orphan_innings", orphan_innings == 0, f"count={orphan_innings}"))

    orphan_deliveries_innings = _fetch_one_int(
        con,
        """
        select count(*)
        from deliveries d
        left join innings i
          on i.match_id = d.match_id and i.innings_index = d.innings_index
        where i.match_id is null
        """,
    )
    results.append(_hard("orphan_deliveries_innings", orphan_deliveries_innings == 0, f"count={orphan_deliveries_innings}"))

    orphan_deliveries_matches = _fetch_one_int(
        con,
        """
        select count(*)
        from deliveries d
        left join matches m on m.match_id = d.match_id
        where m.match_id is null
        """,
    )
    results.append(_hard("orphan_deliveries_matches", orphan_deliveries_matches == 0, f"count={orphan_deliveries_matches}"))

    # E) Enriched view should not drop rows (hard if view exists)
    if _view_exists(con, "deliveries_enriched"):
        deliveries_count = _fetch_one_int(con, "select count(*) from deliveries")
        enriched_count = _fetch_one_int(con, "select count(*) from deliveries_enriched")
        results.append(
            _hard(
                "deliveries_enriched_rowcount",
                deliveries_count == enriched_count,
                f"deliveries={deliveries_count} deliveries_enriched={enriched_count}",
            )
        )

    # F) Null checks (hard for core keys; warn for optional fields)
    null_match_id = _fetch_one_int(con, "select count(*) from matches where match_id is null")
    results.append(_hard("null:matches.match_id", null_match_id == 0, f"count={null_match_id}"))

    null_team_keys = _fetch_one_int(
        con, "select count(*) from matches where team1_key is null or team2_key is null"
    )
    results.append(_hard("null:matches.team_keys", null_team_keys == 0, f"count={null_team_keys}"))

    null_delivery_keys = _fetch_one_int(
        con,
        """
        select count(*)
        from deliveries
        where striker_key is null or bowler_key is null or is_legal_ball is null or counts_ball_bowled is null
        """,
    )
    results.append(_hard("null:deliveries.core", null_delivery_keys == 0, f"count={null_delivery_keys}"))

    null_dates = _fetch_one_int(con, "select count(*) from matches where start_date is null")
    # Some older records can miss dates in datasets; warn instead of hard failing.
    results.append(_warn("null:matches.start_date", null_dates == 0, f"count={null_dates}"))

    # G) Cricket sanity checks (warnings)
    if _view_exists(con, "innings_summary"):
        too_many_legal = _fetch_one_int(
            con,
            """
            select count(*)
            from innings_summary
            where innings_type = 'normal' and balls_legal > 120
            """,
        )
        results.append(_warn("sanity:normal_innings_balls_legal<=120", too_many_legal == 0, f"count={too_many_legal}"))

    if _view_exists(con, "deliveries_enriched"):
        unknown_phase = _fetch_one_int(
            con,
            """
            select count(*)
            from deliveries_enriched
            where innings_type = 'normal' and phase = 'unknown'
            """,
        )
        results.append(_warn("sanity:phase_unknown_normal", unknown_phase == 0, f"count={unknown_phase}"))

        super_over_death = _fetch_one_int(
            con,
            """
            select count(*)
            from deliveries_enriched
            where innings_type = 'super_over' and phase = 'death'
            """,
        )
        results.append(_hard("sanity:super_over_not_death", super_over_death == 0, f"count={super_over_death}"))

    # H) Sample queries (warnings if empty)
    if _view_exists(con, "deliveries_enriched"):
        death_bowling_rows = _fetch_one_int(
            con,
            """
            with agg as (
              select
                bowler_key,
                min(bowler_raw) as bowler_name,
                sum(runs_total) as runs_conceded,
                sum(ball_bowled_int) as balls_legal,
                sum(case when is_bowler_wicket then 1 else 0 end) as wickets
              from deliveries_enriched
              where
                season is not null
                and season >= 2018
                and innings_type = 'normal'
                and phase = 'death'
              group by bowler_key
            )
            select count(*) from agg
            """,
        )
        results.append(_warn("sample:death_bowling_since_2018_nonempty", death_bowling_rows > 0, f"rows={death_bowling_rows}"))

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the IPL DuckDB dataset (integrity + sanity + samples).")
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/duckdb/cricket_ipl_men.duckdb",
        help="Path to DuckDB database.",
    )
    parser.add_argument(
        "--yaml-dir",
        type=str,
        default="data/cricsheet/ipl_men/yaml",
        help="Path to the local YAML directory (used for manifest vs disk count check).",
    )
    args = parser.parse_args(argv)

    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"ERROR: DB not found: {db_path}", file=sys.stderr)
        return 2

    yaml_dir = Path(args.yaml_dir) if args.yaml_dir else None

    con = duckdb.connect(str(db_path))
    try:
        results = run_validations(con, yaml_dir=yaml_dir)
    finally:
        con.close()

    _print_results(results)

    hard_fail = [r for r in results if r.severity == "HARD" and not r.ok]
    return 0 if not hard_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())

