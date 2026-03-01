from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


DERIVED_SCHEMA_SQL = """
create or replace view deliveries_enriched as
select
  d.match_id,
  d.innings_index,
  d.innings_type,
  i.innings_name_raw,
  i.batting_team_raw,
  i.batting_team_key,
  i.bowling_team_raw,
  i.bowling_team_key,
  m.competition,
  m.match_type,
  m.gender,
  m.season,
  m.start_date,
  m.venue_raw,
  m.venue_key,
  m.city_raw,
  m.city_key,
  m.result_type,
  m.winner_team_raw,
  m.winner_team_key,

  d.over_number,
  d.ball_in_over,
  d.ball_number_in_innings,

  case
    when d.innings_type = 'super_over' then 'super_over'
    when d.over_number between 0 and 5 then 'powerplay'
    when d.over_number between 6 and 14 then 'middle'
    when d.over_number between 15 and 19 then 'death'
    else 'unknown'
  end as phase,

  d.striker_raw,
  d.striker_key,
  d.non_striker_raw,
  d.non_striker_key,
  d.bowler_raw,
  d.bowler_key,

  d.runs_batter,
  d.runs_extras,
  d.runs_total,

  d.extra_wides,
  d.extra_noballs,
  d.extra_byes,
  d.extra_legbyes,
  d.extra_penalty,
  d.has_extras,

  (d.extra_wides > 0) as is_wide,
  (d.extra_noballs > 0) as is_no_ball,
  (d.extra_byes > 0) as is_bye,
  (d.extra_legbyes > 0) as is_legbye,
  (d.extra_penalty > 0) as is_penalty,

  d.is_wicket,
  d.wicket_kind,
  d.player_dismissed_raw,
  d.player_dismissed_key,
  d.fielders_json,

  d.is_legal_ball,
  d.counts_ball_faced,
  d.counts_ball_bowled,

  case when d.counts_ball_bowled then 1 else 0 end as ball_bowled_int,
  case when d.counts_ball_faced then 1 else 0 end as ball_faced_int,

  (d.runs_total = 0) as is_dot_total,
  (d.runs_batter = 0) as is_dot_bat,
  (d.is_legal_ball and d.runs_total = 0) as is_legal_dot,

  (d.runs_batter = 4) as is_four,
  (d.runs_batter = 6) as is_six,
  (d.runs_batter = 4 or d.runs_batter = 6) as is_boundary,

  (
    d.is_wicket
    and coalesce(lower(d.wicket_kind), '') not in (
      'run out',
      'retired hurt',
      'retired out',
      'obstructing the field',
      'timed out',
      'hit the ball twice',
      'handled the ball'
    )
  ) as is_bowler_wicket,

  (d.striker_key || '|' || d.bowler_key) as batter_bowler_key,
  (i.batting_team_key || '|' || i.bowling_team_key) as team_matchup_key,
  case
    when i.batting_team_key <= i.bowling_team_key then (i.batting_team_key || '|' || i.bowling_team_key)
    else (i.bowling_team_key || '|' || i.batting_team_key)
  end as team_pair_key
from deliveries d
join innings i
  on i.match_id = d.match_id and i.innings_index = d.innings_index
join matches m
  on m.match_id = d.match_id
;

create or replace view innings_summary as
select
  match_id,
  innings_index,
  innings_type,
  innings_name_raw,
  competition,
  season,
  start_date,
  venue_key,
  venue_raw,
  batting_team_key,
  batting_team_raw,
  bowling_team_key,
  bowling_team_raw,
  sum(runs_total) as runs_total,
  sum(runs_batter) as runs_batter,
  sum(runs_extras) as runs_extras,
  sum(case when is_wicket then 1 else 0 end) as wickets_event,
  sum(case when is_bowler_wicket then 1 else 0 end) as wickets_bowler,
  sum(ball_bowled_int) as balls_legal,
  (sum(ball_bowled_int) / 6.0) as overs_legal,
  (sum(runs_total) / nullif(sum(ball_bowled_int) / 6.0, 0)) as run_rate,
  sum(case when is_legal_dot then 1 else 0 end) as dots_legal,
  sum(case when is_boundary then 1 else 0 end) as boundaries,
  sum(case when is_four then 1 else 0 end) as fours,
  sum(case when is_six then 1 else 0 end) as sixes,
  sum(extra_wides) as wides,
  sum(extra_noballs) as noballs,
  sum(extra_byes) as byes,
  sum(extra_legbyes) as legbyes
from deliveries_enriched
group by
  match_id,
  innings_index,
  innings_type,
  innings_name_raw,
  competition,
  season,
  start_date,
  venue_key,
  venue_raw,
  batting_team_key,
  batting_team_raw,
  bowling_team_key,
  bowling_team_raw
;

create or replace view innings_phase_summary as
select
  match_id,
  innings_index,
  innings_type,
  phase,
  competition,
  season,
  batting_team_key,
  bowling_team_key,
  sum(runs_total) as runs_total,
  sum(case when is_bowler_wicket then 1 else 0 end) as wickets_bowler,
  sum(ball_bowled_int) as balls_legal,
  (sum(runs_total) / nullif(sum(ball_bowled_int) / 6.0, 0)) as run_rate
from deliveries_enriched
group by
  match_id,
  innings_index,
  innings_type,
  phase,
  competition,
  season,
  batting_team_key,
  bowling_team_key
;
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create derived-feature views in the IPL DuckDB.")
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/duckdb/cricket_ipl_men.duckdb",
        help="Path to DuckDB database.",
    )
    args = parser.parse_args(argv)

    db_path = Path(args.db_path)
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")

    con = duckdb.connect(str(db_path))
    try:
        con.execute(DERIVED_SCHEMA_SQL)
        print("Created/updated views: deliveries_enriched, innings_summary, innings_phase_summary")
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())

