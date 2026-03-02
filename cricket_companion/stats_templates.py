from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import duckdb


@dataclass(frozen=True)
class TemplatePlan:
    template_id: str
    sql: str
    params: dict[str, Any]
    description: str


class TemplateBuildError(Exception):
    def __init__(self, message: str, *, clarifying_question: str | None = None) -> None:
        super().__init__(message)
        self.clarifying_question = clarifying_question


def available_template_ids() -> list[str]:
    return [
        "health_summary",
        "death_bowling_leaderboard",
        "powerplay_team_runrate",
        "batter_strike_rate_leaderboard",
        "venue_innings_summary",
    ]


def select_template_id(question: str, spec: dict[str, Any]) -> str | None:
    q = (question or "").lower()

    if "death" in q:
        return "death_bowling_leaderboard"

    if "powerplay" in q or re.search(r"\bpp\b", q):
        return "powerplay_team_runrate"

    if "strike rate" in q or re.search(r"\bsr\b", q):
        return "batter_strike_rate_leaderboard"

    params = spec.get("params") or {}
    if isinstance(params, dict) and (params.get("venue") or params.get("venue_key") or "venue" in q or " at " in q):
        return "venue_innings_summary"

    # If the user asked for "top bowlers" but didn't specify phase, default to death overs is risky.
    return None


def _norm_key(text: str) -> str:
    s = (text or "").strip().lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _resolve_venue_key(con: duckdb.DuckDBPyConnection, venue_query: str) -> str:
    vq = venue_query.strip()
    if not vq:
        raise TemplateBuildError("Missing venue.", clarifying_question="Which venue should I filter on (e.g., Wankhede Stadium)?")

    # Try exact key match first.
    exact = con.execute(
        "select distinct venue_key, venue_raw from matches where venue_key = ? limit 5",
        [_norm_key(vq)],
    ).fetchall()
    if len(exact) == 1:
        return str(exact[0][0])

    # Next try contains match on raw venue.
    candidates = con.execute(
        """
        select distinct venue_key, venue_raw
        from matches
        where lower(venue_raw) like '%' || lower(?) || '%'
        order by venue_raw
        limit 20
        """,
        [vq],
    ).fetchall()

    if not candidates:
        raise TemplateBuildError(
            f"Venue not found for: {venue_query!r}.",
            clarifying_question="Which venue do you mean? Provide the exact stadium name as in scorecards (e.g., 'Wankhede Stadium').",
        )
    if len(candidates) > 1:
        options = ", ".join(sorted({str(r[1]) for r in candidates if r[1]}))[:400]
        raise TemplateBuildError(
            f"Venue is ambiguous for: {venue_query!r}. Candidates: {options}",
            clarifying_question=f"Which venue exactly? Some matches: {options}",
        )
    return str(candidates[0][0])


def build_template_plan(
    *,
    template_id: str,
    spec: dict[str, Any],
    db_path: str,
) -> TemplatePlan:
    """
    Builds a deterministic SQL query plan for an allowlisted, read-only stats query.

    `spec` is expected to match the MCP StatsQuerySpec (question + optional fields).
    """
    question = str(spec.get("question") or "")
    since_year = spec.get("since_year")
    until_year = spec.get("until_year")
    params_in = spec.get("params") or {}
    if not isinstance(params_in, dict):
        params_in = {}

    # Normalization (keep these bounded/simple in 2.4.2).
    params: dict[str, Any] = {
        "since_year": int(since_year) if since_year is not None else None,
        "until_year": int(until_year) if until_year is not None else None,
    }

    if template_id == "health_summary":
        return TemplatePlan(
            template_id=template_id,
            description="Dataset health summary (counts and season range).",
            params={},
            sql="""
            select
              (select count(*) from matches) as matches,
              (select count(*) from innings) as innings,
              (select count(*) from deliveries) as deliveries,
              (select min(season) from matches) as season_min,
              (select max(season) from matches) as season_max
            """,
        )

    if template_id == "death_bowling_leaderboard":
        return TemplatePlan(
            template_id=template_id,
            description="Death-overs bowling leaderboard (economy, wickets) (normal innings only).",
            params=params,
            sql="""
            select
              bowler_key,
              min(bowler_raw) as bowler_name,
              sum(runs_total) as runs_conceded,
              sum(ball_bowled_int) as balls_legal,
              (sum(ball_bowled_int) / 6.0) as overs_legal,
              (sum(runs_total) / nullif(sum(ball_bowled_int) / 6.0, 0)) as economy,
              sum(case when is_bowler_wicket then 1 else 0 end) as wickets,
              sum(case when is_legal_dot then 1 else 0 end) as dots_legal,
              sum(case when is_boundary then 1 else 0 end) as boundaries
            from deliveries_enriched
            where
              innings_type = 'normal'
              and phase = 'death'
              and (:since_year is null or season >= :since_year)
              and (:until_year is null or season <= :until_year)
            group by bowler_key
            order by
              wickets desc,
              economy asc nulls last,
              balls_legal desc,
              bowler_name asc
            """,
        )

    if template_id == "powerplay_team_runrate":
        return TemplatePlan(
            template_id=template_id,
            description="Powerplay team run-rate leaderboard (normal innings only).",
            params=params,
            sql="""
            select
              batting_team_key,
              min(batting_team_raw) as batting_team,
              sum(runs_total) as runs,
              sum(ball_bowled_int) as balls_legal,
              (sum(runs_total) / nullif(sum(ball_bowled_int) / 6.0, 0)) as run_rate
            from deliveries_enriched
            where
              innings_type = 'normal'
              and phase = 'powerplay'
              and (:since_year is null or season >= :since_year)
              and (:until_year is null or season <= :until_year)
            group by batting_team_key
            order by
              run_rate desc nulls last,
              runs desc,
              batting_team asc
            """,
        )

    if template_id == "batter_strike_rate_leaderboard":
        min_balls = params_in.get("min_balls")
        try:
            min_balls_i = int(min_balls) if min_balls is not None else 60
        except Exception:
            min_balls_i = 60
        if min_balls_i < 1:
            min_balls_i = 1
        if min_balls_i > 10000:
            min_balls_i = 10000

        params2 = dict(params)
        params2["min_balls"] = min_balls_i

        return TemplatePlan(
            template_id=template_id,
            description="Batter strike-rate leaderboard (min balls faced filter) (normal innings only).",
            params=params2,
            sql="""
            select
              striker_key,
              min(striker_raw) as batter_name,
              sum(runs_batter) as runs,
              sum(ball_faced_int) as balls,
              (100.0 * sum(runs_batter) / nullif(sum(ball_faced_int), 0)) as strike_rate,
              sum(case when player_dismissed_key = striker_key then 1 else 0 end) as outs
            from deliveries_enriched
            where
              innings_type = 'normal'
              and (:since_year is null or season >= :since_year)
              and (:until_year is null or season <= :until_year)
            group by striker_key
            having sum(ball_faced_int) >= :min_balls
            order by
              strike_rate desc nulls last,
              runs desc,
              batter_name asc
            """,
        )

    if template_id == "venue_innings_summary":
        venue_q = params_in.get("venue") or params_in.get("venue_key") or ""
        con = duckdb.connect(db_path, read_only=True)
        try:
            vk = _resolve_venue_key(con, str(venue_q))
        finally:
            con.close()

        params2 = dict(params)
        params2["venue_key"] = vk
        return TemplatePlan(
            template_id=template_id,
            description=f"Innings totals summary by season at venue_key={vk} (normal innings only).",
            params=params2,
            sql="""
            select
              season,
              count(*) as innings,
              avg(runs_total) as avg_runs,
              median(runs_total) as median_runs,
              min(runs_total) as min_runs,
              max(runs_total) as max_runs
            from innings_summary
            where
              innings_type = 'normal'
              and venue_key = :venue_key
              and (:since_year is null or season >= :since_year)
              and (:until_year is null or season <= :until_year)
            group by season
            order by season asc
            """,
        )

    raise TemplateBuildError(
        f"Unknown template_id: {template_id!r}",
        clarifying_question=f"Choose one of: {', '.join(available_template_ids())}",
    )

