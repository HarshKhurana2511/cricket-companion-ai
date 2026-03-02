from __future__ import annotations

from typing import Any

from cricket_companion.output_models import AssistantOutput, ChartArtifact, ChartType, TableArtifact


def _fmt(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def _has_cols(table: TableArtifact, required: set[str]) -> bool:
    cols = {c.name for c in table.columns}
    return required.issubset(cols)


def infer_charts(table: TableArtifact) -> list[ChartArtifact]:
    """
    Deterministic chart suggestions for UI based on known column patterns.
    """
    cols = {c.name for c in table.columns}
    charts: list[ChartArtifact] = []

    def _add(chart_type: ChartType, title: str, x: str, y: list[str]) -> None:
        charts.append(
            ChartArtifact(
                title=title,
                chart_type=chart_type,
                table_id=table.table_id,
                x=x,
                y=y,
            )
        )

    # Death-overs bowling leaderboard
    if {"bowler_name", "economy"}.issubset(cols):
        _add("bar", "Economy (top results)", x="bowler_name", y=["economy"])
        if "wickets" in cols:
            _add("bar", "Wickets (top results)", x="bowler_name", y=["wickets"])
        return charts

    # Powerplay team run-rate
    if {"batting_team", "run_rate"}.issubset(cols):
        _add("bar", "Powerplay run rate by team", x="batting_team", y=["run_rate"])
        return charts

    # Batter strike-rate leaderboard
    if {"batter_name", "strike_rate"}.issubset(cols):
        _add("bar", "Strike rate by batter", x="batter_name", y=["strike_rate"])
        return charts

    # Venue innings summary over seasons
    if {"season", "avg_runs"}.issubset(cols):
        y = ["avg_runs"]
        if "median_runs" in cols:
            y.append("median_runs")
        _add("line", "Innings runs by season (avg/median)", x="season", y=y)
        return charts

    return charts


def compose_analyst_answer(table: TableArtifact) -> str:
    """
    Deterministic, grounded narrative using ONLY table rows/columns.
    """
    row_count = table.row_count if table.row_count is not None else len(table.rows)
    cols = [c.name for c in table.columns]
    header = table.description or "Analysis results"
    lines: list[str] = [header, f"Rows: {row_count}", f"Columns: {', '.join(cols)}"]

    if not table.rows:
        lines.append("No rows matched the filters.")
        return "\n".join(lines)

    # Show top few rows with a readable subset of columns.
    preview_n = min(5, len(table.rows))

    # Prefer name-ish columns first.
    preferred_order = [
        "bowler_name",
        "batter_name",
        "batting_team",
        "season",
        "economy",
        "wickets",
        "run_rate",
        "strike_rate",
        "runs",
        "balls",
        "balls_legal",
        "runs_conceded",
        "avg_runs",
        "median_runs",
    ]
    ordered = [c for c in preferred_order if c in cols] + [c for c in cols if c not in preferred_order]
    show_cols = ordered[: min(8, len(ordered))]

    lines.append("")
    lines.append("Top rows:")
    for i in range(preview_n):
        r = table.rows[i]
        parts = [f"{k}={_fmt(r.get(k))}" for k in show_cols]
        lines.append(f"{i+1}. " + ", ".join(parts))

    return "\n".join(lines)


def build_analyst_output(
    *,
    table: TableArtifact,
    citations: list[Any] | None = None,
    warnings: list[str] | None = None,
    assumptions: list[str] | None = None,
) -> AssistantOutput:
    charts = infer_charts(table)
    answer = compose_analyst_answer(table)
    return AssistantOutput(
        answer_text=answer,
        citations=citations or [],
        tables=[table],
        charts=charts,
        warnings=warnings or [],
        assumptions=assumptions or [],
    )
