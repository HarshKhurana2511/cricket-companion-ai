from __future__ import annotations

from typing import Any

from cricket_companion.output_models import ArtifactSource, AssistantOutput, ChartArtifact, TableArtifact, TableColumn


def build_sim_output(*, question: str, sim_tool_response: dict[str, Any], request_id: str | None = None) -> AssistantOutput:
    ok = bool(sim_tool_response.get("ok"))
    if not ok:
        err = (sim_tool_response.get("error") or {}) if isinstance(sim_tool_response.get("error"), dict) else {}
        code = err.get("code") or "UNKNOWN"
        msg = err.get("message") or "Simulation failed."
        return AssistantOutput(
            answer_text=f"Couldn't simulate this scenario ({code}): {msg}",
            citations=[],
            tables=[],
            charts=[],
            warnings=["No simulation output was composed because sim tool returned ok=false."],
        )

    data = sim_tool_response.get("data") if isinstance(sim_tool_response.get("data"), dict) else {}
    mode = str(data.get("mode") or "sim")
    fmt = str(data.get("format") or "")
    n_sims = data.get("n_sims")
    model = data.get("model")
    win_p = data.get("win_probability")
    expected = data.get("expected_final_runs")
    pct = data.get("score_percentiles") if isinstance(data.get("score_percentiles"), dict) else {}
    hist = data.get("historical") if isinstance(data.get("historical"), dict) else None
    inputs = data.get("inputs") if isinstance(data.get("inputs"), dict) else {}

    lines: list[str] = []
    model_txt = f", model={model}" if isinstance(model, str) and model else ""
    header = f"{fmt} simulator ({mode}{model_txt})".strip()
    lines.append(header)
    lines.append("")
    if isinstance(win_p, (int, float)):
        lines.append(f"- Win probability: {float(win_p) * 100:.1f}%")
    if isinstance(expected, (int, float)):
        lines.append(f"- Expected final runs: {expected}")
    if pct:
        p10 = pct.get("p10")
        p50 = pct.get("p50")
        p90 = pct.get("p90")
        if p10 is not None and p50 is not None and p90 is not None:
            lines.append(f"- Final runs (P10/P50/P90): {p10} / {p50} / {p90}")
    if isinstance(n_sims, int):
        lines.append(f"- Sims: {n_sims}")
    if isinstance(hist, dict) and isinstance((hist.get("overall") or {}), dict):
        overall = hist.get("overall") or {}
        n_hist = overall.get("n")
        win_hist = overall.get("win_rate")
        if n_hist is not None:
            lines.append(f"- Historical samples (windowed): {n_hist}")
        if isinstance(win_hist, (int, float)):
            lines.append(f"- Historical win rate (overall): {float(win_hist) * 100:.1f}%")

    # Recommendations (deterministic, lightweight).
    try:
        if isinstance(inputs, dict) and mode == "chase":
            runs = int(inputs.get("runs") or 0)
            wkts = int(inputs.get("wkts") or 0)
            balls = int(inputs.get("balls") or 0)
            max_overs = int(inputs.get("max_overs") or 20)
            target_runs = inputs.get("target_runs")
            if isinstance(target_runs, int):
                max_balls = max_overs * 6
                balls_left = max(0, max_balls - balls)
                runs_left = max(0, target_runs - runs)
                wkts_left = max(0, 10 - wkts)
                rpb = (runs_left / balls_left) if balls_left > 0 else 99.0
                rrr = rpb * 6.0

                recs: list[str] = []
                if balls_left <= 0:
                    recs = []
                else:
                    if wkts_left <= 2 and rrr >= 10:
                        recs.append("Low wickets + high RRR: prioritize singles early in the over; take boundary risks on 4th/5th/6th ball.")
                    elif wkts_left >= 6 and rrr >= 10:
                        recs.append("Plenty of wickets + high RRR: maximize boundary attempts; accept 1 wicket if it meaningfully boosts boundary rate.")
                    elif rrr <= 8:
                        recs.append("Manageable RRR: minimize dot balls, keep strike rotating, and target 1 boundary per over.")
                    else:
                        recs.append("Moderate pressure: avoid consecutive dots; identify 1–2 overs to attack (matchups/field).")

                    if isinstance(win_p, (int, float)) and float(win_p) < 0.35:
                        recs.append("Win% is low in this model: treat it as a 'must-win over' situation; increase risk earlier.")
                    elif isinstance(win_p, (int, float)) and float(win_p) > 0.70:
                        recs.append("Win% is high: avoid high-variance shots early in the over; preserve wickets.")

                if recs:
                    lines.append("")
                    lines.append("Recommendations")
                    for r in recs[:4]:
                        lines.append(f"- {r}")
    except Exception:
        pass

    warnings = data.get("warnings") if isinstance(data.get("warnings"), list) else []
    warnings_txt = [str(w) for w in warnings if isinstance(w, str)]

    rows: list[dict[str, Any]] = []
    if isinstance(model, str) and model:
        rows.append({"metric": "model", "value": model})
    if isinstance(win_p, (int, float)):
        rows.append({"metric": "win_probability", "value": float(win_p)})
    if isinstance(expected, (int, float)):
        rows.append({"metric": "expected_final_runs", "value": expected})
    if pct:
        for k in ("p10", "p50", "p90"):
            if k in pct:
                rows.append({"metric": f"final_runs_{k}", "value": pct.get(k)})
    if isinstance(hist, dict) and isinstance((hist.get("overall") or {}), dict):
        overall = hist.get("overall") or {}
        if overall.get("n") is not None:
            rows.append({"metric": "historical_samples_overall", "value": overall.get("n")})
        if overall.get("win_rate") is not None:
            rows.append({"metric": "historical_win_rate_overall", "value": overall.get("win_rate")})

    table = TableArtifact(
        name="simulation_summary",
        description="Baseline Monte Carlo simulation summary.",
        columns=[TableColumn(name="metric", dtype="string"), TableColumn(name="value", dtype="string")],
        rows=rows,
        source=ArtifactSource(tool_name="sim", request_id=request_id),
    )

    charts: list[ChartArtifact] = []
    tables: list[TableArtifact] = []
    if rows:
        tables.append(table)

    # Percentiles table + chart.
    if pct and all(k in pct for k in ("p10", "p50", "p90")):
        p_rows = [
            {"percentile": "p10", "runs": pct.get("p10")},
            {"percentile": "p50", "runs": pct.get("p50")},
            {"percentile": "p90", "runs": pct.get("p90")},
        ]
        p_table = TableArtifact(
            name="final_runs_percentiles",
            description="Final runs distribution percentiles.",
            columns=[TableColumn(name="percentile", dtype="string"), TableColumn(name="runs", dtype="int")],
            rows=p_rows,
            source=ArtifactSource(tool_name="sim", request_id=request_id),
        )
        tables.append(p_table)
        charts.append(
            ChartArtifact(
                title="Final runs percentiles (P10/P50/P90)",
                chart_type="bar",
                table_id=p_table.table_id,
                x="percentile",
                y=["runs"],
                notes="Higher is better for chasing; use alongside win probability.",
            )
        )

    return AssistantOutput(
        answer_text="\n".join(lines).strip() if lines else (question or "Simulation completed."),
        citations=[],
        tables=tables,
        charts=charts,
        warnings=warnings_txt,
        assumptions=[
            "Baseline model uses heuristic ball outcome probabilities by phase/pressure.",
            "Extras and batter/bowler matchups are not modeled in v1.",
        ],
    )
