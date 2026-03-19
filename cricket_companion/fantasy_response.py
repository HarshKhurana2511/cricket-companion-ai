from __future__ import annotations

from typing import Any

from cricket_companion.output_models import ArtifactSource, AssistantOutput, TableArtifact, TableColumn


def build_fantasy_output(*, question: str, fantasy_tool_response: dict[str, Any], request_id: str | None = None) -> AssistantOutput:
    ok = bool(fantasy_tool_response.get("ok"))
    if not ok:
        err = (fantasy_tool_response.get("error") or {}) if isinstance(fantasy_tool_response.get("error"), dict) else {}
        code = err.get("code") or "UNKNOWN"
        msg = err.get("message") or "Fantasy optimization failed."
        return AssistantOutput(
            answer_text=f"Couldn't build a fantasy XI ({code}): {msg}",
            citations=[],
            tables=[],
            charts=[],
            warnings=["No fantasy output was composed because fantasy tool returned ok=false."],
        )

    data = fantasy_tool_response.get("data") if isinstance(fantasy_tool_response.get("data"), dict) else {}
    selected = data.get("selected_team") if isinstance(data.get("selected_team"), list) else []
    captain = data.get("captain")
    vice = data.get("vice_captain")
    total_credits = data.get("total_credits")
    credits_remaining = data.get("credits_remaining")
    projected = data.get("projected_points")
    role_counts = data.get("role_counts") if isinstance(data.get("role_counts"), dict) else {}
    team_counts = data.get("team_counts") if isinstance(data.get("team_counts"), dict) else {}
    warnings = data.get("warnings") if isinstance(data.get("warnings"), list) else []
    alternatives = data.get("alternatives") if isinstance(data.get("alternatives"), list) else []

    lines: list[str] = []
    lines.append("Fantasy XI (optimized)")
    lines.append("")
    if isinstance(projected, (int, float)):
        lines.append(f"- Projected points (incl C/VC): {projected}")
    if total_credits is not None:
        lines.append(f"- Credits used: {total_credits} (remaining: {credits_remaining})")
    if captain:
        lines.append(f"- Captain: {captain}")
    if vice:
        lines.append(f"- Vice-captain: {vice}")
    if role_counts:
        lines.append(f"- Roles: {role_counts}")
    if team_counts:
        lines.append(f"- Teams: {team_counts}")

    rows: list[dict[str, Any]] = []
    for p in selected:
        if not isinstance(p, dict):
            continue
        nm = p.get("name")
        rows.append(
            {
                "name": nm,
                "team": p.get("team"),
                "role": p.get("role"),
                "credits": p.get("credits"),
                "expected_points": p.get("expected_points"),
                "C": "C" if isinstance(captain, str) and nm == captain else "",
                "VC": "VC" if isinstance(vice, str) and nm == vice else "",
                "injury": p.get("injury_status"),
                "probable_xi": p.get("is_probable_xi"),
                "news_reason": p.get("news_reason"),
            }
        )

    table = TableArtifact(
        name="fantasy_team",
        description="Optimized fantasy XI (baseline constraints).",
        columns=[
            TableColumn(name="name", dtype="string"),
            TableColumn(name="team", dtype="string"),
            TableColumn(name="role", dtype="string"),
            TableColumn(name="credits", dtype="float"),
            TableColumn(name="expected_points", dtype="float"),
            TableColumn(name="C", dtype="string"),
            TableColumn(name="VC", dtype="string"),
            TableColumn(name="injury", dtype="string"),
            TableColumn(name="probable_xi", dtype="bool"),
            TableColumn(name="news_reason", dtype="string"),
        ],
        rows=rows,
        source=ArtifactSource(tool_name="fantasy", request_id=request_id),
    )

    alt_rows: list[dict[str, Any]] = []
    # Alternatives are optional; executor may attach them (3.3.4).
    for i, alt in enumerate(alternatives[:3]):
        if not isinstance(alt, dict):
            continue
        team = alt.get("selected_team") if isinstance(alt.get("selected_team"), list) else []
        names = [p.get("name") for p in team if isinstance(p, dict) and isinstance(p.get("name"), str)]
        alt_rows.append(
            {
                "option": f"Option {i+1}",
                "projected_points": alt.get("projected_points"),
                "credits_used": alt.get("total_credits"),
                "credits_remaining": alt.get("credits_remaining"),
                "captain": alt.get("captain"),
                "vice_captain": alt.get("vice_captain"),
                "team": ", ".join(names),
            }
        )

    alt_table = None
    if alt_rows:
        alt_table = TableArtifact(
            name="fantasy_alternatives",
            description="Alternative XIs (near-best) under the same constraints.",
            columns=[
                TableColumn(name="option", dtype="string"),
                TableColumn(name="projected_points", dtype="float"),
                TableColumn(name="credits_used", dtype="float"),
                TableColumn(name="credits_remaining", dtype="float"),
                TableColumn(name="captain", dtype="string"),
                TableColumn(name="vice_captain", dtype="string"),
                TableColumn(name="team", dtype="string"),
            ],
            rows=alt_rows,
            source=ArtifactSource(tool_name="fantasy", request_id=request_id),
        )

    warnings_txt = [str(w) for w in warnings if isinstance(w, str)]
    return AssistantOutput(
        answer_text="\n".join(lines).strip() if lines else (question or "Fantasy optimization completed."),
        citations=[],
        tables=([table] if rows else []) + ([alt_table] if alt_table is not None else []),
        charts=[],
        warnings=warnings_txt,
        assumptions=[
            "Projected points are based on expected_points (or a simple role-based fallback).",
            "News enrichment is best-effort; validate against official playing XI.",
        ],
    )
