from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


FantasyPlatform = Literal["dream11", "my11circle", "generic"]
MatchFormat = Literal["T20", "ODI", "TEST", "IPL"]

PlayerRole = Literal["wk", "bat", "ar", "bowl", "unknown"]
RiskProfile = Literal["safe", "balanced", "high_variance"]


class RoleMinMax(BaseModel):
    min: int = Field(ge=0)
    max: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_min_le_max(self) -> "RoleMinMax":
        if self.min > self.max:
            raise ValueError("role constraint must satisfy min <= max")
        return self


class RoleConstraints(BaseModel):
    wk: RoleMinMax
    bat: RoleMinMax
    ar: RoleMinMax
    bowl: RoleMinMax

    @model_validator(mode="after")
    def _validate_nonzero_max(self) -> "RoleConstraints":
        # Guardrail: at least one role must allow selection.
        if (self.wk.max + self.bat.max + self.ar.max + self.bowl.max) <= 0:
            raise ValueError("role max totals must be > 0")
        return self


class CaptainRules(BaseModel):
    captain_multiplier: float = Field(default=2.0, ge=1.0, le=3.0)
    vice_captain_multiplier: float = Field(default=1.5, ge=1.0, le=3.0)


class ScoringRules(BaseModel):
    """
    Generic scoring schema. Platforms differ; this is intentionally flexible.

    For MVP, you can either:
    - provide a full points table here, or
    - leave it sparse and let the optimizer use `expected_points` directly per player.
    """

    # Batting
    run: float = 1.0
    four: float = 1.0
    six: float = 2.0
    duck: float = -2.0

    # Bowling
    wicket: float = 25.0
    maiden_over: float = 8.0

    # Fielding
    catch: float = 8.0
    stumping: float = 12.0
    run_out: float = 6.0

    # Bonuses (optional, platform-specific)
    strike_rate_bonus: dict[str, float] = Field(default_factory=dict, description="Bucketed SR bonus/penalty, if used.")
    economy_bonus: dict[str, float] = Field(default_factory=dict, description="Bucketed economy bonus/penalty, if used.")


class FantasyRules(BaseModel):
    platform: FantasyPlatform = "generic"
    format: MatchFormat = "T20"

    team_count: int = Field(default=11, ge=1, le=15)
    budget: float = Field(default=100.0, gt=0.0)
    max_from_one_team: int = Field(default=7, ge=1, le=11)

    roles: RoleConstraints
    captain: CaptainRules = Field(default_factory=CaptainRules)
    scoring: ScoringRules = Field(default_factory=ScoringRules)

    @model_validator(mode="after")
    def _validate_team_size_constraints(self) -> "FantasyRules":
        mins = self.roles.wk.min + self.roles.bat.min + self.roles.ar.min + self.roles.bowl.min
        maxs = self.roles.wk.max + self.roles.bat.max + self.roles.ar.max + self.roles.bowl.max
        if mins > self.team_count:
            raise ValueError("sum(role mins) cannot exceed team_count")
        if maxs < self.team_count:
            raise ValueError("sum(role maxs) must be >= team_count")
        if self.max_from_one_team > self.team_count:
            raise ValueError("max_from_one_team cannot exceed team_count")
        return self


class PlayerPoolEntry(BaseModel):
    name: str
    team: str
    role: PlayerRole = "unknown"
    credits: float = Field(gt=0.0)

    # Optional signals used by the optimizer/explanations later.
    expected_points: float | None = Field(default=None, ge=0.0)
    is_probable_xi: bool | None = None
    injury_status: Literal["fit", "doubtful", "out", "unknown"] = "unknown"

    metadata: dict[str, Any] = Field(default_factory=dict)


class FantasyPreferences(BaseModel):
    risk_profile: RiskProfile = "balanced"
    must_include: list[str] = Field(default_factory=list)
    must_exclude: list[str] = Field(default_factory=list)
    use_news: bool = Field(default=True, description="If true, allow web-based injury/availability enrichment (3.3.3).")


class FantasyRequest(BaseModel):
    """
    Task 3.3.1: Fantasy Draft Assistant input schema (rules + player pool).
    """

    rules: FantasyRules
    teams: list[str] = Field(min_length=2, max_length=2, description="Exactly two teams for a match.")
    players: list[PlayerPoolEntry] = Field(min_length=2)
    preferences: FantasyPreferences = Field(default_factory=FantasyPreferences)

    @model_validator(mode="after")
    def _validate_player_pool(self) -> "FantasyRequest":
        team_set = {t.strip() for t in self.teams if isinstance(t, str)}
        if len(team_set) != 2:
            raise ValueError("teams must contain exactly 2 distinct team identifiers")

        bad = [p.name for p in self.players if p.team.strip() not in team_set]
        if bad:
            raise ValueError(f"players contain teams not in `teams`: {bad[:5]}")
        return self


def fantasy_request_json_schema() -> dict:
    return FantasyRequest.model_json_schema()
