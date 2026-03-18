from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


MatchFormat = Literal["T20", "ODI", "TEST", "IPL"]
SimMode = Literal["chase", "set_target"]
Phase = Literal["powerplay", "middle", "death", "unknown"]

PitchType = Literal["flat", "balanced", "slow", "seam", "spin", "unknown"]
DewLevel = Literal["none", "some", "heavy", "unknown"]
BoundarySize = Literal["small", "medium", "large", "unknown"]

SimModel = Literal["baseline", "historical_blend"]


class ScoreState(BaseModel):
    runs: int = Field(ge=0)
    wkts: int = Field(ge=0, le=10)
    balls: int = Field(ge=0, description="Balls elapsed in the current innings (not overs).")


class InningsLimits(BaseModel):
    max_overs: int = Field(ge=1, description="Scheduled overs for the innings (e.g., 20 for T20/IPL, 50 for ODI).")
    reduced_overs: int | None = Field(
        default=None,
        ge=1,
        description="Optional reduced overs (rain/shortened match). If set, this overrides max_overs for ball limits.",
    )

    @property
    def effective_overs(self) -> int:
        return int(self.reduced_overs or self.max_overs)

    @property
    def max_balls(self) -> int:
        return self.effective_overs * 6


class ChaseTarget(BaseModel):
    target_runs: int = Field(ge=1, description="Runs required to win (inclusive target).")
    revised: bool = Field(default=False, description="True if target was revised (e.g., DLS).")


class RecentForm(BaseModel):
    last_n_balls: list[str] = Field(
        default_factory=list,
        description="Optional recent outcomes, e.g. ['0','1','4','W']. Used only as a soft signal.",
        max_length=36,
    )


class MatchState(BaseModel):
    innings: int = Field(ge=1, le=2)
    batting_team: str | None = None
    bowling_team: str | None = None

    score: ScoreState
    limits: InningsLimits

    phase: Phase = "unknown"
    chase: ChaseTarget | None = None
    recent: RecentForm | None = None

    @model_validator(mode="after")
    def _validate_ball_limits(self) -> "MatchState":
        if self.score.balls > self.limits.max_balls:
            raise ValueError(f"balls ({self.score.balls}) cannot exceed max_balls ({self.limits.max_balls}).")
        return self


class Conditions(BaseModel):
    pitch: PitchType = "unknown"
    dew: DewLevel = "unknown"
    boundary_size: BoundarySize = "unknown"
    venue: str | None = None


class TeamStrength(BaseModel):
    batting_rating: float | None = Field(default=None, ge=0.0, le=1.0, description="0..1 coarse team batting strength.")
    bowling_rating: float | None = Field(default=None, ge=0.0, le=1.0, description="0..1 coarse team bowling strength.")


class PlayerStrength(BaseModel):
    name: str
    rating: float = Field(ge=0.0, le=1.0, description="0..1 coarse player strength rating.")
    role: Literal["batter", "bowler", "allrounder", "wk", "unknown"] = "unknown"
    bowling_type: Literal["pace", "spin", "unknown"] = "unknown"


class StrengthModel(BaseModel):
    team: TeamStrength | None = None
    strikers: list[PlayerStrength] = Field(default_factory=list, description="Likely current batters (optional).")
    non_striker: PlayerStrength | None = None
    current_bowler: PlayerStrength | None = None


class SimulationSettings(BaseModel):
    n_sims: int = Field(default=5000, ge=100, le=50000)
    seed: int | None = None
    model: SimModel = "baseline"
    return_distributions: bool = True
    historical_min_samples: int = Field(
        default=250,
        ge=20,
        le=10000,
        description="Minimum historical samples required to trust an empirical bucket (only for historical_blend).",
    )
    historical_blend_max_alpha: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Maximum weight given to historical probabilities when blending with baseline (0..1).",
    )
    historical_window_overs: int = Field(
        default=6,
        ge=1,
        le=12,
        description="How wide (in overs) to search around the scenario's balls_left bucket.",
    )


class SimulationRequest(BaseModel):
    """
    Task 3.2.1: Input schema for Strategy & Scenario Simulator mode.

    This schema is intentionally strict and UI-friendly: it encodes the minimum match state required to simulate
    the remainder of an innings and optionally estimate win probability for chases.
    """

    format: MatchFormat
    mode: SimMode
    match_state: MatchState

    conditions: Conditions | None = None
    strength: StrengthModel | None = None
    simulation: SimulationSettings = Field(default_factory=SimulationSettings)

    @model_validator(mode="after")
    def _validate_mode_consistency(self) -> "SimulationRequest":
        ms = self.match_state

        if self.mode == "chase":
            if ms.innings != 2:
                raise ValueError("mode='chase' requires match_state.innings=2.")
            if ms.chase is None:
                raise ValueError("mode='chase' requires match_state.chase.target_runs.")
            if ms.score.runs >= ms.chase.target_runs:
                raise ValueError("Chase already completed: score.runs must be < target_runs for simulation.")
        else:
            # mode == "set_target"
            if ms.innings != 1:
                raise ValueError("mode='set_target' requires match_state.innings=1.")
            if ms.chase is not None:
                raise ValueError("mode='set_target' must not include match_state.chase.")

        return self


def simulation_request_json_schema() -> dict:
    """
    Convenience helper for UI/forms: returns the JSON schema for SimulationRequest.
    """

    return SimulationRequest.model_json_schema()
