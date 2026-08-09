"""Pydantic request and response contracts for the public REST API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt

from app.models.enums import (
    ActionName,
    AgentFailureType,
    ChaosFailure,
    DecisionSource,
    EventSeverity,
    FireStatus,
    Infrastructure,
    LearningEpisodeSource,
    PolicyName,
    SensorStatus,
    Terrain,
    TerminationReason,
    TrainingDifficulty,
    TrialOutcome,
    TrialScope,
    WindDirection,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ConfigUpdate(StrictModel):
    policy: PolicyName | None = None
    grid_size: StrictInt | None = Field(default=None, ge=5, le=50)
    base_spread_probability: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    wind_direction: WindDirection | None = None
    wind_strength: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    moisture: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    sensor_reliability: StrictFloat | None = Field(default=None, ge=0.25, le=1.0)
    observation_noise: StrictFloat | None = Field(default=None, ge=0.0, le=0.5)
    max_steps: StrictInt | None = Field(default=None, ge=1, le=5_000)
    initial_resources: StrictInt | None = Field(default=None, ge=0, le=1_000)
    resource_effectiveness: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    protection_duration: StrictInt | None = Field(default=None, ge=1, le=50)
    ignition_count: StrictInt | None = Field(default=None, ge=1, le=20)
    burn_duration: StrictInt | None = Field(default=None, ge=1, le=20)
    vegetation_fuel: StrictFloat | None = Field(default=None, ge=0.0, le=2.0)
    open_ground_fuel: StrictFloat | None = Field(default=None, ge=0.0, le=2.0)
    road_fuel: StrictFloat | None = Field(default=None, ge=0.0, le=2.0)
    ember_jump_probability: StrictFloat | None = Field(default=None, ge=0.0, le=0.5)
    wind_drift_probability: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    observation_delay: StrictInt | None = Field(default=None, ge=0, le=5)
    sensor_dropout_probability: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)

    def defined_values(self) -> dict[str, Any]:
        """Return only fields belonging to ``SimulationConfig``."""

        return self.model_dump(exclude_none=True, exclude={"policy"})


class ResetRequest(StrictModel):
    seed: StrictInt | None = Field(default=None, ge=0, le=2**32 - 1)
    policy: PolicyName | None = None
    sensor_reliability: StrictFloat | None = Field(default=None, ge=0.25, le=1.0)
    config: ConfigUpdate | None = None


class StepRequest(StrictModel):
    action: StrictInt | str | None = None
    policy: PolicyName | None = None


class ChaosRequest(StrictModel):
    failure: ChaosFailure | None = None


class CellResponse(StrictModel):
    row: int
    col: int
    terrain: Terrain
    fire: FireStatus
    true_fire: FireStatus
    infrastructure: Infrastructure
    observed: bool
    protection: float = Field(ge=0.0, le=1.0)


class WindResponse(StrictModel):
    direction: WindDirection
    observed_direction: WindDirection | None
    strength: float = Field(ge=0.0, le=1.0)


class SensorResponse(StrictModel):
    reliability: float = Field(ge=0.0, le=1.0)
    configured_reliability: float = Field(ge=0.25, le=1.0)
    status: SensorStatus
    observed_fraction: float = Field(ge=0.0, le=1.0)
    communication_degraded: bool
    recon_active: bool


class ResourceResponse(StrictModel):
    available: int = Field(ge=0)
    initial: int = Field(ge=0)
    consumed: int = Field(ge=0)
    unavailable: int = Field(ge=0)


class MetricsResponse(StrictModel):
    cumulative_reward: StrictFloat
    burned_area: StrictInt = Field(ge=0)
    burned_fraction: StrictFloat = Field(ge=0.0, le=1.0)
    burning_cells: StrictInt = Field(ge=0)
    infrastructure_preserved: StrictInt = Field(ge=0)
    infrastructure_total: StrictInt = Field(ge=0)
    infrastructure_lost: StrictInt = Field(ge=0)
    resources_consumed: StrictInt = Field(ge=0)


class EventResponse(StrictModel):
    step: int = Field(ge=0)
    type: str
    message: str
    severity: EventSeverity


class RewardBreakdownResponse(StrictModel):
    """Auditable components of the most recent one-step reward."""

    new_spread_penalty: StrictFloat = Field(le=0.0)
    infrastructure_loss_penalty: StrictFloat = Field(le=0.0)
    resource_use_penalty: StrictFloat = Field(le=0.0)
    active_fire_penalty: StrictFloat = Field(le=0.0)
    total: StrictFloat


class AgentDecisionResponse(StrictModel):
    """What happened on the latest step, without invented model thoughts."""

    chosen_action: ActionName
    action_label: str = Field(min_length=1)
    decision_source: DecisionSource
    reason: str = Field(min_length=1)
    visible_fire_count: StrictInt = Field(ge=0)
    hidden_cell_count: StrictInt = Field(ge=0)
    resources_before_action: StrictInt = Field(ge=0)
    reward_change: StrictFloat
    reward_breakdown: RewardBreakdownResponse
    # Learning-policy transparency; null for every other policy.
    explored: StrictBool | None = None
    exploration_rate: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    q_values: dict[ActionName, float] | None = None


class AgentFailureResponse(StrictModel):
    """One detected agentic failure, judged against analysis-only truth."""

    step: StrictInt = Field(ge=0)
    policy: str = Field(min_length=1)
    failure_type: AgentFailureType
    message: str = Field(min_length=1)
    hidden_fire_cells: StrictInt = Field(ge=0)
    visible_fire_cells: StrictInt = Field(ge=0)
    true_burning_cells: StrictInt = Field(ge=0)


class LearningEpisodeRecord(StrictModel):
    """One completed learning-policy episode (interactive or headless)."""

    episode: StrictInt = Field(ge=1)
    reward: StrictFloat
    outcome: TrialOutcome
    epsilon: StrictFloat = Field(ge=0.0, le=1.0)
    source: LearningEpisodeSource


class LearningStatsResponse(StrictModel):
    """Honest telemetry for the in-session Q-learning policy."""

    policy: PolicyName
    episodes_trained: StrictInt = Field(ge=0)
    epsilon: StrictFloat = Field(ge=0.0, le=1.0)
    learning_rate: StrictFloat = Field(gt=0.0, le=1.0)
    discount: StrictFloat = Field(ge=0.0, le=1.0)
    updates_applied: StrictInt = Field(ge=0)
    states_visited: StrictInt = Field(ge=0)
    reward_history: list[LearningEpisodeRecord]
    early_average: StrictFloat | None
    recent_average: StrictFloat | None
    improvement: StrictFloat | None
    history_limit: StrictInt = Field(ge=1)
    note: str = Field(min_length=1)


class TrialStatsResponse(StrictModel):
    """Aggregates for this in-memory server session, never global user stats."""

    scope: TrialScope
    trials_started: StrictInt = Field(ge=0)
    trials_completed: StrictInt = Field(ge=0)
    successes: StrictInt = Field(ge=0)
    partials: StrictInt = Field(ge=0)
    needs_improvement: StrictInt = Field(ge=0)
    total_experience_points: StrictInt = Field(ge=0)
    current_level: StrictInt = Field(ge=1)
    average_completed_reward: StrictFloat


class ActionCountsResponse(StrictModel):
    hold: StrictInt = Field(ge=0)
    protect_north: StrictInt = Field(ge=0)
    protect_south: StrictInt = Field(ge=0)
    protect_east: StrictInt = Field(ge=0)
    protect_west: StrictInt = Field(ge=0)
    recon: StrictInt = Field(ge=0)


class RunSummaryResponse(StrictModel):
    trial_id: str = Field(pattern=r"^trial_[0-9a-f]{32}$")
    fire_extinguished: StrictBool
    mission_completed: StrictBool
    success: StrictBool
    outcome: TrialOutcome
    title: str = Field(min_length=1)
    what_happened: list[str] = Field(min_length=1)
    reward_lesson: str = Field(min_length=1)
    improvements: list[str] = Field(min_length=1)
    xp_earned: StrictInt = Field(ge=0)
    action_counts: ActionCountsResponse
    termination_reason: TerminationReason
    final_metrics: MetricsResponse
    failure_counts: dict[AgentFailureType, StrictInt]


class StateResponse(StrictModel):
    grid_size: int
    seed: int
    step_count: int
    max_steps: StrictInt = Field(ge=1)
    cells: list[CellResponse]
    wind: WindResponse
    sensor: SensorResponse
    resources: ResourceResponse
    metrics: MetricsResponse
    policy: PolicyName
    current_action: str
    last_reward: StrictFloat
    agent_decision: AgentDecisionResponse | None
    trial_stats: TrialStatsResponse
    run_summary: RunSummaryResponse | None
    agent_failures: list[AgentFailureResponse]
    learning: LearningStatsResponse
    events: list[EventResponse]
    done: bool
    termination_reason: TerminationReason | None = None


class HealthResponse(StrictModel):
    status: str
    service: str
    version: str


class ActionDescription(StrictModel):
    value: int
    name: str
    description: str


class PolicyDescription(StrictModel):
    id: PolicyName
    label: str
    description: str
    native_available: bool = True


class PoliciesResponse(StrictModel):
    policies: list[PolicyDescription]
    default_policy: PolicyName
    current_policy: PolicyName
    actions: list[ActionDescription]


class TrainRequest(StrictModel):
    """Bounded headless training request; runs synchronously under the session lock."""

    episodes: StrictInt = Field(default=25, ge=1, le=100)
    seed: StrictInt | None = Field(default=None, ge=0, le=2**32 - 1)
    difficulty: TrainingDifficulty = TrainingDifficulty.STANDARD


class TrainEpisodeResult(StrictModel):
    episode: StrictInt = Field(ge=1)
    seed: StrictInt = Field(ge=0)
    steps: StrictInt = Field(ge=1)
    reward: StrictFloat
    outcome: TrialOutcome
    epsilon: StrictFloat = Field(ge=0.0, le=1.0)
    sensor_reliability: StrictFloat = Field(ge=0.25, le=1.0)


class TrainResponse(StrictModel):
    episodes_run: StrictInt = Field(ge=1)
    difficulty: TrainingDifficulty
    results: list[TrainEpisodeResult]
    stats: LearningStatsResponse
    note: str = Field(min_length=1)


class KnowledgeResult(StrictModel):
    """One retrieved field-manual chunk with its citation metadata."""

    rank: StrictInt = Field(ge=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    score: StrictFloat = Field(ge=0.0, le=1.0)


class KnowledgeResponse(StrictModel):
    query: str = Field(min_length=1)
    results: list[KnowledgeResult]
    answer: str | None
    note: str = Field(min_length=1)
