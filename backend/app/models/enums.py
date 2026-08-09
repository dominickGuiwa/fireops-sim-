"""Small, explicit enumerations used by the simulation and REST API."""

from __future__ import annotations

from enum import IntEnum, StrEnum


class Action(IntEnum):
    """Abstract emergency-response actions exposed to agents."""

    HOLD = 0
    PROTECT_NORTH = 1
    PROTECT_SOUTH = 2
    PROTECT_EAST = 3
    PROTECT_WEST = 4
    RECON = 5


class Terrain(StrEnum):
    VEGETATION = "vegetation"
    OPEN_GROUND = "open_ground"
    ROAD = "road"


class FireStatus(StrEnum):
    UNBURNED = "unburned"
    BURNING = "burning"
    BURNED = "burned"
    UNKNOWN = "unknown"


class Infrastructure(StrEnum):
    NONE = "none"
    MEDICAL = "medical"
    POWER = "power"
    COMMUNICATIONS = "communications"


class WindDirection(StrEnum):
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"


class SensorStatus(StrEnum):
    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    OFFLINE = "offline"


class ChaosFailure(StrEnum):
    SENSOR_OFFLINE = "sensor_offline"
    SENSOR_DEGRADED = "sensor_degraded"
    COMMUNICATIONS_DEGRADED = "communications_degraded"
    WIND_CHANGE = "wind_change"
    RESOURCE_UNAVAILABLE = "resource_unavailable"


class EventSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class PolicyName(StrEnum):
    RANDOM = "random"
    RULE = "rule"
    LEARNING = "learning"
    PPO_NOMINAL = "ppo_nominal"
    PPO_ROBUST = "ppo_robust"


class ActionName(StrEnum):
    """Stable, readable action identifiers used by the learning API."""

    HOLD = "HOLD"
    PROTECT_NORTH = "PROTECT_NORTH"
    PROTECT_SOUTH = "PROTECT_SOUTH"
    PROTECT_EAST = "PROTECT_EAST"
    PROTECT_WEST = "PROTECT_WEST"
    RECON = "RECON"


class DecisionSource(StrEnum):
    POLICY = "policy"
    MANUAL = "manual"


class TrialOutcome(StrEnum):
    SUCCESS = "success"
    PARTIAL = "partial"
    NEEDS_IMPROVEMENT = "needs_improvement"


class TrialScope(StrEnum):
    SERVER_SESSION = "server_session"


class TerminationReason(StrEnum):
    FIRE_EXTINGUISHED = "fire_extinguished"
    MAX_STEPS = "max_steps"


class AgentFailureType(StrEnum):
    """Auditable agentic failure modes detected by comparing the degraded
    observation the agent acted on against the true synthetic world state."""

    PERCEPTION_GAP = "perception_gap"
    MISDIRECTED_PROTECTION = "misdirected_protection"
    IDLE_WHILE_BURNING = "idle_while_burning"
    RESOURCE_EXHAUSTED_REQUEST = "resource_exhausted_request"
    WIND_MISREAD = "wind_misread"


class LearningEpisodeSource(StrEnum):
    """Where a learning-policy episode came from."""

    INTERACTIVE = "interactive"
    TRAINING = "training"


class TrainingDifficulty(StrEnum):
    """Named domain-randomization presets for headless training episodes."""

    STANDARD = "standard"
    HARD = "hard"
    BRUTAL = "brutal"


def parse_action(value: Action | int | str) -> Action:
    """Parse either a numeric action or a case-insensitive enum name."""

    if isinstance(value, Action):
        return value
    if isinstance(value, str):
        candidate = value.strip().upper()
        if candidate.isdigit() or (candidate.startswith("-") and candidate[1:].isdigit()):
            value = int(candidate)
        else:
            try:
                return Action[candidate]
            except KeyError as exc:
                raise ValueError(f"Unknown action: {value}") from exc
    try:
        return Action(int(value))
    except (TypeError, ValueError) as exc:
        valid = ", ".join(f"{action.value}={action.name}" for action in Action)
        raise ValueError(f"Action must be one of {valid}") from exc
