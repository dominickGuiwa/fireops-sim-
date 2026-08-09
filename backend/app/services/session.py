"""Thread-safe in-memory owner for the single v0.1 simulation session."""

from __future__ import annotations

import os
from threading import RLock
from typing import Any
from uuid import uuid4

import numpy as np

from app.agents import PPOAgent, QLearningAgent, RandomAgent, RuleBasedAgent
from app.models.config import SimulationConfig
from app.models.enums import (
    Action,
    AgentFailureType,
    ChaosFailure,
    DecisionSource,
    EventSeverity,
    LearningEpisodeSource,
    PolicyName,
    TerminationReason,
    TrainingDifficulty,
    TrialOutcome,
    parse_action,
)
from app.models.schemas import (
    ConfigUpdate,
    PolicyDescription,
    PoliciesResponse,
    StateResponse,
)
from app.services.failure_analysis import detect_step_failures
from app.simulation.engine import BURNING, WIND_TO_CODE, FireSimulation


ACTION_DESCRIPTIONS = {
    Action.HOLD: "Use no response resource this step.",
    Action.PROTECT_NORTH: "Apply one abstract response resource to the north sector.",
    Action.PROTECT_SOUTH: "Apply one abstract response resource to the south sector.",
    Action.PROTECT_EAST: "Apply one abstract response resource to the east sector.",
    Action.PROTECT_WEST: "Apply one abstract response resource to the west sector.",
    Action.RECON: "Use one resource to temporarily improve the sensor observation.",
}

ACTION_LABELS = {
    Action.HOLD: "Hold",
    Action.PROTECT_NORTH: "Protect north",
    Action.PROTECT_SOUTH: "Protect south",
    Action.PROTECT_EAST: "Protect east",
    Action.PROTECT_WEST: "Protect west",
    Action.RECON: "Recon",
}

# Transparent teaching progression: XP is scoped to this process's one
# in-memory SimulationSession. It is not a global account or durable score.
XP_BY_OUTCOME = {
    TrialOutcome.SUCCESS: 100,
    TrialOutcome.PARTIAL: 50,
    TrialOutcome.NEEDS_IMPROVEMENT: 20,
}
XP_PER_LEVEL = 200

FAILURE_LABELS = {
    AgentFailureType.PERCEPTION_GAP: "an unseen fire front",
    AgentFailureType.MISDIRECTED_PROTECTION: "protecting the wrong sector",
    AgentFailureType.IDLE_WHILE_BURNING: "holding while fire burned",
    AgentFailureType.RESOURCE_EXHAUSTED_REQUEST: "acting with no resources left",
    AgentFailureType.WIND_MISREAD: "acting on a wrong wind reading",
}

# Keep the interactive state payload bounded; full history stays in run events.
FAILURE_STATE_LIMIT = 25

# Domain-randomization presets for headless training. Harder presets enable the
# engine's adversarial features and sample lower sensor reliability per episode.
DIFFICULTY_OVERRIDES: dict[TrainingDifficulty, dict[str, Any]] = {
    TrainingDifficulty.STANDARD: {},
    TrainingDifficulty.HARD: {
        "ember_jump_probability": 0.02,
        "wind_drift_probability": 0.05,
        "sensor_dropout_probability": 0.05,
    },
    TrainingDifficulty.BRUTAL: {
        "ember_jump_probability": 0.05,
        "wind_drift_probability": 0.12,
        "sensor_dropout_probability": 0.10,
        "observation_delay": 1,
    },
}
DIFFICULTY_RELIABILITY_RANGE: dict[TrainingDifficulty, tuple[float, float] | None] = {
    TrainingDifficulty.STANDARD: None,
    TrainingDifficulty.HARD: (0.5, 1.0),
    TrainingDifficulty.BRUTAL: (0.25, 0.9),
}


def evaluate_outcome(
    engine: FireSimulation,
) -> tuple[TerminationReason, bool, bool, TrialOutcome]:
    """Apply the documented success rubric to a finished engine episode."""

    termination_reason = TerminationReason(engine.termination_reason)
    metrics = engine.metrics()
    fire_extinguished = termination_reason is TerminationReason.FIRE_EXTINGUISHED
    all_sites_preserved = metrics["infrastructure_preserved"] == metrics["infrastructure_total"]
    mission_completed = fire_extinguished and all_sites_preserved
    if mission_completed:
        outcome = TrialOutcome.SUCCESS
    elif fire_extinguished:
        outcome = TrialOutcome.PARTIAL
    else:
        outcome = TrialOutcome.NEEDS_IMPROVEMENT
    return termination_reason, fire_extinguished, mission_completed, outcome


class SimulationSession:
    """Serialize all mutations while allowing safe use by FastAPI worker threads."""

    def __init__(self, config: SimulationConfig | None = None, seed: int = 42) -> None:
        self._lock = RLock()
        self.config = config or SimulationConfig()
        self.seed = seed
        self.policy = PolicyName.RULE
        self.engine = FireSimulation(self.config, seed=self.seed)
        # The learning agent survives resets and config changes on purpose:
        # its whole point is to keep improving across trials in this session.
        self._learning_agent = QLearningAgent(seed=self.seed + 7)
        self._build_agents()
        self._trials_started = 0
        self._trials_completed = 0
        self._successes = 0
        self._partials = 0
        self._needs_improvement = 0
        self._total_experience_points = 0
        self._completed_reward_total = 0.0
        self._start_trial()

    def _build_agents(self) -> None:
        self.agents = {
            PolicyName.RANDOM: RandomAgent(seed=self.seed + 1),
            PolicyName.RULE: RuleBasedAgent(),
            PolicyName.LEARNING: self._learning_agent,
            PolicyName.PPO_NOMINAL: PPOAgent(
                model_path=os.getenv("FIREOPS_PPO_NOMINAL_MODEL") or None,
                fallback=RuleBasedAgent(),
            ),
            PolicyName.PPO_ROBUST: PPOAgent(
                model_path=os.getenv("FIREOPS_PPO_ROBUST_MODEL") or None,
                fallback=RuleBasedAgent(),
            ),
        }

    def _start_trial(self) -> None:
        """Start fresh per-trial state without erasing server-session totals.

        Resetting or changing configuration abandons an unfinished trial: it
        increases ``trials_started`` but not ``trials_completed``. Only an
        actual simulator termination creates a run summary and outcome.
        """

        self._trials_started += 1
        self._trial_id = f"trial_{uuid4().hex}"
        self._trial_action_counts = {action: 0 for action in Action}
        self._trial_finalized = False
        self._agent_decision: dict[str, Any] | None = None
        self._run_summary: dict[str, Any] | None = None
        self._trial_failures: list[dict[str, Any]] = []
        self._trial_learning_steps = 0

    def _trial_stats(self) -> dict[str, int | float]:
        average_reward = (
            self._completed_reward_total / self._trials_completed
            if self._trials_completed
            else 0.0
        )
        return {
            "scope": "server_session",
            "trials_started": self._trials_started,
            "trials_completed": self._trials_completed,
            "successes": self._successes,
            "partials": self._partials,
            "needs_improvement": self._needs_improvement,
            "total_experience_points": self._total_experience_points,
            "current_level": 1 + self._total_experience_points // XP_PER_LEVEL,
            "average_completed_reward": round(float(average_reward), 4),
        }

    def _action_counts(self) -> dict[str, int]:
        return {action.name.lower(): self._trial_action_counts[action] for action in Action}

    def _state_response(self) -> StateResponse:
        payload = self.engine.state_dict(policy=self.policy.value)
        payload.update(
            agent_decision=self._agent_decision,
            trial_stats=self._trial_stats(),
            run_summary=self._run_summary,
            agent_failures=self._trial_failures[-FAILURE_STATE_LIMIT:],
            learning=self._learning_agent.stats(),
        )
        return StateResponse.model_validate(payload)

    @staticmethod
    def _rule_reason(action: Action, visible_fire_count: int, resources_before: int) -> str:
        if resources_before <= 0:
            return (
                "The rule policy saw that no response resources were left, so it chose Hold. "
                "It used only the visible map and did not learn online."
            )
        if visible_fire_count == 0:
            return (
                "The rule policy saw no visible burning cells, so it chose Recon to improve the next "
                "sensor view. It did not use hidden fire values or learn online."
            )
        direction = ACTION_LABELS[action].removeprefix("Protect ")
        return (
            f"The rule policy saw {visible_fire_count} visible burning cell(s). Its simple map score "
            f"was highest on the {direction} side, so it chose {ACTION_LABELS[action]}. "
            "It did not use hidden fire values or learn online."
        )

    def _learning_reason(self, action: Action, explored: bool | None) -> str:
        label = ACTION_LABELS[action]
        episodes = self._learning_agent.episodes_trained
        epsilon = self._learning_agent.epsilon
        if explored:
            return (
                f"The learning policy explored: with exploration rate {epsilon:.2f} it tried "
                f"{label} instead of its current best-valued move, so it can discover better "
                "strategies. Its value table updates from the reward after every step."
            )
        return (
            f"The learning policy chose {label} because that move has the highest learned value "
            f"for its current observation summary after {episodes} completed episode(s) "
            f"(exploration rate {epsilon:.2f}). It learns only from rewards and never reads "
            "hidden fire values."
        )

    def _decision_reason(
        self,
        *,
        action: Action,
        source: DecisionSource,
        policy: PolicyName,
        visible_fire_count: int,
        resources_before: int,
        ppo_used_rule_fallback: bool,
        explored: bool | None = None,
    ) -> str:
        label = ACTION_LABELS[action]
        if source is DecisionSource.MANUAL:
            if policy is PolicyName.LEARNING:
                return (
                    f"A person chose {label}. The simulator used that exact action, and the "
                    "learning policy still updated its value table from the observed reward "
                    "(off-policy learning), so manual play teaches it too."
                )
            return (
                f"A person chose {label}. The simulator used that exact action; no policy chose it "
                "and no online learning happened."
            )
        if policy is PolicyName.LEARNING:
            return self._learning_reason(action, explored)
        if policy is PolicyName.RANDOM:
            return (
                f"The random policy picked {label} by chance from the six allowed actions. It did not "
                "inspect hidden fire values or learn during this trial."
            )
        if policy is PolicyName.RULE:
            return self._rule_reason(action, visible_fire_count, resources_before)
        if ppo_used_rule_fallback:
            return (
                "The saved PPO model was unavailable or did not match this simulation, so the rule "
                "fallback was used. "
                + self._rule_reason(action, visible_fire_count, resources_before)
            )
        return (
            f"The saved PPO model chose {label} from the observation it was given. Its inner scoring "
            "is opaque, so this API does not invent a private reason. The model did not receive hidden "
            "fire values and does not learn during this trial."
        )

    def _finalize_trial(self) -> None:
        """Apply the synthetic teaching rubric once for a completed trial.

        Success means the fire was extinguished and all three synthetic sites
        were preserved. Extinguished fire with at least one site loss is a
        partial outcome. Reaching ``max_steps`` is ``needs_improvement``.
        """

        if not self.engine.done or self._trial_finalized:
            return

        metrics = self.engine.metrics()
        termination_reason, fire_extinguished, mission_completed, outcome = evaluate_outcome(
            self.engine
        )
        if outcome is TrialOutcome.SUCCESS:
            title = "Mission complete: Every site was protected"
        elif outcome is TrialOutcome.PARTIAL:
            title = "Fire out: Some sites were affected"
        else:
            title = "Time ran out: Fire is still active"

        if self._trial_learning_steps > 0:
            self._learning_agent.finish_episode(
                float(metrics["cumulative_reward"]), outcome, LearningEpisodeSource.INTERACTIVE
            )
            self.engine.record_event(
                "learning",
                (
                    f"Learning policy recorded episode {self._learning_agent.episodes_trained}: "
                    f"reward {metrics['cumulative_reward']:.2f}; exploration rate is now "
                    f"{self._learning_agent.epsilon:.2f}."
                ),
                EventSeverity.INFO,
            )

        xp_earned = XP_BY_OUTCOME[outcome]
        self._trials_completed += 1
        if outcome is TrialOutcome.SUCCESS:
            self._successes += 1
        elif outcome is TrialOutcome.PARTIAL:
            self._partials += 1
        else:
            self._needs_improvement += 1
        self._total_experience_points += xp_earned
        self._completed_reward_total += float(metrics["cumulative_reward"])

        what_happened = [
            (
                "The active fire was out before the step limit."
                if fire_extinguished
                else "The step limit arrived while fire was still active."
            ),
            (
                f"{metrics['infrastructure_preserved']} of {metrics['infrastructure_total']} synthetic "
                "critical sites were preserved."
            ),
            (
                f"Fire affected {metrics['burned_area']} of {self.config.grid_size**2} map cells. "
                f"Response resources used: {metrics['resources_consumed']}."
            ),
        ]
        reward_lesson = (
            f"The total reward was {metrics['cumulative_reward']:.2f}. New spread, damaged sites, "
            "used resources, and active fire subtract points in this simulator, so a higher total is better."
        )
        if outcome is TrialOutcome.SUCCESS:
            improvements = [
                "Try to keep every site safe again while using fewer resources and limiting fire spread."
            ]
        elif outcome is TrialOutcome.PARTIAL:
            improvements = [
                "Protect the side near synthetic critical sites earlier when the visible map shows danger.",
                "Use Recon when many cells are hidden, then act on the clearer next sensor view.",
            ]
        else:
            improvements = [
                "Contain the visible fire before the step limit arrives.",
                "Use Recon when the map is too hidden, and save some resources for later steps.",
            ]

        failure_counts = {failure_type: 0 for failure_type in AgentFailureType}
        for failure in self._trial_failures:
            failure_counts[failure["failure_type"]] += 1
        top_failure = max(failure_counts, key=failure_counts.get)
        if failure_counts[top_failure] > 0:
            improvements.append(
                f"Failure analysis: {FAILURE_LABELS[top_failure]} was flagged "
                f"{failure_counts[top_failure]} time(s) this trial; the flagged steps are in "
                "the event log."
            )

        self._run_summary = {
            "trial_id": self._trial_id,
            "fire_extinguished": fire_extinguished,
            "mission_completed": mission_completed,
            "success": outcome is TrialOutcome.SUCCESS,
            "outcome": outcome,
            "title": title,
            "what_happened": what_happened,
            "reward_lesson": reward_lesson,
            "improvements": improvements,
            "xp_earned": xp_earned,
            "action_counts": self._action_counts(),
            "termination_reason": termination_reason,
            "final_metrics": metrics,
            "failure_counts": failure_counts,
        }
        self._trial_finalized = True

    def state(self) -> StateResponse:
        with self._lock:
            return self._state_response()

    def reset(
        self,
        *,
        seed: int | None = None,
        policy: PolicyName | None = None,
        config_update: ConfigUpdate | None = None,
        sensor_reliability: float | None = None,
    ) -> StateResponse:
        with self._lock:
            updates = config_update.defined_values() if config_update else {}
            if sensor_reliability is not None:
                updates["sensor_reliability"] = sensor_reliability
            if updates:
                self.config = SimulationConfig.model_validate({**self.config.model_dump(), **updates})
            if seed is not None:
                self.seed = int(seed)
            requested_policy = policy or (config_update.policy if config_update else None)
            if requested_policy is not None:
                self.policy = requested_policy
            self.engine = FireSimulation(self.config, seed=self.seed)
            self._build_agents()
            self._start_trial()
            return self._state_response()

    def step(self, *, action: Action | int | str | None = None, policy: PolicyName | None = None) -> StateResponse:
        with self._lock:
            if self.engine.done:
                raise RuntimeError("Episode is complete; reset before stepping again")
            selected_policy = policy or self.policy
            observation = self.engine.gym_observation()
            visible_fire_count = int((observation["fire"] == BURNING).sum())
            hidden_cell_count = int((observation["observed_mask"] == 0).sum())
            resources_before = int(observation["resources"].reshape(-1)[0])
            # Decision-time truth snapshot: analysis-only, never given to agents.
            decision_step = self.engine.step_count
            true_fire_before = self.engine.fire_state.copy()
            true_wind_code_before = WIND_TO_CODE[self.engine.wind_direction]
            observed_wind_before = int(observation["wind"].reshape(-1)[0])
            source = DecisionSource.MANUAL if action is not None else DecisionSource.POLICY
            ppo_used_rule_fallback = False
            explored: bool | None = None
            is_learning_policy = selected_policy is PolicyName.LEARNING
            learning_state = self._learning_agent.encode(observation) if is_learning_policy else None
            if action is not None:
                selected_action = parse_action(action)
            else:
                agent = self.agents[selected_policy]
                ppo_was_native = isinstance(agent, PPOAgent) and agent.native_available
                selected_action = Action(agent.act(observation))
                if isinstance(agent, PPOAgent):
                    ppo_used_rule_fallback = not (ppo_was_native and agent.native_available)
                if is_learning_policy:
                    explored = self._learning_agent.last_explored
            self.engine.step(selected_action)
            if is_learning_policy and learning_state is not None:
                next_observation = self.engine.gym_observation()
                self._learning_agent.learn(
                    learning_state,
                    selected_action,
                    self.engine.last_reward,
                    self._learning_agent.encode(next_observation),
                    self.engine.done,
                )
                self._trial_learning_steps += 1
            step_failures = detect_step_failures(
                step=decision_step,
                policy=selected_policy.value if source is DecisionSource.POLICY else "manual",
                action=selected_action,
                true_fire_before=true_fire_before,
                observed_fire_before=observation["fire"],
                observed_mask_before=observation["observed_mask"],
                observed_wind_before=observed_wind_before,
                true_wind_code=true_wind_code_before,
                resources_before=resources_before,
            )
            for failure in step_failures:
                self.engine.record_event(
                    "agent_failure", failure["message"], EventSeverity.WARNING
                )
            self._trial_failures.extend(step_failures)
            self.policy = selected_policy
            self._trial_action_counts[selected_action] += 1
            self._agent_decision = {
                "chosen_action": selected_action.name,
                "action_label": ACTION_LABELS[selected_action],
                "decision_source": source,
                "reason": self._decision_reason(
                    action=selected_action,
                    source=source,
                    policy=selected_policy,
                    visible_fire_count=visible_fire_count,
                    resources_before=resources_before,
                    ppo_used_rule_fallback=ppo_used_rule_fallback,
                    explored=explored,
                ),
                "visible_fire_count": visible_fire_count,
                "hidden_cell_count": hidden_cell_count,
                "resources_before_action": resources_before,
                "reward_change": self.engine.last_reward,
                "reward_breakdown": self.engine.last_reward_breakdown,
                "explored": explored,
                "exploration_rate": (
                    round(float(self._learning_agent.epsilon), 4) if is_learning_policy else None
                ),
                "q_values": (
                    self._learning_agent.last_q_values
                    if is_learning_policy and source is DecisionSource.POLICY
                    else None
                ),
            }
            self._finalize_trial()
            return self._state_response()

    def chaos(self, failure: ChaosFailure | None = None) -> StateResponse:
        with self._lock:
            self.engine.inject_chaos(failure)
            return self._state_response()

    def update_config(self, update: ConfigUpdate) -> StateResponse:
        with self._lock:
            values = update.defined_values()
            self.config = SimulationConfig.model_validate({**self.config.model_dump(), **values})
            if update.policy is not None:
                self.policy = update.policy
            # Configuration changes define a new comparable episode.
            self.engine = FireSimulation(self.config, seed=self.seed)
            self._build_agents()
            self._start_trial()
            return self._state_response()

    def policies(self) -> PoliciesResponse:
        with self._lock:
            nominal = self.agents[PolicyName.PPO_NOMINAL]
            robust = self.agents[PolicyName.PPO_ROBUST]
            policy_descriptions = [
                PolicyDescription(
                    id=PolicyName.RANDOM,
                    label="Random baseline",
                    description="Seeded uniformly random reference policy.",
                ),
                PolicyDescription(
                    id=PolicyName.RULE,
                    label="Rule-based",
                    description="Auditable heuristic using only visible fire cells and available resources.",
                ),
                PolicyDescription(
                    id=PolicyName.LEARNING,
                    label="Learning (Q-table)",
                    description=(
                        "Transparent tabular Q-learning that improves from rewards; "
                        f"{self._learning_agent.episodes_trained} episode(s) trained this "
                        "server session."
                    ),
                ),
                PolicyDescription(
                    id=PolicyName.PPO_NOMINAL,
                    label="PPO nominal",
                    description=(
                        "Loaded Stable-Baselines3 nominal model."
                        if nominal.native_available
                        else "Optional model is not loaded; currently uses the rule fallback."
                    ),
                    native_available=nominal.native_available,
                ),
                PolicyDescription(
                    id=PolicyName.PPO_ROBUST,
                    label="PPO robust",
                    description=(
                        "Loaded Stable-Baselines3 failure-randomized model."
                        if robust.native_available
                        else "Optional model is not loaded; currently uses the rule fallback."
                    ),
                    native_available=robust.native_available,
                ),
            ]
            return PoliciesResponse(
                policies=policy_descriptions,
                default_policy=PolicyName.RULE,
                current_policy=self.policy,
                actions=[
                    {"value": action.value, "name": action.name, "description": ACTION_DESCRIPTIONS[action]}
                    for action in Action
                ],
            )

    def learning_stats(self) -> dict[str, Any]:
        with self._lock:
            return self._learning_agent.stats()

    def train(
        self,
        *,
        episodes: int,
        seed: int | None = None,
        difficulty: TrainingDifficulty = TrainingDifficulty.STANDARD,
    ) -> dict[str, Any]:
        """Run real headless Q-learning episodes without touching the live trial.

        Every episode uses a fresh seeded ``FireSimulation`` built from the
        current configuration plus the chosen difficulty preset, so training
        worlds are reproducible and the interactive engine is left untouched.
        Harder presets enable ember jumps, wind drift, sensor dropout, delayed
        observations, and per-episode randomized sensor reliability.
        """

        with self._lock:
            agent = self._learning_agent
            if seed is not None:
                base_seed = int(seed)
            else:
                # Advance with training progress so repeated calls see new worlds.
                base_seed = self.seed + 100_000 + agent.episodes_trained * 1_009
            overrides = DIFFICULTY_OVERRIDES[difficulty]
            reliability_range = DIFFICULTY_RELIABILITY_RANGE[difficulty]
            randomization_rng = np.random.default_rng(base_seed)
            results: list[dict[str, Any]] = []
            for index in range(episodes):
                episode_seed = (base_seed + index) % 2**32
                episode_values = {**self.config.model_dump(), **overrides}
                if reliability_range is not None:
                    episode_values["sensor_reliability"] = round(
                        float(randomization_rng.uniform(*reliability_range)), 4
                    )
                episode_config = SimulationConfig.model_validate(episode_values)
                simulation = FireSimulation(episode_config, seed=episode_seed)
                observation = simulation.gym_observation()
                while not simulation.done:
                    state_key = agent.encode(observation)
                    step_action = Action(agent.act(observation))
                    simulation.step(step_action)
                    observation = simulation.gym_observation()
                    agent.learn(
                        state_key,
                        step_action,
                        simulation.last_reward,
                        agent.encode(observation),
                        simulation.done,
                    )
                _, _, _, outcome = evaluate_outcome(simulation)
                episode_reward = float(simulation.metrics()["cumulative_reward"])
                epsilon_used = round(float(agent.epsilon), 4)
                agent.finish_episode(episode_reward, outcome, LearningEpisodeSource.TRAINING)
                results.append(
                    {
                        "episode": agent.episodes_trained,
                        "seed": episode_seed,
                        "steps": simulation.step_count,
                        "reward": round(episode_reward, 4),
                        "outcome": outcome,
                        "epsilon": epsilon_used,
                        "sensor_reliability": episode_config.sensor_reliability,
                    }
                )
            self.engine.record_event(
                "training",
                (
                    f"Headless training ran {episodes} {difficulty.value} episode(s); the "
                    f"learning policy has now trained on {agent.episodes_trained} episode(s) "
                    f"and its exploration rate is {agent.epsilon:.2f}."
                ),
                EventSeverity.INFO,
            )
            return {
                "episodes_run": episodes,
                "difficulty": difficulty,
                "results": results,
                "stats": agent.stats(),
                "note": (
                    "Training used fresh seeded synthetic worlds built from the current "
                    f"configuration with the '{difficulty.value}' preset; the interactive "
                    "trial was not modified."
                ),
            }

    def snapshot_for_test(self) -> dict[str, Any]:
        """Return copied internal arrays for concurrency and determinism tests."""

        with self._lock:
            return {
                "fire": self.engine.fire_state.copy(),
                "resources": self.engine.resources_available,
                "step": self.engine.step_count,
            }
