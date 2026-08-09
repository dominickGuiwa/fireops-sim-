from __future__ import annotations

import numpy as np
import pytest

from app.models.config import SimulationConfig
from app.models.enums import Action, ChaosFailure, WindDirection
from app.simulation.engine import BURNED, BURNING, UNKNOWN, UNBURNED, FireSimulation


def config(**updates: object) -> SimulationConfig:
    defaults: dict[str, object] = {
        "grid_size": 5,
        "ignition_count": 1,
        "base_spread_probability": 0.35,
        "wind_direction": WindDirection.EAST,
        "wind_strength": 0.8,
        "moisture": 0.0,
        "sensor_reliability": 1.0,
        "observation_noise": 0.0,
        "burn_duration": 10,
        "max_steps": 30,
    }
    defaults.update(updates)
    return SimulationConfig(**defaults)


def controlled_fire(engine: FireSimulation, *, burning: list[tuple[int, int]], burned: list[tuple[int, int]] = []) -> None:
    engine.fire_state.fill(UNBURNED)
    engine.burn_age.fill(0)
    for location in burning:
        engine.fire_state[location] = BURNING
    for location in burned:
        engine.fire_state[location] = BURNED
    engine.fuel.fill(1.0)
    engine.cell_moisture.fill(0.0)


def test_neighbours_and_spread_are_bounds_safe() -> None:
    engine = FireSimulation(config(base_spread_probability=1.0), seed=7)
    controlled_fire(engine, burning=[(0, 0)])

    assert set(engine.neighbours(0, 0)) == {(0, 1), (1, 0)}
    engine.step(Action.HOLD)

    assert engine.fire_state.shape == (5, 5)
    assert np.count_nonzero(engine.fire_state == BURNING) <= 3


def test_burned_cells_never_reignite() -> None:
    engine = FireSimulation(config(base_spread_probability=1.0), seed=11)
    controlled_fire(engine, burning=[(2, 1)], burned=[(2, 2)])

    for _ in range(3):
        engine.step(Action.HOLD)
        assert engine.fire_state[2, 2] == BURNED


def test_wind_increases_downwind_and_reduces_upwind_spread() -> None:
    engine = FireSimulation(config(base_spread_probability=0.2, wind_strength=1.0), seed=1)
    controlled_fire(engine, burning=[(2, 2)])

    east = engine.spread_probability((2, 2), (2, 3))
    crosswind = engine.spread_probability((2, 2), (1, 2))
    west = engine.spread_probability((2, 2), (2, 1))

    assert east > crosswind > west
    assert east == pytest.approx(0.5)
    assert west == pytest.approx(0.05)


def test_moisture_fuel_and_protection_reduce_probability() -> None:
    engine = FireSimulation(config(base_spread_probability=0.4, initial_resources=1), seed=2)
    controlled_fire(engine, burning=[(2, 2)])
    target = (2, 3)
    dry = engine.spread_probability((2, 2), target)

    engine.cell_moisture[target] = 0.5
    moist = engine.spread_probability((2, 2), target)
    engine.cell_moisture[target] = 0.0
    engine.fuel[target] = 0.2
    low_fuel = engine.spread_probability((2, 2), target)
    engine.fuel[target] = 1.0
    engine._apply_action(Action.PROTECT_EAST)
    protected = engine.spread_probability((2, 2), target)

    assert dry > moist
    assert dry > low_fuel
    assert dry > protected


def test_reward_does_not_pay_a_per_step_survival_bonus() -> None:
    engine = FireSimulation(config(base_spread_probability=0.0), seed=12)
    controlled_fire(engine, burning=[(2, 2)])

    _, reward, _, _ = engine.step(Action.HOLD)

    assert reward < 0.0


def test_sensor_degradation_reduces_information_without_changing_world() -> None:
    high = FireSimulation(config(sensor_reliability=1.0), seed=33)
    low = FireSimulation(config(sensor_reliability=0.25), seed=33)

    assert np.array_equal(high.fire_state, low.fire_state)
    assert high.observation.observed_mask.mean() == 1.0
    assert high.gym_observation()["wind"][0] == 2
    assert low.observation.observed_mask.mean() < 0.60
    assert low.observation.observed_mask.mean() < high.observation.observed_mask.mean()

    low.inject_chaos(ChaosFailure.SENSOR_OFFLINE)
    assert low.observation.effective_reliability == 0.0
    assert not low.observation.observed_mask.any()
    assert low.gym_observation()["wind"][0] == UNKNOWN


def test_cached_state_reads_do_not_change_dynamics() -> None:
    first = FireSimulation(config(), seed=91)
    second = FireSimulation(config(), seed=91)
    for _ in range(8):
        first.state_dict()
    first.step(Action.HOLD)
    second.step(Action.HOLD)

    assert np.array_equal(first.fire_state, second.fire_state)
    assert first.metrics() == second.metrics()


def test_sensor_chaos_does_not_perturb_physical_dynamics_rng() -> None:
    baseline = FireSimulation(config(), seed=92)
    degraded = FireSimulation(config(), seed=92)
    degraded.inject_chaos(ChaosFailure.SENSOR_DEGRADED)

    for action in (Action.HOLD, Action.PROTECT_EAST, Action.HOLD, Action.HOLD):
        baseline.step(action)
        degraded.step(action)
        assert np.array_equal(baseline.fire_state, degraded.fire_state)


def test_resources_never_become_negative_and_recon_improves_sensor() -> None:
    engine = FireSimulation(config(initial_resources=2, sensor_reliability=0.25), seed=4)
    engine.step(Action.RECON)
    assert engine.resources_available == 1
    assert engine.observation.effective_reliability >= 0.95

    engine.step(Action.PROTECT_NORTH)
    for _ in range(5):
        if engine.done:
            break
        engine.step(Action.PROTECT_SOUTH)
    assert engine.resources_available == 0
    assert engine.resources_consumed == 2
    assert engine.metrics()["resources_consumed"] == 2


def test_infrastructure_loss_reward_and_event_are_recorded_once() -> None:
    engine = FireSimulation(config(), seed=8)
    location = next(iter(engine.infrastructure_locations))
    controlled_fire(engine, burning=[location])

    _, reward, _, _ = engine.step(Action.HOLD)
    assert reward < -20.0
    assert engine.metrics()["infrastructure_lost"] == 1
    assert engine.metrics()["infrastructure_preserved"] == 2
    assert [event["type"] for event in engine.events].count("infrastructure_loss") == 1

    engine.step(Action.HOLD)
    assert engine.metrics()["infrastructure_lost"] == 1
    assert [event["type"] for event in engine.events].count("infrastructure_loss") == 1


@pytest.mark.parametrize("failure", list(ChaosFailure))
def test_every_chaos_failure_is_seeded_safe_and_logged(failure: ChaosFailure) -> None:
    engine = FireSimulation(config(initial_resources=3), seed=55)
    before_resources = engine.resources_available
    before_wind = engine.wind_direction

    actual = engine.inject_chaos(failure)

    assert actual is failure
    assert engine.resources_available >= 0
    assert engine.events[-1]["type"] == "chaos"
    if failure == ChaosFailure.RESOURCE_UNAVAILABLE:
        assert engine.resources_available < before_resources
    if failure == ChaosFailure.WIND_CHANGE:
        assert engine.wind_direction != before_wind


def test_same_seed_and_actions_produce_same_trajectory() -> None:
    first = FireSimulation(config(), seed=2026)
    second = FireSimulation(config(), seed=2026)
    actions = [Action.HOLD, Action.PROTECT_EAST, Action.RECON, Action.HOLD]

    for action in actions:
        result_a = first.step(action)
        result_b = second.step(action)
        assert result_a[1:] == result_b[1:]
        assert np.array_equal(first.fire_state, second.fire_state)
        assert np.array_equal(first.observation.fire, second.observation.fire)
