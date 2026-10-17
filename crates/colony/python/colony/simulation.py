"""Closed-loop (observe -> plan -> step) episode executor and simple safety gate.

Not SLAM, robot dynamics, or a certifiable safety controller. The simulated
movement oracle rejects collisions; real hardware needs independent safeguards.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Callable
from .planner import choose_target, shortest_path
from .world import World, FREE, BLOCKED, UNKNOWN, Point, observe, reachable


@dataclass(frozen=True)
class Telemetry:
    localization_confidence: float = 1.0
    sensor_healthy: bool = True
    anomaly: bool = False

    def may_move(self) -> bool:
        return (self.sensor_healthy and not self.anomaly
                and 0.0 <= self.localization_confidence <= 1.0
                and self.localization_confidence >= 0.70)


@dataclass(frozen=True)
class EpisodeResult:
    policy: str
    seed: int
    reachable_cells: int
    observed_reachable: int
    visited_reachable: int
    coverage: float
    steps: int
    decisions: int
    growth_nodes: int
    collisions: int
    stopped_for_safety: bool
    completed: bool
    exhausted_budget: bool

    def as_dict(self) -> dict:
        return asdict(self)


def run_episode(world: World, policy: str, *, seed: int = 0,
                sensor_radius: int = 3, max_steps: int = 3_000,
                telemetry: Callable[[int, Point], Telemetry] | None = None,
                trace: list[Point] | None = None,
                growth_parameters: dict | None = None) -> EpisodeResult:
    """Explore an unknown map with fixed range-limited obstacle-occluded sensing.

    The evaluation oracle knows true reachable cells, but planning only gets
    a ternary observed map. Each next motion is guarded before execution.
    """
    if sensor_radius < 1 or max_steps < 1:
        raise ValueError("sensor_radius and max_steps must be positive")
    topology = world.topology()
    known = [UNKNOWN] * len(world.cells)
    position = world.start
    visited = {position}
    if trace is not None:
        trace.clear()
        trace.append(position)
    truth_reachable = reachable(world)
    steps = decisions = growth_nodes = collisions = 0
    stopped = False
    observe(world, known, position, sensor_radius)

    while steps < max_steps:
        choice = choose_target(topology, known, position, policy,
                               **(growth_parameters or {}))
        decisions += 1
        growth_nodes += choice.growth_nodes
        if choice.target is None:
            break
        route = shortest_path(topology, known, position, choice.target)
        if route is None or len(route) < 2:
            # A valid frontier candidate must be reachable and not current.
            raise AssertionError("planner selected a non-traversable target")
        for next_pos in route[1:]:
            if steps >= max_steps:
                break
            snapshot = telemetry(steps, position) if telemetry else Telemetry()
            if not snapshot.may_move():
                stopped = True
                break
            if known[world.idx(next_pos)] != FREE:
                raise AssertionError("planner attempted to move into an unknown/blocked cell")
            # Oracle check measures missed collisions in the idealized model.
            if world.tile(next_pos) == BLOCKED:
                collisions += 1
                stopped = True
                break
            position = next_pos
            visited.add(position)
            if trace is not None:
                trace.append(position)
            steps += 1
            observe(world, known, position, sensor_radius)
        if stopped:
            break

    observed_reachable = sum(known[world.idx(p)] == FREE for p in truth_reachable)
    visited_reachable = len(visited.intersection(truth_reachable))
    return EpisodeResult(
        policy=policy, seed=seed, reachable_cells=len(truth_reachable),
        observed_reachable=observed_reachable, visited_reachable=visited_reachable,
        coverage=round(observed_reachable / len(truth_reachable), 6),
        steps=steps, decisions=decisions, growth_nodes=growth_nodes,
        collisions=collisions, stopped_for_safety=stopped,
        completed=observed_reachable == len(truth_reachable),
        exhausted_budget=steps >= max_steps and observed_reachable < len(truth_reachable),
    )
