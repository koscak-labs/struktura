"""Baseline frontier and SCA-inspired growth target selectors.

Both planners receive ONLY the robot's current partially observed map.
All motion is executed via the SAME shortest-path-on-known-free grid planner.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from math import hypot
from .world import UNKNOWN, FREE, Topology, Point


def shortest_path(shape: Topology, known: list[int], source: Point, goal: Point) -> list[Point] | None:
    if known[shape.idx(source)] != FREE or known[shape.idx(goal)] != FREE:
        return None
    queue = deque([source])
    predecessor = {source: None}
    while queue:
        p = queue.popleft()
        if p == goal:
            result = []
            while p is not None:
                result.append(p)
                p = predecessor[p]
            return list(reversed(result))
        for q in shape.neighbors(p):
            if q not in predecessor and known[shape.idx(q)] == FREE:
                predecessor[q] = p
                queue.append(q)
    return None


def distance_map(shape: Topology, known: list[int], source: Point):
    queue = deque([source])
    distances = {source: 0}
    while queue:
        p = queue.popleft()
        for q in shape.neighbors(p):
            if q not in distances and known[shape.idx(q)] == FREE:
                distances[q] = distances[p] + 1
                queue.append(q)
    return distances


def frontiers(shape: Topology, known: list[int]) -> list[Point]:
    """Observed FREE cells bordering at least one UNKNOWN cell."""
    return [(x, y) for y in range(shape.height) for x in range(shape.width)
            if known[shape.idx((x, y))] == FREE
            and any(known[shape.idx(q)] == UNKNOWN for q in shape.neighbors((x, y)))]


def gain(shape: Topology, known: list[int], p: Point) -> int:
    return sum(known[shape.idx(q)] == UNKNOWN for q in shape.neighbors(p))


@dataclass(frozen=True)
class Decision:
    target: Point | None
    growth_nodes: int = 0
    frontier_count: int = 0


def choose_target(shape: Topology, known: list[int], position: Point,
                  policy: str, influence_radius: int = 6,
                  growth_budget: int = 60) -> Decision:
    """Decide target from observed free cells only; never inspect shape.cells.

    SCA is adapted from continuous attraction-point growth to an integer grid:
    at each iteration, frontier points steer the nearest existing branch tip.
    Proposed edges are accepted only through already OBSERVED free cells.
    This is a *candidate-selection heuristic*, not an optimal path planner.
    """
    if policy not in ("frontier", "growth"):
        raise ValueError(f"unknown policy {policy}")
    targets = frontiers(shape, known)
    distances = distance_map(shape, known, position)
    targets = [p for p in targets if p in distances and p != position]
    if not targets:
        return Decision(None)
    # Tie breakers deliberately deterministic; no hidden RNG in policies.
    nearest = min(targets, key=lambda p: (distances[p], -gain(shape, known, p), p[1], p[0]))
    if policy == "frontier":
        return Decision(nearest, 0, len(targets))
    if influence_radius < 1 or growth_budget < 1:
        raise ValueError("positive influence_radius and growth_budget required")

    nodes = [position]
    existing = {position}
    radius2 = influence_radius * influence_radius
    for _ in range(growth_budget):
        directions: dict[Point, tuple[float, float]] = {}
        for a in targets:
            closest = min(nodes, key=lambda n: (n[0]-a[0])**2 + (n[1]-a[1])**2)
            dx, dy = a[0]-closest[0], a[1]-closest[1]
            d2 = dx*dx + dy*dy
            if not 0 < d2 <= radius2:
                continue
            d = hypot(dx, dy)
            weight = 1.0 + 0.35 * gain(shape, known, a)
            prev = directions.get(closest, (0.0, 0.0))
            directions[closest] = (prev[0] + weight*dx/d, prev[1] + weight*dy/d)
        extensions: list[Point] = []
        for tip in nodes:  # only old nodes act in this iteration
            if tip not in directions:
                continue
            dx, dy = directions[tip]
            candidates = [q for q in shape.neighbors(tip)
                          if known[shape.idx(q)] == FREE and q not in existing]
            if not candidates:
                continue
            def alignment(q):
                return (q[0]-tip[0])*dx + (q[1]-tip[1])*dy
            nxt = max(candidates, key=lambda q: (alignment(q), -q[1], -q[0]))
            if alignment(nxt) > 1e-9:
                existing.add(nxt)
                extensions.append(nxt)
        if not extensions:
            break
        nodes.extend(extensions)
        if len(nodes) >= shape.width*shape.height:
            break

    reached = [p for p in targets if p in existing]
    if not reached:
        return Decision(nearest, len(nodes), len(targets))

    # Score is a hypothesis, not an empirically verified improvement.
    # The +1 prevents favoring a nearly zero-distance cell infinitely.
    choice = max(reached, key=lambda p: ((1.0 + gain(shape, known, p)) /
                                         (2.0 + distances[p]),
                                         -distances[p], -p[1], -p[0]))
    return Decision(choice, len(nodes), len(targets))
