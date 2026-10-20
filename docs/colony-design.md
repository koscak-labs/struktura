# COLONY — architecture decision record v0.2

## Decision

Keep COLONY in a separate experimental Rust crate under `crates/colony/`.
Struktura is a telemetry detector; exploration/planning has different safety,
performance, integration and benchmark requirements. The experimental crate
must not become an undocumented production feature of Struktura.

## Data flow

```
world truth -> range sensor -> ternary occupancy map
                                   |
                                   v
                           observed frontiers
                            /            \
                    baseline BFS      SCA growth tree
                            \            /
                             candidate waypoint
                                   |
                          BFS on observed FREE
                                   |
                        simulated telemetry gate
                                   |
                          one step + reobserve
```

The reachability oracle is used *only* by evaluation, not by navigation.
The candidate-selector API accepts dimensions / observed map only in the next
iteration, to enforce information isolation by type instead of by convention.

## Known problems

- Planning selectors receive a topology-only type and observed occupancy.
  Ground truth is absent from their argument types.
- Naive attractor-to-tip association O(A*N) per growth iteration; the growth
  tree is rebuilt each target decision.
- Simulated motion checks truth before executing, i.e. idealized safety oracle.
- Incomplete or invalid telemetry is not calibrated from real robot readings.
- No orientation, time-dependent occupancy, robot footprint or kinodynamic
  feasibility; only four-connected point-agent navigation.
- Python and Rust map generators use different seeded RNG algorithms, so their
  measurements are **not paired across languages**. Compare policies within
  each language only, and do not assert bit-for-bit numerical parity.

## Performance experiment

Primary: per-seed paired distance on identical maps and sensor model.
Secondary: complete coverage fraction, success rate, decisions, CPU cost,
regressions caused by dead ends, sensitivity to influence radius.
Report mean, median, standard deviation, bootstrap confidence intervals,
problem seeds and pre-registered parameter searches on held-out maps.

## Target architecture

- Map builder: ROS 2 / SLAM adapter producing occupancy + confidence.
- Exploration core: map-neutral Rust trait `ObservationMap` + incremental
  growth structure; target hypotheses are never commands to the robot.
- Motion planner: actual G1 kinematics, balance, collision margins, speed limits.
- Supervisor: independent safety controls and monitored health events.
- Telemetry: optional Struktura adapter mapping events to safe response policy.
- Benchmarks: reproducible independent unseen seeds / sensor failures.

## Near-term next experiment

Establish a growth-aware path utility using *expected sensor visibility gain*
over known free space, compared with an equal-compute-budget frontier-based
information-gain planner. The current baseline uses frontier adjacency gain
as a weak proxy; as of 100 paired seeds growth costs +3.8% travel (100% coverage for both).
Seven of those seeds have tiny reachable components (1–7 free cells).
