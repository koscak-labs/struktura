# COLONY v0.2 — partially observed exploration research

**Status:** simulated research prototype. Not a Unitree G1 controller, SLAM
system, safety-certified motion planner, or demonstrated space-robot product.

COLONY tests the hypothesis that an attraction-point (space-colonization)
growth tree can select efficient observation waypoints for exploration.

Both policies see **only a ternary map** (`unknown / observed free / observed
blocked`), updated by an idealized occlusion-aware range sensor. They execute
through the **same BFS shortest-path planner restricted to observed free cells**.
Ground-truth connectivity and obstacles are used by the simulator's sensor,
collision check, and evaluator, not for goal selection or route planning.

## Runnable Python reference (validated locally)

```bash
PYTHONPATH=crates/colony/python python3 -m unittest discover -s crates/colony/python/tests -v
PYTHONPATH=crates/colony/python python3 -m colony.benchmark --seeds 12 \
  --json benchmark.json --csv benchmark.csv
```

No Python dependencies. Fixed-seed paired episodes. Generate an inspection
visualization with:

```bash
PYTHONPATH=crates/colony/python python3 -m colony.visualize --seed 7 --out colony-trajectories.svg
```

Fixed-seed paired episodes. Both policies operate on
the same generated world. CSV per run, JSON summary and paired travel delta.

## Rust exploratory engine (requires Cargo toolchain)

```bash
cargo test --manifest-path crates/colony/Cargo.toml
cargo run --release --manifest-path crates/colony/Cargo.toml --bin colony-bench -- --seeds 30
```

Self-contained Rust library plus CLI, zero external crates. Intended first
step toward reusable ROS 2 or Unitree adapters. Rust source **has not been
compiled in the authoring environment**; GitHub CI is included for validation.

## Policies

- `frontier`: nearest observed free cell bordering unknown, by BFS distance.
- `growth`: observed frontier cells serve as attraction points. A bounded
  influence-radius growth tree expands only over observed free cells. Select
  a reached frontier by local unknown-neighbor information gain per distance.
  Fall back to nearest frontier if growth cannot reach a target.

Python and Rust selectors take a **Topology** (geometry only) plus observed
occupancy; ground-truth cells are unavailable to them by construction.

This is an SCA-inspired **waypoint selector**; it is not full continuous SCA,
RRT, risk-optimal planning, or a learned exploration policy.

## Exploratory evidence (Python reference, 100 paired seeds, 25x19, radius 3)

| Metric | Nearest frontier | Growth |
|---|---:|---:|
| Complete observed reachable coverage | 100/100 | 100/100 |
| Mean traversed steps | 156.71 | 162.68 |
| Mean planner decisions | 43.71 | 39.25 |
| Mean episode runtime (Python, this host) | 0.030 s | 0.053 s |
| Simulated collisions | 0 | 0 |

Growth is **worse on travel distance in these runs** (+5.97 steps, +3.8%;
paired percentile-bootstrap 95% CI +2.51 to +9.47 over these 100 seeds).
This is not a win and must not be advertised as one. Seven seeds produced
small reachable components (1–7 free cells) because the start was isolated;
for the remaining 93 maps, mean travel steps are 168.44 (frontier) versus
174.85 (growth). Seed filtering was examined post hoc, not pre-registered.
Terrain is still simple,
static, axis-aligned, and noise-free. There are no dynamics, slip, localization
errors, rolling maps, depth camera, or physical collision checks.

## Integration boundary

COLONY consumes a topology/occupancy map and structured telemetry health.
`Telemetry::may_move` rejects low localization confidence, unhealthy sensors,
and anomalous status. This is a **simulated supervisory hook**, NOT currently
connected to Struktura's Guard or to a Unitree G1. A future Struktura adapter
should convert calibrated anomaly events into a conservative health state,
with hysteresis and explicit recovery conditions. Do not treat a statistical
anomaly score as a validated real-time emergency stop.

## Research gates before hardware

1. Test >=100 unseen paired maps, corridors and dynamic-obstacle cases.
2. Report coverage, travel, wall-clock CPU time, peak memory and failure modes.
3. Improve candidate growth speed using spatial index + incremental graph.
4. Benchmark path planners and active mapping with fair sensing assumptions.
5. Integrate real SLAM, robot kinematic constraints, human avoidance,
   collision checking and an independent E-stop in simulation.
6. Only then attempt supervised G1 trials in a controlled environment.

The package is stored under `crates/` to avoid changing Struktura's published
anomaly-detection crate. It is not in the root Cargo workspace.


## GENESIS + XENOWORLDS + DISCOVERY (v0.3)

Run a bounded, deterministic, MAP-Elites-inspired growth search followed by
**one post-selection evaluation** on a disjoint environment family:

```bash
PYTHONPATH=crates/colony/python python3 -m colony.genesis \
  --budget 24 --train-seeds 3 --test-seeds 30 --out genesis-report.json
```

- `genesis.py`: four-parameter growth genome, seeded mutation, two-dimensional
  niche archive, competition by training cost and champion selection.
- `world.challenge_world`: rooms, scatter and corridor-maze generator families.
- `discovery_report`: isolated training seeds (1000+) and corridor evaluation
  seeds (50000+); untuned growth and frontier controls, paired costs,
  reproducible genealogy and archived elites.
- No physical robots, independent learning agent, Struktura Guard binding, or
  claims of intelligence. The feature uses the Python reference planner; Rust
  parity and hardware integration are future work.

**First experiment** (seed 17, 24 proposals, 20 distinct genomes, six niches;
training uses three room and three scatter worlds; evaluation 30 corridor worlds):

| Held-out mean traversal/coverage-penalty cost (lower is better) | Cost |
|---|---:|
| Selected evolved genome | 167.17 |
| Untuned growth genome | **166.80** |
| Nearest-frontier control | 167.47 |

The selected genome is 0.30 steps better than frontier and **0.37 steps worse
than untuned growth** on the held-out set. Evolution is not demonstrated to
improve generalization. This is a small toy study and selected protocol; run
multiple independent search seeds and more challenging worlds before claims.
See `evidence/genesis-24-evaluations.json` for full history, training/test
identities and per-world costs.
