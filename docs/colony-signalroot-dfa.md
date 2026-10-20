# SIGNALROOT-DFA — first genuine Struktura-to-COLONY experiment

**Scientific status:** new, synthetic **temporal-correlation** experiment. The Rust example runs the repository's actual `struktura::dfa_short` implementation, not a mock signal, Guard, or C99 anomaly detector. No physical robot or space hardware and no evidence of novel behavior yet.

## Experimental contract

- 25×25 locations, independent length-96 measurement windows. A hidden source perturbs temporal correlation; one decoy family varies amplitude, a dropout family injects invalid readings.
- Python writes `x,y,values...` with no true source coordinates or labels. The Rust binary scores **all** location windows offline, outputting only `alpha`, `r_squared`, and a measurement-validity flag. Invalid/abstained channels remain unknown; they are **never zero-filled**.
- Planner queries one location per observation, sees only the returned score at that location and prior measurements, never the unvisited score map. The hidden source enters only the simulator's signal generation and outcome evaluator.
- Three policies receive the same five initial observations, the same seed/family field, and the same observation budget: geometric coverage, greedy structural signal, SCA-inspired structural signal. Travel distance is reported rather than matched.
- The transformation `max(0, alpha-0.5)` is **a predeclared heuristic**, not an empirical calibration or an anomaly likelihood. All methods are compared on held-out seeds; do not tune this mapping using test outcomes.

## Run

```sh
PYTHONPATH=crates/colony/python python3 -m unittest discover -s crates/colony/python/tests -v
PYTHONPATH=crates/colony/python python3 -m colony.signalroot_dfa --seeds 4 --budget 35 --out /tmp/colony-dfa.json
```

The Python command invokes `cargo run --quiet --example colony_dfa_bridge` from the repo root. Rust compilation is needed to verify the real detector path. Smaller CI smoke uses one seed to prevent repeated expensive full-map DFA analysis.

## Limitations & research follow-ups

1. This is **DFA**, not Struktura Guard. Integrating actual calibrated Guard must be a distinct follow-up using a healthy baseline per sensor/channel and typed anomaly events.
2. A single local 96-sample window has unstable alpha. `r_squared>=0.5` is an experimental abstention rule, not calibrated confidence.
3. The synthetic target is correlated noise, not a scientifically validated planetary anomaly. An apparent navigation benefit could simply follow our hand-designed field generator.
4. Precomputing all measurement windows adds offline compute cost to every method. Report total DFA CPU cost and score-generation wall time for a fair cost comparison.
5. Future test: independent source families, irrelevant correlated decoys, sensor intermittency, correlated measurement errors, realistic motion constraints, and randomized trace collection order.
6. No result should be described as new to humanity without stronger baselines, replication, and literature review.
