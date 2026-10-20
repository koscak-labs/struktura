# COLONY GENESIS v0.3 — experimental protocol

**Status:** research-only toy simulator. The actual SCA implementation is a
grid growth waypoint heuristic, not a biologically faithful continuous SCA.

## Hypothesis

Parameter-space novelty preservation could improve out-of-distribution
exploration compared with untuned growth and nearest-frontier selection.
The null hypothesis is that it does not.

## Three pillars

- **GENESIS:** a seeded mutation and niche-archive evolutionary search.
  Genome = `(influence_radius, growth_budget, gain_weight, distance_penalty)`.
  Niches categorize two *parameter* axes (horizon and budget), not measured
  behavioral diversity. This is MAP-Elites-inspired, not a full MAP-Elites
  implementation. Duplicates are cached.
- **XENOWORLDS:** map families generated separately from the planner.
  Rooms/scatter appear during training; randomized depth-first corridor mazes
  appear only at evaluation. Ground truth is used solely by sensor and oracle.
- **DISCOVERY:** a report with genomic lineage/proposal log, archive, episodes,
  controls, held-out paired differences, and explicit non-winning outcomes.

## Reproduction

```
PYTHONPATH=crates/colony/python python3 -m unittest discover -s crates/colony/python/tests -v
PYTHONPATH=crates/colony/python python3 -m colony.genesis --seed 17 \
  --budget 24 --train-seeds 3 --test-seeds 30 --out genesis-report.json
```

Optimization uses just 6 training maps (3 rooms, 3 scatter), and a deterministic
fixed budget. Once selected, the genome is evaluated exactly once on 30
corridor maps, seeds `50000..50029`. Note that rerunning the development
process and changing algorithms based on held-out results invalidates the
original holdout for confirmatory claims. A fresh evaluation split must then
be frozen, preferably with independent hidden scenario generators.

## First result and interpretation

The selected genome `radius=7, budget=60, gain_weight=0.5,
distance_penalty=2` scores 167.17 average cost vs 167.47 for nearest
frontier, but loses to untuned growth (166.80). In other words, **no evidence
of useful evolution yet**. We also have no independent repeat searches or
statistical power analysis; no publication-grade inference is warranted.

## Next gates

1. Multi-run training with different evolutionary seeds and fixed total
   evaluation budgets; include random search and hill-climbing controls.
2. Measure actual branch diversity, traversal distance, information gain,
   wall time and memory (not just parameter-space niches).
3. Freeze a new confirmatory unseen family / seed split before additional tuning.
4. Introduce noisy sensing, dynamic obstacles, risk controls and resource use.
5. Wire a read-only Struktura Guard telemetry adapter, then ROS 2 simulation.
6. No real G1 actuation without independent validated safety supervisor.
