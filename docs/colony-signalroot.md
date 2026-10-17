# SIGNALROOT — COLONY active sensing experiment v0.1

**Experimental, simulation-only.** This is an SCA-inspired spatial sampling
heuristic, not a new physical G1 controller and not a validated Struktura detector.

## Hypothesis

Growing sensing tips toward locations associated with anomalous observations
could find environmental change more efficiently than a blind coverage survey.
The crucial adversarial hypothesis is that false cues can *worsen* discovery.

## Run

From the Struktura repository root:

```sh
PYTHONPATH=crates/colony/python python3 -m unittest discover -s crates/colony/python/tests -v
PYTHONPATH=crates/colony/python python3 -m colony.signalroot --seeds 12 --budget 35 --out signalroot-evidence.json
```

Only Python standard library is needed. Five strategies: raster scan, random,
max-distance coverage, highest-signal local greedy, SCA-inspired branching.
All start with the same five readings. Four synthetic families: compact,
diffuse, decoy attraction, and deterministic sensor dropout. The true target
and alternate source are unknown to every policy and used solely for the
synthetic sensor/environment and evaluation.

**Cue source:** scalar sampled synthetic field; it is not a DFA result, Hurst
exponent, Struktura Guard alarm, or validated sensor reading. Future Struktura
integration needs a mode-aware stream with honest calibration and event-wise
scoring. Do not label these as Struktura detection metrics.

**Success metric:** at least one *valid* observation within two grid cells of
the hidden source in the observation budget. This is target-vicinity sampling,
not fault classification, safety validation, or source identification.
Euclidean travel assumes traversable open grid and ignores robot kinematics,
obstacles, energy and time. Identical observation budgets are not identical
wall-clock compute or travel budgets.

## Exploratory evidence — 12 seeds × 4 families × 5 policies, 35 readings

| Family | Random success | Coverage success | Signal SCA success | Signal SCA mean travel | Random mean travel |
|---|---:|---:|---:|---:|---:|
| Compact | 6/12 | 6/12 | 9/12 | 179.27 | 474.84 |
| Diffuse | 6/12 | 6/12 | 12/12 | 181.30 | 474.84 |
| Decoy | 6/12 | 6/12 | 6/12 | 173.71 | 474.84 |
| Dropout | 3/12 | 6/12 | 8/12 | 181.33 | 474.84 |

Numbers are exploratory, with only 12 seeds and a simulated oracle. No
inferential statistical claim or physical-world generalization follows.
The negative decoy result matters: attraction toward anomalous signals can
trap the policy at the wrong source. Greedy performs as well as SCA in the
diffuse case, so SCA's distinct value is also unproven.

## Next controlled campaigns

1. **Struktura adapter:** Generate a streaming telemetry input with
   calibration and real Guard events. Bind each alarm to its original spatial
   sample, preserve validity and alarm tick. Compare to equivalent
   generic threshold, random, and oracle-free coverage controls.
2. **SCAR TISSUE:** Damage sensors and graph edges mid-episode; compare
   remembered repair against BFS rebuild, frontier-only and random.
3. **BLINDSPOT:** Train policies to distinguish unavailable sensor readings
   from nominal readings and report calibrated abstention under dropouts.
4. Repeat >100 seeds across multiple generators and pre-register a fully
   held-out test set. Match travel/compute budgets and estimate paired CIs.

Related existing work: Physarum decentralized mesh (Scientific Reports,
2026), Physarum adaptive flow networks (Scientific Reports, 2026),
FrontierNet (RA-L 2025), active perception network (IJRR 2025). This program
makes **no discovery or world-first claim**.
