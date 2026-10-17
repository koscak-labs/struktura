# TRUSTROOT — reliability-aware exploration under failed sensing

**Research-only.** This is an SCA-inspired spatial active sensing experiment
using **real Struktura `dfa_short` calculations** on entirely **synthetic
correlated noise**. `dfa_short` is NOT Struktura Guard; raw `alpha-0.5` is NOT
an anomaly probability. No G1 robotics, sensor firmware, or space hardware.

## Hypothesis

Does a failure-aware growth strategy improve target-vicinity discovery when
some measurements are missing without sacrificing localization of real
correlation changes? Compare against signal-guided SCA, greedy signal tracking,
global coverage, costed coverage, and a fixed 2:1 SCA/coverage mixture.

The first attempted strategy fell back to global geometric coverage after any
missing reading and performed **worse** on local pilots. That version was
rejected before this experiment. This revision only triggers travel-aware
coverage after **5 of the previous 8 observations** are invalid, and resumes
attraction after sensor recovery. This rule is scripted, **not emergent**.

## Experimental controls and isolation

- 25×25 obstacle-free grid; compact shared starting probes around (12,12).
- 24 location observations maximum per method; optionally 85 Euclidean travel
  units. For over-budget proposals, a **common** nearest-affordable waypoint
  projection allows every method to continue rather than stopping long hops.
- Real DFA is computed once for each synthetic location using the Rust example
  `colony_dfa_bridge`. Each policy receives scores only for locations it visits;
  no policy can inspect hidden target coordinates or unvisited DFA scores.
- `correlation`: localized temporal-correlation change.
- `dropout`: deterministic spatial missing windows (invalid scores are `None`).
- `burst_dropout`: temporary instrument outage during observations 10–19.
- `correlated_decoy`: extraneous strong correlation peak away from the target.
- `shuffled`: deterministic temporal shuffling destroys original ordering while
  keeping sample values unchanged. Used as a negative control.
- Paired worlds, sample budgets, random seeds; **two** separate results for
  equal observation cap and equal travel cap. The latter still uses a maximum
  observation count, making it a constrained-budget comparison.
- The seed range **1000+** is frozen before hosted evaluation; local proxy
  diagnostics used seeds 0–11. Do not tune on held-out outcomes.
- Percentile bootstrap CIs are *paired across spatial seeds* and limited by
  small samples; report both beneficial and detrimental effects.

## Running

From Struktura repository root, with Rust and Python installed:

```sh
PYTHONPATH=crates/colony/python python3 -m unittest discover -s crates/colony/python/tests -v
PYTHONPATH=crates/colony/python python3 -m colony.trustroot \
  --seeds 8 --budget 24 --travel-cap 85 --seed-start 1000 \
  --out /tmp/trustroot.json
```

The tool builds and runs `cargo run --quiet --example colony_dfa_bridge` for
5×8 independent synthetic fields; the planner doesn't execute any robot
movement. The same DFA field is shared across competing methods, so results
are paired. Output includes all episode data and CI summaries.

## Important weaknesses

1. This doesn't integrate calibrated Struktura Guard event/quarantine rules.
   DFA R² expresses scaling-law fit quality, not hardware sensor reliability.
2. All windows can be scored offline; wall-time, memory, per-measurement compute
   and route energy have not been equalized. The costs of a real deployment may
   overturn apparently superior sample efficiency.
3. The grid is unobstructed and Euclidean; no SLAM, G1 dynamics, collision
   checks, communication loss or actual telemetry.
4. `shuffled` has a hidden target in the scorer even though its temporal
   structure is randomized. Success there is **chance vicinity discovery**,
   not a valid event-detection claim.
5. The fallback is explicitly hardcoded and only a hypothesis. No novelty or
   general advantage should be asserted from these simulated outcomes.
6. Eight evaluation seeds are an integration trial, not publishable evidence.

Next: calibration sets and proper anomaly event reporting; independent signal
families; 100+ paired held-out maps; collision-aware distance, compute, and
energy accounting; real labeled sensor logs; and ablations of fallback versus
randomized health flags.
