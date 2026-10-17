# VERITAS-2: Explore before claiming a source

This is a **predeclared follow-up** to VERITAS-1's negative result: on eight
synthetic correlation worlds, early independent confirmation produced **no
correct claims about the source location**, although both windows sometimes
showed genuine elevated temporal structure away from the hidden center.
A double confirmation can support the existence of a structural signal;
it cannot, by itself, prove localization of that signal's cause.

VERITAS-2 asks: does **explore first, verify later** improve correct source
attribution without inflating null-world false claims?

## Protocol, frozen before hosted evaluation

- Real `struktura::dfa_short` bridge on two independently generated windows
  at every potential sample location. Same 95th percentile null alpha cutoff
  trained on shuffled-only calibration seeds 6000 and 6001.
- **New independent held-out seeds 3000–3007**; no overlap with preceding
  seed ranges or calibration data.
- Max 24 windows and 85 Euclidean travel units per episode, with five common
  central startup sites and a common reachable-point projection.
- Compare identical sampling policies against one another under two claim
  strategies: **immediate** first independently confirmed point (existing
  VERITAS-1), versus **deferred**.
- Deferred uses up to 18 windows for spatial exploration without claiming.
  It ranks only *visited* locations by first-window DFA alpha and reserves six
  windows to independently repeat the six highest candidates above the frozen
  cutoff. It declares the most strongly confirmed sampled location, not the
  first elevated location.
- A true claim requires the repeated location to lie within two grid cells of
  the hidden synthetic source. Confirmed points outside that radius are false
  **target attribution** claims, not necessarily false anomaly signals.
- Any confirmed claim in shuffled-only worlds is false because the designed
  temporal correlation was destroyed.

Primary outcomes: verified-target rate; false-claim rate; windows and travel
used; paired bootstrap confidence interval on deferred-minus-immediate true
claims. The protocol makes no claim of statistical significance from eight
seeds, and uses no learned localization or physically faithful simulator.

## Run

```bash
PYTHONPATH=crates/colony/python python3 -m unittest discover -s crates/colony/python/tests -v
PYTHONPATH=crates/colony/python python3 -m colony.veritas_deferred \
  --seeds 8 --seed-start 3000 --budget 24 --travel-cap 85 \
  --out /tmp/veritas-deferred-evidence.json
```

### Limits

Exploration may collect more windows than immediate confirmation. The result
measures discovery under a **maximum** budget and reports actual windows used.
It does not compare equal realized sensor energy, computational overhead or
robotic motion. The observed correlation-field profile is constructed in the
simulator; detecting its center is not equivalent to causal diagnosis in space.
`dfa_short` is not a calibrated Guard alarm. A real implementation must
acquire independent sensor streams, perform mode-aware calibration, test
sensor health, and retain case/replay evidence.
