# VERITAS — Scientific claim verification in COLONY

**Question:** How often does an SCA-based active observer "discover" a
synthetic temporal anomaly for which it has no statistically adequate evidence?

The previous SIGNALROOT/TRUSTROOT benchmark rewarded entering a two-cell radius
of a hidden target, even after temporal shuffling removed the designed anomaly.
VERITAS separates *finding a location* from *making a supported event claim*.

## Predeclared method

1. Draw two **independent** length-96 correlated-noise windows at each possible
   location. For shuffled controls, independently shuffle each window, preserving
   the distribution but destroying its designed temporal correlation.
2. Use the repository's actual `struktura::dfa_short` Rust implementation on
   both windows, through the same `colony_dfa_bridge` example.
3. Choose a single alpha threshold using the empirical **95th percentile** of
   valid DFA windows on separate shuffled-only calibration seeds 6000, 6001.
   This estimates an uncorrected single-window null cutoff. It is **not** an
   eventwise or familywise false alarm guarantee, nor a Struktura Guard event.
4. Compare six spatial sampling policies on **held-out seeds 2000–2007** under
   the same 24-window budget and maximum 85 units of Euclidean travel.
5. When the first DFA alpha exceeds the frozen threshold, request a second,
   independent window at the **same location**. The repeated measurement
   consumes another observation, but no travel. Declare a claim only if both
   independently pass. If the first claim is false, count it; do not silently
   retry until a claim looks good. Invalid samples never count as healthy.
6. Declare a *verified target hit* only if a confirmed claim lands within two
   cells of the hidden source. Confirmed claims elsewhere are false TARGET
   claims. For shuffled controls, every claim is a false discovery: there is no
   designed temporal source after shuffling even if the original random source
   position remains available to the evaluation oracle.

## Metrics

Report separately: verified-target rate, false-claim rate, any-claim rate,
windows used (including confirmation), confirmation count, mean travel and
sensor abstentions. Report all methods, all world families and raw outcomes.

An important distinction: a high DFA alpha with adequate fit R² does not
constitute an anomaly **probability**. The `r_squared` validity cutoff is inherited
from the exploratory Rust bridge. Further calibration should use healthy
training data from the *actual* sensor and preserve calibration/test session
separation.

## Reproduction

From the repository root:

```sh
PYTHONPATH=crates/colony/python python3 -m unittest discover -s crates/colony/python/tests -v
PYTHONPATH=crates/colony/python python3 -m colony.veritas \
  --seeds 8 --seed-start 2000 --budget 24 --travel-cap 85 \
  --out /tmp/veritas-evidence.json
```

This runs more expensive than TRUSTROOT because two independent real-DFA
fields are evaluated per synthetic world. No simulation of actual robot
movement, physical energy costs, or live Struktura Guard sensor streams.

## Scientific standards before external claims

- Repeat over many independent null calibration and search seeds; produce
  confidence intervals and report calibration-family mismatch.
- Correct for repeated looks if making statistical significance claims.
- Compare against classical sequential probability ratio tests, Bayesian
  active sensing, matched-travel random search and spatial coverage.
- Separate signal quality, sensor health and physical target priority.
- Replace synthetic windows with independently labeled mission/robot telemetry,
  use Guard's actual mode-aware events, and require offline replayable evidence.
- Review the existing literature before making a novelty or priority claim.

**Outcome is an experiment, not a promise of superiority.** If the two-window
confirmation suppresses legitimate discoveries as much as false discoveries,
we must report that and rethink the measurement model.
