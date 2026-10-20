---
title: How a Rust Crate Measures Structural Shifts in Time-Series Data
published: false
tags: rust, anomaly-detection, algorithms, science
---

> **Status 2026-10-20: DRAFT, not published.** Rewritten from the 2026-09-17 version.
> The old title ("How 98 Lines of Rust Detected a Spacecraft Anomaly 44 Billion Miles
> Away") asserted a detection that the project's own claims audit withdrew, and the "98
> lines" number was a miscount. Both are corrected here. Every other number is checkable
> against `docs/claims.tsv`.

In May 2022, NASA's Voyager 1 — humanity's most distant spacecraft at 14.5 billion miles —
started sending gibberish. The attitude articulation and control system (AACS) was feeding
telemetry through a computer that had stopped working years ago.

The tool I built is called [**struktura**](https://crates.io/crates/struktura), and it is
open source. But before I show you what it does, let me be clear about what it does *not*
claim: it did **not** detect the AACS anomaly. The honest framing is that it measures
structural change in the magnetometer data across a year boundary, and that measurement is
currently inconclusive. Everything below is the part I *can* stand behind.

## The Problem With Thresholds

Every monitoring system you've used works the same way: set a threshold, alert when something
crosses it.

```
if temperature > 85.0 { alert("overheating") }
```

This works until it doesn't. Here's what thresholds miss:

A bearing can develop an inner-race fault while its vibration amplitude stays within normal
bounds. The **structure** of the vibration changes — the scaling relationship between time
scales shifts — but the raw numbers look fine. By the time amplitude crosses your threshold,
you've already lost the bearing.

Spacecraft telemetry is the same. A magnetometer reading can drift structurally while
individual values stay in-band.

## What DFA Actually Measures

Detrended Fluctuation Analysis computes a single number — the scaling exponent α (alpha) —
that captures how a signal is organized across time scales.

The algorithm, in full, is about 68 lines of code — the box loop and the log-log fit are
the math, the rest is bookkeeping:

1. Remove the mean, take the cumulative sum — the "profile".
2. Cut the profile into boxes of increasing size, linearly detrend each box, measure the
   residual fluctuation.
3. Fit a line in log-log space. Its slope is α.

That's it. No matrices, no gradients, no hyperparameters, no training set. α ≈ 0.5 means
white noise. α > 1 means persistent long-range correlation — a trend that persists no matter
how you zoom out.

The reference implementation lives in one public module, `src/dfa_core.rs`, and there's a
test asserting it agrees with the optimized production path to 1e-9. What you read is what
runs.

## The Code

```rust
pub fn dfa_reference(values: &[f64]) -> DfaResult {
    // 1. cumulative profile y_j = cumsum(x - mean)
    // 2. boxes of increasing size, detrend each, measure residual fluctuation
    // 3. log-log least-squares fit -> alpha
}
```

The shipped path is an allocation-free, prefix-sum-optimized equivalent. The point of the
reference version is that you can read the whole detector in one sitting and check it by hand.

The method itself is from 1994 — C.-K. Peng et al., *"Mosaic organization of DNA nucleotide
sequences,"* Physical Review E 49(2), 1685–1689. Crossref counts 4,316 citations. If you
know DFA, the rest of this is just the implementation; if you don't, the three steps above
are the whole thing.

## What It Actually Scores

Here is the honest scoreboard, from `docs/claims.tsv` — every row is a command with a
control, and CI fails if the output drifts:

| Benchmark | Result | Caveat stated in the repo |
|---|---|---|
| SMAP/MSL (NASA Mars rover telemetry) | F1 **0.655** with `--ar 0 --dfa` | JPL's telemanom LSTM scores P=87.5% / R=80.0% on the same data — supervised models win here |
| CWRU bearing faults | α 0.689 → 0.183 | fault vs healthy |
| Numenta Anomaly Benchmark (58 series, 116 labelled windows) | `guard` catches **45** windows with **45** false alarms (0.15 per 1,000 samples) | catches the *fewest* windows of any detector; limit check catches 48 with 240 false alarms |
| Human chr1 genome | α=0.967, R²=0.9803 | 248,956,422 bases |
| HRV (PhysioNet) | α1 AUC **0.829** | SDNN alone separates the groups at least as well (0.897) — α1 adds nothing on top of it. Not a diagnostic. |
| ESA OPS-SAT-AD | AUC-ROC **0.943** (AUC-PR 0.882) on 529 test segments | shuffling values inside a segment drops it to 0.554; on segments under 64 samples the shuffled control still reaches 0.743 |

Two things I want to be explicit about:

**The F1 number was once 0.755, then 0.788.** Neither was reproducible on any commit or any
flag. The 0.788 was an editing error of some kind — it never came from a run. The current
0.655 is the best configuration actually measured, and the plain `struktura smap` scores
0.161 because the default residual leg raises 419 false alarms on JPL's pre-scaled channels.

**"No training" has an exception.** The SMAP path fits a closed-form AR predictor. That's a
closed-form fit, not gradient training — but it's a fit, and I name it rather than hide it
behind the phrase "zero training".

## The Voyager Comparison — and Why It Isn't a Detection

NASA publishes Voyager 1 magnetometer data through the
[HAPI API](https://cdaweb.gsfc.nasa.gov/hapi). Pulling the public dataset spanning the
anomaly window and running the analysis:

```
2021 (healthy)     alpha=0.989   R²=0.9869
2022 (May–Jul)     alpha=0.801   R²=0.9681
                  shift=-0.187
```

α did move. But the CLI's own output is careful about what that means:

> Year-over-year comparison only. Pre-anomaly vs during-anomaly 2022 shows no significant
> shift (p=0.52), and these slices give z=1.5: inconclusive. This is not a detection of the
> AACS anomaly. (the CRITICAL label is a fixed alpha threshold, not significance)

The `CRITICAL` label is a hard-coded α threshold. A label is not a test. The right claim is
"DFA measures a structural shift in Voyager's magnetometer data across a year boundary" —
and even that is currently inconclusive, which is why the repo keeps it in the claims
ledger as `voyager-alpha` with the word *inconclusive* baked into the expected output.

That framing is deliberate. A crate that says "I detected the anomaly" and then can't
reproduce it is worse than useless in flight software.

## Performance

```
signal size   struktura (Rust, measured)   Python nolds (published figure, not run here)
4,096 pts     0.24 ms                      ~15-25 ms              ~85x
16,384 pts    0.93 ms                      ~60-100 ms             ~86x
65,536 pts    2.89 ms                      ~250-400 ms            ~112x
```

(Timings are machine-dependent and not fixture-checked — on a faster box the Rust
column comes out several times quicker. Reproduce and use your own numbers.)

The Rust column is measured — on the reference machine the README's numbers come
from, not on whatever box you are reading this on; timings are machine-dependent and
are not fixture-checked, so treat them as indicative. The Python column is a published
reference figure, not a head-to-head run — I say so in the table header because the
difference matters. Single-threaded, release build, same 4K window. Reproduce with
`cargo run --release --example speed_bench`.

## Where It Fits

Not as a replacement for a supervised model on a problem where a supervised model wins. It
fits where you can't train: a microcontroller with no heap, a satellite downlink path, a
bearing on a spinning shaft where the failure mode is structural rather than amplitude.

That last case is the one I keep coming back to. The 2026-09-17 BASEPROD analysis on ESA's
rover data was instructive precisely because it *failed*: leave-one-traverse-out on four
terrain classes gave 40% against a 25% chance baseline, and the standard deviation of a
single force channel did at least as well. Alpha was not a terrain classifier. On the
reserved traverse, adding α broke two blocks and rescued none.

A method that fails in public, with the control included, is more useful than one that
passes suspiciously.

## The Rest of It

Beyond the DFA core: `guard` (self-calibrating streaming monitor with per-channel
baselines and sensor quarantine), `mission` (detect → decide → adapt without human
calls), `redblue` (adversarial search that evolves its own detector legs — 69% → 97%
peak coverage, 90% at generation 10), C99 codegen for cFS / F´ / ROS, and `no_std` with
a single `libm` dependency.

```
cargo install struktura && struktura demo
```

The README says plainly that DFA detects structural shifts, not point anomalies; that
upstream preprocessing can invalidate a baseline; that α alone is not a decision; and that
supervised models beat it on SMAP/MSL. Those caveats are in the README because a crate
whose README oversells it is a crate you stop trusting.

Full method, evidence lines, and the control for every claim:
[docs/claims.tsv](https://github.com/koscak-labs/struktura/blob/master/docs/claims.tsv) ·
[docs.rs/struktura](https://docs.rs/struktura) ·
[github.com/koscak-labs/struktura](https://github.com/koscak-labs/struktura)
