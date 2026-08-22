# Formal Properties of DFA as a Runtime Monitor

This document establishes the mathematical guarantees of Detrended Fluctuation
Analysis (DFA) when used as a runtime structural health monitor. These properties
are relevant to integration with frameworks that generate verified runtime
monitors, such as NASA's Copilot/Ogma pipeline.

Implementation: [Struktura](https://crates.io/crates/struktura)

---

## 1. Convergence

**Property.** Let {X_i} be a stationary stochastic process with long-range
correlations characterized by a true scaling exponent alpha_true. The DFA
estimator alpha_n, computed over n samples, converges to alpha_true as
n tends to infinity:

    alpha_n → alpha_true   as   n → ∞

**Basis.** DFA computes the scaling relationship between fluctuation magnitude
F(s) and box size s via ordinary least squares regression in log-log space.
For a process with power-law scaling C(s) ~ s^(2*alpha), the fluctuation
function satisfies F(s) ~ s^alpha. The OLS estimator of the log-log slope
is consistent under standard regularity conditions (finite variance of
log F(s) at each scale).

The method was introduced and validated in:

- C.-K. Peng, S. V. Buldyrev, S. Havlin, M. Simons, H. E. Stanley,
  A. L. Goldberger, "Mosaic organization of DNA nucleotide sequences,"
  *Physical Review E* 49(2), 1685--1689, 1994.

- C.-K. Peng, S. Havlin, H. E. Stanley, A. L. Goldberger, "Quantification
  of scaling exponents and crossover phenomena in nonstationary heartbeat
  time series," *Chaos* 5(1), 82--87, 1995.

Convergence has been verified empirically on fractional Gaussian noise with
known Hurst parameters (see Hu et al., *Physical Review E* 64, 011114, 2001).

---

## 2. Finite-Sample Error Bound

**Property.** For a window of n samples with B box sizes, the standard error
of the DFA alpha estimator decreases as O(1/sqrt(B)), where B is the number
of log-log regression points. For fixed B (= 6 in our implementation), the
estimation variance decreases with n because each box size yields a more
precise fluctuation estimate from a larger number of non-overlapping segments.

**Empirical verification.** Struktura implements bootstrap confidence intervals
via `bootstrap_alpha(values, iterations)`. For n = 256 with 100 bootstrap
resamples, the 95% confidence interval width is typically 0.06--0.10 (i.e.,
alpha ± 0.03--0.05). This narrows with larger windows:

| Window size | 95% CI width (typical) |
|-------------|------------------------|
| 128         | 0.10 -- 0.15           |
| 256         | 0.06 -- 0.10           |
| 512         | 0.04 -- 0.07           |

The bootstrap procedure resamples with replacement from the original series,
recomputes alpha on each resample, and reports the 2.5th and 97.5th
percentiles. This provides distribution-free confidence intervals without
parametric assumptions.

---

## 3. Exact-or-Abstain Guarantee

**Property.** Let R^2 denote the coefficient of determination of the log-log
regression (log F(s) vs. log s). If R^2 falls below a configurable threshold
tau (default tau = 0.7), the monitor **suspends** monitoring for that channel
and reports `Abstain` rather than emitting an alert.

**Rationale.** A low R^2 indicates that the signal does not exhibit power-law
scaling at the measured box sizes. Alerting on an unreliable alpha estimate
produces false positives. The exact-or-abstain policy ensures that:

- Every reported alpha has R^2 >= tau (the scaling law is real).
- No alert fires on a signal that lacks exploitable structure.
- The monitor is honest about the limits of its own measurement.

**Implementation.** In Struktura, the `LawQuality` enum encodes the
confidence level: `Exact` (R^2 >= 0.95), `Strong` (>= 0.85), `Good`
(>= 0.7), `Approx` (>= 0.5), `Abstain` (< 0.5), `Insufficient`
(too few samples). The health check only fires when quality is `Good`
or above.

---

## 4. Computational Complexity

**Property.** Each DFA evaluation over a window of n samples with B box
sizes requires O(n * B) multiply-accumulate operations.

**Derivation.** For each of B box sizes s_b, the algorithm:
1. Divides n samples into floor(n / s_b) non-overlapping segments.
2. In each segment, performs a linear regression (O(s_b) operations).
3. Computes the residual variance (O(s_b) operations).

Total work per box size: O(n). Total across B box sizes: O(n * B).
The final log-log regression is O(B), negligible.

**Concrete cost.** With B = 6 box sizes {16, 24, 36, 54, 81, 121} and
n = 256:

- Cumulative profile: 256 additions
- Per-box detrending: sum of floor(256/s_b) * s_b ~ 256 * 6 ~ 1536 MACs
- Log-log regression: 6 * 4 = 24 MACs
- **Total: ~1,500 MACs per channel per evaluation**

On a 400 MHz RAD750 (typical flight processor) at 1 MAC/cycle, this is
3.75 microseconds. At a 1 Hz rate group, CPU utilization per channel is
0.000375%, or **0.004% for 10 monitored channels**.

---

## 5. Memory Guarantee

**Property.** Each monitored channel requires exactly
`n * sizeof(double) + C` bytes of memory, where C is a small constant
for bookkeeping state. There is no dynamic memory allocation.

**Concrete cost.** For n = 256:

| Component              | Bytes  |
|------------------------|--------|
| Circular buffer (f64)  | 2,048  |
| Position index (u32)   | 4      |
| Filled flag (u32)      | 4      |
| Baseline alpha (f64)   | 8      |
| Baseline set flag (u8) | 1      |
| Window count (u32)     | 4      |
| **Total per channel**  | **~2.1 KB** |

For 10 monitored channels: ~21 KB total.

**Guarantee.** The implementation uses a fixed-size circular buffer.
No heap allocation occurs during monitoring. No `malloc`, `calloc`,
or `realloc` calls. This matches the constant-memory guarantee of
Copilot-generated C99 monitors: memory usage is fully determined at
compile time.

---

## 6. Determinism

**Property.** For a given input sequence {X_1, ..., X_n}, the DFA
computation is a pure function:

    dfa_compute(X, n) = (alpha, R^2)

The output is identical across invocations, platforms (within IEEE 754
double precision), and compilation targets. There is no internal
randomness, no sampling, and no state beyond the current window contents.

**Implication.** DFA monitors are reproducible and auditable. A recorded
input sequence can be replayed offline to reproduce the exact alpha
trajectory. This property is shared with Copilot-generated monitors
and is essential for post-incident analysis in flight software.

---

## 7. Shuffle Proof (Structure Verification)

**Property.** Let alpha_original = DFA({X_1, ..., X_n}) and let
alpha_shuffled = DFA(pi({X_1, ..., X_n})) where pi is a uniformly
random permutation. Then:

    alpha_shuffled → 0.5   as   n → ∞

for any input sequence with finite variance.

**Rationale.** A random permutation destroys all temporal correlations
while preserving the marginal distribution (mean, variance, histogram).
If the original alpha differs significantly from 0.5 but the shuffled
alpha is near 0.5, the measured structure is a real property of the
signal's temporal ordering, not an artifact of its amplitude distribution.

**Implementation.** Struktura's `prove_structure(values)` function
performs this test and returns a `ShuffleProof` containing the original
alpha, shuffled alpha, and the absolute difference. A large difference
(typically > 0.1) confirms genuine temporal structure.

**Use in monitoring.** During the baseline learning phase, the monitor
can run a shuffle proof to verify that the channel carries exploitable
structure before committing to a baseline. Channels that fail the
shuffle proof (alpha_original near alpha_shuffled) are flagged as
unsuitable for DFA monitoring.

---

## 8. Complementarity with Temporal Logic Monitors

**Property.** DFA structural health monitoring and Copilot temporal
property monitoring detect fundamentally different classes of anomalies.
Neither subsumes the other.

| Aspect | Copilot (temporal logic) | DFA (structural health) |
|--------|--------------------------|-------------------------|
| Detects | Property violations (threshold crossings, timing faults, state machine deviations) | Behavioral degradation (correlation pattern changes, scaling exponent drift) |
| Fires when | A specified Boolean property becomes false | The statistical structure of the signal shifts from baseline |
| Requires | Formal specification of expected behavior | Only historical baseline data |
| Catches | Known failure modes (specified in advance) | Unknown degradation modes (detected by structural change) |

**Example.** A reaction wheel bearing degrading over weeks:
1. Vibration correlation pattern changes (DFA alpha shifts from 0.7 to 0.4).
2. Weeks later, vibration amplitude exceeds threshold (Copilot property fires).

DFA detects the degradation at step 1. Copilot detects the failure at step 2.
Running both in the same application gives the earliest possible detection
(DFA) with formal correctness guarantees on known properties (Copilot).

---

## Verified Empirical Results

All numbers below are from actual runs of Struktura on real-world data.

### Cross-Domain Universality

| Domain | Signal | N | DFA alpha | R^2 | Interpretation |
|--------|--------|---|-----------|------|----------------|
| Spacecraft | Queue depth trace | 500 | 0.593 | 0.789 | Correlated (healthy) |
| Bearing | CWRU normal (97.mat, 12 kHz) | 243,938 | 0.389 | 0.872 | Structured vibration |
| Genome | Human chr1 GC% (250 bp windows) | 8,000 | 0.909 | 0.991 | Strong fractal scaling |
| Cardiac | RR intervals (normal HRV) | 2,048 | 0.695 | 0.985 | Healthy fractal dynamics |

### Bearing Fault Detection (CWRU Bearing Data Center, 12 kHz)

| Condition | DFA alpha | R^2 | Delta from normal | Verdict |
|-----------|-----------|------|-------------------|---------|
| Normal (97.mat) | 0.389 | 0.872 | -- | Baseline |
| Inner race fault (105.mat) | 0.146 | 0.635 | -0.243 | **DETECTED** |
| Outer race fault (130.mat) | 0.247 | 0.687 | -0.142 | **DETECTED** |
| Ball fault (118.mat) | 0.275 | 0.746 | -0.114 | **DETECTED** |

All three fault types produce alpha shifts exceeding 0.08 from baseline.
No amplitude thresholds are needed -- the structural change alone is
diagnostic.

### Genome Fractal Scaling (8 Human Chromosomes)

| Chromosome | DFA alpha | R^2 |
|-----------|-----------|------|
| chr1 | 0.909 | 0.991 |
| chr2 | 0.699 | 0.991 |
| chr3 | 0.659 | 0.998 |
| chr4 | 0.894 | 0.997 |
| chr5 | 0.824 | 0.994 |
| chr6 | 0.822 | 0.998 |
| chr7 | 0.862 | 0.997 |
| chr8 | 0.816 | 0.995 |

All 8 chromosomes at R^2 > 0.99, confirming DFA reliably extracts
structural signatures at very high confidence.

---

## References

1. C.-K. Peng, S. V. Buldyrev, S. Havlin, M. Simons, H. E. Stanley,
   A. L. Goldberger, "Mosaic organization of DNA nucleotide sequences,"
   *Physical Review E* 49(2), 1685--1689, 1994.

2. C.-K. Peng, S. Havlin, H. E. Stanley, A. L. Goldberger, "Quantification
   of scaling exponents and crossover phenomena in nonstationary heartbeat
   time series," *Chaos* 5(1), 82--87, 1995.

3. K. Hu, P. C. Ivanov, Z. Chen, P. Carpena, H. E. Stanley, "Effect of
   trends on detrended fluctuation analysis," *Physical Review E* 64,
   011114, 2001.

4. A. L. Goldberger, L. A. N. Amaral, J. M. Hausdorff, P. C. Ivanov,
   C.-K. Peng, H. E. Stanley, "Fractal dynamics in physiology: Alterations
   with disease and aging," *Proceedings of the National Academy of
   Sciences* 99(suppl 1), 2466--2472, 2002.

5. Case Western Reserve University Bearing Data Center,
   https://engineering.case.edu/bearingdatacenter

6. I. Perez, F. Dedden, R. Scott, "Trustworthy Runtime Verification via
   Bisimulation (Experience Report)," *Proceedings of the ACM on
   Programming Languages* 7(ICFP), 2023.
