# TWiR Crate of the Week — Self-Nomination

Post at: https://users.rust-lang.org/t/crate-of-the-week/2704

I'd like to suggest [struktura](https://crates.io/crates/struktura) — structural-shift
detection for time-series data.

Not thresholds, not ML. One number: the DFA scaling exponent α. When α shifts from
baseline, the signal's *structure* changed.

- `no_std`, one dependency (`libm`), zero heap on the hot path.
- ~0.24 ms on 4K points, release build.
- Measured on NASA SMAP/MSL rover telemetry, ESA OPS-SAT-AD, CWRU bearing faults,
  Numenta's anomaly benchmark, and the human genome.

It does *not* beat supervised models on SMAP/MSL — F1 0.655 against the JPL telemanom
LSTM's P=87.5% / R=80.0%. The pitch is no training, no GPU, and embedded deployment.
The Voyager 1 magnetometer comparison is year-over-year and currently inconclusive; I
don't claim otherwise.

[GitHub](https://github.com/koscak-labs/struktura) | [docs.rs](https://docs.rs/struktura)
