# Struktura DFA — TimeEval Algorithm Adapter

**The first Rust algorithm in the ESA-ADB benchmark.**

Detrended Fluctuation Analysis (DFA) anomaly detection: computes windowed
scaling exponents and scores deviations from a self-calibrated baseline.
Zero training, zero hyperparameter tuning.

## Method

1. Slide a window of `window_size` samples (default 256) across the time series
   with `step_size` stride (default 128).
2. Compute the DFA scaling exponent α for each window.
3. Self-calibrate: use the first third of the α series as the baseline (mean and
   standard deviation).
4. Score each window as `|α - baseline_mean| / (baseline_std + threshold)`.
5. Expand window-level scores to per-timestep scores by assigning each row the
   maximum score of all windows that cover it.
6. For multivariate input: score each channel independently, take the max across
   channels per timestep.

Higher scores = more anomalous.

## Learning type

Unsupervised. No training step. The `train` execution type returns immediately.

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| window_size | int | 256 | DFA window size (≥ 64) |
| step_size | int | 128 | Sliding window stride |
| threshold | float | 0.05 | Noise floor for the score denominator |

## Source

- Crate: [struktura on crates.io](https://crates.io/crates/struktura)
- Method: Peng et al., "Mosaic organization of DNA nucleotides", Physical Review E, 1994
- Repository: [koscak-labs/struktura](https://github.com/koscak-labs/struktura)

## License

MIT OR Apache-2.0
