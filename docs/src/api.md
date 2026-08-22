# API Reference

## Core functions

### `dfa(values: &[f64]) -> DfaResult`
Compute the DFA scaling exponent. Returns alpha and R-squared.

### `acr(values: &[f64]) -> DfaResult`
Compute autocorrelation decay exponent.

### `analyze(values: &[f64]) -> StructuralLaw`
Full structural analysis: DFA + ACR + statistics.

### `health_check(law: &StructuralLaw, baseline: f64) -> HealthVerdict`
Compare current DFA alpha against a baseline.

## Types

### `DfaResult { alpha: f64, r_squared: f64 }`
### `StructuralLaw { hurst, dfa, acr, mean, std_dev, kurtosis, p99, max, n, quality }`
### `LawQuality` — Exact, Strong, Good, Approx, Abstain, Insufficient
### `HealthVerdict` — Healthy, Watch, Warning, Critical

## Thresholds

| Shift from baseline | Verdict |
|---------------------|---------|
| < 0.03 | Healthy |
| 0.03 - 0.08 | Watch |
| 0.08 - 0.15 | Warning |
| > 0.15 | Critical |
