# Struktura

**Time-series anomaly detection with no training data.**

Struktura is a Rust crate and CLI. It measures the correlation structure of a time series with detrended fluctuation analysis (DFA) and runs a streaming monitor (`struktura guard`) that calibrates itself on the first rows of your data: a CSV goes in, an exit code comes out.

## Why Struktura?

Most monitoring checks whether a value leaves a band ("is the temperature above 80 C?"). That is fast when a fault makes values bigger. It misses faults that change how values follow each other while their spread stays the same, and it false-alarms on healthy signals that wander slowly. Struktura's monitor looks at structure as well as level. Use it next to limit checks, not instead of them.

Measured results, with the command behind each number, are in the [README](https://github.com/koscak-labs/struktura#readme) and in [docs/claims.tsv](https://github.com/koscak-labs/struktura/blob/master/docs/claims.tsv), which CI re-runs.

## Install

```
cargo add struktura
```
