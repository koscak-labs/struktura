# Struktura

**Predict failure before it happens.**

Struktura is a zero-dependency Rust crate for universal anomaly detection. It uses Detrended Fluctuation Analysis (DFA) to measure the structural health of any time series — bearings, heartbeats, spacecraft telemetry, drone motors, DNA sequences.

One algorithm. Any signal. No training data. No domain knowledge.

## Why Struktura?

Current monitoring tools watch for threshold violations: "is the temperature above 80C?" But structural degradation changes the *pattern* of a signal before it changes the *amplitude*. A bearing that's starting to wear changes its vibration structure weeks before it exceeds any alarm threshold.

Struktura catches what threshold monitors miss.

## Install

```
cargo add struktura
```
