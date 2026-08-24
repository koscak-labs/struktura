# Contributing to Struktura

Thanks for your interest in contributing.

## Quick start

```
git clone https://github.com/koscak-labs/struktura
cd struktura
cargo test
cargo run --example benchmark
cargo run --bin struktura -- demo
```

## What we need help with

- **More domains**: got time-series data from a new field? run `struktura check` on it and share the results
- **WASM playground** (#9): interactive browser demo — drag-drop a CSV, see DFA analysis live
- **Python bindings** (#3): PyO3 wrapper so Python users can `pip install struktura`
- **SOTA comparison** (#11): run the SMAP/MSL benchmark through other anomaly detectors and compare honestly
- **Mutation testing**: we're at ~44% mutation score — help us get to 60%+ (see #6 for the methodology)
- **More tests**: boundary cases, adversarial inputs, large-N performance, new datasets

## Guidelines

- One change per PR
- Run `cargo test` and `cargo clippy -- -D warnings` before submitting
- Zero dependencies in the core library (feature-gated deps like `serde` are OK)
- The crate never bluffs: if you add a feature, include a test that proves it works

## Code of conduct

Be kind. Build things. Prove them.
