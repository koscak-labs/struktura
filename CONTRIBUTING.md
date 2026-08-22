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

- **More domains**: got time-series data from a new field? Run `struktura check` on it and share the results.
- **`no_std` support**: make the core library work without std (needs `libm` for `sqrt`/`ln`).
- **WASM target**: compile to WebAssembly for browser-based analysis.
- **Python bindings**: PyO3 wrapper so Python users can `pip install struktura`.
- **More tests**: edge cases, adversarial inputs, large-N performance benchmarks.

## Guidelines

- One change per PR
- Run `cargo test` and `cargo clippy -- -D warnings` before submitting
- Zero dependencies in the core library (feature-gated deps like `serde` are OK)
- The crate never bluffs: if you add a feature, include a test that proves it works

## Code of conduct

Be kind. Build things. Prove them.
