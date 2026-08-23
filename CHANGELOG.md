# Changelog

All notable changes to Struktura are documented here.

## v1.2.x (2026-08-23) — Code Generation

- v1.2.4: Fix all clippy errors — safety docs, remove unreachable patterns
- v1.2.3: cFS app codegen (`struktura codegen --cfs`) + `--name` flag
- v1.2.2: Fix fprime codegen (no format! crash) + clean FPP output
- v1.2.1: F Prime component codegen (`struktura codegen --fprime`)
- v1.2.0: C code generator (`struktura codegen`) + `codegen.rs` module

## v1.1.x (2026-08-22) — C FFI

- v1.1.1: `struktura.h` C header + C test program
- v1.1.0: C FFI layer for cross-language integration

## v1.0.0 (2026-08-22) — STABLE RELEASE

- Semver-stable API: `dfa()`, `acr()`, `analyze()`, `health_check()`
- CLI: `check`, `compare`, `demo`, `report`, `validate`, `self-test`, `codegen`
- Flight-software ready: zero alloc in hot path, `no_std`-compatible core
- CWRU bearing dataset validation (all 3 fault types detected)
- CI matrix: Ubuntu + macOS + Windows
- CITATION.cff, GUARANTEES.md, full rustdoc

## v0.9.x (2026-08-22) — Pre-Release

- v0.9.0: `struktura self-test` — verifies all claims (5/5 pass)
- v0.8.1: Full rustdoc, doc examples, `#[non_exhaustive]`, GUARANTEES.md

## v0.7.x (2026-08-22) — Empirical Proof

- v0.7.0: `--shuffle` flag + `prove_structure()` + ShuffleProof type
- v0.7.2: `bootstrap_alpha()` — confidence intervals on alpha
- v0.7.3: `split_half_validate()` — consistency check
- v0.7.4: PartialEq for StructuralLaw

## v0.6.x (2026-08-22) — Robustness

- v0.6.0: NaN/Inf sanitization + constant-signal Abstain
- v0.6.2: MSRV rust-version = "1.70"
- v0.6.3: `struktura validate` command
- v0.6.6: CI matrix (Ubuntu + macOS + Windows)

## v0.5.x (2026-08-22) — Flight-Software Ready

- v0.5.0: Multi-file batch analysis
- v0.5.1: `--threshold` flag
- v0.5.2: Column-aware CSV parser
- v0.5.4: `struktura report` — markdown output

## v0.4.x (2026-08-21-22) — Hardening

- v0.4.0: serde feature flag (optional)
- v0.4.1: Display impls + space-first README
- v0.4.3: stdin support (`cat data.csv | struktura check -`)
- v0.4.4: `--quiet` flag
- v0.4.5: `--csv` output mode
- v0.4.7: Default for SlidingWindow/BaselineTracker
- v0.4.8: From conversions for SlidingWindow

## v0.3.x (2026-08-21) — Demo Experience

- v0.3.0: `struktura demo` + visual ASCII bars + builtin CWRU data
- v0.3.0: SlidingWindow + BaselineTracker for real-time monitoring
- v0.3.2: `--json` output + `version` command

## v0.2.0 (2026-08-21) — CLI Binary

- `struktura check` and `struktura compare` commands
- CSV and one-value-per-line file support

## v0.1.0 (2026-08-21) — Initial Release

- Core library: `dfa()`, `acr()`, `analyze()`, `health_check()`
- 7 unit tests + 1 doctest
- Benchmark example with shuffle controls
- Published to crates.io
