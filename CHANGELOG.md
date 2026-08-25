# Changelog

All notable changes to Struktura are documented here.

## v1.7.2 (2026-08-25) — Human-Readable Output + Conformal Confidence + Rover FFI

- All alarms now show column names from CSV headers instead of ch0/ch1/ch2
- Conformal confidence (%) on every alarm, changepoint, check, and compare
- `stamp` command: self-certifying CSVs — data carries its own structural fingerprint
- `rover` command + `rover_flight.rs` flight module (6KB, zero-alloc, const-constructible)
- F Prime rover component template (`generate --rover`)
- C FFI exports proven: linked + called from C test program
- no_std build verified with rover_flight
- Shared CSV parser (deduped from 3 copies)
- Package size: 4.9MB → 551KB (excluded 162 NASA CSVs from crate)
- README rebranded: problem-first ("is your data broken?") not math-first
- Stale "98 lines" claims removed from CLI output
- 92 tests pass

## v1.7.0 (2026-08-24) — Guard Mode + Changepoints + 80 Tests

- `guard` command: pipe any CSV and get live anomaly detection with `--watch` tail mode
- `when` command: find WHERE and WHEN structure changed (changepoint detection)
- `nasa` command: zero-setup NASA embedded demo
- Dual-channel residual detector (magnitude + train-calibrated variance) — F1 0.788
- `--help` / `-h` / `help` now works (was broken, returned "Unknown command")
- Doc comments on all major public functions
- 80 tests (up from 73), mutation-tested coverage gaps closed
- CHANGELOG, CITATION.cff, CONTRIBUTING.md all current
- Full 38-command reference table in README
- Preprocessing warning documented (filtering invalidates baselines)

## v1.6.9 (2026-08-24) — Mars Rovers + Autonomous Evolution

- NASA SMAP/MSL Mars rover anomaly benchmark (`struktura smap`) — F1=0.755 zero training
- RED/BLUE adversarial evolution (`struktura redblue`) — coverage 60% → 92% autonomously
- Generational policy optimization (`struktura evolve`)
- Flight-grade streaming hybrid monitor (`monitor.rs`) with multi-rate channels
- Autonomous mission mode (`struktura mission`) — detect → decide → adapt loop
- Prognosis module (`struktura when`, `struktura guard`) — time-to-failure estimation
- Telemetry benchmark: 6 fault types × 20 seeds × null distribution
- `scan` command: auto-classify + trend in one shot
- `watch` command: live monitoring with auto-refresh
- `alert` command: exit-code monitoring for cron/CI/systemd
- `oneline` command: one-line output for logs/Slack
- `batch` command: multi-file CI/CD analysis with `--json`
- `fingerprint` / `dna` command: structural DNA of signals
- `changepoint` module: structural change detection
- Mutation-tested: 80 tests covering core DFA, boundaries, operator directions
- Chrome S logo + social preview banner
- LICENSE-MIT + LICENSE-APACHE files added
- Preprocessing warning documented (filtering invalidates baselines)

## v1.5.0 (2026-08-23) — Spacecraft + Multi-Domain

- Voyager 1 AACS anomaly detection from public NASA data
- Heliopause crossing detection
- Genome sequence analysis (8 chromosomes, R² > 0.99)
- Text rhythm analysis (human vs AI writing)
- Financial market regime detection
- Cardiac HRV analysis
- Multi-fractal DFA (`mfdfa` module)
- Speed benchmark: 85-112x faster than Python nolds

## v1.3.0 (2026-08-23) — no_std Support

- `#![no_std]` with `extern crate alloc` — runs on embedded flight computers
- `std` feature (default on) — existing users unaffected
- `libm` for transcendentals (`ln`, `sqrt`) in no_std mode
- Codegen module gated behind `std` (not needed on embedded)
- FFI uses `core::slice` instead of `std::slice`

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
