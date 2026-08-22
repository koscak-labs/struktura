# Changelog

## v0.3.0 (2026-08-22)

### Added
- **CLI: `struktura demo`** — runs builtin CWRU bearing fault detection with visual output
- **CLI: `struktura bench`** — markdown table of benchmark results
- **CLI: visual output** — ASCII alpha bars + ANSI colored verdicts (CRITICAL in red)
- **Library: `SlidingWindow`** — circular buffer for real-time incremental DFA
- **Library: `BaselineTracker`** — learns baseline during first N samples, then auto-verdicts
- **Builtin data** — 1000 real CWRU samples each for normal + inner race fault (40KB)

### Changed
- CLI `check` and `compare` commands now show visual alpha bars
- Help text updated with all commands

## v0.2.0 (2026-08-22)

### Added
- **CLI binary** — `struktura check`, `struktura compare` commands
- Reads CSV or one-value-per-line files
- `--baseline` flag for health comparison

## v0.1.0 (2026-08-22)

### Added
- Core library: `dfa()`, `acr()`, `analyze()`, `health_check()`
- Types: `DfaResult`, `StructuralLaw`, `LawQuality`, `HealthVerdict`
- 7 unit tests (white noise, brownian, determinism, edge cases, verdicts)
- Benchmark example with shuffle controls
- Published to crates.io
