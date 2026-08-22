# Struktura — Implementation Plan

## Architecture
- Single-file library crate (src/lib.rs)
- Pure Rust, no dependencies, std required (for sqrt/ln)
- All public types: DfaResult, StructuralLaw, LawQuality, HealthVerdict
- All public functions: dfa(), acr(), analyze(), health_check()
- Internal: linreg(), clamp()

## Data Model
- Input: &[f64] slices (caller owns the data)
- Output: Copy types (no lifetimes, no allocations in return path)
- Vec allocation only inside dfa() for the cumulative profile

## Phases
1. Core library (src/lib.rs) — DFA, ACR, analyze, health_check
2. Tests — 7 unit tests covering white noise, brownian, edge cases, verdicts
3. Benchmark example — examples/benchmark.rs with shuffle controls
4. CI — .github/workflows/ci.yml (test + clippy + benchmark)
5. Docs — mdBook site under docs/
6. README — viral format with proof tables
7. Publish — crates.io as struktura

## Technical Decisions
- Fixed box sizes [16,24,36,54,81,121,181,271] — logarithmically spaced
- ACR lags [1,2,3,5,8,13,21,34,55,89] — Fibonacci-spaced for log coverage
- Health thresholds: <0.03 Healthy, 0.03-0.08 Watch, 0.08-0.15 Warning, >0.15 Critical
- Minimum 3 log-log regression points for valid fit
