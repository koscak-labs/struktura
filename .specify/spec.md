# Struktura — Feature Specification

## Overview
Zero-dependency Rust crate for universal anomaly detection via Detrended Fluctuation Analysis (DFA).

## Functional Requirements

### FR-001: DFA Computation
The crate MUST compute the DFA scaling exponent alpha and R-squared for any input slice of f64 values with N >= 64.

### FR-002: Autocorrelation Decay
The crate MUST compute autocorrelation decay exponent and R-squared for input with N >= 20.

### FR-003: Full Structural Analysis
The `analyze()` function MUST return a StructuralLaw containing: hurst, dfa result, acr result, mean, std_dev, kurtosis, p99, max, n, and quality classification.

### FR-004: Health Verdict
The `health_check()` function MUST compare current DFA alpha against a baseline and return Healthy/Watch/Warning/Critical based on shift magnitude.

### FR-005: Law Quality Classification
Quality MUST be classified as Exact (R2>0.95), Strong (R2>0.85), Good (R2>0.7), Approx (R2>0.3), or Abstain (R2<=0.3). Insufficient for N<20.

### FR-006: Exact-or-Abstain Guarantee
The crate MUST never report a health verdict without also reporting R-squared. If R2 < 0.3, quality MUST be Abstain.

### FR-007: Benchmark Example
A runnable example (`cargo run --example benchmark`) MUST demonstrate cross-domain analysis with shuffle controls.

### FR-008: Cross-Domain Proof
The README MUST contain verified result tables for at least 4 domains (spacecraft, bearings, genome, cardiac).

### FR-009: CI Pipeline
GitHub Actions MUST run tests, clippy, and the benchmark example on every push and PR.

### FR-010: Documentation Site
GitHub Pages MUST serve mdBook documentation at koscak-labs.github.io/struktura.

## Success Criteria

### SC-001: All 7 unit tests pass
### SC-002: Benchmark example runs without error
### SC-003: cargo publish --dry-run succeeds
### SC-004: CI workflow passes on GitHub Actions
### SC-005: Published on crates.io as struktura v0.1.0

## User Stories

### US1: Rust Developer Monitoring Bearings
As a Rust developer monitoring industrial bearings, I want to `cargo add struktura` and detect faults from vibration CSV data with 5 lines of code.

**Acceptance Criteria:**
- US1/AC1: `cargo add struktura` resolves from crates.io
- US1/AC2: `analyze(&data)` returns DFA alpha that distinguishes normal from faulted bearings
- US1/AC3: `health_check(&law, baseline)` returns Critical for inner race faults

### US2: Flight Software Engineer
As a flight software engineer, I want to evaluate Struktura for onboard telemetry health monitoring.

**Acceptance Criteria:**
- US2/AC1: The README documents computational cost (MACs per channel, RAM per channel)
- US2/AC2: The crate has zero external dependencies
- US2/AC3: DFA computation works on 256-sample windows (minimum viable for onboard use)
