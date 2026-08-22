# Struktura — Tasks

## Phase 1: Core Library
- [x] T001 Implement DfaResult struct with alpha and r_squared fields
- [x] T002 Implement dfa() function with box-based DFA algorithm per FR-001
- [x] T003 Implement acr() function with autocorrelation decay per FR-002
- [x] T004 Implement StructuralLaw struct per FR-003
- [x] T005 Implement analyze() combining DFA + ACR + statistics per FR-003
- [x] T006 Implement LawQuality enum with classification thresholds per FR-005
- [x] T007 Implement HealthVerdict enum with from_shift() per FR-004
- [x] T008 Implement health_check() function per FR-004
- [x] T009 Implement internal linreg() helper for log-log regression
- [x] T010 Implement internal clamp() helper

## Phase 2: Tests
- [x] T011 Test: white noise DFA alpha near 0.5
- [x] T012 Test: brownian motion DFA alpha near 1.5
- [x] T013 Test: deterministic (same input = same output)
- [x] T014 Test: too-short input returns alpha=0.5, R2=0.0
- [x] T015 Test: analyze() returns non-Insufficient quality for 2048 samples
- [x] T016 Test: health verdict thresholds (Healthy/Watch/Warning/Critical)
- [x] T017 Test: ACR detects correlation in brownian motion

## Phase 3: Benchmark Example
- [x] T018 Create examples/benchmark.rs with synthetic signals per FR-007
- [x] T019 Include shuffle controls proving structure is real per FR-007

## Phase 4: CI and Publishing
- [x] T020 Create .github/workflows/ci.yml per FR-009
- [x] T021 Create .github/workflows/docs.yml for GitHub Pages per FR-010
- [x] T022 Publish to crates.io as struktura v0.1.0 per SC-005

## Phase 5: Documentation
- [x] T023 Write viral README with proof tables per FR-008
- [x] T024 Create mdBook docs site per FR-010
- [x] T025 Document API (dfa, acr, analyze, health_check) per FR-003
- [x] T026 Document bearing fault detection tutorial
- [x] T027 Document cross-domain proof
