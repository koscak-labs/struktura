# Changelog

All notable changes to Struktura are documented here.

## Unreleased

- `guard` checks its own calibration: it calibrates on the first half of the
  calibration rows and warns when the second half already alarms (the rows it
  assumes are healthy may contain a fault). JSON output gains
  `calibration_suspect_row`. Exit codes are unchanged. NAB, default
  calibration: warned on 6/10 series whose calibration contains a labelled
  anomaly and 2/8 whose calibration is clean; skipped below 1,536 calibration
  rows. Found because guard called NAB's machine-temperature failure series
  healthy (examples/calib_selfcheck_eval.rs).
- The core crate's `wasm` feature and `src/wasm.rs` are removed; JavaScript
  bindings live in `crates/struktura-wasm` (npm tarball on `wasm-v*` releases),
  as the Python ones do in `crates/struktura-py`.

## v1.8.4 (2026-09-24): dfa_short, fair NAB comparison, head-to-head bench, claims cleanup

- Known bug, not fixed in this release: `generate-hybrid` emits DFA box sizes
  16..23 while the Rust calibration uses 16..24 at the 96-sample window, so the
  generated C's DFA α differs from Rust by 0.1-0.2 on average. Found by a
  differential test; fix in progress for 1.8.5.
- Python wheels (Linux, macOS, Windows) as GitHub release `py-v1.8.4`:
  `pip install struktura --find-links https://github.com/koscak-labs/struktura/releases/expanded_assets/py-v1.8.4`.

- Real-data evaluation: `examples/nab_eval.rs` runs `guard` and a limit check
  on the 58 labelled NAB series. Alarms are counted in episodes (alarms less
  than 50 ticks apart count once), the same rule for every detector.
  guard: 36/116 windows, 35 false alarms (0.12 per 1,000 samples); limit
  check: 48/116, 240 (0.80). [Corrected before release: a first version
  counted guard's alarms more leniently than the limit check's and reported
  50 vs 420.]
- New opt-in `MonitorConfig::quiet_drift` / `guard --quiet-drift`: the drift
  (residual-CUSUM) leg clips residuals at 4 sigma and rescales residuals that
  are autocorrelated in calibration. Small effect: NAB 35 -> 33 false alarms,
  36 -> 35 windows. Default behaviour is unchanged.
- New `dfa_short`: DFA for series from about 24 samples, `None` instead of a
  placeholder when a series cannot be measured. On ESA OPS-SAT-AD (529 test
  segments) per-channel |z| gives AUC-ROC 0.943 vs 0.770 for `dfa`
  (examples/opssat_eval.rs). `dfa` documents its 0.5/R² 0 placeholder below 64.
- Alarm explanations: the residual-CUSUM leg no longer says "gradual drift"
  (it fires on steps too); no em dashes in explanation text.
- Python bindings moved out of the core crate into `crates/struktura-py`
  (the `python` feature is gone); core stays `no_std`-friendly and rlib-only.
- Docs: withdrawn slogans ("predict failure before it happens", "85x faster",
  the Voyager AACS "detection") removed from the crate docs, the docs site,
  ogma-template/ and the demo image; old drafts marked superseded.
- `guard` prints a note when calibration is under 768 rows.
- Browser playground runs the real monitor via WebAssembly.
- New example `structure_vs_amplitude`: guard vs a limit check on synthetic
  correlation changes, with controls.
- Claims ledger: controls now assert how many samples were monitored, so a
  control that monitors nothing cannot pass; weekly run includes NAB.

## v1.8.3 (2026-09-24): claims ledger, README rewrite, output cleanup

- New: `docs/claims.tsv` lists every public number with its command, plus
  control rows on healthy data that must stay silent. `scripts/check-claims.sh`
  re-runs them and fails on drift, on a control that alarms, or when a withdrawn
  claim reappears in README.md, src/ or command output. CI runs the fast rows
  on every push and all rows weekly (`.github/workflows/claims.yml`).
- `ims` output and `ImsDemoResult`'s Display no longer say "early warning";
  they report the first alarm and note that an RMS threshold trips earlier.
  `--help` describes `voyager` as a 2021 vs 2022 magnetometer comparison.
- README rewritten: plain descriptions, commands corrected (`guard` is the
  monitor, `when` is changepoint detection), `evolve` table shows the final
  generation (92%) next to the peak (97%).
- CLI output and comments: fewer em dashes, no decorative emoji in examples.
- Cargo description and keywords updated for search (anomaly-detection,
  time-series, hurst, telemetry, predictive-maintenance).
- Release workflow: skips `cargo publish` when the version is already on
  crates.io; binaries are uploaded under per-target names.

## v1.8.2 (2026-09-24) — claims corrections + CRLF text fix

- Withdrawn: the IMS "323 samples early warning" claim from 1.8.1. The first
  monitor alarm also fires when only the healthy rows (0-500) are fed, and the
  confirmed fault (row 925) comes after a plain amplitude threshold (row 701).
- Withdrawn: C-MAPSS detection rates. Healthy-prefix control alarmed on 35/36
  engines.
- `copilot-compare` prints rows only (first alarm, first confirmed fault, first
  threshold trip, threshold definition), with no lead-time claim.
- `ims`, `spacecraft`, `rover`, `voyager`, `heliopause` output reworded to match
  docs/CLAIMS-AUDIT-2026-09-17.md: Voyager 2021 vs 2022 is year-over-year only
  (pre vs during anomaly p=0.52, slices z=1.5); heliopause z=0.6, both
  inconclusive. The rover demo is labelled simulated.
- README "what it detects" table replaced by measured α on bundled data; only
  the bearing row is a normal-vs-fault separation on real data.
- Fix: `text` ignores `\r`, so CRLF and LF checkouts of the same file give the
  same sentence lengths and α (Windows gave mean_len=125 vs 124).

## v1.8.1 (2026-09-18) — Copilot integration: copilot-compare + predictive monitoring pitch

- New CLI command: `struktura copilot-compare <file.csv>` — side-by-side comparison
  of DFA structural health vs boolean amplitude threshold. On IMS bearing data:
  [Corrected 2026-09-24: an earlier line here claimed 323 samples of early
  warning. The first monitor alarm on IMS also fires in the healthy period,
  and the confirmed fault comes after the threshold; claim withdrawn.]
- New document: `docs/PREDICTIVE-RUNTIME-MONITORING.md` — 1-page pitch bridging
  Copilot (reactive, onboard) and ProgPy (predictive, offline) with DFA structural
  health monitoring for autonomous deep-space missions (Artemis/Gateway).
- Suppress dead_code warnings for clean demo screen-sharing.

## v1.8.0 (2026-09-18) — Telemetry debugger: investigate, case save, replay

- New telemetry debugger CLI: `struktura investigate`, `struktura case save`, and
  `struktura replay` — five new modules (`context`, `incident`, `case`, `replay`,
  `report`) built on top of the existing detector.
- Operating context (`--context` sidecar): mode/command/annotation columns are parsed
  into a tick-indexed `ContextTimeline` and attached to incidents for narrative
  context. Context **annotates evidence only** — it does not change detection.
- Incidents: alarms are grouped by temporal proximity into `Incident` records (start/end
  tick, evidence, involved channels, attached context) instead of being reported as
  isolated alarms.
- Case save/replay: `struktura case save` snapshots a recording, its investigation, and
  the full detector configuration into a case directory; `struktura replay` re-runs the
  detector on the saved recording and diffs the result against what was saved —
  recording-fingerprint verification (`fingerprint_mismatch`), full saved-vs-fresh
  configuration comparison (`threshold_diffs`), and matched/missed/new/evidence-changed
  incident diffing.
- Config validation on replay: a case's saved `config.json` is now required to parse for
  any case with a `schema_version` — a `config.json` that exists but is corrupt (e.g.
  overwritten with `{}`) is an error instead of being silently skipped; only a case with
  no `config.json` at all (true legacy format) still warns and skips gracefully.
  `ReplayDiff::config_valid` reports which path was taken.
- NaN/Inf imputation policy for calibration windows: non-finite values are replaced with
  the channel's own finite mean, the count is recorded per channel and printed to
  stderr, and a channel with more than 20% of its calibration samples imputed (previously
  50% — too permissive) is now rejected outright. Imputation counts are saved into a
  case's `config.json` under `"imputation"` so replay can see how much of the original
  investigation's calibration was fabricated.
- Generated CLI fixtures for `investigate`/`case save`/`replay` are checked into CI
  alongside the existing `docs/examples/*.cmd` fixtures.
- ESA-ADB TimeEval adapter: the first Rust algorithm entry in the TimeEval anomaly
  detection benchmark (scores pending upstream evaluation).

## v1.7.3 (2026-09-17) — no_std dependents build; every public number re-verified

- Docs: full claims audit (`docs/CLAIMS-AUDIT-2026-09-17.md`). Corrected: IMS early
  warning ~2 h (was ~105 h); SMAP/MSL F1 0.655 with `smap --ar 0 --dfa` (the earlier
  0.788 could not be reproduced on any commit or flag, `docs/evidence/smap-f1-2026-09-17.md`);
  Voyager and heliopause rows labelled inconclusive by their own subsampling z; Python
  speed ratio labelled as a published reference, not run head-to-head; "flight-ready"
  → compiles clean, not mission-qualified; rover variant stated to have no DFA leg;
  "0 false alarms" scoped to one fixed-seed synthetic stream
  (`docs/evidence/monitor-perf-2026-09-17.md`).
- CLI: `guard` prints an observed/threshold ratio instead of a percentage that was
  calibrated on synthetic filler; `check`/`compare` print a subsampling z instead of a
  shuffle-null confidence; `when` rebuilt as a block-alpha changepoint detector with a
  data-derived noise gate (0 changes on shuffled / AR / 1/f controls, 123K samples
  each); `--col`, `prove`, `when --truth`.
- README CLI examples are generated from `docs/examples/*.cmd` fixtures and checked
  in CI (`scripts/check-examples.sh`, `.github/workflows/examples.yml`).
- GitHub Action (`action.yml`) wrapping `struktura guard`, with a self-test workflow.
- `examples/baseprod_terrain.rs`: ESA BASEPROD terrain blocks, leave-one-traverse-out.
- `[lib] crate-type` reduced to `["lib"]`. Observed in 1.7.2: a dependent with
  `default-features = false` failed to build with three errors (no global memory
  allocator found; `#[panic_handler]` function required; unwinding panics are not
  supported without std) and built cleanly with the reduced list. The C artifacts are
  produced on demand with `cargo rustc --lib --release --crate-type cdylib --crate-type staticlib`.
- New `dfa_scratch(values, &mut [f64])`: the DFA with the profile written into a
  caller-owned slice, no heap required. `dfa_into` now wraps it; a test checks the two
  agree bit for bit.
- CI: a `no_std` job (bare-metal `thumbv7em-none-eabihf` library check plus a
  dependent built with `default-features = false`) and a `c-ffi` job that builds the
  static library and runs `tests/c/test_struktura.c`.
- Evidence for the above: `docs/evidence/2026-09-17-nostd-*.log`.

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
- Dual-channel residual detector (magnitude + train-calibrated variance) — F1 0.655 with `smap --ar 0 --dfa` (an earlier figure of 0.788 could not be reproduced; see v1.7.3 notes)
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
