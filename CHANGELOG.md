# Changelog

All notable changes to Struktura are documented here.

## Unreleased

- `guard`: a channel that is constant in the calibration rows (a valve, a
  status flag) has no noise to measure against, so its residual scale sits
  at the 1e-9 floor and any change alarms. That part is deliberate; what was
  wrong is that guard said nothing at calibration and then printed the
  floor's artifact as the alarm's size ("155402461.9x threshold" for a
  valve going 0 -> 1). guard now names such channels at calibration, says a
  `--baseline` covering their normal changes avoids it, and prints
  "changed; constant in its baseline" instead of the ratio (JSON adds
  `"baseline_constant":true`). With a baseline that covers the switching,
  the same valve raises nothing. Ledger row guard-constant-baseline-channel.
  The monitor and the generated C are unchanged. (Audit finding.)
- CSV input (`guard`, `watch`, `report`), two audit findings:
  - A quoted field containing the delimiter (a header like `"Temp, C"`)
    was split in two, so every later name landed on the wrong column: a
    dead `pressure` channel was reported as `C`. Fields now split only
    outside quotes. When a header still has more fields than the data rows,
    channels are numbered (`ch0`, `ch1`, ...) with a warning instead of
    named wrongly.
  - A time column in epoch nanoseconds (~1.7e18) repeats values as f64
    (256 apart), so it was not "always increasing", got monitored, and
    raised three false faults and a "declared dead". A time-named column now
    only has to never decrease and rise overall.
  Ledger rows guard-quoted-header-names and
  control-epoch-ns-timestamp-ignored, and a unit test, each failing before
  the fix. `guard` output is unchanged on all 16 bundled CSVs.
- `analyze()` and `acr()` no longer depend on the units of the data. Both
  treated a series as constant when its spread was below a fixed 1e-12
  (1e-15 for `acr`'s sum of squares), so a real signal recorded in small
  units (a strain gauge in SI, a displacement in metres) got
  `LawQuality::Abstain`. The check is now relative: exactly constant, or a
  standard deviation below 1e-12 of the mean's magnitude. Test
  analyze_does_not_depend_on_units (the same Brownian series scaled by
  1e-15, 1e-9 and 1e9 gives the same quality, α and ACR exponent) fails
  without the fix at 1e-15. Bundled data, claims ledger and examples are
  unchanged. Two unit-dependent thresholds remain: the pivot check in the
  monitor's parity least squares and one CLI variance gate.
- Generated hybrid C, two checks by the github-profile-space-robotics
  session (harness and logs in its proofs/esa-adb-official/hybrid-diff):
  - On ESA-ADB Mission 1 channels 41-46 (7,364,161 rows, one continuous
    stream, reset after every alarm) the C and Rust alarm sequences are
    identical before and after the phase-counter change: 778,977 alarms,
    0 differences, while the ring phase wraps 399 times. A 10% change of
    the C residual threshold gives 471,999 differences.
  - With every channel started at 2^32 - 1000 samples, the C from before
    the 64-bit counters, built for a 32-bit ABI (`-m32`), missed a residual
    alarm pair at the sentinel tick and another at the wrap tick, and
    raised level alarms up to 96 samples late after the wrap. The current C
    at `-m32` and `-m64`, and the old C at `-m64`, match their no-wrap runs
    on all five planted faults.
  - Built for 32-bit x86 with default x87 math, the C raised 199 alarms
    where Rust and SSE math raise 198. The generated header now says to add
    `-msse2 -mfpmath=sse` there. Cortex-M builds use IEEE double and are not
    affected.
- Claims corrected to what the code and data show (audit findings, plus
  what checking them turned up):
  - `struktura demo` printed "The bearing's vibration structure changed
    BEFORE any amplitude threshold would have fired". The demo compares two
    separate recordings, so it shows no timing, and on the IMS run-to-failure
    bearing a plain RMS threshold trips first. It now prints the RMS
    amplitude of both recordings (0.081 -> 0.090, +11%, pinned in the
    claims ledger) next to α (0.689 -> 0.183). The same claim is gone from
    docs/src/dfa.md and USE_CASES.md, whose bearing paragraph also said α
    "increases toward failure" (on the bundled data it drops).
  - `dfa()` and `dfa_scratch()` return the placeholder below 72 samples,
    not 64: from 64 to 71 `dfa_box_sizes` gives fewer than 3 box sizes. The
    docs say so; a test pins the boundary.
  - GUARANTEES.md said there is no unsafe code; the C interface has 8
    `unsafe extern "C"` functions and one `static mut`.
  - src/genome.rs quoted human-vs-chimp α values that cannot be reproduced
    (the chimp file in data/genome is an HTML 404 page); USE_CASES.md claimed
    8 chromosomes with R² > 0.99. Both now quote the ledgered chr1 result
    (α 0.967, R² 0.980). USE_CASES.md no longer claims that GNSS monitoring
    catches degradation before the receiver (the example is synthetic) or
    that AI text has a different structure (not measured).
  - bench/compare/RESULTS.md: 10 of 58 NAB series timed out for the
    isolation forest (the run log has 10), not 11. README: `--help` does not
    list every command. CITATION.cff said 1.8.0; llms.txt said 1.7.2 and
    linked a `main` branch that does not exist.
- Python and WebAssembly bindings (audit findings):
  - wasm `Monitor::new` and `Guard::new` panicked on empty calibration data
    (`RuntimeError: unreachable` in JS) instead of returning their documented
    error; they now return it. Python already raised `ValueError` there.
  - `Monitor.push` in both bindings fed NaN into the detectors instead of
    treating it as a missing reading, as `Guard.push` does. NaN and +/-inf
    now go through `push_with_validity`, so the missingness leg reports them.
  Tests in `crates/struktura-wasm/tests/smoke.mjs` and the Python smoke test
  in `python-wheels.yml`, each failing without its fix (wasm:
  `RuntimeError: unreachable`; NaN: silence in wasm, `None` in Python).
- AutoPilot during a recalibration (found by the six-dimension audit, each
  finding reproduced by an independent verifier):
  - A channel quarantined while a new baseline was collected got its new
    calibration from reconstructed readings, which lack its own noise, so
    its real readings never passed the recovery checks again: quarantined
    for good. The candidate now keeps that channel's previous calibration.
  - A sensor that failed while the baseline was collected raised nothing
    and was calibrated into the new baseline (a stuck run even switched its
    stuck-value check off there). The current monitor now takes whole
    samples with their validity during collection, and a stuck, missing or
    inconsistent sensor ends the collection and is quarantined. A sensor
    failure on the candidate during its trial is quarantined too, instead
    of being reported as an unstable new regime.
  - The current monitor was not fed during the 300-sample trial, so after a
    rollback it resumed with rings and previous values up to 300 samples
    stale, and the first sample after it could raise a spurious drift
    alarm. It is now fed throughout.
  Tests: channel_quarantined_through_a_recalibration_still_recovers,
  sensor_failure_during_recalibration_is_quarantined,
  rollback_resumes_from_an_up_to_date_monitor (all three fail without the
  fix). Effects: NAB default and `--quiet-drift` unchanged; `--sensitivity
  high` 57 -> 55 windows, 55 -> 54 false alarms (two borderline drift
  alarms no longer cross their threshold after the corrected state, on
  ambient_temperature_system_failure and ec2_request_latency_system_failure).
  `examples/rover.csv`: the motor current alarm at row 2201 is 1.2x its
  threshold, not 15.1x; the 15.1x came from the stale monitor after the
  rollback at row 2200.
- Generated hybrid C (`generate-hybrid`): new `hyb_push_sample(m, x, &ch)`
  feeds one sample of every channel and returns the first alarm and its
  channel, and new `hyb_reset(m)` clears what `HybridMonitor::reset` clears.
  Driving the C one channel at a time with `hyb_push` latches after an alarm,
  so the channels after the alarming one skipped that sample, the same fault
  the Rust monitor had until 1.8.7. A differential oracle
  (tests/hybrid_c_alarms_match_rust.rs, by the oura-32 session) compares the
  C with the Rust `HybridMonitor` on 240 seeded streams (1-3 channels; white,
  AR, random walk, quantized; step, stuck, variance, drift and spike faults):
  first alarm 0/240 mismatches (tick, channel and leg) through `hyb_push`;
  full alarm sequence with a reset after each alarm 2/240 through per-channel
  `hyb_push`, 0/240 through `hyb_push_sample`; a planted 10% change of the C
  residual threshold gives 58/240. Both new functions are `static inline`, so
  unused they cause no `-Wunused-function` error. The self-test now uses
  them and prints the alarm tick itself (404) and the channel; it printed one
  past the tick (405) before. `hyb_push` is unchanged.
- Generated hybrid C, two audit findings:
  - One NaN reading switched the C's residual CUSUM leg off on that channel
    for good: the clamp `if (x < 0.0) x = 0.0` keeps a NaN, while Rust's
    `max(0.0)` turns it into 0. The clamp is now `if (!(x > 0.0)) x = 0.0`.
    The oracle gained a NaN-then-step fault: 0/240 mismatches with the fix,
    20/240 with the old clamp.
  - The sample counters were `unsigned long`, 32 bits on most
    microcontrollers, so they wrapped after 2^32 samples (50 days at 1 kHz),
    after which the DFA and level legs went back to their 96-sample warm-up,
    the rings jumped position, and the residual leg's "previous hit"
    sentinel could match a real tick. They are now
    `unsigned long long`, as Rust's `u64`. The ring positions come from a
    32-bit phase counter (the tick modulo WINDOW x ROLL x DFA_STRIDE), so
    no 64-bit division is needed: with the 64-bit tick used directly,
    Cortex-M3 `-O2` flash grew from 6,600 to 7,464 bytes (`__udivmoddi4`,
    and 96 software divisions per DFA evaluation). With the phase counter
    it is 6,680 bytes, and RAM 9,568 bytes (was 9,520); QEMU alarms at the
    same tick and leg at every optimization level. One oracle stream is 40,000 samples long, so the phase
    counter wraps in it; the oracle does not detect a wrong phase period
    (tried: 0/240), since write and read positions stay consistent.
- `guard` no longer monitors a column that always increases and is named
  like a time (time, timestamp, epoch, date, clock, t, ts, utc, ...), for
  example epoch nanoseconds with jitter. Only exactly even steps were left
  out before, so a jittered timestamp was monitored and raised most of the
  alarms: on BASEPROD's FOG log 20 of 21 were on `Timestamp`; now none are
  (10 faults, all on orientation channels). A note names the column. An
  always-increasing column with any other name (a counter) is still
  monitored. Ledger row control-jittered-timestamp-ignored. Found by the
  oura-26 session.
- `guard` CSV parsing: a blank or unparseable cell dropped out of its row,
  so every later column shifted left and the last column read 0. A gap in
  one sensor was reported on another (a 10-row gap in column b of an a,b,c
  file: "c: sensor appears stuck, c declared dead"). Now the cell stays in
  its column as a missing reading: calibration fills it from the previous
  reading, and while streaming it goes to the monitor as invalid, so the
  missingness leg reports the right sensor ("b: data stopped arriving").
  A blank cell in the first data row no longer drops that column, and
  `--watch` parses appended lines with the same delimiter and columns as
  the initial file (it used to split on commas and index raw fields).
  Ledger row guard-blank-cell-right-column. Found by the oura-26 session.
- A quarantined sensor can come back. Until now `guard`, `Guard` in
  Python/JS and AutoPilot quarantined a channel on a stuck run, sustained
  missing data or cross-channel inconsistency, and nothing ever lifted it:
  a data gap filled with repeats silenced that sensor for the rest of the
  run (on ESA-ADB, all six channels by 2007, silent for 6.7 years). While
  quarantined, a channel's own readings are now checked against its
  calibration (no stuck run the calibration did not allow, residuals within
  the residual leg's threshold, consistent with the other channels when
  none of them is quarantined); after `RECOVER_SPAN` (192) such samples in
  a row it is monitored again, its rings refilled with its real readings.
  New event `Unquarantined` (CLI: "readings healthy again"; JSON event
  `unquarantine`; Python/JS kind `unquarantined`). A sensor that stays stuck
  is never released (tested). Effects: NAB default 36 -> 45 of 116 windows,
  35 -> 46 false alarms; `--sensitivity high` 49 -> 57 windows, 48 -> 55
  false alarms; `--quiet-drift` 35 -> 44 windows, 33 -> 45 false alarms;
  clean control series still silent. `examples/rover.csv`: the motor current
  sensor comes back at row 2592 after its fault ends, and the scripted
  battery drain from row 2600, missed before, is reported at 2595.
  Reported by the ESA-ADB evaluation. Checked on non-NAB data by the
  oura-26 session: in 40 seeded runs no still-faulty sensor was released,
  and alarms before the first release are identical on 49 files.
  A channel that fails again soon after release waits twice as long the
  next time (`recovery_span`: 192, 384, 768, ... up to 1024x). On
  forward-filled ESA-ADB data, recovery without this cycled 2,568 times
  (2,356 stuck-value alarms); a sensor sticking 300 of every 550 samples
  now cycles at most 6 times in 20,000 samples instead of 31 (tested).
  NAB, the bundled CSVs and the examples are unchanged by the back-off.
- `bench/flight` (embedded-monitor evidence): the bare-metal ARM Cortex-M3
  QEMU build now passes at `-O2`, struktura's own suggested compile line,
  instead of falling back to `-O1`. At plain `-O2` a stack slot in the test
  harness's replay loop is overwritten once GCC's early inlining folds
  `hyb_push()`/`hyb_dfa_alpha()` into it; `-fno-early-inlining` avoids it.
  Whether that is a GCC 13.2.1 code-generation bug is not established (one
  compiler available, no minimized reproducer). The generated monitor shows
  no undefined behaviour under ASan/UBSan natively, or under UBSan on ARM.
  `bench/flight/run.sh` now builds the ARM harness and `size_probe` with
  `-fno-early-inlining`. Alarm tick/leg (1504, REPEATED)
  and determinism now match Rust and native x86 at every ARM optimization
  level, `-O0` through `-O3` and `-Os`; flash at `-O2` is 6600 B (was 7120 B
  at the previous `-O1` fallback), RAM stays 9520 B. See
  `bench/flight/README.md` for the fault registers, disassembly, and flag
  bisection.
- Parity (cross-channel) leg: each channel is now predicted from the other
  channels' values in the same sample, which is how the model is fitted. When
  a full sample was pushed, channels later in the column order still held the
  previous sample, so channels that move together sample by sample (thrust and
  acceleration, for example) looked inconsistent and a healthy sensor could be
  declared dead. On NASA OnAIR's simulated `standby_communication_error_1.csv`
  the false THRUST alarm at row 770 and its quarantine are gone, and the real
  VOLTAGE fault is still caught at row 995. Channels fed one at a time with
  `push_channel` still use each other channel's latest value.
  Output changes elsewhere: on `data/ims_monitor_stream.csv` the drift alarm
  moves from row 926 to 924; on `examples/rover.csv` the motor current parity
  score is 1.7x instead of 1.8x; rows 2 to 4 of the `redblue` table change
  (its headline, 60.0% to 75.0% RED coverage, does not).

## v1.8.7 (2026-09-26): multi-channel monitors no longer skip samples after an alarm

- Multi-channel monitors no longer skip samples after an alarm. When one
  channel alarmed, `HybridMonitor::push` / `push_with_validity` stopped feeding
  the channels after it for that sample, so each alarm followed by `reset()`
  put every later channel one sample behind (its tick counter and rings).
  `guard`, `Guard` in Python/JS and AutoPilot reset after alarms, so a long
  multi-channel run with many alarms could drift far (seen at 3.3M ticks on
  ESA-ADB telemetry). Every channel now ingests every sample and the first
  alarm of the sample is reported, as before. On the CSVs in `data/` the
  `guard` output is unchanged. On `examples/rover.csv` (motor current steps
  at row 2200) a false wheel_rpm alarm at row 2215 is gone, and the faulty
  motor current sensor is quarantined at row 2210 instead of 2211; that
  wheel_rpm alarm came from wheel_rpm running behind after each motor current
  alarm. Found by the ESA-ADB evaluation.
- `guard` and `copilot-compare` on a file shorter than 192 rows exited with a
  panic (slice out of range); they now report "calibration failed" and exit 2.

## v1.8.6 (2026-09-25): Guard in Python and JS, index columns skipped, OPS-SAT-AD

- Python and JS bindings: new `Guard`, the same monitor as `struktura guard`
  (AutoPilot plus the CLI's 50-sample same-leg dedupe). Until now the bindings
  only exposed `Monitor`, which latches after its first alarm, so binding users
  saw one alarm and then silence. The JS `Guard` reproduces
  `struktura guard --baseline 2000 --json` event for event on a test CSV.
  Reaches users with the py-v1.8.6 and wasm-v1.8.6 packages.

- `guard` no longer monitors a column that rises in even steps (a row index or
  a regular timestamp). On a CSV with a `t,value` layout, 1.8.5 monitored `t`,
  raised a fault on it and declared it dead; now it prints a note and the result
  equals the value-only file. Columns with uneven increments (counters) are still
  monitored, and a file's only column is always kept. This includes epoch
  timestamps with sub-second steps (the tolerance scales with the values).
- GitHub Action: `@v1` now moves to a release only after the action has run
  against that release's published binaries on Linux, Windows and macOS
  (download, SHA-256 check, faults found on a known-fault CSV).
- OPS-SAT-AD (ESA satellite telemetry) result in the README and ledger:
  `dfa_short` AUC-ROC 0.943, 0.554 with shuffled samples. Segment
  classification with a nominal reference from the training labels, not the
  streaming `guard`. Re-checked weekly in CI against the SHA-256-pinned data.
- The residual-CUSUM alarm text no longer says "a step or a drift": it also
  fires on a change in how values follow each other (seen on white noise that
  turns AR(0.9) at the same variance). New text: "the signal keeps deviating
  from what its baseline predicts (a step, a drift, or a change in its pattern)".

## v1.8.5 (2026-09-24): generated C matches Rust, --sensitivity high, calibration self-check

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
- `guard --sensitivity normal|high`: `high` sets the threshold design horizon
  to 1e5 clean samples per expected false alarm (default 1e6). NAB (episode
  counting): windows 36 -> 49 of 116, false alarms 35 -> 48 (limit check:
  48, 240). Chosen from a sweep of 1e3..1e7 on NAB itself. Clean synthetic
  slow-wander streams: 2-3/30 false alarms at high, 0/30 at normal.
- Fix: `generate-hybrid` now emits the same DFA box sizes as the Rust DFA
  (shared `dfa_box_sizes`); in 1.8.4 the C α differed by 0.1-0.2 on average,
  worst 0.85 (1.69 α_sd); now max |Δα| 2.9e-11 (tests/hybrid_c_matches_rust.rs,
  found by a differential test; PR #28).
- Fix: `struktura codegen` (standalone C monitor) and `generate --cfs` computed
  DFA on their ring buffer in storage order once it had wrapped, and
  `codegen` used box sizes different from Rust. Against Rust `dfa()` at a
  512-sample window, 1.8.4 matched on 0/1537 windows (max |Δα| 0.51 random
  walk, 0.20 AR(1) 0.9, 0.76 ramp+noise). Now the ring is unrolled into time
  order and `codegen` emits `dfa_box_sizes(window)`: equal within 1e-10 after
  every sample at windows 512, 64 and 72 (tests/c_monitor_matches_rust.rs,
  PR #29). Regenerate DFA monitors made with 1.8.4 or earlier.
- `dfa()` docs: α is noisy between 64 and ~128 samples; use `dfa_short` there.
- Releases publish `SHA256SUMS`; the GitHub Action verifies the downloaded
  binary against it.

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
