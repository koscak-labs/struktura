# Data Contract — debugger modules (context, incident, case, replay, report)

Version: 0.2 (2026-09-18)

## Shared assumptions

### Timestamps and ticks
- A **tick** is a zero-based sample index (u64). All modules use ticks, not wall-clock time.
- Ticks are **recording-row indices, stamped at the source**: every alarm/event tick is set
  directly to the recording row `run_investigation` is currently iterating on (`t`), not a
  monitor-local or candidate-local counter. A recalibrated candidate's own tick counter
  restarts at zero when it is calibrated partway through the recording, so any tick derived
  from it (rather than stamped from the source loop) would misalign with the recording —
  and with a context sidecar, which is always keyed on the original row number. Grouping
  incidents happens after this stamping, so `Incident` ticks are already recording-aligned.
- If the input has a timestamp column, `ColumnSchema::from_header` detects it by name
  ("time", "timestamp") and assigns `ColumnRole::Timestamp`. It is NOT used as the tick
  index — ticks are row positions. Wall-clock alignment is the caller's responsibility.

### Channel identity
- Channels are identified by their column index in the measurement subset (after filtering
  out Mode/Command/Annotation/Timestamp columns via `ColumnSchema`).
- `Evidence.channel`, `AlarmReport.channel`, and `Incident.channels_involved` all use this
  zero-based measurement-column index.
- Channel names are carried separately (e.g. the CSV header strings, saved to `schema.json`)
  and joined at report time, not embedded in the incident/evidence structs.

### Column roles
- `ColumnRole::Measurement` — fed to HybridMonitor/AutoPilot as f64 time series.
- `ColumnRole::Mode` / `Command` — NOT fed to the detector. Stored in the ContextTimeline
  as ContextEvents for annotation. Context **annotates evidence only** — it does not change
  detection behavior. Mode-aware detection is a future milestone.
- `ColumnRole::Annotation` — free-text, stored as ContextEvents.
- `ColumnRole::Timestamp` — ignored by the detector; available for the report.

### Validity
- `AutoPilot::push(sample, valid)` accepts a per-channel bool slice.
- Validity is derived from the data itself: `valid[c] = cols[c][t].is_finite()` for every
  row past the calibration window — a channel with a NaN/Inf value at a tick is invalid at
  that tick. A future version should also derive validity from context (e.g. a "sensor_off"
  command → false for that channel).

### Imputation policy (calibration window only)
- Detector calibration (`HybridMonitor::calibrate`) has no NaN/Inf guard of its own — even a
  single non-finite value in a calibration window poisons every sum/mean derived from it.
  `run_investigation` imputes every non-finite calibration sample with that channel's mean
  of its own finite calibration samples, preserving row count and alignment (never dropping
  rows).
- **Explicit and recorded**: the number of values imputed per channel is returned from
  `run_investigation` as `(channel_index, imputed_count)` pairs, and a warning is printed to
  stderr for every channel with at least one imputed value: `channel N: M of K calibration
  values imputed with channel mean (X.Y%)`.
- **Bounded**: a channel with *zero* finite calibration samples cannot be imputed from and is
  rejected outright, naming the channel. A channel with **more than 20%** of its calibration
  samples imputed is also rejected — even 20% imputed is already enough to meaningfully
  distort a channel's mean/variance, so the old, more permissive 50% threshold from an
  earlier revision of this policy was tightened.
- **Persisted**: `struktura case save` writes the imputation counts from the original
  investigation into `config.json` under `"imputation"` (see Case directory layout below),
  so `struktura replay` — and anyone reading the saved case — can see how much of the
  original calibration was fabricated from the channel mean rather than observed.

### Context alignment
- `ContextTimeline` is tick-indexed: `ContextEvent.tick` is a row index in the same
  recording. `IncidentBuilder::attach_context` pulls events in `[start_tick - 20, end_tick]`.
- When context is absent (no `--context` sidecar), the timeline is empty and incidents
  carry no context events. The report says nothing about context rather than inventing it.
- When context is stale (the sidecar has fewer rows than the recording), events beyond the
  sidecar's range simply have no context.
- A saved case's context sidecar (`context.json`) is reattached on `struktura replay`, so a
  contextual investigation round-trips through `case save` -> `replay` with the same context
  at the same ticks. Legacy cases with no `context.json` get an empty timeline.

### Reconstructed values
- `HybridMonitor` and `AutoPilot` reconstruct quarantined channels internally via the
  `Reconstructor` (linear LS from the surviving channels).
- The reconstructed values are served by `monitor.virtual_value(ch)` but are NOT stored in
  the incident record. The incident records the Parity alarm that triggered the quarantine.
- A future version should preserve the original measurement alongside the reconstruction
  and flag which values in the output are virtual.

### Malformed CSV rows
- A row with the wrong field count (more or fewer fields than the header) is malformed. The
  entire row is NaN'd across every channel — it is never partially patched (no padding a
  short row, no truncating a long one) — which would otherwise let a bad row masquerade as
  partly-valid data.
- Row count is always preserved (a malformed row becomes a NaN row, not a dropped row), so
  tick alignment with a sidecar context file keyed on the original row number is never
  disturbed by a malformed row elsewhere in the file.
- A row with the *right* field count but a value that doesn't parse as a number (e.g. a
  `mode`/`command` column carrying a string label) is NOT treated as malformed — only that
  one field becomes NaN, and `ColumnSchema` filters non-measurement columns out downstream
  before calibration/detection ever sees them.
- The caller (`prepare_input` in `struktura.rs`) gets back the 0-based data-row indices that
  were malformed and prints a warning naming them.

## Case directory layout

```
cases/<name>/
  manifest.json     — CaseManifest (name, created, version, baseline, channels, samples,
                       input_hash, recording_hash, schema_version)
  recording.csv     — row-per-tick, one column per channel, header "ch0,ch1,..."
  incidents.json    — JSON array of Incident objects
  config.json        — { baseline_samples, detector_version, input_hash, monitor_export,
                       column_schema, imputation }
  context.json      — JSON array of ContextEvent objects (the sidecar timeline, if any)
  schema.json       — JSON array of per-channel display names, in recording-column order
```

- `recording.csv` preserves the original measurements. Reconstructed values are NOT saved.
- `manifest.json` includes `detector_version` (the crate version at save time),
  `samples`/`channels` counts for quick validation without parsing the CSV, an
  `input_hash` (content fingerprint of the original input file), a `recording_hash`
  (content fingerprint of the exact bytes written to `recording.csv` — a different byte
  representation of the same data, used for drift detection on replay), and a
  `schema_version` field (case-file layout version, independent of `detector_version`).
  These four fields are optional on parse for backward compatibility with manifests saved
  before they existed; `schema_version` defaults to `"0.1"` when absent.
- `config.json`'s `monitor_export` is the calibrated detector's full exported state
  (thresholds and every per-channel AR/CUSUM/repeat-detection constant) — not just the
  three global thresholds — so `struktura replay` can diff every field against a fresh
  recalibration, not only spot a global-threshold change.
- Both `input_hash` (the fingerprint) and the FNV-1a-based `fingerprint_content` function
  that computes it are content fingerprints for reproducibility detection, **not**
  cryptographic hashes — no crypto-hash dependency is pulled in, and they must not be used
  for integrity/security purposes.
- A future version should add: reconstructed-value preservation, and independent
  cross-validation of the detector against ground truth beyond the bundled fixtures.

## Config validation on replay

- `struktura replay` treats a case's saved `config.json` as **required and must-parse** for
  any case whose manifest has a `schema_version` — i.e. any case saved by a version of this
  crate that writes `schema_version` at all ("0.1" or later). If `config.json` **exists**
  but fails to parse into a full `MonitorExport` (e.g. it was overwritten with `{}` or is
  otherwise corrupt), replay returns an error: `config.json exists but monitor_export is
  missing or corrupt`. It does not silently skip the saved-configuration comparison.
- Only a **missing** `config.json` file (the file itself absent, e.g. a case saved before
  `config.json` existed) is tolerated: replay warns and skips the saved-configuration
  comparison rather than erroring, since `schema_version` alone cannot distinguish "no
  `config.json` ever written" from "written and later lost."
- `ReplayDiff::config_valid` reports which path was taken: `true` once a saved `config.json`
  was read back into a full `MonitorExport` and compared; `false` when validation failed
  (an error is also returned in that case) or was skipped for a case with a missing
  `config.json`.

## Replay diff semantics

- **matched**: a saved incident and a new incident whose start ticks are within ±5 and
  whose channel sets overlap by at least one channel.
- **missed**: a saved incident with no match in the new run.
- **new**: a new incident with no match in the saved run.
- **timing_delta**: for matched pairs, `new.start_tick - saved.start_tick`.
- **evidence_changes**: for every matched pair, evidence added/removed between the saved run
  and the replay, plus `value_changes` — evidence present on both sides at the same
  `(channel, leg, tick)` whose `observed`/`threshold` differs beyond a relative tolerance —
  and `channel_diff`, channels present in the new incident but not the old one.
- **fingerprint_mismatch**: `Some((saved_hash, current_hash))` when the case's manifest has
  a saved `recording_hash` and a fresh read of `recording.csv` fingerprints differently —
  i.e. the recording file was modified after the case was saved. `None` for an untouched
  case. Diffed against `recording_hash` (the exact bytes `Case::save` wrote), not
  `input_hash` (the original input file's different byte representation of the same data).
- **threshold_diffs**: every saved-vs-fresh field that differs beyond float tolerance across
  the full `MonitorExport` (all three global thresholds plus every per-channel calibration
  field), each entry naming which channel and field.
- **config_valid**: see "Config validation on replay" above.
- Incident grouping changes (one incident splitting into two, or two merging) currently
  surface as missed + new. A future version should detect regrouping explicitly.
- Execution timing and host metadata are NOT compared.

## What this version does NOT provide

- Mode-aware detection (context annotates only)
- Automatic operating-mode discovery
- Independent real-world evaluation (the rover fixtures and the ESA-ADB TimeEval adapter
  are regression tests / an external benchmark harness, not generalization evidence)
- Reconstructed-value preservation in the case file
- Explicit regrouping detection in replay diffs (split/merge surfaces as missed + new)
- Validity derived from context (e.g. a "sensor_off" command does not yet mark a channel
  invalid at that tick — only NaN/Inf does)
