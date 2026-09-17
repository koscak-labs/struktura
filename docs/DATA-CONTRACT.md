# Data Contract — debugger modules (context, incident, case, replay, report)

Version: 0.1 (2026-09-17, pre-release)

## Shared assumptions

### Timestamps and ticks
- A **tick** is a zero-based sample index (u64). All modules use ticks, not wall-clock time.
- If the input has a timestamp column, `ColumnSchema::from_header` detects it by name
  ("time", "timestamp") and assigns `ColumnRole::Timestamp`. It is NOT used as the tick
  index — ticks are row positions. Wall-clock alignment is the caller's responsibility.

### Channel identity
- Channels are identified by their column index in the measurement subset (after filtering
  out Mode/Command/Annotation/Timestamp columns via `ColumnSchema`).
- `Evidence.channel`, `AlarmReport.channel`, and `Incident.channels_involved` all use this
  zero-based measurement-column index.
- Channel names are carried separately (e.g. the CSV header strings) and joined at report
  time, not embedded in the incident/evidence structs.

### Column roles
- `ColumnRole::Measurement` — fed to HybridMonitor/AutoPilot as f64 time series.
- `ColumnRole::Mode` / `Command` — NOT fed to the detector. Stored in the ContextTimeline
  as ContextEvents for annotation. In this milestone, context **annotates evidence only** —
  it does not change detection behavior. Mode-aware detection is a future milestone.
- `ColumnRole::Annotation` — free-text, stored as ContextEvents.
- `ColumnRole::Timestamp` — ignored by the detector; available for the report.

### Validity
- `AutoPilot::push(sample, valid)` accepts a per-channel bool slice.
- `investigate` currently passes `valid = [true; channels]` for every row because the
  input is a plain CSV with no separate validity signal. NaN/Inf values are caught by
  `sanitize()` inside the DFA path. A future version should derive validity from the data
  (NaN → false) and from context (e.g. a "sensor_off" command → false for that channel).

### Context alignment
- `ContextTimeline` is tick-indexed: `ContextEvent.tick` is a row index in the same
  recording. `IncidentBuilder::attach_context` pulls events in `[start_tick - 20, end_tick]`.
- When context is absent (no `--context` sidecar), the timeline is empty and incidents
  carry no context events. The report says nothing about context rather than inventing it.
- When context is stale (the sidecar has fewer rows than the recording), events beyond the
  sidecar's range simply have no context.

### Reconstructed values
- `HybridMonitor` and `AutoPilot` reconstruct quarantined channels internally via the
  `Reconstructor` (linear LS from the surviving channels).
- The reconstructed values are served by `monitor.virtual_value(ch)` but are NOT stored in
  the incident record. The incident records the Parity alarm that triggered the quarantine.
- A future version should preserve the original measurement alongside the reconstruction
  and flag which values in the output are virtual.

## Case directory layout

```
cases/<name>/
  manifest.json     — CaseManifest (name, created, version, baseline, channels, samples)
  recording.csv     — row-per-tick, one column per channel, header "ch0,ch1,..."
  incidents.json    — JSON array of Incident objects
  config.json       — { baseline_samples, detector_version }
```

- `recording.csv` preserves the original measurements. Reconstructed values are NOT saved.
- `manifest.json` includes `detector_version` (the crate version at save time) and
  `samples`/`channels` counts for quick validation without parsing the CSV.
- A future version should add: input file hash (SHA-256), full detector configuration
  (MonitorExport), column schema, and a schema version field.

## Replay diff semantics

- **matched**: a saved incident and a new incident whose start ticks are within ±5 and
  whose channel sets overlap by at least one channel.
- **missed**: a saved incident with no match in the new run.
- **new**: a new incident with no match in the saved run.
- **timing_delta**: for matched pairs, `new.start_tick - saved.start_tick`.
- Incident grouping changes (one incident splitting into two, or two merging) currently
  surface as missed + new. A future version should detect regrouping explicitly.
- Execution timing and host metadata are NOT compared.

## What this milestone does NOT provide

- Mode-aware detection (context annotates only)
- Automatic operating-mode discovery
- Independent real-world evaluation (the rover fixtures are regression tests, not
  generalization evidence)
- Reconstructed-value preservation in the case file
- Input hashes or full configuration capture in the manifest
