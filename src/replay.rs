//! Re-run a detector on a saved case, diff the output against the saved
//! incidents, and report what changed.
#![cfg(feature = "std")]

use crate::autopilot::{AutoPilot, Event};
use crate::case::Case;
use crate::context::ContextTimeline;
use crate::incident::{Incident, IncidentBuilder};
use crate::monitor::{HybridMonitor, Leg, MonitorExport};

/// Alarm-tick tolerance and minimum channel overlap for treating a replayed
/// incident as "the same" incident as one in the saved case.
const TICK_TOLERANCE: i64 = 5;

/// What changed inside one matched incident pair: evidence added/removed
/// between the saved run and the replay, and which channels appear in the
/// new incident but not the old one.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct EvidenceChange {
    pub incident_id: u64,
    pub added_evidence: usize,
    pub removed_evidence: usize,
    /// Evidence items present on both sides at the same `(channel, leg,
    /// tick)` whose `observed` or `threshold` value differs by more than a
    /// relative tolerance (see [`value_differs`]). A changed threshold or
    /// observed value with an unchanged key would otherwise report zero
    /// changes (Finding 4).
    pub value_changes: usize,
    pub channel_diff: Vec<usize>,
}

/// What changed between a case's saved incidents and a fresh replay.
#[derive(Debug, Clone, Default)]
pub struct ReplayDiff {
    /// (saved incident id, new incident id) pairs judged to be the same
    /// real-world event.
    pub matched: Vec<(u64, u64)>,
    /// Saved incident ids with no corresponding new incident.
    pub missed: Vec<u64>,
    /// New incident ids with no corresponding saved incident.
    pub new_alarms: Vec<u64>,
    /// (saved incident id, new_start_tick - saved_start_tick) for every
    /// matched pair.
    pub timing_deltas: Vec<(u64, i64)>,
    /// Evidence-level diff for every matched pair (see [`EvidenceChange`]).
    pub evidence_changes: Vec<EvidenceChange>,
    /// (saved, current) `recording.csv` content fingerprints — set only
    /// when the case's `config.json` has a saved `input_hash` and it
    /// differs from the current recording file (Finding 5).
    pub fingerprint_mismatch: Option<(String, String)>,
    /// `(res_thr, dfa_thr, cusum_thr)` read back from the case's saved
    /// `config.json` `monitor_export`, when present and parseable
    /// (Finding 5's "saved configuration").
    pub saved_thresholds: Option<(f64, f64, f64)>,
    /// `(res_thr, dfa_thr, cusum_thr)` from this replay's fresh
    /// recalibration, for side-by-side comparison against
    /// `saved_thresholds`.
    pub fresh_thresholds: (f64, f64, f64),
}

impl ReplayDiff {
    pub fn summary(&self) -> String {
        let max_delta = self
            .timing_deltas
            .iter()
            .map(|(_, d)| *d)
            .max_by_key(|d| d.abs());
        let delta_text = match max_delta {
            Some(d) if d >= 0 => format!("+{}", d),
            Some(d) => format!("{}", d),
            None => "0".to_string(),
        };
        let changed = self
            .evidence_changes
            .iter()
            .filter(|c| c.added_evidence > 0 || c.removed_evidence > 0 || c.value_changes > 0)
            .count();
        format!(
            "{} matched, {} missed, {} new, {} evidence-changed, max timing delta {} ticks",
            self.matched.len(),
            self.missed.len(),
            self.new_alarms.len(),
            changed,
            delta_text
        )
    }

    pub fn to_json(&self) -> String {
        let mut out = String::from("{");
        out.push_str("\"matched\":[");
        for (i, (s, n)) in self.matched.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&format!("[{},{}]", s, n));
        }
        out.push_str("],\"missed\":[");
        for (i, id) in self.missed.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&format!("{}", id));
        }
        out.push_str("],\"new_alarms\":[");
        for (i, id) in self.new_alarms.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&format!("{}", id));
        }
        out.push_str("],\"timing_deltas\":[");
        for (i, (id, d)) in self.timing_deltas.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&format!("[{},{}]", id, d));
        }
        out.push_str("],\"evidence_changes\":[");
        for (i, c) in self.evidence_changes.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            let channels = c
                .channel_diff
                .iter()
                .map(|ch| ch.to_string())
                .collect::<Vec<_>>()
                .join(",");
            out.push_str(&format!(
                "{{\"incident_id\":{},\"added_evidence\":{},\"removed_evidence\":{},\"value_changes\":{},\"channel_diff\":[{}]}}",
                c.incident_id, c.added_evidence, c.removed_evidence, c.value_changes, channels
            ));
        }
        out.push_str("]}");
        out
    }
}

/// Run the full calibrate -> detect -> group-into-incidents -> attach
/// context pipeline shared by `investigate`, `case save`, and `replay`.
///
/// Calibrates a fresh [`HybridMonitor`] on `cols[..baseline]`, streams the
/// remaining rows through [`AutoPilot`] (validity derived per-row from the
/// data: a channel is invalid at a tick when its value is NaN or
/// infinite), stamping each alarm with the recording-row index it was
/// observed at (not AutoPilot's or a recalibrated candidate's own local
/// tick counter — see Finding 1), and groups the resulting alarms into
/// [`Incident`]s already ticked against the *full* recording — matching
/// how `timeline` (e.g. a context sidecar) ticks its events.
///
/// `cols` is column-major: one `Vec<f64>` per channel, all the same
/// length. Returns the finalized incidents plus the calibrated monitor's
/// exported constants (for callers that persist the calibration, e.g.
/// `case save`).
pub fn run_investigation(
    cols: &[Vec<f64>],
    baseline: usize,
    timeline: &ContextTimeline,
) -> Result<(Vec<Incident>, MonitorExport), String> {
    let channels = cols.len();
    if channels == 0 {
        return Err("case recording has no channels".to_string());
    }
    let samples = cols[0].len();
    if baseline >= samples {
        return Err(format!(
            "baseline_samples {} >= recording samples {}",
            baseline, samples
        ));
    }

    let calib: Vec<Vec<f64>> = cols.iter().map(|c| c[..baseline].to_vec()).collect();

    // Finding 3: HybridMonitor::calibrate has no NaN guard of its own — a
    // channel that's mostly non-finite in the calibration window silently
    // poisons its thresholds with NaN instead of failing loudly. Refuse to
    // calibrate in that case, naming the channel.
    for (ci, c) in calib.iter().enumerate() {
        let finite = c.iter().filter(|v| v.is_finite()).count();
        if finite * 2 < c.len() {
            return Err(format!(
                "channel {} has too many non-finite values in the calibration window ({}/{} finite)",
                ci,
                finite,
                c.len()
            ));
        }
    }

    let monitor = HybridMonitor::calibrate(&calib)
        .ok_or_else(|| "calibration failed on baseline window".to_string())?;
    let export = monitor.export();
    let mut autopilot = AutoPilot::new(monitor);
    let mut builder = IncidentBuilder::default();

    for t in baseline..samples {
        let row: Vec<f64> = (0..channels).map(|c| cols[c][t]).collect();
        let valid: Vec<bool> = (0..channels).map(|c| cols[c][t].is_finite()).collect();
        for mut event in autopilot.push(&row, &valid) {
            // Finding 1: tick alignment through recalibration. Every tick
            // AutoPilot/HybridMonitor attaches to an event is local to
            // whichever monitor is currently active — and a recalibrated
            // candidate's own counter restarts at zero when it's
            // calibrated, partway through the recording. Stamp every event
            // with the recording-row index `t` we're already iterating on,
            // directly at the source, instead of trying to offset
            // candidate-local ticks after the fact.
            match &mut event {
                Event::Alarm { tick, report, .. } => {
                    *tick = t as u64;
                    report.tick = t as u64;
                }
                Event::Quarantined { tick, .. }
                | Event::AdaptationStarted { tick }
                | Event::Recalibrated { tick } => *tick = t as u64,
                Event::RolledBack { tick, guard_report } => {
                    *tick = t as u64;
                    guard_report.tick = t as u64;
                }
            }
            if let Event::Alarm { report, .. } = &event {
                builder.push_alarm(report);
            }
            builder.push_event(&event);
        }
    }

    // Ticks are recording-row indices from the source above, so no
    // post-hoc offset_ticks() is needed (and would double-shift them).
    let mut incidents = builder.finalize();
    crate::incident::attach_context(&mut incidents, timeline);
    Ok((incidents, export))
}

/// Re-run the detector on a case's recording and diff the result against
/// what was saved.
///
/// Calibrates a fresh `HybridMonitor` on the recording's first
/// `baseline_override` samples (or the case manifest's `baseline_samples`
/// if not given), streams the rest through `AutoPilot`, and groups the
/// resulting alarms into incidents the same way the original case was
/// built.
pub fn replay(
    case: &Case,
    baseline_override: Option<usize>,
) -> Result<(Vec<Incident>, ReplayDiff), String> {
    let baseline = baseline_override.unwrap_or(case.manifest.baseline_samples);
    let cols = case.recording()?;

    // A saved case has no context sidecar of its own; replay compares pure
    // detector output against what was saved.
    let timeline = ContextTimeline::new();
    let (new_incidents, export) = run_investigation(&cols, baseline, &timeline)?;

    let saved_incidents = case.incidents()?;
    let mut diff = diff_incidents(&saved_incidents, &new_incidents);
    diff.fresh_thresholds = (export.res_thr, export.dfa_thr, export.cusum_thr);

    // Finding 5: replay used to recalibrate from scratch and never look at
    // what was actually saved. Read config.json back, surface its
    // monitor_export as the "saved configuration" alongside the fresh one
    // above, and warn if recording.csv itself has drifted since the case
    // was saved.
    if let Ok(config_json) = case.config_json() {
        diff.saved_thresholds = crate::case::parse_config_monitor_thresholds(&config_json);
        if let Some(saved_hash) = crate::case::parse_config_input_hash(&config_json) {
            let recording_path = case.dir().join("recording.csv");
            if let Ok(bytes) = std::fs::read(&recording_path) {
                let current_hash = crate::case::fingerprint_content(&bytes);
                if current_hash != saved_hash {
                    eprintln!(
                        "warning: recording.csv has changed since this case was saved (saved: {}, current: {})",
                        saved_hash, current_hash
                    );
                    diff.fingerprint_mismatch = Some((saved_hash, current_hash));
                }
            }
        }
    }

    Ok((new_incidents, diff))
}

/// Greedily match each saved incident to the closest (by start tick) new
/// incident within `TICK_TOLERANCE` ticks that shares at least one channel
/// and has not already been claimed by an earlier saved incident.
fn diff_incidents(saved: &[Incident], new: &[Incident]) -> ReplayDiff {
    let mut matched = Vec::new();
    let mut timing_deltas = Vec::new();
    let mut evidence_changes = Vec::new();
    let mut used_new = vec![false; new.len()];

    for s in saved {
        let mut best: Option<(usize, i64)> = None;
        for (i, n) in new.iter().enumerate() {
            if used_new[i] {
                continue;
            }
            let delta = n.start_tick as i64 - s.start_tick as i64;
            if delta.abs() > TICK_TOLERANCE {
                continue;
            }
            let overlaps = s
                .channels_involved
                .iter()
                .any(|c| n.channels_involved.contains(c));
            if !overlaps {
                continue;
            }
            let better = match best {
                Some((_, bd)) => delta.abs() < bd.abs(),
                None => true,
            };
            if better {
                best = Some((i, delta));
            }
        }
        if let Some((i, delta)) = best {
            used_new[i] = true;
            matched.push((s.id, new[i].id));
            timing_deltas.push((s.id, delta));
            evidence_changes.push(evidence_change(s.id, s, &new[i]));
        }
    }

    let matched_saved: Vec<u64> = matched.iter().map(|(sid, _)| *sid).collect();
    let missed = saved
        .iter()
        .map(|s| s.id)
        .filter(|id| !matched_saved.contains(id))
        .collect();
    let new_alarms = new
        .iter()
        .enumerate()
        .filter(|(i, _)| !used_new[*i])
        .map(|(_, n)| n.id)
        .collect();

    ReplayDiff { matched, missed, new_alarms, timing_deltas, evidence_changes, ..Default::default() }
}

/// Diff a matched pair's evidence: what's in `new` but not `old` (added),
/// what's in `old` but not `new` (removed) — keyed by (channel, leg, tick)
/// since evidence is otherwise just a snapshot of an `AlarmReport` — which
/// evidence items share a key but have a changed `observed`/`threshold`
/// value (Finding 4 — a same-key comparison alone reports zero changes for
/// those), plus which channels appear in `new` but not `old`.
fn evidence_change(saved_id: u64, old: &Incident, new: &Incident) -> EvidenceChange {
    let old_keys: Vec<(usize, Leg, u64)> =
        old.evidence.iter().map(|e| (e.channel, e.leg, e.tick)).collect();
    let new_keys: Vec<(usize, Leg, u64)> =
        new.evidence.iter().map(|e| (e.channel, e.leg, e.tick)).collect();
    let added_evidence = new_keys.iter().filter(|k| !old_keys.contains(k)).count();
    let removed_evidence = old_keys.iter().filter(|k| !new_keys.contains(k)).count();
    let value_changes = old
        .evidence
        .iter()
        .filter(|oe| {
            new.evidence.iter().any(|ne| {
                (oe.channel, oe.leg, oe.tick) == (ne.channel, ne.leg, ne.tick)
                    && (value_differs(oe.observed, ne.observed)
                        || value_differs(oe.threshold, ne.threshold))
            })
        })
        .count();
    let mut channel_diff: Vec<usize> = new
        .channels_involved
        .iter()
        .filter(|c| !old.channels_involved.contains(c))
        .copied()
        .collect();
    channel_diff.sort_unstable();
    EvidenceChange { incident_id: saved_id, added_evidence, removed_evidence, value_changes, channel_diff }
}

/// True if `a` and `b` differ by more than a relative tolerance of `1e-6`
/// (the tolerance floor is 1.0, so near-zero values compare with an
/// absolute tolerance instead of an unstable relative one).
fn value_differs(a: f64, b: f64) -> bool {
    let scale = a.abs().max(b.abs()).max(1.0);
    (a - b).abs() > 1e-6 * scale
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::ContextEvent;
    use crate::incident::Evidence;
    use crate::monitor::Leg;

    fn incident(id: u64, start: u64, end: u64, channels: Vec<usize>) -> Incident {
        Incident {
            id,
            start_tick: start,
            end_tick: end,
            evidence: vec![Evidence {
                channel: channels.first().copied().unwrap_or(0),
                leg: Leg::LevelShift,
                tick: start,
                observed: 4.0,
                threshold: 3.0,
                explanation: "test".to_string(),
            }],
            context: Vec::<ContextEvent>::new(),
            channels_involved: channels,
            resolution: None,
            unresolved: Vec::new(),
        }
    }

    #[test]
    fn matches_within_tolerance_and_overlapping_channels() {
        let saved = vec![incident(0, 100, 105, vec![0])];
        let new = vec![incident(0, 103, 108, vec![0])];
        let diff = diff_incidents(&saved, &new);
        assert_eq!(diff.matched, vec![(0, 0)]);
        assert_eq!(diff.timing_deltas, vec![(0, 3)]);
        assert!(diff.missed.is_empty());
        assert!(diff.new_alarms.is_empty());

        // Evidence ticks differ (100 vs 103) so the single evidence item on
        // each side counts as one added, one removed; same channel set.
        assert_eq!(diff.evidence_changes.len(), 1);
        assert_eq!(diff.evidence_changes[0].incident_id, 0);
        assert_eq!(diff.evidence_changes[0].added_evidence, 1);
        assert_eq!(diff.evidence_changes[0].removed_evidence, 1);
        assert!(diff.evidence_changes[0].channel_diff.is_empty());
    }

    #[test]
    fn beyond_tolerance_is_missed_and_new() {
        let saved = vec![incident(0, 100, 105, vec![0])];
        let new = vec![incident(0, 200, 205, vec![0])];
        let diff = diff_incidents(&saved, &new);
        assert!(diff.matched.is_empty());
        assert_eq!(diff.missed, vec![0]);
        assert_eq!(diff.new_alarms, vec![0]);
    }

    #[test]
    fn no_channel_overlap_does_not_match() {
        let saved = vec![incident(0, 100, 105, vec![0])];
        let new = vec![incident(0, 101, 106, vec![1])];
        let diff = diff_incidents(&saved, &new);
        assert!(diff.matched.is_empty());
        assert_eq!(diff.missed, vec![0]);
        assert_eq!(diff.new_alarms, vec![0]);
    }

    #[test]
    fn summary_formats_sign_and_counts() {
        let diff = ReplayDiff {
            matched: vec![(0, 0), (1, 1), (2, 2)],
            missed: vec![3],
            new_alarms: vec![],
            timing_deltas: vec![(0, 2), (1, -1), (2, 5)],
            evidence_changes: vec![],
            ..Default::default()
        };
        assert_eq!(
            diff.summary(),
            "3 matched, 1 missed, 0 new, 0 evidence-changed, max timing delta +5 ticks"
        );
    }

    #[test]
    fn summary_counts_only_evidence_changes_with_real_deltas() {
        let diff = ReplayDiff {
            matched: vec![(0, 0)],
            missed: vec![],
            new_alarms: vec![],
            timing_deltas: vec![(0, 0)],
            evidence_changes: vec![EvidenceChange {
                incident_id: 0,
                added_evidence: 0,
                removed_evidence: 0,
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(diff.summary().contains("0 evidence-changed"));
    }

    #[test]
    fn to_json_is_well_formed() {
        let diff = ReplayDiff {
            matched: vec![(0, 1)],
            missed: vec![2],
            new_alarms: vec![3],
            timing_deltas: vec![(0, -4)],
            evidence_changes: vec![EvidenceChange {
                incident_id: 0,
                added_evidence: 1,
                removed_evidence: 2,
                channel_diff: vec![5],
                ..Default::default()
            }],
            ..Default::default()
        };
        let json = diff.to_json();
        assert!(json.starts_with('{') && json.ends_with('}'));
        assert_eq!(json.matches('{').count(), json.matches('}').count());
        assert_eq!(json.matches('[').count(), json.matches(']').count());
        assert!(json.contains("\"evidence_changes\":[{\"incident_id\":0"));
        assert!(json.contains("\"channel_diff\":[5]"));
    }

    /// End-to-end: build a case the way the rover example flow does
    /// (calibrate on the first 1000 samples, push the remaining 2000 through
    /// AutoPilot via the shared `run_investigation` pipeline), save it,
    /// replay it, and confirm the replay reproduces the same incident
    /// count. Since `Case::incidents()` fully round-trips the saved
    /// incidents (see `case.rs`), the diff should also show every saved
    /// incident matched and nothing missed/new — including tick alignment,
    /// since both sides now go through the same baseline-offset pipeline.
    #[test]
    fn replay_reproduces_incident_count_on_rover_flow_recording() {
        use crate::case::CaseConfig;
        use crate::context::ColumnSchema;
        use crate::telemetry_bench::{inject_fault, synth_spacecraft, CHANNELS};

        let baseline = 1000;
        let total = baseline + 2000;

        let clean = synth_spacecraft(total, 42); // channel-major: [channel][tick]
        let faulted = inject_fault(&clean, "regime_shift", 42);

        let timeline = ContextTimeline::new();
        let (original_incidents, monitor_export) =
            run_investigation(&faulted, baseline, &timeline).expect("investigates");

        // `Case::save` wants row-per-tick.
        let mut full_rows: Vec<Vec<f64>> = Vec::with_capacity(total);
        for t in 0..total {
            full_rows.push((0..CHANNELS).map(|c| faulted[c][t]).collect());
        }

        let names: Vec<String> = (0..CHANNELS).map(|c| format!("ch{}", c)).collect();
        let column_schema =
            ColumnSchema::from_header(&names.iter().map(|s| s.as_str()).collect::<Vec<_>>());
        let config = CaseConfig {
            input_hash: crate::case::fingerprint_content(b"rover_flow test fixture"),
            monitor_export,
            column_schema,
        };

        let dir = std::env::temp_dir().join("struktura_replay_test_rover_flow");
        let _ = std::fs::remove_dir_all(&dir);
        let case = Case::save(&dir, &full_rows, &original_incidents, baseline, "rover_flow", &config)
            .expect("saves");

        let (new_incidents, diff) = replay(&case, None).expect("replays");

        assert_eq!(new_incidents.len(), original_incidents.len());
        assert!(diff.missed.is_empty());
        assert!(diff.new_alarms.is_empty());
        if !original_incidents.is_empty() {
            assert!(original_incidents[0].start_tick >= baseline as u64);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Finding 1 regression test. A `regime_shift` fault across every
    /// channel is exactly AutoPilot's guarded self-recalibration scenario
    /// (level-shift alarm -> candidate collection -> guard window ->
    /// accepted candidate). Before the fix, alarms raised by the accepted
    /// candidate carried the *candidate's own* local tick counter (which
    /// restarts at zero when the candidate is calibrated partway through
    /// the recording), so evidence ticks would jump backward right after
    /// recalibration instead of continuing to track the recording row.
    #[test]
    fn evidence_ticks_stay_recording_row_aligned_through_recalibration() {
        use crate::telemetry_bench::{inject_fault, synth_spacecraft};

        let baseline = 1000;
        let total = baseline + 2000;
        let clean = synth_spacecraft(total, 42);
        let faulted = inject_fault(&clean, "regime_shift", 42);
        let timeline = ContextTimeline::new();

        let (incidents, _export) =
            run_investigation(&faulted, baseline, &timeline).expect("investigates");

        // Confirm this fixture actually exercises recalibration, or the
        // rest of the test proves nothing.
        let recalibrated =
            incidents.iter().any(|inc| inc.context.iter().any(|c| c.kind == "recalibrated"));
        assert!(recalibrated, "fixture did not trigger a recalibration");

        for inc in &incidents {
            for e in &inc.evidence {
                assert!(
                    e.tick >= baseline as u64 && (e.tick as usize) < total,
                    "evidence tick {} out of recording range [{}, {})",
                    e.tick,
                    baseline,
                    total
                );
            }
        }

        // Alarms are pushed in recording order, so ticks across the whole
        // stream must be non-decreasing. A candidate-local tick after
        // acceptance would regress backward here.
        let all_ticks: Vec<u64> =
            incidents.iter().flat_map(|inc| inc.evidence.iter().map(|e| e.tick)).collect();
        let mut sorted = all_ticks.clone();
        sorted.sort_unstable();
        assert_eq!(
            all_ticks, sorted,
            "evidence ticks are not monotonically non-decreasing through recalibration"
        );
    }

    /// Finding 3 regression test: a calibration window that's mostly NaN on
    /// one channel must fail loudly, naming the channel, instead of
    /// silently poisoning that channel's thresholds with NaN.
    #[test]
    fn run_investigation_rejects_mostly_nan_calibration_window() {
        let baseline = 200;
        let samples = 300;
        let mut good = vec![0.0f64; samples];
        for (i, v) in good.iter_mut().enumerate() {
            *v = (i as f64 * 0.01).sin();
        }
        let mut bad = vec![f64::NAN; samples];
        // Less than half the calibration window is finite.
        for v in bad.iter_mut().take(baseline / 4) {
            *v = 1.0;
        }
        let cols = vec![good, bad];
        let timeline = ContextTimeline::new();

        let err = run_investigation(&cols, baseline, &timeline).unwrap_err();
        assert!(err.contains("channel 1"), "error should name the bad channel: {}", err);
    }

    /// Finding 4 regression test: two incidents whose evidence shares the
    /// same `(channel, leg, tick)` key but has a different threshold must
    /// be reported as a value change, not silently ignored.
    #[test]
    fn evidence_change_counts_same_key_different_threshold_as_value_change() {
        let mut old = incident(0, 100, 100, vec![0]);
        old.evidence[0].threshold = 3.0;
        let mut new = incident(0, 100, 100, vec![0]);
        new.evidence[0].threshold = 5.0;

        let change = evidence_change(0, &old, &new);
        assert_eq!(change.added_evidence, 0);
        assert_eq!(change.removed_evidence, 0);
        assert!(change.value_changes > 0, "changed threshold at the same key must count as a value change");
    }

    #[test]
    fn evidence_change_ignores_within_tolerance_float_noise() {
        let mut old = incident(0, 100, 100, vec![0]);
        old.evidence[0].threshold = 3.0;
        let mut new = incident(0, 100, 100, vec![0]);
        new.evidence[0].threshold = 3.0 + 1e-9;

        let change = evidence_change(0, &old, &new);
        assert_eq!(change.value_changes, 0, "sub-tolerance float noise must not count as a value change");
    }
}
