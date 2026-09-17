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
            .filter(|c| c.added_evidence > 0 || c.removed_evidence > 0)
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
                "{{\"incident_id\":{},\"added_evidence\":{},\"removed_evidence\":{},\"channel_diff\":[{}]}}",
                c.incident_id, c.added_evidence, c.removed_evidence, channels
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
/// infinite), groups the resulting alarms into [`Incident`]s, then shifts
/// every tick by `baseline` so incidents are ticked against the *full*
/// recording rather than the post-baseline stream — matching how
/// `timeline` (e.g. a context sidecar) ticks its events.
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
    let monitor = HybridMonitor::calibrate(&calib)
        .ok_or_else(|| "calibration failed on baseline window".to_string())?;
    let export = monitor.export();
    let mut autopilot = AutoPilot::new(monitor);
    let mut builder = IncidentBuilder::default();

    for t in baseline..samples {
        let row: Vec<f64> = (0..channels).map(|c| cols[c][t]).collect();
        let valid: Vec<bool> = (0..channels).map(|c| cols[c][t].is_finite()).collect();
        for event in autopilot.push(&row, &valid) {
            if let Event::Alarm { report, .. } = &event {
                builder.push_alarm(report);
            }
            builder.push_event(&event);
        }
    }

    let mut incidents = builder.finalize();
    crate::incident::offset_ticks(&mut incidents, baseline as u64);
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
    let (new_incidents, _export) = run_investigation(&cols, baseline, &timeline)?;

    let saved_incidents = case.incidents()?;
    let diff = diff_incidents(&saved_incidents, &new_incidents);
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

    ReplayDiff { matched, missed, new_alarms, timing_deltas, evidence_changes }
}

/// Diff a matched pair's evidence: what's in `new` but not `old` (added),
/// what's in `old` but not `new` (removed) — keyed by (channel, leg, tick)
/// since evidence is otherwise just a snapshot of an `AlarmReport` — plus
/// which channels appear in `new` but not `old`.
fn evidence_change(saved_id: u64, old: &Incident, new: &Incident) -> EvidenceChange {
    let old_keys: Vec<(usize, Leg, u64)> =
        old.evidence.iter().map(|e| (e.channel, e.leg, e.tick)).collect();
    let new_keys: Vec<(usize, Leg, u64)> =
        new.evidence.iter().map(|e| (e.channel, e.leg, e.tick)).collect();
    let added_evidence = new_keys.iter().filter(|k| !old_keys.contains(k)).count();
    let removed_evidence = old_keys.iter().filter(|k| !new_keys.contains(k)).count();
    let mut channel_diff: Vec<usize> = new
        .channels_involved
        .iter()
        .filter(|c| !old.channels_involved.contains(c))
        .copied()
        .collect();
    channel_diff.sort_unstable();
    EvidenceChange { incident_id: saved_id, added_evidence, removed_evidence, channel_diff }
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
            evidence_changes: vec![
                EvidenceChange { incident_id: 0, added_evidence: 0, removed_evidence: 0, channel_diff: vec![] },
            ],
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
            }],
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
}
