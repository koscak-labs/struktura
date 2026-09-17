//! Re-run a detector on a saved case, diff the output against the saved
//! incidents, and report what changed.
#![cfg(feature = "std")]

use crate::autopilot::{AutoPilot, Event};
use crate::case::Case;
use crate::incident::{Incident, IncidentBuilder};
use crate::monitor::HybridMonitor;

/// Alarm-tick tolerance and minimum channel overlap for treating a replayed
/// incident as "the same" incident as one in the saved case.
const TICK_TOLERANCE: i64 = 5;

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
        format!(
            "{} matched, {} missed, {} new, max timing delta {} ticks",
            self.matched.len(),
            self.missed.len(),
            self.new_alarms.len(),
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
        out.push_str("]}");
        out
    }
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

    // `HybridMonitor::calibrate` wants channel-major slices (one inner Vec
    // per channel); `AutoPilot::push` wants row-per-tick (one sample across
    // channels). `cols` is already channel-major, so slice it directly for
    // calibration and build the row-per-tick view separately for push.
    let calib: Vec<Vec<f64>> = cols.iter().map(|c| c[..baseline].to_vec()).collect();
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(samples - baseline);
    for t in baseline..samples {
        rows.push((0..channels).map(|c| cols[c][t]).collect());
    }

    let monitor = HybridMonitor::calibrate(&calib)
        .ok_or_else(|| "calibration failed on baseline window".to_string())?;
    let mut autopilot = AutoPilot::new(monitor);
    let mut builder = IncidentBuilder::default();

    let valid = vec![true; channels];
    for row in &rows {
        for event in autopilot.push(row, &valid) {
            if let Event::Alarm { report, .. } = &event {
                builder.push_alarm(report);
            }
            builder.push_event(&event);
        }
    }

    let new_incidents = builder.finalize();
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

    ReplayDiff { matched, missed, new_alarms, timing_deltas }
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
        };
        assert_eq!(diff.summary(), "3 matched, 1 missed, 0 new, max timing delta +5 ticks");
    }

    #[test]
    fn to_json_is_well_formed() {
        let diff = ReplayDiff {
            matched: vec![(0, 1)],
            missed: vec![2],
            new_alarms: vec![3],
            timing_deltas: vec![(0, -4)],
        };
        let json = diff.to_json();
        assert!(json.starts_with('{') && json.ends_with('}'));
        assert_eq!(json.matches('{').count(), json.matches('}').count());
        assert_eq!(json.matches('[').count(), json.matches(']').count());
    }

    /// End-to-end: build a case the way the rover example flow does
    /// (calibrate on the first 1000 samples, push the remaining 2000 through
    /// AutoPilot), save it, replay it, and confirm the replay reproduces the
    /// same incident count. Since `Case::incidents()` fully round-trips the
    /// saved incidents (see `case.rs`), the diff should also show every
    /// saved incident matched and nothing missed/new.
    #[test]
    fn replay_reproduces_incident_count_on_rover_flow_recording() {
        use crate::telemetry_bench::{inject_fault, synth_spacecraft, CHANNELS};

        let baseline = 1000;
        let total = baseline + 2000;

        let clean = synth_spacecraft(total, 42); // channel-major: [channel][tick]
        let faulted = inject_fault(&clean, "regime_shift", 42);

        // `calibrate` wants channel-major slices; `faulted` already is one.
        // `push` and `Case::save` want row-per-tick — build the full
        // recording once, and calibrate off `faulted` directly.
        let calib: Vec<Vec<f64>> = faulted.iter().map(|c| c[..baseline].to_vec()).collect();
        let mut full_rows: Vec<Vec<f64>> = Vec::with_capacity(total);
        for t in 0..total {
            full_rows.push((0..CHANNELS).map(|c| faulted[c][t]).collect());
        }

        let monitor = HybridMonitor::calibrate(&calib).expect("calibrates");
        let mut autopilot = AutoPilot::new(monitor);
        let mut builder = IncidentBuilder::default();
        let valid = vec![true; CHANNELS];
        for row in &full_rows[baseline..] {
            for event in autopilot.push(row, &valid) {
                if let Event::Alarm { report, .. } = &event {
                    builder.push_alarm(report);
                }
                builder.push_event(&event);
            }
        }
        let original_incidents = builder.finalize();

        let dir = std::env::temp_dir().join("struktura_replay_test_rover_flow");
        let _ = std::fs::remove_dir_all(&dir);
        let case = Case::save(&dir, &full_rows, &original_incidents, baseline, "rover_flow")
            .expect("saves");

        let (new_incidents, diff) = replay(&case, None).expect("replays");

        assert_eq!(new_incidents.len(), original_incidents.len());
        assert!(diff.missed.is_empty());
        assert!(diff.new_alarms.is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
