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
    /// when the case's manifest has a saved `recording_hash` and it
    /// differs from the current recording file (Finding 5 / Issue 2).
    pub fingerprint_mismatch: Option<(String, String)>,
    /// `(res_thr, dfa_thr, cusum_thr)` read back from the case's saved
    /// `config.json` `monitor_export`, when present and parseable
    /// (Finding 5's "saved configuration").
    pub saved_thresholds: Option<(f64, f64, f64)>,
    /// `(res_thr, dfa_thr, cusum_thr)` from this replay's fresh
    /// recalibration, for side-by-side comparison against
    /// `saved_thresholds`.
    pub fresh_thresholds: (f64, f64, f64),
    /// Every saved-vs-fresh field that differs beyond float tolerance
    /// across the *full* `MonitorExport` (all three global thresholds plus
    /// every per-channel calibration field), each entry naming which
    /// channel and field (Issue 4 — `saved_thresholds`/`fresh_thresholds`
    /// alone only ever compared the three global thresholds).
    pub threshold_diffs: Vec<String>,
    /// Whether this case's `config.json` was successfully validated against
    /// the fresh recalibration. `true` when the case has a `schema_version`
    /// of "0.1" or later and `config.json` parsed into a full
    /// `MonitorExport`; `false` when validation failed outright (an error
    /// is also returned in that case — see [`replay`]); absent/unset
    /// (`false`, the `Default`) only for true legacy cases with no
    /// `schema_version` at all, where config validation is skipped.
    pub config_valid: bool,
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
        let mut out = format!(
            "{} matched, {} missed, {} new, {} evidence-changed, max timing delta {} ticks",
            self.matched.len(),
            self.missed.len(),
            self.new_alarms.len(),
            changed,
            delta_text
        );
        // Issue 4: the saved-configuration comparison used to be omitted
        // from the summary entirely.
        if self.fingerprint_mismatch.is_some() {
            out.push_str(", fingerprint mismatch");
        }
        if !self.threshold_diffs.is_empty() {
            out.push_str(&format!(", {} config diff(s)", self.threshold_diffs.len()));
        }
        out
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
        out.push_str("],\"fingerprint_mismatch\":");
        out.push_str(if self.fingerprint_mismatch.is_some() { "true" } else { "false" });
        out.push_str(",\"threshold_diffs\":[");
        for (i, d) in self.threshold_diffs.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&format!("\"{}\"", json_escape_diff(d)));
        }
        out.push_str("],\"config_valid\":");
        out.push_str(if self.config_valid { "true" } else { "false" });
        out.push('}');
        out
    }
}

/// Minimal JSON string escape for the free-text entries in
/// [`ReplayDiff::threshold_diffs`] (mirrors `case::json_escape`, kept local
/// since it's the only string content `to_json` here needs to escape).
fn json_escape_diff(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c => out.push(c),
        }
    }
    out
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
/// length. Returns the finalized incidents, the calibrated monitor's
/// exported constants (for callers that persist the calibration, e.g.
/// `case save`), and per-channel imputation counts as `(channel_index,
/// imputed_count)` pairs (one entry per channel, `0` when nothing was
/// imputed on that channel) — see [`IMPUTATION_REJECT_FRACTION`].
/// Rejection threshold for calibration-window imputation: a channel with
/// more than this fraction of its calibration samples imputed is rejected
/// outright rather than calibrated on mostly-fabricated data. 20%, not the
/// old 50% — even 20% imputed is already enough to meaningfully distort a
/// channel's mean/variance.
const IMPUTATION_REJECT_FRACTION: f64 = 0.2;

pub fn run_investigation(
    cols: &[Vec<f64>],
    baseline: usize,
    timeline: &ContextTimeline,
) -> Result<(Vec<Incident>, MonitorExport, Vec<(usize, usize)>), String> {
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

    let mut calib: Vec<Vec<f64>> = cols.iter().map(|c| c[..baseline].to_vec()).collect();

    // Finding 3 (revisited): HybridMonitor::calibrate has no NaN guard of
    // its own — even a *single* non-finite value in a calibration window
    // poisons every sum/mean derived from it, not just a channel that's
    // mostly non-finite. Impute every non-finite sample with that
    // channel's mean of its own finite calibration samples instead of
    // filtering — this preserves row count and alignment, unlike dropping
    // rows would. A channel with *no* finite calibration samples can't be
    // imputed from and is rejected, naming the channel.
    let mut imputation_counts: Vec<(usize, usize)> = Vec::with_capacity(calib.len());
    for (ci, c) in calib.iter_mut().enumerate() {
        let total = c.len();
        let finite_count = c.iter().filter(|v| v.is_finite()).count();
        if finite_count == 0 {
            return Err(format!(
                "channel {} has no finite values in the calibration window ({} samples)",
                ci, total
            ));
        }
        let mut replaced = 0usize;
        if finite_count < total {
            let finite_sum: f64 = c.iter().filter(|v| v.is_finite()).sum();
            let mean = finite_sum / finite_count as f64;
            for v in c.iter_mut() {
                if !v.is_finite() {
                    *v = mean;
                    replaced += 1;
                }
            }
            let fraction = replaced as f64 / total as f64;
            eprintln!(
                "channel {}: {} of {} calibration values imputed with channel mean ({:.1}%)",
                ci,
                replaced,
                total,
                fraction * 100.0
            );
            if fraction > IMPUTATION_REJECT_FRACTION {
                return Err(format!(
                    "channel {} has {} of {} calibration values ({:.1}%) imputed — \
                     exceeds the {:.0}% imputation rejection threshold",
                    ci,
                    replaced,
                    total,
                    fraction * 100.0,
                    IMPUTATION_REJECT_FRACTION * 100.0
                ));
            }
        }
        imputation_counts.push((ci, replaced));
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
                | Event::Unquarantined { tick, .. }
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
    Ok((incidents, export, imputation_counts))
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

    // Issue 3: a saved case's context sidecar (`context.json`, written by
    // `Case::save`) is reattached here instead of always replaying against
    // an empty timeline — a contextual investigation now round-trips
    // through `case save` -> `replay` with the same context at the same
    // ticks. Legacy cases with no context.json get an empty timeline, same
    // as before.
    let timeline = case.context()?;
    let (new_incidents, export, _imputation_counts) = run_investigation(&cols, baseline, &timeline)?;

    let saved_incidents = case.incidents()?;
    let mut diff = diff_incidents(&saved_incidents, &new_incidents);
    diff.fresh_thresholds = (export.res_thr, export.dfa_thr, export.cusum_thr);

    // Finding 5 / Issue 4: config.json used to be swallowed with
    // `if let Ok(...)`, hiding real read errors. A *missing* config.json is
    // expected for cases saved before it existed (legacy format) and only
    // warns; anything else (corrupt file, permissions) propagates.
    //
    // Fix 1: for any case with a `schema_version` ("0.1" or later — i.e.
    // any case saved by a version of this crate that writes
    // `schema_version` at all), a *present* config.json MUST parse into a
    // full `MonitorExport`. The old `if let Some(saved_export) = ...`
    // silently skipped the whole saved-configuration comparison for a
    // config.json that exists but doesn't parse (e.g. `{}`), which looks
    // identical to "nothing to compare" instead of "this case's saved
    // configuration is corrupt". Only a true legacy case — no
    // `schema_version` at all — still warns and skips gracefully.
    let config_path = case.dir().join("config.json");
    let has_schema_version = !case.manifest.schema_version.is_empty();
    if config_path.exists() {
        let config_json = case.config_json()?;
        diff.saved_thresholds = crate::case::parse_config_monitor_thresholds(&config_json);
        // Issue 4: compare every field of the saved MonitorExport against
        // the fresh recalibration, not just the three global thresholds.
        match crate::case::parse_config_monitor_export(&config_json) {
            Some(saved_export) => {
                diff.threshold_diffs = monitor_export_diffs(&saved_export, &export);
                diff.config_valid = true;
            }
            None if has_schema_version => {
                return Err(
                    "config.json exists but monitor_export is missing or corrupt".to_string()
                );
            }
            None => {
                eprintln!(
                    "warning: legacy case's config.json has no parseable monitor_export; skipping saved-configuration comparison"
                );
            }
        }
    } else {
        // A *missing* config.json (the file itself absent) is the true
        // legacy case Fix 1 leaves alone regardless of `schema_version` —
        // `schema_version` defaults to "0.1" for every manifest (including
        // ones from before `config.json` existed), so it can't distinguish
        // "no config.json ever written" from "config.json written and then
        // lost"; only a config.json that *exists* but fails to parse is
        // treated as corruption.
        eprintln!(
            "warning: case has no config.json (legacy case format); skipping saved-configuration comparison"
        );
    }

    // Finding 5 / Issue 2: `recording.csv` is fingerprinted against
    // `recording_hash` — the fingerprint of the exact bytes `Case::save`
    // wrote to `recording.csv` — not `input_hash`, the original input
    // file's fingerprint. Those are two different byte representations of
    // the same data; diffing against `input_hash` made an untouched case
    // falsely report drift every time.
    if let Some(saved_hash) = &case.manifest.recording_hash {
        let recording_path = case.dir().join("recording.csv");
        if let Ok(bytes) = std::fs::read(&recording_path) {
            let current_hash = crate::case::fingerprint_content(&bytes);
            if &current_hash != saved_hash {
                eprintln!(
                    "warning: recording.csv has changed since this case was saved (saved: {}, current: {})",
                    saved_hash, current_hash
                );
                diff.fingerprint_mismatch = Some((saved_hash.clone(), current_hash));
            }
        }
    }

    Ok((new_incidents, diff))
}

/// Compare a saved case's full `MonitorExport` against a fresh
/// recalibration, noting every threshold or per-channel field that differs
/// beyond float tolerance (Issue 4 — this used to compare only the three
/// global thresholds).
fn monitor_export_diffs(saved: &MonitorExport, fresh: &MonitorExport) -> Vec<String> {
    let mut diffs = Vec::new();
    for (field, s, f) in [
        ("res_thr", saved.res_thr, fresh.res_thr),
        ("dfa_thr", saved.dfa_thr, fresh.dfa_thr),
        ("cusum_thr", saved.cusum_thr, fresh.cusum_thr),
    ] {
        if value_differs(s, f) {
            diffs.push(format!("{}: saved={:.6} fresh={:.6}", field, s, f));
        }
    }

    if saved.channels.len() != fresh.channels.len() {
        diffs.push(format!(
            "channel count: saved={} fresh={}",
            saved.channels.len(),
            fresh.channels.len()
        ));
        return diffs;
    }

    for (ci, (s, f)) in saved.channels.iter().zip(fresh.channels.iter()).enumerate() {
        for (field, sv, fv) in [
            ("ar_a", s.ar_a, f.ar_a),
            ("ar_b", s.ar_b, f.ar_b),
            ("ar_sd", s.ar_sd, f.ar_sd),
            ("alpha_mean", s.alpha_mean, f.alpha_mean),
            ("alpha_sd", s.alpha_sd, f.alpha_sd),
            ("mean", s.mean, f.mean),
            ("roll_thr", s.roll_thr, f.roll_thr),
        ] {
            if value_differs(sv, fv) {
                diffs.push(format!("channel {} {}: saved={:.6} fresh={:.6}", ci, field, sv, fv));
            }
        }
        if s.max_run != f.max_run {
            diffs.push(format!("channel {} max_run: saved={} fresh={}", ci, s.max_run, f.max_run));
        }
        if s.repeat_enabled != f.repeat_enabled {
            diffs.push(format!(
                "channel {} repeat_enabled: saved={} fresh={}",
                ci, s.repeat_enabled, f.repeat_enabled
            ));
        }
    }
    diffs
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
        let (original_incidents, monitor_export, _imputation) =
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
            imputation: vec![],
        };

        let dir = std::env::temp_dir().join("struktura_replay_test_rover_flow");
        let _ = std::fs::remove_dir_all(&dir);
        let case =
            Case::save(&dir, &full_rows, &original_incidents, baseline, "rover_flow", &config, &timeline, &names)
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

        let (incidents, _export, _imputation) =
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

    /// Issue 1 regression test (revised Finding 3): a calibration window
    /// with NO finite values on one channel at all cannot be imputed from
    /// and must fail loudly, naming the channel — the old guard rejected
    /// anything over 50% non-finite, but the fixed behavior only rejects a
    /// channel that's *entirely* non-finite (everything short of that gets
    /// imputed instead, see the tests below).
    #[test]
    fn run_investigation_rejects_calibration_window_with_zero_finite_values() {
        let baseline = 200;
        let samples = 300;
        let mut good = vec![0.0f64; samples];
        for (i, v) in good.iter_mut().enumerate() {
            *v = (i as f64 * 0.01).sin();
        }
        let bad = vec![f64::NAN; samples];
        let cols = vec![good, bad];
        let timeline = ContextTimeline::new();

        let err = run_investigation(&cols, baseline, &timeline).unwrap_err();
        assert!(err.contains("channel 1"), "error should name the bad channel: {}", err);
        assert!(err.contains("no finite values"), "error should say why: {}", err);
    }

    fn sine_channel(samples: usize) -> Vec<f64> {
        (0..samples).map(|i| (i as f64 * 0.01).sin()).collect()
    }

    /// Issue 1: a *single* NaN in a calibration window used to poison every
    /// sum/mean HybridMonitor::calibrate derives from that channel (it has
    /// no NaN guard of its own). It must instead be imputed with the
    /// channel's own finite-sample mean, and calibration must still
    /// succeed with finite calibrated constants.
    #[test]
    fn run_investigation_imputes_single_nan_in_calibration_window() {
        let baseline = 200;
        let samples = 300;
        let good = sine_channel(samples);
        let mut one_nan = sine_channel(samples);
        one_nan[50] = f64::NAN; // inside the calibration window
        let cols = vec![good, one_nan];
        let timeline = ContextTimeline::new();

        let (_incidents, export, _imputation) =
            run_investigation(&cols, baseline, &timeline).expect("a single NaN must not fail calibration");
        assert!(export.channels[1].alpha_mean.is_finite(), "imputed NaN must not propagate into calibration");
        assert!(export.channels[1].ar_a.is_finite());
        assert!(export.channels[1].mean.is_finite());
    }

    /// Issue 1: same as the NaN case above, but for a single `Inf` value —
    /// `is_finite()` (used both by the guard and by imputation) rejects
    /// infinities too, not just NaN.
    #[test]
    fn run_investigation_imputes_single_inf_in_calibration_window() {
        let baseline = 200;
        let samples = 300;
        let good = sine_channel(samples);
        let mut one_inf = sine_channel(samples);
        one_inf[50] = f64::INFINITY; // inside the calibration window
        let cols = vec![good, one_inf];
        let timeline = ContextTimeline::new();

        let (_incidents, export, _imputation) =
            run_investigation(&cols, baseline, &timeline).expect("a single Inf must not fail calibration");
        assert!(export.channels[1].alpha_mean.is_finite(), "imputed Inf must not propagate into calibration");
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

    /// Issue 2 regression test: `Case::save` used to fingerprint the
    /// *original input file* bytes (`input_hash`) but `replay()` compared
    /// that against the *transformed* `recording.csv` it writes
    /// (`ch0,ch1,...` headers, normalized values) — two different byte
    /// representations of the same data, so an untouched, unmodified case
    /// always falsely reported drift. `recording_hash` fingerprints the
    /// exact bytes written to `recording.csv`, and replay now diffs
    /// against that instead.
    #[test]
    fn replay_reports_no_fingerprint_mismatch_on_untouched_case_but_does_after_edit() {
        use crate::case::CaseConfig;
        use crate::context::ColumnSchema;
        use crate::monitor::HybridMonitor;

        let baseline = 200;
        let total = baseline + 100;
        let ch0 = sine_channel(total);
        let recording: Vec<Vec<f64>> = (0..total).map(|t| vec![ch0[t]]).collect();

        let timeline = ContextTimeline::new();
        let calib = vec![ch0[..baseline].to_vec()];
        let monitor = HybridMonitor::calibrate(&calib).expect("calibrates");
        let config = CaseConfig {
            input_hash: crate::case::fingerprint_content(b"original source file bytes, unrelated to recording.csv"),
            monitor_export: monitor.export(),
            column_schema: ColumnSchema::from_header(&["ch0"]),
            imputation: vec![],
        };

        let dir = std::env::temp_dir().join("struktura_replay_test_fingerprint");
        let _ = std::fs::remove_dir_all(&dir);
        let case = Case::save(&dir, &recording, &[], baseline, "fingerprint", &config, &timeline, &["ch0".to_string()])
            .expect("saves");

        let (_incidents, diff) = replay(&case, None).expect("replays");
        assert!(diff.fingerprint_mismatch.is_none(), "untouched case must not report a fingerprint mismatch");

        // Directly edit one value inside recording.csv (as saved).
        let recording_path = case.dir().join("recording.csv");
        let text = std::fs::read_to_string(&recording_path).expect("reads recording.csv");
        let mut lines: Vec<String> = text.lines().map(|s| s.to_string()).collect();
        let row_line_idx = 51; // header (line 0) + row 50
        let original_val: f64 = lines[row_line_idx].parse().expect("row is a single float");
        lines[row_line_idx] = (original_val + 100.0).to_string();
        let edited = lines.join("\n") + "\n";
        std::fs::write(&recording_path, &edited).expect("writes edited recording.csv");

        let (_incidents2, diff2) = replay(&case, None).expect("replays after edit");
        assert!(diff2.fingerprint_mismatch.is_some(), "edited recording.csv must report a fingerprint mismatch");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Issue 3 regression test: `cmd_case save` used to build an empty
    /// timeline and `replay()` always replayed against an empty timeline
    /// too, so a contextual investigation's context never survived the
    /// case round trip. A context event anchored to a real incident's
    /// start tick must reappear on the corresponding replayed incident at
    /// the same tick.
    #[test]
    fn replay_reproduces_saved_context_at_the_same_ticks() {
        use crate::case::CaseConfig;
        use crate::context::{ColumnSchema, ContextEvent};
        use crate::telemetry_bench::{inject_fault, synth_spacecraft, CHANNELS};

        let baseline = 1000;
        let total = baseline + 2000;
        let clean = synth_spacecraft(total, 42);
        let faulted = inject_fault(&clean, "regime_shift", 42);

        // Find a real alarm tick to anchor a context event to, instead of
        // guessing one and hoping it lands inside an incident's lookback
        // window.
        let empty_timeline = ContextTimeline::new();
        let (probe_incidents, _, _) =
            run_investigation(&faulted, baseline, &empty_timeline).expect("investigates");
        assert!(!probe_incidents.is_empty(), "fixture must raise at least one incident");
        let anchor_tick = probe_incidents[0].start_tick;

        let mut timeline = ContextTimeline::new();
        timeline.push(ContextEvent::new(anchor_tick, "mode", "safe_hold"));

        let (original_incidents, monitor_export, _imputation) =
            run_investigation(&faulted, baseline, &timeline).expect("investigates");
        assert!(
            original_incidents.iter().any(|inc| inc.context.iter().any(|c| c.value == "safe_hold")),
            "fixture must actually attach the context event to an incident"
        );

        let mut full_rows: Vec<Vec<f64>> = Vec::with_capacity(total);
        for t in 0..total {
            full_rows.push((0..CHANNELS).map(|c| faulted[c][t]).collect());
        }
        let names: Vec<String> = (0..CHANNELS).map(|c| format!("ch{}", c)).collect();
        let config = CaseConfig {
            input_hash: crate::case::fingerprint_content(b"context round trip fixture"),
            monitor_export,
            column_schema: ColumnSchema::from_header(&names.iter().map(|s| s.as_str()).collect::<Vec<_>>()),
            imputation: vec![],
        };

        let dir = std::env::temp_dir().join("struktura_replay_test_context_roundtrip");
        let _ = std::fs::remove_dir_all(&dir);
        let case = Case::save(
            &dir,
            &full_rows,
            &original_incidents,
            baseline,
            "context_roundtrip",
            &config,
            &timeline,
            &names,
        )
        .expect("saves");

        let loaded_timeline = case.context().expect("reads context.json");
        assert_eq!(loaded_timeline.len(), timeline.len());
        assert_eq!(loaded_timeline.active_at(anchor_tick).unwrap().value, "safe_hold");

        let (replayed_incidents, _diff) = replay(&case, None).expect("replays");
        assert!(
            replayed_incidents
                .iter()
                .any(|inc| inc.context.iter().any(|c| c.tick == anchor_tick && c.value == "safe_hold")),
            "replayed incidents must carry the same context event at the same tick as the original"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Issue 4 regression test: `replay()` used to compare only the three
    /// global thresholds and swallow a missing/unreadable `config.json`
    /// silently; `to_json()`/`summary()` omitted the fingerprint and
    /// threshold-diff fields entirely. A changed per-channel calibration
    /// field (not just a global threshold) in a saved `config.json` must
    /// now be reported, and an untouched case's JSON must show
    /// `fingerprint_mismatch:false` explicitly.
    #[test]
    fn replay_reports_fingerprint_and_per_channel_threshold_diffs_in_json() {
        use crate::case::CaseConfig;
        use crate::context::ColumnSchema;
        use crate::monitor::HybridMonitor;

        let baseline = 200;
        let total = baseline + 100;
        let ch0 = sine_channel(total);
        let recording: Vec<Vec<f64>> = (0..total).map(|t| vec![ch0[t]]).collect();
        let timeline = ContextTimeline::new();
        let calib = vec![ch0[..baseline].to_vec()];
        let monitor = HybridMonitor::calibrate(&calib).expect("calibrates");
        let config = CaseConfig {
            input_hash: crate::case::fingerprint_content(b"config diff fixture"),
            monitor_export: monitor.export(),
            column_schema: ColumnSchema::from_header(&["ch0"]),
            imputation: vec![],
        };
        let dir = std::env::temp_dir().join("struktura_replay_test_config_diff");
        let _ = std::fs::remove_dir_all(&dir);
        let case = Case::save(&dir, &recording, &[], baseline, "config_diff", &config, &timeline, &["ch0".to_string()])
            .expect("saves");

        // Untouched case: no fingerprint mismatch, no threshold diffs, and
        // both now show up explicitly in the JSON.
        let (_incidents, diff) = replay(&case, None).expect("replays");
        assert!(diff.threshold_diffs.is_empty());
        let json = diff.to_json();
        assert!(json.contains("\"fingerprint_mismatch\":false"), "{}", json);
        assert!(json.contains("\"threshold_diffs\":[]"), "{}", json);

        // Change a *per-channel* calibration field (not one of the three
        // global thresholds) directly in config.json.
        let config_path = case.dir().join("config.json");
        let text = std::fs::read_to_string(&config_path).expect("reads config.json");
        let saved_alpha_mean = crate::case::parse_config_monitor_export(&text).unwrap().channels[0].alpha_mean;
        let edited = text.replacen(
            &format!("\"alpha_mean\":{}", saved_alpha_mean),
            &format!("\"alpha_mean\":{}", saved_alpha_mean + 100.0),
            1,
        );
        assert_ne!(text, edited, "test setup must actually change alpha_mean");
        std::fs::write(&config_path, &edited).expect("writes edited config.json");

        let (_incidents2, diff2) = replay(&case, None).expect("replays after config edit");
        assert!(
            diff2.threshold_diffs.iter().any(|d| d.contains("alpha_mean")),
            "threshold_diffs must report the changed per-channel alpha_mean: {:?}",
            diff2.threshold_diffs
        );
        assert!(diff2.to_json().contains("\"threshold_diffs\":[\""));

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Issue 4: a case with no `config.json` at all (legacy format, saved
    /// before it existed) must still replay successfully with a warning,
    /// not error out — only a genuinely broken config.json (present but
    /// unreadable/corrupt) should propagate as an error.
    #[test]
    fn replay_tolerates_a_case_with_no_config_json() {
        use crate::case::CaseConfig;
        use crate::context::ColumnSchema;
        use crate::monitor::HybridMonitor;

        let baseline = 200;
        let total = baseline + 100;
        let ch0 = sine_channel(total);
        let recording: Vec<Vec<f64>> = (0..total).map(|t| vec![ch0[t]]).collect();
        let timeline = ContextTimeline::new();
        let calib = vec![ch0[..baseline].to_vec()];
        let monitor = HybridMonitor::calibrate(&calib).expect("calibrates");
        let config = CaseConfig {
            input_hash: crate::case::fingerprint_content(b"legacy fixture"),
            monitor_export: monitor.export(),
            column_schema: ColumnSchema::from_header(&["ch0"]),
            imputation: vec![],
        };
        let dir = std::env::temp_dir().join("struktura_replay_test_legacy_no_config");
        let _ = std::fs::remove_dir_all(&dir);
        let case = Case::save(&dir, &recording, &[], baseline, "legacy", &config, &timeline, &["ch0".to_string()])
            .expect("saves");
        std::fs::remove_file(case.dir().join("config.json")).expect("removes config.json");

        let (_incidents, diff) = replay(&case, None).expect("replay must tolerate a missing config.json");
        assert!(diff.saved_thresholds.is_none());
        assert!(diff.threshold_diffs.is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Fix 4: full end-to-end regression test covering the whole
    /// investigate -> save -> replay pipeline in one place, including
    /// operating context, calibration-window imputation (Fix 2), and both
    /// of replay's config-validation failure modes (Fix 1's corrupt-config
    /// error, and a genuine recording.csv edit that must only warn).
    #[test]
    fn end_to_end_investigate_save_replay_with_context_and_damaged_data() {
        use crate::case::CaseConfig;
        use crate::context::ColumnSchema;
        use crate::telemetry_bench::{synth_spacecraft, CHANNELS};

        // Sized like the other full-pipeline fixtures in this file (the
        // rover-flow and recalibration tests below): a short baseline lets
        // a channel's naturally noisy calibration produce spurious alarms
        // (and AutoPilot's guarded recalibration after one eats much of a
        // short recording's remaining runway), which flaked this fixture
        // at baseline=200/total=600. 1000/3000 is the scale already proven
        // reliable elsewhere in this file.
        let baseline = 1000;
        let total = 3000;
        let fault_start = 2000;
        let mut cols = synth_spacecraft(total, 7);

        // 5 NaNs injected into channel 1's calibration window (rows 10-14).
        for t in 10..15 {
            cols[1][t] = f64::NAN;
        }

        // Fault channel: channel 2 (temperature) is a continuous,
        // unclamped, noise-driven process in `synth_spacecraft` -- unlike
        // channel 0 (state of charge), which is clamped to [0.2, 0.98] and
        // can naturally plateau at the clamp boundary for a run of ticks,
        // which would otherwise trip the repeat detector on its own.
        const FAULT_CHANNEL: usize = 2;

        // Fault: the channel freezes at a strongly shifted constant from
        // `fault_start` onward -- a level shift plus a stuck/repeat
        // signature, well after both the baseline and the injected NaNs.
        let window = &cols[FAULT_CHANNEL][..fault_start];
        let mean: f64 = window.iter().sum::<f64>() / window.len() as f64;
        let std: f64 =
            (window.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / window.len() as f64).sqrt();
        let stuck_value = mean + 6.0 * std.max(1e-6);
        for v in cols[FAULT_CHANNEL].iter_mut().skip(fault_start) {
            *v = stuck_value;
        }

        let mut timeline = ContextTimeline::new();
        timeline.push(ContextEvent::new(1500, "mode", "science"));

        let (incidents, monitor_export, imputation) =
            run_investigation(&cols, baseline, &timeline).expect("investigates");

        // The fault must be caught on the fault channel somewhere after
        // `fault_start`.
        assert!(
            incidents
                .iter()
                .any(|inc| inc.evidence.iter().any(|e| e.channel == FAULT_CHANNEL && e.tick as usize >= fault_start)),
            "expected a channel-{}-incident at/after tick {}, got: {:?}",
            FAULT_CHANNEL,
            fault_start,
            incidents
        );

        // Channel 1's 5 injected NaNs must be recorded as imputed.
        let ch1_imputed = imputation.iter().find(|(ch, _)| *ch == 1).map(|(_, n)| *n);
        assert_eq!(ch1_imputed, Some(5), "imputation counts: {:?}", imputation);

        let names: Vec<String> = (0..CHANNELS).map(|c| format!("ch{}", c)).collect();
        let column_schema =
            ColumnSchema::from_header(&names.iter().map(|s| s.as_str()).collect::<Vec<_>>());
        let config = CaseConfig {
            input_hash: crate::case::fingerprint_content(b"end to end fixture"),
            monitor_export,
            column_schema,
            imputation,
        };

        let mut rows: Vec<Vec<f64>> = Vec::with_capacity(total);
        for t in 0..total {
            rows.push((0..CHANNELS).map(|c| cols[c][t]).collect());
        }

        let dir = std::env::temp_dir().join("struktura_replay_test_end_to_end");
        let _ = std::fs::remove_dir_all(&dir);
        let case = Case::save(&dir, &rows, &incidents, baseline, "end_to_end", &config, &timeline, &names)
            .expect("saves");

        assert!(case.dir().join("context.json").exists());
        assert!(case.dir().join("schema.json").exists());

        // Untouched case: fingerprint clean, everything matched, config
        // validated.
        let (replayed_incidents, diff) = replay(&case, None).expect("replays");
        assert!(diff.fingerprint_mismatch.is_none());
        assert_eq!(replayed_incidents.len(), incidents.len());
        assert_eq!(diff.matched.len(), incidents.len());
        assert!(diff.missed.is_empty());
        assert!(diff.new_alarms.is_empty());
        assert!(diff.config_valid);

        // Overwrite config.json with `{}`: replay must now error (Fix 1).
        let config_path = case.dir().join("config.json");
        let original_config = std::fs::read_to_string(&config_path).expect("reads config.json");
        std::fs::write(&config_path, "{}").expect("overwrites config.json");
        let err = replay(&case, None).unwrap_err();
        assert!(err.contains("monitor_export"), "unexpected error: {}", err);

        // Restore config.json, then modify one value in recording.csv:
        // replay must succeed but warn (report a fingerprint mismatch).
        std::fs::write(&config_path, &original_config).expect("restores config.json");
        let recording_path = case.dir().join("recording.csv");
        let text = std::fs::read_to_string(&recording_path).expect("reads recording.csv");
        let mut lines: Vec<String> = text.lines().map(|s| s.to_string()).collect();
        let row_line_idx = 51; // header (line 0) + row 50
        let mut fields: Vec<String> = lines[row_line_idx].split(',').collect::<Vec<_>>().iter().map(|s| s.to_string()).collect();
        let original_val: f64 = fields[0].parse().expect("first field is a float");
        fields[0] = (original_val + 100.0).to_string();
        lines[row_line_idx] = fields.join(",");
        let edited = lines.join("\n") + "\n";
        std::fs::write(&recording_path, &edited).expect("writes edited recording.csv");

        let (_incidents3, diff3) = replay(&case, None).expect("replays after recording edit");
        assert!(diff3.fingerprint_mismatch.is_some(), "edited recording.csv must report a fingerprint mismatch");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
