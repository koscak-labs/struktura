//! Autonomic layer over the hybrid monitor: detect → decide → adapt →
//! continue, without a human in the loop.
//!
//! Three autonomous behaviors, each with a conservative policy:
//!
//! 1. **Auto-quarantine.** An alarm whose provenance identifies a sensor
//!    failure on one channel (stuck, sustained missingness, cross-channel
//!    inconsistency) quarantines that channel: its legs go silent, its
//!    reading is served by reconstruction from the survivors, and
//!    monitoring continues degraded. Its real readings are still checked
//!    against its calibration; after [`crate::monitor::RECOVER_SPAN`]
//!    healthy samples in a row it is monitored again (a data gap filled
//!    with repeats, or a sensor that came back, is not dead forever).
//! 2. **Guarded self-recalibration.** A level-shift alarm may mean the
//!    environment changed rather than broke (new operating mode, new
//!    thermal regime). The autopilot collects a candidate window of the
//!    new regime, calibrates a candidate monitor on it, and then streams a
//!    guard window through the candidate: only if the guard stays silent
//!    is the candidate accepted. A guard alarm means the "new regime" is
//!    itself unstable: the adaptation rolls back and the original alarm
//!    stands as a confirmed fault. (This mirrors the guarded-adaptation
//!    accept/rollback discipline used in telemetry-assurance research.)
//! 3. **Continuous operation.** Genuine fault alarms (drift, spike,
//!    structural) are reported as events and the latch is cleared: an
//!    autonomous system logs and keeps watching; it never goes blind
//!    after its first detection.
//!
//! Recalibration allocates (it is a re-initialization); all steady-state
//! monitoring remains allocation-free.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(not(feature = "std"))]
use alloc::vec;

use crate::monitor::{classify_alarm, AlarmReport, HybridMonitor, Leg, RECOVER_SPAN};

/// Samples of the new regime collected before candidate calibration.
pub const RECAL_WINDOW: usize = 400;
/// Samples the candidate must stay silent for before being accepted.
pub const GUARD_WINDOW: usize = 300;
/// A second level-shift alarm on the same channel within this many samples
/// of an accepted recalibration is not another regime change; it is a
/// sustained trend (drift) that each short guard window individually
/// mistakes for a stable new normal. The autopilot then refuses to adapt
/// and reports a confirmed drift instead. (Found empirically: without
/// this, a slow drift walked the autopilot through seven consecutive
/// accepted recalibrations.)
pub const RECAL_COOLDOWN: u64 = 4000;

/// An autonomous decision or observation, timestamped by global tick.
#[derive(Debug, Clone)]
pub enum Event {
    /// A fault alarm (monitoring continues after logging it).
    Alarm { tick: u64, report: AlarmReport, class: &'static str },
    /// A channel was declared dead and switched to virtual mode.
    Quarantined { tick: u64, channel: usize },
    /// A quarantined channel's own readings passed the recovery checks for
    /// [`recovery_span`] samples in a row; it is monitored again.
    Unquarantined { tick: u64, channel: usize },
    /// A level shift triggered adaptation; candidate collection started.
    AdaptationStarted { tick: u64 },
    /// The candidate monitor passed its guard window and took over.
    Recalibrated { tick: u64 },
    /// The candidate alarmed during the guard window; adaptation rolled
    /// back, and the original level-shift alarm stands as a confirmed fault.
    RolledBack { tick: u64, guard_report: AlarmReport },
}

enum Mode {
    Monitoring,
    /// Collecting the candidate calibration window.
    Collecting { buffer: Vec<Vec<f64>>, target: usize },
    /// Streaming the guard window through the candidate.
    Guarding { candidate: HybridMonitor, fed: usize },
}

/// Autonomous wrapper: owns the monitor, applies the policies above.
pub struct AutoPilot {
    monitor: HybridMonitor,
    mode: Mode,
    channels: usize,
    tick: u64,
    quarantined: Vec<bool>,
    /// (tick, triggering channel) of the last ACCEPTED recalibration.
    last_recal: Option<(u64, usize)>,
    /// Channel whose level alarm triggered the adaptation now in progress.
    adapting_channel: usize,
    /// Per-channel confirmed-drift latch: once a channel's trend is
    /// confirmed, further level alarms on it refresh the latch silently
    /// (one fault, one report) and adaptation stays refused while the
    /// trend persists.
    drift_latch: Vec<Option<u64>>,
    /// Per channel: tick of the last release from quarantine, and how many
    /// times in a row it failed again before staying healthy for its
    /// current recovery span. Each such flap doubles the span.
    released_at: Vec<Option<u64>>,
    flaps: Vec<u32>,
}

/// A drift latch decays after this many quiet samples on the channel.
pub const DRIFT_LATCH_DECAY: u64 = 2000;

/// A channel that fails again soon after release has its recovery span
/// doubled each time, up to `RECOVER_SPAN << MAX_FLAPS`. On data where a
/// sensor keeps sticking (forward-filled gaps), quarantine/release cycles
/// then grow logarithmically with the stream length instead of linearly.
pub const MAX_FLAPS: u32 = 10;

/// Healthy samples a quarantined channel needs before release, after
/// `flaps` quick re-failures.
#[must_use]
pub fn recovery_span(flaps: u32) -> usize {
    RECOVER_SPAN << flaps.min(MAX_FLAPS)
}

/// Alarms that point at a broken sensor rather than a change in the system:
/// a stuck value, missing data, or a channel that no longer agrees with the
/// others. These quarantine the channel.
fn is_sensor_failure(r: &AlarmReport, class: &str) -> bool {
    matches!(r.leg, Leg::RepeatedValue | Leg::Missingness)
        || (r.leg == Leg::Parity && class == "cross_channel_inconsistency")
}

/// Flap count after a new quarantine at `tick`: one more if the channel was
/// released less than its current span ago, else back to zero.
fn next_flaps(flaps: u32, released_at: Option<u64>, tick: u64) -> u32 {
    match released_at {
        Some(r) if tick.saturating_sub(r) < recovery_span(flaps) as u64 => (flaps + 1).min(MAX_FLAPS),
        _ => 0,
    }
}

impl AutoPilot {
    #[must_use]
    pub fn new(monitor: HybridMonitor) -> AutoPilot {
        let channels = monitor.channels();
        let mut q = Vec::with_capacity(channels);
        q.resize(channels, false);
        AutoPilot {
            monitor,
            mode: Mode::Monitoring,
            channels,
            tick: 0,
            quarantined: q,
            last_recal: None,
            adapting_channel: 0,
            drift_latch: {
                let mut d = Vec::with_capacity(channels);
                d.resize(channels, None);
                d
            },
            released_at: {
                let mut r = Vec::with_capacity(channels);
                r.resize(channels, None);
                r
            },
            flaps: vec![0; channels],
        }
    }

    /// Access the underlying monitor (e.g. for virtual readings).
    #[must_use]
    pub fn monitor(&self) -> &HybridMonitor {
        &self.monitor
    }

    /// Feed one synchronized sample; returns the autonomous events this
    /// tick produced (empty in the steady state).
    pub fn push(&mut self, sample: &[f64], valid: &[bool]) -> Vec<Event> {
        let mut events = Vec::new();
        let tick = self.tick;
        self.tick += 1;
        // A sensor failure found in any mode is quarantined after the match.
        let mut failure: Option<(AlarmReport, &'static str)> = None;

        match &mut self.mode {
            Mode::Monitoring => {
                let alarm = self.monitor.push_with_validity(sample, valid);
                for ch in 0..self.channels {
                    if self.quarantined[ch]
                        && self.monitor.healthy_run(ch) >= recovery_span(self.flaps[ch])
                    {
                        self.monitor.unquarantine(ch);
                        self.quarantined[ch] = false;
                        self.released_at[ch] = Some(tick);
                        events.push(Event::Unquarantined { tick, channel: ch });
                    }
                }
                if let Some(_leg) = alarm {
                    if let Some(report) = self.monitor.last_alarm() {
                        let class = classify_alarm(&report);
                        match report.leg {
                            // Sensor-failure signatures → quarantine the channel
                            _ if is_sensor_failure(&report, class) => failure = Some((report, class)),
                            // Environment may have changed → guarded adaptation,
                            // unless the same channel already forced a recent
                            // recalibration: that pattern is a sustained trend
                            // (drift), and adapting again would chase the fault.
                            Leg::LevelShift => {
                                self.monitor.reset();
                                let ch = report.channel;
                                // Latched trend: refresh silently, one
                                // fault, one report, no re-adaptation.
                                if matches!(
                                    self.drift_latch[ch],
                                    Some(t0) if tick - t0 < DRIFT_LATCH_DECAY
                                ) {
                                    self.drift_latch[ch] = Some(tick);
                                    return events;
                                }
                                self.drift_latch[ch] = None;
                                let repeat_trend = matches!(
                                    self.last_recal,
                                    Some((rt, rch)) if rch == ch
                                        && tick - rt < RECAL_COOLDOWN
                                );
                                // An along-trend excursion (the channel is
                                // certified to drift and this break points
                                // the same direction, not too far past the
                                // certified band) is more of that same
                                // drift, not a new regime: report it and
                                // refuse adaptation, the same as a repeated
                                // trend. Abrupt along-trend steps (observed
                                // far past 2x threshold) and opposite-sign
                                // breaks keep today's guarded adaptation.
                                let along = self.monitor.level_break_along_trend(ch)
                                    && report.observed <= 2.0 * report.threshold;
                                if repeat_trend || along {
                                    self.drift_latch[ch] = Some(tick);
                                    let drift_class =
                                        if repeat_trend { "drift_confirmed" } else { "drift" };
                                    events.push(Event::Alarm { tick, report, class: drift_class });
                                } else {
                                    events.push(Event::Alarm { tick, report, class });
                                    events.push(Event::AdaptationStarted { tick });
                                    self.adapting_channel = report.channel;
                                    let mut buffer = Vec::with_capacity(self.channels);
                                    for _ in 0..self.channels {
                                        buffer.push(Vec::with_capacity(RECAL_WINDOW));
                                    }
                                    self.mode =
                                        Mode::Collecting { buffer, target: RECAL_WINDOW };
                                }
                            }
                            // Genuine fault → log, clear latch, keep watching
                            _ => {
                                self.monitor.reset();
                                events.push(Event::Alarm { tick, report, class });
                            }
                        }
                    } else {
                        self.monitor.reset();
                    }
                }
            }
            Mode::Collecting { buffer, target } => {
                for ch in 0..self.channels {
                    // A quarantined channel contributes its virtual reading
                    // so the candidate stays full-width.
                    let v = if self.quarantined[ch] || !valid.get(ch).copied().unwrap_or(true)
                    {
                        self.monitor.virtual_value(ch).map(|(x, _)| x).unwrap_or(sample[ch])
                    } else {
                        sample[ch]
                    };
                    buffer[ch].push(v);
                }
                // Keep the current monitor up to date. A sensor failure it finds
                // ends the collection: that channel would otherwise be calibrated
                // into the new baseline (a stuck run would even switch its
                // repeated-value leg off there).
                if self.monitor.push_with_validity(sample, valid).is_some() {
                    if let Some(report) = self.monitor.last_alarm() {
                        let class = classify_alarm(&report);
                        if is_sensor_failure(&report, class) {
                            failure = Some((report, class));
                        }
                    }
                }
                self.monitor.reset();
                if failure.is_none() && buffer[0].len() >= *target {
                    match HybridMonitor::calibrate_with_prior(buffer, &self.monitor) {
                        Some(mut candidate) => {
                            for ch in 0..self.channels {
                                if self.quarantined[ch] {
                                    // Its candidate data were reconstructed,
                                    // not measured: keep its old calibration.
                                    candidate.adopt_channel(ch, &self.monitor);
                                    candidate.quarantine(ch);
                                }
                            }
                            self.mode = Mode::Guarding { candidate, fed: 0 };
                        }
                        None => {
                            // Cannot calibrate; stay on the old monitor.
                            self.mode = Mode::Monitoring;
                        }
                    }
                }
            }
            Mode::Guarding { candidate, fed } => {
                *fed += 1;
                // Keep the current monitor up to date too, so a rollback
                // resumes from the present rather than from before the trial.
                let _ = self.monitor.push_with_validity(sample, valid);
                self.monitor.reset();
                if candidate.push_with_validity(sample, valid).is_some() {
                    let guard_report = candidate.last_alarm().unwrap_or(AlarmReport {
                        leg: Leg::LevelShift,
                        channel: 0,
                        tick,
                        observed: 0.0,
                        threshold: 0.0,
                        hit_gap: 0,
                    });
                    let class = classify_alarm(&guard_report);
                    if is_sensor_failure(&guard_report, class) {
                        // A sensor failed, not the new regime: quarantine it
                        // and drop the candidate.
                        failure = Some((guard_report, class));
                    } else {
                        // New regime is itself unstable → rollback.
                        events.push(Event::RolledBack { tick, guard_report });
                        self.mode = Mode::Monitoring;
                    }
                } else if *fed >= GUARD_WINDOW {
                    // Guard passed; the candidate takes over.
                    let mut accepted = match core::mem::replace(&mut self.mode, Mode::Monitoring)
                    {
                        Mode::Guarding { candidate, .. } => candidate,
                        _ => unreachable!(),
                    };
                    core::mem::swap(&mut self.monitor, &mut accepted);
                    self.last_recal = Some((tick, self.adapting_channel));
                    events.push(Event::Recalibrated { tick });
                }
            }
        }
        if let Some((report, class)) = failure {
            let ch = report.channel;
            self.mode = Mode::Monitoring;
            self.monitor.reset();
            self.monitor.quarantine(ch);
            self.quarantined[ch] = true;
            self.flaps[ch] = next_flaps(self.flaps[ch], self.released_at[ch], tick);
            events.push(Event::Alarm { tick, report, class });
            events.push(Event::Quarantined { tick, channel: ch });
        }
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry_bench::synth_spacecraft;

    /// The gauntlet: sensor death → permanent regime change → drift fault
    /// in the new regime. The autopilot must survive all three alone.
    #[test]
    fn autonomous_gauntlet() {
        let n = 24_000usize;
        let calib = synth_spacecraft(2048, 31_337 + 100);
        let stream = synth_spacecraft(n, 31_337 + 200);
        let mon = HybridMonitor::calibrate(&calib).expect("calibration");
        let mut ap = AutoPilot::new(mon);

        let mut quarantine_at = None;
        let mut recal_at = None;
        let mut drift_alarm_at = None;
        let mut recal_count = 0usize;

        let valid = [true; 6];
        let mut sample = [0.0f64; 6];
        // Channel means for the scripted regime shift
        let ch_sd: Vec<f64> = (0..6)
            .map(|ch| {
                let c = &calib[ch];
                let m = c.iter().sum::<f64>() / c.len() as f64;
                (c.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / c.len() as f64).sqrt()
            })
            .collect();

        for t in 0..n {
            for ch in 0..6 {
                let mut v = stream[ch][t];
                // Event 1 (t>=4000): temp sensor (ch2) freezes, dead sensor.
                if ch == 2 && t >= 4000 {
                    v = stream[2][4000];
                }
                // Event 2 (t>=10000): permanent regime change, all channels
                // shift by 0.8 sigma (new thermal/power operating point).
                if t >= 10_000 {
                    v += 0.8 * ch_sd[ch];
                }
                // Event 3 (t>=18000): drift fault on SOC in the new regime.
                if ch == 0 && t >= 18_000 {
                    v += (t - 18_000) as f64 * 0.0005 * ch_sd[0];
                }
                sample[ch] = v;
            }
            for ev in ap.push(&sample, &valid) {
                match ev {
                    Event::Quarantined { tick, channel }
                        if channel == 2 && quarantine_at.is_none() => {
                            quarantine_at = Some(tick);
                        }
                    // ch2 stays frozen: it must never come back.
                    Event::Unquarantined { tick, channel: 2 } => {
                        panic!("still-frozen sensor released at {}", tick);
                    }
                    Event::Recalibrated { tick } => {
                        recal_count += 1;
                        if recal_at.is_none() {
                            recal_at = Some(tick);
                        }
                    }
                    Event::Alarm { tick, report, class }
                        if tick > 17_000
                            && report.channel == 0
                            && class == "drift_confirmed"
                            && drift_alarm_at.is_none()
                        => {
                            drift_alarm_at = Some(tick);
                        }
                    _ => {}
                }
            }
        }

        let q = quarantine_at.expect("dead temp sensor must be auto-quarantined");
        assert!((4000..6000).contains(&(q as usize)), "quarantine at {}", q);
        let r = recal_at.expect("regime change must trigger accepted recalibration");
        assert!((10_000..13_000).contains(&(r as usize)), "recal at {}", r);
        let d = drift_alarm_at.expect("drift in the NEW regime must be CONFIRMED, not adapted into");
        assert!((18_000..24_000).contains(&(d as usize)), "drift alarm at {}", d);
        // The regime change is ONE event; the drift must not be chased with
        // repeated recalibrations (the pre-cooldown autopilot accepted 7).
        assert!(
            recal_count <= 2,
            "autopilot chased the drift: {} accepted recalibrations",
            recal_count
        );
    }

    /// A fault disguised as a regime change must be rolled back, not
    /// adapted into: the "new regime" here keeps drifting, so the
    /// candidate's guard window alarms.
    #[test]
    fn unstable_regime_rolls_back() {
        let n = 16_000usize;
        let calib = synth_spacecraft(2048, 777 + 100);
        let stream = synth_spacecraft(n, 777 + 200);
        let mon = HybridMonitor::calibrate(&calib).expect("calibration");
        let mut ap = AutoPilot::new(mon);

        let ch_sd: f64 = {
            let c = &calib[5];
            let m = c.iter().sum::<f64>() / c.len() as f64;
            (c.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / c.len() as f64).sqrt()
        };

        let valid = [true; 6];
        let mut sample = [0.0f64; 6];
        let mut rolled_back = false;
        for t in 0..n {
            for ch in 0..6 {
                let mut v = stream[ch][t];
                // From t=6000: payload_current keeps accelerating, not a
                // new stable regime but a runaway.
                if ch == 5 && t >= 6000 {
                    let dt = (t - 6000) as f64;
                    v += ch_sd * (0.8 + dt * dt * 2e-7);
                }
                sample[ch] = v;
            }
            for ev in ap.push(&sample, &valid) {
                match ev {
                    // Either defense counts as "did not adapt into the
                    // runaway": the guard window catching it (rollback), or
                    // the cooldown confirming it as a sustained trend.
                    Event::RolledBack { .. } => rolled_back = true,
                    Event::Alarm { report, class, .. }
                        if report.channel == 5 && class == "drift_confirmed" =>
                    {
                        rolled_back = true;
                    }
                    _ => {}
                }
            }
        }
        assert!(rolled_back, "runaway disguised as regime change must be refused");
    }

    /// A sensor that freezes briefly (a data gap filled with repeats, a
    /// transient stick) is quarantined, must come back once its readings are
    /// healthy again, and a later real fault on it must still be reported.
    /// Before recovery existed, quarantine was permanent and the fault below
    /// raised nothing.
    #[test]
    fn briefly_frozen_sensor_recovers_and_is_monitored_again() {
        let n = 10_000usize;
        let calib = synth_spacecraft(2048, 4242 + 100);
        let stream = synth_spacecraft(n, 4242 + 200);
        let sd2: f64 = {
            let c = &calib[2];
            let m = c.iter().sum::<f64>() / c.len() as f64;
            (c.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / c.len() as f64).sqrt()
        };
        let mon = HybridMonitor::calibrate(&calib).expect("calibration");
        let mut ap = AutoPilot::new(mon);
        let valid = [true; 6];
        let mut sample = [0.0f64; 6];
        let (mut quarantined, mut released, mut fault_alarm) = (None, None, None);
        for t in 0..n {
            for ch in 0..6 {
                let mut v = stream[ch][t];
                // ch2 (temperature) freezes for 200 samples, then is healthy.
                if ch == 2 && (3000..3200).contains(&t) {
                    v = stream[2][3000];
                }
                // A real step fault on ch2 long after it came back.
                if ch == 2 && t >= 8000 {
                    v += 8.0 * sd2;
                }
                sample[ch] = v;
            }
            for ev in ap.push(&sample, &valid) {
                match ev {
                    Event::Quarantined { tick, channel: 2 } => {
                        quarantined.get_or_insert(tick);
                    }
                    Event::Unquarantined { tick, channel: 2 } => {
                        released.get_or_insert(tick);
                    }
                    Event::Alarm { tick, report, .. } if report.channel == 2 && tick >= 8000 => {
                        fault_alarm.get_or_insert(tick);
                    }
                    _ => {}
                }
            }
        }
        let q = quarantined.expect("the frozen stretch must quarantine ch2");
        assert!((3000..3200).contains(&(q as usize)), "quarantined at {q}");
        let r = released.expect("ch2 must come back after its readings recover");
        assert!((3200..8000).contains(&(r as usize)), "released at {r}");
        let a = fault_alarm.expect("the step on the recovered sensor must be reported");
        assert!((8000..8100).contains(&(a as usize)), "fault alarm at {a}");
    }

    /// A sensor that keeps sticking (300 samples stuck, 250 healthy, over
    /// and over, like forward-filled gaps) must not produce a quarantine and
    /// a release every cycle: each quick re-failure doubles its recovery span.
    #[test]
    fn flapping_sensor_backs_off() {
        let n = 20_000usize;
        let calib = synth_spacecraft(2048, 99 + 100);
        let stream = synth_spacecraft(n, 99 + 200);
        let mon = HybridMonitor::calibrate(&calib).expect("calibration");
        let mut ap = AutoPilot::new(mon);
        let valid = [true; 6];
        let mut sample = [0.0f64; 6];
        let (mut quarantines, mut releases) = (0usize, 0usize);
        for t in 0..n {
            for ch in 0..6 {
                let mut v = stream[ch][t];
                if ch == 2 && t >= 3000 && (t - 3000) % 550 < 300 {
                    v = stream[2][t - (t - 3000) % 550];
                }
                sample[ch] = v;
            }
            for ev in ap.push(&sample, &valid) {
                match ev {
                    Event::Quarantined { channel: 2, .. } => quarantines += 1,
                    Event::Unquarantined { channel: 2, .. } => releases += 1,
                    _ => {}
                }
            }
        }
        // Without back-off: a cycle every 550 samples, about 30 here.
        assert!(releases >= 1, "the sensor is healthy between sticks: it must come back sometimes");
        assert!(quarantines <= 6, "{quarantines} quarantines, {releases} releases: no back-off");
    }

    fn sd(c: &[f64]) -> f64 {
        let m = c.iter().sum::<f64>() / c.len() as f64;
        (c.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / c.len() as f64).sqrt()
    }

    /// Three channels: a = s + its own noise, b = 3 s (almost noise-free), c
    /// independent; s is a slow AR(0.99). a's own noise is what the other
    /// channels cannot reconstruct.
    fn coupled(n: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = crate::telemetry_bench::GaussRng::new(seed);
        let (mut s, mut c) = (0.0f64, 0.0f64);
        let mut out: Vec<Vec<f64>> = (0..3).map(|_| Vec::with_capacity(n)).collect();
        for _ in 0..n {
            s = 0.99 * s + rng.normal(0.0, 0.1);
            c = 0.8 * c + rng.normal(0.0, 1.0);
            out[0].push(s + rng.normal(0.0, 0.3));
            out[1].push(3.0 * s + rng.normal(0.0, 0.05));
            out[2].push(c);
        }
        out
    }

    /// A channel that is quarantined when a regime change is adapted to must
    /// still come back: the new monitor's calibration for it was fitted on
    /// reconstructed readings, which lack its own noise, so judged against
    /// that calibration its real readings never passed the recovery checks.
    #[test]
    fn channel_quarantined_through_a_recalibration_still_recovers() {
        let n = 6000usize;
        let calib = coupled(2048, 31);
        let stream = coupled(n, 32);
        let sd_c = sd(&calib[2]);
        let mut ap = AutoPilot::new(HybridMonitor::calibrate(&calib).expect("calibration"));
        let valid = [true; 3];
        let (mut quarantined, mut recal, mut released) = (None, None, None);
        for t in 0..n {
            let mut sample = [stream[0][t], stream[1][t], stream[2][t]];
            if (300..700).contains(&t) {
                sample[0] = stream[0][300]; // a frozen: quarantined
            }
            if t >= 400 {
                sample[2] += 5.0 * sd_c; // regime change on c while a is quarantined
            }
            for ev in ap.push(&sample, &valid) {
                match ev {
                    Event::Quarantined { tick, channel: 0 } => { quarantined.get_or_insert(tick); }
                    Event::Recalibrated { tick } => { recal.get_or_insert(tick); }
                    Event::Unquarantined { tick, channel: 0 } => { released.get_or_insert(tick); }
                    _ => {}
                }
            }
        }
        let q = quarantined.expect("a must be quarantined while frozen");
        let r = recal.expect("the regime change on c must be adapted to");
        assert!(q < r, "quarantined at {q}, recalibrated at {r}: the test needs q < r");
        let rel = released.expect("a is healthy from 700: it must come back after the recalibration");
        assert!(rel > r && rel < r + 1000, "recalibrated at {r}, released at {rel}");
    }

    /// A sensor that fails while a new baseline is being collected must be
    /// quarantined, not calibrated into the new baseline.
    #[test]
    fn sensor_failure_during_recalibration_is_quarantined() {
        let n = 8000usize;
        let calib = synth_spacecraft(2048, 6160 + 100);
        let stream = synth_spacecraft(n, 6160 + 200);
        let sds: Vec<f64> = calib.iter().map(|c| sd(c)).collect();
        let mut ap = AutoPilot::new(HybridMonitor::calibrate(&calib).expect("calibration"));
        let valid = [true; 6];
        let mut sample = [0.0f64; 6];
        let (mut adapting, mut quarantined) = (None, None);
        for t in 0..n {
            for ch in 0..6 {
                let mut v = stream[ch][t] + if t >= 3000 { 0.8 * sds[ch] } else { 0.0 };
                if ch == 4 && t >= 3150 {
                    v = stream[4][3150] + 0.8 * sds[4];
                }
                sample[ch] = v;
            }
            for ev in ap.push(&sample, &valid) {
                match ev {
                    Event::AdaptationStarted { tick } => { adapting.get_or_insert(tick); }
                    Event::Quarantined { tick, channel: 4 } => { quarantined.get_or_insert(tick); }
                    _ => {}
                }
            }
        }
        let a = adapting.expect("the level shift at 3000 must start an adaptation");
        assert!(a < 3150, "adaptation started at {a}: the test needs it before the failure");
        let q = quarantined.expect("ch4, stuck from 3150, must be quarantined");
        assert!(q < 3450, "ch4 quarantined only at {q}");
    }

    /// While a candidate monitor is on trial, the current one keeps taking
    /// samples, so a rollback resumes from an up-to-date monitor.
    #[test]
    fn rollback_resumes_from_an_up_to_date_monitor() {
        let n = 3000usize;
        let calib = coupled(2048, 41);
        let stream = coupled(n, 42);
        let mut ap = AutoPilot::new(HybridMonitor::calibrate(&calib).expect("calibration"));
        let valid = [true; 3];
        let mut rollbacks = 0;
        for t in 0..n {
            let mut sample = [stream[0][t], stream[1][t], stream[2][t]];
            // Shift the common source s (a by d, b by 3 d): a consistent regime
            // change that starts an adaptation, then a runaway during the trial.
            let d = if t >= 400 { 3.0 + if t >= 900 { 0.01 * (t - 900) as f64 } else { 0.0 } } else { 0.0 };
            sample[0] += d;
            sample[1] += 3.0 * d;
            for ev in ap.push(&sample, &valid) {
                if let Event::RolledBack { .. } = ev {
                    rollbacks += 1;
                    for (ch, &k) in ap.monitor().channel_ticks().iter().enumerate() {
                        assert_eq!(k, t as u64 + 1, "at the rollback (sample {t}) channel {ch} had taken {k} samples");
                    }
                }
            }
        }
        assert!(rollbacks >= 1, "the runaway during the trial must roll the adaptation back");
    }

    fn t1_like_stream(seed: u64, n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut rng = crate::telemetry_bench::GaussRng::new(seed);
        let mut e0 = 0.0f64;
        let mut ch0 = Vec::with_capacity(n);
        for i in 0..n {
            e0 = 0.3 * e0 + rng.normal(0.0, 0.076);
            ch0.push(10.0 - 4e-4 * i as f64 + e0);
        }
        let mut e1 = 0.0f64;
        let mut ch1 = Vec::with_capacity(n);
        for _ in 0..n {
            e1 = 0.3 * e1 + rng.normal(0.0, 0.076);
            ch1.push(5.0 + e1);
        }
        (ch0, ch1)
    }

    fn ar03_const(seed: u64, n: usize, level: f64) -> Vec<f64> {
        let mut rng = crate::telemetry_bench::GaussRng::new(seed);
        let mut e = 0.0f64;
        (0..n)
            .map(|_| {
                e = 0.3 * e + rng.normal(0.0, 0.076);
                level + e
            })
            .collect()
    }

    fn t1_ramp(seed: u64, n: usize) -> Vec<f64> {
        let mut rng = crate::telemetry_bench::GaussRng::new(seed);
        let mut e = 0.0f64;
        (0..n)
            .map(|i| {
                e = 0.3 * e + rng.normal(0.0, 0.076);
                10.0 - 4e-4 * i as f64 + e
            })
            .collect()
    }

    /// T7 (FAIL BEFORE): the rover's battery sag runs along its certified
    /// calibration trend — it must be reported as a confirmed drift, not
    /// chased through guarded adaptation.
    #[test]
    fn battery_sag_along_trend_is_reported_not_adapted() {
        use crate::rover::{RoverFault, RoverSim, ROVER_CHANNELS};

        let mut sim = RoverSim::new(3);
        sim.inject(1500, RoverFault::BatteryCell { severity: 0.7 });
        let data = sim.run(3000);
        let calib: Vec<Vec<f64>> = data.iter().map(|c| c[..1000].to_vec()).collect();
        let mon = HybridMonitor::calibrate(&calib).expect("calibration");
        let mut ap = AutoPilot::new(mon);
        let valid = [true; ROVER_CHANNELS];
        let mut sample = [0.0f64; ROVER_CHANNELS];
        let mut first_battery_alarm: Option<(usize, &'static str)> = None;
        for t in 1000..3000usize {
            for ch in 0..ROVER_CHANNELS {
                sample[ch] = data[ch][t];
            }
            let events = ap.push(&sample, &valid);
            // Parity (cross-channel reconstruction) is out of scope for this
            // change (Phase 2) — only the trend-aware legs are asserted on.
            let battery_alarm = events.iter().find_map(|ev| match ev {
                Event::Alarm { report, class, .. }
                    if (report.channel == 5 || report.channel == 6)
                        && matches!(report.leg, Leg::LevelShift | Leg::ResidualCusum) =>
                {
                    Some(*class)
                }
                _ => None,
            });
            if let Some(class) = battery_alarm {
                let has_adapt = events.iter().any(|ev| matches!(ev, Event::AdaptationStarted { .. }));
                assert!(!has_adapt, "the battery alarm at t={t} must not also start an adaptation");
                first_battery_alarm.get_or_insert((t, class));
            }
        }
        let (t, class) = first_battery_alarm.expect("the battery sag must be reported");
        assert!((1500..1800).contains(&t), "first battery alarm at {t}, expected in 1500..1800");
        assert!(
            class == "drift" || class == "drift_confirmed",
            "unexpected class {class} for the battery-sag alarm"
        );
    }

    /// T8 (mutation guard): a recalibration candidate must never certify a
    /// fresh trend on its own buffer — the runaway here has R2~0.68 over a
    /// 400-sample window, which a naive gate would certify.
    #[test]
    fn adaptation_does_not_certify_a_new_drift() {
        let calib_len = 2048usize;
        let n = 5000usize;
        let (mut ch0, ch1) = {
            let mut rng = crate::telemetry_bench::GaussRng::new(555);
            let mut e0 = 0.0f64;
            let ch0: Vec<f64> = (0..n).map(|_| { e0 = 0.3 * e0 + rng.normal(0.0, 0.076); 5.0 + e0 }).collect();
            let mut e1 = 0.0f64;
            let ch1: Vec<f64> = (0..n).map(|_| { e1 = 0.3 * e1 + rng.normal(0.0, 0.076); 5.0 + e1 }).collect();
            (ch0, ch1)
        };
        // A step large enough to cross per-sample residual/CUSUM thresholds
        // immediately would be caught by those (unchanged, existing) legs
        // before the level leg (rolling mean) ever gets a chance to react —
        // that race is a property of the detector architecture, not of
        // this change. Use a small initial offset (below the CUSUM slack)
        // that grows into a clear level shift, the same relative scale
        // `autonomous_gauntlet`'s regime-shift fixture (0.8 sigma) already
        // proves routes through LevelShift.
        for t in 3000..n {
            ch0[t] += 0.06 + 3e-4 * (t - 3000) as f64;
        }
        let calib = vec![ch0[..calib_len].to_vec(), ch1[..calib_len].to_vec()];
        let mut mon = HybridMonitor::calibrate(&calib).expect("calibration");
        mon.set_leg_enabled(Leg::Parity, false);
        let mut ap = AutoPilot::new(mon);
        let valid = [true, true];
        let mut sample = [0.0f64; 2];
        let mut confirmed = false;
        for t in calib_len..n {
            sample[0] = ch0[t];
            sample[1] = ch1[t];
            for ev in ap.push(&sample, &valid) {
                match ev {
                    Event::RolledBack { .. } => confirmed = true,
                    Event::Alarm { report, class, .. } if report.channel == 0 && class == "drift_confirmed" => {
                        confirmed = true;
                    }
                    Event::Recalibrated { .. } => {
                        assert!(
                            ap.monitor().trend(0).is_none(),
                            "a recalibration candidate must never certify a new trend on ch0"
                        );
                    }
                    _ => {}
                }
            }
        }
        assert!(
            confirmed,
            "the runaway disguised as a fresh drift must be refused (RolledBack or drift_confirmed) by t={n}"
        );
    }

    /// T8b (FAIL on the "candidate re-fits its own rate" mutant): an
    /// opposite-sign step takes the guarded-adaptation path; the accepted
    /// candidate must inherit ch0's exact certified rate, not refit it.
    #[test]
    fn recalibration_keeps_the_certified_rate() {
        let n = 6000usize;
        let (mut ch0, ch1) = t1_like_stream(11, n);
        // A step magnitude that routes through the level leg (rolling
        // mean) rather than the per-sample residual/CUSUM legs — see the
        // comment in `adaptation_does_not_certify_a_new_drift` on why a
        // multi-sigma step would win that race regardless of this change.
        for t in 2500..n {
            ch0[t] += 0.08;
        }
        let calib_len = 1000usize;
        let calib = vec![ch0[..calib_len].to_vec(), ch1[..calib_len].to_vec()];
        let mut mon = HybridMonitor::calibrate(&calib).expect("calibration");
        mon.set_leg_enabled(Leg::Parity, false);
        let original_slope = mon.trend(0).expect("ch0 must certify a trend").slope;
        let mut ap = AutoPilot::new(mon);
        let valid = [true, true];
        let mut sample = [0.0f64; 2];
        let mut recal_at: Option<usize> = None;
        let end = n.min(calib_len + 4500);
        for t in calib_len..end {
            sample[0] = ch0[t];
            sample[1] = ch1[t];
            for ev in ap.push(&sample, &valid) {
                match ev {
                    Event::Recalibrated { .. } => {
                        recal_at.get_or_insert(t);
                    }
                    Event::Alarm { report, .. }
                        if report.channel == 0 && matches!(report.leg, Leg::LevelShift | Leg::ResidualCusum) =>
                    {
                        if let Some(r) = recal_at {
                            assert!(
                                t - r >= 1500,
                                "ch0 alarmed ({:?}) at t={t}, only {} samples after recalibration at {r}",
                                report.leg,
                                t - r
                            );
                        }
                    }
                    _ => {}
                }
            }
        }
        let r = recal_at.expect("the opposite-sign step must trigger an accepted recalibration");
        let new_slope = ap.monitor().trend(0).expect("ch0 must still be certified after recalibration").slope;
        assert_eq!(
            new_slope.to_bits(),
            original_slope.to_bits(),
            "recalibration must keep the certified rate exactly (old {original_slope}, new {new_slope})"
        );
        let _ = r;
    }

    /// T9 (mutation guard for `adopt_channel`'s tr_t0 re-base): a trend
    /// channel quarantined through a recalibration must keep its trend
    /// frame through it. Without the re-base, the reference is off by
    /// roughly slope x elapsed-ticks, which gives a false alarm or no
    /// recovery.
    #[test]
    fn quarantined_trend_channel_keeps_its_frame_through_recalibration() {
        let calib_len = 2048usize;
        let n = 6000usize;
        let ch0 = t1_ramp(21, n);
        let ch1 = ar03_const(22, n, 5.0);
        let mut ch2 = ar03_const(23, n, 5.0);

        let calib =
            vec![ch0[..calib_len].to_vec(), ch1[..calib_len].to_vec(), ch2[..calib_len].to_vec()];
        let mut mon = HybridMonitor::calibrate(&calib).expect("calibration");
        mon.set_leg_enabled(Leg::Parity, false);
        assert!(mon.trend(0).is_some(), "ch0 must certify a trend");

        // ch0 frozen for stream ticks [300, 700) -> repeated-value quarantine.
        let mut ch0_stream = ch0.clone();
        let freeze_val = ch0_stream[calib_len + 300];
        for v in ch0_stream[(calib_len + 300)..(calib_len + 700)].iter_mut() {
            *v = freeze_val;
        }
        // ch2 steps by 5 sd from stream tick 400 -> triggers an adaptation
        // while ch0 is quarantined.
        let sd2 = {
            let c = &ch1[..calib_len];
            let m = c.iter().sum::<f64>() / c.len() as f64;
            (c.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / c.len() as f64).sqrt()
        };
        // 0.8 sigma: the same relative magnitude `autonomous_gauntlet`'s
        // regime-shift fixture uses, which reliably routes through
        // LevelShift rather than the per-sample residual/CUSUM legs.
        for v in ch2[(calib_len + 400)..].iter_mut() {
            *v += 0.8 * sd2;
        }

        let mut ap = AutoPilot::new(mon);
        let valid = [true; 3];
        let mut sample = [0.0f64; 3];
        let (mut quarantined_at, mut recal_at, mut unquarantined_at) = (None, None, None);
        let mut post_recal_alarm: Option<(usize, Leg)> = None;
        for t in calib_len..n {
            sample[0] = ch0_stream[t];
            sample[1] = ch1[t];
            sample[2] = ch2[t];
            for ev in ap.push(&sample, &valid) {
                match ev {
                    Event::Quarantined { tick, channel: 0 } => {
                        quarantined_at.get_or_insert(tick as usize);
                    }
                    Event::Recalibrated { tick } => {
                        recal_at.get_or_insert(tick as usize);
                    }
                    Event::Unquarantined { tick, channel: 0 } => {
                        unquarantined_at.get_or_insert(tick as usize);
                    }
                    Event::Alarm { report, .. }
                        if report.channel == 0 && matches!(report.leg, Leg::LevelShift | Leg::ResidualCusum) =>
                    {
                        if let Some(r) = recal_at {
                            if t > r && t - r < 2000 {
                                post_recal_alarm.get_or_insert((t, report.leg));
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        let q = quarantined_at.expect("ch0's frozen stretch must quarantine it");
        let r = recal_at.expect("ch2's step must trigger an accepted recalibration");
        assert!(q < r, "quarantined at {q}, recalibrated at {r}: the test needs q < r");
        let u = unquarantined_at.expect("ch0 must recover and be unquarantined");
        assert!(
            u >= r && u <= r + 1500,
            "unquarantined at {u}, expected within [{r}, {}]",
            r + 1500
        );
        assert!(
            post_recal_alarm.is_none(),
            "false alarm on ch0 at {:?} within 2000 samples after recalibration at {r}",
            post_recal_alarm
        );
    }
}
