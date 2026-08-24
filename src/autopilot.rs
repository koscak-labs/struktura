//! Autonomic layer over the hybrid monitor: detect → decide → adapt →
//! continue, without a human in the loop.
//!
//! Three autonomous behaviors, each with a conservative policy:
//!
//! 1. **Auto-quarantine.** An alarm whose provenance identifies a SENSOR
//!    failure on one channel (stuck, sustained missingness, cross-channel
//!    inconsistency) quarantines that channel: its legs go silent, its
//!    reading is served by reconstruction from the survivors, and
//!    monitoring continues degraded.
//! 2. **Guarded self-recalibration.** A level-shift alarm may mean the
//!    ENVIRONMENT changed rather than broke (new operating mode, new
//!    thermal regime). The autopilot collects a candidate window of the
//!    new regime, calibrates a candidate monitor on it, and then streams a
//!    guard window through the candidate: only if the guard stays silent
//!    is the candidate accepted. A guard alarm means the "new regime" is
//!    itself unstable — the adaptation rolls back and the original alarm
//!    stands as a confirmed fault. (This mirrors the guarded-adaptation
//!    accept/rollback discipline used in telemetry-assurance research.)
//! 3. **Continuous operation.** Genuine fault alarms (drift, spike,
//!    structural) are reported as events and the latch is cleared — an
//!    autonomous system logs and keeps watching; it never goes blind
//!    after its first detection.
//!
//! Recalibration allocates (it is a re-initialization); all steady-state
//! monitoring remains allocation-free.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::monitor::{classify_alarm, AlarmReport, HybridMonitor, Leg};

/// Samples of the new regime collected before candidate calibration.
pub const RECAL_WINDOW: usize = 400;
/// Samples the candidate must stay silent for before being accepted.
pub const GUARD_WINDOW: usize = 300;
/// A second level-shift alarm on the SAME channel within this many samples
/// of an accepted recalibration is not another regime change — it is a
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
    /// A level shift triggered adaptation; candidate collection started.
    AdaptationStarted { tick: u64 },
    /// The candidate monitor passed its guard window and took over.
    Recalibrated { tick: u64 },
    /// The candidate alarmed during the guard window — adaptation rolled
    /// back; the original level-shift alarm stands as a confirmed fault.
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
}

/// A drift latch decays after this many quiet samples on the channel.
pub const DRIFT_LATCH_DECAY: u64 = 2000;

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

        match &mut self.mode {
            Mode::Monitoring => {
                if let Some(_leg) = self.monitor.push_with_validity(sample, valid) {
                    if let Some(report) = self.monitor.last_alarm() {
                        let class = classify_alarm(&report);
                        match report.leg {
                            // Sensor-failure signatures → quarantine the channel
                            Leg::RepeatedValue | Leg::Missingness => {
                                self.monitor.reset();
                                self.monitor.quarantine(report.channel);
                                self.quarantined[report.channel] = true;
                                events.push(Event::Alarm { tick, report, class });
                                events.push(Event::Quarantined {
                                    tick,
                                    channel: report.channel,
                                });
                            }
                            Leg::Parity if class == "cross_channel_inconsistency" => {
                                self.monitor.reset();
                                self.monitor.quarantine(report.channel);
                                self.quarantined[report.channel] = true;
                                events.push(Event::Alarm { tick, report, class });
                                events.push(Event::Quarantined {
                                    tick,
                                    channel: report.channel,
                                });
                            }
                            // Environment may have changed → guarded adaptation,
                            // UNLESS the same channel already forced a recent
                            // recalibration — that pattern is a sustained trend
                            // (drift), and adapting again would chase the fault.
                            Leg::LevelShift => {
                                self.monitor.reset();
                                let ch = report.channel;
                                // Latched trend: refresh silently — one
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
                                if repeat_trend {
                                    self.drift_latch[ch] = Some(tick);
                                    events.push(Event::Alarm {
                                        tick,
                                        report,
                                        class: "drift_confirmed",
                                    });
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
                    // Keep the old monitor's state warm (no alarms read).
                    let _ = self.monitor.push_channel(ch, Some(v));
                }
                self.monitor.reset();
                if buffer[0].len() >= *target {
                    match HybridMonitor::calibrate(buffer) {
                        Some(mut candidate) => {
                            for ch in 0..self.channels {
                                if self.quarantined[ch] {
                                    candidate.quarantine(ch);
                                }
                            }
                            self.mode = Mode::Guarding { candidate, fed: 0 };
                        }
                        None => {
                            // Cannot calibrate — stay on the old monitor.
                            self.mode = Mode::Monitoring;
                        }
                    }
                }
            }
            Mode::Guarding { candidate, fed } => {
                *fed += 1;
                if candidate.push_with_validity(sample, valid).is_some() {
                    // New regime is itself unstable → rollback.
                    let guard_report = candidate.last_alarm().unwrap_or(AlarmReport {
                        leg: Leg::LevelShift,
                        channel: 0,
                        tick,
                        observed: 0.0,
                        threshold: 0.0,
                        hit_gap: 0,
                    });
                    events.push(Event::RolledBack { tick, guard_report });
                    self.mode = Mode::Monitoring;
                } else if *fed >= GUARD_WINDOW {
                    // Guard passed — the candidate takes over.
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
                // Event 1 (t>=4000): temp sensor (ch2) freezes — dead sensor.
                if ch == 2 && t >= 4000 {
                    v = stream[2][4000];
                }
                // Event 2 (t>=10000): PERMANENT regime change, all channels
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
                    Event::Quarantined { tick, channel } => {
                        if channel == 2 && quarantine_at.is_none() {
                            quarantine_at = Some(tick);
                        }
                    }
                    Event::Recalibrated { tick } => {
                        recal_count += 1;
                        if recal_at.is_none() {
                            recal_at = Some(tick);
                        }
                    }
                    Event::Alarm { tick, report, class } => {
                        if tick > 17_000
                            && report.channel == 0
                            && class == "drift_confirmed"
                            && drift_alarm_at.is_none()
                        {
                            drift_alarm_at = Some(tick);
                        }
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
                // From t=6000: payload_current keeps ACCELERATING — not a
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
}
