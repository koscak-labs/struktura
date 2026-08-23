//! Spacecraft health monitoring via DFA structural analysis.
//!
//! Real-time anomaly detection for telemetry channels — reaction wheels,
//! magnetometers, thermal sensors, battery voltage, solar array current.
//! Detects structural degradation before threshold-based monitors trigger.
//!
//! ```
//! use struktura::space::{SpacecraftMonitor, voyager_demo};
//! let result = voyager_demo();
//! assert!(result.anomaly_detected);
//! ```

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(not(feature = "std"))]
use alloc::string::String;
#[cfg(not(feature = "std"))]
use alloc::format;

use crate::{analyze, health_check, dfa, HealthVerdict, BaselineTracker};
use core::fmt;

/// Spacecraft subsystem being monitored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Subsystem {
    ReactionWheel,
    Magnetometer,
    ThermalSensor,
    BatteryVoltage,
    SolarArray,
    Gyroscope,
    StarTracker,
    Thruster,
    Transponder,
    Custom,
}

impl fmt::Display for Subsystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Subsystem::ReactionWheel => write!(f, "RWA"),
            Subsystem::Magnetometer => write!(f, "MAG"),
            Subsystem::ThermalSensor => write!(f, "THM"),
            Subsystem::BatteryVoltage => write!(f, "BAT"),
            Subsystem::SolarArray => write!(f, "SA"),
            Subsystem::Gyroscope => write!(f, "GYR"),
            Subsystem::StarTracker => write!(f, "STR"),
            Subsystem::Thruster => write!(f, "THR"),
            Subsystem::Transponder => write!(f, "XPDR"),
            Subsystem::Custom => write!(f, "CUST"),
        }
    }
}

/// Result of a spacecraft telemetry health assessment.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TelemetryHealth {
    pub subsystem: Subsystem,
    pub channel_name: String,
    pub current_alpha: f64,
    pub baseline_alpha: f64,
    pub shift: f64,
    pub r_squared: f64,
    pub verdict: HealthVerdict,
    pub samples: usize,
}

impl fmt::Display for TelemetryHealth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}:{}] α={:.3} baseline={:.3} shift={:.3} R²={:.4} → {}",
            self.subsystem, self.channel_name,
            self.current_alpha, self.baseline_alpha, self.shift,
            self.r_squared, self.verdict)
    }
}

/// Real-time spacecraft telemetry monitor.
///
/// Wraps [`BaselineTracker`] with spacecraft-specific defaults:
/// - 512-sample sliding window (typical for 1Hz telemetry over ~8 minutes)
/// - 2048-sample learning period (builds baseline over ~34 minutes)
/// - Configurable per-subsystem thresholds
pub struct SpacecraftMonitor {
    tracker: BaselineTracker,
    subsystem: Subsystem,
    channel: String,
    threshold: f64,
}

impl SpacecraftMonitor {
    pub fn new(subsystem: Subsystem, channel: &str) -> Self {
        let (window, learning) = match subsystem {
            Subsystem::ReactionWheel => (256, 1024),
            Subsystem::Magnetometer => (512, 2048),
            Subsystem::BatteryVoltage => (1024, 4096),
            Subsystem::ThermalSensor => (1024, 4096),
            _ => (512, 2048),
        };
        SpacecraftMonitor {
            tracker: BaselineTracker::new(window, learning),
            subsystem,
            channel: String::from(channel),
            threshold: 0.08,
        }
    }

    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    pub fn with_window(subsystem: Subsystem, channel: &str, window: usize, learning: usize) -> Self {
        SpacecraftMonitor {
            tracker: BaselineTracker::new(window, learning),
            subsystem,
            channel: String::from(channel),
            threshold: 0.08,
        }
    }

    pub fn push(&mut self, value: f64) -> Option<HealthVerdict> {
        self.tracker.push(value)
    }

    pub fn baseline(&self) -> Option<f64> {
        self.tracker.baseline()
    }

    pub fn is_learning(&self) -> bool {
        self.tracker.is_learning()
    }

    pub fn assess(&self, values: &[f64]) -> TelemetryHealth {
        let law = analyze(values);
        let baseline = self.baseline().unwrap_or(law.dfa.alpha);
        let shift = law.dfa.alpha - baseline;
        let verdict = HealthVerdict::from_shift_threshold(shift, self.threshold);
        TelemetryHealth {
            subsystem: self.subsystem,
            channel_name: self.channel.clone(),
            current_alpha: law.dfa.alpha,
            baseline_alpha: baseline,
            shift,
            r_squared: law.dfa.r_squared,
            verdict,
            samples: values.len(),
        }
    }
}

/// Analyze a batch of telemetry channels simultaneously.
pub fn multi_channel_health(
    channels: &[(&str, Subsystem, &[f64], f64)],
) -> Vec<TelemetryHealth> {
    channels.iter().map(|(name, subsystem, data, baseline_alpha)| {
        let law = analyze(data);
        let shift = law.dfa.alpha - baseline_alpha;
        let verdict = HealthVerdict::from_shift(shift);
        TelemetryHealth {
            subsystem: *subsystem,
            channel_name: String::from(*name),
            current_alpha: law.dfa.alpha,
            baseline_alpha: *baseline_alpha,
            shift,
            r_squared: law.dfa.r_squared,
            verdict,
            samples: data.len(),
        }
    }).collect()
}

/// Result of the Voyager demo.
#[derive(Debug)]
pub struct VoyagerDemoResult {
    pub healthy_alpha: f64,
    pub healthy_r2: f64,
    pub anomaly_alpha: f64,
    pub anomaly_r2: f64,
    pub shift: f64,
    pub anomaly_detected: bool,
    pub verdict: HealthVerdict,
}

impl fmt::Display for VoyagerDemoResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Voyager 1 Magnetometer (NASA SPDF 48s averages)\n\
                    2021 healthy:     α={:.3} R²={:.4}\n\
                    2022 anomaly:     α={:.3} R²={:.4}\n\
                    Structural shift: {:.3}\n\
                    Verdict:          {}",
            self.healthy_alpha, self.healthy_r2,
            self.anomaly_alpha, self.anomaly_r2,
            self.shift, self.verdict)
    }
}

/// Run DFA on real Voyager 1 magnetometer data.
///
/// Compares 2021 (healthy) vs May-Jul 2022 (AACS anomaly period).
/// The anomaly was a real spacecraft failure — Voyager 1's attitude
/// articulation and control system sent garbled telemetry for months.
/// DFA detects the structural shift in magnetometer readings.
pub fn voyager_demo() -> VoyagerDemoResult {
    let healthy: Vec<f64> = include_str!("../data/voyager1_healthy_4k.csv")
        .lines().filter_map(|l| l.trim().parse().ok()).collect();
    let anomaly: Vec<f64> = include_str!("../data/voyager1_anomaly_4k.csv")
        .lines().filter_map(|l| l.trim().parse().ok()).collect();

    let law_h = dfa(&healthy);
    let law_a = dfa(&anomaly);
    let shift = law_a.alpha - law_h.alpha;
    let verdict = health_check(
        &analyze(&anomaly),
        law_h.alpha,
    );

    VoyagerDemoResult {
        healthy_alpha: law_h.alpha,
        healthy_r2: law_h.r_squared,
        anomaly_alpha: law_a.alpha,
        anomaly_r2: law_a.r_squared,
        shift,
        anomaly_detected: verdict != HealthVerdict::Healthy,
        verdict,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voyager_detects_anomaly() {
        let result = voyager_demo();
        assert!(result.anomaly_detected, "DFA should detect the 2022 AACS anomaly");
        assert!(result.healthy_r2 > 0.9, "healthy R² should be high");
        assert!(result.anomaly_r2 > 0.9, "anomaly R² should be high");
        assert!(result.shift.abs() > 0.03, "shift should be measurable: {}", result.shift);
    }

    #[test]
    fn spacecraft_monitor_learns_baseline() {
        let mut mon = SpacecraftMonitor::new(Subsystem::Magnetometer, "B_total");
        let healthy: Vec<f64> = include_str!("../data/voyager1_healthy_4k.csv")
            .lines().filter_map(|l| l.trim().parse().ok()).collect();
        for &v in &healthy {
            mon.push(v);
        }
        assert!(!mon.is_learning(), "should have finished learning after 4096 samples");
        assert!(mon.baseline().is_some(), "baseline should be established");
    }

    #[test]
    fn multi_channel_produces_verdicts() {
        let healthy: Vec<f64> = include_str!("../data/voyager1_healthy_4k.csv")
            .lines().filter_map(|l| l.trim().parse().ok()).collect();
        let anomaly: Vec<f64> = include_str!("../data/voyager1_anomaly_4k.csv")
            .lines().filter_map(|l| l.trim().parse().ok()).collect();
        let baseline = dfa(&healthy).alpha;

        let results = multi_channel_health(&[
            ("B_total", Subsystem::Magnetometer, &healthy, baseline),
            ("B_anomaly", Subsystem::Magnetometer, &anomaly, baseline),
        ]);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].verdict, HealthVerdict::Healthy);
        assert_ne!(results[1].verdict, HealthVerdict::Healthy);
    }
}
