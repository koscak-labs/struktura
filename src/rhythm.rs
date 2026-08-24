//! Rhythm analysis — DFA on inter-event timing sequences.
//!
//! Works on any sequence of timestamps: git commits, keystrokes,
//! heartbeats, network packets, sensor readings.
//!
//! ```
//! use struktura::rhythm::{intervals_from_timestamps, rhythm_analyze};
//! let timestamps = vec![0.0, 1.2, 2.1, 3.5, 4.0, 5.8, 6.2, 7.9];
//! let intervals = intervals_from_timestamps(&timestamps);
//! // Need 200+ intervals for reliable DFA
//! ```

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::{dfa, analyze, StructuralLaw};
use core::fmt;

/// Rhythm classification based on DFA of inter-event intervals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RhythmType {
    /// α > 0.7 — strongly correlated bursts (creative/flow state)
    Bursty,
    /// 0.55 < α < 0.7 — natural human rhythm
    Natural,
    /// 0.45 < α < 0.55 — random/uncorrelated
    Random,
    /// α < 0.45 — metronomic/automated
    Metronomic,
}

impl fmt::Display for RhythmType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RhythmType::Bursty => write!(f, "BURSTY"),
            RhythmType::Natural => write!(f, "NATURAL"),
            RhythmType::Random => write!(f, "RANDOM"),
            RhythmType::Metronomic => write!(f, "METRONOMIC"),
        }
    }
}

/// Result of rhythm analysis.
#[derive(Debug, Clone)]
pub struct RhythmResult {
    pub alpha: f64,
    pub r_squared: f64,
    pub rhythm: RhythmType,
    pub n_intervals: usize,
    pub mean_interval: f64,
    pub law: StructuralLaw,
}

impl fmt::Display for RhythmResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "α={:.3} R²={:.4} rhythm={} mean_interval={:.1} (n={})",
            self.alpha, self.r_squared, self.rhythm, self.mean_interval, self.n_intervals)
    }
}

/// Convert sorted timestamps to inter-event intervals.
pub fn intervals_from_timestamps(timestamps: &[f64]) -> Vec<f64> {
    if timestamps.len() < 2 { return Vec::new(); }
    timestamps.windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .filter(|&d| d > 0.0)
        .collect()
}

/// Analyze the rhythm of an interval sequence.
pub fn rhythm_analyze(intervals: &[f64]) -> RhythmResult {
    let n = intervals.len();
    let mean = if n > 0 { intervals.iter().sum::<f64>() / n as f64 } else { 0.0 };
    let dfa_result = dfa(intervals);
    let law = analyze(intervals);
    let rhythm = if dfa_result.r_squared < 0.3 || n < 64 {
        RhythmType::Random
    } else if dfa_result.alpha > 0.7 {
        RhythmType::Bursty
    } else if dfa_result.alpha > 0.55 {
        RhythmType::Natural
    } else if dfa_result.alpha > 0.45 {
        RhythmType::Random
    } else {
        RhythmType::Metronomic
    };
    RhythmResult {
        alpha: dfa_result.alpha,
        r_squared: dfa_result.r_squared,
        rhythm,
        n_intervals: n,
        mean_interval: mean,
        law,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intervals_basic() {
        let ts = vec![0.0, 1.0, 3.0, 6.0, 10.0];
        let ints = intervals_from_timestamps(&ts);
        assert_eq!(ints, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn rhythm_of_noise() {
        let mut state = 42u64;
        let intervals: Vec<f64> = (0..512).map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as f64 / (1u64 << 31) as f64) * 10.0 + 0.1
        }).collect();
        let result = rhythm_analyze(&intervals);
        assert!(result.alpha > 0.3 && result.alpha < 0.7,
            "white noise should be near 0.5, got {}", result.alpha);
    }
}
