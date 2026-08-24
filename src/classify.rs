//! Signal classification — what kind of signal is this?
//!
//! Auto-classifies a signal based on its DFA alpha into one of the
//! canonical noise types. People know what "pink noise" and "brownian
//! motion" mean — this bridges the gap between DFA numbers and intuition.
//!
//! ```
//! use struktura::classify::classify;
//! let data: Vec<f64> = (0..256).map(|i| (i as f64 * 0.1).sin()).collect();
//! let result = classify(&data);
//! println!("{}", result.signal_type); // e.g. "CORRELATED"
//! ```

#[cfg(not(feature = "std"))]
use alloc::string::String;

use crate::{analyze, StructuralLaw};
use core::fmt;

/// Canonical signal type based on DFA scaling exponent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalType {
    /// α < 0.3 — anti-correlated (successive values tend to alternate)
    AntiCorrelated,
    /// α ≈ 0.5 — white noise (uncorrelated random)
    WhiteNoise,
    /// 0.5 < α < 0.85 — correlated (pink/1/f noise family)
    Correlated,
    /// 0.85 < α < 1.15 — 1/f noise (the boundary between stationary and non-stationary)
    OneOverF,
    /// 1.15 < α < 1.65 — brownian motion / random walk
    Brownian,
    /// α > 1.65 — strongly persistent (trending/drifting)
    Persistent,
    /// R² too low — signal has no clear scaling law
    Unclassifiable,
}

impl fmt::Display for SignalType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SignalType::AntiCorrelated => write!(f, "ANTI-CORRELATED"),
            SignalType::WhiteNoise => write!(f, "WHITE NOISE"),
            SignalType::Correlated => write!(f, "CORRELATED (pink/1/f family)"),
            SignalType::OneOverF => write!(f, "1/f NOISE (scale-free)"),
            SignalType::Brownian => write!(f, "BROWNIAN MOTION (random walk)"),
            SignalType::Persistent => write!(f, "PERSISTENT (trending)"),
            SignalType::Unclassifiable => write!(f, "UNCLASSIFIABLE (no clear scaling)"),
        }
    }
}

/// Result of signal classification.
#[derive(Debug, Clone)]
pub struct ClassifyResult {
    pub signal_type: SignalType,
    pub alpha: f64,
    pub r_squared: f64,
    pub law: StructuralLaw,
    pub description: &'static str,
}

impl fmt::Display for ClassifyResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "α={:.3} → {} (R²={:.3})", self.alpha, self.signal_type, self.r_squared)
    }
}

/// Classify a signal based on its DFA scaling exponent.
pub fn classify(values: &[f64]) -> ClassifyResult {
    let law = analyze(values);
    let alpha = law.dfa.alpha;
    let r2 = law.dfa.r_squared;

    let (signal_type, description) = if r2 < 0.3 {
        (SignalType::Unclassifiable, "No clear scaling law — possibly chaotic, periodic, or too short")
    } else if alpha < 0.3 {
        (SignalType::AntiCorrelated, "Values tend to alternate — a rise is followed by a fall")
    } else if alpha < 0.6 {
        (SignalType::WhiteNoise, "Uncorrelated random values — no memory, no trend")
    } else if alpha < 0.85 {
        (SignalType::Correlated, "Long-range correlations — like heartbeats, music, natural processes")
    } else if alpha < 1.15 {
        (SignalType::OneOverF, "Scale-free 1/f noise — equal energy at every scale, the most natural")
    } else if alpha < 1.65 {
        (SignalType::Brownian, "Random walk — cumulative random process, like stock prices or diffusion")
    } else {
        (SignalType::Persistent, "Strongly trending — each value builds on the last, high persistence")
    };

    ClassifyResult { signal_type, alpha, r_squared: r2, law, description }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn white_noise_classified_correctly() {
        let mut state = 42u64;
        let data: Vec<f64> = (0..4096).map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as f64 / (1u64 << 31) as f64 - 0.5
        }).collect();
        let result = classify(&data);
        assert!(result.alpha > 0.3 && result.alpha < 0.65,
            "white noise should classify as white noise, got α={:.3}", result.alpha);
    }

    #[test]
    fn brownian_classified_correctly() {
        let mut state = 42u64;
        let mut sum = 0.0;
        let data: Vec<f64> = (0..4096).map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            sum += (state >> 33) as f64 / (1u64 << 31) as f64 - 0.5;
            sum
        }).collect();
        let result = classify(&data);
        assert!(result.alpha > 1.1,
            "brownian motion should have α > 1.1, got {:.3}", result.alpha);
    }
}
