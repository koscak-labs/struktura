//! Structural fingerprint — the "DNA" of a signal.
//!
//! Combines DFA alpha, MFDFA width, ACR decay, kurtosis, and trend
//! into a compact signature that uniquely characterizes a signal's
//! structural properties. Two signals with the same fingerprint
//! have the same structural behavior regardless of amplitude or offset.
//!
//! ```
//! use struktura::fingerprint::fingerprint;
//! # let signal = vec![1.0; 256];
//! let fp = fingerprint(&signal);
//! println!("{}", fp); // "α=0.72 w=0.08 k=3.1 H=0.83 [CORRELATED|monofractal]"
//! ```

#[cfg(not(feature = "std"))]
use alloc::string::String;
#[cfg(not(feature = "std"))]
use alloc::format;

use crate::analyze;
use crate::mfdfa::mfdfa;
use crate::classify::{classify, SignalType};
use core::fmt;

/// Compact structural fingerprint of a signal.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Fingerprint {
    pub alpha: f64,
    pub r_squared: f64,
    pub hurst: f64,
    pub kurtosis: f64,
    pub mfdfa_width: f64,
    pub is_multifractal: bool,
    pub signal_type: SignalType,
    pub n: usize,
}

impl Fingerprint {
    /// How similar are two fingerprints? Returns 0.0 (identical) to 1.0+ (very different).
    pub fn distance(&self, other: &Fingerprint) -> f64 {
        let da = (self.alpha - other.alpha).abs();
        let dw = (self.mfdfa_width - other.mfdfa_width).abs();
        let dk = ((self.kurtosis - other.kurtosis) / (self.kurtosis.max(other.kurtosis).max(1.0))).abs();
        let dh = (self.hurst - other.hurst).abs();
        (da * da + dw * dw + dk * dk * 0.1 + dh * dh).sqrt()
    }
}

impl fmt::Display for Fingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "α={:.3} w={:.3} k={:.1} H={:.3} [{}|{}]",
            self.alpha, self.mfdfa_width, self.kurtosis, self.hurst,
            self.signal_type,
            if self.is_multifractal { "multifractal" } else { "monofractal" })
    }
}

/// Compute the structural fingerprint of a signal.
pub fn fingerprint(values: &[f64]) -> Fingerprint {
    let law = analyze(values);
    let cls = classify(values);
    let qs = [-3.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let spectrum = mfdfa(values, &qs);

    Fingerprint {
        alpha: law.dfa.alpha,
        r_squared: law.dfa.r_squared,
        hurst: law.hurst,
        kurtosis: law.kurtosis,
        mfdfa_width: spectrum.width,
        is_multifractal: spectrum.is_multifractal,
        signal_type: cls.signal_type,
        n: law.n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_signal_same_fingerprint() {
        let mut state = 42u64;
        let data: Vec<f64> = (0..2048).map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as f64 / (1u64 << 31) as f64 - 0.5
        }).collect();
        let fp1 = fingerprint(&data);
        let fp2 = fingerprint(&data);
        assert!(fp1.distance(&fp2) < 1e-10, "same signal should have distance 0");
    }

    #[test]
    fn different_signals_different_fingerprints() {
        let mut state = 42u64;
        let noise: Vec<f64> = (0..2048).map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as f64 / (1u64 << 31) as f64 - 0.5
        }).collect();
        let mut sum = 0.0;
        let brownian: Vec<f64> = noise.iter().map(|&v| { sum += v; sum }).collect();

        let fp_n = fingerprint(&noise);
        let fp_b = fingerprint(&brownian);
        assert!(fp_n.distance(&fp_b) > 0.3,
            "white noise and brownian should be far apart: {:.3}", fp_n.distance(&fp_b));
    }
}
