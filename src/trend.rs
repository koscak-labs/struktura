//! Trend detection — is the signal getting worse over time?
//!
//! Runs DFA on a sliding window across the signal and fits a trend line
//! to the resulting α values. A declining α means the structure is
//! degrading progressively — the early warning before a sudden failure.
//!
//! ```
//! use struktura::trend::alpha_trend;
//! # let signal = vec![1.0; 1024];
//! let trend = alpha_trend(&signal, 256, 64);
//! println!("slope: {:.6}/sample — {}", trend.slope, trend.direction);
//! ```

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::dfa;
use core::fmt;

/// Direction of the α trend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendDirection {
    /// α is increasing (structure strengthening)
    Improving,
    /// α is stable (within ±0.0001/sample)
    Stable,
    /// α is decreasing (structure degrading)
    Degrading,
}

impl fmt::Display for TrendDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrendDirection::Improving => write!(f, "IMPROVING"),
            TrendDirection::Stable => write!(f, "STABLE"),
            TrendDirection::Degrading => write!(f, "DEGRADING"),
        }
    }
}

/// Result of a trend analysis.
#[derive(Debug, Clone)]
pub struct TrendResult {
    /// Slope of α over time (per sample). Negative = degrading.
    pub slope: f64,
    /// R² of the trend line fit.
    pub r_squared: f64,
    /// Direction classification.
    pub direction: TrendDirection,
    /// α values at each window position.
    pub alpha_series: Vec<f64>,
    /// Starting α (first window).
    pub alpha_start: f64,
    /// Ending α (last window).
    pub alpha_end: f64,
}

impl fmt::Display for TrendResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "α {:.3}→{:.3} slope={:.6}/sample R²={:.3} {}",
            self.alpha_start, self.alpha_end, self.slope, self.r_squared, self.direction)
    }
}

/// Compute the trend of DFA α across a sliding window.
///
/// `window_size`: number of samples per DFA computation (≥64).
/// `step`: how many samples to advance between windows.
///
/// Returns the slope of α over time — negative means degradation.
pub fn alpha_trend(signal: &[f64], window_size: usize, step: usize) -> TrendResult {
    let window_size = window_size.max(64);
    let step = step.max(1);
    let n = signal.len();

    let mut alphas = Vec::new();
    let mut pos = 0;
    while pos + window_size <= n {
        let result = dfa(&signal[pos..pos + window_size]);
        if result.r_squared > 0.3 {
            alphas.push(result.alpha);
        }
        pos += step;
    }

    if alphas.len() < 3 {
        return TrendResult {
            slope: 0.0, r_squared: 0.0,
            direction: TrendDirection::Stable,
            alpha_series: alphas.clone(),
            alpha_start: alphas.first().copied().unwrap_or(0.5),
            alpha_end: alphas.last().copied().unwrap_or(0.5),
        };
    }

    let k = alphas.len() as f64;
    let (mut sx, mut sy, mut sxy, mut sx2) = (0.0, 0.0, 0.0, 0.0);
    for (i, &a) in alphas.iter().enumerate() {
        let x = i as f64;
        sx += x; sy += a; sxy += x * a; sx2 += x * x;
    }
    let slope = (k * sxy - sx * sy) / (k * sx2 - sx * sx);
    let ic = (sy - slope * sx) / k;
    let ym = sy / k;
    let mut sst = 0.0;
    let mut ssr = 0.0;
    for (i, &a) in alphas.iter().enumerate() {
        sst += (a - ym) * (a - ym);
        ssr += (a - slope * i as f64 - ic) * (a - slope * i as f64 - ic);
    }
    let r2 = if sst > 1e-15 { 1.0 - ssr / sst } else { 0.0 };

    let slope_per_sample = slope / step as f64;
    let direction = if slope_per_sample.abs() < 0.0001 {
        TrendDirection::Stable
    } else if slope_per_sample < 0.0 {
        TrendDirection::Degrading
    } else {
        TrendDirection::Improving
    };

    TrendResult {
        slope: slope_per_sample,
        r_squared: r2,
        direction,
        alpha_start: alphas[0],
        alpha_end: *alphas.last().unwrap(),
        alpha_series: alphas,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_signal_has_stable_trend() {
        let mut state = 42u64;
        let data: Vec<f64> = (0..4096).map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as f64 / (1u64 << 31) as f64 - 0.5
        }).collect();
        let result = alpha_trend(&data, 512, 128);
        assert!(result.slope.abs() < 0.001, "stable noise should have near-zero slope, got {}", result.slope);
    }

    #[test]
    fn degrading_signal_has_negative_trend() {
        let mut state = 42u64;
        let mut prev = 0.0f64;
        let mut data = Vec::with_capacity(4096);
        for i in 0..4096 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let noise = (state >> 33) as f64 / (1u64 << 31) as f64 - 0.5;
            let correlation = if i < 2048 { 0.8 } else { 0.8 - (i - 2048) as f64 * 0.0003 };
            prev = prev * correlation.max(0.0) + noise * (1.0 - correlation.max(0.0));
            data.push(prev);
        }
        let result = alpha_trend(&data, 512, 128);
        assert!(result.direction == TrendDirection::Degrading || result.slope < 0.0,
            "signal with decreasing correlation should show degrading trend, got {:?}", result.direction);
    }
}
