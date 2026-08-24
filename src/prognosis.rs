//! Prognosis: time-to-threshold estimation from a health-metric trajectory.
//!
//! Fits a least-squares line to the most recent `fit_window` points of a
//! health metric (RMS, DFA α, any scalar trend) and extrapolates to a
//! failure threshold. Returns the ETA in samples together with a 1-sigma
//! interval derived from the regression's slope uncertainty — an honest
//! "how much should you trust this" alongside every prediction.
//!
//! A prediction is only issued when the fitted trend actually moves toward
//! the threshold and the slope is distinguishable from zero (|slope| >
//! its own standard error); otherwise `None` — "no imminent failure
//! evidence" is a valid and important output.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::sqrt;

/// A time-to-threshold estimate.
///
/// Honesty note: the 1-sigma band covers SLOPE NOISE only. When the true
/// degradation is nonlinear (e.g. the IMS bearing's temporary "healing"
/// plateau, where a spall smooths and RMS dips before the final rise), the
/// linear model's error exceeds the band — measured on IMS: predictions
/// -66/+91 recordings (~±13 hours on a 164-hour run) with ±10-recording
/// bands. Treat the band as a floor on uncertainty, never a ceiling.
#[derive(Debug, Clone, Copy)]
pub struct Eta {
    /// Predicted samples until the metric crosses the threshold.
    pub eta: f64,
    /// 1-sigma lower/upper bounds on the ETA from slope uncertainty.
    pub eta_low: f64,
    pub eta_high: f64,
    /// Fitted slope per sample.
    pub slope: f64,
}

/// Estimate time-to-threshold from the last `fit_window` points of `series`.
///
/// `threshold` is the metric level defined as failure. Returns `None` when
/// there is no statistically resolvable trend toward the threshold.
pub fn time_to_threshold(series: &[f64], fit_window: usize, threshold: f64) -> Option<Eta> {
    let n = series.len();
    if n < fit_window || fit_window < 8 {
        return None;
    }
    let w = &series[n - fit_window..];
    let m = fit_window as f64;

    // Least squares y = a + b x over x = 0..fit_window
    let sx = m * (m - 1.0) / 2.0;
    let sx2 = m * (m - 1.0) * (2.0 * m - 1.0) / 6.0;
    let sy: f64 = w.iter().sum();
    let sxy: f64 = w.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
    let det = m * sx2 - sx * sx;
    if det.abs() < 1e-12 {
        return None;
    }
    let a = (sx2 * sy - sx * sxy) / det;
    let b = (m * sxy - sx * sy) / det;

    // Residual variance and slope standard error
    let mut ss = 0.0;
    for (i, &y) in w.iter().enumerate() {
        let r = y - (a + b * i as f64);
        ss += r * r;
    }
    let dof = (m - 2.0).max(1.0);
    let s2 = ss / dof;
    let mean_x = sx / m;
    let sxx = sx2 - m * mean_x * mean_x;
    let se_b = sqrt(s2 / sxx.max(1e-12));

    // Current fitted level at the window's end
    let current = a + b * (m - 1.0);
    let remaining = threshold - current;

    // Trend must move toward the threshold and be resolvable from noise.
    if b.abs() <= se_b || remaining.signum() != b.signum() {
        return None;
    }

    let eta = remaining / b;
    // Slope uncertainty propagated to ETA (1-sigma band).
    let b_lo = b - se_b;
    let b_hi = b + se_b;
    // Same-sign guard: the band stays on this side of zero (checked above).
    let mut bounds = [remaining / b_hi, remaining / b_lo];
    if bounds[0] > bounds[1] {
        bounds.swap(0, 1);
    }
    Some(Eta { eta, eta_low: bounds[0], eta_high: bounds[1], slope: b })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_ramp_predicts_exactly() {
        // y = 0.01 x: from x=99, threshold 2.0 → crossing at x=200, ETA=101.
        let series: Vec<f64> = (0..100).map(|i| 0.01 * i as f64).collect();
        let eta = time_to_threshold(&series, 50, 2.0).expect("trend exists");
        assert!((eta.eta - 101.0).abs() < 1.0, "eta {}", eta.eta);
        assert!(eta.eta_low <= eta.eta && eta.eta <= eta.eta_high);
    }

    #[test]
    fn flat_noise_returns_none() {
        let mut state = 12345u64;
        let series: Vec<f64> = (0..200)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (state >> 33) as f64 / (1u64 << 31) as f64 - 0.5
            })
            .collect();
        assert!(time_to_threshold(&series, 100, 10.0).is_none());
    }

    #[test]
    fn trend_away_from_threshold_returns_none() {
        let series: Vec<f64> = (0..100).map(|i| -0.01 * i as f64).collect();
        assert!(time_to_threshold(&series, 50, 2.0).is_none());
    }
}
