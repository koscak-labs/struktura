//! The DFA detector core — reference implementation.
//!
//! This module is the *whole* anomaly detector in its simplest, auditable form:
//! no ML, no training, no hidden state. Just Detrended Fluctuation Analysis
//! (Peng et al., Phys. Rev. E, 1994):
//!
//! 1. remove the mean and take the cumulative sum — the "profile"
//! 2. cut the profile into boxes, linearly detrend each, measure residual fluctuation
//! 3. fit a line in log-log space — its slope is `alpha`, the scaling exponent
//!
//! When `alpha` shifts from your baseline, the signal's structure changed.
//!
//! The shipped [`crate::dfa_into`] / [`crate::dfa_fast_into`] are allocation-free,
//! prefix-sum-optimized equivalents of [`dfa_reference`]. They are verified to
//! agree with this reference to 1e-9 (see [`tests::reference_matches_production`]),
//! so what you read here is exactly what runs in production — there is no hidden
//! model doing the "heavy lifting".

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::{linreg, ln, powf, powi, sqrt, DfaResult};

/// Canonical DFA reference implementation — the entire detector, ~68 lines of code
/// (the DFA math proper is the box loop and the log-log fit).
///
/// Mirrors the textbook algorithm exactly so the method is auditable at a glance.
/// For hot paths prefer [`crate::dfa`] (reuses a caller buffer) or
/// [`crate::dfa_fast_into`] (prefix-sum, zero-alloc).
///
/// ```
/// use struktura::dfa_core::dfa_reference;
/// // a smooth trend has strong long-range correlation -> high alpha
/// let trend: Vec<f64> = (0..512).map(|i| i as f64 * 0.01).collect();
/// assert!(dfa_reference(&trend).alpha > 1.0);
/// // white noise has no structure -> alpha near 0.5
/// // (same PCG-style LCG the crate's own tests use; the low bits of a plain
/// //  LCG are strongly anti-correlated and give alpha ~0.15, not ~0.5)
/// fn white_noise(n: usize, seed: u64) -> Vec<f64> {
///     let mut state = seed;
///     (0..n).map(|_| {
///         state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
///         (state >> 33) as f64 / (1u64 << 31) as f64 - 0.5
///     }).collect()
/// }
/// let r = dfa_reference(&white_noise(4096, 42));
/// assert!(r.alpha > 0.35 && r.alpha < 0.65, "alpha was {}", r.alpha);
/// ```
#[must_use]
pub fn dfa_reference(values: &[f64]) -> DfaResult {
    let n = values.len();
    if n < 64 {
        return DfaResult { alpha: 0.5, r_squared: 0.0 };
    }
    let mean = values.iter().sum::<f64>() / n as f64;

    // 1. cumulative profile y_j = cumsum(x - mean)
    let mut profile = Vec::with_capacity(n);
    let mut cum = 0.0;
    for &v in values {
        cum += v - mean;
        profile.push(cum);
    }

    // 2. adaptive box sizes: geometric spacing from max(16, n/50) to n/4
    let s_min = 16usize.max(n / 50);
    let s_max = n / 4;
    if s_min >= s_max {
        return DfaResult { alpha: 0.5, r_squared: 0.0 };
    }
    let ratio = powf(s_max as f64 / s_min as f64, 1.0 / 11.0);

    let mut log_s = [0.0f64; 12];
    let mut log_f = [0.0f64; 12];
    let mut pts = 0usize;
    let mut prev_s = 0usize;

    for step in 0..12 {
        let s = (s_min as f64 * powi(ratio, step)) as usize;
        if s == prev_s || s > s_max {
            continue;
        }
        prev_s = s;
        let num_segs = n / s;
        if num_segs == 0 {
            continue;
        }
        // least-squares constants for a line fit inside a box of length s
        let k = s as f64;
        let sx = k * (k - 1.0) / 2.0;
        let sx2 = k * (k - 1.0) * (2.0 * k - 1.0) / 6.0;
        let det = k * sx2 - sx * sx;
        if det.abs() < 1e-15 {
            continue;
        }
        let mut f2_sum = 0.0;
        for seg in 0..num_segs {
            let start = seg * s;
            let mut sy = 0.0;
            let mut sxy = 0.0;
            let mut sy2 = 0.0;
            for i in 0..s {
                let yi = profile[start + i];
                sy += yi;
                sxy += i as f64 * yi;
                sy2 += yi * yi;
            }
            let a0 = (sx2 * sy - sx * sxy) / det;
            let a1 = (k * sxy - sx * sy) / det;
            let resid = (sy2 - a0 * sy - a1 * sxy).max(0.0);
            f2_sum += resid / k;
        }
        let f = sqrt(f2_sum / num_segs as f64);
        if f > 0.0 {
            log_s[pts] = ln(s as f64);
            log_f[pts] = ln(f);
            pts += 1;
        }
    }

    if pts < 3 {
        return DfaResult { alpha: 0.5, r_squared: 0.0 };
    }
    linreg(&log_s[..pts], &log_f[..pts])
}

/// Re-export the production DFA entry points so the whole core lives under one path.
pub use crate::{dfa, dfa_fast_into, dfa_into};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{dfa_fast_into, dfa_into};

    /// The reference impl and the optimized production impl must agree.
    /// This is the proof behind the "no hidden model" claim.
    #[test]
    fn reference_matches_production() {
        let a: Vec<f64> = (0..1024)
            .map(|i| ((i as f64 * 1103515245.0 + 12345.0) % 65536.0) / 65536.0 - 0.5)
            .collect();
        let b: Vec<f64> = (0..2048).map(|i| (i as f64 * 0.013).sin()).collect();
        let c: Vec<f64> = (0..4096)
            .map(|i| (i as f64 * 0.007).sin() + (i as f64 * 0.0003))
            .collect();
        for sig in [&a, &b, &c] {
            let r = dfa_reference(sig);
            let mut buf = Vec::new();
            let p = dfa_into(sig, &mut buf);
            let mut buf2 = Vec::new();
            let q = dfa_fast_into(sig, &mut buf2);
            assert!(
                (r.alpha - p.alpha).abs() < 1e-9,
                "reference vs dfa_into alpha mismatch: {} vs {}",
                r.alpha,
                p.alpha
            );
            assert!((r.r_squared - p.r_squared).abs() < 1e-9);
            assert!((p.alpha - q.alpha).abs() < 1e-9, "dfa_into vs dfa_fast_into differ");
        }
    }

    #[test]
    fn smooth_trend_has_high_alpha() {
        let trend: Vec<f64> = (0..512).map(|i| i as f64 * 0.01).collect();
        assert!(dfa_reference(&trend).alpha > 1.0);
    }

    #[test]
    fn short_signal_is_safe() {
        let tiny = [1.0, 2.0, 3.0];
        let r = dfa_reference(&tiny);
        assert!(r.r_squared == 0.0);
    }
}
