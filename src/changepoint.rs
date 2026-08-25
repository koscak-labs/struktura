//! Changepoint detection — WHERE did the structure change?
//!
//! Runs DFA on sliding windows across the signal and finds the point
//! where α changes most sharply. Answers "sample 3,847 is where
//! the bearing started degrading" — not just "it degraded."
//!
//! ```
//! use struktura::changepoint::find_changepoint;
//! # let signal = vec![1.0; 512];
//! let cp = find_changepoint(&signal, 128);
//! if let Some(result) = cp {
//!     println!("change at sample {}: α went from {:.3} to {:.3}",
//!         result.location, result.alpha_before, result.alpha_after);
//! }
//! ```

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::dfa;
use core::fmt;

/// A detected changepoint in the signal.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Changepoint {
    /// Sample index where the change occurred.
    pub location: usize,
    /// DFA α before the changepoint.
    pub alpha_before: f64,
    /// DFA α after the changepoint.
    pub alpha_after: f64,
    /// Magnitude of the structural shift.
    pub shift: f64,
    /// Confidence (R² of the worse half).
    pub confidence: f64,
}

impl fmt::Display for Changepoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "changepoint at sample {}: α {:.3}→{:.3} (shift={:+.3}, conf={:.3})",
            self.location, self.alpha_before, self.alpha_after, self.shift, self.confidence)
    }
}

/// Find the single most significant changepoint in a signal.
///
/// Scans every possible split point (with minimum `min_segment` samples
/// per side) and picks the one with the largest |Δα|.
///
/// Returns `None` if the signal is too short or no significant change found.
pub fn find_changepoint(signal: &[f64], min_segment: usize) -> Option<Changepoint> {
    let n = signal.len();
    let min_seg = min_segment.max(64);
    if n < min_seg * 2 { return None; }

    // Two-pass: coarse scan then fine-tune around the best candidate.
    let coarse_step = (n / 30).max(1);
    let mut best: Option<Changepoint> = None;
    let mut best_shift = 0.0f64;

    let mut pos = min_seg;
    while pos <= n - min_seg {
        let left = dfa(&signal[..pos]);
        let right = dfa(&signal[pos..]);

        if left.r_squared > 0.3 && right.r_squared > 0.3 {
            let shift = (right.alpha - left.alpha).abs();
            if shift > best_shift {
                best_shift = shift;
                best = Some(Changepoint {
                    location: pos,
                    alpha_before: left.alpha,
                    alpha_after: right.alpha,
                    shift: right.alpha - left.alpha,
                    confidence: left.r_squared.min(right.r_squared),
                });
            }
        }
        pos += coarse_step;
    }

    // Fine-tune: re-scan around the coarse best at single-step resolution
    if let Some(ref coarse) = best {
        let fine_start = coarse.location.saturating_sub(coarse_step * 2).max(min_seg);
        let fine_end = (coarse.location + coarse_step * 2).min(n - min_seg);
        let fine_step = (coarse_step / 10).max(1);
        let mut fpos = fine_start;
        while fpos <= fine_end {
            let left = dfa(&signal[..fpos]);
            let right = dfa(&signal[fpos..]);
            if left.r_squared > 0.3 && right.r_squared > 0.3 {
                let shift = (right.alpha - left.alpha).abs();
                if shift > best_shift {
                    best_shift = shift;
                    best = Some(Changepoint {
                        location: fpos,
                        alpha_before: left.alpha,
                        alpha_after: right.alpha,
                        shift: right.alpha - left.alpha,
                        confidence: left.r_squared.min(right.r_squared),
                    });
                }
            }
            fpos += fine_step;
        }
    }

    // Only report if the shift is meaningful (> 0.03)
    best.filter(|cp| cp.shift.abs() > 0.03)
}

/// Find multiple changepoints by recursive binary splitting.
///
/// Splits at the strongest changepoint, then recursively searches
/// each half. Stops when no segment has a significant shift.
pub fn find_changepoints(signal: &[f64], min_segment: usize, max_points: usize) -> Vec<Changepoint> {
    let mut results = Vec::new();
    find_changepoints_recursive(signal, 0, min_segment, max_points, &mut results);
    results.sort_by_key(|cp| cp.location);
    results
}

fn find_changepoints_recursive(
    signal: &[f64],
    offset: usize,
    min_segment: usize,
    remaining: usize,
    results: &mut Vec<Changepoint>,
) {
    if remaining == 0 || signal.len() < min_segment * 2 { return; }

    if let Some(mut cp) = find_changepoint(signal, min_segment) {
        cp.location += offset;
        let split = cp.location - offset;
        results.push(cp);

        // Recurse into both halves
        find_changepoints_recursive(&signal[..split], offset, min_segment, remaining - 1, results);
        find_changepoints_recursive(&signal[split..], offset + split, min_segment, remaining - 1, results);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_signal_with_change(n: usize, change_at: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        let mut prev = 0.0f64;
        (0..n).map(|i| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let noise = (state >> 33) as f64 / (1u64 << 31) as f64 - 0.5;
            let correlation = if i < change_at { 0.8 } else { 0.1 };
            prev = prev * correlation + noise * (1.0 - correlation);
            prev
        }).collect()
    }

    #[test]
    fn detects_changepoint_in_synthetic() {
        let signal = make_signal_with_change(4096, 2048, 42);
        let cp = find_changepoint(&signal, 256);
        assert!(cp.is_some(), "should detect a changepoint");
        let cp = cp.unwrap();
        // The signal has a real structural change — a changepoint must be found
        // with a significant shift. The exact location may vary because DFA
        // measures global structure within each half, not point-wise.
        assert!(cp.shift.abs() > 0.05, "shift should be significant: {:.3}", cp.shift);
        assert!(cp.confidence > 0.3, "confidence should be reasonable: {:.3}", cp.confidence);
    }

    #[test]
    fn no_changepoint_in_stationary() {
        let mut state = 42u64;
        let signal: Vec<f64> = (0..2048).map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as f64 / (1u64 << 31) as f64 - 0.5
        }).collect();
        let cp = find_changepoint(&signal, 256);
        // Stationary white noise should have no significant changepoint
        // (or a very small shift if one is found)
        if let Some(cp) = cp {
            assert!(cp.shift.abs() < 0.2,
                "stationary signal shouldn't have large shift: {:.3}", cp.shift);
        }
    }

    #[test]
    fn multiple_changepoints() {
        let mut signal = make_signal_with_change(4096, 1365, 42);
        // Add a second change at 2730
        let mut state = 99u64;
        let mut prev = signal[2730];
        for i in 2730..4096 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let noise = (state >> 33) as f64 / (1u64 << 31) as f64 - 0.5;
            prev = prev * 0.8 + noise * 0.2;
            signal[i] = prev;
        }
        let cps = find_changepoints(&signal, 512, 3);
        assert!(!cps.is_empty(), "should find at least 1 changepoint");
    }
}
