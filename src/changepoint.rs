//! Changepoint detection: locates WHERE the structure changed.
//!
//! Reports "sample 3,847 is where the bearing started degrading," giving
//! a location alongside the fact that degradation occurred.
//!
//! ```
//! use struktura::changepoint::find_changepoint;
//! # let signal = vec![1.0; 8192];
//! let cp = find_changepoint(&signal, 1024);
//! if let Some(result) = cp {
//!     println!("change at sample {}: α went from {:.3} to {:.3}",
//!         result.location, result.alpha_before, result.alpha_after);
//! }
//! ```
//!
//! # Method
//!
//! The signal is cut into non-overlapping [`BLOCK`]-sample blocks and α
//! is estimated on each. The block-to-block noise σ is estimated once
//! from the median absolute first difference of the block-α sequence
//! (see [`noise_sigma`]): jumps at the few real changepoints are ignored
//! by the median, so σ measures only estimator noise, whichever regimes
//! the signal contains. A candidate split at block boundary `j` then
//! compares the mean block-α before `j` with the mean after `j`:
//! `z = |Δmean| / (σ · √(1/k_left + 1/k_right))`. The split with the
//! largest z wins if z ≥ [`Z_THRESHOLD`] and the mean difference is at
//! least [`MIN_ALPHA_DIFF`]. Location resolution is one block.
//!
//! A Welch test on each side's own variance was tried first; it fails on
//! three-regime signals because the side that straddles two regimes has
//! a large variance and the true boundary scores below threshold.
//!
//! # Why not `dfa(left)` vs `dfa(right)`
//!
//! The earlier implementation ran DFA on the whole left slice and the
//! whole right slice of every split. Two things go wrong:
//!
//! 1. DFA's box sizes scale with segment length (n/50 … n/4), so a
//!    1K-sample side is measured at scales 20–256 and a 120K-sample side
//!    at 2400–30000. For any process with a scaling crossover (AR-like
//!    short-range correlation, most real telemetry) those α values differ
//!    legitimately, and the scan reports a "change" at every early split.
//!    Stationary AR(1) φ=0.95 produced a changepoint at every 1024
//!    boundary; a stationary rover force channel produced 86.
//! 2. A short side's α is noisy (std 0.14 at 128 samples) and biased
//!    (−0.12 at 128), so a fixed shift threshold (the old 0.03) is passed
//!    by noise alone.
//!
//! Scoring every block at one scale removes (1); averaging blocks per
//! side and testing against the within-side spread removes (2).

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::{dfa, sqrt};
use core::fmt;

/// Block length on which α is estimated. At 1024 samples the DFA fit has
/// ~4 box sizes (20–256) and std(α) ≈ 0.05 on 1/f noise; at 128 it is
/// 0.14 and biased.
pub const BLOCK: usize = 1024;

/// Kept for callers that used the old name.
pub const MIN_SEGMENT: usize = BLOCK;

/// Fewest blocks per side: a variance needs at least this many.
pub const MIN_BLOCKS_PER_SIDE: usize = 3;

/// z-score a split must reach. The scan takes the maximum over ~k
/// candidate splits, so this is deliberately above the usual 3.
pub const Z_THRESHOLD: f64 = 4.0;

/// Smallest mean-α difference that counts as a structural change, even
/// when statistically resolvable from many blocks.
pub const MIN_ALPHA_DIFF: f64 = 0.10;

/// Fewest samples on which detection is attempted at all.
pub const MIN_SAMPLES: usize = 2 * MIN_BLOCKS_PER_SIDE * BLOCK;

/// A detected changepoint in the signal.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Changepoint {
    /// Sample index where the change occurred (a multiple of [`BLOCK`]
    /// relative to the start of the analysed slice).
    pub location: usize,
    /// Mean block-α before the changepoint.
    pub alpha_before: f64,
    /// Mean block-α after the changepoint.
    pub alpha_after: f64,
    /// `alpha_after - alpha_before`.
    pub shift: f64,
    /// z-score of the shift against block-to-block noise (≥ [`Z_THRESHOLD`]).
    pub confidence: f64,
}

impl fmt::Display for Changepoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "changepoint at sample {}: α {:.3}→{:.3} (shift={:+.3}, z={:.1})",
            self.location, self.alpha_before, self.alpha_after, self.shift, self.confidence)
    }
}

/// α of each full [`BLOCK`] in `signal`, with its block index. Blocks
/// whose DFA fit has R² ≤ 0.3 are dropped.
pub fn block_alphas(signal: &[f64]) -> Vec<(usize, f64)> {
    signal
        .chunks_exact(BLOCK)
        .enumerate()
        .map(|(i, c)| (i, dfa(c)))
        .filter(|(_, r)| r.r_squared > 0.3)
        .map(|(i, r)| (i, r.alpha))
        .collect()
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn median_in_place(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let m = v.len() / 2;
    if v.len() % 2 == 0 { (v[m - 1] + v[m]) / 2.0 } else { v[m] }
}

/// Block-to-block noise of a block-α sequence: the median absolute first
/// difference, scaled to a standard deviation (÷0.6745 for the normal
/// MAD, ÷√2 because a difference of two estimates has twice the
/// variance). Floored at 0.02.
pub fn noise_sigma(alphas: &[f64]) -> f64 {
    let mut d: Vec<f64> = alphas.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
    if d.is_empty() { return 0.02; }
    let sigma = median_in_place(&mut d) / 0.6745 / core::f64::consts::SQRT_2;
    if sigma < 0.02 { 0.02 } else { sigma }
}

/// z-score for the difference of two block-α means given the
/// block-to-block noise `sigma`.
pub fn z_score(left: &[f64], right: &[f64], sigma: f64) -> f64 {
    let se = sigma * sqrt(1.0 / left.len() as f64 + 1.0 / right.len() as f64);
    (mean(right) - mean(left)).abs() / se
}

/// Find the single most significant changepoint in a signal.
///
/// `min_segment` is the fewest samples allowed on either side; it is
/// rounded up to whole blocks and never below [`MIN_BLOCKS_PER_SIDE`].
///
/// Returns `None` if the signal has fewer than [`MIN_SAMPLES`] usable
/// samples or no split reaches [`Z_THRESHOLD`] with a mean difference of
/// at least [`MIN_ALPHA_DIFF`].
pub fn find_changepoint(signal: &[f64], min_segment: usize) -> Option<Changepoint> {
    let blocks = block_alphas(signal);
    let alphas: Vec<f64> = blocks.iter().map(|b| b.1).collect();
    let k = alphas.len();
    let min_side = ((min_segment + BLOCK - 1) / BLOCK).max(MIN_BLOCKS_PER_SIDE);
    if k < 2 * min_side { return None; }

    let sigma = noise_sigma(&alphas);
    let mut best: Option<Changepoint> = None;
    let mut best_z = Z_THRESHOLD;
    for j in min_side..=(k - min_side) {
        let (l, r) = alphas.split_at(j);
        let ml = mean(l);
        let mr = mean(r);
        if (mr - ml).abs() < MIN_ALPHA_DIFF { continue; }
        let z = z_score(l, r, sigma);
        if z > best_z {
            best_z = z;
            best = Some(Changepoint {
                location: blocks[j].0 * BLOCK,
                alpha_before: ml,
                alpha_after: mr,
                shift: mr - ml,
                confidence: z,
            });
        }
    }
    best
}

/// Find up to `max_points` changepoints by recursive binary splitting.
///
/// Splits at the strongest changepoint, then searches each side. Stops
/// when `max_points` have been found or no segment clears the test.
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
    max_total: usize,
    results: &mut Vec<Changepoint>,
) {
    if results.len() >= max_total || signal.len() < MIN_SAMPLES { return; }
    if let Some(mut cp) = find_changepoint(signal, min_segment) {
        let split = cp.location;
        cp.location += offset;
        results.push(cp);
        find_changepoints_recursive(&signal[..split], offset, min_segment, max_total, results);
        find_changepoints_recursive(&signal[split..], offset + split, min_segment, max_total, results);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(state: &mut u64) -> f64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*state >> 33) as f64 / (1u64 << 31) as f64 - 0.5
    }

    /// White noise (α≈0.5) followed by its running sum (α≈1.5): a shift
    /// visible at every scale.
    fn white_then_walk(n: usize, change_at: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        let mut out = Vec::with_capacity(n);
        let mut acc = 0.0;
        for i in 0..n {
            let e = lcg(&mut state);
            if i < change_at { out.push(e); } else { acc += e; out.push(acc * 0.05); }
        }
        out
    }

    #[test]
    fn detects_changepoint_in_synthetic() {
        let signal = white_then_walk(16384, 8192, 42);
        let cp = find_changepoint(&signal, 1024).expect("should detect a changepoint");
        assert!(cp.shift > 0.5, "shift should be large and positive: {:.3}", cp.shift);
        assert!((cp.location as i64 - 8192).abs() <= BLOCK as i64, "location {} not near 8192", cp.location);
        assert!(cp.confidence >= Z_THRESHOLD);
    }

    #[test]
    fn no_changepoint_in_short_or_stationary() {
        let mut state = 42u64;
        let short: Vec<f64> = (0..4096).map(|_| lcg(&mut state)).collect();
        assert!(find_changepoint(&short, 256).is_none(), "4 blocks cannot form two sides of 3");
        let white: Vec<f64> = (0..65536).map(|_| lcg(&mut state)).collect();
        assert!(find_changepoints(&white, 128, 20).is_empty(), "white noise must give no changepoints");
    }

    /// Stationary AR(1) at φ=0.7 and φ=0.95 must yield zero changepoints.
    /// Under the old whole-side comparison φ=0.95 fired at every 1024
    /// boundary because the sides were measured at different scales.
    #[test]
    fn no_changepoints_in_stationary_ar1_long() {
        for phi in [0.7f64, 0.95] {
            for seed in [1u64, 7, 42] {
                let mut state = seed;
                let mut prev = 0.0f64;
                let signal: Vec<f64> = (0..65536).map(|_| { prev = prev * phi + lcg(&mut state); prev }).collect();
                let cps = find_changepoints(&signal, 128, 20);
                assert!(cps.is_empty(),
                    "phi {} seed {}: stationary AR(1) must give no changepoints, got {:?}",
                    phi, seed, cps.iter().map(|c| (c.location, c.shift, c.confidence)).collect::<Vec<_>>());
            }
        }
    }

    #[test]
    fn multiple_changepoints() {
        // white | walk | white, 8192 each
        let mut signal = white_then_walk(16384, 8192, 42);
        let mut state = 99u64;
        signal.extend((0..8192).map(|_| lcg(&mut state)));
        let cps = find_changepoints(&signal, 1024, 5);
        let locs: Vec<usize> = cps.iter().map(|c| c.location).collect();
        assert!(cps.len() >= 2, "expected >= 2 changepoints, got {:?}", locs);
        assert!(locs.iter().any(|&l| (l as i64 - 8192).abs() <= BLOCK as i64), "missing change near 8192: {:?}", locs);
        assert!(locs.iter().any(|&l| (l as i64 - 16384).abs() <= BLOCK as i64), "missing change near 16384: {:?}", locs);
    }

    #[test]
    fn max_points_is_a_total_cap() {
        let mut signal = white_then_walk(16384, 8192, 42);
        let mut state = 99u64;
        signal.extend((0..8192).map(|_| lcg(&mut state)));
        assert!(find_changepoints(&signal, 1024, 1).len() <= 1);
    }

    #[test]
    fn noise_and_z_basics() {
        // Two regimes with one jump: the median difference ignores the jump.
        let seq = [0.50, 0.52, 0.48, 0.51, 0.49, 1.50, 1.52, 1.48, 1.51, 1.49];
        let sigma = noise_sigma(&seq);
        assert!(sigma < 0.05, "sigma {} should reflect the 0.02-ish wobble, not the 1.0 jump", sigma);
        let (a, b) = seq.split_at(5);
        assert!(z_score(a, b, sigma) > 20.0);
        assert!(z_score(a, a, sigma) < 1e-9);
    }
}
