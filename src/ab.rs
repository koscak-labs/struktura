//! A/B level comparison: is sample B higher, lower, or the same as sample A?
//!
//! [`compare`](crate::compare) measures a change in correlation structure
//! (the DFA exponent α). It says nothing about level: two series of
//! per-request latencies can differ by +84% in their median and still have
//! the same α. This module compares the level of two independent samples
//! and is meant for benchmark-style A/B runs (step times, throughput, run
//! durations), where the values are heavy-tailed and n is 10 to a few
//! thousand.
//!
//! What it reports for B against A:
//!
//! - a location statistic per arm: median (default), mean, or 20% trimmed mean;
//! - the percent difference `100 * (stat_b - stat_a) / |stat_a|`;
//! - a 95% percentile-bootstrap interval for that percent difference, each arm
//!   resampled independently with a seeded generator, so the same seed gives
//!   the same interval;
//! - the Mann-Whitney U test, two-sided, normal approximation with tie and
//!   continuity correction;
//! - Cliff's delta, `P(b > a) - P(b < a)` over all pairs.
//!
//! The verdict uses the interval and a smallest effect worth reporting
//! (`min_effect_pct`):
//!
//! - [`AbVerdict::Higher`] / [`AbVerdict::Lower`]: the interval excludes 0
//!   and the point estimate is at least `min_effect_pct` away from 0;
//! - [`AbVerdict::Equivalent`]: the whole interval lies within
//!   `±min_effect_pct`;
//! - [`AbVerdict::Inconclusive`]: anything else, and always when an arm has
//!   fewer than [`MIN_N_VERDICT`] values.
//!
//! The percentile bootstrap undercovers at small n. With A and B drawn from
//! the same heavy-tailed distribution (`examples/ab_null_rate.rs`, 1,000
//! pairs per n) it called a direction in 10.7% of pairs at n = 3 and 6.8% at
//! n = 4, and in 2.4-4.6% from n = 5 to 100; hence the floor. Below
//! [`SMALL_N`] values the result also carries a warning
//! ([`AbResult::small_sample`]). The samples are treated as exchangeable within
//! each arm: autocorrelated series (a warm-up drift, a slow leak) make the
//! interval too narrow.
//!
//! ```
//! use struktura::ab::{ab_compare, AbConfig, AbVerdict};
//! let a: Vec<f64> = (0..200).map(|i| 70.0 + (i % 17) as f64).collect();
//! let b: Vec<f64> = a.iter().map(|x| x * 1.5).collect();
//! let r = ab_compare(&a, &b, &AbConfig::default()).unwrap();
//! assert_eq!(r.verdict, AbVerdict::Higher);
//! assert!((r.diff_pct - 50.0).abs() < 1e-9);
//! ```

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use core::fmt;

/// Fewer values than this in either arm: the result carries a small-sample
/// warning ([`AbResult::small_sample`]).
pub const SMALL_N: usize = 10;

/// Fewer values than this in either arm: the verdict is always
/// [`AbVerdict::Inconclusive`] (see the module docs for the measurement).
pub const MIN_N_VERDICT: usize = 5;

/// Fraction trimmed from each end by [`Statistic::Trimmed`].
pub const TRIM_FRACTION: f64 = 0.2;

/// Location statistic compared between the arms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Statistic {
    /// Median (average of the two middle values for even n).
    Median,
    /// Arithmetic mean. Sensitive to the tail.
    Mean,
    /// Mean of the values left after dropping [`TRIM_FRACTION`] from each end.
    Trimmed,
}

impl Statistic {
    /// Lower-case name, as accepted by the CLI's `--stat`.
    pub fn name(self) -> &'static str {
        match self {
            Statistic::Median => "median",
            Statistic::Mean => "mean",
            Statistic::Trimmed => "trimmed",
        }
    }

    /// Parse `median`, `mean` or `trimmed`.
    pub fn parse(s: &str) -> Option<Statistic> {
        match s {
            "median" => Some(Statistic::Median),
            "mean" => Some(Statistic::Mean),
            "trimmed" => Some(Statistic::Trimmed),
            _ => None,
        }
    }
}

/// Settings for [`ab_compare`].
#[derive(Clone, Copy, Debug)]
pub struct AbConfig {
    /// Location statistic (default median).
    pub statistic: Statistic,
    /// Bootstrap resamples (default 10,000; values below 1 are raised to 1).
    pub resamples: usize,
    /// Seed of the bootstrap generator (default 0).
    pub seed: u64,
    /// Smallest percent difference reported as a change, and the
    /// equivalence margin (default 1.0, i.e. ±1%).
    pub min_effect_pct: f64,
}

impl Default for AbConfig {
    fn default() -> Self {
        AbConfig { statistic: Statistic::Median, resamples: 10_000, seed: 0, min_effect_pct: 1.0 }
    }
}

/// Outcome of the comparison, read as "B against A".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AbVerdict {
    /// Interval above 0 and effect at least `min_effect_pct`.
    Higher,
    /// Interval below 0 and effect at least `min_effect_pct` in size.
    Lower,
    /// Interval entirely within `±min_effect_pct`.
    Equivalent,
    /// Neither: the data cannot tell at this n and margin, or an arm has
    /// fewer than [`MIN_N_VERDICT`] values.
    Inconclusive,
}

impl AbVerdict {
    /// Label for output. With `lower_is_better` (latencies, step times),
    /// `Lower` reads `B BETTER` and `Higher` reads `B WORSE`.
    pub fn label(self, lower_is_better: bool) -> &'static str {
        match (self, lower_is_better) {
            (AbVerdict::Higher, false) => "B HIGHER",
            (AbVerdict::Lower, false) => "B LOWER",
            (AbVerdict::Higher, true) => "B WORSE",
            (AbVerdict::Lower, true) => "B BETTER",
            (AbVerdict::Equivalent, _) => "EQUIVALENT",
            (AbVerdict::Inconclusive, _) => "INCONCLUSIVE",
        }
    }
}

/// Mann-Whitney U test and Cliff's delta for B against A.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MannWhitney {
    /// U statistic of B: pairs with b > a, plus half the tied pairs.
    pub u_b: f64,
    /// Normal-approximation z (tie- and continuity-corrected); positive when
    /// B tends to be higher.
    pub z: f64,
    /// Two-sided p-value from `z`. 1.0 when every value is tied.
    pub p_value: f64,
    /// Cliff's delta, `P(b > a) - P(b < a)`, in [-1, 1].
    pub cliffs_delta: f64,
}

/// Result of [`ab_compare`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AbResult {
    pub n_a: usize,
    pub n_b: usize,
    /// The chosen statistic on each arm.
    pub stat_a: f64,
    pub stat_b: f64,
    /// `100 * (stat_b - stat_a) / |stat_a|`.
    pub diff_pct: f64,
    /// 95% percentile-bootstrap interval of `diff_pct`.
    pub ci_low_pct: f64,
    pub ci_high_pct: f64,
    /// Resamples that entered the interval (a resample whose A statistic is
    /// exactly 0 has no percent difference and is skipped).
    pub resamples_used: usize,
    pub mann_whitney: MannWhitney,
    pub verdict: AbVerdict,
}

impl AbResult {
    /// True when either arm has fewer than [`SMALL_N`] values.
    pub fn small_sample(&self) -> bool {
        self.n_a < SMALL_N || self.n_b < SMALL_N
    }
}

/// Why [`ab_compare`] could not produce a result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AbError {
    /// An arm has fewer than 2 values (`'A'` or `'B'`, and its n).
    TooFew(char, usize),
    /// An arm contains NaN or an infinity.
    NonFinite(char),
    /// The A statistic is 0, so a percent difference is undefined.
    ZeroBaseline,
}

impl fmt::Display for AbError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AbError::TooFew(arm, n) => write!(f, "arm {} has {} value(s), need at least 2", arm, n),
            AbError::NonFinite(arm) => write!(f, "arm {} contains NaN or infinity", arm),
            AbError::ZeroBaseline => write!(f, "the A statistic is 0, percent difference undefined"),
        }
    }
}

/// SplitMix64 (Steele, Lea, Flood 2014): small, seedable, no dependencies.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform index in `0..n` (multiply-shift; bias below 2^-40 for n < 2^24).
    fn below(&mut self, n: usize) -> usize {
        ((self.next_u64() as u128 * n as u128) >> 64) as usize
    }
}

fn cmp_f64(a: &f64, b: &f64) -> core::cmp::Ordering {
    a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal)
}

/// The statistic of `buf`, reordering it in place. `buf` must be non-empty.
fn stat_in_place(buf: &mut [f64], stat: Statistic) -> f64 {
    let n = buf.len();
    match stat {
        Statistic::Mean => buf.iter().sum::<f64>() / n as f64,
        Statistic::Median => {
            let k = n / 2;
            let (lower, hi, _) = buf.select_nth_unstable_by(k, cmp_f64);
            let hi = *hi;
            if n % 2 == 1 {
                hi
            } else {
                let lo = lower.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                0.5 * (lo + hi)
            }
        }
        Statistic::Trimmed => {
            buf.sort_unstable_by(cmp_f64);
            let g = (TRIM_FRACTION * n as f64) as usize;
            let kept = &buf[g..n - g];
            kept.iter().sum::<f64>() / kept.len() as f64
        }
    }
}

/// The statistic of `values` (copied, the input is not reordered).
/// Returns NaN for an empty slice.
pub fn statistic(values: &[f64], stat: Statistic) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut buf = values.to_vec();
    stat_in_place(&mut buf, stat)
}

/// Mann-Whitney U test of B against A, two-sided, normal approximation with
/// tie correction and a 0.5 continuity correction, plus Cliff's delta.
/// Both slices must be non-empty and free of NaN.
pub fn mann_whitney(a: &[f64], b: &[f64]) -> MannWhitney {
    let (na, nb) = (a.len() as f64, b.len() as f64);
    let mut all: Vec<(f64, bool)> = Vec::with_capacity(a.len() + b.len());
    all.extend(a.iter().map(|&x| (x, false)));
    all.extend(b.iter().map(|&x| (x, true)));
    all.sort_unstable_by(|x, y| cmp_f64(&x.0, &y.0));

    let n = all.len();
    let (mut rank_sum_b, mut tie_term) = (0.0f64, 0.0f64);
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && all[j].0 == all[i].0 {
            j += 1;
        }
        // positions i..j share ranks i+1..=j
        let avg_rank = (i + 1 + j) as f64 / 2.0;
        let in_b = all[i..j].iter().filter(|p| p.1).count();
        rank_sum_b += avg_rank * in_b as f64;
        let t = (j - i) as f64;
        tie_term += t * t * t - t;
        i = j;
    }

    let u_b = rank_sum_b - nb * (nb + 1.0) / 2.0;
    let cliffs_delta = 2.0 * u_b / (na * nb) - 1.0;
    let nn = na + nb;
    let mean_u = na * nb / 2.0;
    let var_u = na * nb / 12.0 * ((nn + 1.0) - tie_term / (nn * (nn - 1.0)));
    let (z, p_value) = if var_u > 0.0 {
        let d = u_b - mean_u;
        let z = d.signum() * (d.abs() - 0.5).max(0.0) / crate::sqrt(var_u);
        (z, libm::erfc(z.abs() / core::f64::consts::SQRT_2).min(1.0))
    } else {
        (0.0, 1.0)
    };
    MannWhitney { u_b, z, p_value, cliffs_delta }
}

/// Linear-interpolated quantile of an ascending slice.
fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = pos - lo as f64;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

/// Compare the level of B against A. See the module docs for the method.
pub fn ab_compare(a: &[f64], b: &[f64], cfg: &AbConfig) -> Result<AbResult, AbError> {
    for (arm, v) in [('A', a), ('B', b)] {
        if v.len() < 2 {
            return Err(AbError::TooFew(arm, v.len()));
        }
        if v.iter().any(|x| !x.is_finite()) {
            return Err(AbError::NonFinite(arm));
        }
    }
    let stat_a = statistic(a, cfg.statistic);
    let stat_b = statistic(b, cfg.statistic);
    if stat_a == 0.0 {
        return Err(AbError::ZeroBaseline);
    }
    let pct = |sa: f64, sb: f64| 100.0 * (sb - sa) / sa.abs();
    let diff_pct = pct(stat_a, stat_b);

    let mut rng = SplitMix64(cfg.seed);
    let mut buf_a = vec_of(a.len());
    let mut buf_b = vec_of(b.len());
    let resamples = cfg.resamples.max(1);
    let mut diffs: Vec<f64> = Vec::with_capacity(resamples);
    for _ in 0..resamples {
        for slot in buf_a.iter_mut() {
            *slot = a[rng.below(a.len())];
        }
        for slot in buf_b.iter_mut() {
            *slot = b[rng.below(b.len())];
        }
        let sa = stat_in_place(&mut buf_a, cfg.statistic);
        let sb = stat_in_place(&mut buf_b, cfg.statistic);
        if sa != 0.0 {
            diffs.push(pct(sa, sb));
        }
    }
    if diffs.is_empty() {
        return Err(AbError::ZeroBaseline);
    }
    diffs.sort_unstable_by(cmp_f64);
    let ci_low_pct = quantile_sorted(&diffs, 0.025);
    let ci_high_pct = quantile_sorted(&diffs, 0.975);

    let m = cfg.min_effect_pct.abs();
    let verdict = if a.len() < MIN_N_VERDICT || b.len() < MIN_N_VERDICT {
        AbVerdict::Inconclusive
    } else if ci_low_pct > 0.0 && diff_pct.abs() >= m {
        AbVerdict::Higher
    } else if ci_high_pct < 0.0 && diff_pct.abs() >= m {
        AbVerdict::Lower
    } else if ci_low_pct >= -m && ci_high_pct <= m {
        AbVerdict::Equivalent
    } else {
        AbVerdict::Inconclusive
    };

    Ok(AbResult {
        n_a: a.len(),
        n_b: b.len(),
        stat_a,
        stat_b,
        diff_pct,
        ci_low_pct,
        ci_high_pct,
        resamples_used: diffs.len(),
        mann_whitney: mann_whitney(a, b),
        verdict,
    })
}

fn vec_of(n: usize) -> Vec<f64> {
    let mut v = Vec::with_capacity(n);
    v.resize(n, 0.0);
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Heavy-tailed positive samples: `base * (1 + 0.15 * (Pareto(alpha 2.2) - 1))`.
    /// Shape similar to per-request step times: a dense floor and a long right tail.
    fn heavy(n: usize, base: f64, seed: u64) -> Vec<f64> {
        let mut rng = SplitMix64(seed);
        (0..n)
            .map(|_| {
                let u = ((rng.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
                base * (1.0 + 0.15 * (u.powf(-1.0 / 2.2) - 1.0))
            })
            .collect()
    }

    fn excess_kurtosis(v: &[f64]) -> f64 {
        let n = v.len() as f64;
        let m = v.iter().sum::<f64>() / n;
        let m2 = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n;
        let m4 = v.iter().map(|x| (x - m).powi(4)).sum::<f64>() / n;
        m4 / (m2 * m2) - 3.0
    }

    #[test]
    fn generator_is_heavy_tailed() {
        // The detection tests below are only meaningful if the data really
        // has the tails they claim to be robust to.
        let a = heavy(229, 70.0, 11);
        assert!(excess_kurtosis(&a) > 10.0, "kurtosis {}", excess_kurtosis(&a));
    }

    #[test]
    fn statistics_on_known_values() {
        assert_eq!(statistic(&[3.0, 1.0, 2.0], Statistic::Median), 2.0);
        assert_eq!(statistic(&[4.0, 1.0, 3.0, 2.0], Statistic::Median), 2.5);
        assert_eq!(statistic(&[1.0, 2.0, 6.0], Statistic::Mean), 3.0);
        // 10 values, 20% trim drops 2 from each end: mean of 3..=8
        let v: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        assert_eq!(statistic(&v, Statistic::Trimmed), 5.5);
        let mut with_outlier = v.clone();
        with_outlier[9] = 1e9;
        assert_eq!(statistic(&with_outlier, Statistic::Trimmed), 5.5);
    }

    #[test]
    fn mann_whitney_known_values() {
        // complete separation, n=5 each: U=25, z=(12.5-0.5)/sqrt(25*11/12)=2.5067,
        // two-sided p=0.01219 (matches the asymptotic, continuity-corrected test)
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [6.0, 7.0, 8.0, 9.0, 10.0];
        let mw = mann_whitney(&a, &b);
        assert_eq!(mw.u_b, 25.0);
        assert_eq!(mw.cliffs_delta, 1.0);
        assert!((mw.z - 2.5067).abs() < 1e-4, "z {}", mw.z);
        assert!((mw.p_value - 0.01219).abs() < 1e-4, "p {}", mw.p_value);
        let rev = mann_whitney(&b, &a);
        assert_eq!(rev.cliffs_delta, -1.0);
        assert!((rev.p_value - mw.p_value).abs() < 1e-12);
        // all tied
        let t = mann_whitney(&[1.0, 1.0], &[1.0, 1.0, 1.0]);
        assert_eq!((t.p_value, t.cliffs_delta, t.z), (1.0, 0.0, 0.0));
    }

    #[test]
    fn identical_arms_are_equivalent() {
        let a = heavy(2000, 70.0, 1);
        let r = ab_compare(&a, &a, &AbConfig::default()).unwrap();
        assert_eq!(r.diff_pct, 0.0);
        assert!(r.ci_low_pct < 0.0 && r.ci_high_pct > 0.0);
        assert_eq!(r.verdict, AbVerdict::Equivalent, "{:?}", r);
        assert!((r.mann_whitney.p_value - 1.0).abs() < 1e-9);
        assert_eq!(r.mann_whitney.cliffs_delta, 0.0);
    }

    #[test]
    fn large_shift_in_heavy_tailed_arms_is_detected() {
        // Shape of the motivating case: n=229 vs 222, B about 1.84x A.
        let a = heavy(229, 70.0, 11);
        let b = heavy(222, 70.0 * 1.84, 12);
        assert!(excess_kurtosis(&b) > 10.0);
        let r = ab_compare(&a, &b, &AbConfig::default()).unwrap();
        assert_eq!(r.verdict, AbVerdict::Higher, "{:?}", r);
        assert_eq!(r.verdict.label(true), "B WORSE");
        assert!(r.ci_low_pct > 60.0 && r.ci_high_pct < 110.0, "{:?}", r);
        assert!(r.mann_whitney.p_value < 1e-10);
        assert!(r.mann_whitney.cliffs_delta > 0.8);
    }

    #[test]
    fn small_shift_detected_and_direction_follows_arms() {
        // +10% at n=300 per arm, for every statistic.
        let a = heavy(300, 100.0, 21);
        let b = heavy(300, 110.0, 22);
        for stat in [Statistic::Median, Statistic::Mean, Statistic::Trimmed] {
            let cfg = AbConfig { statistic: stat, ..AbConfig::default() };
            let up = ab_compare(&a, &b, &cfg).unwrap();
            assert_eq!(up.verdict, AbVerdict::Higher, "{:?} {:?}", stat, up);
            let down = ab_compare(&b, &a, &cfg).unwrap();
            assert_eq!(down.verdict, AbVerdict::Lower, "{:?} {:?}", stat, down);
            assert_eq!(down.verdict.label(true), "B BETTER");
        }
    }

    #[test]
    fn tiny_n_is_inconclusive() {
        // Below MIN_N_VERDICT no call is made, even with complete separation
        // (where the bootstrap interval alone would exclude 0).
        let r = ab_compare(&[1.0, 2.0, 3.0, 4.0], &[10.0, 11.0, 12.0, 13.0], &AbConfig::default()).unwrap();
        assert!(r.ci_low_pct > 0.0);
        assert_eq!(r.verdict, AbVerdict::Inconclusive);
        assert!(r.small_sample());
        let same = ab_compare(&[5.0, 5.0, 5.0], &[5.0, 5.0, 5.0], &AbConfig::default()).unwrap();
        assert_eq!(same.verdict, AbVerdict::Inconclusive);
        // From 5 values the verdict rule applies and the warning stays. A +10%
        // shift at n=5, over 20 data seeds: no call may point the wrong way.
        for s in 0..20u64 {
            let a = heavy(5, 100.0, 100 + 2 * s);
            let b = heavy(5, 110.0, 101 + 2 * s);
            let r = ab_compare(&a, &b, &AbConfig::default()).unwrap();
            assert!(r.small_sample());
            assert_ne!(r.verdict, AbVerdict::Lower, "seed {} {:?}", s, r);
        }
    }

    #[test]
    fn same_seed_same_result_different_seed_close() {
        let a = heavy(150, 70.0, 31);
        let b = heavy(150, 77.0, 32);
        let cfg = AbConfig { seed: 42, ..AbConfig::default() };
        let r1 = ab_compare(&a, &b, &cfg).unwrap();
        let r2 = ab_compare(&a, &b, &cfg).unwrap();
        assert_eq!(r1, r2);
        let r3 = ab_compare(&a, &b, &AbConfig { seed: 43, ..cfg }).unwrap();
        assert_ne!(r1.ci_low_pct, r3.ci_low_pct);
        assert!((r1.ci_low_pct - r3.ci_low_pct).abs() < 1.0);
        assert_eq!(r1.diff_pct, r3.diff_pct);
    }

    #[test]
    fn null_false_call_rate_is_low() {
        // A and B drawn from the same heavy-tailed distribution, n=50 each,
        // 200 pairs: count how often a direction is called.
        let cfg = AbConfig { resamples: 2000, ..AbConfig::default() };
        let mut called = 0;
        for s in 0..200u64 {
            let a = heavy(50, 70.0, 1000 + 2 * s);
            let b = heavy(50, 70.0, 1001 + 2 * s);
            let r = ab_compare(&a, &b, &cfg).unwrap();
            if matches!(r.verdict, AbVerdict::Higher | AbVerdict::Lower) {
                called += 1;
            }
        }
        // nominal 5% would be 10 of 200
        assert!(called <= 16, "{} of 200 null pairs called a direction", called);
    }

    #[test]
    fn errors() {
        let ok = [1.0, 2.0, 3.0];
        let cfg = AbConfig::default();
        assert_eq!(ab_compare(&[1.0], &ok, &cfg), Err(AbError::TooFew('A', 1)));
        assert_eq!(ab_compare(&ok, &[], &cfg), Err(AbError::TooFew('B', 0)));
        assert_eq!(ab_compare(&ok, &[1.0, f64::NAN], &cfg), Err(AbError::NonFinite('B')));
        assert_eq!(ab_compare(&[0.0, 0.0, 0.0], &ok, &cfg), Err(AbError::ZeroBaseline));
    }
}
