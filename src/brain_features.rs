//! Situations for the native brain ([`crate::brain`]) built from struktura's
//! own signal fingerprint.
//!
//! A brain can only tell situations apart if their features differ. Level
//! and spread alone cannot separate faults with the same mean and variance
//! but different dynamics (a slow persistent drift versus white-noise
//! bursts). The structural fingerprint can: DFA alpha and the Hurst
//! exponent measure persistence, the MFDFA width measures multifractality,
//! kurtosis measures heavy tails (bursts), and the DFA fit quality says how
//! clean the scaling law is.
//!
//! Every feature is mapped to `[0, 1]` so Euclidean distances in the brain's
//! memory weigh them comparably. Windows shorter than [`MIN_WINDOW`] give a
//! fingerprint too noisy to trust; the caller should buffer more samples.
//! Builds without `std` (the fingerprint uses `alloc` internally).

use crate::fingerprint::fingerprint;

/// Features per situation: 5 structural + 2 context + 1 constant.
pub const D: usize = 8;

/// Smallest window the fingerprint is meaningful on.
pub const MIN_WINDOW: usize = 128;

/// Smoothly map a non-negative quantity to [0, 1): x / (x + scale).
fn squash(x: f64, scale: f64) -> f32 {
    if x.is_finite() && x > 0.0 { (x / (x + scale)) as f32 } else { 0.0 }
}

fn clamp01(x: f64) -> f32 {
    if x.is_finite() { (if x < 0.0 { 0.0 } else if x > 1.0 { 1.0 } else { x }) as f32 } else { 0.0 }
}

/// Situation for one sensor window plus up to two context values (each
/// clamped to [0, 1], e.g. temperature fraction of limit, wheel load):
/// `[alpha / 2, hurst, mfdfa_width / 2, kurtosis / (kurtosis + 3), dfa r², ctx0, ctx1, 1]`.
///
/// A Gaussian window has kurtosis ≈ 3, so the tail feature sits near 0.5 for
/// noise and approaches 1 for bursty data; white noise has alpha ≈ 0.5
/// (feature 0.25) and a random walk alpha ≈ 1.5 (feature 0.75).
pub fn rover_situation(window: &[f64], ctx: &[f32]) -> [f32; D] {
    let fp = fingerprint(window);
    let c = |i: usize| ctx.get(i).map(|&v| clamp01(v as f64)).unwrap_or(0.0);
    [
        clamp01(fp.alpha / 2.0),
        clamp01(fp.hurst),
        clamp01(fp.mfdfa_width / 2.0),
        squash(fp.kurtosis, 3.0),
        clamp01(fp.r_squared),
        c(0),
        c(1),
        1.0,
    ]
}

/// The level-and-spread situation a naive monitor would use:
/// `[mean / (|mean| + 1) mapped to [0, 1], std / (std + 1), 0, 0, 0, ctx0, ctx1, 1]`.
/// Kept as the baseline that [`rover_situation`] must beat.
pub fn level_situation(window: &[f64], ctx: &[f32]) -> [f32; D] {
    let n = window.len().max(1) as f64;
    let mean = window.iter().sum::<f64>() / n;
    let var = window.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
    let c = |i: usize| ctx.get(i).map(|&v| clamp01(v as f64)).unwrap_or(0.0);
    let m = mean / (if mean < 0.0 { -mean } else { mean } + 1.0);
    [clamp01(0.5 + 0.5 * m), squash(crate::sqrt(var), 1.0), 0.0, 0.0, 0.0, c(0), c(1), 1.0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain::Brain;

    struct Lcg(u64);
    impl Lcg {
        fn u(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / (1u64 << 53) as f64
        }
        /// Standard normal (Box-Muller).
        fn n(&mut self) -> f64 {
            let (a, b) = (self.u().max(1e-12), self.u());
            crate::sqrt(-2.0 * libm::log(a)) * libm::cos(2.0 * core::f64::consts::PI * b)
        }
    }

    fn standardize(v: &mut [f64]) {
        let n = v.len() as f64;
        let m = v.iter().sum::<f64>() / n;
        let s = crate::sqrt(v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n).max(1e-12);
        for x in v.iter_mut() { *x = (*x - m) / s; }
    }

    /// Fault 0: slow persistent drift (random walk). Fault 1: white-noise bursts.
    /// Both standardized to mean 0, variance 1: identical level and spread.
    fn window(kind: u8, r: &mut Lcg) -> Vec<f64> {
        let mut v = vec![0.0; 256];
        if kind == 0 {
            let mut s = 0.0;
            for x in v.iter_mut() { s += r.n(); *x = s; }
        } else {
            let mut burst = false;
            for x in v.iter_mut() {
                if r.u() < 0.03 { burst = !burst; }
                *x = r.n() * if burst { 4.0 } else { 0.5 };
            }
        }
        standardize(&mut v);
        v
    }

    fn accuracy(r: &mut Lcg, features: fn(&[f64], &[f32]) -> [f32; D]) -> (usize, usize) {
        // action 0 = recalibrate (right for drift), action 1 = quarantine (right for bursts)
        let mut br: Brain<128, D, 2> = Brain::new(0.3, 0.5);
        let (mut ok, mut n) = (0, 0);
        for t in 0..400u32 {
            let kind = (r.u() < 0.5) as u8;
            let x = features(&window(kind, r), &[]);
            let d = br.decide(&x, &[true; 2], 0);
            let a = if d.abstained && t < 40 { (t % 2) as u8 } else { d.action };
            if t >= 200 { n += 1; if a == kind { ok += 1; } }
            br.learn(&x, a, if a == kind { 1.0 } else { 0.0 });
        }
        (ok, n)
    }

    #[test]
    fn same_level_and_variance_different_structure_are_separated() {
        let mut r = Lcg(2026);
        let (fo, fn_) = accuracy(&mut r, rover_situation);
        let mut r = Lcg(2026);
        let (lo, ln) = accuracy(&mut r, level_situation);
        let (fp, lv) = (100.0 * fo as f64 / fn_ as f64, 100.0 * lo as f64 / ln as f64);
        std::println!("rover faults, last 200 decisions: fingerprint situations {:.1}% right, level/spread situations {:.1}% right", fp, lv);
        assert!(fp >= 90.0, "fingerprint {:.1}%", fp);
        assert!(fp >= lv + 30.0, "fingerprint {:.1}% vs level {:.1}%", fp, lv);
    }

    #[test]
    fn features_are_in_unit_range_and_structure_differs() {
        let mut r = Lcg(7);
        let a = rover_situation(&window(0, &mut r), &[0.3, 2.0]);
        let b = rover_situation(&window(1, &mut r), &[-1.0]);
        for x in a.iter().chain(b.iter()) { assert!((0.0..=1.0).contains(x), "{:?} {:?}", a, b); }
        assert_eq!((a[5], a[6], b[5], b[6], a[7]), (0.3, 1.0, 0.0, 0.0, 1.0), "context clamped, missing context = 0");
        assert!(a[0] > b[0] + 0.2, "drift is more persistent: alpha {} vs {}", a[0], b[0]);
        assert!(b[3] > a[3], "bursts have heavier tails: {} vs {}", b[3], a[3]);
        let la = level_situation(&window(0, &mut r), &[]);
        let lb = level_situation(&window(1, &mut r), &[]);
        assert!((la[0] - lb[0]).abs() < 1e-3 && (la[1] - lb[1]).abs() < 1e-3, "level features cannot tell them apart");
    }
}
