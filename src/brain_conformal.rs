//! Conformal abstention for [`crate::brain::Brain`]: act autonomously only when
//! the evidence clears a calibrated bar, otherwise hand control to a safe default.
//! No heap, no model, deterministic.
//!
//! Two rules, both calibrated on the brain's own recent outcomes:
//!
//! 1. **Interval separation** ([`Conformal::decide`]). A ring of past absolute
//!    residuals `|reward - expected|` (for actions actually taken) gives the
//!    split-conformal quantile `q = R_(k)`, the k-th smallest of n residuals with
//!    `k = ceil((n + 1)(1 - alpha))`. The best allowed action is taken only if
//!    `best.expected - q > other.expected + q` for every other allowed action.
//!
//!    Guarantee and its limits: if the stored residuals and the next one are
//!    exchangeable, `P(|reward - expected| <= q) >= 1 - alpha` for the next
//!    decision (marginal, finite-sample). Union-bounding the A intervals gives
//!    `P(an acted-on choice is not the best) <= A * alpha`. Caveats: the brain
//!    keeps learning, so residuals drift (the ring holds only the recent M);
//!    residuals are only seen for actions taken, so intervals for actions not
//!    taken are not directly validated; with rewards in [0, 1] and a wide q the
//!    rule can abstain almost always.
//!
//! 2. **Selective risk control** ([`Conformal::decide_selective`]). A ring of past
//!    decisions records (gap between best and runner-up expectation, whether the
//!    choice turned out right). It acts only when the current gap is at least the
//!    smallest threshold whose exact binomial (Clopper-Pearson) upper confidence
//!    bound on the calibrated error rate, at confidence 1 - delta, is at most `alpha`. If the outcomes are
//!    i.i.d. (an exchangeable stream), with probability >= 1 - delta the error
//!    rate of acted-on decisions at that threshold is <= alpha (one threshold per
//!    look; choosing among thresholds is not union-bounded). "Right" must be
//!    observable, e.g. reward above a success level.
//!
//! Memory: two rings of `M` entries: `M * 12` bytes plus a few counters
//! (256 entries: about 3 KB).

use crate::brain::{Brain, Decision};

/// Result of a guarded decision.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Guarded {
    /// What the brain itself would do (unguarded).
    pub brain: Decision,
    /// The action to take: `brain.action` when acting, otherwise `safe_default`.
    pub action: u8,
    pub abstained: bool,
    /// Calibrated half-width (rule 1) or gap threshold (rule 2); infinite when uncalibrated.
    pub bar: f32,
    /// Margin by which the best action cleared the runner-up (best - runner-up expected).
    pub gap: f32,
}

pub struct Conformal<const M: usize> {
    res: [f32; M],
    res_len: usize,
    res_next: usize,
    gaps: [f32; M],
    oks: [bool; M],
    sel_len: usize,
    sel_next: usize,
}

impl<const M: usize> Default for Conformal<M> {
    fn default() -> Self { Self::new() }
}

impl<const M: usize> Conformal<M> {
    pub const fn new() -> Self {
        Conformal { res: [0.0; M], res_len: 0, res_next: 0, gaps: [0.0; M], oks: [false; M], sel_len: 0, sel_next: 0 }
    }

    /// Residuals stored (at most M).
    pub fn len(&self) -> usize { self.res_len }
    pub fn is_empty(&self) -> bool { self.res_len == 0 }

    /// Record a taken action's expectation and the reward it got (rule 1 calibration).
    pub fn observe(&mut self, expected: f32, reward: f32) {
        if M == 0 { return; }
        let r = expected - reward;
        self.res[self.res_next] = if r < 0.0 { -r } else { r };
        self.res_next = (self.res_next + 1) % M;
        if self.res_len < M { self.res_len += 1; }
    }

    /// Record a decision's gap (best minus runner-up expectation) and whether it was right (rule 2).
    pub fn observe_outcome(&mut self, gap: f32, right: bool) {
        if M == 0 { return; }
        self.gaps[self.sel_next] = gap;
        self.oks[self.sel_next] = right;
        self.sel_next = (self.sel_next + 1) % M;
        if self.sel_len < M { self.sel_len += 1; }
    }

    /// Split-conformal quantile of the stored residuals at level 1 - alpha:
    /// the ceil((n+1)(1-alpha))-th smallest, or None (infinite) when n is too small.
    pub fn quantile(&self, alpha: f32) -> Option<f32> {
        let n = self.res_len;
        if n == 0 || !(alpha > 0.0 && alpha < 1.0) { return None; }
        let kf = (n as f32 + 1.0) * (1.0 - alpha);
        let mut k = kf as usize;
        if (k as f32) < kf { k += 1; }
        if k == 0 { k = 1; }
        if k > n { return None; }
        let mut buf = [0.0f32; M];
        buf[..n].copy_from_slice(&self.res[..n]);
        Some(kth_smallest(&mut buf[..n], k - 1))
    }

    fn best_two<const N: usize, const D: usize, const A: usize>(br: &Brain<N, D, A>, x: &[f32; D], allowed: &[bool; A])
        -> (Option<(usize, f32)>, f32) {
        let est = br.estimates(x);
        let mut best: Option<(usize, f32)> = None;
        for a in 0..A {
            if !allowed[a] { continue; }
            if best.map(|b| est[a].expected > b.1).unwrap_or(true) { best = Some((a, est[a].expected)); }
        }
        let mut second = f32::NEG_INFINITY;
        if let Some((b, _)) = best {
            for a in 0..A { if allowed[a] && a != b && est[a].expected > second { second = est[a].expected; } }
        }
        (best, second)
    }

    /// Rule 1: act only when the best allowed action's conformal interval separates
    /// from every other allowed action's.
    pub fn decide<const N: usize, const D: usize, const A: usize>(&self, br: &Brain<N, D, A>, x: &[f32; D],
        allowed: &[bool; A], safe_default: u8, alpha: f32) -> Guarded {
        let d = br.decide(x, allowed, safe_default);
        let (best, second) = Self::best_two(br, x, allowed);
        let q = self.quantile(alpha);
        let mut g = Guarded { brain: d, action: safe_default, abstained: true, bar: q.unwrap_or(f32::INFINITY), gap: 0.0 };
        if let Some((b, e)) = best {
            g.gap = if second.is_finite() { e - second } else { f32::INFINITY };
            if let Some(q) = q {
                if !d.abstained && (b as u8) == d.action && g.gap > 2.0 * q { g.action = d.action; g.abstained = false; }
            }
        }
        g
    }

    /// Smallest gap threshold whose calibrated selective error has a Clopper-Pearson
    /// upper bound (confidence 1 - delta) <= alpha, or None if no threshold qualifies.
    pub fn gap_threshold(&self, alpha: f32, delta: f32) -> Option<f32> {
        let n = self.sel_len;
        if n == 0 || !(alpha > 0.0 && delta > 0.0 && delta < 1.0) { return None; }
        let mut order = [0usize; M];
        for (i, o) in order.iter_mut().enumerate().take(n) { *o = i; }
        // sort indices by gap descending (insertion sort; bounded by M^2)
        for i in 1..n {
            let mut j = i;
            while j > 0 && self.gaps[order[j - 1]] < self.gaps[order[j]] { order.swap(j - 1, j); j -= 1; }
        }
        let (mut acc, mut wrong) = (0usize, 0usize);
        let mut best: Option<f32> = None;
        for &i in order.iter().take(n) {
            acc += 1;
            if !self.oks[i] { wrong += 1; }
            let bound = binom_upper(wrong, acc, delta as f64);
            // A threshold equal to this gap accepts exactly the `acc` largest gaps (ties aside).
            if bound <= alpha as f64 { best = Some(self.gaps[i]); }
        }
        best
    }

    /// Rule 2: act only when the gap clears the calibrated threshold (selective risk control).
    /// Recomputes the threshold (O(M^2) plus M bisections); on a flight computer compute
    /// [`Conformal::gap_threshold`] periodically and use [`Conformal::decide_with_threshold`].
    pub fn decide_selective<const N: usize, const D: usize, const A: usize>(&self, br: &Brain<N, D, A>, x: &[f32; D],
        allowed: &[bool; A], safe_default: u8, alpha: f32, delta: f32) -> Guarded {
        self.decide_with_threshold(br, x, allowed, safe_default, self.gap_threshold(alpha, delta))
    }

    /// Rule 2 with a precomputed threshold (None = abstain).
    pub fn decide_with_threshold<const N: usize, const D: usize, const A: usize>(&self, br: &Brain<N, D, A>, x: &[f32; D],
        allowed: &[bool; A], safe_default: u8, thr: Option<f32>) -> Guarded {
        let d = br.decide(x, allowed, safe_default);
        let (best, second) = Self::best_two(br, x, allowed);
        let mut g = Guarded { brain: d, action: safe_default, abstained: true, bar: thr.unwrap_or(f32::INFINITY), gap: 0.0 };
        if let Some((b, e)) = best {
            g.gap = if second.is_finite() { e - second } else { f32::INFINITY };
            if let Some(t) = thr {
                if !d.abstained && (b as u8) == d.action && g.gap >= t { g.action = d.action; g.abstained = false; }
            }
        }
        g
    }
}

/// Exact one-sided (Clopper-Pearson) upper confidence bound on an error rate:
/// the p with P(Bin(n, p) <= k) = delta, found by bisection (60 steps).
pub fn binom_upper(k: usize, n: usize, delta: f64) -> f64 {
    if n == 0 { return 1.0; }
    if k >= n { return 1.0; }
    let cdf = |p: f64| -> f64 {
        // sum_{i<=k} C(n,i) p^i (1-p)^(n-i), terms built by ratio from (1-p)^n
        let mut term = crate::powf(1.0 - p, n as f64);
        let mut s = term;
        for i in 0..k { term *= (n - i) as f64 / (i + 1) as f64 * p / (1.0 - p); s += term; }
        s
    };
    let (mut lo, mut hi) = (k as f64 / n as f64, 1.0f64);
    for _ in 0..60 { let mid = 0.5 * (lo + hi); if cdf(mid) > delta { lo = mid } else { hi = mid } }
    hi
}

/// k-th smallest (0-based) by quickselect-free partial selection: bounded O(n * k).
fn kth_smallest(v: &mut [f32], k: usize) -> f32 {
    for i in 0..=k {
        let mut m = i;
        for j in (i + 1)..v.len() { if v[j] < v[m] { m = j; } }
        v.swap(i, m);
    }
    v[k]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantile_is_the_split_conformal_order_statistic() {
        let mut c: Conformal<16> = Conformal::new();
        assert_eq!(c.quantile(0.1), None);
        for r in [0.1f32, 0.5, 0.3, 0.9, 0.2, 0.7, 0.4, 0.8, 0.6] { c.observe(0.0, r); } // n = 9
        // k = ceil(10 * 0.9) = 9 -> 9th smallest = 0.9
        assert_eq!(c.quantile(0.1), Some(0.9));
        // k = ceil(10 * 0.5) = 5 -> 0.5
        assert_eq!(c.quantile(0.5), Some(0.5));
        // alpha too small for n = 9: k = ceil(10 * 0.95) = 10 > 9 -> infinite
        assert_eq!(c.quantile(0.05), None);
    }

    #[test]
    fn clopper_pearson_matches_closed_forms() {
        // k = 0: upper bound is 1 - delta^(1/n)
        let u = binom_upper(0, 100, 0.05);
        assert!((u - (1.0 - 0.05f64.powf(0.01))).abs() < 1e-9, "{}", u);
        // k = n - 1, n = 1 ... and monotone in k
        assert!(binom_upper(1, 100, 0.05) > u);
        assert_eq!(binom_upper(5, 5, 0.05), 1.0);
    }

    #[test]
    fn memory_budget() {
        let sz = core::mem::size_of::<Conformal<256>>();
        assert!(sz <= 2400, "{} bytes", sz);
        static _C: Conformal<64> = Conformal::new();
    }

    #[test]
    fn ring_keeps_only_the_last_m() {
        let mut c: Conformal<4> = Conformal::new();
        for i in 0..10 { c.observe(0.0, i as f32); }
        assert_eq!(c.len(), 4);
        // residuals 6,7,8,9; alpha 0.5 -> k = ceil(5*0.5) = 3 -> 8
        assert_eq!(c.quantile(0.5), Some(8.0));
    }
}

/// The falsifier's rover worlds and run, shared by the test and the brain gym (`crate::brain_gym`).
#[cfg(feature = "std")]
pub mod falsify {
    use super::*;

    struct Xs(u64);
    impl Xs { fn f(&mut self) -> f32 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; ((self.0 >> 40) as f32) / (1u64 << 24) as f32 } }

    pub fn truth_a(x: &[f32; 4]) -> u8 { if x[2] > 0.8 { 3 } else if x[1] > 0.6 { 2 } else if x[0] > 0.6 { 1 } else { 0 } }
    pub fn truth_b(x: &[f32; 4]) -> u8 {
        if x[2] > 0.6 && x[0] > 0.6 { 3 } else if x[1] > 0.6 && x[2] < 0.4 { 2 } else if (x[0] > 0.5) != (x[1] > 0.5) { 1 } else { 0 }
    }
    fn reward(t: u8, a: u8) -> f32 { if a == t { 1.0 } else if a == 3 { 0.2 } else { 0.0 } }

    pub struct Stats { pub rule: &'static str, pub acted: u32, pub errors: u32, pub n: u32, pub covered: u32, pub cov_n: u32, pub base_err: u32 }

    /// Train 3000 steps, then 2000 held-out steps where both guards run in shadow.
    pub fn run(truth: fn(&[f32; 4]) -> u8, alpha: f32, delta: f32, seed: u64) -> (Stats, Stats) {
        let mut br: Brain<256, 4, 4> = Brain::new(0.3, 0.5);
        let mut c: Conformal<256> = Conformal::new();
        let mut r = Xs(seed);
        let mut s1 = Stats { rule: "interval", acted: 0, errors: 0, n: 0, covered: 0, cov_n: 0, base_err: 0 };
        let mut thr: Option<f32> = None;
        let mut s2 = Stats { rule: "selective", acted: 0, errors: 0, n: 0, covered: 0, cov_n: 0, base_err: 0 };
        for t in 0..5000u32 {
            let x = [r.f(), r.f(), r.f(), 1.0];
            let tr = truth(&x);
            let est = br.estimates(&x);
            let d = br.decide(&x, &[true; 4], 3);
            if t >= 3000 {
                let g1 = c.decide(&br, &x, &[true; 4], 3, alpha);
                if t % 25 == 0 { thr = c.gap_threshold(alpha, delta); }
                let g2 = c.decide_with_threshold(&br, &x, &[true; 4], 3, thr);
                if d.action != tr { s1.base_err += 1; s2.base_err += 1; }
                for (g, s) in [(g1, &mut s1), (g2, &mut s2)] {
                    s.n += 1;
                    if !g.abstained { s.acted += 1; if g.action != tr { s.errors += 1; } }
                }
                // coverage of the rule-1 interval for the action actually taken
                if let Some(q) = c.quantile(alpha) {
                    let a = d.action as usize;
                    s1.cov_n += 1;
                    if (reward(tr, d.action) - est[a].expected).abs() <= q { s1.covered += 1; }
                }
            }
            let a = if d.abstained && t < 400 { (t % 4) as u8 } else { d.action };
            let rw = reward(tr, a);
            c.observe(est[a as usize].expected, rw);
            if a == d.action && !d.abstained {
                let mut ex = [f32::NEG_INFINITY; 4];
                for (i, e) in est.iter().enumerate() { ex[i] = e.expected; }
                let best = ex[a as usize];
                let second = ex.iter().enumerate().filter(|(i, _)| *i != a as usize).map(|(_, v)| *v).fold(f32::NEG_INFINITY, f32::max);
                c.observe_outcome(best - second, a == tr);
            }
            br.learn(&x, a, rw);
        }
        (s1, s2)
    }

    /// The pre-registered verdict on one run (the test's asserts): error among acted <= alpha for
    /// both rules, interval coverage >= 1 - alpha (2-point slack), selective rule acts on >= half.
    pub fn holds(s1: &Stats, s2: &Stats, alpha: f32) -> bool {
        [s1, s2].iter().all(|s| s.acted == 0 || s.errors as f32 <= alpha * s.acted as f32)
            && s1.covered as f32 >= (1.0 - alpha) * s1.cov_n as f32 - 0.02 * s1.cov_n as f32
            && s2.acted >= s2.n / 2
    }
}

#[cfg(test)]
mod falsifier {
    extern crate std;
    use std::println;
    use super::falsify::{holds, run, truth_a, truth_b};

    fn pct(a: u32, b: u32) -> f32 { if b == 0 { f32::NAN } else { 100.0 * a as f32 / b as f32 } }

    #[test]
    fn falsifier_rover_worlds() {
        let alpha = 0.1;
        for (name, truth) in [("A thresholds", truth_a as fn(&[f32; 4]) -> u8), ("B interactions", truth_b as fn(&[f32; 4]) -> u8)] {
            for seed in [0x9E3779B97F4A7C15u64, 0xD1B54A32D192ED03, 0x2545F4914F6CDD1D] {
                let (s1, s2) = run(truth, alpha, 0.05, seed);
                println!("world {} seed {:x}: unguarded brain error {:.2}%", name, seed, pct(s1.base_err, s1.n));
                for s in [&s1, &s2] {
                    println!("  rule {:<9}: acted {}/{} (abstain {:.1}%), error among acted {:.2}% (target <= {:.0}%)",
                        s.rule, s.acted, s.n, 100.0 - pct(s.acted, s.n), pct(s.errors, s.acted), 100.0 * alpha);
                    if s.acted > 0 { assert!(s.errors as f32 <= alpha * s.acted as f32, "{} error above target", s.rule); }
                }
                println!("  interval coverage on held-out: {:.2}% (target >= {:.0}%)", pct(s1.covered, s1.cov_n), 100.0 * (1.0 - alpha));
                assert!(s1.covered as f32 >= (1.0 - alpha) * s1.cov_n as f32 - 0.02 * s1.cov_n as f32, "coverage far below target");
                assert!(s2.acted >= s2.n / 2, "selective rule must act on at least half the decisions to be useful");
                assert!(holds(&s1, &s2, alpha), "the gym verdict agrees with the asserts");
            }
        }
    }
}
