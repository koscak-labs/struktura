//! Compression falsifier v3: does the growth cycle's compression beat no compression on held-out
//! accuracy, on 12 fresh seeds?
//!
//! [`crate::brain_grok2`] failed its pre-registered grokking test (Fourier 2/5, one-hot 0/5) but left
//! one directional lead that was NOT pre-registered: on all 5 Fourier seeds the cycle WITH
//! compression (arm A) ended above the identical cycle WITHOUT compression (arm B) on held-out
//! (0.10..0.38 vs ~0.03; one-sided sign test p = 0.0625). This module pre-registers exactly that
//! claim as the primary, on fresh seeds, with nothing else changed. The jump ("a sudden grokking
//! event") is not part of the verdict here.
//!
//! ## Pre-registration (written and committed before any fresh seed was run)
//!
//! - **World, data, arms, knobs, budget: unchanged** from [`crate::brain_grok2`] (`(a + b) mod 7`,
//!   20 fixed training pairs of which 4 are the validation slice, 29 test pairs, 300 epochs, arms
//!   A / B / C, [`Knobs::NONE`]). The code path is [`crate::brain_grok2::run`].
//! - **Encoding: Fourier** (where the lead was seen). One-hot is reported, not judged (its held-out
//!   is predicted at chance in every arm, as in v1 and v2).
//! - **Metric per seed and arm**: held-out = test accuracy of the brain's own decision, mean of the
//!   last 10 epochs ([`crate::brain_grok2::GrokTrial2`]`::heldout_*`).
//! - **Primary claim**: A > B. A seed is a WIN iff `heldout_a > heldout_b` strictly; a tie counts
//!   as a loss (conservative). **PASS iff the one-sided exact sign test over the 12 seeds gives
//!   `p <= 0.05`**, i.e. at least [`MIN_WINS`] = 10 wins of 12 (`P(X >= 10 | n = 12, 1/2) = 0.0193`;
//!   9 wins gives 0.0730 and fails).
//! - **Fresh seeds** [`FRESH_SEEDS`]: 12 seeds never run by any brain falsifier before this
//!   pre-registration (disjoint from v1, v2 and the dev seeds; checked by a test).
//! - **Reported, not judged**: mean paired difference A - B and its exact one-sided sign-flip
//!   permutation p (all 4096 sign patterns); the same sign test for A > C (C = grow-only, every slot
//!   open: is A also ahead of a non-gated control?); A's jumps under v2's rule; one-hot held-out.
//! - **Analysis prediction** (pre-registered): PASS on Fourier. A's advantage in v2 came from
//!   compression removing the 11-12 overfitting base senses, which B can never do; nothing about
//!   that is seed-specific. Caveat stated in advance: a PASS shows "this compression helps this
//!   linear brain generalize", not grokking, and not that compression is needed in general.
//!
//! ## RESULT
//!
//! (added after the one run; rule, thresholds and seeds unchanged)

use crate::brain_grok::{Encoding, END};
use crate::brain_grok2::{self as g2, GrokTrial2};

/// Pre-registered fresh seeds.
pub const FRESH_SEEDS: [u64; 12] = [7103, 7211, 7307, 7417, 7523, 7621, 7727, 7829, 7933, 8039, 8147, 8243];
/// Wins of 12 needed for a one-sided sign-test p <= 0.05.
pub const MIN_WINS: usize = 10;
/// Significance level of the primary test.
pub const ALPHA: f64 = 0.05;

/// One-sided exact sign test: `P(X >= wins)` for `X ~ Binomial(n, 1/2)`.
pub fn sign_p(wins: usize, n: usize) -> f64 {
    let mut c = 1.0f64; // C(n, 0)
    let mut tail = 0.0f64;
    for k in 0..=n {
        if k >= wins { tail += c; }
        c = c * (n - k) as f64 / (k + 1) as f64;
    }
    tail / (2.0f64).powi(n as i32)
}

/// Exact one-sided sign-flip permutation p for the mean of paired differences (`n <= 20`).
pub fn perm_p(d: &[f64]) -> f64 {
    let n = d.len();
    let obs: f64 = d.iter().sum();
    let mut ge = 0u64;
    for mask in 0u64..(1u64 << n) {
        let s: f64 = d.iter().enumerate().map(|(i, x)| if mask >> i & 1 == 1 { -x } else { *x }).sum();
        if s >= obs - 1e-12 { ge += 1; }
    }
    ge as f64 / (1u64 << n) as f64
}

/// The primary comparison over the fresh seeds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Verdict3 {
    pub wins: usize,
    pub n: usize,
    pub p: f64,
    pub mean_diff: f64,
    pub perm_p: f64,
    pub pass: bool,
}

/// Wins / sign test / permutation for `x > y` over paired values.
pub fn compare(x: &[f32], y: &[f32]) -> Verdict3 {
    let n = x.len().min(y.len());
    let wins = (0..n).filter(|&i| x[i] > y[i]).count();
    let d: std::vec::Vec<f64> = (0..n).map(|i| x[i] as f64 - y[i] as f64).collect();
    let p = sign_p(wins, n);
    Verdict3 { wins, n, p, mean_diff: d.iter().sum::<f64>() / n.max(1) as f64, perm_p: perm_p(&d), pass: p <= ALPHA }
}

/// Run every fresh seed (in parallel threads; each trial is deterministic).
pub fn trials(enc: Encoding) -> [GrokTrial2; 12] {
    std::thread::scope(|s| FRESH_SEEDS.map(|seed| s.spawn(move || g2::trial(seed, enc))).map(|h| h.join().unwrap()))
}

/// The pre-registered primary verdict (A > B, Fourier) from computed trials.
pub fn primary(t: &[GrokTrial2]) -> Verdict3 {
    let a: std::vec::Vec<f32> = t.iter().map(|x| x.heldout_a).collect();
    let b: std::vec::Vec<f32> = t.iter().map(|x| x.heldout_b).collect();
    compare(&a, &b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_test_values() {
        assert!((sign_p(10, 12) - 0.019287).abs() < 1e-5);
        assert!((sign_p(9, 12) - 0.072998).abs() < 1e-5);
        assert!((sign_p(5, 5) - 0.03125).abs() < 1e-9);
        assert!(sign_p(MIN_WINS, 12) <= ALPHA && sign_p(MIN_WINS - 1, 12) > ALPHA);
        assert_eq!(sign_p(0, 12), 1.0);
    }

    #[test]
    fn permutation_values() {
        assert!((perm_p(&[1.0, 1.0, 1.0]) - 0.125).abs() < 1e-12);
        assert_eq!(perm_p(&[0.0, 0.0]), 1.0);
        let v = compare(&[0.3, 0.2, 0.1], &[0.0, 0.2, 0.0]);
        assert_eq!((v.wins, v.n), (2, 3), "a tie is not a win");
    }

    /// The pre-registration, pinned.
    #[test]
    fn preregistration_is_pinned() {
        use crate::brain_grok as v1;
        assert_eq!(FRESH_SEEDS, [7103, 7211, 7307, 7417, 7523, 7621, 7727, 7829, 7933, 8039, 8147, 8243]);
        for s in FRESH_SEEDS.iter() {
            assert!(!v1::FRESH_SEEDS.contains(s) && !g2::FRESH_SEEDS.contains(s) && !g2::DEV_SEEDS.contains(s), "seed {} is fresh", s);
        }
        assert_eq!((MIN_WINS, ALPHA, END), (10, 0.05, 10));
        assert_eq!(g2::Knobs::NONE, g2::Knobs::default());
    }

    /// PRE-REGISTERED primary falsifier v3: compression > no compression on held-out, Fourier, 12
    /// fresh seeds, one-sided sign test p <= 0.05.
    /// Run: cargo test --release --lib brain_grok3 -- --ignored --nocapture
    #[test]
    #[ignore = "slow: 12 seeds x 3 arms x 300 epochs (run explicitly)"]
    fn falsifier_v3_compression_beats_no_compression() {
        let t = trials(Encoding::Fourier);
        for x in t.iter() {
            std::println!("  fourier seed {}: held-out A {:.3} B {:.3} C {:.3} | train A {:.3} B {:.3} | jump A {:?} B {:?} | A size {:?} base {} | B size {:?} base {} | groks A {} B {}",
                x.seed, x.heldout_a, x.heldout_b, x.heldout_c, x.train_a, x.train_b, x.jump_epoch, x.jump_b, x.size_after, x.base_after, x.size_b, x.base_b, x.groks_a, x.groks_b);
        }
        let v = primary(&t);
        let ac = compare(&t.iter().map(|x| x.heldout_a).collect::<std::vec::Vec<_>>(), &t.iter().map(|x| x.heldout_c).collect::<std::vec::Vec<_>>());
        std::println!("PRIMARY A>B: wins {}/{} sign p {:.4} | mean diff {:+.3} perm p {:.4} -> {}", v.wins, v.n, v.p, v.mean_diff, v.perm_p, if v.pass { "PASS" } else { "FAIL" });
        std::println!("reported A>C: wins {}/{} sign p {:.4} | mean diff {:+.3} perm p {:.4}", ac.wins, ac.n, ac.p, ac.mean_diff, ac.perm_p);
        std::println!("reported A jumps (v2 rule): {}/12", t.iter().filter(|x| x.jump_epoch.is_some()).count());
        let oh = trials(Encoding::OneHot);
        let vo = primary(&oh);
        std::println!("reported one-hot A>B: wins {}/{} p {:.4}; held-out A max {:.3} B max {:.3}", vo.wins, vo.n, vo.p,
            oh.iter().map(|x| x.heldout_a).fold(0.0f32, f32::max), oh.iter().map(|x| x.heldout_b).fold(0.0f32, f32::max));
        assert!(v.pass, "A > B on {}/12 seeds, p = {:.4} > {}", v.wins, v.p, ALPHA);
    }
}
