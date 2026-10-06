//! Neuromodulation: the brain's exploration and forgetting set by its own surprise.
//!
//! Three signals, loosely after the neuromodulators they are named for:
//!
//! - **Dopamine**: the signed reward prediction error `reward - expected`,
//!   kept as a running average (reported; positive = outcomes better than expected).
//! - **Acetylcholine (expected uncertainty)**: the slow running average of the
//!   unsigned error. It sets the exploration level the brain settles at: a world
//!   it predicts well gets little exploration, a noisy one more.
//! - **Norepinephrine (unexpected uncertainty, "network reset")**: when the fast
//!   running error rises well above the slow one (`fast > ratio * slow`), the
//!   world has probably changed. The modulator forgets part of the model
//!   ([`Brain::forget`]), forgets memories older than the recent window
//!   ([`Brain::forget_slot`]) and boosts exploration to its maximum, then lets
//!   exploration decay back. A cooldown stops repeated resets.
//!
//! Exploration always stays within `[explore_min, explore_max]`.
//! no_std, no heap, bounded work per step (one pass over memory on a reset),
//! deterministic.

use crate::brain::Brain;

/// What the modulator did this step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Signal {
    /// Signed reward prediction error (running average).
    pub dopamine: f32,
    /// Fast running unsigned error.
    pub fast: f32,
    /// Slow running unsigned error.
    pub slow: f32,
    /// Exploration weight now set on the brain.
    pub explore: f32,
    /// A reset (forget + exploration boost) fired this step.
    pub reset: bool,
}

pub struct Neuromod {
    fast: f32,
    slow: f32,
    dopamine: f32,
    n: u32,
    since_reset: u32,
    /// Running-average rates (per step) of the fast and slow error.
    pub alpha_fast: f32,
    pub alpha_slow: f32,
    pub explore_min: f32,
    pub explore_max: f32,
    /// Slow error at which exploration reaches its maximum (acetylcholine scale).
    pub err_scale: f32,
    /// Fast/slow error ratio that triggers a reset.
    pub ratio: f32,
    /// Fast error below this never triggers a reset (absolute floor).
    pub min_fast: f32,
    /// Model confidence kept on a reset (passed to `Brain::forget`).
    pub reset_keep: f32,
    /// On a reset, keep only memories learned in the last `keep_recent` steps.
    pub keep_recent: u32,
    /// Steps after a reset (and at start) before another reset may fire.
    pub cooldown: u32,
    /// Consecutive surprised steps required before a reset (a real change is sustained;
    /// a noise spike is not).
    pub confirm: u32,
    streak: u32,
    /// Per-step pull of exploration toward its acetylcholine target.
    pub settle: f32,
    /// Resets fired so far.
    pub resets: u32,
    explore: f32,
}

impl Neuromod {
    /// Defaults for rewards in [0, 1].
    pub const fn new() -> Self {
        Neuromod { fast: 0.0, slow: 0.0, dopamine: 0.0, n: 0, since_reset: 0,
            alpha_fast: 0.1, alpha_slow: 0.005, explore_min: 0.05, explore_max: 0.6, err_scale: 0.5,
            ratio: 2.0, min_fast: 0.25, reset_keep: 0.05, keep_recent: 50, cooldown: 400, confirm: 25, streak: 0, settle: 0.02,
            resets: 0, explore: 0.3 }
    }

    /// Feed one outcome: `expected` is what the brain predicted for the action it took,
    /// `reward` what happened. Updates the signals and sets `brain.explore` (and, on a
    /// reset, forgets part of the model and old memories). Call after `learn`.
    pub fn observe<const N: usize, const D: usize, const A: usize>(&mut self, brain: &mut Brain<N, D, A>, expected: f32, reward: f32) -> Signal {
        let rpe = reward - expected;
        let err = if rpe < 0.0 { -rpe } else { rpe };
        self.n = self.n.saturating_add(1);
        self.since_reset = self.since_reset.saturating_add(1);
        if self.n == 1 { self.fast = err; self.slow = err; }
        self.fast += self.alpha_fast * (err - self.fast);
        self.slow += self.alpha_slow * (err - self.slow);
        self.dopamine += self.alpha_fast * (rpe - self.dopamine);

        let mut reset = false;
        let surprised = self.fast > self.min_fast && self.fast > self.ratio * self.slow;
        self.streak = if surprised { self.streak.saturating_add(1) } else { 0 };
        if self.since_reset > self.cooldown && self.streak >= self.confirm {
            self.streak = 0;
            reset = true;
            self.resets += 1;
            self.since_reset = 0;
            brain.forget(self.reset_keep);
            let now = brain.clock();
            for s in 0..Brain::<N, D, A>::CAPACITY {
                let old = brain.episode(s).map(|e| now.wrapping_sub(e.t) > self.keep_recent).unwrap_or(false);
                if old { brain.forget_slot(s); }
            }
            // The new world's error level becomes the reference.
            self.slow = self.fast;
            self.explore = self.explore_max;
        } else {
            let x = self.slow / self.err_scale;
            let x = if x > 1.0 { 1.0 } else if x < 0.0 { 0.0 } else { x };
            let target = self.explore_min + (self.explore_max - self.explore_min) * x;
            self.explore += self.settle * (target - self.explore);
        }
        if self.explore < self.explore_min { self.explore = self.explore_min; }
        if self.explore > self.explore_max { self.explore = self.explore_max; }
        brain.explore = self.explore;
        Signal { dopamine: self.dopamine, fast: self.fast, slow: self.slow, explore: self.explore, reset }
    }

    pub fn explore(&self) -> f32 { self.explore }
}

impl Default for Neuromod {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng { fn f(&mut self) -> f32 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; ((self.0 >> 40) as f32) / (1u64 << 24) as f32 } }

    /// Rover thresholds world (`struktura brain demo` world A); after `flip` the
    /// actions are permuted (the world's rule changes: e.g. lunar day -> night).
    fn truth(x: &[f32; 4], flipped: bool) -> u8 {
        let a = if x[2] > 0.8 { 3 } else if x[1] > 0.6 { 2 } else if x[0] > 0.6 { 1 } else { 0 };
        if flipped { [2u8, 3, 0, 1][a as usize] } else { a }
    }

    struct Run { rate: [f32; 40], regret: f32, resets: u32 }

    /// 4000 steps; `flip` = step at which the rule changes (None = stationary).
    fn run(seed: u64, flip: Option<usize>, modulated: bool) -> Run {
        let mut r = Rng(seed);
        let mut br: Brain<256, 4, 4> = Brain::new(0.3, 0.5);
        let mut nm = Neuromod::new();
        let mut rate = [0.0f32; 40];
        let mut regret = 0.0f32;
        for t in 0..4000usize {
            let x = [r.f(), r.f(), r.f(), 1.0];
            let flipped = flip.map(|f| t >= f).unwrap_or(false);
            let d = br.decide(&x, &[true; 4], 3);
            let a = if d.abstained && t < 400 { (t % 4) as u8 } else { d.action };
            let expected = br.estimates(&x)[a as usize].expected;
            let right = a == truth(&x, flipped);
            let rw = if right { 1.0 } else if a == 3 { 0.2 } else { 0.0 };
            regret += 1.0 - rw;
            if right { rate[t / 100] += 0.01; }
            br.learn(&x, a, rw);
            if modulated { nm.observe(&mut br, expected, rw); }
        }
        Run { rate, regret, resets: nm.resets }
    }

    /// Steps after `flip` until a 100-step window reaches 90% of the pre-flip steady rate
    /// (steady = mean of the 5 windows before the flip); 2000 if never.
    fn recovery(r: &Run, flip: usize) -> f32 {
        let f = flip / 100;
        let steady: f32 = r.rate[f - 5..f].iter().sum::<f32>() / 5.0;
        (f..40).find(|&w| r.rate[w] >= 0.9 * steady).map(|w| ((w + 1) * 100 - flip) as f32).unwrap_or(2000.0)
    }

    /// Pre-registered falsifier (set before the run), on 8 seeds and on 8 fresh seeds:
    /// (a) after a mid-stream rule flip, the neuromodulated brain recovers to 90% of its
    ///     pre-flip rate in fewer steps (mean) than the fixed-setting brain;
    /// (b) in a stationary world it is not worse than the fixed brain by more than 1 point
    ///     (right-action rate over the last 1000 steps);
    /// (c) total regret over the flip stream is lower.
    #[test]
    fn falsifier_neuromodulation_recovers_faster_without_hurting_stationary() {
        let flip = 2000usize;
        for seeds in [[11u64, 23, 37, 41, 59, 61, 73, 89], [101, 103, 107, 109, 113, 127, 131, 137]] {
            let k = seeds.len() as f32;
            let (mut rec_f, mut rec_m, mut reg_f, mut reg_m, mut st_f, mut st_m, mut resets_flip, mut resets_stat) = (0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0u32, 0u32);
            for &s in &seeds {
                let (f, m) = (run(s, Some(flip), false), run(s, Some(flip), true));
                rec_f += recovery(&f, flip); rec_m += recovery(&m, flip);
                reg_f += f.regret; reg_m += m.regret; resets_flip += m.resets;
                let (sf, sm) = (run(s, None, false), run(s, None, true));
                st_f += sf.rate[30..40].iter().sum::<f32>() / 10.0;
                st_m += sm.rate[30..40].iter().sum::<f32>() / 10.0;
                resets_stat += sm.resets;
            }
            std::println!("neuromod falsifier (seeds {:?}..): flip world recovery to 90% fixed {:.0} vs neuromod {:.0} steps; regret fixed {:.0} vs neuromod {:.0}; stationary last-1000 rate fixed {:.3} vs neuromod {:.3}; resets: {:.1}/run in flip world, {:.1}/run stationary",
                seeds[0], rec_f / k, rec_m / k, reg_f / k, reg_m / k, st_f / k, st_m / k, resets_flip as f32 / k, resets_stat as f32 / k);
            assert!(rec_m < rec_f, "(a) recovery neuromod {} vs fixed {}", rec_m / k, rec_f / k);
            assert!(st_m / k >= st_f / k - 0.01, "(b) stationary neuromod {} vs fixed {}", st_m / k, st_f / k);
            assert!(reg_m < reg_f, "(c) regret neuromod {} vs fixed {}", reg_m / k, reg_f / k);
        }
    }

    #[test]
    fn exploration_stays_in_bounds_and_is_deterministic() {
        let mut br: Brain<32, 4, 4> = Brain::new(0.3, 0.5);
        let mut nm = Neuromod::new();
        let mut r = Rng(5);
        for i in 0..3000u32 {
            let x = [r.f(), r.f(), r.f(), 1.0];
            let rw = if i % 997 < 500 { r.f() } else { 1.0 - r.f() };
            let s = nm.observe(&mut br, 0.5, rw);
            assert!(s.explore >= nm.explore_min - 1e-6 && s.explore <= nm.explore_max + 1e-6);
            br.learn(&x, (i % 4) as u8, rw);
        }
        let a = run(7, Some(2000), true);
        let b = run(7, Some(2000), true);
        assert_eq!(a.regret, b.regret);
        assert_eq!(a.resets, b.resets);
    }

    #[test]
    fn a_shock_triggers_one_reset_and_quiet_does_not() {
        let mut br: Brain<32, 4, 4> = Brain::new(0.3, 0.5);
        let mut nm = Neuromod::new();
        for _ in 0..1000 { nm.observe(&mut br, 0.9, 1.0); }
        assert_eq!(nm.resets, 0, "a well-predicted world never resets");
        let (mut fired, mut boosted) = (0, false);
        for _ in 0..100 { let s = nm.observe(&mut br, 0.9, 0.0); if s.reset { fired += 1; boosted = (s.explore - nm.explore_max).abs() < 1e-6; } }
        assert_eq!(fired, 1, "cooldown: one reset per shock");
        assert!(boosted, "exploration set to its maximum at the reset");
        // a short spike (shorter than `confirm`) does not reset
        let mut nm2 = Neuromod::new();
        for _ in 0..1000 { nm2.observe(&mut br, 0.9, 1.0); }
        for _ in 0..10 { nm2.observe(&mut br, 0.9, 0.0); }
        for _ in 0..100 { nm2.observe(&mut br, 0.9, 1.0); }
        assert_eq!(nm2.resets, 0, "a 10-step spike is noise, not a change");
    }
}
