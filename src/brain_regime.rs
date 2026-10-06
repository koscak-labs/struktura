//! Regime awareness for [`crate::brain::Brain`]: notice that the world changed,
//! forget the model's confidence, and trust only memories from the current regime.
//!
//! A rover's right response can flip with its environment (lunar day to night,
//! a dust storm, an ageing wheel). The plain brain keeps acting on old evidence:
//! its model is confident and its memory is full of episodes from the old world.
//! [`Regime`] wraps a brain without changing it:
//!
//! 1. **Detect.** Every outcome's surprise (|reward - what it expected|) feeds a
//!    one-sided Page-Hinkley test, the same family as the CUSUM leg of
//!    [`crate::monitor`]: it accumulates surprise above its running mean (minus a
//!    drift allowance `delta`) and declares a change when the accumulation rises
//!    `lambda` above its minimum. Falling surprise (learning) never alarms.
//! 2. **Forget.** On a change it calls [`Brain::forget`] (the model's confidence
//!    shrinks, so new outcomes dominate and the uncertainty bonus re-explores) and
//!    starts a new regime id.
//! 3. **Prefer the current regime.** Every stored episode is tagged with the
//!    regime it was learned in (aligned with memory slots via the slot that
//!    [`Brain::learn`] returns). Decisions use the model plus a memory
//!    correction from the K nearest episodes **of the current regime only**.
//!
//! The model-only prediction comes from [`Brain::predict`], so the current-regime
//! memory correction is computed exactly as the core does it (residual of each
//! neighbour against the model at the neighbour's own situation).
//!
//! No heap, bounded loops, deterministic. `size_of::<Regime<256>>()` is 2 bytes
//! per memory slot plus ~40 bytes.

use crate::brain::{Brain, Decision, K};

/// One-sided Page-Hinkley test for an increase in the mean of a stream.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PageHinkley {
    n: u32,
    mean: f32,
    cum: f32,
    min: f32,
    /// Allowed drift: increases smaller than this per step never accumulate.
    pub delta: f32,
    /// Alarm threshold on the accumulated excess.
    pub lambda: f32,
    /// Samples after a reset before an alarm may fire.
    pub warmup: u32,
}

impl PageHinkley {
    pub const fn new(delta: f32, lambda: f32, warmup: u32) -> Self {
        PageHinkley { n: 0, mean: 0.0, cum: 0.0, min: 0.0, delta, lambda, warmup }
    }

    pub fn reset(&mut self) { self.n = 0; self.mean = 0.0; self.cum = 0.0; self.min = 0.0; }

    /// Feed one value; true when an increase in its mean is declared.
    pub fn update(&mut self, x: f32) -> bool {
        self.n = self.n.saturating_add(1);
        self.mean += (x - self.mean) / self.n as f32;
        self.cum += x - self.mean - self.delta;
        if self.cum < self.min { self.min = self.cum; }
        self.n > self.warmup && self.cum - self.min > self.lambda
    }

    /// Current accumulated excess over its minimum (0 = calm).
    pub fn excess(&self) -> f32 { self.cum - self.min }
}

/// Regime tracking for a brain with memory capacity `N`.
pub struct Regime<const N: usize> {
    ids: [u16; N],
    /// Id of the current regime (0 at start; +1 at every declared change).
    pub current: u16,
    /// Changes declared so far.
    pub changes: u32,
    /// Brain clock at the last declared change.
    pub last_change: u32,
    /// Model confidence kept at a change (passed to [`Brain::forget`]).
    pub keep: f32,
    pub detector: PageHinkley,
}

/// Per-action estimate restricted to the current regime.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegimeEstimate {
    pub expected: f32,
    pub width: f32,
    pub evidence: f32,
    /// Current-regime neighbours of this action that contributed.
    pub same_regime: u8,
}

impl<const N: usize> Regime<N> {
    /// Defaults for rewards in [0, 1]: delta 0.05, lambda 12, warmup 50, keep 0.05.
    /// Tuned on 8 seeds x (stationary + flip) x 6000 steps of the rover world: lambda 10
    /// gave 0 false alarms and 26-68 steps detection delay; 12 adds margin.
    pub const fn new() -> Self { Self::with(0.05, PageHinkley::new(0.05, 12.0, 50)) }

    pub const fn with(keep: f32, detector: PageHinkley) -> Self {
        Regime { ids: [0; N], current: 0, changes: 0, last_change: 0, keep, detector }
    }

    /// Regime id of the episode in `slot`.
    pub fn regime_of(&self, slot: usize) -> Option<u16> { self.ids.get(slot).copied() }

    /// Model + current-regime memory estimate for every action.
    pub fn estimates<const D: usize, const A: usize>(&self, br: &Brain<N, D, A>, x: &[f32; D]) -> [RegimeEstimate; A] {
        let core = br.estimates(x);
        // K nearest episodes of the current regime (bounded scan of memory).
        let mut cs = [0u16; K];
        let mut cd = [f32::INFINITY; K];
        let mut cn = 0usize;
        for s in 0..N {
            let Some(e) = br.episode(s) else { continue };
            if self.ids[s] != self.current { continue; }
            let mut d = 0.0f32;
            for i in 0..D { let t = x[i] - e.key[i]; d += t * t; }
            if cn < K || d < cd[cn - 1] {
                let mut p = if cn < K { cn } else { K - 1 };
                while p > 0 && cd[p - 1] > d { cd[p] = cd[p - 1]; cs[p] = cs[p - 1]; p -= 1; }
                cd[p] = d; cs[p] = s as u16;
                if cn < K { cn += 1; }
            }
        }
        let mut out = [RegimeEstimate { expected: 0.0, width: 0.0, evidence: 0.0, same_regime: 0 }; A];
        for a in 0..A {
            // Exact model-only prediction from the core (same residual correction as Brain::estimate).
            let p = br.predict(a as u8, x);
            let (mut w_cur, mut wres, mut m) = (0.0f32, 0.0f32, 0u8);
            for k in 0..cn {
                if let Some(e) = br.episode(cs[k] as usize) {
                    if e.action as usize != a { continue; }
                    let w = 1.0 / (1.0 + cd[k]);
                    w_cur += w; wres += w * (e.reward - br.predict(a as u8, &e.key)); m += 1;
                }
            }
            let pulls = if br.use_model { br.pulls(a) as f32 } else { 0.0 };
            out[a] = RegimeEstimate { expected: p + wres / (w_cur + 1.0), width: core[a].width, evidence: pulls + w_cur, same_regime: m };
        }
        out
    }

    /// Decide with current-regime memory. Same rules as [`Brain::decide`]: score =
    /// expected + explore x width; disallowed actions never chosen; abstain to
    /// `safe_default` when the best allowed action has less than `min_evidence`.
    pub fn decide<const D: usize, const A: usize>(&self, br: &Brain<N, D, A>, x: &[f32; D], allowed: &[bool; A], safe_default: u8) -> Decision {
        let est = self.estimates(br, x);
        let (slots, _, n) = br.neighbours(x);
        let mut best: Option<(usize, f32)> = None;
        for a in 0..A {
            if !allowed[a] { continue; }
            let score = est[a].expected + br.explore * est[a].width;
            if best.map(|b| score > b.1).unwrap_or(true) { best = Some((a, score)); }
        }
        let mut d = Decision { action: safe_default, abstained: true, expected: 0.0, bonus: 0.0, evidence: 0.0, cited: slots, n_cited: n as u8 };
        if let Some((a, _)) = best {
            d.expected = est[a].expected;
            d.bonus = br.explore * est[a].width;
            d.evidence = est[a].evidence;
            if est[a].evidence >= br.min_evidence { d.action = a as u8; d.abstained = false; }
        }
        d
    }

    /// Learn an outcome: feed its surprise to the detector, tag the stored episode
    /// with the current regime, and on a declared change forget and start a new
    /// regime. Returns true when this outcome triggered a change.
    pub fn learn<const D: usize, const A: usize>(&mut self, br: &mut Brain<N, D, A>, x: &[f32; D], action: u8, reward: f32) -> bool {
        let a = action as usize;
        let expected = if a < A { self.estimates(br, x)[a].expected } else { reward };
        let surprise = if reward > expected { reward - expected } else { expected - reward };
        if let Some(slot) = br.learn(x, action, reward) { if slot < N { self.ids[slot] = self.current; } }
        if self.detector.update(surprise) {
            br.forget(self.keep);
            self.current = self.current.wrapping_add(1);
            self.changes = self.changes.saturating_add(1);
            self.last_change = br.clock();
            self.detector.reset();
            return true;
        }
        false
    }
}

impl<const N: usize> Default for Regime<N> {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Xs(u64);
    impl Xs { fn f(&mut self) -> f32 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; ((self.0 >> 40) as f32) / (1u64 << 24) as f32 } }

    /// Rover fault response [drift, spike, temperature, 1]. Lunar day: overheating
    /// (temp > 0.8) needs safe-mode. Lunar night: freezing (temp < 0.3) does, and a
    /// hot reading is normal. Spike -> quarantine, drift -> recalibrate in both.
    fn truth(x: &[f32; 4], night: bool) -> u8 {
        let thermal = if night { x[2] < 0.3 } else { x[2] > 0.8 };
        if thermal { 3 } else if x[1] > 0.6 { 2 } else if x[0] > 0.6 { 1 } else { 0 }
    }
    fn reward(x: &[f32; 4], a: u8, night: bool) -> f32 { if a == truth(x, night) { 1.0 } else if a == 3 { 0.2 } else { 0.0 } }

    const STEPS: usize = 6000;
    const FLIP: usize = 2000;
    const WIN: usize = 200;

    /// Run one rover; returns (per-step correctness, changes declared, first change step).
    fn run(seed: u64, flip: bool, regime: bool) -> (Vec<bool>, u32, Option<usize>) {
        let mut br: Brain<256, 4, 4> = Brain::new(0.3, 0.5);
        let mut rg: Regime<256> = Regime::new();
        let mut r = Xs(seed);
        let mut ok = Vec::with_capacity(STEPS);
        let mut first = None;
        for t in 0..STEPS {
            let night = flip && t >= FLIP;
            let x = [r.f(), r.f(), r.f(), 1.0];
            let d = if regime { rg.decide(&br, &x, &[true; 4], 3) } else { br.decide(&x, &[true; 4], 3) };
            let a = if d.abstained && t < 400 { (t % 4) as u8 } else { d.action };
            ok.push(a == truth(&x, night));
            let rw = reward(&x, a, night);
            if regime { if rg.learn(&mut br, &x, a, rw) && first.is_none() { first = Some(t); } } else { br.learn(&x, a, rw); }
        }
        (ok, rg.changes, first)
    }

    fn rate(ok: &[bool]) -> f32 { ok.iter().filter(|b| **b).count() as f32 / ok.len() as f32 }

    /// Steps after the flip until the rolling right-action rate is back to 90% of
    /// the pre-flip steady rate (None = never within the run).
    fn recovery(ok: &[bool]) -> (f32, Option<usize>) {
        let steady = rate(&ok[FLIP - 500..FLIP]);
        let target = 0.9 * steady;
        for t in FLIP + WIN..ok.len() {
            if rate(&ok[t - WIN..t]) >= target { return (steady, Some(t - FLIP)); }
        }
        (steady, None)
    }

    #[test]
    fn detector_fires_on_a_jump_and_not_on_learning() {
        let mut ph = PageHinkley::new(0.05, 12.0, 50);
        // Learning: surprise falls from 0.8 to 0.1 -> never an alarm.
        for i in 0..500 { assert!(!ph.update(0.8 - 0.7 * (i as f32 / 500.0).min(1.0)), "false alarm at {}", i); }
        for i in 0..500 { assert!(!ph.update(if i % 7 == 0 { 0.9 } else { 0.1 }), "false alarm on stationary noise at {}", i); }
        let mut fired = None;
        for i in 0..200 { if ph.update(0.8) { fired = Some(i); break; } }
        assert!(fired.map(|i| i < 60).unwrap_or(false), "{:?}", fired);
    }

    #[test]
    fn episodes_are_tagged_with_their_regime() {
        let mut br: Brain<16, 4, 2> = Brain::new(0.0, 0.0);
        let mut rg: Regime<16> = Regime::with(0.1, PageHinkley::new(0.0, 0.5, 0));
        for _ in 0..3 { rg.learn(&mut br, &[0.1, 0.1, 0.1, 1.0], 0, 0.0); }
        let before = rg.current;
        // A big surprise trips the sensitive detector.
        for _ in 0..5 { if rg.learn(&mut br, &[0.1, 0.1, 0.1, 1.0], 0, 1.0) { break; } }
        assert!(rg.current > before);
        rg.learn(&mut br, &[0.9, 0.9, 0.9, 1.0], 1, 1.0);
        let tagged: Vec<u16> = (0..16).filter(|s| br.episode(*s).is_some()).map(|s| rg.regime_of(s).unwrap()).collect();
        assert!(tagged.contains(&before) && tagged.contains(&rg.current), "{:?}", tagged);
    }

    #[test]
    fn falsifier_recovers_faster_after_lunar_night_and_no_false_changes() {
        // Fresh seeds: the detector defaults were tuned on 11..88.
        let seeds = [101u64, 102, 103, 104, 105];
        let (mut plain_t, mut regime_t) = (Vec::new(), Vec::new());
        let mut false_changes = 0u32;
        let mut detect_delay = Vec::new();
        for &s in &seeds {
            let (okp, _, _) = run(s, true, false);
            let (okr, _, first) = run(s, true, true);
            let (sp, rp) = recovery(&okp);
            let (sr, rr) = recovery(&okr);
            let post_p = rate(&okp[FLIP..FLIP + 1000]);
            let post_r = rate(&okr[FLIP..FLIP + 1000]);
            std::println!("seed {}: steady plain {:.3} regime {:.3}; recovery (steps to 90% of steady) plain {:?} regime {:?}; first 1000 after flip plain {:.3} regime {:.3}; change declared at {:?}",
                s, sp, sr, rp, rr, post_p, post_r, first);
            plain_t.push(rp.unwrap_or(STEPS - FLIP));
            regime_t.push(rr.unwrap_or(STEPS - FLIP));
            if let Some(f) = first { detect_delay.push(f as i64 - FLIP as i64); }
            let (_, changes, _) = run(s, false, true);
            false_changes += changes;
        }
        let mean = |v: &[usize]| v.iter().sum::<usize>() as f32 / v.len() as f32;
        std::println!("FALSIFIER: mean recovery plain {:.0} steps, regime {:.0} steps; detection delay {:?}; false changes in {} stationary runs of {} steps: {}",
            mean(&plain_t), mean(&regime_t), detect_delay, seeds.len(), STEPS, false_changes);
        assert_eq!(false_changes, 0, "no false regime change in stationary runs");
        assert!(mean(&regime_t) < mean(&plain_t), "regime awareness must recover faster");
        assert!(regime_t.iter().zip(&plain_t).filter(|(r, p)| r < p).count() >= 4, "faster on at least 4 of 5 seeds");
    }

    #[test]
    fn stationary_accuracy_is_not_worse() {
        let (okp, _, _) = run(7, false, false);
        let (okr, changes, _) = run(7, false, true);
        assert_eq!(changes, 0);
        assert!(rate(&okr[STEPS - 2000..]) >= rate(&okp[STEPS - 2000..]) - 0.02,
            "regime {:.3} vs plain {:.3}", rate(&okr[STEPS - 2000..]), rate(&okp[STEPS - 2000..]));
    }

    #[test]
    fn fits_the_flight_budget() {
        let sz = core::mem::size_of::<Regime<256>>();
        assert!(sz <= 2 * 256 + 64, "{} bytes", sz);
        static _R: Regime<64> = Regime::new();
    }
}
