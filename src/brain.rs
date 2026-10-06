//! Native brain: retrieval-augmented contextual decisions in fixed memory.
//!
//! A decision maker small and predictable enough for a flight computer:
//! no heap, no model, no unbounded loop, deterministic. It learns which
//! action pays off in which situation from its own outcomes.
//!
//! - **Situation**: a feature vector of `D` numbers chosen by the caller
//!   (e.g. a signal [`crate::fingerprint`], a residual z, a temperature).
//!   Scale features to comparable ranges; distances are plain Euclidean.
//! - **Memory (retrieval)**: a ring of `N` past episodes (situation, action,
//!   reward). A decision looks up the `K` nearest past situations and uses
//!   their outcomes as local evidence, and reports which episodes it used.
//!   When full, the episode that is least surprising (weighted by age) is
//!   forgotten, so rare and unexpected outcomes are kept longest.
//! - **Model (deliberation)**: per action a ridge-regression estimate of
//!   reward as a linear function of the situation (LinUCB), updated with a
//!   rank-one inverse update in O(D^2), plus an uncertainty bonus that sends
//!   it to try actions it knows little about.
//! - **Safety**: actions the caller marks as not allowed are never chosen;
//!   when the evidence behind the best allowed action is below
//!   `min_evidence`, it abstains and returns the caller's safe default.
//!
//! Memory: `size_of::<Brain<N, D, A>>()` is about `N * (4 D + 16) + A * 4 (D^2 + D + 1)`
//! bytes; a 256-episode, 8-feature, 4-action brain is about 13 KB.

use crate::sqrt;

#[inline]
fn absf(x: f32) -> f32 { if x < 0.0 { -x } else { x } }

/// Neighbours consulted per decision.
pub const K: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Episode<const D: usize> {
    pub key: [f32; D],
    pub action: u8,
    pub reward: f32,
    /// |reward - what the brain expected| when it was stored.
    pub surprise: f32,
    /// Brain clock when stored.
    pub t: u32,
    pub used: bool,
}

impl<const D: usize> Episode<D> {
    pub const EMPTY: Self = Episode { key: [0.0; D], action: 0, reward: 0.0, surprise: 0.0, t: 0, used: false };
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Decision {
    pub action: u8,
    /// True when the evidence was too thin and the safe default was returned.
    pub abstained: bool,
    /// Expected reward of the chosen action (blend of model and memory).
    pub expected: f32,
    /// Uncertainty bonus that was added to rank it.
    pub bonus: f32,
    /// Evidence behind the chosen action: its updates + similarity weight of its neighbours.
    pub evidence: f32,
    /// Memory slots of the neighbours consulted (first `n_cited` are valid), nearest first.
    pub cited: [u16; K],
    pub n_cited: u8,
}

pub struct Brain<const N: usize, const D: usize, const A: usize> {
    mem: [Episode<D>; N],
    len: usize,
    clock: u32,
    a_inv: [[[f32; D]; D]; A],
    b: [[f32; D]; A],
    pulls: [u32; A],
    /// Exploration weight on the uncertainty bonus.
    pub explore: f32,
    /// Minimum evidence for the chosen action before the brain will act on its own.
    pub min_evidence: f32,
    /// Episodes older than this many decisions count half as much when deciding what to forget.
    pub half_life: u32,
    /// Use the linear model (false = memory only; for ablations and very small situations).
    pub use_model: bool,
}

impl<const N: usize, const D: usize, const A: usize> Brain<N, D, A> {
    /// A blank brain. `const`, so it can live in a `static` on bare metal.
    pub const fn new(explore: f32, min_evidence: f32) -> Self {
        let mut a_inv = [[[0.0f32; D]; D]; A];
        let mut a = 0;
        while a < A {
            let mut i = 0;
            while i < D { a_inv[a][i][i] = 1.0; i += 1; }
            a += 1;
        }
        Brain { mem: [Episode::EMPTY; N], len: 0, clock: 0, a_inv, b: [[0.0; D]; A], pulls: [0; A],
            explore, min_evidence, half_life: 64, use_model: true }
    }

    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }
    pub fn episode(&self, slot: usize) -> Option<&Episode<D>> { self.mem.get(slot).filter(|e| e.used) }
    pub fn pulls(&self, action: usize) -> u32 { if action < A { self.pulls[action] } else { 0 } }

    fn theta(&self, a: usize) -> [f32; D] {
        let mut t = [0.0f32; D];
        for i in 0..D { let mut s = 0.0; for j in 0..D { s += self.a_inv[a][i][j] * self.b[a][j]; } t[i] = s; }
        t
    }

    fn width(&self, a: usize, x: &[f32; D]) -> f32 {
        let mut q = 0.0f32;
        for i in 0..D { let mut s = 0.0; for j in 0..D { s += self.a_inv[a][i][j] * x[j]; } q += x[i] * s; }
        sqrt(q.max(0.0) as f64) as f32
    }

    fn dist2(a: &[f32; D], b: &[f32; D]) -> f32 {
        let mut s = 0.0; for i in 0..D { let d = a[i] - b[i]; s += d * d; } s
    }

    /// The K nearest stored episodes to `x` (slots and squared distances), nearest first.
    fn nearest(&self, x: &[f32; D]) -> ([u16; K], [f32; K], usize) {
        let mut slots = [0u16; K];
        let mut d2 = [f32::INFINITY; K];
        let mut n = 0usize;
        for (s, e) in self.mem.iter().enumerate() {
            if !e.used { continue; }
            let d = Self::dist2(x, &e.key);
            if n < K || d < d2[n - 1] {
                let mut p = if n < K { n } else { K - 1 };
                while p > 0 && d2[p - 1] > d { d2[p] = d2[p - 1]; slots[p] = slots[p - 1]; p -= 1; }
                d2[p] = d; slots[p] = s as u16;
                if n < K { n += 1; }
            }
        }
        (slots, d2, n)
    }

    /// Model + memory estimate for one action: (expected reward, evidence).
    /// The model gives a global linear prediction; memory corrects it locally with the
    /// model's average error on the nearest past episodes of the same action (shrunk
    /// toward no correction when there are few), so remembered situations fix the
    /// regions a linear model cannot represent.
    fn estimate(&self, a: usize, x: &[f32; D], nb: &([u16; K], [f32; K], usize)) -> (f32, f32) {
        let th = self.theta(a);
        let predict = |v: &[f32; D]| -> f32 { if self.use_model { let mut s = 0.0; for i in 0..D { s += th[i] * v[i]; } s } else { 0.0 } };
        let (mut w, mut wres) = (0.0f32, 0.0f32);
        for k in 0..nb.2 {
            let e = &self.mem[nb.0[k] as usize];
            if e.action as usize != a { continue; }
            let wk = 1.0 / (1.0 + nb.1[k]);
            w += wk; wres += wk * (e.reward - predict(&e.key));
        }
        let correction = wres / (w + 1.0);
        let model_n = if self.use_model { self.pulls[a] as f32 } else { 0.0 };
        (predict(x) + correction, model_n + w)
    }

    /// Choose an action for situation `x`. `allowed[a] == false` is never chosen;
    /// `safe_default` is returned (abstained) when the best allowed action has
    /// less than `min_evidence` behind it, or when nothing is allowed.
    pub fn decide(&self, x: &[f32; D], allowed: &[bool; A], safe_default: u8) -> Decision {
        let nb = self.nearest(x);
        let mut best: Option<(usize, f32, f32, f32, f32)> = None; // (a, score, expected, bonus, evidence)
        for a in 0..A {
            if !allowed[a] { continue; }
            let (expected, evidence) = self.estimate(a, x, &nb);
            let bonus = self.explore * self.width(a, x);
            let score = expected + bonus;
            if best.map(|b| score > b.1).unwrap_or(true) { best = Some((a, score, expected, bonus, evidence)); }
        }
        let mut d = Decision { action: safe_default, abstained: true, expected: 0.0, bonus: 0.0, evidence: 0.0,
            cited: nb.0, n_cited: nb.2 as u8 };
        if let Some((a, _, expected, bonus, evidence)) = best {
            d.expected = expected; d.bonus = bonus; d.evidence = evidence;
            if evidence >= self.min_evidence { d.action = a as u8; d.abstained = false; }
        }
        d
    }

    /// Record the outcome of taking `action` in situation `x`: update the model and remember the episode.
    pub fn learn(&mut self, x: &[f32; D], action: u8, reward: f32) {
        let a = action as usize;
        if a >= A { return; }
        let nb = self.nearest(x);
        let (expected, _) = self.estimate(a, x, &nb);
        // Sherman-Morrison: A^-1 <- A^-1 - (A^-1 x)(x^T A^-1) / (1 + x^T A^-1 x)
        let mut u = [0.0f32; D];
        for i in 0..D { let mut s = 0.0; for j in 0..D { s += self.a_inv[a][i][j] * x[j]; } u[i] = s; }
        let mut den = 1.0; for i in 0..D { den += x[i] * u[i]; }
        for i in 0..D { for j in 0..D { self.a_inv[a][i][j] -= u[i] * u[j] / den; } }
        for i in 0..D { self.b[a][i] += reward * x[i]; }
        self.pulls[a] = self.pulls[a].saturating_add(1);
        self.clock = self.clock.wrapping_add(1);
        let ep = Episode { key: *x, action, reward, surprise: absf(reward - expected), t: self.clock, used: true };
        if self.len < N {
            self.mem[self.len] = ep;
            self.len += 1;
        } else if N > 0 {
            // Forget the least valuable memory: least surprising, discounted by age.
            let mut worst = 0usize;
            let mut worst_v = f32::INFINITY;
            for (s, e) in self.mem.iter().enumerate() {
                let age = self.clock.wrapping_sub(e.t) as f32;
                let v = e.surprise / (1.0 + age / self.half_life.max(1) as f32);
                if v < worst_v { worst_v = v; worst = s; }
            }
            self.mem[worst] = ep;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u64);
    impl Lcg { fn f(&mut self) -> f32 { self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((self.0 >> 40) as f32) / (1u64 << 24) as f32 } }

    /// Rover-like rule: feature 0 = drift, feature 1 = spike, feature 2 = bias 1.
    /// Best action: 0 ignore (calm), 1 recalibrate (drift), 2 quarantine (spike).
    fn world(x: &[f32; 3], a: u8) -> f32 {
        let best = if x[1] > 0.6 { 2 } else if x[0] > 0.6 { 1 } else { 0 };
        if a == best { 1.0 } else { 0.0 }
    }

    #[test]
    fn learns_a_contextual_rule_and_cites_memory() {
        let mut br: Brain<256, 3, 3> = Brain::new(0.3, 0.5);
        let mut r = Lcg(1);
        for _ in 0..1500 {
            let x = [r.f(), r.f(), 1.0];
            let d = br.decide(&x, &[true; 3], 0);
            // When abstaining, still explore deterministically so it gathers evidence.
            let a = if d.abstained { (br.clock % 3) as u8 } else { d.action };
            br.learn(&x, a, world(&x, a));
        }
        br.explore = 0.0;
        let (mut ok, mut n) = (0, 0);
        for _ in 0..400 {
            let x = [r.f(), r.f(), 1.0];
            if absf(x[0] - 0.6) < 0.08 || absf(x[1] - 0.6) < 0.08 { continue; } // skip the exact boundary
            let d = br.decide(&x, &[true; 3], 0);
            n += 1;
            if world(&x, d.action) == 1.0 { ok += 1; }
            assert!(d.n_cited as usize == K, "a full memory always returns K neighbours");
        }
        assert!(ok as f32 / n as f32 > 0.9, "{}/{}", ok, n);
    }

    #[test]
    fn never_picks_a_disallowed_action() {
        let mut br: Brain<32, 3, 3> = Brain::new(0.0, 0.0);
        for _ in 0..50 { br.learn(&[0.9, 0.0, 1.0], 1, 1.0); br.learn(&[0.9, 0.0, 1.0], 0, 0.0); }
        let d = br.decide(&[0.9, 0.0, 1.0], &[true, false, true], 0);
        assert_ne!(d.action, 1);
        let none = br.decide(&[0.9, 0.0, 1.0], &[false; 3], 2);
        assert_eq!((none.action, none.abstained), (2, true));
    }

    #[test]
    fn abstains_to_the_safe_default_without_evidence() {
        let br: Brain<16, 3, 4> = Brain::new(1.0, 2.0);
        let d = br.decide(&[0.5, 0.5, 1.0], &[true; 4], 3);
        assert!(d.abstained);
        assert_eq!(d.action, 3);
        assert_eq!(d.n_cited, 0);
    }

    #[test]
    fn cites_the_episode_it_remembers() {
        let mut br: Brain<64, 3, 2> = Brain::new(0.0, 0.0);
        let mut r = Lcg(5);
        for _ in 0..40 { br.learn(&[r.f(), r.f(), 1.0], 0, 0.1); }
        let x = [0.123, 0.987, 1.0];
        br.learn(&x, 1, 1.0);
        let d = br.decide(&x, &[true; 2], 0);
        let slot = d.cited[0] as usize;
        assert_eq!(br.episode(slot).unwrap().key, x, "nearest citation is the identical past situation");
    }

    #[test]
    fn memory_is_bounded_and_keeps_the_surprise() {
        let mut br: Brain<32, 3, 2> = Brain::new(0.0, 0.0);
        for _ in 0..20 { br.learn(&[0.5, 0.5, 1.0], 0, 0.0); } // routine
        br.learn(&[0.9, 0.1, 1.0], 0, 10.0);                  // one shock
        for i in 0..400 { br.learn(&[0.5, 0.5 + (i % 3) as f32 * 0.001, 1.0], 0, 0.0); }
        assert_eq!(br.len(), 32);
        assert!((0..32).any(|s| br.episode(s).map(|e| e.reward == 10.0).unwrap_or(false)), "the surprising episode is still remembered");
    }

    #[test]
    fn memory_beats_the_model_alone_where_the_world_has_interactions() {
        // quarantine only on a spike while cool; safe-mode only when hot AND drifting; drift XOR spike: recalibrate
        let truth = |x: &[f32; 4]| -> u8 { if x[2] > 0.6 && x[0] > 0.6 { 3 } else if x[1] > 0.6 && x[2] < 0.4 { 2 } else if (x[0] > 0.5) != (x[1] > 0.5) { 1 } else { 0 } };
        fn late<const N: usize>(truth: &dyn Fn(&[f32; 4]) -> u8) -> u32 {
            let mut br: Brain<N, 4, 4> = Brain::new(0.3, 0.5);
            let mut r = Lcg(77);
            let mut ok = 0;
            for t in 0..4000u32 {
                let x = [r.f(), r.f(), r.f(), 1.0];
                let d = br.decide(&x, &[true; 4], 3);
                let a = if d.abstained && t < 400 { (t % 4) as u8 } else { d.action };
                if t >= 3000 && a == truth(&x) { ok += 1; }
                br.learn(&x, a, if a == truth(&x) { 1.0 } else if a == 3 { 0.2 } else { 0.0 });
            }
            ok
        }
        let (with_mem, model_only) = (late::<256>(&truth), late::<0>(&truth));
        assert!(with_mem >= model_only + 50, "memory+model {}/1000 vs model only {}/1000", with_mem, model_only);
    }

    #[test]
    fn deterministic() {
        let run = || { let mut br: Brain<64, 3, 3> = Brain::new(0.2, 0.5); let mut r = Lcg(9);
            let mut out = 0u32;
            for _ in 0..300 { let x = [r.f(), r.f(), 1.0]; let d = br.decide(&x, &[true; 3], 0);
                let a = if d.abstained { (out % 3) as u8 } else { d.action }; out = out.wrapping_mul(31).wrapping_add(a as u32 + 1); br.learn(&x, a, world(&x, a)); }
            out };
        assert_eq!(run(), run());
    }

    #[test]
    fn fits_a_flight_computer_budget() {
        let sz = core::mem::size_of::<Brain<256, 8, 4>>();
        assert!(sz < 16 * 1024, "{} bytes", sz);
        // const-constructible: can live in a static on bare metal
        static _B: Brain<16, 4, 3> = Brain::new(0.1, 1.0);
    }
}
