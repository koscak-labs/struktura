//! Growth: the brain grows new senses (features) it was never given.
//!
//! A situation is `B` base features the caller measures plus `G` growth slots,
//! zero until filled (so the brain's situation size `D` must be `B + G`). Every
//! `every` decisions the grower proposes candidate features from a bounded
//! grammar over the base features:
//!
//! - `Prod(i, j)`: `x_i · x_j` (an interaction),
//! - `Step(i, t)`: `1` if `x_i > t` else `0` (a threshold),
//! - `Gt(i, j)`: `1` if `x_i > x_j` else `0` (a comparison).
//!
//! A candidate is adopted only if it explains the brain's mistakes on memories it
//! was NOT fitted to. In the stochastic sparse spirit of SS-LoRA it draws `R`
//! random half-splits of memory; on one half it fits, per action, how much of the
//! brain's error the candidate accounts for; on the other half it measures the
//! reduction in squared error. It adopts the best candidate only if the held-out
//! reduction is positive in every split and its mean exceeds `min_gain` of the
//! held-out error. Fitting the memories it already has is never enough: the brain
//! grows what generalizes, not what memorizes. Adopting a feature re-computes every
//! stored situation ([`crate::brain::Brain::rekey`]) so memory sees the new sense.
//!
//! no_std, no heap, bounded work per growth check (candidates x R x memory),
//! deterministic for a seed. Growth is opt-in: a brain without a grower is unchanged.

use crate::brain::{Brain, K};

/// A grown feature over the base features.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Feat {
    Off,
    Prod(u8, u8),
    Step(u8, f32),
    Gt(u8, u8),
}

impl Feat {
    pub fn eval(&self, base: &[f32]) -> f32 {
        match *self {
            Feat::Off => 0.0,
            Feat::Prod(i, j) => base[i as usize] * base[j as usize],
            Feat::Step(i, t) => if base[i as usize] > t { 1.0 } else { 0.0 },
            Feat::Gt(i, j) => if base[i as usize] > base[j as usize] { 1.0 } else { 0.0 },
        }
    }
}

/// Thresholds tried for `Step` features.
pub const STEPS: [f32; 3] = [0.25, 0.5, 0.75];

pub struct Grower<const B: usize, const G: usize> {
    feats: [Feat; G],
    grown: usize,
    limit: usize,
    /// Base features that are constants (bias terms) are never used in candidates.
    pub constant: [bool; B],
    /// Decisions between growth checks.
    pub every: u32,
    /// Random half-splits per candidate.
    pub splits: u8,
    /// Minimum mean held-out error reduction, as a fraction of held-out error.
    pub min_gain: f32,
    since: u32,
    rng: u64,
}

/// What a growth check found.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Growth {
    pub adopted: Option<Feat>,
    pub slot: usize,
    /// Mean held-out error reduction fraction of the best candidate.
    pub gain: f32,
    pub candidates: u16,
}

impl<const B: usize, const G: usize> Grower<B, G> {
    pub const fn new(seed: u64) -> Self {
        Grower { feats: [Feat::Off; G], grown: 0, limit: G, constant: [false; B], every: 250, splits: 4, min_gain: 0.05, since: 0, rng: seed | 1 }
    }

    pub fn grown(&self) -> &[Feat] { &self.feats[..self.grown] }

    /// Grown senses currently active (pruned slots are `Off`).
    pub fn active(&self) -> usize { self.feats[..self.grown].iter().filter(|f| **f != Feat::Off).count() }

    /// Slots usable now (a capacity tier); growth never fills past it. Default: all G.
    pub fn set_limit(&mut self, limit: usize) { self.limit = limit.min(G); }
    pub fn limit(&self) -> usize { self.limit }

    /// Prune grown sense `slot` (it stops contributing; memory is re-keyed so the sense reads 0).
    pub fn prune<const N: usize, const D: usize, const A: usize>(&mut self, slot: usize, brain: &mut Brain<N, D, A>) -> bool {
        if slot >= self.grown || self.feats[slot] == Feat::Off || B + G != D { return false; }
        self.feats[slot] = Feat::Off;
        brain.rekey(|k| { let mut y = *k; y[B + slot] = 0.0; y });
        true
    }

    /// Base features followed by the grown features (unfilled slots are 0). `D` must be `B + G`.
    pub fn situation<const D: usize>(&self, base: &[f32; B]) -> [f32; D] {
        let mut x = [0.0f32; D];
        for i in 0..B.min(D) { x[i] = base[i]; }
        for g in 0..self.grown { if B + g < D { x[B + g] = self.feats[g].eval(base); } }
        x
    }

    fn coin(&mut self) -> bool {
        self.rng ^= self.rng << 13; self.rng ^= self.rng >> 7; self.rng ^= self.rng << 17;
        self.rng & 1 == 1
    }

    fn candidate(&self, idx: usize) -> Option<Feat> {
        // Enumerate the grammar deterministically: Prod(i<=j), Step(i,t), Gt(i!=j).
        let mut k = 0usize;
        for i in 0..B { for j in i..B { if self.constant[i] || self.constant[j] { continue; } if k == idx { return Some(Feat::Prod(i as u8, j as u8)); } k += 1; } }
        for i in 0..B { if self.constant[i] { continue; } for t in STEPS { if k == idx { return Some(Feat::Step(i as u8, t)); } k += 1; } }
        for i in 0..B { for j in 0..B { if i == j || self.constant[i] || self.constant[j] { continue; } if k == idx { return Some(Feat::Gt(i as u8, j as u8)); } k += 1; } }
        None
    }

    /// Call after every learned outcome. Every `every` calls it runs a growth check and,
    /// when a candidate generalizes, adopts it into the next free slot and re-keys memory.
    pub fn after_learn<const N: usize, const D: usize, const A: usize>(&mut self, brain: &mut Brain<N, D, A>) -> Option<Growth> {
        self.since += 1;
        let free = (0..self.limit).find(|&s| s >= self.grown || self.feats[s] == Feat::Off);
        if self.since < self.every || free.is_none() || B + G != D { return None; }
        self.since = 0;
        let g = self.check(brain);
        if let Some(f) = g.adopted {
            let slot = free.unwrap();
            self.feats[slot] = f;
            if slot >= self.grown { self.grown = slot + 1; }
            brain.rekey(|k| { let mut base = [0.0f32; B]; for i in 0..B { base[i] = k[i]; } let mut y = *k; y[B + slot] = f.eval(&base); y });
        }
        Some(g)
    }

    /// Evaluate every candidate on random half-splits of memory (held-out error reduction).
    pub fn check<const N: usize, const D: usize, const A: usize>(&mut self, brain: &Brain<N, D, A>) -> Growth {
        let _ = K;
        let mut best = (None, 0.0f32);
        let mut cands = 0u16;
        let mut idx = 0usize;
        while let Some(f) = self.candidate(idx) {
            idx += 1;
            if self.feats[..self.grown].contains(&f) { continue; }
            cands += 1;
            let mut total_gain = 0.0f32;
            let mut all_positive = true;
            for _ in 0..self.splits {
                let seed_state = self.rng;
                // fit: per action, beta = sum(f r) / sum(f^2) on the "train" half
                let (mut sfr, mut sff) = ([0.0f32; A], [0.0f32; A]);
                for s in 0..N {
                    let Some(e) = brain.episode(s) else { continue };
                    if !self.coin() { continue; }
                    let mut base = [0.0f32; B]; for i in 0..B { base[i] = e.key[i]; }
                    let fv = f.eval(&base);
                    let r = e.reward - brain.estimates(&e.key)[e.action as usize].expected;
                    sfr[e.action as usize] += fv * r; sff[e.action as usize] += fv * fv;
                }
                // score on the other half (replay the same coins)
                self.rng = seed_state;
                let (mut before, mut after) = (0.0f32, 0.0f32);
                for s in 0..N {
                    let Some(e) = brain.episode(s) else { continue };
                    if self.coin() { continue; }
                    let a = e.action as usize;
                    let mut base = [0.0f32; B]; for i in 0..B { base[i] = e.key[i]; }
                    let fv = f.eval(&base);
                    let r = e.reward - brain.estimates(&e.key)[a].expected;
                    let beta = if sff[a] > 1e-6 { sfr[a] / sff[a] } else { 0.0 };
                    before += r * r;
                    let rr = r - beta * fv; after += rr * rr;
                }
                // advance the stream so the next split differs
                for _ in 0..7 { self.coin(); }
                let gain = if before > 1e-9 { (before - after) / before } else { 0.0 };
                if gain <= 0.0 { all_positive = false; }
                total_gain += gain;
            }
            let mean = total_gain / self.splits.max(1) as f32;
            if all_positive && mean > best.1 { best = (Some(f), mean); }
        }
        let adopted = if best.1 >= self.min_gain { best.0 } else { None };
        Growth { adopted, slot: self.grown, gain: best.1, candidates: cands }
    }
}

/// The falsifier's worlds and runs, shared by the tests and the brain gym (`crate::brain_gym`).
#[cfg(feature = "std")]
pub mod falsify {
    use super::*;

    struct Rng(u64);
    impl Rng { fn f(&mut self) -> f32 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; ((self.0 >> 40) as f32) / (1u64 << 24) as f32 } }

    /// World B from `struktura brain demo` (interactions a linear model cannot represent).
    fn truth_b(x: &[f32; 4]) -> u8 {
        if x[2] > 0.6 && x[0] > 0.6 { 3 } else if x[1] > 0.6 && x[2] < 0.4 { 2 } else if (x[0] > 0.5) != (x[1] > 0.5) { 1 } else { 0 }
    }
    /// A world the base senses already explain (reward linear in the situation per action).
    fn linear_reward(x: &[f32; 4], a: u8) -> f32 {
        let w = [[0.6f32, 0.1, 0.0], [0.0, 0.7, 0.1], [0.2, 0.0, 0.6], [0.3, 0.3, 0.3]];
        let a = a as usize; w[a][0] * x[0] + w[a][1] * x[1] + w[a][2] * x[2]
    }

    /// Returns (right-action rate over the last 1000, features grown).
    pub fn run_b(seed: u64, grow: bool) -> (f32, usize) {
        let mut r = Rng(seed);
        let mut br: Brain<256, 8, 4> = Brain::new(0.3, 0.5);
        let mut gr: Grower<4, 4> = Grower::new(seed ^ 0xBEEF);
        gr.constant[3] = true;
        let steps = 4000usize;
        let mut ok = 0;
        for t in 0..steps {
            let base = [r.f(), r.f(), r.f(), 1.0];
            let x: [f32; 8] = gr.situation(&base);
            let d = br.decide(&x, &[true; 4], 3);
            let a = if d.abstained && t < 400 { (t % 4) as u8 } else { d.action };
            let right = a == truth_b(&base);
            if t >= steps - 1000 && right { ok += 1; }
            br.learn(&x, a, if right { 1.0 } else if a == 3 { 0.2 } else { 0.0 });
            if grow { gr.after_learn(&mut br); }
        }
        (ok as f32 / 1000.0, gr.grown().len())
    }

    pub fn run_linear(seed: u64) -> usize {
        let mut r = Rng(seed);
        let mut br: Brain<256, 8, 4> = Brain::new(0.3, 0.5);
        let mut gr: Grower<4, 4> = Grower::new(seed ^ 0xBEEF);
        gr.constant[3] = true;
        for t in 0..4000usize {
            let base = [r.f(), r.f(), r.f(), 1.0];
            let x: [f32; 8] = gr.situation(&base);
            let d = br.decide(&x, &[true; 4], 3);
            let a = if d.abstained && t < 400 { (t % 4) as u8 } else { d.action };
            let noise = (r.f() - 0.5) * 0.2;
            br.learn(&x, a, linear_reward(&base, a) + noise);
            gr.after_learn(&mut br);
        }
        gr.grown().len()
    }

    /// The falsifier's accumulators over a seed set (sums, as the test asserts on them);
    /// `k` = number of seeds.
    #[derive(Clone, Copy, Debug)]
    pub struct Report { pub k: f32, pub plain: f32, pub grow: f32, pub grown_b: usize, pub grown_lin: usize }

    impl Report {
        /// The pre-registered verdict, the test's asserts: (1) growth beats no growth by
        /// >= 3 points; (2) < 1 feature grown per run in the linear world.
        pub fn holds(&self) -> bool {
            let k = self.k;
            self.grow / k >= self.plain / k + 0.03 && (self.grown_lin as f32 / k) < 1.0
        }
    }

    pub fn falsifier(seeds: &[u64]) -> Report {
        let (mut plain, mut grow, mut grown_b, mut grown_lin) = (0.0f32, 0.0f32, 0usize, 0usize);
        for &s in seeds {
            plain += run_b(s, false).0;
            let (g, n) = run_b(s, true); grow += g; grown_b += n;
            grown_lin += run_linear(s);
        }
        Report { k: seeds.len() as f32, plain, grow, grown_b, grown_lin }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::falsify::{falsifier, Report};

    /// Pre-registered falsifiers (set before the run), 8 seeds and 8 fresh seeds:
    /// (1) interaction world: the growing brain's right-action rate (last 1000) beats the
    ///     same brain without growth by at least 3 points on average;
    /// (2) a world the base senses already explain: on average fewer than 1 feature grown
    ///     per run (it does not grow without held-out evidence).
    #[test]
    fn falsifier_grows_what_generalizes_and_nothing_else() {
        for seeds in [[11u64, 23, 37, 41, 59, 61, 73, 89], [101, 103, 107, 109, 113, 127, 131, 137]] {
            let rep = falsifier(&seeds);
            let Report { k, plain, grow, grown_b, grown_lin } = rep;
            std::println!("growth falsifier: interaction world right-action (last 1000) plain {:.3} vs growing {:.3} ({:.1} features grown/run); linear world: {:.2} features grown/run",
                plain / k, grow / k, grown_b as f32 / k, grown_lin as f32 / k);
            assert!(grow / k >= plain / k + 0.03, "growth {} vs plain {}", grow / k, plain / k);
            assert!((grown_lin as f32 / k) < 1.0, "grew {} features/run in a world the base senses explain", grown_lin as f32 / k);
            assert!(rep.holds(), "the gym verdict agrees with the asserts");
        }
    }

    #[test]
    fn grammar_is_bounded_and_skips_constants() {
        let mut g: Grower<4, 2> = Grower::new(1);
        g.constant[3] = true;
        let mut n = 0; while g.candidate(n).is_some() { n += 1; }
        assert_eq!(n, 6 + 9 + 6, "prod(i<=j) 6 + step 3x3 + gt 6 over 3 non-constant features");
        assert_eq!(Feat::Prod(0, 1).eval(&[0.5, 0.4, 0.0, 1.0]), 0.2);
        assert_eq!(Feat::Gt(0, 1).eval(&[0.5, 0.4, 0.0, 1.0]), 1.0);
        let x: [f32; 6] = g.situation(&[0.1, 0.2, 0.3, 1.0]);
        assert_eq!(&x[4..], &[0.0, 0.0], "unfilled slots are zero");
    }
}
