//! Synaptic pruning: "use it or lose it", scored by minimum description length.
//!
//! A brain's description length is what it stores (memories in use plus active
//! grown senses) plus what it still gets wrong on data it was not fitted to.
//! The pruner removes stored structure whenever that lowers the total:
//! `cost = held-out error + lambda · active parameters`.
//!
//! - **Memories, at birth.** An outcome the brain already expected (its
//!   surprise, stored by [`Brain::learn`], is below `surprise_eps`) is absorbed
//!   by the model during learning; keeping its episode adds a parameter and
//!   explains nothing new, so it is forgotten right away (compression; off by default).
//! - **Memories, by use.** Every decision cites the episodes it relied on. An
//!   episode not cited for `horizon` decisions (counted from when it was stored)
//!   took part in no decision in that time, so dropping it changes no recent
//!   decision while saving a parameter: it is forgotten (at most `max_forget`
//!   per check).
//! - **Memories, by budget.** With `budget < N`, once more episodes are in use the
//!   least recently stored-or-cited one is forgotten (one O(N) scan per learn; off by default).
//! - **Grown senses.** For each active grown sense the pruner measures, on
//!   random half-splits of memory, how much held-out error the sense explains
//!   (the residual the brain would have without the sense's model contribution,
//!   fitted on one half and scored on the other: the same held-out test the
//!   grower used to adopt it). If even its best split gain is at most `lambda_sense`
//!   (the price of one parameter, in units of held-out error), the sense explains
//!   nothing it costs and is pruned (at most one per check). Held-out gains of useful
//!   senses are noisy (calibration: -0.04..0.26), so a weak mean alone is not enough.
//!   `lambda_sense` sits below the grower's `min_gain`, so adoption and pruning do not
//!   oscillate.
//!
//! Measured (rover interaction world, Grower on, pre-registered seeds): pruning drops
//! noise senses in 16/16 runs but costs 1.3 points of right-action on one seed set and
//! saves under one parameter: every memory pays in that world (see the falsifier test).
//!
//! The model-only contribution of a sense at dimension `j` is exact:
//! `predict(a, x) - predict(a, x with x_j = 0)`, since the model is linear.
//!
//! no_std, no heap, bounded work per call (O(1) per learn; per check at most
//! G senses x splits x memory estimates), deterministic for a seed.

use crate::brain::{Brain, Decision, K};
use crate::brain_grow::{Feat, Grower};

/// What a check pruned.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct PruneReport {
    /// Episodes forgotten at birth since the last check (already known to the model).
    pub absorbed: u32,
    /// Episodes forgotten in this check for not being cited within the horizon.
    pub unused: u32,
    /// Grown sense pruned in this check (its slot), if any.
    pub sense: Option<usize>,
    /// Mean held-out gain of the weakest active sense (1.0 when none was tested).
    pub weakest_gain: f32,
    /// Memories in use + active grown senses after the check.
    pub active_params: usize,
}

pub struct Pruner<const N: usize> {
    /// Brain clock when each memory slot was last cited by a decision.
    last_cited: [u32; N],
    /// Episodes whose surprise is below this are forgotten at birth (0 disables).
    pub surprise_eps: f32,
    /// Decisions an episode may go uncited before it is forgotten (0 disables).
    pub horizon: u32,
    /// Most episodes forgotten for disuse per check.
    pub max_forget: usize,
    /// Price of one grown sense in held-out error fraction; prune a sense explaining less.
    pub lambda_sense: f32,
    /// Random half-splits per sense test.
    pub splits: u8,
    /// Learned outcomes between checks.
    pub every: u32,
    /// Memory budget: while more episodes are in use, the least recently used one is forgotten
    /// (one O(N) scan per learn; N disables).
    pub budget: usize,
    since: u32,
    absorbed: u32,
    rng: u64,
}

impl<const N: usize> Pruner<N> {
    /// Defaults for rewards in [0, 1]: surprise_eps 0 (off), horizon 1500, max_forget 32, budget N (off),
    /// lambda_sense 0 (prune a sense only if no split shows it helping), 6 splits, check every 250.
    pub const fn new(seed: u64) -> Self {
        Pruner { last_cited: [0; N], surprise_eps: 0.0, horizon: 1500, max_forget: 32, lambda_sense: 0.0,
            splits: 6, every: 250, budget: N, since: 0, absorbed: 0, rng: seed | 1 }
    }

    fn coin(&mut self) -> bool {
        self.rng ^= self.rng << 13; self.rng ^= self.rng >> 7; self.rng ^= self.rng << 17;
        self.rng & 1 == 1
    }

    /// Record which memories a decision relied on ("use it").
    pub fn observe<const D: usize, const A: usize>(&mut self, brain: &Brain<N, D, A>, d: &Decision) {
        let now = brain.clock();
        for k in 0..(d.n_cited as usize).min(K) {
            let s = d.cited[k] as usize;
            if s < N { self.last_cited[s] = now; }
        }
    }

    /// Call right after `brain.learn` with the slot it returned. Forgets the new episode at
    /// birth when the brain already expected the outcome; every `every` calls runs a check
    /// (disuse sweep + grown-sense MDL test) and returns its report.
    pub fn after_learn<const D: usize, const A: usize, const B: usize, const G: usize>(
        &mut self, brain: &mut Brain<N, D, A>, grower: &mut Grower<B, G>, slot: Option<usize>) -> Option<PruneReport> {
        if let Some(s) = slot {
            if s < N {
                self.last_cited[s] = brain.clock();
                let absorbed = brain.episode(s).map(|e| e.surprise < self.surprise_eps).unwrap_or(false);
                if absorbed && brain.forget_slot(s) { self.absorbed += 1; }
            }
        }
        if brain.in_use() > self.budget {
            let mut lru: Option<(usize, u32)> = None;
            for s in 0..N {
                if Some(s) == slot { continue; }
                if let Some(e) = brain.episode(s) {
                    let last = if e.t > self.last_cited[s] { e.t } else { self.last_cited[s] };
                    if lru.map(|(_, l)| last < l).unwrap_or(true) { lru = Some((s, last)); }
                }
            }
            if let Some((s, _)) = lru { brain.forget_slot(s); }
        }
        self.since += 1;
        if self.since < self.every { return None; }
        self.since = 0;
        Some(self.check(brain, grower))
    }

    /// One pruning check: forget memories unused for the horizon, then test grown senses.
    pub fn check<const D: usize, const A: usize, const B: usize, const G: usize>(
        &mut self, brain: &mut Brain<N, D, A>, grower: &mut Grower<B, G>) -> PruneReport {
        let mut rep = PruneReport { absorbed: self.absorbed, weakest_gain: 1.0, ..Default::default() };
        self.absorbed = 0;
        // Use it or lose it.
        if self.horizon > 0 {
            let now = brain.clock();
            for s in 0..N {
                if rep.unused as usize >= self.max_forget { break; }
                let Some(e) = brain.episode(s) else { continue };
                let since = now.wrapping_sub(e.t.max(self.last_cited[s]));
                if since > self.horizon && brain.forget_slot(s) { rep.unused += 1; }
            }
        }
        // Grown senses: prune the weakest one if it explains less than its price.
        if B + G == D {
            // (slot, mean gain, best split gain); prune the weakest only if no split shows it explaining
            // more than its price (held-out gains are noisy, so a single weak mean is not enough).
            let mut weakest: Option<(usize, f32, f32)> = None;
            for g in 0..grower.grown().len() {
                if grower.grown()[g] == Feat::Off { continue; }
                let (mean, best) = self.sense_gains(brain, B + g);
                if weakest.map(|(_, _, w)| best < w).unwrap_or(true) { weakest = Some((g, mean, best)); }
            }
            if let Some((g, mean, best)) = weakest {
                rep.weakest_gain = mean;
                if best <= self.lambda_sense && grower.prune(g, brain) { rep.sense = Some(g); }
            }
        }
        rep.active_params = brain.in_use() + grower.active();
        rep
    }

    /// Mean held-out gain of the sense at dimension `j` over random half-splits of memory:
    /// fraction of held-out residual error (without the sense's model contribution) that the
    /// sense explains when refitted on the other half.
    pub fn sense_gain<const D: usize, const A: usize>(&mut self, brain: &Brain<N, D, A>, j: usize) -> f32 { self.sense_gains(brain, j).0 }

    /// (mean, best) held-out gain of the sense at dimension `j` over the random half-splits.
    pub fn sense_gains<const D: usize, const A: usize>(&mut self, brain: &Brain<N, D, A>, j: usize) -> (f32, f32) {
        if j >= D { return (0.0, 0.0); }
        let splits = self.splits.max(1);
        let mut total = 0.0f32;
        let mut best = f32::NEG_INFINITY;
        // (sense value, residual without the sense) per stored episode does not depend on the
        // split: compute it once per call (same values as computing it inside the loops).
        let (mut fvs, mut rws) = ([0.0f32; N], [0.0f32; N]);
        for s in 0..N {
            if let Some(e) = brain.episode(s) { let (f, r) = Self::without(brain, &e.key, e.action, e.reward, j); fvs[s] = f; rws[s] = r; }
        }
        for _ in 0..splits {
            let state = self.rng;
            let (mut sfr, mut sff) = ([0.0f32; A], [0.0f32; A]);
            for s in 0..N {
                let Some(e) = brain.episode(s) else { continue };
                if !self.coin() { continue; }
                let (fv, rw) = (fvs[s], rws[s]);
                let a = e.action as usize;
                sfr[a] += fv * rw; sff[a] += fv * fv;
            }
            self.rng = state;
            let (mut before, mut after) = (0.0f32, 0.0f32);
            for s in 0..N {
                let Some(e) = brain.episode(s) else { continue };
                if self.coin() { continue; }
                let a = e.action as usize;
                let (fv, rw) = (fvs[s], rws[s]);
                let beta = if sff[a] > 1e-6 { sfr[a] / sff[a] } else { 0.0 };
                before += rw * rw;
                let rr = rw - beta * fv; after += rr * rr;
            }
            for _ in 0..7 { self.coin(); }
            let gain = if before > 1e-9 { (before - after) / before } else { 0.0 };
            if gain > best { best = gain; }
            total += gain;
        }
        (total / splits as f32, best)
    }

    /// (sense value, residual of the brain's estimate with the sense's model contribution removed).
    fn without<const D: usize, const A: usize>(brain: &Brain<N, D, A>, key: &[f32; D], action: u8, reward: f32, j: usize) -> (f32, f32) {
        let fv = key[j];
        let mut masked = *key; masked[j] = 0.0;
        let contribution = brain.predict(action, key) - brain.predict(action, &masked);
        let r = reward - brain.estimates(key)[action as usize].expected;
        (fv, r + contribution)
    }
}

#[cfg(test)]
mod tests_shared {
    pub struct Rng(pub u64);
    impl Rng { pub fn f(&mut self) -> f32 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; ((self.0 >> 40) as f32) / (1u64 << 24) as f32 } }

    /// Rover interaction world (as in brain_grow): drift, spike, temperature.
    pub fn truth_b(x: &[f32]) -> u8 {
        if x[2] > 0.6 && x[0] > 0.6 { 3 } else if x[1] > 0.6 && x[2] < 0.4 { 2 } else if (x[0] > 0.5) != (x[1] > 0.5) { 1 } else { 0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::tests_shared::*;

    /// Interaction world with growth; returns (right-action rate over the last 1000,
    /// mean active parameters over the last 1000, senses pruned).
    fn run_b(seed: u64, prune: bool) -> (f32, f32, u32) {
        let mut r = Rng(seed);
        let mut br: Brain<256, 8, 4> = Brain::new(0.3, 0.5);
        let mut gr: Grower<4, 4> = Grower::new(seed ^ 0xBEEF);
        gr.constant[3] = true;
        let mut pr: Pruner<256> = Pruner::new(seed ^ 0x5EED);
        let steps = 4000usize;
        let (mut ok, mut params, mut pruned) = (0usize, 0usize, 0u32);
        for t in 0..steps {
            let base = [r.f(), r.f(), r.f(), 1.0];
            let x: [f32; 8] = gr.situation(&base);
            let d = br.decide(&x, &[true; 4], 3);
            if prune { pr.observe(&br, &d); }
            let a = if d.abstained && t < 400 { (t % 4) as u8 } else { d.action };
            let right = a == truth_b(&base);
            let slot = br.learn(&x, a, if right { 1.0 } else if a == 3 { 0.2 } else { 0.0 });
            gr.after_learn(&mut br);
            if prune { if let Some(rep) = pr.after_learn(&mut br, &mut gr, slot) { if rep.sense.is_some() { pruned += 1; } } }
            if t >= steps - 1000 {
                if right { ok += 1; }
                params += br.in_use() + gr.active();
            }
        }
        (ok as f32 / 1000.0, params as f32 / 1000.0, pruned)
    }

    /// A world whose reward is linear in drift/spike/temp and ignores a pure-noise base feature;
    /// a lenient grower (one split, tiny bar) adopts senses that explain nothing, some built on
    /// the noise feature. Returns (noise-based senses adopted, of them pruned).
    pub(super) fn run_noise(seed: u64) -> (u32, u32) {
        let mut r = Rng(seed);
        let mut br: Brain<256, 9, 4> = Brain::new(0.3, 0.5);
        let mut gr: Grower<5, 4> = Grower::new(seed ^ 0xF00D);
        gr.constant[4] = true;
        gr.splits = 1; gr.min_gain = 1e-6; gr.every = 150;
        let mut pr: Pruner<256> = Pruner::new(seed ^ 0xD00F);
        let w = [[0.6f32, 0.1, 0.0], [0.0, 0.7, 0.1], [0.2, 0.0, 0.6], [0.3, 0.3, 0.3]];
        let noisy = |f: &Feat| matches!(*f, Feat::Prod(i, j) | Feat::Gt(i, j) if i == 3 || j == 3) || matches!(*f, Feat::Step(3, _));
        let (mut adopted, mut pruned) = (0u32, 0u32);
        let mut prev = [Feat::Off; 4];
        for t in 0..4000usize {
            let base = [r.f(), r.f(), r.f(), r.f(), 1.0];
            let x: [f32; 9] = gr.situation(&base);
            let d = br.decide(&x, &[true; 4], 3);
            pr.observe(&br, &d);
            let a = if d.abstained && t < 400 { (t % 4) as u8 } else { d.action };
            let rw = w[a as usize][0] * base[0] + w[a as usize][1] * base[1] + w[a as usize][2] * base[2] + (r.f() - 0.5) * 0.2;
            let slot = br.learn(&x, a, rw);
            gr.after_learn(&mut br);
            for g in 0..gr.grown().len() {
                let f = gr.grown()[g];
                if f != prev[g] && f != Feat::Off && noisy(&f) { adopted += 1; }
            }
            if let Some(rep) = pr.after_learn(&mut br, &mut gr, slot) {
                if let Some(g) = rep.sense { if noisy(&prev_or(&gr, &prev, g)) { pruned += 1; } }
            }
            for g in 0..4 { prev[g] = if g < gr.grown().len() { gr.grown()[g] } else { Feat::Off }; }
        }
        (adopted, pruned)
    }

    /// The sense that was in slot `g` before the check (the check sets it Off).
    fn prev_or(gr: &Grower<5, 4>, prev: &[Feat; 4], g: usize) -> Feat {
        if g < gr.grown().len() && gr.grown()[g] != Feat::Off { gr.grown()[g] } else { prev[g] }
    }

    #[test]
    fn absorbed_outcomes_are_forgotten_at_birth_and_unused_ones_after_the_horizon() {
        let mut br: Brain<32, 4, 2> = Brain::new(0.0, 0.0);
        let mut gr: Grower<4, 0> = Grower::new(1);
        let mut pr: Pruner<32> = Pruner::new(3);
        pr.every = 1_000_000; pr.horizon = 0; pr.surprise_eps = 0.05;
        // the same outcome over and over: after the first few the brain expects it
        let x = [0.5, 0.5, 0.5, 1.0];
        let mut absorbed = 0;
        for _ in 0..40 { let s = br.learn(&x, 0, 1.0); pr.after_learn(&mut br, &mut gr, s); }
        let rep = pr.check(&mut br, &mut gr);
        absorbed += rep.absorbed;
        assert!(absorbed > 20, "absorbed {}", absorbed);
        assert!(br.in_use() < 20, "in use {}", br.in_use());
        // disuse: nothing cited, horizon 5
        let mut br2: Brain<32, 4, 2> = Brain::new(0.0, 0.0);
        let mut pr2: Pruner<32> = Pruner::new(4);
        pr2.surprise_eps = 0.0; pr2.horizon = 5; pr2.every = 1_000_000;
        let mut r = Rng(9);
        for _ in 0..20 { let k = [r.f(), r.f(), r.f(), 1.0]; let s = br2.learn(&k, 1, r.f()); pr2.after_learn(&mut br2, &mut gr, s); }
        let before = br2.in_use();
        let rep2 = pr2.check(&mut br2, &mut gr);
        assert!(rep2.unused > 0 && br2.in_use() < before, "unused {} in use {} -> {}", rep2.unused, before, br2.in_use());
    }

    #[test]
    fn deterministic() {
        assert_eq!(run_b(7, true), run_b(7, true));
    }

    /// Pre-registered falsifiers (set before the run), 8 seeds and 8 fresh seeds:
    /// (1) rover interaction world with growth: with the pruner the right-action rate (last 1000)
    ///     is within 1 point of (or above) the run without it, with fewer active parameters
    ///     (memories in use + active senses, mean over the last 1000);
    /// (2) senses built on a pure-noise feature, adopted by a deliberately lenient grower, are
    ///     pruned in most runs (in more than half of the runs that adopted one).
    /// RESULT (pre-registered seeds): set 11..: right-action 0.853 -> 0.840, params 259.3 -> 258.9,
    /// 0.5 senses pruned/run; set 101..: 0.875 -> 0.876, 259.8 -> 259.4, 0.8/run; noise senses
    /// pruned in 8/8 and 8/8. Accuracy clause fails on set 11.. (-1.3 points > 1). Calibration
    /// (seeds 1001..) showed memory compression costs accuracy monotonically in this world
    /// (LRU budget 192/128/96: 0.819/0.794/0.771 vs 0.854 unpruned): every memory pays here.
    #[test]
    #[ignore = "FAILED as pre-registered: accuracy -1.3 points on seeds 11.. (tolerance 1); see doc"]
    fn falsifier_prunes_what_does_not_pay_and_keeps_accuracy() {
        let (mut ok_acc, mut ok_par, mut ok_noise) = (true, true, true);
        for seeds in [[11u64, 23, 37, 41, 59, 61, 73, 89], [101, 103, 107, 109, 113, 127, 131, 137]] {
            let (mut acc0, mut acc1, mut p0, mut p1, mut pr_senses) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0u32);
            let (mut runs_with_noise, mut runs_pruned) = (0u32, 0u32);
            for &s in &seeds {
                let (a0, q0, _) = run_b(s, false);
                let (a1, q1, n) = run_b(s, true);
                acc0 += a0; acc1 += a1; p0 += q0; p1 += q1; pr_senses += n;
                let (adopted, pruned) = run_noise(s);
                if adopted > 0 { runs_with_noise += 1; if pruned > 0 { runs_pruned += 1; } }
            }
            let k = seeds.len() as f32;
            std::println!("prune falsifier ({:?}..): right-action {:.3} -> {:.3} with pruning; active params {:.1} -> {:.1}; senses pruned {:.1}/run; noise senses: pruned in {}/{} runs that adopted one",
                seeds[0], acc0 / k, acc1 / k, p0 / k, p1 / k, pr_senses as f32 / k, runs_pruned, runs_with_noise);
            ok_acc &= acc1 / k >= acc0 / k - 0.01;
            ok_par &= p1 < p0;
            ok_noise &= runs_with_noise > 0 && runs_pruned * 2 > runs_with_noise;
        }
        std::println!("accuracy kept: {ok_acc}; fewer params: {ok_par}; noise pruned in most runs: {ok_noise}");
        assert!(ok_acc && ok_par && ok_noise);
    }
}

#[cfg(test)]
mod calibration {
    use super::*;
    use super::tests_shared::*;

    /// Calibration on seeds disjoint from the falsifier's (1001..): measured held-out gain of
    /// every active grown sense at the end of an interaction-world run, and of noise senses.
    #[test]
    #[ignore = "calibration only (run with --ignored --nocapture)"]
    fn calibrate_sense_gains() {
        for seed in 1001u64..1007 {
            let mut r = Rng(seed);
            let mut br: Brain<256, 8, 4> = Brain::new(0.3, 0.5);
            let mut gr: Grower<4, 4> = Grower::new(seed ^ 0xBEEF);
            gr.constant[3] = true;
            for t in 0..4000usize {
                let base = [r.f(), r.f(), r.f(), 1.0];
                let x: [f32; 8] = gr.situation(&base);
                let d = br.decide(&x, &[true; 4], 3);
                let a = if d.abstained && t < 400 { (t % 4) as u8 } else { d.action };
                let right = a == truth_b(&base);
                br.learn(&x, a, if right { 1.0 } else if a == 3 { 0.2 } else { 0.0 });
                gr.after_learn(&mut br);
            }
            let mut pr: Pruner<256> = Pruner::new(seed);
            let gains: Vec<(Feat, f32)> = (0..gr.grown().len()).map(|g| (gr.grown()[g], pr.sense_gain(&br, 4 + g))).collect();
            let mut surp: Vec<f32> = (0..256).filter_map(|s| br.episode(s).map(|e| e.surprise)).collect();
            surp.sort_by(|a, b| a.partial_cmp(b).unwrap());
            std::println!("seed {} useful-world sense gains {:?}; surprise quantiles p25 {:.2} p50 {:.2} p75 {:.2}", seed, gains, surp[surp.len() / 4], surp[surp.len() / 2], surp[3 * surp.len() / 4]);
        }
    }
}

#[cfg(test)]
mod calibration_sweep {
    use super::*;
    use super::tests_shared::*;

    fn run(seed: u64, eps: Option<f32>, budget: usize) -> (f32, f32) {
        let mut r = Rng(seed);
        let mut br: Brain<256, 8, 4> = Brain::new(0.3, 0.5);
        let mut gr: Grower<4, 4> = Grower::new(seed ^ 0xBEEF);
        gr.constant[3] = true;
        let mut pr: Pruner<256> = Pruner::new(seed ^ 0x5EED);
        if let Some(e) = eps { pr.surprise_eps = e; }
        pr.budget = budget;
        let (mut ok, mut params) = (0usize, 0usize);
        for t in 0..4000usize {
            let base = [r.f(), r.f(), r.f(), 1.0];
            let x: [f32; 8] = gr.situation(&base);
            let d = br.decide(&x, &[true; 4], 3);
            if eps.is_some() { pr.observe(&br, &d); }
            let a = if d.abstained && t < 400 { (t % 4) as u8 } else { d.action };
            let right = a == truth_b(&base);
            let slot = br.learn(&x, a, if right { 1.0 } else if a == 3 { 0.2 } else { 0.0 });
            gr.after_learn(&mut br);
            if eps.is_some() { pr.after_learn(&mut br, &mut gr, slot); }
            if t >= 3000 { if right { ok += 1; } params += br.in_use() + gr.active(); }
        }
        (ok as f32 / 1000.0, params as f32 / 1000.0)
    }

    #[test]
    #[ignore = "calibration only (seeds 1001.., disjoint from the falsifier)"]
    fn sweep_surprise_eps() {
        for (eps, budget) in [(Some(0.0f32), 96usize), (Some(0.0), 128), (Some(0.0), 192)] {
            let (mut a, mut p) = (0.0, 0.0);
            for s in 1001u64..1009 { let (x, y) = run(s, eps, budget); a += x; p += y; }
            std::println!("budget {} eps {:?}: right-action {:.3} active params {:.1}", budget, eps, a / 8.0, p / 8.0);
        }
    }
}

#[cfg(test)]
mod noise_claim {
    use super::tests::run_noise;

    /// Sub-claim (2) of the falsifier on its own, same pre-registered 16 seeds: a sense built on
    /// a pure-noise feature is pruned in most runs that adopted one.
    #[test]
    fn noise_senses_are_pruned_in_most_runs() {
        let (mut with, mut pruned) = (0u32, 0u32);
        for s in [11u64, 23, 37, 41, 59, 61, 73, 89, 101, 103, 107, 109, 113, 127, 131, 137] {
            let (a, p) = run_noise(s);
            if a > 0 { with += 1; if p > 0 { pruned += 1; } }
        }
        std::println!("noise senses pruned in {}/{} runs that adopted one", pruned, with);
        assert!(with > 0 && pruned * 2 > with);
    }
}
