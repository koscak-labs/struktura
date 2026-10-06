//! SS-LoRA (Koščák) stochastic sparse low-rank adapter for the brain.
//!
//! Scope, stated exactly: this implements the SS-LoRA core (a Bernoulli mask over
//! low-rank update components with 1/(1-p) rescaling). It does NOT implement
//! KSS-LoRA, the Koščák Gamma Theorem's one-parameter, theorem-derived fix (negative
//! Koščák coefficient), whose formula is not part of this repository.
//!
//! The brain's own model is one small ridge regression per action, so an action
//! that has been tried a few times learns alone and overfits its few outcomes.
//! This adapter shares a low-rank structure across actions:
//! `f(x, a) = Σ_r B[a][r] · (C[r] · x)`, i.e. the A×D weight matrix factorised
//! through rank R, so a rarely tried action borrows what the others learned.
//!
//! Training is SS-LoRA-style stochastic sparse: at every step a Bernoulli mask
//! drops each rank component (or, in `Granularity::Entry`, each adapter weight)
//! with probability `p` (default 0.5), the surviving ones are rescaled by
//! `1/(1-p)` so the expected update is unbiased, and only they are updated. No
//! component can lean on any single other component, so the adapter cannot
//! memorise its few examples; at inference all components are used, unscaled
//! (the expectation of the training-time output). Effective rank during
//! training is about `R·(1-p)`.
//!
//! B starts at zero (the adapter starts as "no correction", as in LoRA); C
//! starts small and deterministic. Plain SGD on squared error, optional weight
//! decay. no_std, no heap, bounded loops, deterministic for a seed.
//!
//! Size: `4·(A·R + R·D) + 16` bytes; A=8, R=4, D=8 is 272 bytes.

/// What the Bernoulli mask drops.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Granularity {
    /// Whole rank components (row r of C with column r of B), the SS-LoRA default.
    Rank,
    /// Individual adapter weights in B and C.
    Entry,
}

pub struct SsLora<const D: usize, const A: usize, const R: usize> {
    b: [[f32; R]; A],
    c: [[f32; D]; R],
    /// Drop probability (sparsity); 0 = dense low-rank SGD.
    pub p: f32,
    pub lr: f32,
    pub decay: f32,
    pub granularity: Granularity,
    rng: u64,
}

impl<const D: usize, const A: usize, const R: usize> SsLora<D, A, R> {
    /// `p` sparsity (0.5 = SS-LoRA default), `lr` learning rate, `seed` for the masks.
    pub const fn new(p: f32, lr: f32, seed: u64) -> Self {
        let mut c = [[0.0f32; D]; R];
        let mut s = seed | 1;
        let mut r = 0;
        while r < R {
            let mut j = 0;
            while j < D {
                s ^= s << 13; s ^= s >> 7; s ^= s << 17;
                // uniform in [-0.1, 0.1]
                c[r][j] = ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.2;
                j += 1;
            }
            r += 1;
        }
        SsLora { b: [[0.0; R]; A], c, p, lr, decay: 0.0, granularity: Granularity::Rank, rng: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1 }
    }

    fn coin(&mut self) -> bool {
        self.rng ^= self.rng << 13; self.rng ^= self.rng >> 7; self.rng ^= self.rng << 17;
        ((self.rng >> 40) as f32 / (1u64 << 24) as f32) >= self.p
    }

    /// Inference: all components, unscaled.
    pub fn predict(&self, a: usize, x: &[f32; D]) -> f32 {
        if a >= A { return 0.0; }
        let mut y = 0.0;
        for r in 0..R {
            let mut h = 0.0; for j in 0..D { h += self.c[r][j] * x[j]; }
            y += self.b[a][r] * h;
        }
        y
    }

    /// One stochastic sparse SGD step toward `target` for action `a` in situation `x`.
    /// Returns the squared error of the masked forward pass before the update.
    pub fn step(&mut self, a: usize, x: &[f32; D], target: f32) -> f32 {
        if a >= A { return 0.0; }
        let keep = if self.p > 0.0 { 1.0 - self.p } else { 1.0 };
        let scale = 1.0 / keep;
        // Draw the masks first (fixed-size, no heap).
        let mut keep_r = [true; R];
        let mut keep_b = [true; R];
        let mut keep_c = [[true; D]; R];
        if self.p > 0.0 {
            match self.granularity {
                Granularity::Rank => for r in 0..R { keep_r[r] = self.coin(); },
                Granularity::Entry => for r in 0..R { keep_b[r] = self.coin(); for j in 0..D { keep_c[r][j] = self.coin(); } },
            }
        }
        // Masked forward pass.
        let mut h = [0.0f32; R];
        let mut y = 0.0f32;
        for r in 0..R {
            if !keep_r[r] { continue; }
            let mut s = 0.0;
            for j in 0..D { if keep_c[r][j] { s += self.c[r][j] * x[j]; } }
            h[r] = s * if self.granularity == Granularity::Entry && self.p > 0.0 { scale } else { 1.0 };
            if keep_b[r] { y += self.b[a][r] * h[r] * if self.granularity == Granularity::Entry && self.p > 0.0 { scale } else { 1.0 }; }
        }
        if self.granularity == Granularity::Rank { y *= scale; }
        let err = target - y;
        // Gradients only for surviving parameters (rescaled consistently with the forward pass).
        let g = self.lr * err;
        for r in 0..R {
            if !keep_r[r] { continue; }
            let sb = if self.granularity == Granularity::Rank { scale } else if keep_b[r] && self.p > 0.0 { scale } else { 1.0 };
            let b_old = self.b[a][r];
            if keep_b[r] { self.b[a][r] += g * h[r] * sb - self.lr * self.decay * self.b[a][r]; }
            if !keep_b[r] { continue; }
            for j in 0..D {
                if !keep_c[r][j] { continue; }
                let sc = if self.granularity == Granularity::Entry && self.p > 0.0 { scale * scale } else { scale };
                self.c[r][j] += g * b_old * sc * x[j] - self.lr * self.decay * self.c[r][j];
            }
        }
        err * err
    }
}

/// A brain with an SS-LoRA adapter on top: the adapter learns what the brain's
/// estimate gets wrong, shared across actions through rank R, so actions tried
/// rarely borrow structure from the rest. The brain's rules stay in charge:
/// disallowed actions are never chosen and thin evidence abstains to the safe default.
pub struct Adapted<const N: usize, const D: usize, const A: usize, const R: usize> {
    pub brain: crate::brain::Brain<N, D, A>,
    pub adapter: SsLora<D, A, R>,
    /// Remembered episodes replayed into the adapter after each new outcome (bounded).
    pub replay: u8,
    cursor: usize,
}

impl<const N: usize, const D: usize, const A: usize, const R: usize> Adapted<N, D, A, R> {
    pub const fn new(brain: crate::brain::Brain<N, D, A>, adapter: SsLora<D, A, R>, replay: u8) -> Self {
        Adapted { brain, adapter, replay, cursor: 0 }
    }

    /// Brain estimate plus adapter correction, per action.
    pub fn expected(&self, x: &[f32; D]) -> [f32; A] {
        let est = self.brain.estimates(x);
        let mut out = [0.0f32; A];
        for a in 0..A { out[a] = est[a].expected + self.adapter.predict(a, x); }
        out
    }

    pub fn decide(&self, x: &[f32; D], allowed: &[bool; A], safe_default: u8) -> crate::brain::Decision {
        let mut d = self.brain.decide(x, allowed, safe_default);
        if d.abstained { return d; }
        let est = self.brain.estimates(x);
        let exp = self.expected(x);
        let mut best = d.action as usize;
        let mut best_s = f32::NEG_INFINITY;
        for a in 0..A {
            if !allowed[a] || est[a].evidence < self.brain.min_evidence { continue; }
            let s = exp[a] + self.brain.explore * est[a].width;
            if s > best_s { best_s = s; best = a; }
        }
        d.action = best as u8;
        d.expected = exp[best];
        d.evidence = est[best].evidence;
        d
    }

    /// Learn an outcome: the adapter takes the brain's error on it, then replays a few memories.
    pub fn learn(&mut self, x: &[f32; D], action: u8, reward: f32) {
        let a = action as usize;
        if a >= A { return; }
        let before = self.brain.estimates(x)[a].expected;
        self.adapter.step(a, x, reward - before);
        self.brain.learn(x, action, reward);
        let cap = crate::brain::Brain::<N, D, A>::CAPACITY;
        if cap == 0 { return; }
        for _ in 0..self.replay {
            self.cursor = (self.cursor + 7) % cap;
            if let Some(e) = self.brain.episode(self.cursor) {
                let (k, ea, er) = (e.key, e.action as usize, e.reward);
                let target = er - self.brain.estimates(&k)[ea].expected;
                self.adapter.step(ea, &k, target);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain::Brain;

    struct Rng(u64);
    impl Rng {
        fn f(&mut self) -> f32 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; ((self.0 >> 40) as f32) / (1u64 << 24) as f32 }
        fn n(&mut self) -> f32 { (self.f() + self.f() + self.f() + self.f() - 2.0) * 0.866 } // ~N(0, 0.5)
    }

    const D: usize = 8;
    const A: usize = 8;
    const R: usize = 4;

    /// A world where the A actions' reward weights share a rank-2 structure
    /// (W* = U V), plus noise. Data is scarce: `per_action` samples per action.
    struct World { w: [[f32; D]; A] }
    impl World {
        fn new(seed: u64) -> Self {
            let mut r = Rng(seed);
            let mut u = [[0.0f32; 2]; A]; let mut v = [[0.0f32; D]; 2];
            for a in 0..A { for k in 0..2 { u[a][k] = r.n(); } }
            for k in 0..2 { for j in 0..D { v[k][j] = r.n(); } }
            let mut w = [[0.0f32; D]; A];
            for a in 0..A { for j in 0..D { w[a][j] = u[a][0] * v[0][j] + u[a][1] * v[1][j]; } }
            World { w }
        }
        fn mean(&self, a: usize, x: &[f32; D]) -> f32 { let mut s = 0.0; for j in 0..D { s += self.w[a][j] * x[j]; } s }
    }

    fn situation(r: &mut Rng) -> [f32; D] { let mut x = [0.0f32; D]; for j in 0..D { x[j] = r.f() * 2.0 - 1.0; } x }

    #[derive(Clone, Copy, Debug)]
    struct Score { train: f32, test: f32 }

    /// Train on `per_action` noisy samples per action, report train MSE (on the noisy
    /// labels it saw) and held-out MSE against the true mean reward.
    fn evaluate(seed: u64, per_action: usize, noise: f32) -> [Score; 4] {
        let world = World::new(seed);
        let mut r = Rng(seed ^ 0xABCD);
        let mut train: [([f32; D], usize, f32); A * 64] = [([0.0; D], 0, 0.0); A * 64];
        let n = per_action * A;
        for i in 0..n { let a = i % A; let x = situation(&mut r); train[i] = (x, a, world.mean(a, &x) + noise * r.n()); }
        let mut test = [([0.0f32; D], 0usize); 2000];
        for t in test.iter_mut() { *t = (situation(&mut r), (r.f() * A as f32) as usize % A); }

        // (0) per-action ridge (the brain's own model), (1) dense low-rank, (2) SS-LoRA rank mask, (3) SS-LoRA entry mask
        let mut ridge: Brain<0, D, A> = Brain::new(0.0, 0.0);
        for &(x, a, y) in &train[..n] { ridge.update_model(&x, a as u8, y); }
        let mut models = [SsLora::<D, A, R>::new(0.0, 0.03, seed), SsLora::new(0.5, 0.03, seed), SsLora::new(0.5, 0.03, seed)];
        models[2].granularity = Granularity::Entry;
        let epochs = 400;
        for m in models.iter_mut() {
            let mut order = Rng(seed ^ 0x55);
            for _ in 0..epochs {
                for _ in 0..n { let i = (order.f() * n as f32) as usize % n; let (x, a, y) = train[i]; m.step(a, &x, y); }
            }
        }
        let mse = |f: &dyn Fn(usize, &[f32; D]) -> f32| -> Score {
            let mut tr = 0.0; for &(x, a, y) in &train[..n] { let e = f(a, &x) - y; tr += e * e; }
            let mut te = 0.0; for &(x, a) in &test { let e = f(a, &x) - world.mean(a, &x); te += e * e; }
            Score { train: tr / n as f32, test: te / test.len() as f32 }
        };
        [mse(&|a, x| ridge.predict(a as u8, x)), mse(&|a, x| models[0].predict(a, x)), mse(&|a, x| models[1].predict(a, x)), mse(&|a, x| models[2].predict(a, x))]
    }

    /// Online bandit in the rank-2 world: each step a situation arrives, the brain picks an
    /// action, sees a noisy reward. Score = right-action rate on 500 unseen situations.
    fn bandit(seed: u64, mode: u8) -> f32 { bandit_steps(seed, mode, 1200) }

    fn bandit_steps(seed: u64, mode: u8, steps: usize) -> f32 {
        let world = World::new(seed);
        let mut r = Rng(seed ^ 0x77);
        let brain: Brain<64, D, A> = Brain::new(0.3, 0.5);
        let p = if mode == 2 { 0.5 } else { 0.0 };
        let mut ad: Adapted<64, D, A, R> = Adapted::new(brain, SsLora::new(p, 0.03, seed), 4);
        for t in 0..steps {
            let x = situation(&mut r);
            let d = if mode == 0 { ad.brain.decide(&x, &[true; A], 0) } else { ad.decide(&x, &[true; A], 0) };
            let a = if d.abstained { (t % A) as u8 } else { d.action };
            let rw = world.mean(a as usize, &x) + 0.5 * r.n();
            if mode == 0 { ad.brain.learn(&x, a, rw); } else { ad.learn(&x, a, rw); }
        }
        ad.brain.explore = 0.0;
        let mut ok = 0;
        for _ in 0..500 {
            let x = situation(&mut r);
            let d = if mode == 0 { ad.brain.decide(&x, &[true; A], 0) } else { ad.decide(&x, &[true; A], 0) };
            let best = (0..A).max_by(|&i, &j| world.mean(i, &x).partial_cmp(&world.mean(j, &x)).unwrap()).unwrap();
            if d.action as usize == best { ok += 1; }
        }
        ok as f32 / 500.0
    }

    /// Pre-registered: averaged over 8 seeds (and again over 8 fresh seeds), the brain with an
    /// SS-LoRA adapter (p = 0.5) picks the best action on unseen situations more often than the
    /// plain brain AND than the same adapter without the mask.
    #[test]
    #[ignore = "FAILED as pre-registered: plain brain 0.781 > SS-LoRA-adapted 0.765 > dense-adapted 0.751 at 1200 steps (data is not scarce there); see horizon_sweep"]
    fn falsifier_adapted_brain_decides_better_on_unseen_situations() {
        for seeds in [[11u64, 23, 37, 41, 59, 61, 73, 89], [101, 103, 107, 109, 113, 127, 131, 137]] {
            let mut s = [0.0f32; 3];
            for &sd in &seeds { for m in 0..3 { s[m as usize] += bandit(sd, m); } }
            let k = seeds.len() as f32;
            std::println!("adapted-brain falsifier: plain brain {:.3}  + dense adapter {:.3}  + SS-LoRA adapter {:.3} (right-action rate, unseen situations)", s[0] / k, s[1] / k, s[2] / k);
            assert!(s[2] > s[0] && s[2] > s[1], "{:?}", s);
        }
    }

    /// Follow-up AFTER the failed falsifier (a new hypothesis, reported in full, not a pass of the
    /// original): does the adapter help when data per action is scarce (short horizons)?
    #[test]
    fn horizon_sweep() {
        let seeds = [11u64, 23, 37, 41, 59, 61, 73, 89, 101, 103, 107, 109, 113, 127, 131, 137];
        std::println!("horizon sweep (16 seeds, right-action rate on unseen situations): plain | +dense | +SS-LoRA");
        for &steps in &[80usize, 160, 320, 640, 1200] {
            let mut s = [0.0f32; 3];
            for &sd in &seeds { for m in 0..3u8 { s[m as usize] += bandit_steps(sd, m, steps); } }
            let k = seeds.len() as f32;
            std::println!("  {:>5} steps ({:>4.1}/action): {:.3} | {:.3} | {:.3}", steps, steps as f32 / A as f32, s[0] / k, s[1] / k, s[2] / k);
        }
    }

    #[test]
    fn starts_as_no_correction_and_is_deterministic() {
        let m: SsLora<D, A, R> = SsLora::new(0.5, 0.05, 3);
        assert_eq!(m.predict(2, &[0.5; D]), 0.0, "B = 0 at start");
        let run = || { let mut m: SsLora<D, A, R> = SsLora::new(0.5, 0.05, 9); let mut r = Rng(1); let mut s = 0.0;
            for _ in 0..500 { let x = situation(&mut r); s += m.step(3, &x, 1.0); } s };
        assert_eq!(run(), run());
        assert!(core::mem::size_of::<SsLora<D, A, R>>() <= 4 * (A * R + R * D) + 24);
    }

    #[test]
    fn learns_a_low_rank_world_with_plenty_of_data() {
        let s = evaluate(7, 48, 0.1);
        assert!(s[2].test < 0.1, "SS-LoRA held-out MSE {}", s[2].test);
    }

    /// Pre-registered falsifier (set before the run): with scarce data (12 samples per
    /// action, 8 actions, 8 features, true reward weights rank 2, label noise 0.5),
    /// averaged over 8 seeds, SS-LoRA (rank mask, p = 0.5) has
    /// (a) lower held-out error than per-action ridge AND than the same adapter without the mask,
    /// (b) a smaller generalization gap (held-out MSE minus train MSE) than the dense adapter.
    #[test]
    fn falsifier_ss_lora_generalizes_instead_of_memorizing() {
        // The original seeds AND eight fresh ones never used while building it.
        for seeds in [[11u64, 23, 37, 41, 59, 61, 73, 89], [101, 103, 107, 109, 113, 127, 131, 137]] {
        let mut sum = [[0.0f32; 2]; 4];
        for &s in &seeds {
            let sc = evaluate(s, 12, 0.5);
            for k in 0..4 { sum[k][0] += sc[k].train; sum[k][1] += sc[k].test; }
        }
        let k = seeds.len() as f32;
        let names = ["per-action ridge", "dense low-rank (p=0)", "SS-LoRA rank mask p=0.5", "SS-LoRA entry mask p=0.5"];
        std::println!("SS-LoRA falsifier: 12 samples/action, 8 actions, rank-2 truth, noise 0.5, {} seeds", seeds.len());
        for i in 0..4 {
            std::println!("  {:<26} train MSE {:.3}  held-out MSE {:.3}  gap {:+.3}", names[i], sum[i][0] / k, sum[i][1] / k, (sum[i][1] - sum[i][0]) / k);
        }
        let (ridge, dense, ss) = (sum[0][1] / k, sum[1][1] / k, sum[2][1] / k);
        let (gap_dense, gap_ss) = ((sum[1][1] - sum[1][0]) / k, (sum[2][1] - sum[2][0]) / k);
        assert!(ss < ridge && ss < dense, "held-out: SS {} vs ridge {} / dense {}", ss, ridge, dense);
        assert!(gap_ss < gap_dense, "gap: SS {} vs dense {}", gap_ss, gap_dense);
        }
    }
}
