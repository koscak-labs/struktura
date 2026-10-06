//! Safety shield for irreversible actions, around a [`crate::brain::Brain`].
//!
//! Some actions cannot be undone: leaving safe-mode, driving onto a slope,
//! firing a pyro, committing a lander power decision. Exploring them to learn
//! their value is how a vehicle is lost. The shield changes how the brain
//! ranks actions, without changing what it learns:
//!
//! - **Reversible** actions are ranked optimistically: expected reward plus an
//!   exploration bonus on the model's uncertainty width.
//! - **Irreversible** actions are never explored. They are ranked by a lower
//!   confidence bound, `expected - pessimism * noise * width` (noise = RMS of
//!   this action's past non-catastrophic surprises, width = the model's
//!   uncertainty), capped by the worst reward the action produced within
//!   `radius` of the situation, and need `min_evidence` behind them.
//! - **Veto (memory)**: an action that ever produced a reward below
//!   `catastrophe` within `radius` of the situation is not allowed there. The
//!   scan covers the whole memory (one bounded pass), so a remembered
//!   catastrophe cannot be crowded out of the K-nearest list; the brain keeps
//!   surprising episodes longest, so catastrophes are forgotten last.
//! - **Veto (certification)**: with `risk_budget = eps > 0`, an irreversible
//!   action is allowed in a region only after `ceil(3 / eps)` trials there with
//!   no catastrophe (rule of three: then the catastrophe rate is below eps with
//!   95% confidence), and never in a region where it ever caused one. Regions
//!   are cells of a fixed grid over the situation (each feature, scaled to
//!   [0, 1], cut into `bins` bins; cells beyond 64 hash together). Counts are
//!   exact, independent of the lossy episode memory, and must be fed by
//!   [`Shield::observe`] - including the ground test campaign, because an
//!   irreversible action is never tried live until it is certified.
//!
//! What this cannot do: a hazard never seen anywhere cannot be vetoed before it
//! first happens. Without certification the guarantee is "at most one
//! catastrophe per neighbourhood"; with it, "catastrophe rate below eps (95%)
//! wherever the action is used", which needs a ground campaign large enough to
//! certify the regions where the action should be allowed.
//!
//! No heap. `size_of::<Shield<A>>()` is about `A * 256 + 32` bytes (1 KB at A = 4).

use crate::brain::{Brain, K};

/// Certification grid cells.
pub const CELLS: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shield<const A: usize> {
    /// `irreversible[a]`: action `a` cannot be undone.
    pub irreversible: [bool; A],
    /// Multiplier of (noise x width) in the lower bound for irreversible actions.
    pub pessimism: f32,
    /// Exploration bonus multiplier for reversible actions.
    pub explore: f32,
    /// A reward below this is a catastrophe.
    pub catastrophe: f32,
    /// Squared situation distance within which past outcomes of an action count as "here".
    pub radius2: f32,
    /// Evidence an irreversible action needs before it can be chosen.
    pub min_evidence: f32,
    /// Acceptable catastrophe rate for irreversible actions (0 = no certification required).
    pub risk_budget: f32,
    /// Bins per feature for the certification grid.
    pub bins: u8,
    trials: [[u16; CELLS]; A],
    hits: [[u16; CELLS]; A],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShieldedDecision {
    pub action: u8,
    /// True when no allowed action survived and the safe default was returned.
    pub abstained: bool,
    /// Ranking score of the chosen action (upper bound if reversible, lower bound if irreversible).
    pub score: f32,
    /// Bit `a` set: action `a` was vetoed (catastrophe nearby or in its cell, too little evidence, or not certified).
    pub vetoed_mask: u32,
    /// Neighbours the brain consulted (nearest first; first `n_cited` valid).
    pub cited: [u16; K],
    pub n_cited: u8,
}

impl<const A: usize> Shield<A> {
    pub const fn new(irreversible: [bool; A]) -> Self {
        Shield { irreversible, pessimism: 2.0, explore: 0.3, catastrophe: -1.0, radius2: 0.05, min_evidence: 5.0,
            risk_budget: 0.0, bins: 4, trials: [[0; CELLS]; A], hits: [[0; CELLS]; A] }
    }

    /// Certification cell of a situation (features expected in [0, 1]; clamped).
    pub fn cell<const D: usize>(&self, x: &[f32; D]) -> usize {
        let b = self.bins.max(1) as usize;
        let mut idx = 0usize;
        let mut mul = 1usize;
        for v in x.iter() {
            let c = if *v <= 0.0 { 0 } else if *v >= 1.0 { b - 1 } else { ((*v * b as f32) as usize).min(b - 1) };
            idx = idx.wrapping_add(c.wrapping_mul(mul));
            mul = mul.wrapping_mul(b);
        }
        idx % CELLS
    }

    /// Record an outcome (ground campaign or live): exact per-cell trial and catastrophe counts.
    pub fn observe<const D: usize>(&mut self, x: &[f32; D], action: u8, reward: f32) {
        let a = action as usize;
        if a >= A { return; }
        let c = self.cell(x);
        self.trials[a][c] = self.trials[a][c].saturating_add(1);
        if reward < self.catastrophe { self.hits[a][c] = self.hits[a][c].saturating_add(1); }
    }

    /// (trials, catastrophes) of action `a` in the cell of `x`.
    pub fn record<const D: usize>(&self, x: &[f32; D], a: usize) -> (u16, u16) {
        if a >= A { return (0, 0); }
        let c = self.cell(x);
        (self.trials[a][c], self.hits[a][c])
    }

    /// Clean trials needed to certify `risk_budget` (rule of three), 0 when certification is off.
    pub fn trials_needed(&self) -> u32 {
        if self.risk_budget > 0.0 {
            let n = 3.0 / self.risk_budget;
            let f = n as u32;
            if (f as f32) < n { f + 1 } else { f }
        } else { 0 }
    }

    /// Noise scale of action `a`'s outcomes: RMS of the surprise (|reward - what the brain
    /// expected when it stored the episode) over its remembered non-catastrophic episodes, floored.
    pub fn noise<const N: usize, const D: usize>(&self, brain: &Brain<N, D, A>, a: usize) -> f32 {
        let (mut s2, mut n) = (0.0f32, 0u32);
        for s in 0..N {
            if let Some(e) = brain.episode(s) {
                // Catastrophes are the veto's job; the noise scale is ordinary variation.
                if e.action as usize == a && e.reward >= self.catastrophe { s2 += e.surprise * e.surprise; n += 1; }
            }
        }
        if n == 0 { return 1.0; }
        let r = crate::sqrt((s2 / n as f32) as f64) as f32;
        if r < 0.01 { 0.01 } else { r }
    }

    /// Worst reward of action `a` within `radius` of `x` over the whole memory, if any.
    pub fn worst_nearby<const N: usize, const D: usize>(&self, brain: &Brain<N, D, A>, x: &[f32; D], a: usize) -> Option<f32> {
        let mut worst: Option<f32> = None;
        for s in 0..N {
            let Some(e) = brain.episode(s) else { continue };
            if e.action as usize != a { continue; }
            let mut d2 = 0.0;
            for i in 0..D { let d = e.key[i] - x[i]; d2 += d * d; }
            if d2 <= self.radius2 && worst.map(|w| e.reward < w).unwrap_or(true) { worst = Some(e.reward); }
        }
        worst
    }

    /// Choose an action for `x`. `allowed[a] == false` is never chosen.
    pub fn decide<const N: usize, const D: usize>(&self, brain: &Brain<N, D, A>, x: &[f32; D],
                                                  allowed: &[bool; A], safe_default: u8) -> ShieldedDecision {
        let est = brain.estimates(x);
        let nb = brain.neighbours(x);
        let mut out = ShieldedDecision { action: safe_default, abstained: true, score: 0.0, vetoed_mask: 0,
            cited: nb.0, n_cited: nb.2 as u8 };
        let need = self.trials_needed();
        let mut best: Option<(usize, f32)> = None;
        for a in 0..A {
            if !allowed[a] { continue; }
            let e = est[a];
            let score = if self.irreversible[a] {
                let worst = self.worst_nearby(brain, x, a);
                let (trials, hits) = self.record(x, a);
                let veto = worst.map(|w| w < self.catastrophe).unwrap_or(false)
                    || e.evidence < self.min_evidence
                    || hits > 0
                    || (need > 0 && (trials as u32) < need);
                if veto { if a < 32 { out.vetoed_mask |= 1 << a; } continue; }
                // Lower confidence bound: mean minus pessimism x (noise scale x model width).
                let lower = e.expected - self.pessimism * self.noise(brain, a) * e.width;
                match worst { Some(w) if w < lower => w, _ => lower }
            } else {
                e.expected + self.explore * e.width
            };
            if best.map(|b| score > b.1).unwrap_or(true) { best = Some((a, score)); }
        }
        if let Some((a, s)) = best { out.action = a as u8; out.abstained = false; out.score = s; }
        out
    }
}

/// The falsifiers' rover traverse, shared by the tests and the brain gym (`crate::brain_gym`).
#[cfg(feature = "std")]
pub mod falsify {
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn f(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32) / (1u64 << 24) as f32
        }
    }

    // Rover traverse. Situation [slope, soil softness, 1]. Actions:
    // 0 wait (0.2), 1 drive slow (0.8), 2 drive fast: IRREVERSIBLE (momentum on a slope),
    // 1.2 almost everywhere, but on a steep soft patch (slope > 0.8 and soft > 0.8,
    // 4% of the terrain) it has a 3% chance of -10 (the rover slides and is lost). Its
    // mean there is still 0.864 > 0.8 for driving slow, so a mean-maximising brain keeps
    // driving fast on the patch: rare-catastrophe risk that averages hide;
    // 3 safe-mode (0.05).
    pub const IRREV: [bool; 4] = [false, false, true, false];
    fn danger(x: &[f32; 3]) -> bool { x[0] > 0.8 && x[1] > 0.8 }
    fn outcome(x: &[f32; 3], a: u8, r: &mut Lcg) -> f32 {
        let noise = (r.f() - 0.5) * 0.1;
        match a {
            0 => 0.2 + noise,
            1 => 0.8 + noise,
            2 => if danger(x) && r.f() < 0.03 { -10.0 } else { 1.2 + noise },
            _ => 0.05 + noise,
        }
    }

    pub struct Run { pub reward: f32, pub catastrophes: u32, pub risky_picks: u32, pub picks: [u32; 4] }

    /// Ground test campaign (`ground` situations; a quarter on the patch the team worried
    /// about; every action tried on and off it), then `n` live decisions. Both brains see the
    /// same campaign and the same terrain.
    pub fn traverse(shielded: bool, risk_budget: f32, ground: usize, n: usize, world_seed: u64, terrain_seed: u64) -> Run {
        let mut br: Brain<256, 3, 4> = Brain::new(0.3, 0.5);
        let mut sh: Shield<4> = Shield::new(IRREV);
        sh.catastrophe = -2.0;
        sh.radius2 = 0.09; // within 0.3 in the (slope, softness) plane
        sh.risk_budget = risk_budget;
        let mut world = Lcg(world_seed);
        let mut terrain = Lcg(terrain_seed);
        for i in 0..ground {
            let x = if i % 4 == 0 { [0.8 + 0.2 * terrain.f(), 0.8 + 0.2 * terrain.f(), 1.0] } else { [terrain.f(), terrain.f(), 1.0] };
            let a = ((i / 4) % 4) as u8;
            let r = outcome(&x, a, &mut world);
            br.learn(&x, a, r);
            sh.observe(&x, a, r);
        }
        let mut run = Run { reward: 0.0, catastrophes: 0, risky_picks: 0, picks: [0; 4] };
        for _ in 0..n {
            let x = [terrain.f(), terrain.f(), 1.0];
            let a = if shielded { sh.decide(&br, &x, &[true; 4], 3).action } else { br.decide(&x, &[true; 4], 3).action };
            let r = outcome(&x, a, &mut world);
            run.picks[a as usize] += 1;
            if a == 2 && danger(&x) { run.risky_picks += 1; }
            if r < -5.0 { run.catastrophes += 1; }
            run.reward += r;
            br.learn(&x, a, r);
            sh.observe(&x, a, r);
        }
        run
    }

    /// The tests' eight (world, terrain) seed pairs.
    pub fn test_seeds() -> [(u64, u64); 8] { core::array::from_fn(|k| (100 + k as u64 * 7, 900 + k as u64 * 13)) }

    /// Unshielded vs shielded on identical worlds, one (world, terrain) seed pair each.
    /// Returns (unshielded catastrophes, shielded, max shielded per seed, regret %);
    /// `label` = Some(..) prints the per-seed table.
    pub fn campaign(risk_budget: f32, ground: usize, seeds: &[(u64, u64)], label: Option<&str>) -> (u32, u32, u32, f32) {
        let (mut pc, mut sc, mut smax, mut pr, mut sr) = (0u32, 0u32, 0u32, 0.0f32, 0.0f32);
        if let Some(label) = label { std::println!("{} (ground campaign {} trials, risk budget {}):", label, ground, risk_budget); }
        for (k, &(ws, ts)) in seeds.iter().enumerate() {
            let p = traverse(false, risk_budget, ground, 3000, ws, ts);
            let s = traverse(true, risk_budget, ground, 3000, ws, ts);
            if label.is_some() {
                std::println!("  seed {}: unshielded catastrophes {:>2} risky {:>3} reward {:>7.1} | shielded catastrophes {:>2} risky {:>3} reward {:>7.1}",
                    k, p.catastrophes, p.risky_picks, p.reward, s.catastrophes, s.risky_picks, s.reward);
            }
            pc += p.catastrophes; sc += s.catastrophes; smax = smax.max(s.catastrophes); pr += p.reward; sr += s.reward;
        }
        let regret = 100.0 * (pr - sr) / pr;
        if label.is_some() { std::println!("  total: unshielded {} catastrophes, shielded {} (max {} per seed); shield regret {:+.2}%", pc, sc, smax, regret); }
        (pc, sc, smax, regret)
    }

    /// The certified falsifier's verdict on a campaign result (the test's asserts).
    pub fn certified_holds(pc: u32, sc: u32, regret: f32) -> bool { pc > 0 && sc == 0 && regret <= 10.0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::falsify::{certified_holds, test_seeds, traverse, IRREV};

    /// Eight seeds, unshielded vs shielded on identical worlds.
    /// Returns (unshielded catastrophes, shielded, max shielded per seed, regret %).
    fn campaign(risk_budget: f32, ground: usize, label: &str) -> (u32, u32, u32, f32) {
        falsify::campaign(risk_budget, ground, &test_seeds(), Some(label))
    }

    #[test]
    fn falsifier_memory_veto_allows_at_most_one_per_neighbourhood() {
        // No certification: a hazard nobody has seen yet cannot be vetoed, so ZERO is NOT
        // claimed here. What holds: never a second catastrophe in the same neighbourhood.
        let (pc, sc, smax, regret) = campaign(0.0, 1200, "memory veto only");
        assert!(pc > sc, "the shield must lose fewer rovers than the unshielded brain");
        assert!(smax <= 1, "never a second catastrophe in the same neighbourhood");
        assert!(regret <= 10.0);
    }

    #[test]
    fn falsifier_certified_zero_catastrophes_with_bounded_regret() {
        // Certification at eps = 3%: fast driving is allowed in a cell only after 100 clean
        // trials there. The campaign is sized by the rule of three: 100 clean fast trials in each
        // of the 16 (slope, softness) cells needs ~12000 ground steps (3/4 random x 1/4 fast);
        // an 8000-step campaign leaves ordinary cells uncertified (measured: 0 catastrophes but
        // 24.9% regret, the shield correctly refuses to drive fast where it cannot certify).
        let (pc, sc, _, regret) = campaign(0.03, 12000, "certified (rule of three, eps 3%)");
        assert!(pc > 0, "the falsifier needs the unshielded brain to lose rovers");
        assert_eq!(sc, 0, "zero catastrophic outcomes over 8 x 3000 live decisions");
        assert!(regret <= 10.0, "regret {:.2}% over the 10% budget", regret);
        assert!(certified_holds(pc, sc, regret), "the gym verdict agrees with the asserts");
    }

    #[test]
    fn irreversible_is_never_explored_without_evidence() {
        let br: Brain<16, 3, 4> = Brain::new(1.0, 0.0);
        let sh: Shield<4> = Shield::new(IRREV);
        let d = sh.decide(&br, &[0.5, 0.5, 1.0], &[true; 4], 3);
        assert_ne!(d.action, 2, "no evidence: the irreversible action is not even tried");
        assert!(d.vetoed_mask & (1 << 2) != 0);
    }

    #[test]
    fn uncertified_cell_is_vetoed_until_enough_clean_trials() {
        let mut br: Brain<64, 3, 4> = Brain::new(0.0, 0.0);
        let mut sh: Shield<4> = Shield::new(IRREV);
        sh.risk_budget = 0.1; // 30 clean trials
        assert_eq!(sh.trials_needed(), 30);
        let x = [0.1, 0.1, 1.0];
        for _ in 0..29 { br.learn(&x, 2, 1.2); br.learn(&x, 1, 0.8); sh.observe(&x, 2, 1.2); }
        assert_ne!(sh.decide(&br, &x, &[true; 4], 3).action, 2, "29 trials do not certify 10%");
        sh.observe(&x, 2, 1.2);
        assert_eq!(sh.decide(&br, &x, &[true; 4], 3).action, 2, "30 clean trials do");
        sh.observe(&x, 2, -10.0);
        assert_ne!(sh.decide(&br, &x, &[true; 4], 3).action, 2, "one catastrophe in the cell revokes it for good");
    }

    #[test]
    fn remembered_catastrophe_vetoes_its_neighbourhood_only() {
        let mut br: Brain<64, 3, 4> = Brain::new(0.0, 0.0);
        for i in 0..40 {
            let x = [i as f32 / 40.0, 0.1, 1.0];
            br.learn(&x, 2, 1.2);
            br.learn(&x, 1, 0.8);
        }
        br.learn(&[0.9, 0.9, 1.0], 2, -10.0);
        let mut sh: Shield<4> = Shield::new(IRREV);
        sh.catastrophe = -2.0;
        sh.radius2 = 0.02;
        let near = sh.decide(&br, &[0.92, 0.88, 1.0], &[true; 4], 3);
        assert_ne!(near.action, 2);
        assert!(near.vetoed_mask & (1 << 2) != 0);
        let far = sh.decide(&br, &[0.2, 0.1, 1.0], &[true; 4], 3);
        assert_eq!(far.action, 2, "far from the catastrophe the fast drive is still used");
    }

    #[test]
    fn nothing_allowed_returns_safe_default() {
        let br: Brain<8, 3, 4> = Brain::new(0.0, 0.0);
        let sh: Shield<4> = Shield::new(IRREV);
        let d = sh.decide(&br, &[0.0, 0.0, 1.0], &[false; 4], 3);
        assert!(d.abstained);
        assert_eq!(d.action, 3);
    }

    #[test]
    fn deterministic_and_small() {
        let a = traverse(true, 0.03, 2000, 500, 1, 2);
        let b = traverse(true, 0.03, 2000, 500, 1, 2);
        assert_eq!((a.reward, a.picks), (b.reward, b.picks));
        let sz = core::mem::size_of::<Shield<4>>();
        std::println!("size_of::<Shield<4>>() = {} bytes", sz);
        assert!(sz <= 1100, "{} bytes", sz);
        static _S: Shield<4> = Shield::new(IRREV);
    }
}
