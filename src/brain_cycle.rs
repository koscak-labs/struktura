//! Growth cycle: grow -> compress -> grokking event -> grow bigger.
//!
//! A developmental controller over a [`Brain`] and its [`Grower`]. Capacity comes
//! in tiers (1, 2, 4, ... growth slots). The brain only gets a bigger tier after it
//! has compressed what it has, and, beyond the first tier, only after the previous
//! tier produced a grokking event.
//!
//! - **Grow**: the grower may adopt senses into the slots of the current tier
//!   (held-out-gated, see [`crate::brain_grow`]). The phase ends after
//!   `grow_patience` growth checks without an adoption, or after `grow_max` steps.
//! - **Compress** (`compress_steps` long, acting every `compress_every` steps):
//!   forget memories the model alone already predicts within `absorb_eps` (their
//!   knowledge is in the model) and prune grown senses whose contribution vanished: the sense is (near) constant
//!   over everything still in memory, so it separates nothing. (An error-based
//!   criterion, the model-only fit loss on memory when the sense is zeroed, was
//!   tried and cost 2-3 points on dev seeds: a sense also pays through recall.)
//! - **Grokking event**: a sudden, sustained DROP in the prequential error (each
//!   outcome predicted before it is learned: an honest online held-out signal),
//!   detected by a Page-Hinkley decrease test that is armed only after a compress
//!   phase. At the end of a compress phase the next tier unlocks if this is the
//!   first tier or the current tier produced a grokking event; otherwise the brain
//!   stays at its size (capacity that does not pay off is not grown).
//!
//! no_std, no heap, bounded work per step, deterministic.

use crate::brain::Brain;
use crate::brain_grow::{Feat, Grower};

#[inline]
fn absf(x: f32) -> f32 { if x < 0.0 { -x } else { x } }

/// Page-Hinkley test for a sustained DECREASE in the mean of a stream.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DropDetector {
    n: u32,
    mean: f32,
    cum: f32,
    max: f32,
    /// Allowed drift per step.
    pub delta: f32,
    /// Alarm threshold on the accumulated shortfall below the running mean.
    pub lambda: f32,
    /// Samples after a reset before an alarm may fire.
    pub warmup: u32,
}

impl DropDetector {
    pub const fn new(delta: f32, lambda: f32, warmup: u32) -> Self {
        DropDetector { n: 0, mean: 0.0, cum: 0.0, max: 0.0, delta, lambda, warmup }
    }
    pub fn reset(&mut self) { self.n = 0; self.mean = 0.0; self.cum = 0.0; self.max = 0.0; }
    /// Feed one value; true when a decrease in its mean is declared.
    pub fn update(&mut self, x: f32) -> bool {
        self.n = self.n.saturating_add(1);
        self.mean += (x - self.mean) / self.n as f32;
        self.cum += x - self.mean + self.delta;
        if self.cum > self.max { self.max = self.cum; }
        self.n > self.warmup && self.max - self.cum > self.lambda
    }
    /// Running mean since the last reset.
    pub fn mean(&self) -> f32 { self.mean }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Phase { Grow, Compress }

/// A recorded grokking event.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GrokEvent {
    /// Learned outcomes when it fired.
    pub t: u32,
    /// Mean prequential error since the detector was armed (before the drop).
    pub before: f32,
    /// Recent prequential error (fast average) when it fired.
    pub after: f32,
    pub tier: u8,
    /// Active parameters when it fired (see [`Cycle::active_params`]).
    pub active: u32,
}

/// Maximum grokking events kept.
pub const MAX_EVENTS: usize = 16;

pub struct Cycle<const B: usize, const G: usize> {
    pub grower: Grower<B, G>,
    phase: Phase,
    tier: u8,
    pub detector: DropDetector,
    armed: bool,
    grok_in_tier: bool,
    compressed_once: bool,
    steps_in_phase: u32,
    checks_without_growth: u8,
    /// Growth checks without an adoption that end a Grow phase.
    pub grow_patience: u8,
    /// Longest Grow phase (steps), e.g. when the tier is full.
    pub grow_max: u32,
    pub compress_steps: u32,
    pub compress_every: u32,
    /// A memory is absorbed when the model alone predicts its reward within this.
    pub absorb_eps: f32,
    /// A grown sense is pruned when its value spans less than this over the memories in use.
    pub prune_range: f32,
    fast: f32,
    t: u32,
    events: [GrokEvent; MAX_EVENTS],
    n_events: usize,
    /// Memories forgotten / senses pruned by compression so far.
    pub forgotten: u32,
    pub pruned: u32,
}

impl<const B: usize, const G: usize> Cycle<B, G> {
    pub fn new(seed: u64) -> Self {
        let mut grower = Grower::new(seed);
        grower.set_limit(1);
        Cycle {
            grower, phase: Phase::Grow, tier: 0,
            detector: DropDetector::new(0.002, 2.0, 150),
            armed: false, grok_in_tier: false, compressed_once: false,
            steps_in_phase: 0, checks_without_growth: 0,
            grow_patience: 2, grow_max: 1500, compress_steps: 300, compress_every: 100,
            absorb_eps: 0.1, prune_range: 1e-3,
            fast: 0.0, t: 0,
            events: [GrokEvent { t: 0, before: 0.0, after: 0.0, tier: 0, active: 0 }; MAX_EVENTS], n_events: 0,
            forgotten: 0, pruned: 0,
        }
    }

    pub fn phase(&self) -> Phase { self.phase }
    pub fn tier(&self) -> u8 { self.tier }
    pub fn events(&self) -> &[GrokEvent] { &self.events[..self.n_events] }
    /// Fast moving average of the prequential squared error.
    pub fn prequential(&self) -> f32 { self.fast }

    /// Base features followed by the grown senses. `D` must be `B + G`.
    pub fn situation<const D: usize>(&self, base: &[f32; B]) -> [f32; D] { self.grower.situation(base) }

    /// Memories in use + active grown senses + model dimensions in use (base + active senses).
    pub fn active_params<const N: usize, const D: usize, const A: usize>(&self, brain: &Brain<N, D, A>) -> u32 {
        (brain.in_use() + self.grower.active() + B + self.grower.active()) as u32
    }

    /// Learn one outcome (situation `x` must come from [`Cycle::situation`]) and advance the cycle.
    /// Returns a grokking event when one fires.
    pub fn learn<const N: usize, const D: usize, const A: usize>(&mut self, brain: &mut Brain<N, D, A>, x: &[f32; D], action: u8, reward: f32) -> Option<GrokEvent> {
        if (action as usize) >= A || B + G != D { return None; }
        // Prequential error: predicted before this outcome is learned.
        let e = reward - brain.estimates(x)[action as usize].expected;
        let err = e * e;
        self.fast += (err - self.fast) * 0.02;
        brain.learn(x, action, reward);
        self.t = self.t.wrapping_add(1);
        self.steps_in_phase += 1;

        let mut fired = None;
        if self.armed && self.detector.update(err) {
            let ev = GrokEvent { t: self.t, before: self.detector.mean(), after: self.fast, tier: self.tier, active: self.active_params(brain) };
            if self.n_events < MAX_EVENTS { self.events[self.n_events] = ev; self.n_events += 1; }
            self.grok_in_tier = true;
            self.armed = false;
            fired = Some(ev);
        }

        match self.phase {
            Phase::Grow => {
                if let Some(g) = self.grower.after_learn(brain) {
                    if g.adopted.is_some() { self.checks_without_growth = 0; } else { self.checks_without_growth = self.checks_without_growth.saturating_add(1); }
                }
                if self.checks_without_growth >= self.grow_patience || self.steps_in_phase >= self.grow_max {
                    self.phase = Phase::Compress;
                    self.steps_in_phase = 0;
                }
            }
            Phase::Compress => {
                if self.steps_in_phase % self.compress_every.max(1) == 0 { self.compress(brain); }
                if self.steps_in_phase >= self.compress_steps {
                    if self.tier == 0 || self.grok_in_tier {
                        self.tier = self.tier.saturating_add(1);
                        let cap = if self.tier >= 16 { G } else { (1usize << self.tier).min(G) };
                        self.grower.set_limit(cap);
                        self.grok_in_tier = false;
                    }
                    self.compressed_once = true;
                    // Arm the grokking detector: from here a sustained drop counts.
                    self.detector.reset();
                    self.armed = true;
                    self.phase = Phase::Grow;
                    self.steps_in_phase = 0;
                    self.checks_without_growth = 0;
                }
            }
        }
        fired
    }

    /// One compression pass: forget absorbed memories, prune senses that stopped contributing.
    pub fn compress<const N: usize, const D: usize, const A: usize>(&mut self, brain: &mut Brain<N, D, A>) {
        for s in 0..N {
            let absorbed = match brain.episode(s) {
                Some(ep) => absf(ep.reward - brain.predict(ep.action, &ep.key)) < self.absorb_eps,
                None => false,
            };
            if absorbed && brain.forget_slot(s) { self.forgotten += 1; }
        }
        let grown = self.grower.grown().len();
        for slot in 0..grown {
            if self.grower.grown()[slot] == Feat::Off { continue; }
            let (mut lo, mut hi, mut n) = (f32::MAX, f32::MIN, 0u32);
            for s in 0..N {
                let Some(ep) = brain.episode(s) else { continue };
                let v = ep.key[B + slot];
                if v < lo { lo = v; }
                if v > hi { hi = v; }
                n += 1;
            }
            if n >= 16 && hi - lo < self.prune_range && self.grower.prune(slot, brain) {
                self.pruned += 1;
            }
        }
    }

    /// Whether at least one compress phase has completed.
    pub fn compressed_once(&self) -> bool { self.compressed_once }
}

/// The falsifier's layered rover world and runs, shared by the tests and the brain gym
/// (`crate::brain_gym`).
#[cfg(feature = "std")]
pub mod falsify {
    use super::*;

    struct Rng(u64);
    impl Rng { fn f(&mut self) -> f32 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; ((self.0 >> 40) as f32) / (1u64 << 24) as f32 } }

    /// Layered rover world: [drift, spike, temp, 1]. Thermal runaway (drift AND temp high)
    /// -> safe-mode 3, EXCEPT when a spike also shows -> quarantine 2 (sensor fault). The
    /// spike refinement is only worth a sense once the brain can see the thermal region.
    fn truth(x: &[f32; 4]) -> u8 {
        if x[0] > 0.6 && x[2] > 0.6 { if x[1] > 0.7 { 2 } else { 3 } } else if x[0] > 0.5 { 1 } else { 0 }
    }
    pub fn reward(x: &[f32; 4], a: u8) -> f32 { if a == truth(x) { 1.0 } else if a == 3 { 0.2 } else { 0.0 } }

    pub const STEPS: usize = 6000;

    /// No knob overrides (the falsifier's and the gym's setting).
    pub fn defaults(_: &mut Cycle<4, 4>) {}

    /// (right-action rate last 1000, mean active params last 1000, grokking events, senses active at end).
    /// `knobs` may override the cycle's settings (dev tuning); `log` prints the grokking events.
    pub fn run(seed: u64, cycle: bool, knobs: fn(&mut Cycle<4, 4>), log: bool) -> (f32, f32, usize, usize) {
        let mut r = Rng(seed);
        let mut br: Brain<256, 8, 4> = Brain::new(0.3, 0.5);
        let mut cy: Cycle<4, 4> = Cycle::new(seed ^ 0xC1C1);
        cy.grower.constant[3] = true;
        if !cycle { cy.grower.set_limit(4); }
        knobs(&mut cy);
        let (mut ok, mut params) = (0usize, 0.0f32);
        for t in 0..STEPS {
            let base = [r.f(), r.f(), r.f(), 1.0];
            let x: [f32; 8] = cy.situation(&base);
            let d = br.decide(&x, &[true; 4], 3);
            let a = if d.abstained && t < 400 { (t % 4) as u8 } else { d.action };
            let rw = reward(&base, a);
            if cycle {
                cy.learn(&mut br, &x, a, rw);
            } else {
                br.learn(&x, a, rw);
                cy.grower.after_learn(&mut br);
            }
            if t >= STEPS - 1000 {
                if a == truth(&base) { ok += 1; }
                params += cy.active_params(&br) as f32;
            }
        }
        if cycle && log { for e in cy.events() { std::println!("    grok t={} tier={} err {:.3} -> {:.3} active {}", e.t, e.tier, e.before, e.after, e.active); } }
        (ok as f32 / 1000.0, params / 1000.0, cy.events().len(), cy.grower.active())
    }

    /// The falsifier's accumulators over a seed set (sums, as the test asserts on them):
    /// right-action rate (`rc` cycle, `rg` grow-only), active params, senses, runs with a
    /// grokking event; `k` = number of seeds.
    #[derive(Clone, Copy, Debug)]
    pub struct Report { pub k: f32, pub rc: f32, pub rg: f32, pub pc: f32, pub pg: f32, pub with_event: usize, pub sc: usize, pub sg: usize }

    impl Report {
        /// The pre-registered verdict (the test's asserts): accuracy within 1 point of grow-only,
        /// strictly fewer active params, a grokking event in >= 5 of 8 runs (5/8 of the seeds).
        pub fn holds(&self) -> bool {
            let k = self.k;
            self.rc / k >= self.rg / k - 0.01 && self.pc / k < self.pg / k && 8.0 * self.with_event as f32 >= 5.0 * k
        }
    }

    /// `knobs` as in [`run`]; `log` prints the per-seed line and grokking events.
    pub fn falsifier(seeds: &[u64], knobs: fn(&mut Cycle<4, 4>), log: bool) -> Report {
        let (mut rc, mut rg, mut pc, mut pg, mut with_event, mut sc, mut sg) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0usize, 0usize, 0usize);
        for &s in seeds {
            let (r1, p1, e1, s1) = run(s, true, knobs, log);
            let (r2, p2, _, s2) = run(s, false, knobs, log);
            if log { std::println!("  seed {:>3}: cycle {:.3} ({:.0} params, {} grok events, {} senses) | grow-only {:.3} ({:.0} params, {} senses)", s, r1, p1, e1, s1, r2, p2, s2); }
            rc += r1; rg += r2; pc += p1; pg += p2; sc += s1; sg += s2;
            if e1 >= 1 { with_event += 1; }
        }
        Report { k: seeds.len() as f32, rc, rg, pc, pg, with_event, sc, sg }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::falsify::{falsifier, reward, run, Report};

    struct Rng(u64);
    impl Rng { fn f(&mut self) -> f32 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; ((self.0 >> 40) as f32) / (1u64 << 24) as f32 } }

    /// Dev-only knob overrides (env CY_*), used to tune on dev seeds 1000..1007; unset in the falsifier run.
    fn tune(cy: &mut Cycle<4, 4>) {
        let g = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok());
        if let Some(v) = g("CY_EPS") { cy.absorb_eps = v; }
        if let Some(v) = g("CY_PRUNE") { cy.prune_range = v; }
        if let Some(v) = g("CY_CSTEPS") { cy.compress_steps = v as u32; }
        if let Some(v) = g("CY_CEVERY") { cy.compress_every = v as u32; }
        if let Some(v) = g("CY_GMAX") { cy.grow_max = v as u32; }
        if let Some(v) = g("CY_PAT") { cy.grow_patience = v as u8; }
        if let Some(v) = g("CY_LAMBDA") { cy.detector.lambda = v; }
        if let Some(v) = g("CY_DELTA") { cy.detector.delta = v; }
    }

    #[test]
    #[ignore]
    fn dev_seeds() {
        let (mut rc, mut rg, mut pc, mut pg, mut ev) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0usize);
        for s in 1000u64..1008 { let (a, b, c, _) = run(s, true, tune, true); let (d, e, _, _) = run(s, false, tune, true); rc += a; pc += b; rg += d; pg += e; if c >= 1 { ev += 1; } }
        std::println!("DEV cycle {:.3} vs grow {:.3}; params {:.1} vs {:.1}; grok runs {}/8", rc / 8.0, rg / 8.0, pc / 8.0, pg / 8.0, ev);
    }

    #[test]
    fn drop_detector_fires_on_a_sustained_drop_only() {
        let mut d = DropDetector::new(0.002, 2.0, 150);
        let mut r = Rng(3);
        for _ in 0..2000 { assert!(!d.update(0.3 + 0.2 * (r.f() - 0.5)), "no drop in a stationary stream"); }
        let mut fired = false;
        for _ in 0..300 { if d.update(0.1 + 0.2 * (r.f() - 0.5)) { fired = true; break; } }
        assert!(fired, "a sustained drop 0.3 -> 0.1 is declared");
    }

    #[test]
    fn compression_forgets_only_absorbed_memories_and_is_deterministic() {
        let go = || {
            let mut br: Brain<64, 8, 4> = Brain::new(0.0, 0.0);
            let mut cy: Cycle<4, 4> = Cycle::new(1);
            cy.grower.constant[3] = true;
            let mut r = Rng(9);
            for _ in 0..400 { let base = [r.f(), r.f(), r.f(), 1.0]; let x: [f32; 8] = cy.situation(&base); let a = (r.f() * 4.0) as u8 % 4; cy.learn(&mut br, &x, a, reward(&base, a)); }
            let before = br.in_use();
            cy.compress(&mut br);
            for s in 0..64 { if let Some(e) = br.episode(s) { assert!(absf(e.reward - br.predict(e.action, &e.key)) >= cy.absorb_eps, "kept memories are not absorbed"); } }
            (before, br.in_use(), cy.forgotten)
        };
        let (a, b) = (go(), go());
        assert_eq!(a, b);
        assert!(a.1 <= a.0);
    }

    /// Pre-registered falsifier (set before the run), 8 seeds and 8 fresh seeds:
    /// the cycle's right-action rate (last 1000) >= grow-only's minus 1 point, with strictly
    /// fewer active parameters on average (last 1000), and >= 1 grokking event in most runs (>= 5 of 8).
    ///
    /// RESULT (knobs tuned on dev seeds 1000..1007 only): FAILS on accuracy, kept ignored as the record.
    /// Seeds 11..89: cycle 0.943 vs grow-only 0.948 (pass), params 261.8 vs 265.7, grok 8/8.
    /// Fresh 101..137: cycle 0.944 vs grow-only 0.961 (FAIL, -1.7 pts), params 263.7 vs 265.5, grok 8/8.
    /// Run: cargo test --release --lib brain_cycle -- --ignored --nocapture
    #[test]
    #[ignore]
    fn falsifier_cycle_matches_grow_only_with_fewer_parameters_and_groks() {
        for seeds in [[11u64, 23, 37, 41, 59, 61, 73, 89], [101, 103, 107, 109, 113, 127, 131, 137]] {
            let rep = falsifier(&seeds, tune, true);
            let Report { k, rc, rg, pc, pg, with_event, sc, sg } = rep;
            std::println!("cycle falsifier: right-action cycle {:.3} vs grow-only {:.3}; active params {:.1} vs {:.1}; senses {:.2} vs {:.2}; runs with a grokking event {}/{}",
                rc / k, rg / k, pc / k, pg / k, sc as f32 / k, sg as f32 / k, with_event, seeds.len());
            assert_eq!(rep.holds(), rc / k >= rg / k - 0.01 && pc / k < pg / k && with_event >= 5, "the gym verdict agrees with the asserts");
            assert!(rc / k >= rg / k - 0.01, "accuracy: cycle {} vs grow-only {}", rc / k, rg / k);
            assert!(pc / k < pg / k, "params: cycle {} vs grow-only {}", pc / k, pg / k);
            assert!(with_event >= 5, "grokking events in {}/8 runs", with_event);
        }
    }
}
