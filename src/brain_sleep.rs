//! Sleep: consolidation from memory into the model, then freeing what was absorbed.
//!
//! Inspired by two ideas from neuroscience:
//! - **Hippocampus -> cortex replay**: during sleep, recent episodes held in fast
//!   memory are replayed into slow, general structure (here: the brain's model).
//! - **Synaptic homeostasis**: waking learning keeps strengthening synapses; sleep
//!   scales them back down so that only what is replayed keeps its weight.
//!
//! While awake, sleep pressure builds with every learned outcome (plus a share of
//! its surprise). When it crosses `threshold` (or when the caller asks, e.g. in an
//! idle window or the lunar night), [`Sleep::sleep`] runs one bounded sleep:
//! 1. downscale the model ([`Brain::forget`] with `keep`): old evidence keeps only part of its weight;
//! 2. replay up to `replay` remembered episodes into the model ([`Brain::update_model`]),
//!    most surprising first, so what memory knows is consolidated into the model;
//! 3. forget episodes the model now predicts within `eps` ([`Brain::forget_slot`]):
//!    their knowledge lives in the model, so memory space is freed for new surprises.
//!    At most `max_free` are freed per sleep; surprising episodes are always kept.
//!
//! Measured (rover worlds, memory N = 32, 6000 outcomes, 8 + 8 fresh seeds): sleep frees
//! 89-214 episodes per run with no measurable held-out loss (change -0.003 .. +0.013), but it does
//! NOT raise held-out accuracy (pre-registered falsifier failed; kept as an ignored test).
//! The defaults (keep 1.0 = no downscale, replay 16 in memory order) were picked by a sweep
//! on seed set 1; the homeostatic downscale (keep < 1) did not pay off in that sweep.
//!
//! Downscaling before replay is what prevents double counting: every replayed
//! episode was already learned once while awake, so without the downscale it would
//! count twice.
//!
//! no_std, no heap, bounded work per sleep (two passes over memory plus `replay`
//! model updates of O(D^2)), deterministic.

use crate::brain::Brain;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sleep {
    /// Current sleep pressure.
    pub pressure: f32,
    /// Pressure at which [`Sleep::tick`] asks for sleep.
    pub threshold: f32,
    /// Share of a learned outcome's surprise added to the pressure (on top of 1 per outcome).
    pub surprise_weight: f32,
    /// Model weight kept by the homeostatic downscale (0 < keep <= 1).
    pub keep: f32,
    /// Most episodes replayed into the model per sleep.
    pub replay: u16,
    /// An episode the model predicts within this absolute error is absorbed and may be freed.
    pub eps: f32,
    /// Episodes with surprise at or above this are never freed.
    pub keep_surprise: f32,
    /// Most episodes freed per sleep.
    pub max_free: u16,
    /// Replay the most surprising episodes first (true) or in memory order (false).
    pub surprising_first: bool,
    /// Sleeps so far.
    pub sleeps: u32,
    /// Episodes freed so far.
    pub freed: u32,
}

/// What one sleep did.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SleepReport {
    pub replayed: u16,
    pub freed: u16,
    pub in_use_before: u16,
    pub in_use_after: u16,
}

impl Sleep {
    /// Defaults for rewards in [0, 1]: sleep every ~64 outcomes, keep half the model weight,
    /// replay up to 64 episodes, free episodes predicted within 0.15, never free surprise >= 0.5.
    pub const fn new() -> Self {
        Sleep { pressure: 0.0, threshold: 64.0, surprise_weight: 1.0, keep: 1.0, replay: 16, eps: 0.15,
            keep_surprise: 0.5, max_free: 16, surprising_first: false, sleeps: 0, freed: 0 }
    }

    /// Call after each learned outcome with its surprise (|reward - expected|, e.g. from the
    /// stored episode). True when pressure has crossed the threshold.
    pub fn tick(&mut self, surprise: f32) -> bool {
        let s = if surprise.is_finite() && surprise > 0.0 { surprise } else { 0.0 };
        self.pressure += 1.0 + self.surprise_weight * s;
        self.pressure >= self.threshold
    }

    /// [`Sleep::tick`] using the surprise stored in `slot` (the slot [`Brain::learn`] returned).
    pub fn tick_learned<const N: usize, const D: usize, const A: usize>(&mut self, br: &Brain<N, D, A>, slot: Option<usize>) -> bool {
        let s = slot.and_then(|s| br.episode(s)).map(|e| e.surprise).unwrap_or(0.0);
        self.tick(s)
    }

    /// One bounded sleep: downscale, replay most-surprising-first, free what the model absorbed.
    pub fn sleep<const N: usize, const D: usize, const A: usize>(&mut self, br: &mut Brain<N, D, A>) -> SleepReport {
        let in_use_before = br.in_use() as u16;
        // 1. synaptic homeostasis: scale the model back down
        br.forget(self.keep);
        // 2. replay, most surprising first: repeatedly pick the next most surprising not yet
        //    replayed (bounded: replay x N comparisons; a bitset marks replayed slots).
        let mut replayed = 0u16;
        let mut done = [0u64; 64]; // up to 4096 slots
        let cap = N.min(64 * 64);
        while replayed < self.replay {
            let mut best: Option<usize> = None;
            let mut best_s = f32::NEG_INFINITY;
            for s in 0..cap {
                if done[s / 64] >> (s % 64) & 1 == 1 { continue; }
                if let Some(e) = br.episode(s) {
                    let key = if self.surprising_first { e.surprise } else { -(s as f32) };
                    if key > best_s { best_s = key; best = Some(s); }
                }
            }
            let Some(s) = best else { break };
            done[s / 64] |= 1u64 << (s % 64);
            let e = *br.episode(s).unwrap();
            br.update_model(&e.key, e.action, e.reward);
            replayed += 1;
        }
        // 3. free episodes the consolidated model now predicts well (absorbed), least surprising first
        let mut freed = 0u16;
        while freed < self.max_free {
            let mut pick: Option<usize> = None;
            let mut pick_s = f32::INFINITY;
            for s in 0..N {
                let Some(e) = br.episode(s) else { continue };
                if e.surprise >= self.keep_surprise { continue; }
                let err = e.reward - br.predict(e.action, &e.key);
                let err = if err < 0.0 { -err } else { err };
                if err < self.eps && e.surprise < pick_s { pick_s = e.surprise; pick = Some(s); }
            }
            let Some(s) = pick else { break };
            br.forget_slot(s);
            freed += 1;
        }
        self.pressure = 0.0;
        self.sleeps = self.sleeps.saturating_add(1);
        self.freed = self.freed.saturating_add(freed as u32);
        SleepReport { replayed, freed, in_use_before, in_use_after: br.in_use() as u16 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng { fn f(&mut self) -> f32 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; ((self.0 >> 40) as f32) / (1u64 << 24) as f32 } }

    // The two rover worlds from `struktura brain demo`. Situation: [drift, spike, temp, 1].
    fn truth_a(x: &[f32; 4]) -> u8 { if x[2] > 0.8 { 3 } else if x[1] > 0.6 { 2 } else if x[0] > 0.6 { 1 } else { 0 } }
    fn truth_b(x: &[f32; 4]) -> u8 {
        if x[2] > 0.6 && x[0] > 0.6 { 3 } else if x[1] > 0.6 && x[2] < 0.4 { 2 } else if (x[0] > 0.5) != (x[1] > 0.5) { 1 } else { 0 }
    }
    fn reward(right: bool, a: u8) -> f32 { if right { 1.0 } else if a == 3 { 0.2 } else { 0.0 } }

    /// Long stream into a SMALL memory (N = 32). Returns (held-out right-action rate,
    /// memory in use at the end, sleeps, freed).
    fn run(world: fn(&[f32; 4]) -> u8, seed: u64, with_sleep: bool) -> (f32, usize, u32, u32) { run_with(world, seed, with_sleep, Sleep::new()) }

    fn run_with(world: fn(&[f32; 4]) -> u8, seed: u64, with_sleep: bool, cfg: Sleep) -> (f32, usize, u32, u32) {
        let mut r = Rng(seed);
        let mut br: Brain<32, 4, 4> = Brain::new(0.3, 0.5);
        let mut sl = cfg;
        for t in 0..6000u32 {
            let x = [r.f(), r.f(), r.f(), 1.0];
            let d = br.decide(&x, &[true; 4], 3);
            let a = if d.abstained && t < 400 { (t % 4) as u8 } else { d.action };
            let slot = br.learn(&x, a, reward(a == world(&x), a));
            if with_sleep && sl.tick_learned(&br, slot) { sl.sleep(&mut br); }
        }
        br.explore = 0.0;
        let mut h = Rng(seed ^ 0xD00D);
        let mut ok = 0;
        for _ in 0..1000 {
            let x = [h.f(), h.f(), h.f(), 1.0];
            if br.decide(&x, &[true; 4], 3).action == world(&x) { ok += 1; }
        }
        (ok as f32 / 1000.0, br.in_use(), sl.sleeps, sl.freed)
    }

    /// Pre-registered falsifier (set before the run), 8 seeds and 8 fresh seeds, both worlds,
    /// memory N = 32 under a 6000-outcome stream:
    /// (1) held-out right-action rate with sleep > without sleep (average per world and seed set);
    /// (2) sleep frees memory (episodes freed > 0) and held-out accuracy is not lost
    ///     (covered by (1): with-sleep accuracy is at least the no-sleep accuracy).
    #[test]
    #[ignore = "FAILED as pre-registered: sleep does not raise held-out accuracy (fresh seeds: A 0.828 -> 0.830, B 0.707 -> 0.704; set 1 after a sweep on it: A 0.843 -> 0.845, B 0.705 -> 0.718)"]
    fn falsifier_sleep_consolidates_and_frees_memory() {
        let worlds: [(&str, fn(&[f32; 4]) -> u8); 2] = [("A thresholds", truth_a), ("B interactions", truth_b)];
        for seeds in [[11u64, 23, 37, 41, 59, 61, 73, 89], [101, 103, 107, 109, 113, 127, 131, 137]] {
            for (name, w) in worlds {
                let (mut plain, mut slept, mut freed, mut sleeps, mut used) = (0.0f32, 0.0f32, 0u32, 0u32, 0usize);
                for &s in &seeds {
                    plain += run(w, s, false).0;
                    let (acc, u, n, f) = run(w, s, true);
                    slept += acc; used += u; sleeps += n; freed += f;
                }
                let k = seeds.len() as f32;
                std::println!("sleep falsifier, world {} (seeds {}..): held-out right-action no sleep {:.3} vs sleep {:.3}; {:.0} sleeps, {:.0} episodes freed, {:.1}/32 memory in use at end",
                    name, seeds[0], plain / k, slept / k, sleeps as f32 / k, freed as f32 / k, used as f32 / k);
                assert!(slept > plain, "world {}: sleep {} vs no sleep {}", name, slept / k, plain / k);
                assert!(freed > 0, "sleep freed nothing");
            }
        }
    }

    #[test]
    #[ignore]
    fn sweep_seed_set_1() {
        let seeds = [11u64, 23, 37, 41, 59, 61, 73, 89];
        let worlds: [(&str, fn(&[f32; 4]) -> u8); 2] = [("A", truth_a), ("B", truth_b)];
        for (name, w) in worlds {
            let base: f32 = seeds.iter().map(|&s| run(w, s, false).0).sum::<f32>() / 8.0;
            std::println!("world {} no sleep {:.3}", name, base);
            for keep in [0.5f32, 0.8, 1.0] { for replay in [0u16, 16, 64] { for sf in [true, false] { for eps in [0.05f32, 0.15] {
                let mut c = Sleep::new(); c.keep = keep; c.replay = replay; c.surprising_first = sf; c.eps = eps;
                let v: f32 = seeds.iter().map(|&s| run_with(w, s, true, c).0).sum::<f32>() / 8.0;
                std::println!("  keep {:.1} replay {:>2} surprising_first {} eps {:.2}: {:.3} ({:+.3})", keep, replay, sf, eps, v, v - base);
            } } } }
        }
    }

    /// Post-hoc, weaker claim (added AFTER the pre-registered falsifier failed, reported as such):
    /// sleep frees memory every run without a measurable held-out loss (change >= -0.01, both
    /// worlds, both seed sets).
    #[test]
    fn sleep_frees_memory_without_measurable_loss() {
        let worlds: [fn(&[f32; 4]) -> u8; 2] = [truth_a, truth_b];
        for seeds in [[11u64, 23, 37, 41, 59, 61, 73, 89], [101, 103, 107, 109, 113, 127, 131, 137]] {
            for w in worlds {
                let (mut plain, mut slept, mut freed) = (0.0f32, 0.0f32, 0u32);
                for &s in &seeds { plain += run(w, s, false).0; let (a, _, _, f) = run(w, s, true); slept += a; freed += f; }
                let d = (slept - plain) / seeds.len() as f32;
                assert!(freed > 0 && d >= -0.01, "freed {} change {}", freed, d);
            }
        }
    }

    #[test]
    fn sleep_is_bounded_and_keeps_surprises() {
        let mut br: Brain<16, 3, 2> = Brain::new(0.0, 0.0);
        for i in 0..16 { br.learn(&[i as f32 / 16.0, 0.5, 1.0], 0, 0.5); }
        br.learn(&[0.9, 0.9, 1.0], 1, 10.0); // a shock: high surprise
        let mut sl = Sleep::new();
        sl.replay = 4;
        sl.max_free = 3;
        let rep = sl.sleep(&mut br);
        assert_eq!(rep.replayed, 4);
        assert!(rep.freed <= 3);
        assert_eq!(rep.in_use_after, rep.in_use_before - rep.freed);
        assert!((0..16).any(|s| br.episode(s).map(|e| e.reward == 10.0).unwrap_or(false)), "the shock is never freed");
        assert_eq!(sl.pressure, 0.0);
    }

    #[test]
    fn pressure_builds_with_learning_and_surprise() {
        let mut sl = Sleep::new();
        sl.threshold = 5.0;
        assert!(!sl.tick(0.0));
        assert!(!sl.tick(1.0));
        assert!(sl.tick(2.0), "1 + 2 + 3 >= 5");
    }

    #[test]
    fn deterministic() {
        assert_eq!(run(truth_b, 7, true), run(truth_b, 7, true));
    }
}
