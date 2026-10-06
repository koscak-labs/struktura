//! brain_dream: transition memory and bounded look-ahead over the native brain.
//!
//! The core [`Brain`] chooses the action with the best *immediate* expected
//! reward. That is greedy: when the best-looking first move blocks a better
//! second move (drive now and overheat, versus cool down and then drive), it
//! takes the trap. A [`Dreamer`] wraps a brain and also remembers, for every
//! stored episode, the situation that followed it (or that the episode ended).
//! Deciding, it plans up to [`MAX_DEPTH`] steps ahead:
//!
//! `value(x, a) = expected(x, a) + gamma * (1 - p_end) * max_a' value(next(x, a), a')`
//!
//! where `next(x, a)` is the similarity-weighted average of the next
//! situations of the nearest stored episodes that took `a` near `x`, and
//! `p_end` is the weighted share of those episodes that ended. Planning is
//! used only when every allowed action has at least `min_transition_evidence`
//! of transition evidence behind it; otherwise it falls back to the brain's
//! own one-step decision. If the brain abstains (thin evidence), so does the
//! dreamer.
//!
//! No heap, bounded loops (depth <= 3, A actions, N memory slots, so at most
//! A^2 * (depth - 1) memory scans per decision), deterministic, `const`
//! constructible. Cost over the bare brain: `N * (4 D + 1)` bytes
//! (8.4 KB for 256 episodes of 8 features).
//!
//! Not done here: "dreaming" in the Dyna sense (replaying memory to refine
//! the model during idle time) needs a core call that updates the model
//! without storing a new episode; [`Brain::learn`] always stores one.

use crate::brain::Brain;

/// Deepest look-ahead (decision step + this many - 1 future steps).
pub const MAX_DEPTH: u8 = 3;
/// Transitions consulted per (situation, action).
const KT: usize = 4;

const UNKNOWN: u8 = 0;
const ENDED: u8 = 1;
const NEXT: u8 = 2;

/// A planned decision.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plan {
    pub action: u8,
    /// Planned value of the chosen action (one-step expected value when not planned).
    pub value: f32,
    /// What the brain alone (one step, greedy) would have chosen.
    pub greedy_action: u8,
    /// True when the look-ahead was used (enough transition evidence for every allowed action).
    pub planned: bool,
    /// True when the brain abstained to the safe default.
    pub abstained: bool,
}

pub struct Dreamer<const N: usize, const D: usize, const A: usize> {
    pub brain: Brain<N, D, A>,
    next: [[f32; D]; N],
    kind: [u8; N],
    /// Discount on future reward, in [0, 1].
    pub gamma: f32,
    /// Planning depth, clamped to 1..=MAX_DEPTH (1 = greedy).
    pub depth: u8,
    /// Minimum similarity weight of stored transitions per allowed action before planning.
    pub min_transition_evidence: f32,
}

impl<const N: usize, const D: usize, const A: usize> Dreamer<N, D, A> {
    pub const fn new(explore: f32, min_evidence: f32, gamma: f32, depth: u8) -> Self {
        Dreamer { brain: Brain::new(explore, min_evidence), next: [[0.0; D]; N], kind: [UNKNOWN; N],
            gamma, depth, min_transition_evidence: 0.5 }
    }

    /// Learn one step: situation `x`, `action`, `reward`, and the situation that followed
    /// (`None` when the episode ended). Returns the memory slot used.
    pub fn learn(&mut self, x: &[f32; D], action: u8, reward: f32, next: Option<&[f32; D]>) -> Option<usize> {
        let slot = self.brain.learn(x, action, reward)?;
        if slot < N {
            match next {
                Some(n) => { self.next[slot] = *n; self.kind[slot] = NEXT; }
                None => { self.next[slot] = [0.0; D]; self.kind[slot] = ENDED; }
            }
        }
        Some(slot)
    }

    fn dist2(a: &[f32; D], b: &[f32; D]) -> f32 {
        let mut s = 0.0; for i in 0..D { let d = a[i] - b[i]; s += d * d; } s
    }

    /// Predicted situation after taking `action` in `x`: (prediction, evidence, share of episodes that ended).
    /// Evidence is the summed similarity weight of the nearest stored transitions with that action.
    pub fn predict_next(&self, x: &[f32; D], action: u8) -> ([f32; D], f32, f32) {
        let mut slots = [0usize; KT];
        let mut d2 = [f32::INFINITY; KT];
        let mut n = 0usize;
        for s in 0..N {
            if self.kind[s] == UNKNOWN { continue; }
            let Some(e) = self.brain.episode(s) else { continue };
            if e.action != action { continue; }
            let d = Self::dist2(x, &e.key);
            if n < KT || d < d2[n - 1] {
                let mut p = if n < KT { n } else { KT - 1 };
                while p > 0 && d2[p - 1] > d { d2[p] = d2[p - 1]; slots[p] = slots[p - 1]; p -= 1; }
                d2[p] = d; slots[p] = s;
                if n < KT { n += 1; }
            }
        }
        let mut out = [0.0f32; D];
        let (mut w_all, mut w_next, mut w_end) = (0.0f32, 0.0f32, 0.0f32);
        for k in 0..n {
            let w = 1.0 / (1.0 + d2[k]);
            w_all += w;
            if self.kind[slots[k]] == ENDED { w_end += w; continue; }
            w_next += w;
            for i in 0..D { out[i] += w * self.next[slots[k]][i]; }
        }
        if w_next > 0.0 { for v in out.iter_mut() { *v /= w_next; } }
        let ended = if w_all > 0.0 { w_end / w_all } else { 0.0 };
        (out, w_all, ended)
    }

    /// Planned value of taking `action` in `x`, looking `depth - 1` steps further.
    /// Branches with thin transition evidence count only their immediate value.
    pub fn value(&self, x: &[f32; D], action: u8, depth: u8) -> f32 {
        let a = action as usize;
        if a >= A { return f32::NEG_INFINITY; }
        let now = self.brain.estimates(x)[a].expected;
        if depth <= 1 { return now; }
        let (nx, evidence, ended) = self.predict_next(x, action);
        if evidence < self.min_transition_evidence || ended >= 1.0 { return now; }
        let mut best = f32::NEG_INFINITY;
        for a2 in 0..A {
            let v = self.value(&nx, a2 as u8, depth - 1);
            if v > best { best = v; }
        }
        if !best.is_finite() { best = 0.0; }
        now + self.gamma * (1.0 - ended) * best
    }

    /// Decide in situation `x`. Plans when every allowed action has enough transition
    /// evidence; otherwise returns the brain's own (greedy) decision.
    pub fn decide(&self, x: &[f32; D], allowed: &[bool; A], safe_default: u8) -> Plan {
        let g = self.brain.decide(x, allowed, safe_default);
        let greedy = Plan { action: g.action, value: g.expected, greedy_action: g.action, planned: false, abstained: g.abstained };
        let depth = if self.depth < 1 { 1 } else if self.depth > MAX_DEPTH { MAX_DEPTH } else { self.depth };
        if g.abstained || depth == 1 { return greedy; }
        for a in 0..A {
            if allowed[a] && self.predict_next(x, a as u8).1 < self.min_transition_evidence { return greedy; }
        }
        let est = self.brain.estimates(x);
        let mut best: Option<(usize, f32, f32)> = None; // (action, score, value)
        for a in 0..A {
            if !allowed[a] { continue; }
            let v = self.value(x, a as u8, depth);
            let score = v + self.brain.explore * est[a].width;
            if best.map(|b| score > b.1).unwrap_or(true) { best = Some((a, score, v)); }
        }
        match best {
            Some((a, _, v)) => Plan { action: a as u8, value: v, greedy_action: g.action, planned: true, abstained: false },
            None => greedy,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn f(&mut self) -> f32 { self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((self.0 >> 40) as f32) / (1u64 << 24) as f32 }
        fn bit(&mut self) -> u8 { (self.f() < 0.5) as u8 }
    }

    // Two-step rover task. Situation [temperature, step, 1].
    // Step 1 (hot): 0 = drive now (+0.3, overheats), 1 = cool down (0, cools).
    // Step 2: cool -> drive (0) earns +1; overheated -> nothing earns anything. Then the episode ends.
    const DRIVE: u8 = 0;
    fn start(r: &mut Lcg) -> [f32; 3] { [0.85 + 0.1 * r.f(), 0.0, 1.0] }
    const HOT: [f32; 3] = [1.0, 1.0, 1.0];
    const COOL: [f32; 3] = [0.2, 1.0, 1.0];
    fn step1(a: u8) -> (f32, [f32; 3]) { if a == DRIVE { (0.3, HOT) } else { (0.0, COOL) } }
    fn step2(x: &[f32; 3], a: u8) -> f32 { if x[0] < 0.5 && a == DRIVE { 1.0 } else { 0.0 } }

    #[test]
    fn two_step_rover_planner_beats_greedy() {
        let mut d: Dreamer<256, 3, 2> = Dreamer::new(0.0, 0.5, 0.9, 2);
        let mut g: Brain<256, 3, 2> = Brain::new(0.0, 0.5);
        let mut r = Lcg(11);
        // Identical exploration data for both (120 episodes, 240 steps: no eviction).
        for _ in 0..120 {
            let x0 = start(&mut r);
            let a0 = r.bit();
            let (r0, x1) = step1(a0);
            let a1 = r.bit();
            let r1 = step2(&x1, a1);
            d.learn(&x0, a0, r0, Some(&x1)); d.learn(&x1, a1, r1, None);
            g.learn(&x0, a0, r0); g.learn(&x1, a1, r1);
        }
        let (mut tg, mut tp, mut planned, n) = (0.0f32, 0.0f32, 0u32, 200u32);
        for _ in 0..n {
            let x0 = start(&mut r);
            let a = g.decide(&x0, &[true; 2], 1).action;
            let (r0, x1) = step1(a);
            tg += r0 + step2(&x1, g.decide(&x1, &[true; 2], 1).action);
            let p = d.decide(&x0, &[true; 2], 1);
            if p.planned { planned += 1; }
            let (r0, x1) = step1(p.action);
            tp += r0 + step2(&x1, d.decide(&x1, &[true; 2], 1).action);
        }
        let (mg, mp) = (tg / n as f32, tp / n as f32);
        println!("two-step rover: greedy {:.3} / episode, planner {:.3} / episode, planned {}/{}", mg, mp, planned, n);
        assert!(mp >= mg + 0.5, "planner {:.3} vs greedy {:.3}", mp, mg);
    }

    #[test]
    fn no_multistep_structure_is_not_worse_than_greedy() {
        // Step 1: action 0 earns +0.5, action 1 earns +0.2; the next situation is random and
        // step 2 pays +0.4 for action 0 whatever happened before. Greedy is already optimal.
        let mut d: Dreamer<256, 3, 2> = Dreamer::new(0.0, 0.5, 0.9, 3);
        let mut g: Brain<256, 3, 2> = Brain::new(0.0, 0.5);
        let mut r = Lcg(5);
        let s1 = |a: u8| if a == 0 { 0.5 } else { 0.2 };
        let s2 = |a: u8| if a == 0 { 0.4 } else { 0.0 };
        for _ in 0..120 {
            let x0 = [0.3 + 0.1 * r.f(), 0.0, 1.0];
            let a0 = r.bit();
            let x1 = [r.f(), 1.0, 1.0];
            let a1 = r.bit();
            d.learn(&x0, a0, s1(a0), Some(&x1)); d.learn(&x1, a1, s2(a1), None);
            g.learn(&x0, a0, s1(a0)); g.learn(&x1, a1, s2(a1));
        }
        let (mut tg, mut tp, n) = (0.0f32, 0.0f32, 200u32);
        for _ in 0..n {
            let x0 = [0.3 + 0.1 * r.f(), 0.0, 1.0];
            let x1 = [r.f(), 1.0, 1.0];
            tg += s1(g.decide(&x0, &[true; 2], 1).action) + s2(g.decide(&x1, &[true; 2], 1).action);
            tp += s1(d.decide(&x0, &[true; 2], 1).action) + s2(d.decide(&x1, &[true; 2], 1).action);
        }
        let (mg, mp) = (tg / n as f32, tp / n as f32);
        println!("no multi-step structure: greedy {:.3} / episode, planner {:.3} / episode", mg, mp);
        assert!(mp >= mg - 0.02, "planner {:.3} vs greedy {:.3}", mp, mg);
    }

    #[test]
    fn single_step_world_gives_the_greedy_decision() {
        // Every episode ends after one step: no future, so planning must agree with the brain.
        let mut d: Dreamer<128, 3, 3> = Dreamer::new(0.0, 0.5, 0.9, 3);
        let mut r = Lcg(9);
        let best = |x: &[f32; 3]| if x[0] > 0.5 { 1u8 } else { 2 };
        for t in 0..300u32 {
            let x = [r.f(), r.f(), 1.0];
            let a = (t % 3) as u8;
            d.learn(&x, a, if a == best(&x) { 1.0 } else { 0.0 }, None);
        }
        for _ in 0..200 {
            let x = [r.f(), r.f(), 1.0];
            let p = d.decide(&x, &[true; 3], 0);
            assert_eq!(p.action, p.greedy_action, "{:?}", p);
        }
    }

    #[test]
    fn thin_transition_evidence_falls_back_to_greedy() {
        let mut d: Dreamer<32, 3, 2> = Dreamer::new(0.0, 0.0, 0.9, 3);
        // Only action 0 has transitions; action 1 has none -> no planning.
        for _ in 0..5 { d.learn(&[0.5, 0.0, 1.0], 0, 0.3, Some(&[0.5, 1.0, 1.0])); }
        let p = d.decide(&[0.5, 0.0, 1.0], &[true; 2], 1);
        assert!(!p.planned);
        assert_eq!(p.action, d.brain.decide(&[0.5, 0.0, 1.0], &[true; 2], 1).action);
        // Disallowed actions are never chosen, planned or not.
        let q = d.decide(&[0.5, 0.0, 1.0], &[false, true], 1);
        assert_eq!(q.action, 1);
    }

    #[test]
    fn depth_is_clamped_and_cost_is_bounded() {
        static _D: Dreamer<16, 4, 3> = Dreamer::new(0.1, 1.0, 0.9, 9);
        let mut d: Dreamer<64, 3, 2> = Dreamer::new(0.0, 0.0, 0.9, 200);
        let mut r = Lcg(3);
        for _ in 0..60 { let x = [r.f(), 0.0, 1.0]; let a = r.bit(); d.learn(&x, a, r.f(), Some(&[r.f(), 1.0, 1.0])); }
        let p = d.decide(&[0.5, 0.0, 1.0], &[true; 2], 0);
        assert!(p.value.is_finite());
        let extra = core::mem::size_of::<Dreamer<256, 8, 4>>() - core::mem::size_of::<Brain<256, 8, 4>>();
        println!("dreamer cost over the brain at N=256, D=8, A=4: {} bytes", extra);
        assert!(extra <= 256 * (4 * 8 + 1) + 16, "{}", extra);
    }

    #[test]
    fn deterministic() {
        let run = || {
            let mut d: Dreamer<128, 3, 2> = Dreamer::new(0.1, 0.5, 0.9, 2);
            let mut r = Lcg(21);
            let mut h = 0u32;
            for _ in 0..60 {
                let x0 = start(&mut r);
                let p = d.decide(&x0, &[true; 2], 1);
                let a0 = if p.abstained { r.bit() } else { p.action };
                let (r0, x1) = step1(a0);
                let a1 = r.bit();
                d.learn(&x0, a0, r0, Some(&x1)); d.learn(&x1, a1, step2(&x1, a1), None);
                h = h.wrapping_mul(31).wrapping_add(a0 as u32 + 1);
            }
            h
        };
        assert_eq!(run(), run());
    }
}
