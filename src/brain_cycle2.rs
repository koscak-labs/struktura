//! Growth cycle v2: grow -> compress -> grokking on a HELD-OUT probe -> grow bigger.
//!
//! Cycle v1 ([`crate::brain_cycle`]) failed the grokking falsifier ([`crate::brain_grok`]) for
//! measured reasons: its grokking detector watched the prequential error of the TRAINING stream
//! (memory fits a replayed training set in one pass, so nothing is left to drop and the tier gate
//! stays shut), and its growth test (half-splits of memory) compared copies of the same memorized
//! examples, so it adopted junk as readily as useful senses. v2:
//!
//! - **Held-out probe.** The caller carves a validation slice out of its TRAINING data and hands it
//!   to the cycle with [`Cycle2::hold_out`] instead of learning it: the brain never learns those
//!   outcomes, and the cycle never sees anything else held out (never a test set). Every
//!   `probe_every` learned outcomes it measures the validation error: the mean squared error of
//!   the brain's own estimate (model + memory) on the slice.
//! - **Grokking event.** Armed at the end of the first compress phase and whenever a tier opens,
//!   with the validation error at that moment as the reference; it fires when the validation error
//!   stays at or below `(1 - grok_drop)` of the reference for `grok_hold` consecutive probes.
//!   A memorized training set cannot fire it; a held-out improvement can.
//! - **Tier gate (v1's rule) and a budget that compression frees.** Capacity comes in tiers of 1,
//!   2, 4, ... growth slots; at the end of a compress phase the next tier opens if this is the
//!   first tier or the current tier produced a grokking event. The model's input budget is
//!   `B + 2^tier`: every base sense compression removes from the model frees one growth slot
//!   ([`Cycle2::budget`]). A brain that does not compress only grows by grokking.
//! - **Growth: Hebbian proposals** ([`Cycle2::grow`]). At a growth check (every `grow_every`
//!   learned outcomes of a Grow phase) the model-only residual (reward minus the linear model's
//!   prediction, memory not consulted) is computed for every stored episode. If fewer than
//!   `min_erring` episodes are still off by `erring_eps`, there is nothing to grow for. Otherwise
//!   every candidate of the grammar ([`Feat`]: products, thresholds and comparisons of base
//!   senses) that is not grown and was not recently removed is scored by the sum over actions of
//!   its squared correlation with the residual ("fires together with the error"), and the best is
//!   adopted into a free slot of the budget. Adoption refits the model exactly as if the sense had
//!   always been there ([`Cycle2::refit`]: reset, then every distinct stored (situation, action)
//!   with the weight the replay gave it). No held-out gate at adoption: a single sense often pays
//!   only together with others, so compression decides on held-out error what stays.
//! - **Compression** (only with `compress_on`): (1) a learned outcome replaces its older stored copy
//!   (`dedup`: one memory per replayed example); (2) every `compress_every` steps of a compress
//!   phase, backward elimination of model inputs (non-constant base senses and grown senses) by
//!   held-out error on the replay: [`Cycle2::press`], the exact leave-one-situation-out error of
//!   the linear model (PRESS: k-fold on the replay with one fold per distinct situation), computed
//!   with each input removed exactly ([`Brain::loo_residual`], [`Brain::predict_without`]). The
//!   input whose removal gives the lowest PRESS is removed ([`Brain::drop_input`]) if that is at
//!   most `(1 + drop_tol)` of the PRESS with it, up to `max_drop` times per pass. A removed grown
//!   sense frees its slot and is not proposed again until the next tier opens (a tabu list).
//! - [`Cycle2::without_compression`] is the no-compression control: the same tiers, budget rule,
//!   probe, gate, growth and refit; compress phases are pauses of the same length that only
//!   evaluate the gate. [`Cycle2::grow_only`]: the same growth with every slot open from the start,
//!   no phases, no gate, no compression (a diagnostic).
//!
//! Development record (dev seeds 9001..9016 of [`crate::brain_grok2`], Fourier senses): growth by
//! a held-out transfer test of single senses rejected true products (their slope alone does not
//! transfer on 16 of 49 pairs: the products are far from orthogonal there), exhaustive growth by
//! PRESS rose only gradually with half junk, and growth every 112 outcomes finished before the
//! 10-epoch plateau the falsifier requires; the settings above (Hebbian growth every 250, v1's
//! default, PRESS compression with `drop_tol` 0 and `max_drop` 4) were kept.
//!
//! no_std, no heap, bounded work per step (a compression pass is O(D * N * D^2) for D inputs
//! and N memories), deterministic (no randomness at all).

use crate::brain::Brain;
use crate::brain_grow::{Feat, STEPS};

/// Validation outcomes kept (a ring: the oldest is replaced when full).
pub const VMAX: usize = 64;
/// Maximum grokking events kept.
pub const MAX_EVENTS: usize = 16;
/// Senses removed by compression that growth will not propose again until the next tier opens.
pub const TABU: usize = 16;

/// One held-out outcome: base senses (grown senses are re-computed), action, reward.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Held<const B: usize> {
    pub base: [f32; B],
    pub action: u8,
    pub reward: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase { Grow, Compress }

/// A recorded grokking event (validation-driven).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GrokEvent {
    /// Learned outcomes when it fired.
    pub t: u32,
    /// Validation error when the detector was armed.
    pub v_ref: f32,
    /// Validation error when it fired.
    pub v: f32,
    pub tier: u8,
    /// Active parameters when it fired (see [`Cycle2::active_params`]).
    pub active: u32,
}

pub struct Cycle2<const B: usize, const G: usize> {
    feats: [Feat; G],
    grown: usize,
    limit: usize,
    /// Base senses that are constants (bias terms): never in candidates, never removed.
    pub constant: [bool; B],
    base_on: [bool; B],
    held: [Held<B>; VMAX],
    n_held: usize,
    next_held: usize,
    /// Compression on (arm A) or off (the no-compression control).
    pub compress_on: bool,
    /// Capacity tiers and the grokking gate (false + `compress_on` false: plain growth, all slots).
    pub tiered: bool,
    phase: Phase,
    tier: u8,
    steps_in_phase: u32,
    checks_without_growth: u8,
    /// Learned outcomes between growth checks.
    pub grow_every: u32,
    since_grow: u32,
    /// Growth checks without an adoption that end a Grow phase.
    pub grow_patience: u8,
    /// Longest Grow phase (steps).
    pub grow_max: u32,
    pub compress_steps: u32,
    pub compress_every: u32,
    /// A stored episode the model alone mispredicts by this much or more is "erring".
    pub erring_eps: f32,
    /// Growth checks run only while at least this many stored episodes are erring.
    pub min_erring: usize,
    /// Remove an input when the held-out error (PRESS) without it is at most `(1 + drop_tol)` times
    /// the error with it (0: remove whatever does not pay for itself on held-out data).
    pub drop_tol: f32,
    /// Most inputs removed per compression pass.
    pub max_drop: u8,
    /// Compression also forgets an older stored copy of the same (situation, action) when it is
    /// learned again (one copy per replayed example; acts only with `compress_on`).
    pub dedup: bool,
    /// Learned outcomes between validation probes.
    pub probe_every: u32,
    since_probe: u32,
    /// Relative validation-error drop that counts as a grokking event.
    pub grok_drop: f32,
    /// Consecutive probes the drop must hold.
    pub grok_hold: u8,
    armed: bool,
    v_ref: f32,
    below: u8,
    grok_in_tier: bool,
    compressed_once: bool,
    last_val: f32,
    t: u32,
    events: [GrokEvent; MAX_EVENTS],
    n_events: usize,
    /// Memories forgotten, base inputs removed, grown senses pruned, senses adopted (so far).
    pub forgotten: u32,
    pub dropped: u32,
    pub pruned: u32,
    pub adopted: u32,
    tabu: [Feat; TABU],
    n_tabu: usize,
}

impl<const B: usize, const G: usize> Cycle2<B, G> {
    /// Arm-A defaults: tiered, compression on.
    pub fn new() -> Self {
        Cycle2 {
            feats: [Feat::Off; G], grown: 0, limit: 1.min(G), constant: [false; B], base_on: [true; B],
            held: [Held { base: [0.0; B], action: 0, reward: 0.0 }; VMAX], n_held: 0, next_held: 0,
            compress_on: true, tiered: true, phase: Phase::Grow, tier: 0, steps_in_phase: 0, checks_without_growth: 0,
            grow_every: 250, since_grow: 0, grow_patience: 2, grow_max: 1500, compress_steps: 300, compress_every: 100,
            erring_eps: 0.1, min_erring: 28, drop_tol: 0.0, max_drop: 4, dedup: true,
            probe_every: 100, since_probe: 0, grok_drop: 0.2, grok_hold: 3,
            armed: false, v_ref: 0.0, below: 0, grok_in_tier: false, compressed_once: false, last_val: f32::NAN,
            t: 0, events: [GrokEvent { t: 0, v_ref: 0.0, v: 0.0, tier: 0, active: 0 }; MAX_EVENTS], n_events: 0,
            forgotten: 0, dropped: 0, pruned: 0, adopted: 0, tabu: [Feat::Off; TABU], n_tabu: 0,
        }
    }

    /// The no-compression control: the same cycle (tiers, probe, gate, growth), compression off.
    pub fn without_compression() -> Self { let mut c = Self::new(); c.compress_on = false; c }

    /// Plain growth: every slot open from the start, no phases, no gate, no compression.
    pub fn grow_only() -> Self { let mut c = Self::new(); c.compress_on = false; c.tiered = false; c.limit = G; c }

    pub fn phase(&self) -> Phase { self.phase }
    pub fn tier(&self) -> u8 { self.tier }
    pub fn limit(&self) -> usize { self.limit }
    pub fn events(&self) -> &[GrokEvent] { &self.events[..self.n_events] }
    pub fn grown(&self) -> &[Feat] { &self.feats[..self.grown] }
    /// Grown senses currently active.
    pub fn active_senses(&self) -> usize { self.feats[..self.grown].iter().filter(|f| **f != Feat::Off).count() }
    /// Base senses still in the model (constants included).
    pub fn base_inputs(&self) -> usize { self.base_on.iter().filter(|b| **b).count() }
    /// Whether base sense `i` is still in the model.
    pub fn base_on(&self, i: usize) -> bool { i < B && self.base_on[i] }
    /// Validation error at the latest probe (NaN before the first).
    pub fn last_val(&self) -> f32 { self.last_val }
    /// Held-out outcomes in the validation slice.
    pub fn held(&self) -> &[Held<B>] { &self.held[..self.n_held] }
    pub fn compressed_once(&self) -> bool { self.compressed_once }

    /// Memories in use + model inputs (base senses still in the model + active grown senses).
    pub fn active_params<const N: usize, const D: usize, const A: usize>(&self, brain: &Brain<N, D, A>) -> u32 {
        (brain.in_use() + self.base_inputs() + self.active_senses()) as u32
    }

    /// Base senses followed by the grown senses (free slots are 0). `D` must be `B + G`.
    /// Base senses removed from the model stay in the situation (memory still reads them).
    pub fn situation<const D: usize>(&self, base: &[f32; B]) -> [f32; D] {
        let mut x = [0.0f32; D];
        for i in 0..B.min(D) { x[i] = base[i]; }
        for g in 0..self.grown { if B + g < D { x[B + g] = self.feats[g].eval(base); } }
        x
    }

    /// Put one outcome in the validation slice instead of learning it (it must come from the
    /// caller's training data; the brain never learns it).
    pub fn hold_out(&mut self, base: &[f32; B], action: u8, reward: f32) {
        self.held[self.next_held] = Held { base: *base, action, reward };
        self.next_held = (self.next_held + 1) % VMAX;
        if self.n_held < VMAX { self.n_held += 1; }
    }

    /// Validation error of the brain's own estimate (model + memory): mean squared error.
    pub fn val_error<const N: usize, const D: usize, const A: usize>(&self, brain: &Brain<N, D, A>) -> f32 {
        if self.n_held == 0 || B + G != D { return f32::NAN; }
        let mut s = 0.0f32;
        for h in self.held() {
            let x: [f32; D] = self.situation(&h.base);
            let e = h.reward - brain.estimates(&x)[(h.action as usize).min(A - 1)].expected;
            s += e * e;
        }
        s / self.n_held as f32
    }

    /// Model-only validation error, optionally with model input `without` removed exactly.
    pub fn val_model_error<const N: usize, const D: usize, const A: usize>(&self, brain: &Brain<N, D, A>, without: Option<usize>) -> f32 {
        if self.n_held == 0 || B + G != D { return f32::NAN; }
        let mut s = 0.0f32;
        for h in self.held() {
            let x: [f32; D] = self.situation(&h.base);
            let p = match without { Some(j) => brain.predict_without(h.action, &x, j), None => brain.predict(h.action, &x) };
            let e = h.reward - p;
            s += e * e;
        }
        s / self.n_held as f32
    }

    fn arm<const N: usize, const D: usize, const A: usize>(&mut self, brain: &Brain<N, D, A>) {
        let v = self.val_error(brain);
        if v.is_finite() { self.v_ref = v; self.armed = true; self.below = 0; }
    }

    /// Learn one outcome (situation `x` from [`Cycle2::situation`]) and advance the cycle.
    /// Returns a grokking event when one fires.
    pub fn learn<const N: usize, const D: usize, const A: usize>(&mut self, brain: &mut Brain<N, D, A>, x: &[f32; D], action: u8, reward: f32) -> Option<GrokEvent> {
        if (action as usize) >= A || B + G != D { return None; }
        if self.compress_on && self.dedup {
            for s in 0..N {
                let dup = brain.episode(s).map(|e| e.action == action && e.key == *x).unwrap_or(false);
                if dup && brain.forget_slot(s) { self.forgotten += 1; }
            }
        }
        brain.learn(x, action, reward);
        self.t = self.t.wrapping_add(1);
        self.steps_in_phase += 1;
        self.since_probe += 1;
        let mut fired = None;
        if self.since_probe >= self.probe_every.max(1) {
            self.since_probe = 0;
            let v = self.val_error(brain);
            self.last_val = v;
            if self.armed && v.is_finite() {
                if v <= (1.0 - self.grok_drop) * self.v_ref { self.below = self.below.saturating_add(1); } else { self.below = 0; }
                if self.below >= self.grok_hold.max(1) {
                    let ev = GrokEvent { t: self.t, v_ref: self.v_ref, v, tier: self.tier, active: self.active_params(brain) };
                    if self.n_events < MAX_EVENTS { self.events[self.n_events] = ev; self.n_events += 1; }
                    self.grok_in_tier = true;
                    self.armed = false;
                    fired = Some(ev);
                }
            }
        }
        let phased = self.tiered || self.compress_on;
        match self.phase {
            Phase::Grow => {
                self.since_grow += 1;
                if self.since_grow >= self.grow_every.max(1) {
                    self.since_grow = 0;
                    let adopted = self.free_slot().is_some() && self.grow(brain).is_some();
                    if adopted { self.checks_without_growth = 0; } else { self.checks_without_growth = self.checks_without_growth.saturating_add(1); }
                }
                if phased && (self.checks_without_growth >= self.grow_patience || self.steps_in_phase >= self.grow_max) {
                    self.phase = Phase::Compress;
                    self.steps_in_phase = 0;
                }
            }
            Phase::Compress => {
                if self.compress_on && self.steps_in_phase % self.compress_every.max(1) == 0 { self.compress(brain); }
                if self.steps_in_phase >= self.compress_steps {
                    let open = self.tiered && self.limit < G && (self.tier == 0 || self.grok_in_tier);
                    if open {
                        self.tier = self.tier.saturating_add(1);
                        self.limit = if self.tier >= 16 { G } else { (1usize << self.tier).min(G) };
                        self.grok_in_tier = false;
                        self.tabu = [Feat::Off; TABU];
                        self.n_tabu = 0;
                    }
                    if open || !self.compressed_once { self.arm(brain); }
                    self.compressed_once = true;
                    self.phase = Phase::Grow;
                    self.steps_in_phase = 0;
                    self.checks_without_growth = 0;
                    self.since_grow = 0;
                }
            }
        }
        fired
    }

    /// Stored episodes with a distinct (situation, action) (the first copy of each), their number,
    /// and per action their count.
    fn distinct<const N: usize, const D: usize, const A: usize>(brain: &Brain<N, D, A>) -> ([u16; N], usize, [u32; A]) {
        let (mut slots, mut m, mut count) = ([0u16; N], 0usize, [0u32; A]);
        for s in 0..N {
            let Some(e) = brain.episode(s) else { continue };
            let dup = slots[..m].iter().any(|&q| brain.episode(q as usize).map(|o| o.action == e.action && o.key == e.key).unwrap_or(false));
            if dup { continue; }
            slots[m] = s as u16;
            m += 1;
            count[e.action as usize] += 1;
        }
        (slots, m, count)
    }

    /// How many times the model learned each distinct (situation, action) of an action: the replay
    /// is balanced, so `pulls / distinct count`.
    fn reps<const N: usize, const D: usize, const A: usize>(brain: &Brain<N, D, A>, count: &[u32; A]) -> [f32; A] {
        let mut r = [0.0f32; A];
        for (a, v) in r.iter_mut().enumerate() { if count[a] > 0 { *v = brain.pulls(a) as f32 / count[a] as f32; } }
        r
    }

    /// Refit the model exactly with the current inputs, as if they had been there from the start:
    /// reset, then every distinct stored (situation, action) with the weight the replay gave it;
    /// base senses and free slots that are out of the model are removed again.
    pub fn refit<const N: usize, const D: usize, const A: usize>(&self, brain: &mut Brain<N, D, A>) {
        if B + G != D { return; }
        let (slots, m, count) = Self::distinct(brain);
        let reps = Self::reps(brain, &count);
        brain.reset_model();
        for &s in slots[..m].iter() {
            if let Some(e) = brain.episode(s as usize).copied() {
                brain.update_model_n(&e.key, e.action, e.reward, ((reps[e.action as usize] + 0.5) as u32).max(1));
            }
        }
        for i in 0..B { if !self.base_on[i] { brain.drop_input(i); } }
        for g in 0..G { if g >= self.grown || self.feats[g] == Feat::Off { brain.drop_input(B + g); } }
    }

    /// Growth slots usable now: the tier's capacity plus one per base sense compression removed
    /// (the model's input budget is `B + 2^tier`; compression frees budget for growth).
    pub fn budget(&self) -> usize { if self.tiered { (self.limit + self.dropped as usize).min(G) } else { G } }

    fn free_slot(&self) -> Option<usize> {
        if self.active_senses() >= self.budget() { return None; }
        (0..G).find(|&s| s >= self.grown || self.feats[s] == Feat::Off)
    }

    /// The grammar, enumerated deterministically: Prod(i <= j), Step(i, t), Gt(i != j) over
    /// non-constant base senses (as [`crate::brain_grow::Grower`]).
    pub fn candidate(&self, idx: usize) -> Option<Feat> {
        let mut k = 0usize;
        for i in 0..B { for j in i..B { if self.constant[i] || self.constant[j] { continue; } if k == idx { return Some(Feat::Prod(i as u8, j as u8)); } k += 1; } }
        for i in 0..B { if self.constant[i] { continue; } for t in STEPS { if k == idx { return Some(Feat::Step(i as u8, t)); } k += 1; } }
        for i in 0..B { for j in 0..B { if i == j || self.constant[i] || self.constant[j] { continue; } if k == idx { return Some(Feat::Gt(i as u8, j as u8)); } k += 1; } }
        None
    }

    /// Leave-one-situation-out error of the model (PRESS, mean over the distinct stored
    /// (situation, action) pairs `slots`), optionally refitted exactly without model input `without`.
    fn press_on<const N: usize, const D: usize, const A: usize>(brain: &Brain<N, D, A>, slots: &[u16], reps: &[f32; A], without: Option<usize>) -> f32 {
        let mut sum = 0.0f32;
        for &s in slots {
            let Some(e) = brain.episode(s as usize) else { continue };
            let r = brain.loo_residual(e.action, &e.key, e.reward, reps[e.action as usize], without);
            sum += r * r;
        }
        sum / slots.len().max(1) as f32
    }

    /// The held-out error on the replay ([`Cycle2::press_on`] over memory); NaN when memory holds
    /// fewer than `4 A` distinct (situation, action) pairs.
    pub fn press<const N: usize, const D: usize, const A: usize>(&self, brain: &Brain<N, D, A>, without: Option<usize>) -> f32 {
        let (slots, m, count) = Self::distinct(brain);
        if m < 4 * A { return f32::NAN; }
        Self::press_on(brain, &slots[..m], &Self::reps(brain, &count), without)
    }

    /// One growth check (normally run by [`Cycle2::learn`]), Hebbian: the candidate that fires
    /// together with the model's error. For every stored episode the model-only residual (reward
    /// minus the linear model's prediction) is computed; a check runs only if at least
    /// `min_erring` of them are still off by `erring_eps` or more (nothing to explain, nothing to
    /// grow). Every candidate not already grown and not recently removed by compression is scored
    /// by the sum over actions of its squared correlation with the residual; the best one is
    /// adopted into a free slot of the budget and the model is refitted. No held-out gate:
    /// compression decides on held-out error what stays.
    pub fn grow<const N: usize, const D: usize, const A: usize>(&mut self, brain: &mut Brain<N, D, A>) -> Option<Feat> {
        if B + G != D { return None; }
        let slot = self.free_slot()?;
        let mut res = [0.0f32; N];
        let mut erring = 0usize;
        for (s, r) in res.iter_mut().enumerate() {
            if let Some(e) = brain.episode(s) {
                *r = e.reward - brain.predict(e.action, &e.key);
                if *r >= self.erring_eps || *r <= -self.erring_eps { erring += 1; }
            }
        }
        if erring < self.min_erring.max(1) { return None; }
        let mut best: Option<(Feat, f32)> = None;
        let mut idx = 0usize;
        while let Some(f) = self.candidate(idx) {
            idx += 1;
            if self.feats[..self.grown].contains(&f) || self.tabu[..self.n_tabu.min(TABU)].contains(&f) { continue; }
            // per action: n, sum f, sum f^2, sum r, sum r^2, sum f r
            let mut st = [[0.0f32; 6]; A];
            for (s, &r) in res.iter().enumerate() {
                let Some(e) = brain.episode(s) else { continue };
                let fv = f.eval(&e.key[..B]);
                let a = &mut st[e.action as usize];
                a[0] += 1.0; a[1] += fv; a[2] += fv * fv; a[3] += r; a[4] += r * r; a[5] += fv * r;
            }
            let mut score = 0.0f32;
            for s in st.iter() {
                let n = s[0];
                if n < 2.0 { continue; }
                let (vf, ve, c) = (s[2] - s[1] * s[1] / n, s[4] - s[3] * s[3] / n, s[5] - s[1] * s[3] / n);
                if vf > 1e-9 && ve > 1e-12 { score += c * c / (vf * ve); }
            }
            if score > 0.0 && best.map(|b| score > b.1).unwrap_or(true) { best = Some((f, score)); }
        }
        let (f, _) = best?;
        self.feats[slot] = f;
        if slot >= self.grown { self.grown = slot + 1; }
        brain.open_input(B + slot);
        brain.rekey(|key| { let mut y = *key; y[B + slot] = f.eval(&key[..B]); y });
        self.refit(brain);
        self.adopted += 1;
        Some(f)
    }

    /// One compression pass: backward elimination of model inputs (non-constant base senses and
    /// grown senses) by held-out error on the replay: the input whose exact removal gives the lowest
    /// [`Cycle2::press`] is removed if that is at most `(1 + drop_tol)` of the PRESS with it
    /// (repeated up to `max_drop` times). A removed grown sense frees its slot, and is not regrown
    /// until the next tier opens.
    pub fn compress<const N: usize, const D: usize, const A: usize>(&mut self, brain: &mut Brain<N, D, A>) {
        if B + G != D { return; }
        for _ in 0..self.max_drop {
            let (slots, m, count) = Self::distinct(brain);
            if m < 4 * A { break; }
            let reps = Self::reps(brain, &count);
            let p0 = Self::press_on(brain, &slots[..m], &reps, None);
            let mut best: Option<(usize, f32)> = None;
            for j in 0..D {
                let live = if j < B { self.base_on[j] && !self.constant[j] } else { j - B < self.grown && self.feats[j - B] != Feat::Off };
                if !live { continue; }
                let p = Self::press_on(brain, &slots[..m], &reps, Some(j));
                if best.map(|b| p < b.1).unwrap_or(true) { best = Some((j, p)); }
            }
            let Some((j, p)) = best else { break };
            if !(p <= (1.0 + self.drop_tol) * p0) { break; }
            brain.drop_input(j);
            if j < B {
                self.base_on[j] = false;
                self.dropped += 1;
            } else {
                let s = j - B;
                self.tabu[self.n_tabu % TABU] = self.feats[s];
                self.n_tabu += 1;
                self.feats[s] = Feat::Off;
                brain.rekey(|key| { let mut y = *key; y[j] = 0.0; y });
                self.pruned += 1;
            }
        }
    }
}

impl<const B: usize, const G: usize> Default for Cycle2<B, G> {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng { fn f(&mut self) -> f32 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; ((self.0 >> 40) as f32) / (1u64 << 24) as f32 } }

    /// A fixed set of situations [x0, x1, 1], x in {-1, 1} + noise-free jitter, and a reward rule.
    fn set(seed: u64, n: usize) -> std::vec::Vec<[f32; 3]> {
        let mut r = Rng(seed);
        (0..n).map(|_| [if r.f() < 0.5 { -1.0 } else { 1.0 } * (0.5 + r.f()), if r.f() < 0.5 { -1.0 } else { 1.0 } * (0.5 + r.f()), 1.0]).collect()
    }

    /// Replay `train` for `epochs` (every action's outcome, full information).
    fn replay<const N: usize, const D: usize>(cy: &mut Cycle2<3, 2>, br: &mut Brain<N, D, 2>, train: &[[f32; 3]], epochs: usize, reward: fn(&[f32; 3], u8) -> f32) -> usize {
        let mut events = 0;
        for _ in 0..epochs { for b in train { for a in 0..2u8 { let x: [f32; D] = cy.situation(b); if cy.learn(br, &x, a, reward(b, a)).is_some() { events += 1; } } } }
        events
    }

    fn product(b: &[f32; 3], a: u8) -> f32 { if a == 1 { b[0] * b[1] } else { 0.0 } }
    fn linear(b: &[f32; 3], a: u8) -> f32 { if a == 1 { 0.6 * b[0] } else { -0.6 * b[0] } }

    #[test]
    fn held_out_outcomes_are_never_learned_and_the_control_never_compresses() {
        let (train, val) = (set(1, 12), set(2, 4));
        for compress in [true, false] {
            let mut cy: Cycle2<3, 2> = if compress { Cycle2::new() } else { Cycle2::without_compression() };
            cy.constant[2] = true;
            for b in val.iter() { for a in 0..2u8 { cy.hold_out(b, a, product(b, a)); } }
            let mut br: Brain<64, 5, 2> = Brain::new(0.0, 0.0);
            replay(&mut cy, &mut br, &train, 40, product);
            assert_eq!(br.clock() as usize, 40 * 12 * 2, "only learned outcomes reach the brain");
            for s in 0..64 { if let Some(e) = br.episode(s) { assert!(val.iter().all(|v| e.key[..3] != v[..]), "a held-out situation is never stored"); } }
            if !compress { assert_eq!((cy.forgotten, cy.dropped, cy.pruned), (0, 0, 0)); }
        }
    }

    #[test]
    fn hebbian_growth_finds_the_interaction_the_world_needs() {
        let train = set(3, 16);
        let mut cy: Cycle2<3, 2> = Cycle2::new();
        cy.constant[2] = true;
        cy.grow_every = u32::MAX; // growth only when asked below
        cy.min_erring = 8; // 2 actions here (the default suits 7)
        let mut br: Brain<64, 5, 2> = Brain::new(0.0, 0.0);
        replay(&mut cy, &mut br, &train, 3, product);
        assert_eq!(cy.grow(&mut br), Some(Feat::Prod(0, 1)));
        let x: [f32; 5] = cy.situation(&train[0]);
        assert!((br.predict(1, &x) - product(&train[0], 1)).abs() < 0.05, "the refit model uses the new sense at once");
    }

    #[test]
    fn compression_removes_an_input_that_does_not_pay_on_held_out_data() {
        // the reward is linear in x0; x1 carries nothing: PRESS without x1 is lower
        let train = set(4, 12);
        let mut cy: Cycle2<3, 2> = Cycle2::new();
        cy.constant[2] = true;
        cy.grow_every = u32::MAX;
        let mut br: Brain<64, 5, 2> = Brain::new(0.0, 0.0);
        replay(&mut cy, &mut br, &train, 5, linear);
        let p = cy.press(&br, None);
        assert!(cy.press(&br, Some(0)) > 10.0 * p, "x0 pays");
        cy.compress(&mut br);
        assert!(cy.base_on(0) && !cy.base_on(1) && cy.base_on(2), "x1 removed, x0 and the bias kept");
        assert_eq!(cy.budget(), 2, "the removed base sense freed a growth slot (tier 0: 1 + 1)");
    }

    #[test]
    fn grokking_event_needs_a_held_out_drop() {
        // The detector is armed at the end of the first compress phase; growth is switched on only
        // after that, so the product (the only way to generalize) arrives after arming: a held-out
        // drop fires an event. Without growth the training set is still fitted (memory), and
        // nothing fires: memorization is not grokking.
        let (train, val) = (set(5, 16), set(6, 6));
        for grow in [true, false] {
            let mut cy: Cycle2<3, 2> = Cycle2::new();
            cy.constant[2] = true;
            cy.min_erring = 8;
            cy.grow_max = 500;
            cy.grow_every = u32::MAX;
            for b in val.iter() { for a in 0..2u8 { cy.hold_out(b, a, product(b, a)); } }
            let mut br: Brain<128, 5, 2> = Brain::new(0.0, 0.0);
            let mut events = replay(&mut cy, &mut br, &train, 30, product);
            assert!(cy.compressed_once() && events == 0);
            if grow { cy.grow_every = 100; }
            events += replay(&mut cy, &mut br, &train, 30, product);
            if grow {
                assert_eq!(cy.grown(), &[Feat::Prod(0, 1)]);
                assert!(events >= 1 && cy.events()[0].v <= 0.8 * cy.events()[0].v_ref, "a held-out drop fires");
            } else {
                assert_eq!(events, 0, "memorizing the training set does not fire");
            }
        }
    }


    #[test]
    fn deterministic_and_the_grow_only_arm_has_every_slot() {
        let train = set(7, 10);
        let go = || { let mut cy: Cycle2<3, 2> = Cycle2::new(); cy.constant[2] = true; let mut br: Brain<64, 5, 2> = Brain::new(0.0, 0.0);
            for b in set(8, 3).iter() { cy.hold_out(b, 1, product(b, 1)); }
            replay(&mut cy, &mut br, &train, 30, product); (cy.grown().to_vec(), cy.adopted, cy.dropped, cy.pruned, cy.forgotten, br.clock()) };
        assert_eq!(go(), go());
        let g: Cycle2<3, 2> = Cycle2::grow_only();
        assert_eq!((g.budget(), g.compress_on, g.tiered), (2, false, false));
    }
}
