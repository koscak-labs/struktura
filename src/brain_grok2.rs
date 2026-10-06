//! Grokking falsifier v2: the world, thresholds and rule of [`crate::brain_grok`], re-run with the
//! growth cycle v2 ([`crate::brain_cycle2`]) on fresh seeds.
//!
//! Thesis under test (Phil): the native brain "has to grow and compress and compress, reach a
//! grokking event, and then it can grow to become a bigger model". v1 failed 0/5 on both encodings
//! because the cycle's grokking detector watched the memorized training stream and its growth
//! search proposed junk (see [`crate::brain_grok`]). v2 fixes both and keeps everything else.
//!
//! ## Pre-registration (written and committed before any fresh seed was run)
//!
//! - **World and data (unchanged).** `(a + b) mod 7`; one-hot senses (primary) or the Fourier
//!   re-encoding (secondary); the seed's fixed split ([`crate::brain_grok::split`]): 20 training
//!   pairs, 29 test pairs; 300 replay epochs, shuffled each epoch; full information (every
//!   action's outcome of a replayed pair is learned).
//! - **Validation slice (new).** [`N_VAL`] = 4 of the 20 training pairs, chosen by the seed
//!   ([`carve`]), are handed to the cycle's held-out probe ([`Cycle2::hold_out`], all 7 actions'
//!   outcomes) and never learned. The brain learns the other [`N_FIT`] = 16. The 29 test pairs
//!   are only measured, after every epoch; nothing reads them (no test-set accuracy drives anything).
//! - **Measured per epoch, per arm**: "train" = accuracy of the brain's own decision
//!   ([`Brain::decide`], exploration off) on the 16 learned pairs; "held-out" = on the 29 test
//!   pairs (the same test pairs as v1 for that seed); validation accuracy and sizes as diagnostics.
//! - **Arm A (compression on)**: [`Cycle2::new`] with its defaults (`B + 16` situation, bias
//!   marked constant): tiers 1, 2, 4, ... growth slots, input budget `B + 2^tier` with every base
//!   sense removed by compression freeing one growth slot; Hebbian growth every 250 learned
//!   outcomes (no held-out gate; only while >= 28 stored episodes are off by >= 0.1); compression
//!   = one stored copy per replayed example + backward elimination of model inputs by exact
//!   leave-one-situation-out error (PRESS; `drop_tol` 0, at most 4 per pass, a pass every 100
//!   steps of a 300-step compress phase); grokking event = validation error (brain's estimate,
//!   probed every 100 learned outcomes) at or below 0.8 of its reference for 3 consecutive probes;
//!   the next tier opens only after a grokking event (v1's gate). Grow phases end after 2 checks
//!   without adoption or 1500 steps.
//! - **Arm B (no-compression control)**: [`Cycle2::without_compression`]: the identical cycle
//!   (tiers, budget rule, validation probe, grokking gate, Hebbian growth, refit, phase lengths)
//!   with compression off (no copy forgetting, no input removal: compress phases are pauses).
//!   Same seed, split, validation slice, replay order and budget. A pass is therefore
//!   attributable to compression.
//! - **Arm C (diagnostic, NOT part of the verdict)**: [`Cycle2::grow_only`]: the same Hebbian
//!   growth with all 16 slots open from the start, no phases, no gate, no compression.
//! - **Rule (unchanged, v1's code and constants)**: a jump in an arm is
//!   [`crate::brain_grok::find_jump`] on its (train, held-out) curves with `J = 0.30`, `W = 10`,
//!   `E = 10`, train `>= 0.95`; a seed passes iff A has a jump, B has none, and B ends (mean of the
//!   last 10 epochs) at least `J / 2` below A. **PASS for an encoding iff >= 4 of the 5 fresh
//!   seeds [`FRESH_SEEDS`] pass; the thesis passes iff BOTH encodings pass** (one-hot primary).
//! - **Fresh seeds** [`FRESH_SEEDS`] = 6113, 6229, 6337, 6451, 6563: never run before this
//!   pre-registration was committed; disjoint from v1's seeds (5101..5505) and the dev seeds.
//! - **Analysis predictions** (pre-registered): (1) one-hot: all three arms end at or below chance
//!   + 0.10 held-out on all 5 seeds (v1's a-priori argument is unchanged: the depth-1 grammar over
//!   one-hot senses cannot link a cell to its anti-diagonal), so the one-hot falsifier FAILS and
//!   with it the thesis; (2) Fourier: on the 16 dev seeds (below) A jumped on 8 and passed on 7, B
//!   and C never jumped; with a per-seed pass rate near 0.44, `P(>= 4 of 5)` is about 0.12, so a
//!   Fourier FAIL is the likely outcome too, with A ahead of B on most seeds.
//!
//! ## Development (before the pre-registration; dev seeds only)
//!
//! The cycle's knobs and design were set on [`DEV_SEEDS`] (9001..9016) only; their test-pair curves
//! were looked at during development (the cycle never reads them). Final dev result, Fourier:
//! A jump 8/16, pass 7/16, mean held-out at the end A 0.316, B 0.027, C 0.039; one-hot: no jump in
//! any arm, held-out A 0.003, B 0.020, C 0.006. On the dev seeds A's jumps mostly came at tier 1
//! with no validation grokking event: removing the 12 non-constant base senses (which overfit: the
//! validation error with them is about 0.6, with the bias alone about 0.13) freed 12 growth slots,
//! and Hebbian growth found enough of the 12 same-frequency products; B keeps every base sense, its
//! validation error never drops and it stays at 2 senses.
//!
//! ## RESULT (added after the one run; rule, thresholds and seeds unchanged)
//!
//! **FAIL on both encodings, as predicted: Fourier 2 of 5 fresh seeds pass, one-hot 0 of 5. The
//! thesis fails its pre-registered test.**
//!
//! Fourier (held-out = test accuracy, mean of the last 10 epochs; chance 0.143; sizes = memories /
//! grown senses, base = base senses still in the model; train is 1.000 in every arm from epoch 1):
//!
//! | seed | A jump (epoch) | held-out A / B / C | A size plateau -> jump start -> end, base | B end, base | groks A / B | pass |
//! |------|----------------|--------------------|-------------------------------------------|-------------|-------------|------|
//! | 6113 | none (+31 pts over epochs 9..21, slower than W) | 0.345 / 0.034 / 0.034 | 112/0 -> 112/9, 13 -> 1 | 256/2, 13 | 0 / 0 | fail |
//! | 6229 | 35 | 0.345 / 0.048 / 0.000 | 112/0 -> 112/3 -> 112/7, 1 | 256/2, 13 | 2 / 0 | PASS |
//! | 6337 | none | 0.207 / 0.028 / 0.034 | 112/0 -> 112/10, 13 -> 2 | 256/2, 13 | 0 / 0 | fail |
//! | 6451 | 19 | 0.379 / 0.034 / 0.034 | 112/0 -> 112/2 -> 112/5, 1 | 256/2, 13 | 1 / 0 | PASS |
//! | 6563 | none | 0.103 / 0.000 / 0.034 | 112/0 -> 112/11, 13 -> 1 | 256/2, 13 | 0 / 0 | fail |
//!
//! One-hot: no jump in any arm on any seed; held-out A 0.000 on 5/5, B 0.000..0.069, C
//! 0.000..0.034. The pre-registered analysis prediction holds (every arm <= chance + 0.10, 5/5).
//!
//! What the numbers say:
//! - **B never jumps** (0/10 runs): it keeps all base senses, its validation error never drops,
//!   no grokking event fires, so its gate stays shut at 2 senses. **C** (grow-only, every slot
//!   open) never generalizes either (<= 0.034): with the base senses in, the model fits the 16
//!   pairs after 3 senses (5/5 seeds) and growth stops.
//! - **A ends above B on 5/5 Fourier seeds** (+0.10 to +0.35), and the cause is compression:
//!   removing 11-12 base senses (they overfit) frees growth budget, and Hebbian growth then finds
//!   some of the 12 same-frequency products. Validation grokking events are rare (0-2 per run;
//!   A ends at tier 1-3): the gain does not come from a sequence of gated tiers.
//! - **Why it still fails**: A ends at 0.10-0.38, far from the 1.000 the right 12 products give
//!   (v1's post-hoc ridge), with 5-11 senses mixing true products and junk: on 16 of 49 pairs a
//!   single product does not pay on its own, so neither Hebbian growth nor PRESS elimination
//!   reliably assembles the full set; and when A does rise, the rise is not reliably sudden
//!   (6113: +31 pts over 12 epochs is not a jump under W = 10).
//!
//! no_std, no heap (the curves are fixed arrays), deterministic for a seed, CPU only.

use crate::brain::Brain;
use crate::brain_cycle2::Cycle2;
use crate::brain_grok::{find_jump, first_plateau, fourier, onehot, split, Encoding, Size, B_FOURIER, B_ONEHOT,
    END, EPOCHS, J, MEM, N_HELD, N_TRAIN, P, PAIRS, PASS_SEEDS, SLOTS};
use crate::brain_grow::Feat;

/// Validation pairs carved from the [`N_TRAIN`] training pairs (never learned; never test pairs).
pub const N_VAL: usize = 4;
/// Training pairs the brain learns.
pub const N_FIT: usize = N_TRAIN - N_VAL;
/// Pre-registered fresh seeds (never run before the pre-registration was committed; disjoint
/// from the v1 seeds and the dev seeds).
pub const FRESH_SEEDS: [u64; 5] = [6113, 6229, 6337, 6451, 6563];
/// Seeds used while developing cycle v2 (its knobs were set on these only).
pub const DEV_SEEDS: [u64; 16] = [9001, 9002, 9003, 9004, 9005, 9006, 9007, 9008, 9009, 9010, 9011, 9012, 9013, 9014, 9015, 9016];

/// Which arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arm {
    /// Cycle v2 with compression.
    A,
    /// The same cycle v2 (tiers, validation probe, gate, growth), compression off.
    B,
    /// Diagnostic, not in the verdict: the same growth search, every slot open, no cycle, no compression.
    C,
}

/// One arm's measurements after one epoch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArmEpoch {
    /// Accuracy of the brain's own decision on the [`N_FIT`] learned pairs.
    pub train: f32,
    /// On the [`N_VAL`] validation pairs (the cycle's probe; diagnostic here).
    pub val: f32,
    /// On the [`N_HELD`] test pairs (recorded only; drives nothing).
    pub held: f32,
    /// Test accuracy of the linear model alone (diagnostic).
    pub model_held: f32,
    pub mem: u16,
    pub senses: u8,
    /// Base senses still in the model.
    pub base: u8,
    pub tier: u8,
    pub groks: u8,
    /// Senses adopted, grown senses pruned and base senses removed so far (diagnostic).
    pub adopted: u16,
    pub pruned: u16,
    pub dropped: u8,
    /// The cycle's validation error at its latest probe (diagnostic).
    pub val_err: f32,
}

impl ArmEpoch {
    pub const ZERO: Self = ArmEpoch { train: 0.0, val: 0.0, held: 0.0, model_held: 0.0, mem: 0, senses: 0, base: 0, tier: 0, groks: 0, adopted: 0, pruned: 0, dropped: 0, val_err: 0.0 };
}

/// All arms after one epoch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Epoch2 {
    pub a: ArmEpoch,
    pub b: ArmEpoch,
    pub c: ArmEpoch,
}

impl Epoch2 {
    pub const ZERO: Self = Epoch2 { a: ArmEpoch::ZERO, b: ArmEpoch::ZERO, c: ArmEpoch::ZERO };
    pub fn arm(&self, arm: Arm) -> &ArmEpoch { match arm { Arm::A => &self.a, Arm::B => &self.b, Arm::C => &self.c } }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn below(&mut self, n: usize) -> usize { (self.next() % n as u64) as usize }
}

/// The validation slice for a seed: `true` for [`N_VAL`] of the training pairs of [`split`].
pub fn carve(seed: u64) -> [bool; PAIRS] {
    let train = split(seed);
    let mut idx = [0u8; N_TRAIN];
    let mut n = 0;
    for (k, &t) in train.iter().enumerate() { if t && n < N_TRAIN { idx[n] = k as u8; n += 1; } }
    let mut r = Rng(seed ^ 0x7A11_DA7E_0000_0001);
    for i in (1..N_TRAIN).rev() { let j = r.below(i + 1); idx.swap(i, j); }
    let mut val = [false; PAIRS];
    for &k in idx[..N_VAL].iter() { val[k as usize] = true; }
    val
}


fn new_cycle<const B: usize>(arm: Arm) -> Cycle2<B, SLOTS> {
    let mut cy: Cycle2<B, SLOTS> = match arm { Arm::A => Cycle2::new(), Arm::B => Cycle2::without_compression(), Arm::C => Cycle2::grow_only() };
    cy.constant[B - 1] = true;
    cy
}

/// Knob overrides for every arm's cycle (dev tuning only; the falsifier uses [`Knobs::NONE`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Knobs {
    pub grok_drop: Option<f32>,
    pub grok_hold: Option<u8>,
    pub min_erring: Option<usize>,
    pub drop_tol: Option<f32>,
    pub max_drop: Option<u8>,
    pub dedup: Option<bool>,
    pub grow_every: Option<u32>,
    pub probe_every: Option<u32>,
}

impl Knobs {
    /// No overrides: the pre-registered settings (the cycle's defaults).
    pub const NONE: Knobs = Knobs { grok_drop: None, grok_hold: None, min_erring: None, drop_tol: None, max_drop: None,
        dedup: None, grow_every: None, probe_every: None };

    fn apply<const B: usize>(&self, cy: &mut Cycle2<B, SLOTS>) {
        if let Some(v) = self.grok_drop { cy.grok_drop = v; }
        if let Some(v) = self.grok_hold { cy.grok_hold = v; }
        if let Some(v) = self.min_erring { cy.min_erring = v; }
        if let Some(v) = self.drop_tol { cy.drop_tol = v; }
        if let Some(v) = self.max_drop { cy.max_drop = v; }
        if let Some(v) = self.dedup { cy.dedup = v; }
        if let Some(v) = self.grow_every { cy.grow_every = v; }
        if let Some(v) = self.probe_every { cy.probe_every = v; }
    }
}

/// Run one arm for `epochs` epochs, writing its column of `out`. Returns the grown senses at the end.
fn run_arm<const B: usize, const D: usize>(seed: u64, enc: fn(usize, usize) -> [f32; B], arm: Arm, knobs: &Knobs,
    epochs: usize, out: &mut [Epoch2; EPOCHS]) -> [Feat; SLOTS] {
    let (train, val) = (split(seed), carve(seed));
    let mut br: Brain<MEM, D, P> = Brain::new(0.0, 0.0);
    let mut cy: Cycle2<B, SLOTS> = new_cycle(arm);
    knobs.apply(&mut cy);
    // the validation slice: every action's outcome of the carved pairs, held out (never learned)
    for k in 0..PAIRS {
        if !val[k] { continue; }
        let (a, b) = (k / P, k % P);
        for c in 0..P { cy.hold_out(&enc(a, b), c as u8, if c == (a + b) % P { 1.0 } else { 0.0 }); }
    }
    let mut order = [0u8; N_FIT];
    let mut n = 0;
    for k in 0..PAIRS { if train[k] && !val[k] && n < N_FIT { order[n] = k as u8; n += 1; } }
    let mut rng = Rng(seed ^ 0x0DE5_0DE5);
    for row in out.iter_mut().take(epochs.min(EPOCHS)) {
        for i in (1..N_FIT).rev() { let j = rng.below(i + 1); order.swap(i, j); }
        for &k in order.iter() {
            let (a, b) = (k as usize / P, k as usize % P);
            let base = enc(a, b);
            for c in 0..P {
                // recomputed per outcome: a sense may have been grown (or pruned) by the previous one
                let x: [f32; D] = cy.situation(&base);
                cy.learn(&mut br, &x, c as u8, if c == (a + b) % P { 1.0 } else { 0.0 });
            }
        }
        let (mut tr, mut va, mut ho, mut mo) = (0usize, 0usize, 0usize, 0usize);
        for k in 0..PAIRS {
            let (a, b) = (k / P, k % P);
            let x: [f32; D] = cy.situation(&enc(a, b));
            let right = br.decide(&x, &[true; P], 0).action as usize == (a + b) % P;
            if !train[k] {
                if right { ho += 1; }
                let mut best = 0usize;
                for c in 1..P { if br.predict(c as u8, &x) > br.predict(best as u8, &x) { best = c; } }
                if best == (a + b) % P { mo += 1; }
            } else if val[k] { if right { va += 1; } } else if right { tr += 1; }
        }
        let e = ArmEpoch {
            train: tr as f32 / N_FIT as f32, val: va as f32 / N_VAL as f32, held: ho as f32 / N_HELD as f32, model_held: mo as f32 / N_HELD as f32,
            mem: br.in_use() as u16, senses: cy.active_senses() as u8, base: cy.base_inputs() as u8, tier: cy.tier(), groks: cy.events().len() as u8,
            adopted: cy.adopted as u16, pruned: cy.pruned as u16, dropped: cy.dropped as u8, val_err: cy.last_val(),
        };
        match arm { Arm::A => row.a = e, Arm::B => row.b = e, Arm::C => row.c = e }
    }
    let mut grown = [Feat::Off; SLOTS];
    for (o, g) in grown.iter_mut().zip(cy.grown()) { *o = *g; }
    grown
}

/// All arms' per-epoch curves for one seed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GrokRun2 {
    pub seed: u64,
    pub encoding: Encoding,
    pub epochs: usize,
    pub curve: [Epoch2; EPOCHS],
    /// Grown senses at the end (slot order; pruned slots are `Off`), arms A, B, C.
    pub grown: [[Feat; SLOTS]; 3],
}

/// Run every arm on one seed for `epochs` (at most [`EPOCHS`]) epochs.
pub fn run_for(seed: u64, enc: Encoding, epochs: usize, knobs: &Knobs) -> GrokRun2 {
    let epochs = epochs.min(EPOCHS);
    let mut curve = [Epoch2::ZERO; EPOCHS];
    let mut grown = [[Feat::Off; SLOTS]; 3];
    for (i, arm) in [Arm::A, Arm::B, Arm::C].into_iter().enumerate() {
        grown[i] = match enc {
            Encoding::OneHot => run_arm::<B_ONEHOT, { B_ONEHOT + SLOTS }>(seed, onehot, arm, knobs, epochs, &mut curve),
            Encoding::Fourier => run_arm::<B_FOURIER, { B_FOURIER + SLOTS }>(seed, fourier, arm, knobs, epochs, &mut curve),
        };
    }
    GrokRun2 { seed, encoding: enc, epochs, curve, grown }
}

/// Run every arm on one seed, full budget, pre-registered settings.
pub fn run(seed: u64, enc: Encoding) -> GrokRun2 { run_for(seed, enc, EPOCHS, &Knobs::NONE) }

/// One seed's verdict under the pre-registered rule (unchanged from [`crate::brain_grok`]).
/// Epochs are counted from 1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GrokTrial2 {
    pub seed: u64,
    pub encoding: Encoding,
    /// Arm A: first epoch of the train plateau before the jump (no jump: of the first plateau).
    pub plateau_epoch: Option<u16>,
    /// Arm A: epoch the jump completed.
    pub jump_epoch: Option<u16>,
    /// Arm B's jump (a pass requires none).
    pub jump_b: Option<u16>,
    /// Arm C's jump (diagnostic).
    pub jump_c: Option<u16>,
    /// Test accuracy at the end (mean of the last [`END`] epochs), arms A, B, C.
    pub heldout_a: f32,
    pub heldout_b: f32,
    pub heldout_c: f32,
    /// Train accuracy (learned pairs) at the end, arms A and B.
    pub train_a: f32,
    pub train_b: f32,
    /// Validation accuracy at the end, arms A and B (diagnostic).
    pub val_a: f32,
    pub val_b: f32,
    /// Arm A's size (memories / grown senses) at the plateau start, jump start, jump end (or final).
    pub size_plateau: Size,
    pub size_before: Size,
    pub size_after: Size,
    /// Arm A's base senses in the model at the jump start and end (or final).
    pub base_before: u8,
    pub base_after: u8,
    /// Arm B's final size and base senses.
    pub size_b: Size,
    pub base_b: u8,
    /// Final tier and grokking events (validation-driven), arms A and B.
    pub tier_a: u8,
    pub tier_b: u8,
    pub groks_a: u8,
    pub groks_b: u8,
    pub pass: bool,
}

impl GrokRun2 {
    fn column(&self, arm: Arm, f: impl Fn(&ArmEpoch) -> f32) -> [f32; EPOCHS] {
        let mut v = [0.0f32; EPOCHS];
        for (o, e) in v.iter_mut().zip(self.curve.iter()).take(self.epochs) { *o = f(e.arm(arm)); }
        v
    }

    fn at(&self, arm: Arm, epoch: usize) -> &ArmEpoch { self.curve[epoch.min(self.epochs.max(1) - 1)].arm(arm) }

    fn size(&self, arm: Arm, epoch: usize) -> Size { let e = self.at(arm, epoch); Size { memories: e.mem, senses: e.senses } }

    fn end(&self, col: &[f32]) -> f32 {
        let n = self.epochs;
        let k = END.min(n).max(1);
        col[n.saturating_sub(k)..n].iter().sum::<f32>() / k as f32
    }

    /// The pre-registered verdict for this seed.
    pub fn verdict(&self) -> GrokTrial2 {
        let n = self.epochs;
        let (ta, ha) = (self.column(Arm::A, |e| e.train), self.column(Arm::A, |e| e.held));
        let (tb, hb) = (self.column(Arm::B, |e| e.train), self.column(Arm::B, |e| e.held));
        let (tc, hc) = (self.column(Arm::C, |e| e.train), self.column(Arm::C, |e| e.held));
        let ja = find_jump(&ta[..n], &ha[..n]);
        let jb = find_jump(&tb[..n], &hb[..n]);
        let jc = find_jump(&tc[..n], &hc[..n]);
        let (heldout_a, heldout_b) = (self.end(&ha), self.end(&hb));
        let plateau = ja.map(|j| j.plateau).or_else(|| first_plateau(&ta[..n]));
        let size_plateau = self.size(Arm::A, plateau.unwrap_or(0));
        let last = n.saturating_sub(1);
        let (s0, s1) = match ja { Some(j) => (j.start, j.end), None => (plateau.unwrap_or(0), last) };
        let pass = ja.is_some() && jb.is_none() && heldout_b <= heldout_a - J / 2.0;
        GrokTrial2 {
            seed: self.seed, encoding: self.encoding,
            plateau_epoch: plateau.map(|p| p as u16 + 1), jump_epoch: ja.map(|j| j.end as u16 + 1),
            jump_b: jb.map(|j| j.end as u16 + 1), jump_c: jc.map(|j| j.end as u16 + 1),
            heldout_a, heldout_b, heldout_c: self.end(&hc),
            train_a: self.end(&ta), train_b: self.end(&tb),
            val_a: self.end(&self.column(Arm::A, |e| e.val)), val_b: self.end(&self.column(Arm::B, |e| e.val)),
            size_plateau, size_before: self.size(Arm::A, s0), size_after: self.size(Arm::A, s1),
            base_before: self.at(Arm::A, s0).base, base_after: self.at(Arm::A, s1).base,
            size_b: self.size(Arm::B, last), base_b: self.at(Arm::B, last).base,
            tier_a: self.at(Arm::A, last).tier, tier_b: self.at(Arm::B, last).tier,
            groks_a: self.at(Arm::A, last).groks, groks_b: self.at(Arm::B, last).groks, pass,
        }
    }
}

/// One seed of the v2 falsifier with a chosen encoding: every arm, full budget, verdict.
pub fn trial(seed: u64, enc: Encoding) -> GrokTrial2 { run(seed, enc).verdict() }

/// The pre-registered falsifier: every fresh seed, and whether at least [`PASS_SEEDS`] passed.
pub fn falsifier(enc: Encoding) -> ([GrokTrial2; 5], bool) {
    let t = FRESH_SEEDS.map(|s| trial(s, enc));
    let passed = t.iter().filter(|x| x.pass).count();
    (t, passed >= PASS_SEEDS)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn print_trial(t: &GrokTrial2) {
        std::println!("  {} seed {}: plateau {:?} jump A {:?} B {:?} (C {:?}) | held-out A {:.3} B {:.3} C {:.3} | train A {:.3} B {:.3} | val A {:.3} B {:.3} | A size {:?} -> {:?} -> {:?} base {} -> {} | B size {:?} base {} | tier A {} B {} | groks A {} B {} | pass {}",
            t.encoding.name(), t.seed, t.plateau_epoch, t.jump_epoch, t.jump_b, t.jump_c, t.heldout_a, t.heldout_b, t.heldout_c, t.train_a, t.train_b,
            t.val_a, t.val_b, t.size_plateau, t.size_before, t.size_after, t.base_before, t.base_after, t.size_b, t.base_b, t.tier_a, t.tier_b, t.groks_a, t.groks_b, t.pass);
    }

    /// Dev-only knob overrides from env (G2_*), used on [`DEV_SEEDS`] only.
    fn env_knobs() -> Knobs {
        let g = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok());
        Knobs { grok_drop: g("G2_DROP"), grok_hold: g("G2_HOLD").map(|v| v as u8), min_erring: g("G2_ERR").map(|v| v as usize), drop_tol: g("G2_TOL"),
            max_drop: g("G2_MAXDROP").map(|v| v as u8), dedup: g("G2_DEDUP").map(|v| v > 0.0), grow_every: g("G2_EVERY").map(|v| v as u32),
            probe_every: g("G2_PROBE").map(|v| v as u32) }
    }

    #[test]
    fn carve_is_inside_the_training_set_and_sized() {
        for s in [1u64, 2, 3, 9001] {
            let (t, v) = (split(s), carve(s));
            assert_eq!(v.iter().filter(|&&x| x).count(), N_VAL);
            assert!((0..PAIRS).all(|k| !v[k] || t[k]), "validation pairs come from the training pairs only");
            assert_eq!(v, carve(s));
        }
    }

    #[test]
    fn deterministic() {
        let (a, b) = (run_for(7, Encoding::Fourier, 3, &Knobs::NONE), run_for(7, Encoding::Fourier, 3, &Knobs::NONE));
        assert_eq!(a, b);
        assert!(a.curve[2].a.train > 0.0 && a.curve[2].b.train > 0.0, "it learns something");
    }

    /// Dev runs (knobs from env G2_*), DEV_SEEDS only: cargo test --release --lib brain_grok2::tests::dev -- --ignored --nocapture
    #[test]
    #[ignore]
    fn dev() {
        let k = env_knobs();
        std::println!("knobs {:?}", k);
        let enc = if std::env::var("G2_ONEHOT").is_ok() { Encoding::OneHot } else { Encoding::Fourier };
        let t: [GrokTrial2; 16] = std::thread::scope(|s| DEV_SEEDS.map(|seed| s.spawn(move || run_for(seed, enc, EPOCHS, &k).verdict())).map(|h| h.join().unwrap()));
        if std::env::var("G2_V").is_ok() { for x in t.iter() { print_trial(x); } }
        std::println!("A held-out per seed {:?} jumps {:?}", t.iter().map(|x| (x.heldout_a * 1000.0).round() / 1000.0).collect::<std::vec::Vec<_>>(), t.iter().map(|x| x.jump_epoch).collect::<std::vec::Vec<_>>());
        std::println!("DEV {}: A jump {}/16, B jump {}/16, C jump {}/16, pass {}/16; mean held-out A {:.3} B {:.3} C {:.3}", enc.name(),
            t.iter().filter(|x| x.jump_epoch.is_some()).count(), t.iter().filter(|x| x.jump_b.is_some()).count(), t.iter().filter(|x| x.jump_c.is_some()).count(),
            t.iter().filter(|x| x.pass).count(), t.iter().map(|x| x.heldout_a).sum::<f32>() / 16.0, t.iter().map(|x| x.heldout_b).sum::<f32>() / 16.0, t.iter().map(|x| x.heldout_c).sum::<f32>() / 16.0);
    }

    /// Dev curve of one dev seed (G2_SEED, default 9001; every G2_EVERY_EP epochs).
    #[test]
    #[ignore]
    fn dev_curve() {
        let seed = std::env::var("G2_SEED").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(9001);
        assert!(!FRESH_SEEDS.contains(&seed), "fresh seeds are run only by the falsifier");
        let every = std::env::var("G2_EVERY_EP").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(5);
        let enc = if std::env::var("G2_ONEHOT").is_ok() { Encoding::OneHot } else { Encoding::Fourier };
        let r = run_for(seed, enc, EPOCHS, &env_knobs());
        for (i, e) in r.curve[..r.epochs].iter().enumerate() {
            if (i + 1) % every != 0 && i != 0 { continue; }
            std::println!("{:>3} | A tr {:.2} va {:.2} ho {:.3} mh {:.3} mem {:>3} s {:>2} b {:>2} t {} g {} | B tr {:.2} va {:.2} ho {:.3} mh {:.3} s {:>2} t {} g {} ad {} pr {} | vA {:.4} vB {:.4} | C ho {:.3} s {:>2}",
                i + 1, e.a.train, e.a.val, e.a.held, e.a.model_held, e.a.mem, e.a.senses, e.a.base, e.a.tier, e.a.groks,
                e.b.train, e.b.val, e.b.held, e.b.model_held, e.b.senses, e.b.tier, e.b.groks, e.a.adopted, e.a.pruned, e.a.val_err, e.b.val_err, e.c.held, e.c.senses);
        }
        for (n, g) in ["A", "B", "C"].iter().zip(r.grown.iter()) {
            std::println!("{}: {:?}", n, g.iter().filter(|f| **f != Feat::Off).collect::<std::vec::Vec<_>>());
        }
        print_trial(&r.verdict());
    }

    /// Each encoding's 5 fresh-seed trials, computed once and shared by the tests that read them.
    fn fresh(enc: Encoding) -> &'static ([GrokTrial2; 5], bool) {
        use std::sync::OnceLock;
        static ONEHOT: OnceLock<([GrokTrial2; 5], bool)> = OnceLock::new();
        static FOURIER: OnceLock<([GrokTrial2; 5], bool)> = OnceLock::new();
        let cell = match enc { Encoding::OneHot => &ONEHOT, Encoding::Fourier => &FOURIER };
        cell.get_or_init(|| {
            // the seeds in parallel threads (each trial is deterministic, so this equals `falsifier(enc)`)
            let t: [GrokTrial2; 5] = std::thread::scope(|s| FRESH_SEEDS.map(|seed| s.spawn(move || trial(seed, enc))).map(|h| h.join().unwrap()));
            for x in t.iter() { print_trial(x); }
            let n = t.iter().filter(|x| x.pass).count();
            std::println!("{} falsifier v2: {}/5 seeds pass -> {}", enc.name(), n, if n >= PASS_SEEDS { "PASS" } else { "FAIL" });
            (t, n >= PASS_SEEDS)
        })
    }

    /// The pre-registration, pinned: seeds, thresholds, split sizes and every arm's settings.
    #[test]
    fn preregistration_is_pinned() {
        use crate::brain_grok as v1;
        assert_eq!(FRESH_SEEDS, [6113, 6229, 6337, 6451, 6563]);
        assert!(FRESH_SEEDS.iter().all(|s| !v1::FRESH_SEEDS.contains(s) && !DEV_SEEDS.contains(s)), "fresh seeds are new");
        assert_eq!((v1::J, v1::W, v1::E, v1::TRAIN_OK, v1::END, v1::PASS_SEEDS), (0.30, 10, 10, 0.95, 10, 4), "thresholds unchanged");
        assert_eq!((N_VAL, N_FIT, N_TRAIN, N_HELD, EPOCHS, SLOTS), (4, 16, 20, 29, 300, 16));
        assert_eq!(Knobs::NONE, Knobs::default());
        let settings = |c: &Cycle2<13, SLOTS>| ((c.grow_every, c.grow_patience, c.grow_max, c.compress_steps, c.compress_every, c.erring_eps, c.min_erring),
            (c.drop_tol, c.max_drop, c.dedup, c.probe_every, c.grok_drop, c.grok_hold));
        let (a, b, c): (Cycle2<13, SLOTS>, Cycle2<13, SLOTS>, Cycle2<13, SLOTS>) = (Cycle2::new(), Cycle2::without_compression(), Cycle2::grow_only());
        assert_eq!(settings(&a), ((250, 2, 1500, 300, 100, 0.1, 28), (0.0, 4, true, 100, 0.2, 3)));
        assert_eq!((settings(&b), settings(&c)), (settings(&a), settings(&a)), "the arms differ only in compression (and C in tiers)");
        assert_eq!((a.compress_on, a.tiered, b.compress_on, b.tiered, c.compress_on, c.tiered), (true, true, false, true, false, false));
        assert_eq!((a.budget(), b.budget(), c.budget()), (1, 1, SLOTS));
    }

    /// PRE-REGISTERED primary falsifier v2 (one-hot senses, 5 fresh seeds; criteria in the module docs).
    /// RESULT: FAILED, 0/5 seeds; no jump in any arm, held-out A 0.000 on every seed (see module docs).
    /// Run: cargo test --release --lib brain_grok2 -- --ignored --nocapture
    #[test]
    #[ignore = "FAILED as pre-registered: 0/5 seeds; one-hot held-out stays at or below chance in every arm (see module docs)"]
    fn falsifier_v2_onehot_groks_only_with_compression() {
        let (t, pass) = fresh(Encoding::OneHot);
        assert!(*pass, "one-hot: {}/5 seeds pass (need {})", t.iter().filter(|x| x.pass).count(), PASS_SEEDS);
    }

    /// PRE-REGISTERED secondary falsifier v2 (Fourier senses, same seeds and criteria).
    /// RESULT: FAILED, 2/5 seeds (6229, 6451); held-out A 0.103-0.379 vs B 0.000-0.048, B never jumps (see module docs).
    #[test]
    #[ignore = "FAILED as pre-registered: 2/5 seeds; A beats B on 5/5 but jumps on 2 (see module docs)"]
    fn falsifier_v2_fourier_groks_only_with_compression() {
        let (t, pass) = fresh(Encoding::Fourier);
        assert!(*pass, "fourier: {}/5 seeds pass (need {})", t.iter().filter(|x| x.pass).count(), PASS_SEEDS);
    }

    /// PRE-REGISTERED analysis prediction: with one-hot senses every arm ends at most chance + 0.10
    /// held-out on every fresh seed. RESULT: holds on 5/5 (A 0.000, B <= 0.069, C <= 0.034).
    #[test]
    fn analysis_prediction_v2_onehot_stays_at_chance() {
        let (t, _) = fresh(Encoding::OneHot);
        let bound = 1.0 / P as f32 + crate::brain_grok::CHANCE_MARGIN;
        for x in t.iter() {
            assert!(x.heldout_a <= bound && x.heldout_b <= bound && x.heldout_c <= bound, "seed {}: held-out A {:.3} B {:.3} C {:.3} > {:.3}", x.seed, x.heldout_a, x.heldout_b, x.heldout_c, bound);
        }
    }
}
