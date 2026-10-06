//! Grokking falsifier: modular addition, a fixed training set replayed for many epochs.
//!
//! Thesis under test (Phil): the native brain "has to grow and compress and compress, reach a
//! grokking event, and then it can grow to become a bigger model". The rover streams used so far
//! cannot show grokking (every memory earns its keep there). This is the classic grokking world.
//!
//! - **World**: `(a + b) mod P`, `P = 7`. Senses: one-hot(a) ++ one-hot(b) ++ 1 (primary), or the
//!   equivalent Fourier re-encoding (secondary, see below). The answer is the action (`P` actions).
//! - **Data**: a FIXED training set of 20 of the 49 pairs (40 %, chosen by the seed); the other 29
//!   are held out and never learned. Training replays the fixed set, shuffled every epoch, for 300
//!   epochs. After every epoch, train and held-out accuracy of the brain's own decision
//!   ([`Brain::decide`], exploration off, memory + model) are recorded, with the brain's size.
//! - **Supervision: full information**, through the brain's own API: for every replayed pair the
//!   brain learns the outcome of EVERY action (`reward 1` for the answer, `0` otherwise), as a
//!   bandit would after trying every arm. This is the supervised setting of the grokking literature
//!   (cross-entropy sees every class) and it takes exploration out as a confound: with bandit
//!   feedback a failure could be blamed on exploring, not on what the brain can represent. Memory
//!   therefore also stores negative outcomes; `P = 7 <= K = 8` neighbours, so one stored visit of a
//!   pair can be recalled for all actions at once.
//! - **Arm A (compression on)**: [`Cycle`] (grow in tiers of 1, 2, 4, 8, 16 senses; compress phases
//!   forget memories the model alone predicts and prune constant senses; the Page-Hinkley grokking
//!   detector gates the next tier) + [`Pruner`] (forget at birth an outcome the brain already
//!   expected within 0.1; held-out MDL test of grown senses every 250 outcomes; the disuse horizon
//!   is off because training makes no decisions to cite) + [`Sleep`] (defaults: replay 16 episodes
//!   into the model, free up to 16 absorbed ones, about every 64 outcomes). Other knobs: defaults.
//! - **Arm B (compression off)**: the identical brain and the same [`Grower`](crate::brain_grow::Grower) (same seed, grammar
//!   and 16 slots, all available from the start, as the grow-only arm of `brain_cycle`); no cycle,
//!   pruner or sleep. Same seed (same split, replay order, grower seed) and budget (300 epochs,
//!   42 000 learned outcomes).
//!
//! ## Pre-registration (written and committed before any result was seen)
//!
//! A **jump** in one arm: epochs `e0 < e1 <= e0 + W` with `held(e1) - held(e0) >= J`, train
//! accuracy `>= 0.95` at every epoch `e0 - E ..= e0` (a plateau of at least `E` epochs before the
//! jump starts), and sustained: `held >= held(e0) + J / 2` at every epoch `e1 .. e1 + W` (clipped
//! at the last epoch). The first such `e1` (and its latest `e0`) counts. `J = 0.30`, `W = 10`, `E = 10`.
//!
//! A seed **passes** iff (1) arm A has a jump, and (2) arm B has none and ends (mean held-out over
//! the last 10 epochs) at least `J / 2` below arm A. **PASS** iff at least 4 of the 5 fresh seeds
//! [`FRESH_SEEDS`] pass. Both encodings are pre-registered with the same thresholds and seeds; the
//! one-hot run is the primary (faithful) test, the Fourier run the secondary one.
//!
//! Sizes (reported, not part of PASS): arm A's memories in use and active grown senses at the
//! plateau start, at the jump start and at the jump end. "Compression precedes the jump" would
//! show as fewer memories at the jump start than at the plateau start.
//!
//! **Analysis prediction** (also pre-registered, from the analysis below): with one-hot senses both
//! arms end at or below chance plus 10 points (held-out `<= 1/7 + 0.10`) on all 5 seeds.
//!
//! ## A priori analysis (before running)
//!
//! One-hot senses, why the linear + kNN brain should NOT generalize, with or without growth:
//! 1. **Base senses carry nothing.** For every `a` and answer `c` exactly one `b` gives
//!    `a + b = c`: the label is independent of `a` alone and of `b` alone. Over the full table the
//!    best additive fit `u_c[a] + v_c[b] + w_c` of the reward is the constant `1/P`. (Also: the sum
//!    over all pairs of `s_{a+b+k}(a, b)` is the same for every shift `k`, so no additive scorer is
//!    right on every pair.) On a 40 % subset the additive fit is worse than nothing for held-out
//!    cells: for a held-out `(i, j)` with answer `c = i + j`, row `i` and column `j` have no
//!    training cell where `c` pays (the only one is `(i, j)` itself), while other rows do, so
//!    `u_c[i]` and `v_c[j]` are pushed below average: expected held-out accuracy below chance.
//! 2. **Memory recalls the wrong answers.** A held-out pair is at squared distance 2 from the
//!    training pairs in its row and column, and every one of them has a different answer; pairs
//!    with the same answer share neither (distance 4).
//! 3. **Growth builds a lookup table.** Over one-hot senses the grammar yields `Prod(a_i, b_j)` =
//!    `[a = i and b = j]` (one cell), `Gt(a_i, b_j)` = `[a = i and b != j]`, `Gt(b_j, a_i)`, copies of
//!    base senses or constant zeros. Each grown sense lives on one row, one column and one cell; a
//!    held-out cell's own indicator is 0 on all training data (its weight stays at the prior), and
//!    its row and column say "the right answer pays 0 here" (point 1). Nothing in the grammar links
//!    a cell to the other cells of its anti-diagonal `a + b = c`, which is where the answer lives.
//!
//! So the one-hot test is predicted to FAIL criterion 1 in both arms, whatever compression does.
//!
//! **Fourier senses** (`cos`, `sin` of `2 pi k a / P` and of `2 pi k b / P`, `k = 1..=3`, plus 1)
//! are an invertible linear map of the centred one-hots: the linear model over base senses is the
//! same function class, and all distinct values of `a` are equidistant, so memory sees the same
//! neighbours. Only the reach of the growth grammar differs: `Prod(cos_k a, cos_k b) -
//! Prod(sin_k a, sin_k b) = cos(w_k (a + b))`, and with the four products of one frequency
//! (`cc, ss, sc, cs`) a linear model per action represents `cos(w_k (a + b - c))`, whose argmax over
//! `c` is exactly `(a + b) mod P`. Representable with 4 grown senses (checked by a test). Whether the
//! grower finds them, whether the jump comes late (after a memorization plateau), and whether only
//! arm A has it are open. Prior: growth, not compression, carries any jump, and arm B can grow all
//! 16 senses from the start, so criterion 2 is likely to fail. This variant hands the brain the
//! cyclic structure a grokking network has to discover, so a PASS here would be weaker evidence.
//!
//! Note on the cycle's own "grokking events": its detector watches the prequential error of the
//! replayed TRAINING stream, so in this world it measures memorization, not held-out generalization.
//!
//! ## RESULT
//!
//! (added after the run; see the falsifier tests)
//!
//! no_std, no heap (the curves are fixed arrays), deterministic for a seed, CPU only.

use crate::brain::Brain;
use crate::brain_cycle::Cycle;
use crate::brain_prune::Pruner;
use crate::brain_sleep::Sleep;

/// Modulus (prime) and number of actions.
pub const P: usize = 7;
/// All `(a, b)` pairs.
pub const PAIRS: usize = P * P;
/// Training pairs (40 % of 49, rounded); the other pairs are held out.
pub const N_TRAIN: usize = 20;
/// Held-out pairs.
pub const N_HELD: usize = PAIRS - N_TRAIN;
/// Replay epochs per arm.
pub const EPOCHS: usize = 300;
/// Memory capacity (episodes).
pub const MEM: usize = 256;
/// Growth slots (grown senses).
pub const SLOTS: usize = 16;
/// Arm A: an outcome the brain already expected within this is forgotten at birth.
pub const ABSORB: f32 = 0.1;

/// Pre-registered: smallest held-out rise (fraction) that counts as a jump.
pub const J: f32 = 0.30;
/// Pre-registered: the rise must happen within this many epochs, and then hold for as many.
pub const W: usize = 10;
/// Pre-registered: train accuracy must have been at least [`TRAIN_OK`] for this many epochs first.
pub const E: usize = 10;
/// Pre-registered: train accuracy that counts as fitted.
pub const TRAIN_OK: f32 = 0.95;
/// Pre-registered: "ends" = mean over this many last epochs.
pub const END: usize = 10;
/// Pre-registered fresh seeds (never run before the falsifier).
pub const FRESH_SEEDS: [u64; 5] = [5101, 5202, 5303, 5404, 5505];
/// Pre-registered: seeds that must pass.
pub const PASS_SEEDS: usize = 4;
/// Pre-registered analysis prediction: one-hot held-out ends at most chance plus this.
pub const CHANCE_MARGIN: f32 = 0.10;

/// Base senses, one-hot: one-hot(a) ++ one-hot(b) ++ 1.
pub const B_ONEHOT: usize = 2 * P + 1;
/// Base senses, Fourier: (cos, sin)(2 pi k a / P) for k = 1..=(P-1)/2, the same for b, then 1.
pub const B_FOURIER: usize = 2 * (P - 1) + 1;

/// How `(a, b)` is presented to the brain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Encoding { OneHot, Fourier }

impl Encoding {
    pub fn name(&self) -> &'static str { match self { Encoding::OneHot => "onehot", Encoding::Fourier => "fourier" } }
}

/// One-hot senses of a pair.
pub fn onehot(a: usize, b: usize) -> [f32; B_ONEHOT] {
    let mut x = [0.0f32; B_ONEHOT];
    x[a % P] = 1.0;
    x[P + b % P] = 1.0;
    x[2 * P] = 1.0;
    x
}

/// Fourier senses of a pair (an invertible linear map of the centred one-hots, plus 1).
pub fn fourier(a: usize, b: usize) -> [f32; B_FOURIER] {
    let mut x = [0.0f32; B_FOURIER];
    let w = 2.0 * core::f64::consts::PI / P as f64;
    for k in 1..=(P - 1) / 2 {
        let i = 2 * (k - 1);
        let (pa, pb) = (((k * a) % P) as f64 * w, ((k * b) % P) as f64 * w);
        x[i] = libm::cos(pa) as f32;
        x[i + 1] = libm::sin(pa) as f32;
        x[P - 1 + i] = libm::cos(pb) as f32;
        x[P - 1 + i + 1] = libm::sin(pb) as f32;
    }
    x[2 * (P - 1)] = 1.0;
    x
}

/// One epoch of both arms' curves and sizes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Epoch {
    pub train_a: f32,
    pub held_a: f32,
    pub train_b: f32,
    pub held_b: f32,
    /// Memories in use.
    pub mem_a: u16,
    pub mem_b: u16,
    /// Active grown senses.
    pub senses_a: u8,
    pub senses_b: u8,
    /// Arm A: capacity tier and grokking events (cycle detector) so far.
    pub tier_a: u8,
    pub groks_a: u8,
}

impl Epoch {
    pub const ZERO: Self = Epoch { train_a: 0.0, held_a: 0.0, train_b: 0.0, held_b: 0.0, mem_a: 0, mem_b: 0, senses_a: 0, senses_b: 0, tier_a: 0, groks_a: 0 };
}

/// A model's size: memories in use and active grown senses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Size {
    pub memories: u16,
    pub senses: u8,
}

/// A detected jump (epochs, 0-based): plateau start, jump start `e0`, jump end `e1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Jump {
    pub plateau: usize,
    pub start: usize,
    pub end: usize,
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn below(&mut self, n: usize) -> usize { (self.next() % n as u64) as usize }
}

/// The fixed training set for a seed: `true` for the [`N_TRAIN`] training pairs (index `a * P + b`).
pub fn split(seed: u64) -> [bool; PAIRS] {
    let mut idx = [0u8; PAIRS];
    for (i, v) in idx.iter_mut().enumerate() { *v = i as u8; }
    let mut r = Rng(seed ^ 0x5B11_7000_0000_0001);
    for i in (1..PAIRS).rev() { let j = r.below(i + 1); idx.swap(i, j); }
    let mut train = [false; PAIRS];
    for &k in idx[..N_TRAIN].iter() { train[k as usize] = true; }
    train
}

/// Slot of the episode stored by the latest learn (None if it was not stored or already forgotten).
fn newest<const D: usize>(br: &Brain<MEM, D, P>) -> Option<usize> {
    let t = br.clock();
    (0..MEM).find(|&s| br.episode(s).map(|e| e.t == t).unwrap_or(false))
}

/// Run one arm for `epochs` epochs, writing its columns of `out`.
fn run_arm<const B: usize, const D: usize>(seed: u64, enc: fn(usize, usize) -> [f32; B], train: &[bool; PAIRS],
    compress: bool, epochs: usize, out: &mut [Epoch; EPOCHS]) {
    let mut br: Brain<MEM, D, P> = Brain::new(0.0, 0.0);
    let mut cy: Cycle<B, SLOTS> = Cycle::new(seed ^ 0xC1C1);
    cy.grower.constant[B - 1] = true;
    if !compress { cy.grower.set_limit(SLOTS); }
    let mut pr: Pruner<MEM> = Pruner::new(seed ^ 0x5EED);
    pr.horizon = 0;
    pr.surprise_eps = ABSORB;
    let mut sl = Sleep::new();
    let mut order = [0u8; N_TRAIN];
    let mut n = 0;
    for (k, &t) in train.iter().enumerate() { if t && n < N_TRAIN { order[n] = k as u8; n += 1; } }
    let mut rng = Rng(seed ^ 0x0DE5_0DE5);
    for row in out.iter_mut().take(epochs.min(EPOCHS)) {
        for i in (1..N_TRAIN).rev() { let j = rng.below(i + 1); order.swap(i, j); }
        for &k in order.iter() {
            let (a, b) = (k as usize / P, k as usize % P);
            let y = (a + b) % P;
            let base = enc(a, b);
            for c in 0..P {
                // recomputed per outcome: a sense may have been grown (or pruned) by the previous one
                let x: [f32; D] = cy.situation(&base);
                let r = if c == y { 1.0 } else { 0.0 };
                if compress {
                    cy.learn(&mut br, &x, c as u8, r);
                    let slot = newest(&br);
                    pr.after_learn(&mut br, &mut cy.grower, slot);
                    if sl.tick_learned(&br, slot) { sl.sleep(&mut br); }
                } else {
                    br.learn(&x, c as u8, r);
                    cy.grower.after_learn(&mut br);
                }
            }
        }
        let (mut tr, mut ho) = (0usize, 0usize);
        for (k, &is_train) in train.iter().enumerate() {
            let (a, b) = (k / P, k % P);
            let x: [f32; D] = cy.situation(&enc(a, b));
            if br.decide(&x, &[true; P], 0).action as usize == (a + b) % P { if is_train { tr += 1; } else { ho += 1; } }
        }
        let (t, h) = (tr as f32 / N_TRAIN as f32, ho as f32 / N_HELD as f32);
        let (m, s) = (br.in_use() as u16, cy.grower.active() as u8);
        if compress {
            row.train_a = t; row.held_a = h; row.mem_a = m; row.senses_a = s;
            row.tier_a = cy.tier(); row.groks_a = cy.events().len() as u8;
        } else {
            row.train_b = t; row.held_b = h; row.mem_b = m; row.senses_b = s;
        }
    }
}

/// Both arms' per-epoch curves for one seed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GrokRun {
    pub seed: u64,
    pub encoding: Encoding,
    /// Epochs run (the first `epochs` rows of `curve` are valid).
    pub epochs: usize,
    /// `true` for training pairs (index `a * P + b`).
    pub train: [bool; PAIRS],
    pub curve: [Epoch; EPOCHS],
}

/// Run both arms on one seed for the full [`EPOCHS`].
pub fn run(seed: u64, enc: Encoding) -> GrokRun { run_for(seed, enc, EPOCHS) }

/// Run both arms on one seed for `epochs` (at most [`EPOCHS`]) epochs.
pub fn run_for(seed: u64, enc: Encoding, epochs: usize) -> GrokRun {
    let epochs = epochs.min(EPOCHS);
    let train = split(seed);
    let mut curve = [Epoch::ZERO; EPOCHS];
    match enc {
        Encoding::OneHot => {
            run_arm::<B_ONEHOT, { B_ONEHOT + SLOTS }>(seed, onehot, &train, true, epochs, &mut curve);
            run_arm::<B_ONEHOT, { B_ONEHOT + SLOTS }>(seed, onehot, &train, false, epochs, &mut curve);
        }
        Encoding::Fourier => {
            run_arm::<B_FOURIER, { B_FOURIER + SLOTS }>(seed, fourier, &train, true, epochs, &mut curve);
            run_arm::<B_FOURIER, { B_FOURIER + SLOTS }>(seed, fourier, &train, false, epochs, &mut curve);
        }
    }
    GrokRun { seed, encoding: enc, epochs, train, curve }
}

/// The pre-registered jump detector (see the module docs) over one arm's curves.
pub fn find_jump(train: &[f32], held: &[f32]) -> Option<Jump> {
    let n = train.len().min(held.len());
    for e1 in 1..n {
        for e0 in (e1.saturating_sub(W)..e1).rev() {
            if held[e1] - held[e0] < J { continue; }
            if e0 < E || !(e0 - E..=e0).all(|t| train[t] >= TRAIN_OK) { continue; }
            let hi = (e1 + W).min(n);
            if !(e1..hi).all(|t| held[t] >= held[e0] + J / 2.0) { continue; }
            let mut p = e0;
            while p > 0 && train[p - 1] >= TRAIN_OK { p -= 1; }
            return Some(Jump { plateau: p, start: e0, end: e1 });
        }
    }
    None
}

/// First epoch that starts a run of at least `E + 1` epochs with train accuracy >= [`TRAIN_OK`].
pub fn first_plateau(train: &[f32]) -> Option<usize> {
    (0..train.len()).find(|&p| p + E < train.len() && (p..=p + E).all(|t| train[t] >= TRAIN_OK))
}

/// One seed's verdict (see the module docs for the pre-registered criteria). Epochs are counted
/// from 1 (epoch 1 = after the first replay of the training set).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GrokTrial {
    pub seed: u64,
    pub encoding: Encoding,
    /// Arm A: first epoch of the train plateau before the jump (no jump: of the first plateau).
    pub plateau_epoch: Option<u16>,
    /// Arm A: epoch the jump completed (`e1`).
    pub jump_epoch: Option<u16>,
    /// Arm B's jump, if any (a pass requires none).
    pub jump_b: Option<u16>,
    /// Held-out accuracy at the end (mean of the last [`END`] epochs).
    pub heldout_a: f32,
    pub heldout_b: f32,
    /// Train accuracy at the end (mean of the last [`END`] epochs).
    pub train_a: f32,
    pub train_b: f32,
    /// Arm A's size at the plateau start (no plateau: after epoch 0).
    pub size_plateau: Size,
    /// Arm A's size at the jump start (no jump: same as `size_plateau`).
    pub size_before: Size,
    /// Arm A's size at the jump end (no jump: at the last epoch).
    pub size_after: Size,
    pub pass: bool,
}

impl GrokRun {
    fn column(&self, f: impl Fn(&Epoch) -> f32) -> [f32; EPOCHS] {
        let mut v = [0.0f32; EPOCHS];
        for (o, e) in v.iter_mut().zip(self.curve.iter()).take(self.epochs) { *o = f(e); }
        v
    }

    fn size_a(&self, epoch: usize) -> Size {
        let e = &self.curve[epoch.min(self.epochs.max(1) - 1)];
        Size { memories: e.mem_a, senses: e.senses_a }
    }

    /// Mean of the last [`END`] epochs of a column.
    fn end(&self, col: &[f32]) -> f32 {
        let n = self.epochs;
        let k = END.min(n).max(1);
        col[n.saturating_sub(k)..n].iter().sum::<f32>() / k as f32
    }

    /// The pre-registered verdict for this seed.
    pub fn verdict(&self) -> GrokTrial {
        let n = self.epochs;
        let (ta, ha) = (self.column(|e| e.train_a), self.column(|e| e.held_a));
        let (tb, hb) = (self.column(|e| e.train_b), self.column(|e| e.held_b));
        let ja = find_jump(&ta[..n], &ha[..n]);
        let jb = find_jump(&tb[..n], &hb[..n]);
        let (heldout_a, heldout_b) = (self.end(&ha), self.end(&hb));
        let plateau = ja.map(|j| j.plateau).or_else(|| first_plateau(&ta[..n]));
        let size_plateau = self.size_a(plateau.unwrap_or(0));
        let (size_before, size_after) = match ja {
            Some(j) => (self.size_a(j.start), self.size_a(j.end)),
            None => (size_plateau, self.size_a(n.saturating_sub(1))),
        };
        let pass = ja.is_some() && jb.is_none() && heldout_b <= heldout_a - J / 2.0;
        GrokTrial {
            seed: self.seed, encoding: self.encoding,
            plateau_epoch: plateau.map(|p| p as u16 + 1), jump_epoch: ja.map(|j| j.end as u16 + 1), jump_b: jb.map(|j| j.end as u16 + 1),
            heldout_a, heldout_b, train_a: self.end(&ta), train_b: self.end(&tb),
            size_plateau, size_before, size_after, pass,
        }
    }
}

/// One seed of the primary (one-hot) grokking falsifier: both arms, full budget, verdict.
pub fn trial(seed: u64) -> GrokTrial { run(seed, Encoding::OneHot).verdict() }

/// One seed with a chosen encoding.
pub fn trial_with(seed: u64, enc: Encoding) -> GrokTrial { run(seed, enc).verdict() }

/// The pre-registered falsifier: every fresh seed, and whether at least [`PASS_SEEDS`] passed.
pub fn falsifier(enc: Encoding) -> ([GrokTrial; 5], bool) {
    let t = FRESH_SEEDS.map(|s| trial_with(s, enc));
    let passed = t.iter().filter(|x| x.pass).count();
    (t, passed >= PASS_SEEDS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;

    /// Each encoding's 5 fresh-seed trials, computed once and shared by the tests that read them.
    fn fresh(enc: Encoding) -> &'static ([GrokTrial; 5], bool) {
        static ONEHOT: OnceLock<([GrokTrial; 5], bool)> = OnceLock::new();
        static FOURIER: OnceLock<([GrokTrial; 5], bool)> = OnceLock::new();
        let cell = match enc { Encoding::OneHot => &ONEHOT, Encoding::Fourier => &FOURIER };
        cell.get_or_init(|| {
            // the seeds in parallel threads (each trial is deterministic, so this equals `falsifier(enc)`)
            let t: [GrokTrial; 5] = std::thread::scope(|s| FRESH_SEEDS.map(|seed| s.spawn(move || trial_with(seed, enc))).map(|h| h.join().unwrap()));
            let r = (t, t.iter().filter(|x| x.pass).count() >= PASS_SEEDS);
            for t in r.0.iter() {
                std::println!("  {} seed {}: plateau {:?} jump {:?} (B jump {:?}) | held-out A {:.3} B {:.3} | train A {:.3} B {:.3} | size A plateau {:?} before {:?} after {:?} | pass {}",
                    enc.name(), t.seed, t.plateau_epoch, t.jump_epoch, t.jump_b, t.heldout_a, t.heldout_b, t.train_a, t.train_b,
                    t.size_plateau, t.size_before, t.size_after, t.pass);
            }
            std::println!("{} falsifier: {}/5 seeds pass -> {}", enc.name(), r.0.iter().filter(|t| t.pass).count(), if r.1 { "PASS" } else { "FAIL" });
            r
        })
    }

    #[test]
    fn split_is_fixed_and_sized() {
        let s = split(5101);
        assert_eq!(s.iter().filter(|&&t| t).count(), N_TRAIN);
        assert_eq!(s, split(5101));
        assert_ne!(s, split(5202));
    }

    #[test]
    fn encodings_are_equivalent_for_memory_and_fourier_products_represent_the_answer() {
        // all distinct a (and b) are equidistant in both encodings: memory sees the same neighbours
        let d2 = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(p, q)| (p - q) * (p - q)).sum::<f32>();
        for a in 0..P { for a2 in 0..P { if a == a2 { continue; }
            assert!((d2(&onehot(a, 0), &onehot(a2, 0)) - 2.0).abs() < 1e-5);
            assert!((d2(&fourier(a, 0), &fourier(a2, 0)) - P as f32).abs() < 1e-4);
        } }
        // the four products of frequency 1 give cos(w (a + b - c)), maximal exactly at c = a + b mod P
        let w = 2.0 * core::f32::consts::PI / P as f32;
        for a in 0..P { for b in 0..P {
            let x = fourier(a, b);
            let (ca, sa, cb, sb) = (x[0], x[1], x[P - 1], x[P]);
            let (cc, ss, sc, cs) = (ca * cb, sa * sb, sa * cb, ca * sb);
            let best = (0..P).max_by(|&c1, &c2| {
                let s = |c: usize| { let (co, si) = ((w * c as f32).cos(), (w * c as f32).sin()); co * cc - co * ss + si * sc + si * cs };
                s(c1).partial_cmp(&s(c2)).unwrap()
            }).unwrap();
            assert_eq!(best, (a + b) % P);
        } }
    }

    #[test]
    fn jump_detector_is_the_preregistered_one() {
        let n = 60;
        let mut train = [0.0f32; 60];
        let mut held = [0.1f32; 60];
        for t in 5..n { train[t] = 1.0; }
        // plateau from epoch 5, held-out rises 0.1 -> 0.5 over epochs 30..34 and stays
        for t in 30..n { held[t] = (0.1 + 0.1 * (t - 29) as f32).min(0.5); }
        let j = find_jump(&train, &held).expect("plateau then sustained jump");
        assert_eq!((j.plateau, j.start), (5, 29));
        assert!(held[j.end] - held[j.start] >= J && j.end - j.start <= W);
        // a transient spike is not a jump
        let mut spike = [0.1f32; 60]; spike[40] = 0.9;
        assert_eq!(find_jump(&train, &spike), None);
        // a jump without a preceding plateau of E epochs is not grokking
        let mut early = [0.0f32; 60];
        for t in 25..n { early[t] = 1.0; }
        let mut h2 = [0.1f32; 60]; for t in 30..n { h2[t] = 0.6; }
        assert_eq!(find_jump(&early, &h2), None);
        // a rise of less than J is not a jump
        let mut small = [0.1f32; 60]; for t in 30..n { small[t] = 0.1 + J - 0.02; }
        assert_eq!(find_jump(&train, &small), None);
    }

    #[test]
    fn deterministic() {
        let (a, b) = (run_for(7, Encoding::OneHot, 4), run_for(7, Encoding::OneHot, 4));
        assert_eq!(a, b);
        assert!(a.curve[3].train_a > 0.0 || a.curve[3].train_b > 0.0, "it learns something");
    }

    /// PRE-REGISTERED primary falsifier (one-hot senses, 5 fresh seeds; criteria in the module docs).
    /// Run: cargo test --release --lib brain_grok -- --nocapture
    #[test]
    fn falsifier_onehot_groks_only_with_compression() {
        let (t, pass) = fresh(Encoding::OneHot);
        assert!(*pass, "one-hot: {}/5 seeds pass (need {})", t.iter().filter(|x| x.pass).count(), PASS_SEEDS);
    }

    /// PRE-REGISTERED secondary falsifier (Fourier senses, same seeds and criteria).
    #[test]
    fn falsifier_fourier_groks_only_with_compression() {
        let (t, pass) = fresh(Encoding::Fourier);
        assert!(*pass, "fourier: {}/5 seeds pass (need {})", t.iter().filter(|x| x.pass).count(), PASS_SEEDS);
    }

    /// PRE-REGISTERED analysis prediction: with one-hot senses both arms end at most chance + 0.10
    /// held-out on every fresh seed (the linear + kNN brain and its growth grammar cannot generalize).
    #[test]
    fn analysis_prediction_onehot_stays_at_chance() {
        let (t, _) = fresh(Encoding::OneHot);
        let bound = 1.0 / P as f32 + CHANCE_MARGIN;
        for x in t.iter() {
            assert!(x.heldout_a <= bound && x.heldout_b <= bound, "seed {}: held-out A {:.3} B {:.3} > {:.3}", x.seed, x.heldout_a, x.heldout_b, bound);
        }
    }
}
