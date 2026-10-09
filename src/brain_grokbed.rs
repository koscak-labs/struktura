//! Grokking testbed: modular addition learned by a tiny network with a LEARNED representation.
//!
//! [`crate::brain_grok3`] showed that compression reliably lifts the linear + kNN brain's held-out
//! accuracy on `(a + b) mod p`, but only to 0.05..0.52: a discrete search over fixed feature
//! products does not assemble the representation. Grokking, as documented (Power et al. 2022;
//! Nanda et al. 2023; Gromov 2023), is a property of a model that LEARNS its representation by
//! gradient descent on a small fixed training set, trained far past fitting, under a compression
//! pressure (weight decay): train accuracy hits 1.0 early by memorizing, and held-out accuracy
//! jumps much later, when the weights reorganize into a compact (Fourier) circuit.
//!
//! This module is that world, std-only and CPU-only, deterministic for a seed:
//!
//! - **Task**: `(a + b) mod p`. All `p^2` pairs; a seed-chosen fixed training set of `n_train`
//!   pairs; the rest are the test set (measured, never trained on).
//! - **Model**: `pre = Ea[a] + Eb[b]` (two learned `p x h` embeddings), `z = act(pre)`,
//!   `logits = W z` (`p x h`), softmax cross-entropy. `3 p h` parameters.
//! - **Training**: full batch, AdamW (decoupled weight decay = the compression pressure).
//! - **Measured** every `eval_every` steps: train / test accuracy and loss, the squared weight norm
//!   (compression), the Fourier concentration of the embeddings (fraction of each neuron's
//!   non-constant energy in its single strongest frequency, energy-weighted: about `H_m / m` for
//!   random weights with `m = (p - 1) / 2` frequencies, 1.0 for a pure Fourier circuit) and the
//!   number of active neurons (squared norm at least [`ACTIVE_FRAC`] of the largest).
//! - **Growth hooks** (for [`crate::brain_growth`]): [`Net::grow`] appends hidden neurons,
//!   [`Net::prune`] removes neurons whose weights decayed to nothing (compression by structure).
//!
//! ## Pre-registration (written and committed before any fresh seed was run)
//!
//! **Claim:** with a compression pressure (weight decay) the network GROKS: it fits the training
//! set early by memorizing, and generalizes much later; the identical network without weight decay
//! does not.
//!
//! - **World** [`PREREG`]: `p = 23`, 265 of the 529 pairs for training (50 %), 128 hidden neurons,
//!   quadratic activation, embeddings `N(0, 0.1^2)`, AdamW `lr 1e-3`, betas `0.9 / 0.98`, full
//!   batch, 12 000 steps, measured every 100. **Arm D**: weight decay 1.0. **Arm N** (control,
//!   [`PREREG_NO_DECAY`]): weight decay 0, everything else identical (same seed: same split, same
//!   initial weights).
//! - **Grokking event** in a run ([`analyse`]): `t_fit` = first step with train accuracy >= 0.99;
//!   `t_gen` = first step with test accuracy >= 0.95. Event iff both exist, test accuracy at `t_fit`
//!   is <= 0.20 (memorized, not generalized; chance is 1/23 = 0.043), `t_gen >= 3 t_fit` (delayed)
//!   and the mean test accuracy of the last 10 measurements is >= 0.95 (it stays).
//! - **Seed passes** iff D has an event, N has none, and N's end test accuracy is at least 0.50
//!   below D's. **PASS iff at least 9 of the 10 fresh seeds [`FRESH_SEEDS`] pass** (if a seed
//!   passed with probability 1/2, `P(>= 9 of 10) = 0.0107`).
//! - **Reported, not judged** (compression precedes generalization): D's squared weight norm at
//!   `t_gen` vs its peak before `t_gen`; D's Fourier concentration at `t_fit` vs `t_gen`.
//! - **Development** (before this pre-registration, on [`DEV_SEEDS`] only): the world was chosen
//!   after grids over init, lr, decay, train size, `p` and activation. ReLU at `p = 23` generalized
//!   only partly within 50 000 steps (test 0.2..0.5); quadratic generalized. On dev seeds 100..102
//!   arm D fit at step ~400 (test 0.01..0.06) and reached test 0.95 at steps 3 800..4 000, 1.00 by
//!   ~5 800; its weight norm peaked near step 2 000 and fell ~11 % by `t_gen`; Fourier concentration
//!   rose 0.44 -> 0.73. Arm N fit at the same step and stayed at test 0.08..0.12, its norm growing
//!   all run. At 30 % training data neither arm generalized within 12 000 steps.
//! - **Analysis prediction:** PASS (10/10). Risk stated in advance: the transition here is an
//!   S-curve over ~3 000 steps, delayed ~10x after the fit, not a single-step jump.
//!
//! ## RESULT
//!
//! (added after the one run; rule, thresholds and seeds unchanged)


use std::vec;
use std::vec::Vec;

/// Activation of the hidden layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Act { Relu, Quad }

impl Act {
    #[inline]
    fn f(self, x: f32) -> f32 { match self { Act::Relu => x.max(0.0), Act::Quad => x * x } }
    #[inline]
    fn df(self, x: f32) -> f32 { match self { Act::Relu => if x > 0.0 { 1.0 } else { 0.0 }, Act::Quad => 2.0 * x } }
}

/// A neuron is active when its squared weight norm is at least this fraction of the largest.
pub const ACTIVE_FRAC: f32 = 0.01;

/// Deterministic xorshift64* generator.
#[derive(Clone, Debug)]
pub struct Rng(pub u64);

impl Rng {
    pub fn new(seed: u64) -> Self { Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03 | 1) }
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12; self.0 ^= self.0 << 25; self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn below(&mut self, n: usize) -> usize { (self.next() % n as u64) as usize }
    /// Uniform in (0, 1].
    pub fn unit(&mut self) -> f32 { ((self.next() >> 40) as f32 + 1.0) / (1u64 << 24) as f32 }
    /// Standard normal (Box-Muller).
    pub fn normal(&mut self) -> f32 {
        let (u, v) = (self.unit(), self.unit());
        (-2.0 * u.ln()).sqrt() * (2.0 * core::f32::consts::PI * v).cos()
    }
}

/// A labelled example `(a, b, op, y)`: op 0 is `y = (a + b) mod p`, op 1 is `y = (a - b) mod p`.
pub type Pair = (u16, u16, u8, u16);

/// The answer of operation `op` on `(a, b)`.
pub fn answer(op: u8, a: usize, b: usize, p: usize) -> usize { if op == 0 { (a + b) % p } else { (a + p - b % p) % p } }

/// The seed's fixed split of all `p^2` pairs: `(train, test)`.
pub fn split(seed: u64, p: usize, n_train: usize) -> (Vec<Pair>, Vec<Pair>) { split_op(seed, p, n_train, 0) }

/// The seed's fixed split of operation `op`'s `p^2` examples (op 0 = [`split`]; other ops use
/// an independent shuffle).
pub fn split_op(seed: u64, p: usize, n_train: usize, op: u8) -> (Vec<Pair>, Vec<Pair>) {
    let mut all: Vec<Pair> = (0..p * p).map(|k| ((k / p) as u16, (k % p) as u16, op, answer(op, k / p, k % p, p) as u16)).collect();
    let mut r = Rng::new(seed ^ 0x5B11_7000 ^ ((op as u64) << 40));
    for i in (1..all.len()).rev() { let j = r.below(i + 1); all.swap(i, j); }
    let test = all.split_off(n_train.min(all.len()));
    (all, test)
}

/// The network and its optimizer state.
#[derive(Clone, Debug)]
pub struct Net {
    pub p: usize,
    pub h: usize,
    pub act: Act,
    /// Operations (op tokens). With 1 there is no op embedding.
    pub ops: usize,
    /// `[Ea (p x h) | Eb (p x h) | W (p x h) | Eo (ops x h, only if ops > 1)]`, row-major, row =
    /// token / class / op.
    pub w: Vec<f32>,
    m: Vec<f32>,
    v: Vec<f32>,
    t: u32,
}

/// AdamW constants (the usual defaults).
const BETA1: f32 = 0.9;
const BETA2: f32 = 0.98;
const EPS: f32 = 1e-8;

impl Net {
    /// Embeddings `N(0, init^2)`, output weights `N(0, 1 / h)`.
    pub fn new(p: usize, h: usize, act: Act, init: f32, rng: &mut Rng) -> Self {
        let mut w = vec![0.0f32; 3 * p * h];
        for x in w[..2 * p * h].iter_mut() { *x = init * rng.normal(); }
        let s = 1.0 / (h as f32).sqrt();
        for x in w[2 * p * h..].iter_mut() { *x = s * rng.normal(); }
        let n = w.len();
        Net { p, h, act, ops: 1, w, m: vec![0.0; n], v: vec![0.0; n], t: 0 }
    }

    pub fn params(&self) -> usize { self.w.len() }

    #[inline]
    fn ea(&self) -> usize { 0 }
    #[inline]
    fn eb(&self) -> usize { self.p * self.h }
    #[inline]
    fn wo(&self) -> usize { 2 * self.p * self.h }
    #[inline]
    fn eo(&self) -> usize { 3 * self.p * self.h }
    /// Rows of each block.
    fn rows(&self) -> [usize; 4] { [self.p, self.p, self.p, if self.ops > 1 { self.ops } else { 0 }] }

    /// Logits of one pair into `out` (length `p`); `pre` and `z` are scratch of length `h`.
    fn forward(&self, a: usize, b: usize, op: usize, pre: &mut [f32], z: &mut [f32], out: &mut [f32]) {
        let (h, p) = (self.h, self.p);
        let (ea, eb, wo) = (&self.w[self.ea() + a * h..][..h], &self.w[self.eb() + b * h..][..h], self.wo());
        for j in 0..h { pre[j] = ea[j] + eb[j]; }
        if self.ops > 1 { let eo = &self.w[self.eo() + op * h..][..h]; for j in 0..h { pre[j] += eo[j]; } }
        for j in 0..h { z[j] = self.act.f(pre[j]); }
        for c in 0..p {
            let row = &self.w[wo + c * h..][..h];
            out[c] = row.iter().zip(z.iter()).map(|(x, y)| x * y).sum();
        }
    }

    /// Predicted class of a pair.
    pub fn predict(&self, a: usize, b: usize, op: usize) -> usize {
        let (mut pre, mut z, mut out) = (vec![0.0; self.h], vec![0.0; self.h], vec![0.0; self.p]);
        self.forward(a, b, op, &mut pre, &mut z, &mut out);
        argmax(&out)
    }

    /// `(accuracy, mean cross-entropy)` on a set of pairs.
    pub fn eval(&self, data: &[Pair]) -> (f32, f32) {
        if data.is_empty() { return (0.0, 0.0); }
        let (mut pre, mut z, mut out) = (vec![0.0; self.h], vec![0.0; self.h], vec![0.0; self.p]);
        let (mut right, mut loss) = (0usize, 0.0f64);
        for &(a, b, op, y) in data {
            self.forward(a as usize, b as usize, op as usize, &mut pre, &mut z, &mut out);
            if argmax(&out) == y as usize { right += 1; }
            loss += xent(&mut out, y as usize) as f64;
        }
        (right as f32 / data.len() as f32, (loss / data.len() as f64) as f32)
    }

    /// One full-batch AdamW step on `data` with learning rate `lr` and decoupled weight decay `wd`.
    /// Returns the training loss before the step.
    pub fn step(&mut self, data: &[Pair], lr: f32, wd: f32) -> f32 {
        let (loss, g) = self.grad(data);
        self.t += 1;
        let (c1, c2) = (1.0 - BETA1.powi(self.t as i32), 1.0 - BETA2.powi(self.t as i32));
        for i in 0..self.w.len() {
            let gi = g[i];
            self.m[i] = BETA1 * self.m[i] + (1.0 - BETA1) * gi;
            self.v[i] = BETA2 * self.v[i] + (1.0 - BETA2) * gi * gi;
            let upd = (self.m[i] / c1) / ((self.v[i] / c2).sqrt() + EPS);
            self.w[i] -= lr * (upd + wd * self.w[i]);
        }
        loss
    }

    /// Mean cross-entropy on `data` and its gradient with respect to every weight.
    pub fn grad(&self, data: &[Pair]) -> (f32, Vec<f32>) {
        let (h, p) = (self.h, self.p);
        let (ea0, eb0, wo0, eo0) = (self.ea(), self.eb(), self.wo(), self.eo());
        let mut g = vec![0.0f32; self.w.len()];
        let (mut pre, mut z, mut out, mut dz) = (vec![0.0; h], vec![0.0; h], vec![0.0; p], vec![0.0; h]);
        let mut loss = 0.0f64;
        for &(a, b, op, y) in data {
            let (a, b, op, y) = (a as usize, b as usize, op as usize, y as usize);
            self.forward(a, b, op, &mut pre, &mut z, &mut out);
            loss += xent(&mut out, y) as f64; // out is now the softmax
            out[y] -= 1.0;
            for d in dz.iter_mut() { *d = 0.0; }
            for c in 0..p {
                let dl = out[c];
                let row = &self.w[wo0 + c * h..][..h];
                let grow = &mut g[wo0 + c * h..][..h];
                for j in 0..h { grow[j] += dl * z[j]; dz[j] += dl * row[j]; }
            }
            for j in 0..h {
                let d = dz[j] * self.act.df(pre[j]);
                g[ea0 + a * h + j] += d;
                g[eb0 + b * h + j] += d;
                if self.ops > 1 { g[eo0 + op * h + j] += d; }
            }
        }
        let n = data.len().max(1) as f32;
        for x in g.iter_mut() { *x /= n; }
        ((loss / n as f64) as f32, g)
    }

    /// Squared norm of all weights.
    pub fn norm2(&self) -> f32 { self.w.iter().map(|x| x * x).sum() }

    /// Squared norm of hidden neuron `j` (its two embedding columns and its output column).
    pub fn neuron_norm2(&self, j: usize) -> f32 {
        let h = self.h;
        let rows: usize = self.rows().iter().sum();
        (0..rows).map(|r| { let x = self.w[r * h + j]; x * x }).sum()
    }

    /// Neurons whose squared norm is at least [`ACTIVE_FRAC`] of the largest.
    pub fn active(&self) -> usize {
        let n: Vec<f32> = (0..self.h).map(|j| self.neuron_norm2(j)).collect();
        let mx = n.iter().cloned().fold(0.0f32, f32::max);
        n.iter().filter(|&&x| mx > 0.0 && x >= ACTIVE_FRAC * mx).count()
    }

    /// Energy-weighted Fourier concentration of both embeddings (see the module docs).
    pub fn fourier_conc(&self) -> f32 {
        let (h, p) = (self.h, self.p);
        let m = (p - 1) / 2;
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for off in [self.ea(), self.eb()] {
            for j in 0..h {
                let (mut best, mut tot) = (0.0f64, 0.0f64);
                for k in 1..=m {
                    let (mut re, mut im) = (0.0f64, 0.0f64);
                    for a in 0..p {
                        let x = self.w[off + a * h + j] as f64;
                        let th = 2.0 * core::f64::consts::PI * (k * a) as f64 / p as f64;
                        re += x * th.cos(); im += x * th.sin();
                    }
                    let e = re * re + im * im;
                    tot += e; if e > best { best = e; }
                }
                num += best; den += tot;
            }
        }
        if den > 0.0 { (num / den) as f32 } else { 0.0 }
    }

    /// Re-layout with `new_h` neurons; `keep[j]` = old neuron kept (in order). New neurons get
    /// embeddings `N(0, init^2)` and zero output weights (the function is unchanged at the
    /// moment of growth). Optimizer state is kept for old neurons, zero for new ones.
    fn relayout(&mut self, keep: &[usize], extra: usize, init: f32, rng: &mut Rng) {
        let h0 = self.h;
        let h1 = keep.len() + extra;
        let rows = self.rows();
        let n: usize = rows.iter().sum::<usize>() * h1;
        let (mut w, mut m, mut v) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
        let mut r0 = 0usize; // first row of the block, in rows
        for (blk, &nr) in rows.iter().enumerate() {
            for r in r0..r0 + nr {
                for (nj, &oj) in keep.iter().enumerate() {
                    let (o, d) = (r * h0 + oj, r * h1 + nj);
                    w[d] = self.w[o]; m[d] = self.m[o]; v[d] = self.v[o];
                }
                // new neurons: random input embeddings, zero output and op weights
                if blk < 2 { for nj in keep.len()..h1 { w[r * h1 + nj] = init * rng.normal(); } }
            }
            r0 += nr;
        }
        self.h = h1; self.w = w; self.m = m; self.v = v;
    }

    /// Add an operation token. Going from 1 to 2 ops creates the op embedding with both rows zero,
    /// so the function on op 0 is unchanged and op 1 starts as a copy of op 0.
    pub fn add_op(&mut self) {
        let extra = if self.ops == 1 { 2 * self.h } else { self.h };
        self.ops += 1;
        let n = self.w.len() + extra;
        self.w.resize(n, 0.0); self.m.resize(n, 0.0); self.v.resize(n, 0.0);
    }

    /// Append `extra` hidden neurons (function unchanged at the moment of growth).
    pub fn grow(&mut self, extra: usize, init: f32, rng: &mut Rng) {
        let keep: Vec<usize> = (0..self.h).collect();
        self.relayout(&keep, extra, init, rng);
    }

    /// Remove neurons whose squared norm is below `frac` of the largest; returns how many.
    /// Keeps at least one neuron.
    pub fn prune(&mut self, frac: f32) -> usize {
        let n: Vec<f32> = (0..self.h).map(|j| self.neuron_norm2(j)).collect();
        let mx = n.iter().cloned().fold(0.0f32, f32::max);
        let mut keep: Vec<usize> = (0..self.h).filter(|&j| n[j] >= frac * mx).collect();
        if keep.is_empty() { keep.push(0); }
        let removed = self.h - keep.len();
        if removed > 0 { let mut r = Rng::new(0); self.relayout(&keep, 0, 0.0, &mut r); }
        removed
    }
}

fn argmax(x: &[f32]) -> usize {
    let mut b = 0;
    for i in 1..x.len() { if x[i] > x[b] { b = i; } }
    b
}

/// Cross-entropy of `logits` against `y`; leaves the softmax in `logits`.
fn xent(logits: &mut [f32], y: usize) -> f32 {
    let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0.0f32;
    for l in logits.iter_mut() { *l = (*l - mx).exp(); s += *l; }
    for l in logits.iter_mut() { *l /= s; }
    -(logits[y].max(1e-30)).ln()
}

/// One run's settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    pub p: usize,
    pub hidden: usize,
    pub n_train: usize,
    pub act: Act,
    pub init: f32,
    pub lr: f32,
    pub wd: f32,
    pub steps: usize,
    pub eval_every: usize,
}

/// One measurement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub step: u32,
    pub train_acc: f32,
    pub test_acc: f32,
    pub train_loss: f32,
    pub test_loss: f32,
    pub norm2: f32,
    pub conc: f32,
    pub active: u16,
    pub hidden: u16,
}

/// Measure the net now.
pub fn measure(net: &Net, step: usize, train: &[Pair], test: &[Pair]) -> Point {
    let (ta, tl) = net.eval(train);
    let (ha, hl) = net.eval(test);
    Point { step: step as u32, train_acc: ta, test_acc: ha, train_loss: tl, test_loss: hl, norm2: net.norm2(),
        conc: net.fourier_conc(), active: net.active() as u16, hidden: net.h as u16 }
}

/// Train one seed under `cfg`, measuring at step 0 and every `eval_every` steps.
pub fn run(seed: u64, cfg: &Config) -> Vec<Point> {
    let (train, test) = split(seed, cfg.p, cfg.n_train);
    let mut rng = Rng::new(seed);
    let mut net = Net::new(cfg.p, cfg.hidden, cfg.act, cfg.init, &mut rng);
    let mut curve = Vec::with_capacity(cfg.steps / cfg.eval_every.max(1) + 2);
    curve.push(measure(&net, 0, &train, &test));
    for s in 1..=cfg.steps {
        net.step(&train, cfg.lr, cfg.wd);
        if s % cfg.eval_every == 0 { curve.push(measure(&net, s, &train, &test)); }
    }
    curve
}

/// The pre-registered world (see "Pre-registration" in the module docs).
pub const PREREG: Config = Config { p: 23, hidden: 128, n_train: 265, act: Act::Quad, init: 0.1, lr: 1e-3, wd: 1.0, steps: 12_000, eval_every: 100 };
/// The no-compression control: identical, weight decay 0.
pub const PREREG_NO_DECAY: Config = Config { wd: 0.0, ..PREREG };
/// Pre-registered fresh seeds (never run before the pre-registration was committed).
pub const FRESH_SEEDS: [u64; 10] = [31337, 31415, 32003, 32771, 33013, 33533, 34061, 34511, 35027, 35543];
/// Seeds used while choosing the world (100..=105).
pub const DEV_SEEDS: [u64; 6] = [100, 101, 102, 103, 104, 105];
/// Train accuracy that counts as fitted.
pub const FIT: f32 = 0.99;
/// Test accuracy that counts as generalized.
pub const GEN: f32 = 0.95;
/// Test accuracy at the fit step must be at most this (memorized, not generalized).
pub const MEMO_MAX: f32 = 0.20;
/// Generalization must come at least this many times later than the fit.
pub const DELAY: u32 = 3;
/// Evaluations averaged for "end" accuracy.
pub const END_EVALS: usize = 10;
/// The control must end at least this far below the decay arm.
pub const GAP: f32 = 0.50;
/// Seeds of 10 that must pass.
pub const PASS_SEEDS: usize = 9;

/// A run's grokking analysis under the pre-registered rule.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grok {
    /// First step with train accuracy >= [`FIT`].
    pub t_fit: Option<u32>,
    /// Test accuracy at `t_fit`.
    pub test_at_fit: f32,
    /// First step with test accuracy >= [`GEN`].
    pub t_gen: Option<u32>,
    /// Mean test accuracy of the last [`END_EVALS`] evaluations.
    pub end_test: f32,
    /// Largest squared weight norm before `t_gen` (or over the run), and the norm at `t_gen` (or the end).
    pub peak_norm2: f32,
    pub norm2_at_gen: f32,
    /// Fourier concentration at `t_fit` and at `t_gen` (or the end).
    pub conc_at_fit: f32,
    pub conc_at_gen: f32,
    /// The pre-registered grokking event.
    pub event: bool,
}

/// Analyse one curve under the pre-registered rule.
pub fn analyse(c: &[Point]) -> Grok {
    let fit = c.iter().position(|p| p.train_acc >= FIT);
    let gen = c.iter().position(|p| p.test_acc >= GEN);
    let k = END_EVALS.min(c.len()).max(1);
    let end_test = c[c.len() - k..].iter().map(|p| p.test_acc).sum::<f32>() / k as f32;
    let g = gen.unwrap_or(c.len() - 1);
    let peak_norm2 = c[..=g].iter().map(|p| p.norm2).fold(0.0f32, f32::max);
    let test_at_fit = fit.map(|i| c[i].test_acc).unwrap_or(0.0);
    let (t_fit, t_gen) = (fit.map(|i| c[i].step), gen.map(|i| c[i].step));
    let event = match (t_fit, t_gen) {
        (Some(f), Some(t)) => test_at_fit <= MEMO_MAX && t >= DELAY * f.max(1) && end_test >= GEN,
        _ => false,
    };
    Grok { t_fit, test_at_fit, t_gen, end_test, peak_norm2, norm2_at_gen: c[g].norm2,
        conc_at_fit: fit.map(|i| c[i].conc).unwrap_or(0.0), conc_at_gen: c[g].conc, event }
}

/// One seed: both arms and the verdict.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Trial {
    pub seed: u64,
    pub decay: Grok,
    pub control: Grok,
    pub pass: bool,
}

/// Run both arms on one seed under the pre-registered settings.
pub fn trial(seed: u64) -> Trial {
    let (d, n) = (analyse(&run(seed, &PREREG)), analyse(&run(seed, &PREREG_NO_DECAY)));
    Trial { seed, decay: d, control: n, pass: d.event && !n.event && n.end_test <= d.end_test - GAP }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The analytic gradient matches finite differences (the instrument first).
    #[test]
    fn gradient_matches_finite_differences() {
        for act in [Act::Relu, Act::Quad] {
            let mut rng = Rng::new(3);
            let mut net = Net::new(5, 7, act, 0.5, &mut rng);
            net.add_op();
            let eo = net.eo();
            for x in net.w[eo..].iter_mut() { *x = 0.5 * rng.normal(); }
            let data: Vec<Pair> = vec![(1, 2, 0, 3), (4, 4, 0, 3), (0, 3, 0, 3), (2, 2, 0, 4), (4, 1, 1, 3), (0, 2, 1, 3)];
            let (_, g) = net.grad(&data);
            let mut checked = 0;
            for i in 0..net.w.len() {
                let e = 1e-2f32;
                let mut np = net.clone(); np.w[i] += e;
                let mut nm = net.clone(); nm.w[i] -= e;
                let fd = (np.eval(&data).1 as f64 - nm.eval(&data).1 as f64) / (2.0 * e as f64);
                // ReLU kinks make single coordinates non-smooth; skip weights whose pre-activation is near 0
                if act == Act::Relu && (fd - g[i] as f64).abs() > 1e-2 && fd.abs() < 1e-1 { continue; }
                assert!((fd - g[i] as f64).abs() <= 2e-3 + 2e-2 * fd.abs(), "{:?} weight {}: fd {} vs analytic {}", act, i, fd, g[i]);
                checked += 1;
            }
            assert!(checked >= net.w.len() * 9 / 10, "{:?}: only {} of {} checked", act, checked, net.w.len());
        }
    }

    #[test]
    fn split_is_a_partition_and_labels_are_right() {
        let (tr, te) = split(9, 11, 40);
        assert_eq!((tr.len(), te.len()), (40, 81));
        let mut seen = vec![false; 121];
        for &(a, b, _, y) in tr.iter().chain(te.iter()) {
            assert_eq!(y as usize, (a as usize + b as usize) % 11);
            assert!(!seen[a as usize * 11 + b as usize]); seen[a as usize * 11 + b as usize] = true;
        }
        assert_eq!(split(9, 11, 40), split(9, 11, 40));
    }

    #[test]
    fn grow_keeps_the_function_and_prune_removes_dead_neurons() {
        let mut rng = Rng::new(5);
        let mut net = Net::new(7, 6, Act::Relu, 0.5, &mut rng);
        let before: Vec<usize> = (0..49).map(|k| net.predict(k / 7, k % 7, 0)).collect();
        net.grow(4, 0.5, &mut rng);
        assert_eq!(net.h, 10);
        let after: Vec<usize> = (0..49).map(|k| net.predict(k / 7, k % 7, 0)).collect();
        assert_eq!(before, after, "growth with zero output weights leaves the function unchanged");
        // kill neuron 2 entirely, then prune
        let (p, h) = (net.p, net.h);
        for blk in 0..3 { for r in 0..p { net.w[blk * p * h + r * h + 2] = 0.0; } }
        let before: Vec<usize> = (0..49).map(|k| net.predict(k / 7, k % 7, 0)).collect();
        let removed = net.prune(1e-6);
        assert!(removed >= 1 && net.h == 10 - removed);
        let pruned: Vec<usize> = (0..49).map(|k| net.predict(k / 7, k % 7, 0)).collect();
        assert_eq!(before, pruned, "removing dead neurons leaves the function unchanged");
    }

    #[test]
    fn add_op_keeps_op0_and_growth_keeps_both_ops() {
        let mut rng = Rng::new(8);
        let mut net = Net::new(7, 6, Act::Relu, 0.5, &mut rng);
        let f0: Vec<usize> = (0..49).map(|k| net.predict(k / 7, k % 7, 0)).collect();
        net.add_op();
        assert_eq!(net.ops, 2);
        assert_eq!(f0, (0..49).map(|k| net.predict(k / 7, k % 7, 0)).collect::<Vec<_>>());
        assert_eq!(f0, (0..49).map(|k| net.predict(k / 7, k % 7, 1)).collect::<Vec<_>>(), "op 1 starts as a copy of op 0");
        let eo = net.eo();
        for x in net.w[eo..].iter_mut() { *x = rng.normal(); }
        let both: Vec<usize> = (0..98).map(|k| net.predict(k % 49 / 7, k % 7, k / 49)).collect();
        net.grow(5, 0.5, &mut rng);
        assert_eq!(both, (0..98).map(|k| net.predict(k % 49 / 7, k % 7, k / 49)).collect::<Vec<_>>());
        assert_eq!(net.w.len(), 3 * 7 * 11 + 2 * 11);
    }

    #[test]
    fn fourier_concentration_reads_a_pure_circuit_as_one() {
        let mut rng = Rng::new(1);
        let mut net = Net::new(11, 4, Act::Quad, 0.0, &mut rng);
        let (p, h) = (net.p, net.h);
        for j in 0..h { for a in 0..p {
            let th = 2.0 * core::f32::consts::PI * ((j + 1) * a) as f32 / p as f32;
            net.w[a * h + j] = th.cos(); net.w[p * h + a * h + j] = th.sin();
        } }
        assert!(net.fourier_conc() > 0.999);
        let rnd = Net::new(11, 64, Act::Quad, 1.0, &mut rng);
        let c = rnd.fourier_conc();
        assert!(c > 0.25 && c < 0.65, "random embeddings: {}", c);
    }

    #[test]
    fn it_fits_a_small_training_set() {
        let cfg = Config { p: 7, hidden: 32, n_train: 30, act: Act::Relu, init: 1.0, lr: 1e-2, wd: 0.0, steps: 300, eval_every: 300 };
        let c = run(1, &cfg);
        assert!(c.last().unwrap().train_acc >= 0.99, "{:?}", c.last());
    }

    fn curve(points: &[(u32, f32, f32)]) -> Vec<Point> {
        points.iter().map(|&(s, tr, te)| Point { step: s, train_acc: tr, test_acc: te, train_loss: 0.0, test_loss: 0.0,
            norm2: 1000.0 + s as f32 - if s > 2000 { 2.0 * (s - 2000) as f32 } else { 0.0 }, conc: te, active: 1, hidden: 1 }).collect()
    }

    /// The analyser on hand-made curves (the instrument first).
    #[test]
    fn analyse_fixtures() {
        let mut g = vec![(0, 0.05, 0.05), (400, 1.0, 0.04)];
        for s in (500..=4000).step_by(100) { g.push((s, 1.0, 0.04 + 0.92 * (s - 500) as f32 / 3500.0)); }
        for s in (4100..=5000).step_by(100) { g.push((s, 1.0, 0.99)); }
        let a = analyse(&curve(&g));
        assert!(a.event, "a delayed rise is a grokking event: {:?}", a);
        assert_eq!((a.t_fit, a.t_gen), (Some(400), Some(4000)));
        assert!(a.norm2_at_gen < a.peak_norm2);
        // generalizes at the same time as it fits: no event
        let a = analyse(&curve(&[(0, 0.05, 0.05), (400, 1.0, 0.96), (500, 1.0, 0.99), (600, 1.0, 0.99)]));
        assert!(!a.event);
        // memorizes and never generalizes: no event
        let a = analyse(&curve(&[(0, 0.05, 0.05), (400, 1.0, 0.04), (5000, 1.0, 0.10)]));
        assert!(!a.event && a.t_gen.is_none());
        // generalizes late but does not stay: no event
        let mut h = g.clone(); for x in h.iter_mut().rev().take(10) { x.2 = 0.5; }
        assert!(!analyse(&curve(&h)).event);
        // too early (t_gen < 3 t_fit): no event
        assert!(!analyse(&curve(&[(0, 0.05, 0.05), (400, 1.0, 0.04), (1100, 1.0, 0.96), (1200, 1.0, 0.99)])).event);
    }

    /// The pre-registration, pinned.
    #[test]
    fn preregistration_is_pinned() {
        assert_eq!(PREREG, Config { p: 23, hidden: 128, n_train: 265, act: Act::Quad, init: 0.1, lr: 1e-3, wd: 1.0, steps: 12_000, eval_every: 100 });
        assert_eq!(PREREG_NO_DECAY, Config { wd: 0.0, ..PREREG });
        assert_eq!(FRESH_SEEDS, [31337, 31415, 32003, 32771, 33013, 33533, 34061, 34511, 35027, 35543]);
        assert!(FRESH_SEEDS.iter().all(|s| !DEV_SEEDS.contains(s)));
        assert_eq!((FIT, GEN, MEMO_MAX, DELAY, END_EVALS, GAP, PASS_SEEDS), (0.99, 0.95, 0.20, 3, 10, 0.50, 9));
        assert_eq!((BETA1, BETA2), (0.9, 0.98));
    }

    /// PRE-REGISTERED falsifier: weight decay groks, no decay does not; 10 fresh seeds, >= 9 pass.
    /// Run: cargo test --release --lib brain_grokbed::tests::falsifier -- --ignored --nocapture
    #[test]
    #[ignore = "slow: 10 seeds x 2 arms x 12 000 steps (run explicitly)"]
    fn falsifier_decay_groks_and_no_decay_does_not() {
        let t: Vec<Trial> = std::thread::scope(|s| FRESH_SEEDS.iter().map(|&sd| s.spawn(move || trial(sd))).collect::<Vec<_>>().into_iter().map(|h| h.join().unwrap()).collect());
        for x in t.iter() {
            let (d, n) = (x.decay, x.control);
            std::println!("  seed {}: D fit {:?} (test {:.3}) gen {:?} end {:.3} event {} | norm peak {:.0} -> at gen {:.0} | conc {:.2} -> {:.2} || N fit {:?} gen {:?} end {:.3} event {} || pass {}",
                x.seed, d.t_fit, d.test_at_fit, d.t_gen, d.end_test, d.event, d.peak_norm2, d.norm2_at_gen, d.conc_at_fit, d.conc_at_gen, n.t_fit, n.t_gen, n.end_test, n.event, x.pass);
        }
        let n = t.iter().filter(|x| x.pass).count();
        let ratio: Vec<f32> = t.iter().filter_map(|x| match (x.decay.t_fit, x.decay.t_gen) { (Some(f), Some(g)) => Some(g as f32 / f as f32), _ => None }).collect();
        std::println!("FALSIFIER: {}/10 seeds pass -> {} | delay t_gen/t_fit {:?} | norm fell before gen on {}/10 | conc rose >= 0.1 on {}/10", n, if n >= PASS_SEEDS { "PASS" } else { "FAIL" }, ratio,
            t.iter().filter(|x| x.decay.norm2_at_gen < x.decay.peak_norm2).count(), t.iter().filter(|x| x.decay.conc_at_gen >= x.decay.conc_at_fit + 0.1).count());
        assert!(n >= PASS_SEEDS, "{}/10 seeds pass (need {})", n, PASS_SEEDS);
    }

    fn env<T: core::str::FromStr>(k: &str, d: T) -> T { std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }

    /// Dev runs (env GB_*), dev seeds only: cargo test --release --lib brain_grokbed::tests::dev -- --ignored --nocapture
    #[test]
    #[ignore]
    fn dev() {
        let cfg = Config { p: env("GB_P", 23), hidden: env("GB_H", 128), n_train: env("GB_N", 212),
            act: if env("GB_QUAD", 0) == 1 { Act::Quad } else { Act::Relu }, init: env("GB_INIT", 1.0),
            lr: env("GB_LR", 1e-3), wd: env("GB_WD", 1.0), steps: env("GB_STEPS", 20000), eval_every: env("GB_EVERY", 500) };
        let seeds: Vec<u64> = (0..env("GB_SEEDS", 4u64)).map(|i| 100 + i).collect();
        assert!(seeds.iter().all(|s| !FRESH_SEEDS.contains(s)), "fresh seeds are run only by the falsifier");
        std::println!("{:?}", cfg);
        let runs: Vec<Vec<Point>> = std::thread::scope(|s| seeds.iter().map(|&sd| s.spawn(move || run(sd, &cfg))).collect::<Vec<_>>().into_iter().map(|h| h.join().unwrap()).collect());
        for (sd, c) in seeds.iter().zip(runs.iter()) {
            std::println!("seed {}", sd);
            for pt in c.iter() {
                std::println!("  {:>6} tr {:.3} te {:.3} trl {:.4} tel {:.3} |w|2 {:>8.1} conc {:.3} act {:>3}/{}", pt.step, pt.train_acc, pt.test_acc, pt.train_loss, pt.test_loss, pt.norm2, pt.conc, pt.active, pt.hidden);
            }
        }
    }
}
