//! Growth gated by grokking: grow -> compress -> grokking event -> grow bigger.
//!
//! Phil (2026-10-06): "it can grow, but it has to grow and compress and compress, reach a grokking
//! event and then it can grow to become a bigger model". [`crate::brain_grokbed`] showed the event
//! is real in a network with a learned representation (decay groks on 10/10 fresh seeds, no decay
//! on 0/10). This module tests the growth rule built on it:
//!
//! - **Task 1**: `(a + b) mod p` ([`crate::brain_grokbed`]'s world). A validation slice is carved
//!   from its training pairs; the slice is never trained on and is the ONLY thing the online
//!   detector reads besides training accuracy and the weight norm (the test sets drive nothing).
//! - **Compress + generalize event** (online, every measurement): train accuracy >= `fit`,
//!   validation accuracy >= `val_ok` for `val_hold` consecutive measurements, and the squared weight
//!   norm at most `1 - norm_drop` of its running peak (the network compressed).
//! - **At the growth point**: structural compression ([`Net::prune`]: neurons whose weights decayed
//!   to nothing are removed), then growth: `grow_by` new neurons ([`Net::grow`], function unchanged)
//!   and a new output head ([`Net::add_op`]): **task 2** = `(a - b) mod p`, with a small fixed
//!   training set. From then on the network trains on both tasks' training pairs.
//! - **Arms** (same seed: same splits, same initial weights for the shared part, same total steps):
//!   - **G (gated)**: grows at the event (or at `cap` steps if it never fires);
//!   - **U (ungated)**: grows as soon as task 1's training set is fitted (memorized, before grokking);
//!   - **S (scratch)**: the final-size network (`h + grow_by` neurons, 2 ops) trained on both tasks
//!     from step 0.
//! - **Measured**: `t_both` = first step at which BOTH tasks' test accuracy is >= 0.95 (total steps
//!   from the start, so G pays for the steps it spent on task 1).
//!
//! ## Pre-registration (written and committed before any fresh seed was run)
//!
//! - **World** [`PREREG`]: task 1 as in [`crate::brain_grokbed::PREREG`] (`p = 23`, quadratic,
//!   128 neurons, AdamW `lr 1e-3`, decay 1.0) with 290 training pairs of which 25 are the
//!   validation slice (so 265 are trained, as in the grokbed falsifier); growth = prune (< 1 % of
//!   the largest neuron) + 64 neurons + a task-2 head; 16 000 total steps; G's cap 8 000. Gate
//!   thresholds [`GateCfg::DEFAULT`]: train >= 0.99, validation >= 0.95 for 3 measurements, norm
//!   <= 0.95 x peak.
//! - **Primary claim (R1, [`R1`], 53 task-2 training pairs = 10 %)**: G ends (mean task-2 test
//!   accuracy of the last 10 measurements) above S on >= 20 of the 30 fresh seeds [`FRESH_SEEDS`]
//!   AND above U on >= 20 of 30 (each a one-sided sign test at p <= 0.05; a tie is a loss).
//! - **Secondary (R2, [`R2`], 132 pairs = 25 %)**: the same comparisons, plus `t_both` (G faster
//!   than S / U) and parameter-steps to `t_both` (G cheaper than S); reported, not judged.
//! - **Detector check** (yin-23's ask: a negative control for the gate): task 1 only, 12 000
//!   steps, the gate watching, 30 fresh seeds per arm. Positive: the grokking world. Negative 1: no
//!   weight decay. Negative 2: decay with 30 % data (184 pairs incl. the slice; it compresses but did
//!   not generalize within 12 000 steps on dev seeds). A FALSE POSITIVE is a firing while task 1's
//!   test accuracy at that measurement is < 0.90 (the test set is read only to score the gate).
//!   **Pass iff 0 false positives in all 90 runs and >= 27 of 30 positives fire with test >= 0.90.**
//! - **Development** (dev seeds 100..102 only). First design: a task-2 op TOKEN added to the
//!   hidden pre-activation. Every arm failed, scratch included: with a quadratic neuron
//!   `(A(a) + B(b) + O(op))^2` the `a x b` interaction does not depend on the op, so no op token
//!   can switch `+` to `-` (an instrument bug, fixed by one output head per op: the grokked
//!   neurons' `A B` products already contain `cos w (a - b)`). With heads, dev results: R2 (25 %):
//!   `t_both` S 2 500..3 500 < U 3 200..4 000 < G 5 100..6 600 steps; R1 (10 %): end task-2 test
//!   G 0.78 / 0.87 / 0.86, U 0.78 / 0.90 / 0.87, S 0.81 / 0.88 / 0.89 (G last on 6 of 6
//!   comparisons); 5 % and 15 % look the same. Every arm keeps training task 1, so every arm groks
//!   it by ~4 000 steps and shares its Fourier neurons with the task-2 head: gating buys no edge.
//! - **Analysis prediction: the primary claim FAILS** (G does not beat S and U), on the dev
//!   evidence above. A FAIL here falsifies "growing only after the grokking event yields a better
//!   bigger model" in this world; it does not touch step 2 (the event itself is real) or the gate's
//!   value as a detector, which is tested separately.
//!
//! ## RESULT (added after the one run; rule, thresholds and seeds unchanged; raven CPU, 40 threads)
//!
//! **Growth claim: FAIL, as predicted.** **Gate check: FAIL, narrowly** (one firing at test 0.895).
//!
//! | regime | end task-2 test, mean G / U / S | G > S | G > U | `t_both`: G faster than S / U | G events |
//! |--------|----------------------------------|-------|-------|-------------------------------|----------|
//! | R1 (10 %, primary) | 0.820 / 0.823 / 0.826 | 9/30 (p 0.99) | 12/30 (p 0.90) | 0/30 / 0/30 (no arm reaches 0.95 on task 2) | 30/30 |
//! | R2 (25 %) | 0.994 / 0.994 / 0.995 | 3/30 (p 1.00) | 6/30 (p 1.00) | 0/30 / 0/30 | 30/30 |
//!
//! R2: G is cheaper than S in parameter-steps to `t_both` on 5/30. (R1's "cheaper 30/30" is an
//! artifact of the cap: no arm reached `t_both`, so it only counts G's smaller early net; not claimed.)
//! Growing only after the grokking event does not give a better (or faster) bigger model here:
//! every arm keeps training task 1, so every arm groks it and shares the same neurons. The structural
//! prune at the growth point removed 0 neurons on every seed: compression here is a smaller weight
//! norm and a few dominant frequencies, not dead neurons.
//!
//! **Gate (90 runs, 12 000 steps, 37 s):** positives fired with test >= 0.90 on 29/30; the 30th
//! (seed 40151) fired at step 3 200 with test 0.895 and ended at 1.000 (it fired mid-rise, 0.005
//! under the bar): 1 false positive by the pre-registered rule. Negative arms: **0 firings in 60
//! runs**, including all 30 low-data runs, whose weight norm ended 16-18 % below its peak: they
//! compress without generalizing, and the train-carved validation slice keeps the gate shut.
//! Compression alone is not a grokking signal; the validation condition is what does the work.
//! A stricter gate would need a new pre-registration and new seeds.
//!
//! ## Pre-registration 3b: continual world, no rehearsal (committed before any 3b seed was run)
//!
//! yin-23's hypothesis: gating should matter where growth INTERFERES, e.g. when task 1's data is
//! gone after the growth point. Fair version (equal task-1 exposure; only the growth time differs):
//!
//! - **Switch step** per seed = the gate's event on a task-1-only run (cap 8 000): [`switch_step`].
//!   Task 1 is trained until the switch, task 2 ONLY after it (no task-1 data), 16 000 steps total,
//!   settings [`CONT`] (= R2: 132 task-2 pairs). [`run_continual`].
//! - **Arms**: G grows (prune + 64 neurons) at the switch; U grows at the first fit (~step 300) and
//!   keeps training task 1 until the same switch; N never grows. All add the task-2 head at the switch.
//! - **Claim**: G's end task-1 test accuracy (retention) > U's on >= 20 of the 30 fresh seeds
//!   [`CONT_SEEDS`] (sign test p <= 0.05; a tie is a loss) AND G's mean task-2 test accuracy is at
//!   least U's minus 0.05. Reported: the same against N.
//! - **Dev** (seeds 100..102): retention G 0.79 / 0.82 / 0.80, U 0.85 / 0.75 / 0.85, N 0.84 /
//!   0.84 / 0.82; acquisition G 0.29 / 0.26 / 0.18 (lowest on 3/3), U 0.37 / 0.32 / 0.35, N 0.37 /
//!   0.39 / 0.30. With 53 task-2 pairs G retained worst on 3/3. G's 64 new neurons arrive exactly
//!   when task 2 starts, with random embeddings, and are trained by task 2 alone.
//! - **Analysis prediction: FAIL** (G retains no better than U, and acquires worse).
//!
//! ## RESULT 3b (added after the one run; raven CPU, 40 threads, 78 s)
//!
//! **FAIL, as predicted, decisively.** Retention (end task-1 test) mean G 0.775, U 0.828, N 0.840;
//! G > U on 1/30 seeds (p 1.00), G > N on 0/30. Acquisition (end task-2 test) mean G 0.226, U 0.318,
//! N 0.349. In this continual world growing at the grokking event is the WORST option on both
//! counts, and not growing is the best: new random neurons that arrive with the new task are
//! shaped by the new task alone and pull the shared readout toward it.
//!
//! ## Pre-registration 3c: grow on a measured capacity shortfall (the #14 successor; committed before its seeds ran)
//!
//! - **World** [`SHORT`]: task 1 only (grokbed world, 265 trained + 25 validation pairs), start at 8
//!   neurons, double (function unchanged) up to 128 when the trigger metric improved by < 0.01 over
//!   the last 10 measurements while below its target; 24 000 steps. [`run_shortfall`].
//! - **Arms**: Val (trigger = validation accuracy below 0.95), Big (128 from the start), Train
//!   (trigger = training accuracy below 0.99), Small (8, never grows).
//! - **Claim**: Val reaches test >= 0.95 with fewer parameter-steps than Big on >= 20 of 30 fresh
//!   seeds [`SF_SEEDS`] (sign test; a Val run that never generalizes loses). Reported: generalized
//!   counts and end test per arm; Train > Small.
//! - **Dev** (100..102): capacity sweep at 12 000 steps: 4 neurons never fit; 8 fit 0.84..0.95; 16 fit
//!   at ~1 500 but test 0.41..0.54; 32 test 0.77..0.89; 128 groks to 1.000 at ~4 000. Fitting is not
//!   generalizing: generalization needs spare capacity. At 24 000 steps: Train stops at 16 neurons
//!   (test 0.45..0.50); Val reaches 128 at 8 000..11 000 and generalizes at 13 900 / 18 900 / never
//!   (end 0.98 / 0.90 / 1.00 for seeds 100 / 101 / 102); Big generalizes at 3 700..4 100 (~33-36 M
//!   parameter-steps), Val's total by 24 000 steps 134..154 M.
//! - **Analysis prediction: FAIL** (Big is cheaper to generalization).
//!
//! ## RESULT 3c (added after the one run; raven CPU, 40 threads, 62 s)
//!
//! **FAIL, as predicted.** Val-triggered growth reached test 0.95 with fewer parameter-steps than
//! Big on 3/30 seeds. Generalized: Val 24/30, Big 30/30, Train 0/30, Small 0/30. Mean end test Val
//! 0.969, Big 1.000, Train 0.442, Small 0.306. Train-triggered growth beats staying small (29/30)
//! but stops at the size that fits and memorizes. Growing on a measured shortfall gets there late:
//! spare capacity from the start is both faster and cheaper.

//!
//! ## Pre-registration 3d: the gate as a STOP / consolidate signal (#18; committed before its seeds ran)
//!
//! - **Rule** ([`stop_run`]): train task 1 with the gate watching; stop [`CONSOLIDATE`] = 2 000 steps
//!   after it fires; budget [`STOP_BUDGET`] = 12 000 (a run whose gate never fires trains the budget).
//! - **Claim**, 30 fresh seeds [`ST_SEEDS`]: on the grokking world (290 / decay 1) test accuracy at
//!   the stop is >= the full-budget test accuracy - 0.02 on >= 27/30 AND the stop comes at <= 60 % of
//!   the budget on >= 27/30; on the 60 negative runs (no decay; 30 % data) the gate stops none early.
//! - **Dev** (100..105): fired at 3 100..4 400; test at fire 0.85..0.97, +1 000: 0.958..0.992,
//!   +2 000: 0.996 on 6/6, budget 1.000; negatives never fired (12/12).
//! - **Analysis prediction: PASS** (~50 % of the compute saved for <= 0.004 accuracy).
//!
//! ## RESULT 3d
//!
//! (added after the one run)







use std::vec::Vec;

use crate::brain_grokbed::{split_op, Act, Net, Pair, Rng};
use crate::grok_gate::{GateCfg, GrokGate};

/// Settings of one growth experiment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GrowCfg {
    pub p: usize,
    /// Task 1 training pairs (including the validation slice).
    pub n_train1: usize,
    /// Validation pairs carved from task 1's training pairs.
    pub n_val: usize,
    /// Task 2 training pairs.
    pub n_train2: usize,
    pub hidden: usize,
    pub grow_by: usize,
    pub act: Act,
    pub init: f32,
    pub lr: f32,
    pub wd: f32,
    /// Total steps for every arm.
    pub total: usize,
    pub eval_every: usize,
    /// G grows at this step at the latest.
    pub cap: usize,
    pub fit: f32,
    pub val_ok: f32,
    pub val_hold: u8,
    pub norm_drop: f32,
    /// Prune neurons below this fraction of the largest squared norm at the growth point.
    pub prune_frac: f32,
}

/// Which arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arm { Gated, Ungated, Scratch }

/// One measurement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GPoint {
    pub step: u32,
    pub train1: f32,
    pub val: f32,
    pub test1: f32,
    pub train2: f32,
    pub test2: f32,
    pub norm2: f32,
    pub hidden: u16,
}

/// One arm's run.
#[derive(Clone, Debug, PartialEq)]
pub struct GrowRun {
    pub arm: Arm,
    pub curve: Vec<GPoint>,
    /// Step at which the network grew (S: 0).
    pub grow_step: Option<u32>,
    /// Whether G's event fired (false for U and S).
    pub event: bool,
    /// Neurons removed by the structural compression at the growth point.
    pub pruned: usize,
    /// Parameters just before pruning, after pruning, and after growth.
    pub params: [usize; 3],
}

/// Test accuracy both tasks must reach.
pub const BOTH: f32 = 0.95;

impl GrowRun {
    /// First step with both tasks' test accuracy >= [`BOTH`].
    pub fn t_both(&self) -> Option<u32> { self.curve.iter().find(|p| p.test1 >= BOTH && p.test2 >= BOTH).map(|p| p.step) }
    /// First step with task 2's test accuracy >= [`BOTH`].
    pub fn t_task2(&self) -> Option<u32> { self.curve.iter().find(|p| p.test2 >= BOTH).map(|p| p.step) }
    /// Mean of the last `k` measurements of a field.
    pub fn end(&self, k: usize, f: impl Fn(&GPoint) -> f32) -> f32 {
        let k = k.min(self.curve.len()).max(1);
        self.curve[self.curve.len() - k..].iter().map(f).sum::<f32>() / k as f32
    }
}

/// The seed's data: task 1 train (without the slice), validation slice, task 1 test, task 2 train, task 2 test.
pub struct Data { pub tr1: Vec<Pair>, pub val: Vec<Pair>, pub te1: Vec<Pair>, pub tr2: Vec<Pair>, pub te2: Vec<Pair> }

pub fn data(seed: u64, c: &GrowCfg) -> Data {
    let (mut tr1, te1) = split_op(seed, c.p, c.n_train1, 0);
    let val = tr1.split_off(c.n_train1 - c.n_val.min(c.n_train1));
    let (tr2, te2) = split_op(seed, c.p, c.n_train2, 1);
    Data { tr1, val, te1, tr2, te2 }
}

fn point(net: &Net, step: usize, d: &Data) -> GPoint {
    let two = net.ops > 1;
    GPoint { step: step as u32, train1: net.eval(&d.tr1).0, val: net.eval(&d.val).0, test1: net.eval(&d.te1).0,
        train2: if two { net.eval(&d.tr2).0 } else { 0.0 }, test2: if two { net.eval(&d.te2).0 } else { 0.0 },
        norm2: net.norm2(), hidden: net.h as u16 }
}

/// Run one arm.
pub fn run_arm(seed: u64, c: &GrowCfg, arm: Arm) -> GrowRun {
    let d = data(seed, c);
    let mut both: Vec<Pair> = d.tr1.clone();
    both.extend_from_slice(&d.tr2);
    let mut rng = Rng::new(seed);
    let mut net = Net::new(c.p, c.hidden, c.act, c.init, &mut rng);
    let mut grow_rng = Rng::new(seed ^ 0x6A0F_0000);
    let (mut grown, mut grow_step, mut event, mut pruned, mut params) = (false, None, false, 0usize, [0usize; 3]);
    if arm == Arm::Scratch {
        net.grow(c.grow_by, c.init, &mut grow_rng);
        // scratch: the grown neurons get random output weights like the rest (a fresh network)
        let (p, h) = (net.p, net.h);
        let s = 1.0 / (h as f32).sqrt();
        for r in 0..p { for j in c.hidden..h { net.w[2 * p * h + r * h + j] = s * grow_rng.normal(); } }
        net.add_op();
        let h1 = 3 * p * h;
        for x in net.w[h1..].iter_mut() { *x = s * grow_rng.normal(); }
        grown = true; grow_step = Some(0);
        params = [net.params(); 3];
    }
    let mut curve = Vec::with_capacity(c.total / c.eval_every.max(1) + 2);
    curve.push(point(&net, 0, &d));
    let mut gate = GrokGate::new(GateCfg { fit: c.fit, val_ok: c.val_ok, val_hold: c.val_hold, drop: c.norm_drop });
    for s in 1..=c.total {
        net.step(if grown { &both } else { &d.tr1 }, c.lr, c.wd);
        if s % c.eval_every != 0 { continue; }
        let pt = point(&net, s, &d);
        curve.push(pt);
        if grown { continue; }
        let e = gate.observe(pt.train1, pt.val, pt.norm2);
        let fire = match arm {
            Arm::Gated => {
                if e { event = true; }
                e || s >= c.cap
            }
            Arm::Ungated => pt.train1 >= c.fit,
            Arm::Scratch => false,
        };
        if fire {
            params[0] = net.params();
            pruned = net.prune(c.prune_frac);
            params[1] = net.params();
            net.grow(c.grow_by, c.init, &mut grow_rng);
            net.add_op();
            params[2] = net.params();
            grown = true; grow_step = Some(s as u32);
        }
    }
    GrowRun { arm, curve, grow_step, event, pruned, params }
}

/// All three arms of one seed.
#[derive(Clone, Debug, PartialEq)]
pub struct GrowTrial { pub seed: u64, pub g: GrowRun, pub u: GrowRun, pub s: GrowRun }

pub fn trial(seed: u64, c: &GrowCfg) -> GrowTrial {
    GrowTrial { seed, g: run_arm(seed, c, Arm::Gated), u: run_arm(seed, c, Arm::Ungated), s: run_arm(seed, c, Arm::Scratch) }
}

/// Pre-registered base settings (task 2 size set per regime).
pub const PREREG: GrowCfg = GrowCfg { p: 23, n_train1: 290, n_val: 25, n_train2: 53, hidden: 128, grow_by: 64, act: Act::Quad, init: 0.1,
    lr: 1e-3, wd: 1.0, total: 16_000, eval_every: 100, cap: 8_000, fit: 0.99, val_ok: 0.95, val_hold: 3, norm_drop: 0.05, prune_frac: 0.01 };
/// Regime R1 (primary): task 2 has 53 training pairs (10 %).
pub const R1: GrowCfg = PREREG;
/// Regime R2 (secondary): task 2 has 132 training pairs (25 %).
pub const R2: GrowCfg = GrowCfg { n_train2: 132, ..PREREG };
/// Pre-registered fresh seeds: 30, never run before the pre-registration was committed.
pub const FRESH_SEEDS: [u64; 30] = [40009, 40013, 40031, 40037, 40039, 40063, 40087, 40093, 40099, 40111, 40123, 40127, 40129, 40151, 40153,
    40163, 40169, 40177, 40189, 40193, 40213, 40231, 40237, 40241, 40253, 40277, 40283, 40289, 40343, 40351];
/// Wins of 30 for a one-sided sign test p <= 0.05 (`P(X >= 20 | 30, 1/2) = 0.0494`).
pub const MIN_WINS: usize = 20;
/// Evaluations averaged for "end" accuracy.
pub const END_EVALS: usize = 10;

/// One regime's primary comparison: wins of G over another arm on task 2's end test accuracy.
pub fn wins_end_task2(t: &[GrowTrial], other: impl Fn(&GrowTrial) -> &GrowRun) -> usize {
    t.iter().filter(|x| x.g.end(END_EVALS, |p| p.test2) > other(x).end(END_EVALS, |p| p.test2)).count()
}

/// `t_both` with "never" scored as `total + 1`.
pub fn t_both_or_cap(r: &GrowRun, total: usize) -> u32 { r.t_both().unwrap_or(total as u32 + 1) }

/// Detector check: task 1 only, the gate watching, no growth. `fired_at` and test accuracy there.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GateRun { pub fired_at: Option<u32>, pub test_at_fire: f32, pub end_test: f32, pub norm_drop: f32 }

/// Run task 1 with the pre-registered gate for `steps`; never grows.
pub fn gate_run(seed: u64, n_train1: usize, wd: f32, steps: usize) -> GateRun {
    let c = GrowCfg { n_train1, wd, total: steps, cap: usize::MAX, ..PREREG };
    let d = data(seed, &c);
    let mut rng = Rng::new(seed);
    let mut net = Net::new(c.p, c.hidden, c.act, c.init, &mut rng);
    let mut gate = GrokGate::new(GateCfg::DEFAULT);
    let (mut fired_at, mut test_at_fire, mut last) = (None, 0.0f32, point(&net, 0, &d));
    for s in 1..=steps {
        net.step(&d.tr1, c.lr, c.wd);
        if s % c.eval_every != 0 { continue; }
        last = point(&net, s, &d);
        if gate.observe(last.train1, last.val, last.norm2) { fired_at = Some(s as u32); test_at_fire = last.test1; }
    }
    GateRun { fired_at, test_at_fire, end_test: last.test1, norm_drop: 1.0 - last.norm2 / gate.peak().max(1e-30) }
}

/// Detector arms: positive (the grokking world), negative 1 (no weight decay), negative 2 (decay,
/// 30 % training data: compresses but does not generalize within the budget).
pub const GATE_STEPS: usize = 12_000;
pub const GATE_POS: (usize, f32) = (290, 1.0);
pub const GATE_NEG_NODECAY: (usize, f32) = (290, 0.0);
pub const GATE_NEG_LOWDATA: (usize, f32) = (184, 1.0);
/// A firing is a false positive when task 1's test accuracy at that measurement is below this.
pub const TRUE_GEN: f32 = 0.90;

/// Continual pre-registration (3b): settings = [`R2`] (132 task-2 pairs), task 1 data gone after the switch.
pub const CONT: GrowCfg = R2;
/// 3b fresh seeds (disjoint from [`FRESH_SEEDS`] and every earlier set).
pub const CONT_SEEDS: [u64; 30] = [50021, 50023, 50033, 50047, 50051, 50053, 50069, 50077, 50087, 50093, 50101, 50111, 50119, 50123, 50129,
    50131, 50147, 50153, 50159, 50177, 50207, 50221, 50227, 50231, 50261, 50263, 50273, 50287, 50291, 50311];
/// G may acquire task 2 at most this much worse than U (mean) for the claim to hold.
pub const ACQ_SLACK: f32 = 0.05;

/// Continual world (no rehearsal): arms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContArm {
    /// Grows at the gate's event, then trains on task 2 only.
    Gated,
    /// Grows at the first fit, keeps training task 1 until the same switch step, then task 2 only.
    Ungated,
    /// Never grows (adds only the task-2 head at the switch step), then task 2 only.
    NoGrowth,
}

/// One continual run: the switch step and the end accuracies.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContRun {
    pub arm: ContArm,
    pub switch: u32,
    pub grow_step: Option<u32>,
    /// Task 1 test accuracy at the switch (what there is to retain) and at the end.
    pub test1_at_switch: f32,
    pub retain: f32,
    /// Task 2 test accuracy at the end.
    pub acquire: f32,
}

/// The switch step of a seed: G's gate event (or the cap), from a task-1-only run with the gate.
pub fn switch_step(seed: u64, c: &GrowCfg) -> u32 {
    let g = gate_run_cfg(seed, c, c.cap);
    g.fired_at.unwrap_or(c.cap as u32)
}

fn gate_run_cfg(seed: u64, c: &GrowCfg, steps: usize) -> GateRun {
    let d = data(seed, c);
    let mut rng = Rng::new(seed);
    let mut net = Net::new(c.p, c.hidden, c.act, c.init, &mut rng);
    let mut gate = GrokGate::new(GateCfg { fit: c.fit, val_ok: c.val_ok, val_hold: c.val_hold, drop: c.norm_drop });
    let (mut fired_at, mut test_at_fire, mut last) = (None, 0.0f32, point(&net, 0, &d));
    for s in 1..=steps {
        net.step(&d.tr1, c.lr, c.wd);
        if s % c.eval_every != 0 { continue; }
        last = point(&net, s, &d);
        if gate.observe(last.train1, last.val, last.norm2) { fired_at = Some(s as u32); test_at_fire = last.test1; break; }
    }
    GateRun { fired_at, test_at_fire, end_test: last.test1, norm_drop: 1.0 - last.norm2 / gate.peak().max(1e-30) }
}

/// Run one continual arm with a given switch step (task 1 until `switch`, task 2 only after).
pub fn run_continual(seed: u64, c: &GrowCfg, arm: ContArm, switch: u32) -> ContRun {
    let d = data(seed, c);
    let mut rng = Rng::new(seed);
    let mut net = Net::new(c.p, c.hidden, c.act, c.init, &mut rng);
    let mut grow_rng = Rng::new(seed ^ 0x6A0F_0000);
    let (mut grow_step, mut test1_at_switch) = (None, 0.0f32);
    for s in 1..=c.total {
        let phase2 = s as u32 > switch;
        net.step(if phase2 { &d.tr2 } else { &d.tr1 }, c.lr, c.wd);
        if s % c.eval_every == 0 && grow_step.is_none() && arm == ContArm::Ungated && net.eval(&d.tr1).0 >= c.fit {
            net.prune(c.prune_frac); net.grow(c.grow_by, c.init, &mut grow_rng); grow_step = Some(s as u32);
        }
        if s as u32 == switch {
            test1_at_switch = net.eval(&d.te1).0;
            match arm {
                ContArm::Gated => { net.prune(c.prune_frac); net.grow(c.grow_by, c.init, &mut grow_rng); grow_step = Some(s as u32); }
                ContArm::Ungated => { if grow_step.is_none() { net.prune(c.prune_frac); net.grow(c.grow_by, c.init, &mut grow_rng); grow_step = Some(s as u32); } }
                ContArm::NoGrowth => {}
            }
            net.add_op();
        }
    }
    let (retain, acquire) = (net.eval(&d.te1).0, if net.ops > 1 { net.eval(&d.te2).0 } else { 0.0 });
    ContRun { arm, switch, grow_step, test1_at_switch, retain, acquire }
}

/// Capacity-shortfall growth (the #14 successor): which signal triggers a doubling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Short {
    /// Double while TRAINING accuracy has plateaued below `fit`.
    Train,
    /// Double while VALIDATION accuracy (slice carved from training data) has plateaued below `val_ok`.
    Val,
    /// Never grow (stays at the start size).
    Small,
    /// Start at the maximum size.
    Big,
}

/// Settings of the shortfall experiment (task 1 only).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShortCfg {
    pub base: GrowCfg,
    pub h0: usize,
    pub h_max: usize,
    /// A plateau = the trigger metric improved by less than `delta` over the last `window` measurements.
    pub window: usize,
    pub delta: f32,
}

/// One shortfall run.
#[derive(Clone, Debug, PartialEq)]
pub struct ShortRun {
    pub mode: Short,
    pub grows: Vec<u32>,
    pub final_h: usize,
    pub end_test: f32,
    pub end_train: f32,
    /// First step with test >= [`BOTH`] (0.95), if any.
    pub t_gen: Option<u32>,
    /// Sum over steps of parameters in use (compute proxy).
    pub param_steps: f64,
    /// Parameter-steps spent until `t_gen` (None: never generalized).
    pub param_steps_at_gen: Option<f64>,
}

/// Run one shortfall arm.
pub fn run_shortfall(seed: u64, s: &ShortCfg, mode: Short) -> ShortRun {
    let c = &s.base;
    let d = data(seed, c);
    let mut rng = Rng::new(seed);
    let mut net = Net::new(c.p, if mode == Short::Big { s.h_max } else { s.h0 }, c.act, c.init, &mut rng);
    let mut grow_rng = Rng::new(seed ^ 0x5F0F_0000);
    let (mut grows, mut hist, mut param_steps, mut t_gen, mut ps_gen) = (Vec::new(), Vec::<f32>::new(), 0.0f64, None, None);
    let mut tail: Vec<f32> = Vec::new();
    for st in 1..=c.total {
        net.step(&d.tr1, c.lr, c.wd);
        param_steps += net.params() as f64;
        if st % c.eval_every != 0 { continue; }
        let (tr, va, te) = (net.eval(&d.tr1).0, net.eval(&d.val).0, net.eval(&d.te1).0);
        if t_gen.is_none() && te >= BOTH { t_gen = Some(st as u32); ps_gen = Some(param_steps); }
        tail.push(te); if tail.len() > END_EVALS { tail.remove(0); }
        let (metric, target) = match mode { Short::Train => (tr, c.fit), Short::Val => (va, c.val_ok), _ => continue };
        hist.push(metric);
        if hist.len() > s.window && metric < target && net.h < s.h_max {
            let then = hist[hist.len() - 1 - s.window];
            let best_recent = hist[hist.len() - s.window..].iter().cloned().fold(f32::MIN, f32::max);
            if best_recent - then < s.delta {
                let add = net.h.min(s.h_max - net.h);
                net.grow(add, c.init, &mut grow_rng);
                grows.push(st as u32);
                hist.clear();
            }
        }
    }
    let end_test = tail.iter().sum::<f32>() / tail.len().max(1) as f32;
    ShortRun { mode, grows, final_h: net.h, end_test, end_train: net.eval(&d.tr1).0, t_gen, param_steps, param_steps_at_gen: ps_gen }
}

/// Pre-registered shortfall settings (#14 successor): task 1, start 8 neurons, double up to 128,
/// plateau = < 0.01 improvement over 10 measurements, 24 000 steps.
pub const SHORT: ShortCfg = ShortCfg { base: GrowCfg { total: 24_000, ..PREREG }, h0: 8, h_max: 128, window: 10, delta: 0.01 };
/// Fresh seeds for the shortfall falsifier.
pub const SF_SEEDS: [u64; 30] = [63029, 63031, 63059, 63067, 63073, 63079, 63097, 63103, 63113, 63127, 63131, 63149, 63179, 63197, 63199,
    63211, 63241, 63247, 63277, 63281, 63299, 63311, 63313, 63317, 63331, 63337, 63347, 63353, 63361, 63367];

/// #18: the gate as a STOP signal. One task-1 run to `steps` with the gate watching; test accuracy is
/// recorded at the firing plus each of `extra` steps after it (consolidation) and at the full budget.
#[derive(Clone, Debug, PartialEq)]
pub struct StopRun {
    pub fired_at: Option<u32>,
    /// Test accuracy at fire + extra[i] (None if not reached or the gate never fired).
    pub test_after: Vec<Option<f32>>,
    pub test_budget: f32,
}

pub fn stop_run(seed: u64, n_train1: usize, wd: f32, steps: usize, extra: &[u32]) -> StopRun {
    let c = GrowCfg { n_train1, wd, total: steps, cap: usize::MAX, ..PREREG };
    let d = data(seed, &c);
    let mut rng = Rng::new(seed);
    let mut net = Net::new(c.p, c.hidden, c.act, c.init, &mut rng);
    let mut gate = GrokGate::new(GateCfg::DEFAULT);
    let (mut fired_at, mut test_after, mut last) = (None, vec![None; extra.len()], 0.0f32);
    for s in 1..=steps {
        net.step(&d.tr1, c.lr, c.wd);
        if s % c.eval_every != 0 { continue; }
        let pt = point(&net, s, &d);
        last = pt.test1;
        if fired_at.is_none() && gate.observe(pt.train1, pt.val, pt.norm2) { fired_at = Some(s as u32); }
        if let Some(f) = fired_at { for (i, &e) in extra.iter().enumerate() { if s as u32 == f + e { test_after[i] = Some(pt.test1); } } }
    }
    StopRun { fired_at, test_after, test_budget: last }
}

/// #18 rule: stop [`CONSOLIDATE`] steps after the gate fires (budget [`STOP_BUDGET`]).
pub const CONSOLIDATE: u32 = 2_000;
pub const STOP_BUDGET: usize = 12_000;
/// #18 fresh seeds.
pub const ST_SEEDS: [u64; 30] = [64013, 64019, 64033, 64037, 64063, 64067, 64081, 64091, 64109, 64123, 64151, 64153, 64157, 64171, 64187,
    64189, 64217, 64223, 64231, 64237, 64271, 64279, 64283, 64301, 64303, 64319, 64327, 64333, 64373, 64381];

#[cfg(test)]
mod tests {
    use super::*;

    fn env<T: core::str::FromStr>(k: &str, d: T) -> T { std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }

    fn dev_cfg() -> GrowCfg {
        GrowCfg { p: 23, n_train1: env("GR_N1", 265), n_val: env("GR_VAL", 25), n_train2: env("GR_N2", 132), hidden: env("GR_H", 128), grow_by: env("GR_GROW", 64),
            act: Act::Quad, init: 0.1, lr: 1e-3, wd: env("GR_WD", 1.0), total: env("GR_TOTAL", 16000), eval_every: 100, cap: env("GR_CAP", 8000),
            fit: 0.99, val_ok: env("GR_VALOK", 0.95), val_hold: env("GR_HOLD", 3), norm_drop: env("GR_DROP", 0.05), prune_frac: env("GR_PRUNE", 0.01) }
    }

    #[test]
    fn data_is_disjoint_and_sized() {
        let c = GrowCfg { n_train1: 100, n_val: 10, n_train2: 50, ..dev_cfg() };
        let d = data(4, &c);
        assert_eq!((d.tr1.len(), d.val.len(), d.te1.len(), d.tr2.len(), d.te2.len()), (90, 10, 429, 50, 479));
        for v in d.val.iter() { assert!(!d.tr1.contains(v) && !d.te1.contains(v)); }
        assert!(d.tr2.iter().chain(d.te2.iter()).all(|x| x.2 == 1));
    }

    #[test]
    fn arms_grow_when_they_should() {
        let c = GrowCfg { p: 7, n_train1: 30, n_val: 5, n_train2: 20, hidden: 16, grow_by: 8, total: 600, eval_every: 50, cap: 400, ..dev_cfg() };
        let (g, u, s) = (run_arm(1, &c, Arm::Gated), run_arm(1, &c, Arm::Ungated), run_arm(1, &c, Arm::Scratch));
        assert_eq!(s.grow_step, Some(0));
        let fit = u.curve.iter().find(|p| p.train1 >= c.fit).map(|p| p.step);
        assert_eq!(u.grow_step, fit, "U grows at the first fitted measurement");
        assert!(if g.event { g.grow_step >= fit } else { g.grow_step == Some(c.cap as u32) }, "G grows at the event, or at the cap");
        assert!(g.grow_step.unwrap() <= 400);
        for r in [&g, &u, &s] { assert_eq!(r.curve.len(), 600 / 50 + 1); assert_eq!(r.params[2], 4 * 7 * r.curve.last().unwrap().hidden as usize); }
        assert_eq!(s.curve.last().unwrap().hidden as usize, 24);
    }

    /// Map `f` over `items` on `threads` workers (env GR_THREADS, default 16); order preserved.
    fn par_map<T: Sync, R: Send + Clone>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Mutex;
        let threads: usize = env("GR_THREADS", 16usize).max(1);
        let next = AtomicUsize::new(0);
        let out: Mutex<Vec<Option<R>>> = Mutex::new(vec![None; items.len()]);
        std::thread::scope(|s| {
            for _ in 0..threads.min(items.len()) {
                s.spawn(|| loop {
                    let i = next.fetch_add(1, Ordering::SeqCst);
                    if i >= items.len() { break; }
                    let r = f(&items[i]);
                    out.lock().unwrap()[i] = Some(r);
                });
            }
        });
        out.into_inner().unwrap().into_iter().map(|r| r.unwrap()).collect()
    }

    #[test]
    fn preregistration_is_pinned() {
        assert_eq!(PREREG, GrowCfg { p: 23, n_train1: 290, n_val: 25, n_train2: 53, hidden: 128, grow_by: 64, act: Act::Quad, init: 0.1,
            lr: 1e-3, wd: 1.0, total: 16_000, eval_every: 100, cap: 8_000, fit: 0.99, val_ok: 0.95, val_hold: 3, norm_drop: 0.05, prune_frac: 0.01 });
        assert_eq!((R1.n_train2, R2.n_train2), (53, 132));
        assert_eq!(GateCfg::DEFAULT, GateCfg { fit: PREREG.fit, val_ok: PREREG.val_ok, val_hold: PREREG.val_hold, drop: PREREG.norm_drop });
        assert_eq!(FRESH_SEEDS.len(), 30);
        let mut s = FRESH_SEEDS.to_vec(); s.sort(); s.dedup(); assert_eq!(s.len(), 30);
        for x in FRESH_SEEDS.iter() { assert!(!crate::brain_grokbed::FRESH_SEEDS.contains(x) && !crate::brain_grokbed::DEV_SEEDS.contains(x)); }
        assert!(crate::brain_grok3::sign_p(MIN_WINS, 30) <= 0.05 && crate::brain_grok3::sign_p(MIN_WINS - 1, 30) > 0.05);
        assert_eq!((GATE_STEPS, GATE_POS, GATE_NEG_NODECAY, GATE_NEG_LOWDATA, TRUE_GEN, END_EVALS), (12_000, (290, 1.0), (290, 0.0), (184, 1.0), 0.90, 10));
    }

    fn growth_falsifier(name: &str, c: &GrowCfg) -> (usize, usize) {
        let t: Vec<GrowTrial> = par_map(&FRESH_SEEDS, |&sd| trial(sd, c));
        for x in t.iter() {
            let f = |r: &GrowRun| format!("grow {:?} t_both {:?} end t1 {:.3} t2 {:.3}", r.grow_step, r.t_both(), r.end(END_EVALS, |p| p.test1), r.end(END_EVALS, |p| p.test2));
            std::println!("  {} seed {}: G[{} event {} pruned {}] U[{}] S[{}]", name, x.seed, f(&x.g), x.g.event, x.g.pruned, f(&x.u), f(&x.s));
        }
        let (ws, wu) = (wins_end_task2(&t, |x| &x.s), wins_end_task2(&t, |x| &x.u));
        let p = crate::brain_grok3::sign_p;
        let mean = |f: &dyn Fn(&GrowTrial) -> f32| t.iter().map(|x| f(x)).sum::<f32>() / t.len() as f32;
        let tb = |r: &GrowRun| t_both_or_cap(r, c.total);
        let faster = |o: &dyn Fn(&GrowTrial) -> &GrowRun| t.iter().filter(|x| tb(&x.g) < tb(o(x))).count();
        // compute to t_both in parameter-steps (G: small net until it grows)
        let cost = |r: &GrowRun, small: f64, big: f64| -> f64 { let g = r.grow_step.unwrap_or(0) as f64; let tb = tb(r) as f64; if tb <= g { small * tb } else { small * g + big * (tb - g) } };
        let (small, big) = ((4 * c.p * c.hidden) as f64 * 0.75, (4 * c.p * (c.hidden + c.grow_by)) as f64);
        let cheaper = t.iter().filter(|x| cost(&x.g, small, big) < cost(&x.s, big, big)).count();
        std::println!("{}: end task-2 test mean G {:.3} U {:.3} S {:.3} | G>S {}/30 (p {:.4}) G>U {}/30 (p {:.4}) | t_both faster G<S {}/30 G<U {}/30 | G events {}/30 | G cheaper param-steps to t_both than S {}/30",
            name, mean(&|x| x.g.end(END_EVALS, |p| p.test2)), mean(&|x| x.u.end(END_EVALS, |p| p.test2)), mean(&|x| x.s.end(END_EVALS, |p| p.test2)),
            ws, p(ws, 30), wu, p(wu, 30), faster(&|x| &x.s), faster(&|x| &x.u), t.iter().filter(|x| x.g.event).count(), cheaper);
        (ws, wu)
    }

    /// PRE-REGISTERED primary (R1): G beats S AND U on task 2's end test accuracy, >= 20/30 each.
    /// Run: GR_THREADS=40 cargo test --release --lib brain_growth::tests::falsifier_r1 -- --ignored --nocapture
    #[test]
    #[ignore = "FAILED as pre-registered (predicted): G>S 9/30, G>U 12/30 (see module docs)"]
    fn falsifier_r1_gated_growth_beats_scratch_and_ungated() {
        let (ws, wu) = growth_falsifier("R1", &R1);
        assert!(ws >= MIN_WINS && wu >= MIN_WINS, "G>S {}/30, G>U {}/30 (need {} each)", ws, wu, MIN_WINS);
    }

    /// PRE-REGISTERED secondary (R2, reported): the same comparisons with 132 task-2 pairs.
    #[test]
    #[ignore = "FAILED (reported): G>S 3/30, G>U 6/30 (see module docs)"]
    fn falsifier_r2_reported() {
        let (ws, wu) = growth_falsifier("R2", &R2);
        assert!(ws >= MIN_WINS && wu >= MIN_WINS, "G>S {}/30, G>U {}/30 (need {} each)", ws, wu, MIN_WINS);
    }

    /// PRE-REGISTERED detector check: no false positive on 60 negative runs; fires with test >= 0.90
    /// on >= 27 of 30 positive runs.
    #[test]
    #[ignore = "FAILED as pre-registered: 29/30 true fires, 1 fire at test 0.895, 0/60 negative firings (see module docs)"]
    fn falsifier_gate_detects_grokking_without_false_positives() {
        let jobs: Vec<(u64, usize, (usize, f32))> = FRESH_SEEDS.iter().flat_map(|&s| [(s, 0, GATE_POS), (s, 1, GATE_NEG_NODECAY), (s, 2, GATE_NEG_LOWDATA)]).collect();
        let r: Vec<GateRun> = par_map(&jobs, |&(s, _, (n, wd))| gate_run(s, n, wd, GATE_STEPS));
        let names = ["pos", "neg-nodecay", "neg-lowdata"];
        let (mut tp, mut fp, mut fired_neg, mut late_true) = (0usize, 0usize, 0usize, 0usize);
        for (j, g) in jobs.iter().zip(r.iter()) {
            std::println!("  gate seed {} {}: fired {:?} test there {:.3} end test {:.3} norm below peak {:.3}", j.0, names[j.1], g.fired_at, g.test_at_fire, g.end_test, g.norm_drop);
            let fired_true = g.fired_at.is_some() && g.test_at_fire >= TRUE_GEN;
            if j.1 == 0 { if fired_true { tp += 1; } }
            else if g.fired_at.is_some() { fired_neg += 1; if fired_true { late_true += 1; } }
            if g.fired_at.is_some() && g.test_at_fire < TRUE_GEN { fp += 1; }
        }
        std::println!("GATE: positives fired correctly {}/30 | false positives (fired, test < {}) {}/90 | negative-arm firings {}/60 (of which truly generalized there {})", tp, TRUE_GEN, fp, fired_neg, late_true);
        assert!(fp == 0 && tp >= 27, "tp {}/30 fp {}", tp, fp);
    }

    #[test]
    fn continual_preregistration_is_pinned() {
        assert_eq!(CONT, R2);
        let mut s = CONT_SEEDS.to_vec(); s.sort(); s.dedup(); assert_eq!(s.len(), 30);
        for x in CONT_SEEDS.iter() { assert!(!FRESH_SEEDS.contains(x) && !crate::brain_grokbed::FRESH_SEEDS.contains(x) && !crate::brain_grokbed::DEV_SEEDS.contains(x)); }
        assert_eq!(ACQ_SLACK, 0.05);
    }

    /// PRE-REGISTERED 3b (continual, no rehearsal): G retains task 1 better than U on >= 20/30 seeds
    /// AND G's mean task-2 acquisition is within 0.05 of U's.
    /// Run: GR_THREADS=40 cargo test --release --lib brain_growth::tests::falsifier_continual -- --ignored --nocapture
    #[test]
    #[ignore = "FAILED as pre-registered (predicted): retention G>U 1/30, G>N 0/30 (see module docs)"]
    fn falsifier_continual_gated_growth_retains_better() {
        let c = CONT;
        let sw: Vec<u32> = par_map(&CONT_SEEDS, |&s| switch_step(s, &c));
        let jobs: Vec<(usize, ContArm)> = (0..30).flat_map(|k| [(k, ContArm::Gated), (k, ContArm::Ungated), (k, ContArm::NoGrowth)]).collect();
        let r: Vec<ContRun> = par_map(&jobs, |&(k, a)| run_continual(CONT_SEEDS[k], &c, a, sw[k]));
        for k in 0..30 {
            let (g, u, n) = (&r[3 * k], &r[3 * k + 1], &r[3 * k + 2]);
            std::println!("  cont seed {}: switch {} | test1@switch G {:.3} U {:.3} N {:.3} | retain G {:.3} U {:.3} N {:.3} | acquire G {:.3} U {:.3} N {:.3}",
                CONT_SEEDS[k], g.switch, g.test1_at_switch, u.test1_at_switch, n.test1_at_switch, g.retain, u.retain, n.retain, g.acquire, u.acquire, n.acquire);
        }
        let arm = |a: usize| -> Vec<&ContRun> { (0..30).map(|k| &r[3 * k + a]).collect() };
        let (g, u, n) = (arm(0), arm(1), arm(2));
        let mean = |v: &[&ContRun], f: fn(&ContRun) -> f32| v.iter().map(|x| f(x)).sum::<f32>() / 30.0;
        let wins_gu = (0..30).filter(|&k| g[k].retain > u[k].retain).count();
        let wins_gn = (0..30).filter(|&k| g[k].retain > n[k].retain).count();
        let (ag, au) = (mean(&g, |x| x.acquire), mean(&u, |x| x.acquire));
        std::println!("CONTINUAL: retain mean G {:.3} U {:.3} N {:.3} | G>U {}/30 (p {:.4}) G>N {}/30 (p {:.4}) | acquire mean G {:.3} U {:.3} N {:.3}",
            mean(&g, |x| x.retain), mean(&u, |x| x.retain), mean(&n, |x| x.retain), wins_gu, crate::brain_grok3::sign_p(wins_gu, 30), wins_gn, crate::brain_grok3::sign_p(wins_gn, 30), ag, au, mean(&n, |x| x.acquire));
        assert!(wins_gu >= MIN_WINS && ag >= au - ACQ_SLACK, "G>U retain {}/30, acquire G {:.3} vs U {:.3}", wins_gu, ag, au);
    }

    #[test]
    fn shortfall_preregistration_is_pinned() {
        assert_eq!((SHORT.h0, SHORT.h_max, SHORT.window, SHORT.delta, SHORT.base.total, SHORT.base.n_train1, SHORT.base.n_val), (8, 128, 10, 0.01, 24_000, 290, 25));
        let mut s = SF_SEEDS.to_vec(); s.sort(); s.dedup(); assert_eq!(s.len(), 30);
        for x in SF_SEEDS.iter() { assert!(!FRESH_SEEDS.contains(x) && !CONT_SEEDS.contains(x) && !crate::brain_hxo::HX_SEEDS.contains(x) && !crate::brain_grokbed::F_SEEDS.contains(x)); }
    }

    /// PRE-REGISTERED (#14 successor): growth on a measured VALIDATION shortfall reaches test >= 0.95 with
    /// fewer parameter-steps than the big network from the start, on >= 20/30 seeds (never = loss).
    /// Run: GR_THREADS=40 cargo test --release --lib brain_growth::tests::falsifier_shortfall -- --ignored --nocapture
    #[test]
    #[ignore = "FAILED as pre-registered (predicted): Val cheaper than Big 3/30 (see module docs)"]
    fn falsifier_shortfall_growth_is_cheaper() {
        let jobs: Vec<(u64, Short)> = SF_SEEDS.iter().flat_map(|&x| [(x, Short::Val), (x, Short::Big), (x, Short::Train), (x, Short::Small)]).collect();
        let r: Vec<ShortRun> = par_map(&jobs, |&(x, m)| run_shortfall(x, &SHORT, m));
        for k in 0..30 {
            let (v, b, t, s) = (&r[4 * k], &r[4 * k + 1], &r[4 * k + 2], &r[4 * k + 3]);
            std::println!("  sf seed {}: Val grows {:?} end {:.3} gen {:?} Mps@gen {:?} | Big end {:.3} gen {:?} Mps@gen {:?} | Train h {} end {:.3} | Small end {:.3}", SF_SEEDS[k],
                v.grows, v.end_test, v.t_gen, v.param_steps_at_gen.map(|x| (x / 1e6).round()), b.end_test, b.t_gen, b.param_steps_at_gen.map(|x| (x / 1e6).round()), t.final_h, t.end_test, s.end_test);
        }
        let cheaper = (0..30).filter(|&k| match (r[4 * k].param_steps_at_gen, r[4 * k + 1].param_steps_at_gen) { (Some(v), Some(b)) => v < b, (Some(_), None) => true, _ => false }).count();
        let m = |o: usize| (0..30).map(|k| r[4 * k + o].end_test).sum::<f32>() / 30.0;
        let gen = |o: usize| (0..30).filter(|&k| r[4 * k + o].t_gen.is_some()).count();
        let tr_gt_small = (0..30).filter(|&k| r[4 * k + 2].end_test > r[4 * k + 3].end_test).count();
        std::println!("SHORTFALL: Val cheaper than Big {}/30 (p {:.4}) | generalized Val {}/30 Big {}/30 Train {}/30 Small {}/30 | end test Val {:.3} Big {:.3} Train {:.3} Small {:.3} | Train > Small {}/30",
            cheaper, crate::brain_grok3::sign_p(cheaper, 30), gen(0), gen(1), gen(2), gen(3), m(0), m(1), m(2), m(3), tr_gt_small);
        assert!(cheaper >= MIN_WINS, "Val cheaper on {}/30", cheaper);
    }

    fn short_cfg() -> ShortCfg {
        ShortCfg { base: GrowCfg { n_train1: 290, total: env("SF_TOTAL", 12000), ..dev_cfg() }, h0: env("SF_H0", 8), h_max: env("SF_HMAX", 128),
            window: env("SF_WIN", 10), delta: env("SF_DELTA", 0.01) }
    }

    /// Shortfall dev, dev seeds only: cargo test --release --lib brain_growth::tests::dev_shortfall -- --ignored --nocapture
    #[test]
    #[ignore]
    fn dev_shortfall() {
        let s = short_cfg();
        std::println!("{:?}", s);
        let seeds: Vec<u64> = (0..env("GR_SEEDS", 3u64)).map(|i| 100 + i).collect();
        assert!(seeds.iter().all(|x| !FRESH_SEEDS.contains(x) && !CONT_SEEDS.contains(x)));
        let jobs: Vec<(u64, Short)> = seeds.iter().flat_map(|&x| [(x, Short::Train), (x, Short::Val), (x, Short::Small), (x, Short::Big)]).collect();
        let r: Vec<ShortRun> = par_map(&jobs, |&(x, m)| run_shortfall(x, &s, m));
        for (j, x) in jobs.iter().zip(r.iter()) {
            std::println!("seed {} {:?}: grows {:?} final_h {} end train {:.3} test {:.3} t_gen {:?} Mparam-steps {:.0}", j.0, x.mode, x.grows, x.final_h, x.end_train, x.end_test, x.t_gen, x.param_steps / 1e6);
        }
    }

    #[test]
    fn stop_preregistration_is_pinned() {
        assert_eq!((CONSOLIDATE, STOP_BUDGET, GATE_POS, GATE_NEG_NODECAY, GATE_NEG_LOWDATA), (2_000, 12_000, (290, 1.0), (290, 0.0), (184, 1.0)));
        let mut s = ST_SEEDS.to_vec(); s.sort(); s.dedup(); assert_eq!(s.len(), 30);
        for x in ST_SEEDS.iter() { assert!(!SF_SEEDS.contains(x) && !FRESH_SEEDS.contains(x) && !CONT_SEEDS.contains(x) && !crate::brain_hxo::HX_SEEDS.contains(x)); }
    }

    /// PRE-REGISTERED #18: stop CONSOLIDATE steps after the gate fires. On the grokking world: test at the stop
    /// >= budget test - 0.02 AND stop <= 60 % of the budget, each on >= 27/30; on 60 negative runs: 0 early stops.
    /// Run: GR_THREADS=40 cargo test --release --lib brain_growth::tests::falsifier_stop -- --ignored --nocapture
    #[test]
    #[ignore = "slow: 90 runs x 12 000 steps (run explicitly)"]
    fn falsifier_stop_signal_saves_compute_at_equal_accuracy() {
        let jobs: Vec<(u64, usize, (usize, f32))> = ST_SEEDS.iter().flat_map(|&s| [(s, 0, GATE_POS), (s, 1, GATE_NEG_NODECAY), (s, 2, GATE_NEG_LOWDATA)]).collect();
        let r: Vec<StopRun> = par_map(&jobs, |&(s, _, (n, wd))| stop_run(s, n, wd, STOP_BUDGET, &[CONSOLIDATE]));
        let (mut ok_acc, mut ok_cost, mut early_neg) = (0usize, 0usize, 0usize);
        let mut saved = 0.0f64;
        for (j, x) in jobs.iter().zip(r.iter()) {
            let stop = x.fired_at.map(|f| (f + CONSOLIDATE).min(STOP_BUDGET as u32));
            let at = x.test_after[0].unwrap_or(x.test_budget);
            std::println!("  stop seed {} arm {}: fired {:?} stop {:?} test at stop {:.3} budget {:.3}", j.0, j.1, x.fired_at, stop, at, x.test_budget);
            if j.1 == 0 {
                if at >= x.test_budget - 0.02 { ok_acc += 1; }
                if let Some(s) = stop { if (s as f64) <= 0.6 * STOP_BUDGET as f64 { ok_cost += 1; } saved += 1.0 - s as f64 / STOP_BUDGET as f64; }
            } else if x.fired_at.is_some() { early_neg += 1; }
        }
        std::println!("STOP: accuracy kept (>= budget - 0.02) {}/30 | stop <= 60% budget {}/30 | mean compute saved {:.1}% | negative-arm early stops {}/60", ok_acc, ok_cost, 100.0 * saved / 30.0, early_neg);
        assert!(ok_acc >= 27 && ok_cost >= 27 && early_neg == 0, "acc {}/30 cost {}/30 early_neg {}", ok_acc, ok_cost, early_neg);
    }

    /// #18 dev, dev seeds only: cargo test --release --lib brain_growth::tests::dev_stop -- --ignored --nocapture
    #[test]
    #[ignore]
    fn dev_stop() {
        let extra = [0u32, 500, 1000, 2000, 3000];
        let seeds: Vec<u64> = (0..env("GR_SEEDS", 6u64)).map(|i| 100 + i).collect();
        let jobs: Vec<(u64, usize, f32)> = seeds.iter().flat_map(|&s| [(s, 290usize, 1.0f32), (s, 184, 1.0), (s, 290, 0.0)]).collect();
        let r: Vec<StopRun> = par_map(&jobs, |&(s, n, wd)| stop_run(s, n, wd, 12_000, &extra));
        for (j, x) in jobs.iter().zip(r.iter()) { std::println!("seed {} n{} wd{}: fired {:?} test after +{:?} = {:?} | budget {:.3}", j.0, j.1, j.2, x.fired_at, extra, x.test_after.iter().map(|t| t.map(|v| (v * 1000.0).round() / 1000.0)).collect::<Vec<_>>(), x.test_budget); }
    }

    /// Continual dev, dev seeds only: GR_N2=132 cargo test --release --lib brain_growth::tests::dev_continual -- --ignored --nocapture
    #[test]
    #[ignore]
    fn dev_continual() {
        let c = dev_cfg();
        let seeds: Vec<u64> = (0..env("GR_SEEDS", 3u64)).map(|i| 100 + i).collect();
        assert!(seeds.iter().all(|s| !FRESH_SEEDS.contains(s)));
        let jobs: Vec<(u64, ContArm)> = seeds.iter().flat_map(|&s| [(s, ContArm::Gated), (s, ContArm::Ungated), (s, ContArm::NoGrowth)]).collect();
        let sw: Vec<u32> = par_map(&seeds, |&s| switch_step(s, &c));
        let r: Vec<ContRun> = par_map(&jobs, |&(s, a)| { let k = seeds.iter().position(|&x| x == s).unwrap(); run_continual(s, &c, a, sw[k]) });
        for (j, x) in jobs.iter().zip(r.iter()) { std::println!("seed {} {:?}: switch {} grow {:?} test1@switch {:.3} retain {:.3} acquire {:.3}", j.0, x.arm, x.switch, x.grow_step, x.test1_at_switch, x.retain, x.acquire); }
    }

    /// Dev runs, dev seeds only: cargo test --release --lib brain_growth::tests::dev -- --ignored --nocapture
    #[test]
    #[ignore]
    fn dev() {
        let c = dev_cfg();
        std::println!("{:?}", c);
        let seeds: Vec<u64> = (0..env("GR_SEEDS", 3u64)).map(|i| 100 + i).collect();
        assert!(seeds.iter().all(|s| !FRESH_SEEDS.contains(s)), "fresh seeds are run only by the falsifiers");
        let t: Vec<GrowTrial> = std::thread::scope(|s| seeds.iter().map(|&sd| s.spawn(move || trial(sd, &c))).collect::<Vec<_>>().into_iter().map(|h| h.join().unwrap()).collect());
        let every = env("GR_PRINT", 1000u32);
        for x in t.iter() {
            for r in [&x.g, &x.u, &x.s] {
                std::print!("seed {} {:?}: grow {:?} event {} pruned {} params {:?} t_task2 {:?} t_both {:?} end t1 {:.3} t2 {:.3} |", x.seed, r.arm, r.grow_step, r.event, r.pruned, r.params, r.t_task2(), r.t_both(), r.end(10, |p| p.test1), r.end(10, |p| p.test2));
                for p in r.curve.iter().filter(|p| p.step % every == 0) { std::print!(" {}:{:.2}/{:.2}/{:.2}", p.step, p.val, p.test1, p.test2); }
                std::println!();
            }
        }
    }
}
