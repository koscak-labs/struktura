//! Hebbian growth: propose new senses from what fires together where the brain errs.
//!
//! [`crate::brain_grow::Grower`] scores its whole grammar (every product,
//! threshold and comparison of the base features) on held-out half-splits at
//! every growth check. Here the expensive held-out test only runs on a short,
//! ranked list of candidates proposed by Hebbian traces ("cells that fire
//! together wire together"):
//!
//! - At each check the brain's error (reward minus its own estimate) is computed
//!   once for every stored episode. The **high-error** half of memory is where a
//!   missing sense would pay.
//! - For every pair of inputs it measures how strongly their joint activity
//!   (centred product, or `x_i > x_j`) co-varies with the signed error on those
//!   episodes, per action; for every input and threshold, how strongly crossing
//!   the threshold does. Scores are normalised by the candidate's own spread, so
//!   they are correlation-like and comparable across kinds.
//! - Those statistics feed **decaying traces** (Hebbian memory across checks), so
//!   a co-activation must recur to rise to the top.
//! - Only the top `shortlist` candidates get the held-out half-split test; one is
//!   adopted only if its held-out error reduction is positive in every split and
//!   at least `min_gain` (same rule as `Grower`).
//!
//! **Dendritic depth.** After `depth2_after` consecutive checks without an
//! adoption (growth has stabilised on what base features can express), on
//! request ([`Hebb::unlock_depth2`]) or at once when `depth2_after == 0`,
//! candidates may also be depth-2 senses:
//! - a grown sense combined with a base feature (`Prod`, `Gt`) or thresholded
//!   (`Step(grown, t)`);
//! - a **dendritic pair**: a threshold on a product of two base features,
//!   `(drift × temp) > 0.25`, traced directly on the high-error memories and
//!   adopted as two senses (the product into one free slot, the threshold reading
//!   it into the next). The product alone may be worth too little to be adopted
//!   first, so depth 2 would never be reached by stacking depth-1 adoptions.
//!
//! Situation layout: `B` base features then `G` growth slots, so the brain's `D`
//! must be `B + G`; grown slot `g` may only read base features and slots `< g`.
//! At most [`MAXF`] inputs, [`MAXB`] non-constant base features for dendritic
//! pairs and [`MAXA`] actions (fixed-size traces).
//! no_std, no heap, bounded work per check, deterministic for a seed.

use crate::brain::Brain;
use crate::brain_grow::STEPS;
use crate::sqrt;

/// Maximum inputs (base + grown) the traces cover.
pub const MAXF: usize = 16;
/// Maximum non-constant base features the dendritic-pair traces cover.
pub const MAXB: usize = 8;
/// Maximum actions the traces cover.
pub const MAXA: usize = 8;
/// Upper bound on the shortlist size.
pub const SHORT_MAX: usize = 16;
const NT: usize = 3;

#[inline]
fn absf(x: f32) -> f32 { if x < 0.0 { -x } else { x } }

/// A grown sense. Indices address the situation: `< B` base features, `>= B` grown slots.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Sense {
    Off,
    Prod(u8, u8),
    Step(u8, f32),
    Gt(u8, u8),
}

impl Sense {
    pub fn eval(&self, x: &[f32]) -> f32 {
        match *self {
            Sense::Off => 0.0,
            Sense::Prod(i, j) => x[i as usize] * x[j as usize],
            Sense::Step(i, t) => if x[i as usize] > t { 1.0 } else { 0.0 },
            Sense::Gt(i, j) => if x[i as usize] > x[j as usize] { 1.0 } else { 0.0 },
        }
    }

    /// Whether the sense reads situation index `k`.
    pub fn reads(&self, k: usize) -> bool {
        match *self {
            Sense::Off => false,
            Sense::Prod(i, j) | Sense::Gt(i, j) => i as usize == k || j as usize == k,
            Sense::Step(i, _) => i as usize == k,
        }
    }

    /// Whether the sense reads a grown slot (a depth-2, "dendritic" sense) given `b` base features.
    pub fn is_deep(&self, b: usize) -> bool {
        match *self {
            Sense::Off => false,
            Sense::Prod(i, j) | Sense::Gt(i, j) => i as usize >= b || j as usize >= b,
            Sense::Step(i, _) => i as usize >= b,
        }
    }
}

/// What a Hebbian growth check found.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HebbGrowth {
    pub adopted: Option<Sense>,
    pub slot: usize,
    /// A dendritic-pair adoption grows two senses: this inner product into `slot`, and
    /// `adopted` (a threshold reading it) into `slot2`.
    pub scaffold: Option<Sense>,
    pub slot2: usize,
    /// Mean held-out error reduction of the best shortlisted candidate.
    pub gain: f32,
    /// Candidates that got the held-out test this check.
    pub evaluated: u16,
    /// Whether depth-2 candidates were allowed this check.
    pub depth2: bool,
}

/// A shortlisted candidate: `inner` (Off unless a dendritic pair) then `outer`.
#[derive(Clone, Copy)]
struct Cand { inner: Sense, outer: Sense, score: f32 }

pub struct Hebb<const B: usize, const G: usize> {
    senses: [Sense; G],
    grown: usize,
    limit: usize,
    /// Base features that are constants (bias terms): never used in candidates.
    pub constant: [bool; B],
    /// Decisions between growth checks.
    pub every: u32,
    /// Random half-splits per shortlisted candidate.
    pub splits: u8,
    /// Minimum mean held-out error reduction, as a fraction of held-out error.
    pub min_gain: f32,
    /// Candidates given the held-out test per check (<= [`SHORT_MAX`]).
    pub shortlist: usize,
    /// Trace decay per check (0 = only the current check counts).
    pub decay: f32,
    /// 1 = base-feature senses only; 2 = also senses built on grown senses and dendritic pairs.
    pub max_depth: u8,
    /// Consecutive checks without adoption before depth 2 unlocks (0 = immediately).
    pub depth2_after: u8,
    quiet: u8,
    depth2: bool,
    prod: [[f32; MAXF * MAXF]; MAXA],
    gt: [[f32; MAXF * MAXF]; MAXA],
    step: [[[f32; NT]; MAXF]; MAXA],
    dstep: [[[f32; NT]; MAXB * MAXB]; MAXA],
    since: u32,
    rng: u64,
    /// Held-out candidate evaluations so far (the expensive part).
    pub evaluations: u32,
    /// Growth checks run so far.
    pub checks: u32,
}

impl<const B: usize, const G: usize> Hebb<B, G> {
    pub const fn new(seed: u64) -> Self {
        Hebb {
            senses: [Sense::Off; G], grown: 0, limit: G, constant: [false; B], every: 250, splits: 4, min_gain: 0.05,
            shortlist: 10, decay: 0.5, max_depth: 2, depth2_after: 2, quiet: 0, depth2: false,
            prod: [[0.0; MAXF * MAXF]; MAXA], gt: [[0.0; MAXF * MAXF]; MAXA], step: [[[0.0; NT]; MAXF]; MAXA],
            dstep: [[[0.0; NT]; MAXB * MAXB]; MAXA],
            since: 0, rng: seed | 1, evaluations: 0, checks: 0,
        }
    }

    pub fn grown(&self) -> &[Sense] { &self.senses[..self.grown] }
    pub fn active(&self) -> usize { self.senses[..self.grown].iter().filter(|s| **s != Sense::Off).count() }
    pub fn set_limit(&mut self, limit: usize) { self.limit = limit.min(G); }
    pub fn depth2_unlocked(&self) -> bool { self.max_depth >= 2 && (self.depth2 || self.depth2_after == 0) }

    /// Unlock depth-2 candidates now (on request).
    pub fn unlock_depth2(&mut self) { if self.max_depth >= 2 { self.depth2 = true; } }

    /// Base features followed by the grown senses, evaluated in slot order. `D` must be `B + G`.
    pub fn situation<const D: usize>(&self, base: &[f32; B]) -> [f32; D] {
        let mut x = [0.0f32; D];
        for i in 0..B.min(D) { x[i] = base[i]; }
        for g in 0..self.grown { if B + g < D { x[B + g] = self.senses[g].eval(&x); } }
        x
    }

    /// Prune grown sense `slot` (it reads 0 from now on; memory is re-keyed).
    /// Senses that read it are pruned too, so nothing depends on a dead slot.
    pub fn prune<const N: usize, const D: usize, const A: usize>(&mut self, slot: usize, brain: &mut Brain<N, D, A>) -> bool {
        if slot >= self.grown || self.senses[slot] == Sense::Off || B + G != D { return false; }
        self.senses[slot] = Sense::Off;
        for g in (slot + 1)..self.grown {
            let dead = (B..B + g).any(|k| self.senses[k - B] == Sense::Off && self.senses[g].reads(k));
            if dead { self.senses[g] = Sense::Off; }
        }
        let senses = self.senses;
        let grown = self.grown;
        brain.rekey(|k| { let mut y = *k; for g in 0..grown { y[B + g] = senses[g].eval(&y); } y });
        true
    }

    fn coin(&mut self) -> bool {
        self.rng ^= self.rng << 13; self.rng ^= self.rng >> 7; self.rng ^= self.rng << 17;
        self.rng & 1 == 1
    }

    fn is_free(&self, s: usize) -> bool { s >= self.grown || self.senses[s] == Sense::Off }

    /// Call after every learned outcome. Every `every` calls it runs a Hebbian growth check.
    pub fn after_learn<const N: usize, const D: usize, const A: usize>(&mut self, brain: &mut Brain<N, D, A>) -> Option<HebbGrowth> {
        self.since += 1;
        if self.since < self.every || B + G != D || B + G > MAXF || A > MAXA { return None; }
        let free = (0..self.limit).find(|&s| self.is_free(s))?;
        let free2 = ((free + 1)..self.limit).find(|&s| self.is_free(s));
        self.since = 0;
        let g = self.check(brain, free, free2);
        if let Some(s) = g.adopted {
            let mut set = |slot: usize, s: Sense, me: &mut Self| {
                me.senses[slot] = s;
                if slot >= me.grown { me.grown = slot + 1; }
                brain.rekey(|k| { let mut y = *k; y[B + slot] = s.eval(&y); y });
            };
            if let Some(inner) = g.scaffold { set(g.slot, inner, self); set(g.slot2, s, self); } else { set(g.slot, s, self); }
            self.quiet = 0;
        } else {
            self.quiet = self.quiet.saturating_add(1);
            if self.max_depth >= 2 && self.quiet >= self.depth2_after { self.depth2 = true; }
        }
        Some(g)
    }

    /// One Hebbian check for filling slot `free` (and `free2` for a dendritic pair):
    /// update traces, shortlist, held-out test the shortlist.
    pub fn check<const N: usize, const D: usize, const A: usize>(&mut self, brain: &Brain<N, D, A>, free: usize, free2: Option<usize>) -> HebbGrowth {
        self.checks += 1;
        let depth2 = self.depth2_unlocked();
        let none = HebbGrowth { adopted: None, slot: free, scaffold: None, slot2: free, gain: 0.0, evaluated: 0, depth2 };
        if B + G > MAXF || A > MAXA || free >= G { return none; }
        // Inputs candidates may read: non-constant base features, plus (depth 2) active grown slots < free.
        let mut inputs = [0u8; MAXF];
        let mut ni = 0usize;
        for i in 0..B { if !self.constant[i] { inputs[ni] = i as u8; ni += 1; } }
        let nb = ni;
        if depth2 { for g in 0..free.min(self.grown) { if self.senses[g] != Sense::Off { inputs[ni] = (B + g) as u8; ni += 1; } } }
        // Dendritic pairs need two free slots, slot order free < free2.
        let pairs = depth2 && free2.is_some() && nb <= MAXB;
        let nd = if pairs { nb } else { 0 };

        // 1. The brain's error on every stored episode, computed once.
        let mut res = [0.0f32; N];
        let mut used = [false; N];
        let mut n = 0usize;
        let mut mean = [0.0f32; MAXF];
        for s in 0..N {
            if let Some(e) = brain.episode(s) {
                res[s] = e.reward - brain.estimates(&e.key)[e.action as usize].expected;
                used[s] = true; n += 1;
                for k in 0..ni { mean[k] += e.key[inputs[k] as usize]; }
            }
        }
        if n < 8 { return none; }
        for k in 0..ni { mean[k] /= n as f32; }
        // Median |error| (insertion sort of a fixed-size copy).
        let mut abs = [0.0f32; N];
        let mut m = 0usize;
        for s in 0..N { if used[s] { let v = absf(res[s]); let mut p = m; while p > 0 && abs[p - 1] > v { abs[p] = abs[p - 1]; p -= 1; } abs[p] = v; m += 1; } }
        let cut = abs[m / 2];

        // 2. Co-activation statistics on the high-error half, per action; spreads over all memory.
        let mut sp = [[0.0f32; MAXF * MAXF]; MAXA];
        let mut sg = [[0.0f32; MAXF * MAXF]; MAXA];
        let mut ss = [[[0.0f32; NT]; MAXF]; MAXA];
        let mut sd = [[[0.0f32; NT]; MAXB * MAXB]; MAXA];
        let mut nh = [0u32; MAXA];
        let mut rsum = [0.0f32; MAXA];
        let mut vp = [0.0f32; MAXF * MAXF];
        let mut fg = [0.0f32; MAXF * MAXF];
        let mut fs = [[0.0f32; NT]; MAXF];
        let mut fd = [[0.0f32; NT]; MAXB * MAXB];
        for s in 0..N {
            if !used[s] { continue; }
            let Some(e) = brain.episode(s) else { continue };
            let a = e.action as usize;
            let high = absf(res[s]) >= cut;
            if high { nh[a] += 1; rsum[a] += res[s]; }
            let r = if high { res[s] } else { 0.0 };
            for p in 0..ni {
                let xp = e.key[inputs[p] as usize];
                let zp = xp - mean[p];
                for q in 0..ni {
                    let idx = p * MAXF + q;
                    let xq = e.key[inputs[q] as usize];
                    if q >= p {
                        let zq = xq - mean[q];
                        vp[idx] += zp * zp * zq * zq;
                        sp[a][idx] += r * zp * zq;
                    }
                    if q != p && xp > xq { fg[idx] += 1.0; sg[a][idx] += r; }
                    if p < nd && q < nd && q >= p {
                        for t in 0..NT { if xp * xq > STEPS[t] { fd[p * MAXB + q][t] += 1.0; sd[a][p * MAXB + q][t] += r; } }
                    }
                }
                for t in 0..NT { if xp > STEPS[t] { fs[p][t] += 1.0; ss[a][p][t] += r; } }
            }
        }

        // 3. Decaying traces: T <- decay * T + (1 - decay) * normalised statistic.
        //    Indicator statistics are centred (minus the action's error sum times the base rate)
        //    and divided by the Bernoulli spread, products by their own spread.
        let nf = n as f32;
        let ind = |sum: f32, count: f32, rs: f32, k: f32| -> f32 {
            let f = count / nf;
            let sdv = sqrt((f * (1.0 - f)) as f64) as f32;
            if sdv > 1e-6 { (sum - rs * f) * k / sdv } else { 0.0 }
        };
        let (d, w) = (self.decay, 1.0 - self.decay);
        for a in 0..A {
            let k = if nh[a] > 0 { 1.0 / nh[a] as f32 } else { 0.0 };
            for p in 0..ni { for q in 0..ni {
                let idx = p * MAXF + q;
                if q >= p {
                    let sdv = sqrt((vp[idx] / nf) as f64) as f32;
                    let v = if sdv > 1e-6 { sp[a][idx] * k / sdv } else { 0.0 };
                    self.prod[a][idx] = d * self.prod[a][idx] + w * v;
                }
                if q != p { self.gt[a][idx] = d * self.gt[a][idx] + w * ind(sg[a][idx], fg[idx], rsum[a], k); }
                if p < nd && q < nd && q >= p {
                    for t in 0..NT { let i = p * MAXB + q; self.dstep[a][i][t] = d * self.dstep[a][i][t] + w * ind(sd[a][i][t], fd[i][t], rsum[a], k); }
                }
            } }
            for p in 0..ni { for t in 0..NT { self.step[a][p][t] = d * self.step[a][p][t] + w * ind(ss[a][p][t], fs[p][t], rsum[a], k); } }
        }

        // 4. Rank candidates by trace strength (summed over actions) into a fixed shortlist.
        let want = self.shortlist.min(SHORT_MAX).max(1);
        let mut short = [Cand { inner: Sense::Off, outer: Sense::Off, score: 0.0 }; SHORT_MAX];
        let mut ns = 0usize;
        let grown = &self.senses[..self.grown];
        let mut offer = |c: Cand| {
            if c.score <= 0.0 || grown.contains(&c.outer) || (c.inner != Sense::Off && grown.contains(&c.inner)) { return; }
            if ns < want || c.score > short[ns - 1].score {
                let mut p = if ns < want { ns } else { want - 1 };
                while p > 0 && short[p - 1].score < c.score { short[p] = short[p - 1]; p -= 1; }
                short[p] = c;
                if ns < want { ns += 1; }
            }
        };
        let total = |t: &dyn Fn(usize) -> f32| -> f32 { (0..A).map(|a| absf(t(a))).sum() };
        for p in 0..ni { for q in 0..ni {
            // depth-2 senses combine exactly one grown sense with a base feature
            if (p >= nb || q >= nb) && (p >= nb) == (q >= nb) { continue; }
            let idx = p * MAXF + q;
            if q >= p { offer(Cand { inner: Sense::Off, outer: Sense::Prod(inputs[p], inputs[q]), score: total(&|a| self.prod[a][idx]) }); }
            if q != p { offer(Cand { inner: Sense::Off, outer: Sense::Gt(inputs[p], inputs[q]), score: total(&|a| self.gt[a][idx]) }); }
        } }
        for p in 0..ni { for t in 0..NT {
            offer(Cand { inner: Sense::Off, outer: Sense::Step(inputs[p], STEPS[t]), score: total(&|a| self.step[a][p][t]) });
        } }
        for p in 0..nd { for q in p..nd { for t in 0..NT {
            let i = p * MAXB + q;
            offer(Cand { inner: Sense::Prod(inputs[p], inputs[q]), outer: Sense::Step((B + free) as u8, STEPS[t]), score: total(&|a| self.dstep[a][i][t]) });
        } } }

        // 5. Held-out half-split test of the shortlist only.
        let value = |c: &Cand, key: &[f32; D]| -> f32 {
            if c.inner == Sense::Off { return c.outer.eval(key); }
            let mut y = *key;
            y[B + free] = c.inner.eval(&y);
            c.outer.eval(&y)
        };
        let mut best: (Option<Cand>, f32) = (None, 0.0);
        for k in 0..ns {
            let cand = short[k];
            self.evaluations += 1;
            let mut sum = 0.0f32;
            let mut all_positive = true;
            for _ in 0..self.splits {
                let mut train = [false; N];
                for s in 0..N { train[s] = self.coin(); }
                let (mut sfr, mut sff) = ([0.0f32; MAXA], [0.0f32; MAXA]);
                for s in 0..N {
                    if !used[s] || !train[s] { continue; }
                    let Some(e) = brain.episode(s) else { continue };
                    let v = value(&cand, &e.key);
                    sfr[e.action as usize] += v * res[s]; sff[e.action as usize] += v * v;
                }
                let (mut before, mut after) = (0.0f32, 0.0f32);
                for s in 0..N {
                    if !used[s] || train[s] { continue; }
                    let Some(e) = brain.episode(s) else { continue };
                    let a = e.action as usize;
                    let beta = if sff[a] > 1e-6 { sfr[a] / sff[a] } else { 0.0 };
                    before += res[s] * res[s];
                    let r2 = res[s] - beta * value(&cand, &e.key); after += r2 * r2;
                }
                let gain = if before > 1e-9 { (before - after) / before } else { 0.0 };
                if gain <= 0.0 { all_positive = false; }
                sum += gain;
            }
            let mean = sum / self.splits.max(1) as f32;
            if all_positive && mean > best.1 { best = (Some(cand), mean); }
        }
        let mut out = HebbGrowth { gain: best.1, evaluated: ns as u16, ..none };
        if let (Some(c), true) = (best.0, best.1 >= self.min_gain) {
            out.adopted = Some(c.outer);
            if c.inner != Sense::Off {
                // The threshold moves to free2 and reads the product grown at free.
                out.scaffold = Some(c.inner);
                out.slot2 = free2.unwrap_or(free);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain_grow::Grower;

    struct Rng(u64);
    impl Rng { fn f(&mut self) -> f32 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; ((self.0 >> 40) as f32) / (1u64 << 24) as f32 } }

    /// World B from `struktura brain demo` (interactions a linear model cannot represent).
    fn truth_b(x: &[f32; 4]) -> u8 {
        if x[2] > 0.6 && x[0] > 0.6 { 3 } else if x[1] > 0.6 && x[2] < 0.4 { 2 } else if (x[0] > 0.5) != (x[1] > 0.5) { 1 } else { 0 }
    }
    /// A world that needs a depth-2 sense: act on (drift x temp) > 0.25.
    fn truth_d2(x: &[f32; 4]) -> u8 {
        if x[0] * x[2] > 0.25 { 1 } else if x[1] > 0.6 { 2 } else { 0 }
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    enum Mode { Plain, Grower, Hebb1, Hebb2 }

    /// (right-action rate over the last 1000 decisions, held-out candidate evaluations).
    fn run(seed: u64, mode: Mode, truth: fn(&[f32; 4]) -> u8) -> (f32, u32) {
        let mut r = Rng(seed);
        let mut br: Brain<256, 8, 4> = Brain::new(0.3, 0.5);
        let mut gr: Grower<4, 4> = Grower::new(seed ^ 0xBEEF);
        gr.constant[3] = true;
        let mut hb: Hebb<4, 4> = Hebb::new(seed ^ 0xBEEF);
        hb.constant[3] = true;
        hb.max_depth = if mode == Mode::Hebb2 { 2 } else { 1 };
        let steps = 4000usize;
        let (mut ok, mut evals) = (0, 0u32);
        for t in 0..steps {
            let base = [r.f(), r.f(), r.f(), 1.0];
            let x: [f32; 8] = match mode { Mode::Grower => gr.situation(&base), Mode::Hebb1 | Mode::Hebb2 => hb.situation(&base),
                Mode::Plain => { let mut v = [0.0; 8]; v[..4].copy_from_slice(&base); v } };
            let d = br.decide(&x, &[true; 4], 3);
            let a = if d.abstained && t < 400 { (t % 4) as u8 } else { d.action };
            let right = a == truth(&base);
            if t >= steps - 1000 && right { ok += 1; }
            br.learn(&x, a, if right { 1.0 } else if a == 3 { 0.2 } else { 0.0 });
            match mode {
                Mode::Grower => if let Some(g) = gr.after_learn(&mut br) { evals += g.candidates as u32; },
                Mode::Hebb1 | Mode::Hebb2 => { hb.after_learn(&mut br); }
                Mode::Plain => {}
            }
        }
        if matches!(mode, Mode::Hebb1 | Mode::Hebb2) { evals = hb.evaluations; }
        (ok as f32 / 1000.0, evals)
    }

    fn mean(seeds: &[u64], mode: Mode, truth: fn(&[f32; 4]) -> u8) -> (f32, f32) {
        let (mut a, mut e) = (0.0f32, 0.0f32);
        for &s in seeds { let (x, n) = run(s, mode, truth); a += x; e += n as f32; }
        (a / seeds.len() as f32, e / seeds.len() as f32)
    }

    const SEEDS: [[u64; 8]; 2] = [[11, 23, 37, 41, 59, 61, 73, 89], [101, 103, 107, 109, 113, 127, 131, 137]];

    /// Pre-registered falsifier (a), 8 + 8 fresh seeds: on the rover interaction world, Hebbian
    /// proposals (depth 1, like the Grower) reach the same or better right-action rate (last 1000)
    /// as the exhaustive Grower — "same" = within 1 point — while giving the held-out test
    /// STRICTLY fewer candidates.
    #[test]
    fn falsifier_a_hebb_matches_exhaustive_growth_with_fewer_evaluations() {
        for seeds in SEEDS {
            let (plain, _) = mean(&seeds, Mode::Plain, truth_b);
            let (grow, ge) = mean(&seeds, Mode::Grower, truth_b);
            let (hebb, he) = mean(&seeds, Mode::Hebb1, truth_b);
            std::println!("hebb (a) interaction world: plain {:.3} | exhaustive grower {:.3} ({:.0} held-out evaluations/run) | hebb depth-1 {:.3} ({:.0} evaluations/run, {:.0}% saved)",
                plain, grow, ge, hebb, he, 100.0 * (1.0 - he / ge.max(1.0)));
            assert!(hebb >= grow - 0.01, "hebb {} vs grower {}", hebb, grow);
            assert!(he < ge, "hebb evaluations {} not fewer than grower {}", he, ge);
        }
    }

    /// Pre-registered falsifier (b), 8 + 8 fresh seeds: on a world that needs a depth-2 sense
    /// ((drift x temp) > 0.25), Hebb with depth 2 beats depth-1 growth (the exhaustive Grower)
    /// by >= 3 points (right-action rate, last 1000). Hebb depth-1 and no growth are reported.
    #[test]
    fn falsifier_b_depth2_beats_depth1_growth() {
        for seeds in SEEDS {
            let (plain, _) = mean(&seeds, Mode::Plain, truth_d2);
            let (grow, _) = mean(&seeds, Mode::Grower, truth_d2);
            let (h1, _) = mean(&seeds, Mode::Hebb1, truth_d2);
            let (h2, e2) = mean(&seeds, Mode::Hebb2, truth_d2);
            std::println!("hebb (b) depth-2 world: plain {:.3} | grower depth-1 {:.3} | hebb depth-1 {:.3} | hebb depth-2 {:.3} ({:.0} evaluations/run)",
                plain, grow, h1, h2, e2);
            assert!(h2 >= grow + 0.03, "depth-2 hebb {} vs depth-1 grower {}", h2, grow);
        }
    }

    #[test]
    fn depth2_sense_reads_a_grown_slot_and_prune_cascades() {
        let mut h: Hebb<4, 4> = Hebb::new(1);
        h.senses[0] = Sense::Prod(0, 2);
        h.senses[1] = Sense::Step(4, 0.25);
        h.grown = 2;
        let x: [f32; 8] = h.situation(&[0.6, 0.1, 0.5, 1.0]);
        assert!((x[4] - 0.3).abs() < 1e-6 && x[5] == 1.0, "{:?}", x);
        assert!(h.senses[1].is_deep(4) && !h.senses[0].is_deep(4));
        let mut br: Brain<16, 8, 2> = Brain::new(0.0, 0.0);
        br.learn(&x, 0, 1.0);
        assert!(h.prune(0, &mut br));
        assert_eq!(h.senses[1], Sense::Off, "a sense reading a pruned slot is pruned too");
        let k = br.episode(0).unwrap().key;
        assert_eq!((k[4], k[5]), (0.0, 0.0), "memory re-keyed");
    }

    #[test]
    fn depth1_never_grows_deep_senses() {
        let mut r = Rng(5);
        let mut br: Brain<256, 8, 4> = Brain::new(0.3, 0.5);
        let mut hb: Hebb<4, 4> = Hebb::new(5);
        hb.constant[3] = true;
        hb.max_depth = 1;
        for t in 0..3000usize {
            let base = [r.f(), r.f(), r.f(), 1.0];
            let x: [f32; 8] = hb.situation(&base);
            let d = br.decide(&x, &[true; 4], 3);
            let a = if d.abstained && t < 400 { (t % 4) as u8 } else { d.action };
            br.learn(&x, a, if a == truth_d2(&base) { 1.0 } else { 0.0 });
            if let Some(g) = hb.after_learn(&mut br) { assert!(g.scaffold.is_none() && !g.depth2 && g.evaluated as usize <= hb.shortlist); }
        }
        assert!(hb.grown().iter().all(|s| !s.is_deep(4)), "{:?}", hb.grown());
    }

    #[test]
    fn deterministic_and_bounded() {
        assert_eq!(run(7, Mode::Hebb2, truth_d2), run(7, Mode::Hebb2, truth_d2));
        assert!(core::mem::size_of::<Hebb<4, 4>>() < 32 * 1024, "{}", core::mem::size_of::<Hebb<4, 4>>());
    }
}
