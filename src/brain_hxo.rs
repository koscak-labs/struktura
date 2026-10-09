//! HxO doctrine seeds, falsified on CPU before anything is built on them.
//!
//! `Oura/HXO_ARCHITECTURE.md` specifies, among others: strands fused by **Born-rule routing**
//! ("quantum-inspired probabilistic routing across the substrate. Not softmax-attention"). This
//! module builds the smallest network where that choice is the only difference and tests it in the
//! grokking world of [`crate::brain_grokbed`] (the Grokfast seed and the Friston loss are tested
//! there, on its [`crate::brain_grokbed::Net`]).
//!
//! **Strand network**: `K` strands of `hb` hidden neurons. Shared embeddings `pre = Ea[a] + Eb[b]`
//! (`p x K hb`), quadratic neurons, each strand reads its own neurons into `p` logits; a gate gives
//! each strand an amplitude `amp_s = Ga[a]_s + Gb[b]_s`, and the strands' logits are fused with
//! weights `w = route(amp)`:
//! - **Born**: `w_s = amp_s^2 / sum amp^2` (the Born rule on real amplitudes);
//! - **Softmax**: `w_s = exp(amp_s) / sum exp(amp)`.
//!
//! Softmax cross-entropy on the fused logits, full batch, AdamW (decoupled weight decay).
//! std-only, CPU, deterministic for a seed.
//!
//! ## Pre-registration: Born-rule routing (committed before any of its fresh seeds was run)
//!
//! - **World** [`HX_PREREG`]: the grokbed world (`p = 23`, 265 training pairs, quadratic, AdamW
//!   `lr 1e-3`, decay 1, 12 000 steps) with 8 strands of 16 neurons; gates `N(0, 0.5^2)`.
//! - **Claim**: Born routing ends (mean test accuracy of the last 10 measurements) above softmax
//!   routing on >= 20 of 30 fresh seeds [`HX_SEEDS`] (sign test p <= 0.05; tie = loss).
//! - **Dev** (100..105): neither routing generalizes within 12 000 steps (the plain 128-neuron net
//!   does at ~4 000). End test Born 0.009..0.027, softmax 0.065..0.148: Born lower on 6/6. Born
//!   collapses the routing to ~1-1.5 strands per pair (routing entropy 0.27..0.40 nats; softmax
//!   1.54..1.56, ~4.7 strands), so each pair sees ~16 neurons: the regime where 16 neurons memorize.
//! - **Analysis prediction: FAIL.**
//!
//! ## RESULT
//!
//! (added after the one run)


use std::vec;
use std::vec::Vec;

use crate::brain_grokbed::{split, Pair, Rng};

/// How strands are fused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route { Born, Softmax }

const BETA1: f32 = 0.9;
const BETA2: f32 = 0.98;
const EPS: f32 = 1e-8;
const BORN_EPS: f32 = 1e-6;

/// The strand network and its optimizer state.
#[derive(Clone, Debug)]
pub struct StrandNet {
    pub p: usize,
    pub k: usize,
    pub hb: usize,
    pub route: Route,
    /// `[Ea (p x h) | Eb (p x h) | W (p x h) | Ga (p x k) | Gb (p x k)]`, `h = k hb`.
    pub w: Vec<f32>,
    m: Vec<f32>,
    v: Vec<f32>,
    t: u32,
}

impl StrandNet {
    pub fn new(p: usize, k: usize, hb: usize, route: Route, init: f32, gate_init: f32, rng: &mut Rng) -> Self {
        let h = k * hb;
        let n = 3 * p * h + 2 * p * k;
        let mut w = vec![0.0f32; n];
        for x in w[..2 * p * h].iter_mut() { *x = init * rng.normal(); }
        let s = 1.0 / (h as f32).sqrt();
        for x in w[2 * p * h..3 * p * h].iter_mut() { *x = s * rng.normal(); }
        for x in w[3 * p * h..].iter_mut() { *x = gate_init * rng.normal(); }
        StrandNet { p, k, hb, route, w, m: vec![0.0; n], v: vec![0.0; n], t: 0 }
    }

    fn h(&self) -> usize { self.k * self.hb }
    fn off(&self) -> [usize; 5] { let (p, h, k) = (self.p, self.h(), self.k); [0, p * h, 2 * p * h, 3 * p * h, 3 * p * h + p * k] }

    fn weights(&self, amp: &[f32], w: &mut [f32]) {
        match self.route {
            Route::Born => {
                let s: f32 = amp.iter().map(|a| a * a).sum::<f32>() + BORN_EPS;
                for (o, a) in w.iter_mut().zip(amp) { *o = a * a / s; }
            }
            Route::Softmax => {
                let mx = amp.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut s = 0.0;
                for (o, a) in w.iter_mut().zip(amp) { *o = (a - mx).exp(); s += *o; }
                for o in w.iter_mut() { *o /= s; }
            }
        }
    }

    /// Fused logits of one pair; fills the scratch buffers used by [`Self::grad`].
    fn forward(&self, a: usize, b: usize, s: &mut Scratch) {
        let (p, h, k, hb) = (self.p, self.h(), self.k, self.hb);
        let o = self.off();
        for j in 0..h { s.pre[j] = self.w[o[0] + a * h + j] + self.w[o[1] + b * h + j]; s.z[j] = s.pre[j] * s.pre[j]; }
        for st in 0..k {
            s.amp[st] = self.w[o[3] + a * k + st] + self.w[o[4] + b * k + st];
            for c in 0..p {
                let row = &self.w[o[2] + c * h + st * hb..][..hb];
                s.ls[st * p + c] = row.iter().zip(&s.z[st * hb..(st + 1) * hb]).map(|(x, y)| x * y).sum();
            }
        }
        let amp = s.amp.clone();
        self.weights(&amp, &mut s.wt);
        for c in 0..p { s.out[c] = (0..k).map(|st| s.wt[st] * s.ls[st * p + c]).sum(); }
    }

    /// `(accuracy, mean cross-entropy)`.
    pub fn eval(&self, data: &[Pair]) -> (f32, f32) {
        let mut s = Scratch::new(self);
        let (mut right, mut loss) = (0usize, 0.0f64);
        for &(a, b, _, y) in data {
            self.forward(a as usize, b as usize, &mut s);
            if argmax(&s.out) == y as usize { right += 1; }
            loss += xent(&mut s.out, y as usize) as f64;
        }
        (right as f32 / data.len().max(1) as f32, (loss / data.len().max(1) as f64) as f32)
    }

    /// Mean cross-entropy and its gradient.
    pub fn grad(&self, data: &[Pair]) -> (f32, Vec<f32>) {
        let (p, h, k, hb) = (self.p, self.h(), self.k, self.hb);
        let o = self.off();
        let mut g = vec![0.0f32; self.w.len()];
        let mut s = Scratch::new(self);
        let (mut dz, mut gw) = (vec![0.0f32; h], vec![0.0f32; k]);
        let mut loss = 0.0f64;
        for &(a, b, _, y) in data {
            let (a, b, y) = (a as usize, b as usize, y as usize);
            self.forward(a, b, &mut s);
            loss += xent(&mut s.out, y) as f64;
            s.out[y] -= 1.0; // delta = softmax - onehot
            for st in 0..k {
                gw[st] = (0..p).map(|c| s.out[c] * s.ls[st * p + c]).sum();
                for j in st * hb..(st + 1) * hb { dz[j] = 0.0; }
                for c in 0..p {
                    let dl = s.out[c] * s.wt[st];
                    let base = o[2] + c * h + st * hb;
                    for jj in 0..hb {
                        let j = st * hb + jj;
                        g[base + jj] += dl * s.z[j];
                        dz[j] += dl * self.w[base + jj];
                    }
                }
            }
            for j in 0..h {
                let d = dz[j] * 2.0 * s.pre[j];
                g[o[0] + a * h + j] += d;
                g[o[1] + b * h + j] += d;
            }
            let gbar: f32 = (0..k).map(|st| s.wt[st] * gw[st]).sum();
            let sum2: f32 = s.amp.iter().map(|x| x * x).sum::<f32>() + BORN_EPS;
            for m in 0..k {
                let da = match self.route {
                    Route::Born => 2.0 * s.amp[m] / sum2 * (gw[m] - gbar),
                    Route::Softmax => s.wt[m] * (gw[m] - gbar),
                };
                g[o[3] + a * k + m] += da;
                g[o[4] + b * k + m] += da;
            }
        }
        let n = data.len().max(1) as f32;
        for x in g.iter_mut() { *x /= n; }
        ((loss / n as f64) as f32, g)
    }

    /// One full-batch AdamW step.
    pub fn step(&mut self, data: &[Pair], lr: f32, wd: f32) {
        let (_, g) = self.grad(data);
        self.t += 1;
        let (c1, c2) = (1.0 - BETA1.powi(self.t as i32), 1.0 - BETA2.powi(self.t as i32));
        for i in 0..self.w.len() {
            self.m[i] = BETA1 * self.m[i] + (1.0 - BETA1) * g[i];
            self.v[i] = BETA2 * self.v[i] + (1.0 - BETA2) * g[i] * g[i];
            self.w[i] -= lr * ((self.m[i] / c1) / ((self.v[i] / c2).sqrt() + EPS) + wd * self.w[i]);
        }
    }

    /// Mean over a data set of the routing entropy, in nats (how many strands a pair uses).
    pub fn route_entropy(&self, data: &[Pair]) -> f32 {
        let mut s = Scratch::new(self);
        let mut e = 0.0f64;
        for &(a, b, _, _) in data {
            self.forward(a as usize, b as usize, &mut s);
            e += -s.wt.iter().map(|&q| if q > 0.0 { (q * q.ln()) as f64 } else { 0.0 }).sum::<f64>();
        }
        (e / data.len().max(1) as f64) as f32
    }
}

struct Scratch { pre: Vec<f32>, z: Vec<f32>, amp: Vec<f32>, wt: Vec<f32>, ls: Vec<f32>, out: Vec<f32> }
impl Scratch {
    fn new(n: &StrandNet) -> Self {
        let h = n.h();
        Scratch { pre: vec![0.0; h], z: vec![0.0; h], amp: vec![0.0; n.k], wt: vec![0.0; n.k], ls: vec![0.0; n.k * n.p], out: vec![0.0; n.p] }
    }
}

fn argmax(x: &[f32]) -> usize { let mut b = 0; for i in 1..x.len() { if x[i] > x[b] { b = i; } } b }

fn xent(logits: &mut [f32], y: usize) -> f32 {
    let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0.0f32;
    for l in logits.iter_mut() { *l = (*l - mx).exp(); s += *l; }
    for l in logits.iter_mut() { *l /= s; }
    -(logits[y].max(1e-30)).ln()
}

/// One strand run's settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StrandCfg {
    pub p: usize,
    pub n_train: usize,
    pub k: usize,
    pub hb: usize,
    pub init: f32,
    pub gate_init: f32,
    pub lr: f32,
    pub wd: f32,
    pub steps: usize,
    pub eval_every: usize,
}

/// One run's summary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StrandRun {
    pub route: Route,
    pub t_fit: Option<u32>,
    pub t_gen: Option<u32>,
    pub end_test: f32,
    pub route_entropy: f32,
}

/// Train one route; `t_fit` / `t_gen` = first step with train >= 0.99 / test >= 0.95; end = mean of the last 10.
pub fn run_strands(seed: u64, c: &StrandCfg, route: Route) -> StrandRun {
    let (train, test) = split(seed, c.p, c.n_train);
    let mut rng = Rng::new(seed);
    let mut net = StrandNet::new(c.p, c.k, c.hb, route, c.init, c.gate_init, &mut rng);
    let (mut t_fit, mut t_gen, mut tail) = (None, None, Vec::new());
    for s in 1..=c.steps {
        net.step(&train, c.lr, c.wd);
        if s % c.eval_every != 0 { continue; }
        let (tr, te) = (net.eval(&train).0, net.eval(&test).0);
        if t_fit.is_none() && tr >= 0.99 { t_fit = Some(s as u32); }
        if t_gen.is_none() && te >= 0.95 { t_gen = Some(s as u32); }
        tail.push(te); if tail.len() > 10 { tail.remove(0); }
    }
    StrandRun { route, t_fit, t_gen, end_test: tail.iter().sum::<f32>() / tail.len().max(1) as f32, route_entropy: net.route_entropy(&test) }
}

/// Pre-registered strand world (the grokbed world with 8 strands of 16 neurons = 128).
pub const HX_PREREG: StrandCfg = StrandCfg { p: 23, n_train: 265, k: 8, hb: 16, init: 0.1, gate_init: 0.5, lr: 1e-3, wd: 1.0, steps: 12_000, eval_every: 100 };
/// Fresh seeds for the routing falsifier.
pub const HX_SEEDS: [u64; 30] = [62003, 62011, 62017, 62039, 62047, 62053, 62057, 62071, 62081, 62099, 62119, 62129, 62131, 62137, 62141,
    62143, 62171, 62189, 62191, 62201, 62207, 62213, 62219, 62233, 62273, 62297, 62299, 62303, 62311, 62323];

#[cfg(test)]
mod tests {
    use super::*;

    fn env<T: core::str::FromStr>(k: &str, d: T) -> T { std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }

    #[test]
    fn gradient_matches_finite_differences_for_both_routes() {
        for route in [Route::Born, Route::Softmax] {
            let mut rng = Rng::new(9);
            let net = StrandNet::new(5, 3, 4, route, 0.5, 0.8, &mut rng);
            let data: Vec<Pair> = vec![(1, 2, 0, 3), (4, 4, 0, 3), (0, 3, 0, 3), (2, 2, 0, 4), (3, 1, 0, 4)];
            let (_, g) = net.grad(&data);
            for i in 0..net.w.len() {
                let e = 1e-2f32;
                let mut np = net.clone(); np.w[i] += e;
                let mut nm = net.clone(); nm.w[i] -= e;
                let fd = (np.eval(&data).1 as f64 - nm.eval(&data).1 as f64) / (2.0 * e as f64);
                assert!((fd - g[i] as f64).abs() <= 2e-3 + 3e-2 * fd.abs(), "{:?} weight {}: fd {} vs analytic {}", route, i, fd, g[i]);
            }
        }
    }

    #[test]
    fn routing_weights_are_a_distribution() {
        let mut rng = Rng::new(1);
        for route in [Route::Born, Route::Softmax] {
            let net = StrandNet::new(5, 4, 2, route, 0.5, 1.0, &mut rng);
            let mut w = vec![0.0; 4];
            net.weights(&[0.3, -1.2, 0.0, 2.0], &mut w);
            assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-5 && w.iter().all(|&x| x >= 0.0));
            if route == Route::Born { assert!((w[1] - 1.44 / 5.53).abs() < 1e-4, "Born: |a|^2 / sum"); }
        }
    }

    #[test]
    fn preregistration_is_pinned() {
        assert_eq!(HX_PREREG, StrandCfg { p: 23, n_train: 265, k: 8, hb: 16, init: 0.1, gate_init: 0.5, lr: 1e-3, wd: 1.0, steps: 12_000, eval_every: 100 });
        let mut s = HX_SEEDS.to_vec(); s.sort(); s.dedup(); assert_eq!(s.len(), 30);
        for x in HX_SEEDS.iter() { assert!(!crate::brain_grokbed::GF_SEEDS.contains(x) && !crate::brain_grokbed::F_SEEDS.contains(x) && !crate::brain_grokbed::FRESH_SEEDS.contains(x)); }
    }

    /// PRE-REGISTERED (HxO seed): Born-rule routing ends with higher test accuracy than softmax routing on >= 20/30 seeds.
    /// Run: HX_THREADS=40 cargo test --release --lib brain_hxo::tests::falsifier_born -- --ignored --nocapture
    #[test]
    #[ignore = "slow: 30 seeds x 2 routes x 12 000 steps (run explicitly)"]
    fn falsifier_born_routing_beats_softmax() {
        let jobs: Vec<(u64, Route)> = HX_SEEDS.iter().flat_map(|&s| [(s, Route::Born), (s, Route::Softmax)]).collect();
        let th = env("HX_THREADS", 16usize);
        let out = std::sync::Mutex::new(vec![None; jobs.len()]); let next = std::sync::atomic::AtomicUsize::new(0);
        std::thread::scope(|s| for _ in 0..th.min(jobs.len()) { s.spawn(|| loop { let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst); if i >= jobs.len() { break; }
            let r = run_strands(jobs[i].0, &HX_PREREG, jobs[i].1); out.lock().unwrap()[i] = Some(r); }); });
        let r: Vec<StrandRun> = out.into_inner().unwrap().into_iter().map(|x| x.unwrap()).collect();
        for k in 0..30 { let (b, s) = (&r[2 * k], &r[2 * k + 1]); std::println!("  route seed {}: end Born {:.3} Softmax {:.3} | gen {:?} {:?} | route-entropy {:.3} {:.3}", HX_SEEDS[k], b.end_test, s.end_test, b.t_gen, s.t_gen, b.route_entropy, s.route_entropy); }
        let wins = (0..30).filter(|&k| r[2 * k].end_test > r[2 * k + 1].end_test).count();
        let m = |o: usize, f: fn(&StrandRun) -> f32| (0..30).map(|k| f(&r[2 * k + o])).sum::<f32>() / 30.0;
        std::println!("BORN: Born > Softmax {}/30 (p {:.4}) | end test Born {:.3} Softmax {:.3} | route entropy Born {:.3} Softmax {:.3} | generalized Born {}/30 Softmax {}/30",
            wins, crate::brain_grok3::sign_p(wins, 30), m(0, |x| x.end_test), m(1, |x| x.end_test), m(0, |x| x.route_entropy), m(1, |x| x.route_entropy),
            (0..30).filter(|&k| r[2 * k].t_gen.is_some()).count(), (0..30).filter(|&k| r[2 * k + 1].t_gen.is_some()).count());
        assert!(wins >= 20, "Born > Softmax on {}/30", wins);
    }

    fn dev_cfg() -> StrandCfg {
        StrandCfg { p: 23, n_train: 265, k: env("HX_K", 8), hb: env("HX_HB", 16), init: 0.1, gate_init: env("HX_GATE", 0.5), lr: 1e-3, wd: 1.0,
            steps: env("HX_STEPS", 12000), eval_every: 100 }
    }

    /// Dev, dev seeds only: cargo test --release --lib brain_hxo::tests::dev -- --ignored --nocapture
    #[test]
    #[ignore]
    fn dev() {
        let c = dev_cfg();
        std::println!("{:?}", c);
        let seeds: Vec<u64> = (0..env("HX_SEEDS", 6u64)).map(|i| 100 + i).collect();
        let jobs: Vec<(u64, Route)> = seeds.iter().flat_map(|&s| [(s, Route::Born), (s, Route::Softmax)]).collect();
        let r: Vec<StrandRun> = std::thread::scope(|sc| jobs.iter().map(|&(s, ro)| sc.spawn(move || run_strands(s, &c, ro))).collect::<Vec<_>>().into_iter().map(|h| h.join().unwrap()).collect());
        for (j, x) in jobs.iter().zip(r.iter()) { std::println!("seed {} {:?}: fit {:?} gen {:?} end {:.3} route-entropy {:.3}", j.0, x.route, x.t_fit, x.t_gen, x.end_test, x.route_entropy); }
    }
}
