//! Brain gym: the brain modules' pre-registered falsifiers, re-run on FRESH seeds.
//!
//! Every `brain_*` evolution module carries falsifier tests whose seeds were fixed when the
//! claim was registered. A claim that only held on lucky seeds stays green in `cargo test`
//! forever. The gym re-runs the cheapest of those falsifiers on seeds derived from a gym seed
//! (`struktura brain gym --seed N`; the CLI's default is the number of days since the UNIX
//! epoch, so a nightly run is fresh every night and reproducible from its seed).
//!
//! Each trial runs the SAME code as its test (the test bodies live in each module's
//! `falsify` submodule and the tests call them), with the same thresholds. A trial records the
//! status pre-registered today (`expect`: the test passes, or it is `#[ignore]`d as a recorded
//! failure). A **flip** is a trial whose result disagrees with that record:
//!
//! - expected pass, now failing: a regression, or a claim that only held on its seeds;
//! - expected fail, now passing: notable (look at it; one lucky seed is not a reversal).
//!
//! Shrinks against the tests (fewer seeds or runs, to keep the whole gym within its time
//! budget) are stated in each trial's `claim`. Trials run in parallel threads; each is a pure
//! function of its seed, so the output does not depend on scheduling.
//!
//! Deterministic, no model weights, no I/O. Needs `std` (threads, heap).

use std::string::String;
use std::vec::Vec;

/// The status pre-registered for a trial's claim today.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Expect { Pass, Fail }

impl Expect {
    pub fn as_str(self) -> &'static str { match self { Expect::Pass => "pass", Expect::Fail => "fail" } }
}

/// One falsifier, run once on fresh seeds.
#[derive(Clone, Debug, PartialEq)]
pub struct Trial {
    pub name: &'static str,
    /// The claim as tested here (including any shrink against the module's test).
    pub claim: &'static str,
    /// What `value` and `baseline` measure.
    pub metric: &'static str,
    pub value: f64,
    pub baseline: f64,
    /// The falsifier's verdict on these seeds (the test's asserts).
    pub pass: bool,
    pub expect: Expect,
}

impl Trial {
    /// The result disagrees with the pre-registered status.
    pub fn flipped(&self) -> bool { self.pass != (self.expect == Expect::Pass) }
    /// An expected-pass trial now failing.
    pub fn regression(&self) -> bool { self.expect == Expect::Pass && !self.pass }

    /// One human-readable line.
    pub fn line(&self) -> String {
        format!("  {:<4}  {:<11} expect {:<4} {:<5} value {:>10} vs baseline {:>10}  {} | {}",
            if self.pass { "pass" } else { "FAIL" }, self.name, self.expect.as_str(), if self.flipped() { "FLIP" } else { "" },
            num(self.value), num(self.baseline), self.metric, self.claim)
    }

    /// One JSON line.
    pub fn json(&self) -> String {
        format!("{{\"event\":\"trial\",\"name\":\"{}\",\"claim\":\"{}\",\"metric\":\"{}\",\"value\":{},\"baseline\":{},\"pass\":{},\"expect\":\"{}\",\"flip\":{}}}",
            esc(self.name), esc(self.claim), esc(self.metric), num(self.value), num(self.baseline), self.pass, self.expect.as_str(), self.flipped())
    }
}

/// A trial: a pure function of the gym seed.
pub type TrialFn = fn(u64) -> Trial;

/// The registry, in output order. Register a new trial with one line here.
pub const TRIALS: &[(&str, TrialFn)] = &[
    ("neuromod", trials::neuromod),
    ("hebb", trials::hebb),
    ("grow", trials::grow),
    ("shield", trials::shield),
    ("conformal", trials::conformal),
    ("regime", trials::regime),
    ("recorder", trials::recorder),
    ("sleep", trials::sleep),
    ("sleep-frees", trials::sleep_frees),
    ("genome", trials::genome),
    ("cycle", trials::cycle),
    ("grok", trials::grok),
    ("compress", trials::compress),
    ("grokking", trials::grokking),
];

/// The gym's result summary.
#[derive(Clone, Debug, PartialEq)]
pub struct Summary {
    pub seed: u64,
    pub passed: usize,
    pub failed: usize,
    /// (trial, its pre-registered status) for every flipped trial.
    pub flips: Vec<(&'static str, Expect)>,
}

impl Summary {
    pub fn new(seed: u64, trials: &[Trial]) -> Self {
        Summary { seed, passed: trials.iter().filter(|t| t.pass).count(), failed: trials.iter().filter(|t| !t.pass).count(),
            flips: trials.iter().filter(|t| t.flipped()).map(|t| (t.name, t.expect)).collect() }
    }
    /// An expected-pass trial failed on these seeds.
    pub fn regression(&self) -> bool { self.flips.iter().any(|(_, e)| *e == Expect::Pass) }

    fn flip_str(name: &str, e: Expect) -> String { format!("{} ({} -> {})", name, e.as_str(), if e == Expect::Pass { "fail" } else { "pass" }) }

    pub fn line(&self) -> String {
        let flips: Vec<String> = self.flips.iter().map(|(n, e)| Self::flip_str(n, *e)).collect();
        format!("summary: seed {}, passed {}, failed {}, flips: {}", self.seed, self.passed, self.failed,
            if flips.is_empty() { String::from("none") } else { flips.join(", ") })
    }

    pub fn json(&self) -> String {
        let flips: Vec<String> = self.flips.iter()
            .map(|(n, e)| format!("{{\"name\":\"{}\",\"expect\":\"{}\",\"regression\":{}}}", esc(n), e.as_str(), *e == Expect::Pass)).collect();
        format!("{{\"event\":\"summary\",\"seed\":{},\"passed\":{},\"failed\":{},\"flips\":[{}]}}", self.seed, self.passed, self.failed, flips.join(","))
    }
}

/// Runs the registered trials (all, or those named in `only`) in parallel threads; results in
/// registry order. `Err(name)` if a name in `only` is not registered.
pub fn run(seed: u64, only: Option<&[&str]>) -> Result<Vec<Trial>, String> {
    if let Some(names) = only {
        if let Some(bad) = names.iter().find(|n| !TRIALS.iter().any(|(t, _)| t == *n)) { return Err(String::from(*bad)); }
    }
    let picked: Vec<TrialFn> = TRIALS.iter().filter(|(n, _)| only.map(|o| o.contains(n)).unwrap_or(true)).map(|(_, f)| *f).collect();
    Ok(std::thread::scope(|sc| {
        let handles: Vec<_> = picked.iter().map(|f| { let f = *f; sc.spawn(move || f(seed)) }).collect();
        handles.into_iter().map(|h| h.join().expect("a gym trial panicked")).collect()
    }))
}

/// The default gym seed: whole days since the UNIX epoch (a new seed every night).
pub fn today() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() / 86_400).unwrap_or(0)
}

/// SplitMix64 step.
fn splitmix(z: u64) -> u64 {
    let mut z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `K` fresh seeds for trial `name` from the gym seed (independent per trial, never 0: 0 is a
/// fixed point of the tests' xorshift generators).
pub fn seeds<const K: usize>(seed: u64, name: &str) -> [u64; K] {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for b in name.bytes() { h ^= b as u64; h = h.wrapping_mul(0x0100_0000_01B3); }
    let mut s = splitmix(seed ^ h);
    core::array::from_fn(|_| { s = splitmix(s); if s == 0 { 1 } else { s } })
}

/// f32 rounded to 4 decimals, as f64 through its shortest decimal (stable, readable output:
/// 3340.6, not 3340.60009765625 or 3340.6001).
fn dec(x: f32) -> f64 { let y = (x * 1e4).round() / 1e4; format!("{}", y).parse().unwrap_or(f64::NAN) }

fn num(x: f64) -> String { if x.is_finite() { format!("{}", x) } else { String::from("null") } }

fn esc(s: &str) -> String { s.replace('\\', "\\\\").replace('"', "\\\"") }

/// The trials. Each calls its module's `falsify` code (the test's own) on fresh seeds.
mod trials {
    use super::{dec, seeds, Expect, Trial};
    use crate::{brain_conformal, brain_cycle, brain_genome, brain_grok, brain_grow, brain_hebb, brain_neuromod, brain_recorder,
        brain_regime, brain_shield, brain_sleep};

    pub fn neuromod(seed: u64) -> Trial {
        let r = brain_neuromod::falsify::falsifier(&seeds::<8>(seed, "neuromod"));
        Trial { name: "neuromod", expect: Expect::Pass,
            claim: "after a rule flip the neuromodulated brain recovers to 90% faster, has lower regret, and is not worse than 1 pt when stationary (8 fresh seeds; test: 8+8)",
            metric: "recovery steps (neuromod vs fixed)", value: dec(r.rec_m / r.k), baseline: dec(r.rec_f / r.k), pass: r.holds() }
    }

    pub fn hebb(seed: u64) -> Trial {
        let r = brain_hebb::falsify::falsifier_a(&seeds::<8>(seed, "hebb"));
        Trial { name: "hebb", expect: Expect::Pass,
            claim: "Hebbian depth-1 proposals match the exhaustive grower within 1 pt (or better) with strictly fewer held-out evaluations (falsifier (a); 8 fresh seeds; test: 8+8)",
            metric: "right-action last 1000 (hebb vs grower)", value: dec(r.hebb), baseline: dec(r.grow), pass: r.holds() }
    }

    pub fn grow(seed: u64) -> Trial {
        let r = brain_grow::falsify::falsifier(&seeds::<8>(seed, "grow"));
        Trial { name: "grow", expect: Expect::Pass,
            claim: "growth beats no growth by >= 3 pts on the interaction world and grows < 1 feature/run on a linear world (8 fresh seeds; test: 8+8)",
            metric: "right-action rate last 1000 (growing vs plain)", value: dec(r.grow / r.k), baseline: dec(r.plain / r.k), pass: r.holds() }
    }

    pub fn shield(seed: u64) -> Trial {
        let s = seeds::<16>(seed, "shield");
        let pairs: [(u64, u64); 8] = core::array::from_fn(|k| (s[2 * k], s[2 * k + 1]));
        let (pc, sc, _, regret) = brain_shield::falsify::campaign(0.03, 12000, &pairs, None);
        Trial { name: "shield", expect: Expect::Pass,
            claim: "certified shield (rule of three, eps 3%, 12000-step ground campaign): zero catastrophes over 8 x 3000 live decisions, regret <= 10%, while the unshielded brain loses rovers (8 fresh world/terrain seed pairs, as the test)",
            metric: "catastrophes (shielded vs unshielded)", value: sc as f64, baseline: pc as f64, pass: brain_shield::falsify::certified_holds(pc, sc, regret) }
    }

    pub fn conformal(seed: u64) -> Trial {
        use brain_conformal::falsify::{holds, run, truth_a, truth_b};
        let alpha = 0.1f32;
        let (mut pass, mut worst) = (true, 0.0f32);
        for (w, truth) in [truth_a as fn(&[f32; 4]) -> u8, truth_b].into_iter().enumerate() {
            for s in seeds::<3>(seed, if w == 0 { "conformal-a" } else { "conformal-b" }) {
                let (s1, s2) = run(truth, alpha, 0.05, s);
                pass &= holds(&s1, &s2, alpha);
                for st in [&s1, &s2] { if st.acted > 0 { worst = worst.max(100.0 * st.errors as f32 / st.acted as f32); } }
            }
        }
        Trial { name: "conformal", expect: Expect::Pass,
            claim: "both conformal guards keep error among acted decisions <= alpha 10%, interval coverage >= 90% (2-pt slack), selective rule acts on >= half (2 worlds x 3 fresh seeds, as the test)",
            metric: "worst error among acted, % (vs target alpha)", value: dec(worst), baseline: dec(100.0 * alpha), pass }
    }

    pub fn regime(seed: u64) -> Trial {
        use brain_regime::falsify::{falsifier, mean};
        let r = falsifier(&seeds::<5>(seed, "regime"), false);
        Trial { name: "regime", expect: Expect::Pass,
            claim: "after the lunar-night flip the regime-aware brain recovers faster (mean, and on >= 4 of 5 seeds) with no false change in stationary runs (5 fresh seeds, as the test)",
            metric: "recovery steps (regime vs plain)", value: dec(mean(&r.regime_t)), baseline: dec(mean(&r.plain_t)), pass: r.holds() }
    }

    pub fn recorder(seed: u64) -> Trial {
        use brain_recorder::falsify::{falsifier, STEPS};
        let s = seeds::<2>(seed, "recorder");
        let flip = ((s[1] % STEPS as u64) as usize, ((s[1] >> 32) % 3) as usize, 0u32);
        let r = falsifier(s[0], flip);
        Trial { name: "recorder", expect: Expect::Pass,
            claim: "the ground replay reproduces every flight record bit-exact, and one flipped low mantissa bit in one situation is caught at exactly that decision (fresh flight stream and flip position; test: seed 42, decision 1234)",
            metric: "records reproduced bit-exact (of 2000)", value: r.exact.matched as f64, baseline: r.exact.checked as f64, pass: r.holds() }
    }

    fn sleep_reports(seed: u64) -> [brain_sleep::falsify::Report; 2] {
        use brain_sleep::falsify::{falsifier, truth_a, truth_b};
        let s = seeds::<8>(seed, "sleep");
        [falsifier(truth_a, &s), falsifier(truth_b, &s)]
    }

    pub fn sleep(seed: u64) -> Trial {
        let r = sleep_reports(seed);
        Trial { name: "sleep", expect: Expect::Fail,
            claim: "sleep raises held-out accuracy in both worlds and frees memory (N = 32, 6000 outcomes; 8 fresh seeds; test: 8+8, ignored as FAILED)",
            metric: "held-out right-action, mean of 2 worlds (sleep vs none)",
            value: dec((r[0].slept / r[0].k + r[1].slept / r[1].k) / 2.0), baseline: dec((r[0].plain / r[0].k + r[1].plain / r[1].k) / 2.0),
            pass: r.iter().all(|w| w.holds()) }
    }

    pub fn sleep_frees(seed: u64) -> Trial {
        let r = sleep_reports(seed);
        let worst = r.iter().map(|w| (w.slept - w.plain) / w.k).fold(f32::INFINITY, f32::min);
        Trial { name: "sleep-frees", expect: Expect::Pass,
            claim: "post-hoc weaker claim: sleep frees memory without a measurable held-out loss (change >= -0.01, both worlds; 8 fresh seeds; test: 8+8)",
            metric: "worst held-out change with sleep (vs floor)", value: dec(worst), baseline: -0.01, pass: r.iter().all(|w| w.frees_without_loss()) }
    }

    pub fn genome(seed: u64) -> Trial {
        let s = seeds::<7>(seed, "genome");
        let r = brain_genome::falsify::falsifier(&s[..6], &s[6..]);
        Trial { name: "genome", expect: Expect::Fail,
            claim: "an evolved genome (8 x 12 generations on one training stream) beats the defaults on a held-out world, with elitism (SHRUNK: 1 evolution run instead of 3, pass = it wins; 6 fresh held-out seeds as the test)",
            metric: "held-out right-action (evolved vs default)", value: dec(r.runs[0].held), baseline: dec(r.default_held), pass: r.holds() }
    }

    pub fn grok(seed: u64) -> Trial {
        let t = brain_grok::trial(seeds::<1>(seed, "grok")[0]);
        Trial { name: "grok", expect: Expect::Fail,
            claim: "(a+b) mod 7, fixed train set replayed: brain + compression shows a late held-out jump of >= 30 pts after a train plateau, the same brain without compression does not (ONE fresh seed, one-hot; test: 5 seeds x 2 encodings, ignored as FAILED 0/5: the cycle watches the memorized train stream)",
            metric: "held-out accuracy at the end (compression vs none)", value: dec(t.heldout_a), baseline: dec(t.heldout_b), pass: t.pass }
    }


    pub fn compress(seed: u64) -> Trial {
        use crate::brain_grok::Encoding;
        let s = seeds::<6>(seed, "compress");
        let t: [crate::brain_grok2::GrokTrial2; 6] = std::thread::scope(|sc| s.map(|x| sc.spawn(move || crate::brain_grok2::trial(x, Encoding::Fourier))).map(|h| h.join().unwrap()));
        let wins = t.iter().filter(|x| x.heldout_a > x.heldout_b).count();
        Trial { name: "compress", expect: Expect::Pass,
            claim: "(a+b) mod 7, Fourier: the growth cycle with compression ends above the same cycle without compression on held-out on all 6 fresh seeds (sign p 0.016; test brain_grok3: 12/12, p 0.0002)",
            metric: "mean held-out at the end (compression vs none)", value: dec(t.iter().map(|x| x.heldout_a).sum::<f32>() / 6.0),
            baseline: dec(t.iter().map(|x| x.heldout_b).sum::<f32>() / 6.0), pass: wins == 6 }
    }

    pub fn grokking(seed: u64) -> Trial {
        let s = seeds::<3>(seed, "grokking");
        let t: [crate::brain_grokbed::Trial; 3] = std::thread::scope(|sc| s.map(|x| sc.spawn(move || crate::brain_grokbed::trial(x))).map(|h| h.join().unwrap()));
        Trial { name: "grokking", expect: Expect::Pass,
            claim: "(a+b) mod 23, learned embeddings + quadratic MLP: with weight decay it fits early and generalizes >= 3x later (test >= 0.95, stays); without decay it does not (all 3 fresh seeds; test brain_grokbed: 10/10)",
            metric: "end test accuracy (decay vs none)", value: dec(t.iter().map(|x| x.decay.end_test).sum::<f32>() / 3.0),
            baseline: dec(t.iter().map(|x| x.control.end_test).sum::<f32>() / 3.0), pass: t.iter().all(|x| x.pass) }
    }

    pub fn cycle(seed: u64) -> Trial {
        use brain_cycle::falsify::{defaults, falsifier};
        let r = falsifier(&seeds::<8>(seed, "cycle"), defaults, false);
        Trial { name: "cycle", expect: Expect::Fail,
            claim: "grow -> compress cycle matches grow-only within 1 pt with fewer active params and a grokking event in >= 5 of 8 runs (8 fresh seeds; test: 8+8, ignored as FAILED on fresh seeds)",
            metric: "right-action last 1000 (cycle vs grow-only)", value: dec(r.rc / r.k), baseline: dec(r.rg / r.k), pass: r.holds() }
    }
}
