//! Mind: the loop's native brain. It replays every scored prediction in the
//! ledger into a [`crate::brain::Brain`] that learns which experiments teach
//! the lab something, then estimates the information yield of each candidate
//! experiment and cites the past predictions its estimate rests on.
//!
//! Situation (9 features), known before an experiment runs:
//! - the claim: relative (A vs B), threshold ambition in noise bands,
//!   equivalence test, speed metric, quality metric;
//! - the knob's own history: prior effect in noise bands, how often the knob
//!   was measured, and how often its past tests were decisive;
//! - a constant.
//!
//! A past prediction is attributed to a knob when a token of its name equals
//! one of the knob's arm labels (e.g. `ub1024`, `draft3`); history features of
//! a past prediction use only predictions scored before it (no leakage).
//!
//! Reward, known after it was scored: 1 = decisive verdict; 0.5 = an easy pass
//! (margin > 3 noise floors: it was a safe bet); 0 = fragile (margin inside the
//! single-run band, for a noisy measurement: [`crate::lab::band_applies`], the rule
//! the lab report, lessons and oracle use) or void (instrument bug: the run taught
//! nothing). An exact answer (a count, a needle, identical output) at its threshold
//! cannot flip within throughput noise, so it is decisive, not fragile.
//!
//! A yield is the brain's estimate shrunk toward the base rate (mean reward so
//! far) by [`Mind::trust`], which the mind earns only from its own prequential
//! record: with no demonstrated skill its yields are the base rate.
//!
//! The mind only re-orders challengers among themselves; fragile re-measures,
//! constraints and missing instruments are decided by the agenda's rules.

#![cfg(feature = "std")]

use crate::brain::Brain;
use crate::lab::LabReport;
use super::knobs::{parse_knobs, Knob, BUILTIN};

pub const D: usize = 9;
pub type Situation = [f32; D];
/// Growth slots: senses the mind may grow from its base features while replaying the ledger.
pub const G: usize = 4;
/// Brain situation size: base features plus growth slots.
pub const DG: usize = D + G;
/// Names of the base features, for describing grown senses.
pub const BASE_NAMES: [&str; D] = ["relative", "ambition", "equivalence", "speed", "quality", "prior_effect", "times_measured", "decisive_rate", "1"];

/// One row of the lab's feature registry (`~/.oura/features.tsv`): a feature on the path to prod
/// and the predictions that prove it. This is the main goal the mind's reward is aimed at.
#[derive(Clone, Debug, PartialEq)]
pub struct GoalFeature { pub feature: String, pub pred: String, pub required: Vec<String> }

/// Parse the registry: `feature commit env needs pred required value flags` (tab-separated, `#` comments).
pub fn parse_goal_features(text: &str) -> Vec<GoalFeature> {
    text.lines().filter(|l| !l.trim_start().starts_with('#') && !l.trim().is_empty()).filter_map(|l| {
        let c: Vec<&str> = l.split('\t').collect();
        if c.len() < 6 { return None; }
        let required = if c[5].trim() == "-" { Vec::new() } else { c[5].split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect() };
        Some(GoalFeature { feature: c[0].trim().into(), pred: c[4].trim().into(), required })
    }).collect()
}

fn describe(f: &crate::brain_grow::Feat) -> String {
    use crate::brain_grow::Feat;
    match *f {
        Feat::Prod(i, j) => format!("{} x {}", BASE_NAMES[i as usize], BASE_NAMES[j as usize]),
        Feat::Step(i, t) => format!("{} > {:.2}", BASE_NAMES[i as usize], t),
        Feat::Gt(i, j) => format!("{} > {}", BASE_NAMES[i as usize], BASE_NAMES[j as usize]),
        Feat::Off => "-".into(),
    }
}

fn has(name: &str, words: &[&str]) -> bool {
    let n = name.to_ascii_lowercase();
    words.iter().any(|w| n.contains(w))
}

const SPEED: &[&str] = &["tps", "ms", "fast", "slow", "pp", "prefill", "decode", "ttd", "speed", "_s", "latency", "throughput"];
const QUALITY: &[&str] = &["pass", "found", "needle", "correct", "acc", "identical", "quality", "solved", "ppl"];

/// What the lab knew about a knob before an experiment.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct History {
    /// Scored predictions attributed to the knob.
    pub n: usize,
    /// Of those, decisive verdicts (reward 1).
    pub decisive: usize,
    /// Sum of |measured effect| in noise bands over relative predictions, and their count.
    pub effect_bands_sum: f64,
    pub effect_n: usize,
}

impl History {
    fn features(&self, prior_effect_bands: Option<f64>) -> [f32; 3] {
        let eff = prior_effect_bands.or(if self.effect_n > 0 { Some(self.effect_bands_sum / self.effect_n as f64) } else { None });
        let e = eff.map(|e| (e.abs().min(10.0) / 10.0) as f32).unwrap_or(0.0);
        let times = self.n as f32 / (self.n as f32 + 2.0);
        let rate = (self.decisive as f32 + 0.5) / (self.n as f32 + 1.0);
        [e, times, rate]
    }
}

fn claim(relative: bool, ambition: f64, equivalence: bool, name_or_metric: &str) -> [f32; 5] {
    let quality = has(name_or_metric, QUALITY);
    let speed = has(name_or_metric, SPEED) || (relative && !quality);
    [relative as u8 as f32, ambition as f32, equivalence as u8 as f32, speed as u8 as f32, quality as u8 as f32]
}

fn join(c: [f32; 5], h: [f32; 3]) -> Situation {
    [c[0], c[1], c[2], c[3], c[4], h[0], h[1], h[2], 1.0]
}

fn ambition(thr: f64, band: f64) -> f64 {
    if band > 0.0 && thr.is_finite() { (thr.abs() / band).min(10.0) / 10.0 } else { 0.0 }
}

/// Situation of a scored prediction (from lab-score's op and value text) as it looked before the run,
/// given the history of its knob up to then.
pub fn situation_of(op: &str, value: &str, name: &str, band_pct: f64, history: &History) -> Situation {
    let relative = op.ends_with('%');
    let thr = if relative { super::learn::parse_relative(value).map(|(_, _, p)| p).unwrap_or(0.0) } else { 0.0 };
    join(claim(relative, if relative { ambition(thr, band_pct) } else { 0.0 }, op == "~%", name), history.features(None))
}

/// Situation of a planned relative test on `metric` at `threshold_pct`, with no knob history.
/// Kept for callers that do not know the knob; prefer [`Mind::situation_for`].
pub fn situation_planned(metric: &str, threshold_pct: f64, band_pct: f64) -> Situation {
    join(claim(true, ambition(threshold_pct, band_pct), false, metric), History::default().features(None))
}

/// Tokens of a prediction name (split on anything that is not a letter or digit).
fn tokens(name: &str) -> impl Iterator<Item = String> + '_ {
    name.split(|c: char| !c.is_ascii_alphanumeric()).filter(|t| !t.is_empty()).map(|t| t.to_ascii_lowercase())
}

/// Knob a prediction is about: a name token equal to one of the knob's arm labels that mixes
/// letters and digits (`ub512`, `draft3`), or to the knob name followed by a value (`chunk248`).
/// Plain words and numbers (`stock`, `64`) are too generic to attribute by.
fn knob_of<'a>(name: &str, knobs: &'a [Knob]) -> Option<&'a Knob> {
    let toks: Vec<String> = tokens(name).collect();
    knobs.iter().find(|k| k.values.iter().any(|v| {
        let label = k.arm_of(v).1.to_ascii_lowercase();
        let specific = label.chars().any(|c| c.is_ascii_digit()) && label.chars().any(|c| c.is_ascii_alphabetic());
        let named = format!("{}{}", k.name.to_ascii_lowercase(), v.to_ascii_lowercase());
        toks.iter().any(|t| (specific && *t == label) || *t == named)
    }))
}

pub struct Mind {
    brain: Box<Brain<256, DG, 1>>,
    grower: crate::brain_grow::Grower<D, G>,
    /// Senses grown while replaying the ledger: (after how many predictions, sense, held-out gain).
    pub grown: Vec<(usize, String, f32)>,
    /// Whether the reward was aimed at the lab's feature registry (main goal).
    pub goal_aware: bool,
    /// Scored predictions that prove a registered feature.
    pub goal_predictions: usize,
    /// "pred::name" per memory slot, for citations.
    labels: Vec<String>,
    pub episodes: usize,
    band_pct: f64,
    /// Noise floor (percent): a pass by more than 3 floors is an easy pass.
    floor_pct: f64,
    /// Knob history after the whole ledger, by knob name.
    history: Vec<(String, History)>,
    /// Sum and count of the rewards learned so far: their mean is the base rate.
    reward_sum: f64,
    reward_n: usize,
    /// The mind's own prequential record (see [`Mind::trust`]): over learned predictions, each forecast
    /// before it was learned, sums of (forecast - base rate) x (reward - base rate) and (forecast - base rate)^2.
    skill_xy: f64,
    skill_xx: f64,
}

/// Reward of one scored verdict: 1 = decisive; 0.5 = easy pass (margin > 3 floors); 0 = fragile
/// (margin inside the single-run band, only for a `noisy` measurement: pass
/// [`crate::lab::band_applies`] of the prediction's name and value) or void. `band` and `floor` are percent.
pub fn reward(verdict: &str, margin_pct: Option<f64>, band: f64, floor: f64, noisy: bool) -> f64 {
    match (verdict, margin_pct) {
        ("void", _) => 0.0,
        (_, Some(m)) if noisy && m.abs() < band => 0.0,
        ("pass", Some(m)) if m > 3.0 * floor => 0.5,
        _ => 1.0,
    }
}

/// Whether a prediction proves a registered feature of the lab's goal registry.
pub fn proves_goal(goal: &[GoalFeature], pred: &str, name: &str) -> bool {
    goal.iter().any(|g| g.pred == pred && (g.required.is_empty() || g.required.iter().any(|r| r == name)))
}

/// Reward aimed at the main goal: full reward for a prediction that proves a registered feature,
/// 0.6 of it for any other (no discount when the registry is empty).
pub fn goal_reward(goal: &[GoalFeature], pred: &str, name: &str, reward: f64) -> f64 {
    if goal.is_empty() || proves_goal(goal, pred, name) { reward } else { reward * 0.6 }
}

/// The scored predictions (latest verdict each: pass, fail or void) the mind learns from, in its
/// learning order: time, then pred file, then name.
pub fn learning_order(lab: &LabReport) -> Vec<&crate::lab::Prediction> {
    let mut preds: Vec<&crate::lab::Prediction> = lab.predictions.iter()
        .filter(|p| matches!(p.verdict.as_str(), "pass" | "fail" | "void")).collect();
    preds.sort_by(|a, b| a.ts.partial_cmp(&b.ts).unwrap_or(std::cmp::Ordering::Equal).then(a.pred.cmp(&b.pred)).then(a.name.cmp(&b.name)));
    preds
}

#[derive(Clone, Debug)]
pub struct Yield {
    /// Expected information yield in [0, 1] (1 = decisive): the brain's estimate shrunk toward the
    /// base rate by the mind's [`Mind::trust`].
    pub expected: f64,
    /// The brain's own estimate (model + memory), before shrinkage.
    pub raw: f64,
    pub evidence: f64,
    pub abstained: bool,
    /// Past predictions this estimate rests on, nearest first.
    pub cites: Vec<String>,
}

impl Mind {
    /// Learn from every scored prediction in the ledger, attributing predictions to the built-in knobs.
    pub fn from_lab(lab: &LabReport) -> Self {
        let knobs = parse_knobs(BUILTIN).unwrap_or_default();
        Self::from_lab_knobs(lab, &knobs)
    }

    /// Learn from every scored prediction (latest verdict each, in time order), attributing each to
    /// one of `knobs` when its name names one of the knob's arms.
    pub fn from_lab_knobs(lab: &LabReport, knobs: &[Knob]) -> Self { Self::from_lab_goal(lab, knobs, &[]) }

    /// Like [`Mind::from_lab_knobs`], with the reward aimed at the main goal: when `goal` (the lab's
    /// feature registry) is given, a prediction that proves a registered feature keeps its full
    /// reward and every other prediction is discounted to 0.6 of it. While replaying, the mind may
    /// grow up to G new senses from its base features, each only if it explains its mistakes on
    /// held-out predictions (see [`crate::brain_grow`]).
    pub fn from_lab_goal(lab: &LabReport, knobs: &[Knob], goal: &[GoalFeature]) -> Self {
        let mut m = Self::blank(lab, knobs, goal);
        for p in learning_order(lab) { m.learn_prediction(p, knobs, goal); }
        m
    }

    /// A mind that has learned nothing yet: the ledger's noise band and floor, empty knob histories.
    /// [`Mind::from_lab_goal`] is this plus [`Mind::learn_prediction`] over [`learning_order`].
    pub fn blank(lab: &LabReport, knobs: &[Knob], goal: &[GoalFeature]) -> Self {
        let brain: Box<Brain<256, DG, 1>> = Box::new(Brain::new(0.0, 1.0));
        let mut grower: crate::brain_grow::Grower<D, G> = crate::brain_grow::Grower::new(0x6D696E64);
        grower.constant[D - 1] = true;
        grower.every = 12;
        Mind { brain, grower, grown: Vec::new(), goal_aware: !goal.is_empty(), goal_predictions: 0, labels: vec![String::new(); 256], episodes: 0,
            band_pct: lab.pair_band_pct.max(lab.floor_pct), floor_pct: lab.floor_pct,
            history: knobs.iter().map(|k| (k.name.clone(), History::default())).collect(),
            reward_sum: 0.0, reward_n: 0, skill_xy: 0.0, skill_xx: 0.0 }
    }

    /// The reward this mind learns for a scored prediction (aimed at the goal when `goal` is given).
    pub fn reward_of(&self, p: &crate::lab::Prediction, goal: &[GoalFeature]) -> f64 {
        goal_reward(goal, &p.pred, &p.name, reward(&p.verdict, p.margin_pct, self.band_pct, self.floor_pct, crate::lab::band_applies(&p.name, &p.value)))
    }

    /// Situation of a scored prediction as the mind sees it now: its claim plus its knob's history so far.
    /// Before [`Mind::learn_prediction`] of `p`, this is exactly the situation that step learns from.
    pub fn situation_of_prediction(&self, p: &crate::lab::Prediction, knobs: &[Knob]) -> Situation {
        let h = knob_of(&p.name, knobs).and_then(|k| self.history.iter().find(|(n, _)| *n == k.name)).map(|(_, h)| *h).unwrap_or_default();
        situation_of(&p.op, &p.value, &p.name, self.band_pct, &h)
    }

    /// One replay step: learn a scored prediction, then update its knob's history. Returns the reward.
    pub fn learn_prediction(&mut self, p: &crate::lab::Prediction, knobs: &[Knob], goal: &[GoalFeature]) -> f64 {
        let raw = reward(&p.verdict, p.margin_pct, self.band_pct, self.floor_pct, crate::lab::band_applies(&p.name, &p.value));
        let is_goal = proves_goal(goal, &p.pred, &p.name);
        if is_goal { self.goal_predictions += 1; }
        let reward = goal_reward(goal, &p.pred, &p.name, raw);
        let knob = knob_of(&p.name, knobs).map(|k| k.name.clone());
        let base = self.situation_of_prediction(p, knobs);
        let x: [f32; DG] = self.grower.situation(&base);
        // The mind's own record: its forecast for this prediction, made before learning it.
        if let Some(b) = self.base_rate() {
            let d = self.brain.decide(&x, &[true], 0);
            if !d.abstained {
                let dev = (d.expected as f64).clamp(0.0, 1.0) - b;
                self.skill_xy += dev * (reward - b);
                self.skill_xx += dev * dev;
            }
        }
        self.reward_sum += reward;
        self.reward_n += 1;
        let class = if reward == 0.0 { "fragile/void" } else if p.verdict == "pass" && p.margin_pct.map(|m| m > 3.0 * self.floor_pct).unwrap_or(false) { "easy" } else { "decisive" };
        if let Some(s) = self.brain.learn(&x, 0, reward as f32) { self.labels[s] = format!("{}::{} ({}{})", p.pred, p.name, class, if is_goal { ", proves a feature" } else { "" }); }
        self.episodes += 1;
        if let Some(g) = self.grower.after_learn(&mut self.brain) { if let Some(f) = g.adopted { self.grown.push((self.episodes, describe(&f), g.gain)); } }
        // Update the knob's history only after its situation was taken (no leakage).
        let band = self.band_pct;
        if let Some(k) = knob {
            if let Some((_, h)) = self.history.iter_mut().find(|(n, _)| *n == k) {
                h.n += 1;
                if reward >= 1.0 { h.decisive += 1; }
                if let Some((d, _, _)) = super::learn::parse_relative(&p.value) {
                    if band > 0.0 { h.effect_bands_sum += d.abs() / band; h.effect_n += 1; }
                }
            }
        }
        reward
    }

    /// History of `knob` over the whole ledger.
    pub fn history(&self, knob: &str) -> History {
        self.history.iter().find(|(n, _)| n == knob).map(|(_, h)| *h).unwrap_or_default()
    }

    /// Situation of a planned challenger test: the knob's metric at `threshold_pct`, with the knob's
    /// history and, when job logs measured this value already, its observed effect (percent, positive
    /// = better) taken as the prior effect.
    pub fn situation_for(&self, knob: &Knob, threshold_pct: f64, observed_effect_pct: Option<f64>) -> Situation {
        let prior = observed_effect_pct.and_then(|e| if self.band_pct > 0.0 { Some(e / self.band_pct) } else { None });
        join(claim(true, ambition(threshold_pct, self.band_pct), false, &knob.metric), self.history(&knob.name).features(prior))
    }

    /// Expected information yield of a challenger test (see [`Mind::situation_for`]).
    pub fn estimate_challenger(&self, knob: &Knob, threshold_pct: f64, observed_effect_pct: Option<f64>) -> Yield {
        self.estimate(&self.situation_for(knob, threshold_pct, observed_effect_pct))
    }

    /// Mean reward learned so far (what `track` calls the running-mean baseline); None before any.
    pub fn base_rate(&self) -> Option<f64> {
        if self.reward_n > 0 { Some(self.reward_sum / self.reward_n as f64) } else { None }
    }

    /// How much of its own deviation from the base rate the mind's yields keep, in [0, 1].
    ///
    /// Earned from its prequential record only: over the predictions it learned, its forecast
    /// before learning each one versus the reward it then learned. Regressing (reward - base) on
    /// (forecast - base) through the origin, with a N(0, 1) prior on the slope and the largest
    /// variance a reward in [0, 1] can have (1/4) as noise: trust = Sxy / (Sxx + 1/4), clamped to
    /// [0, 1]. No record or no demonstrated skill gives 0: its yields are then the base rate, the
    /// best forecast it can justify. Deviations that turned out right raise trust; deviations
    /// that turned out wrong (stale or noisy memories) lower it.
    pub fn trust(&self) -> f64 {
        let t = self.skill_xy / (self.skill_xx + 0.25);
        if t.is_finite() { t.clamp(0.0, 1.0) } else { 0.0 }
    }

    pub fn estimate(&self, x: &Situation) -> Yield {
        let xx: [f32; DG] = self.grower.situation(x);
        let d = self.brain.decide(&xx, &[true], 0);
        let cites = (0..d.n_cited as usize).take(3).filter_map(|k| {
            let s = d.cited[k] as usize;
            self.brain.episode(s).map(|_| self.labels[s].clone())
        }).collect();
        let raw = (d.expected as f64).clamp(0.0, 1.0);
        let expected = match self.base_rate() { Some(b) => b + self.trust() * (raw - b), None => raw };
        Yield { expected, raw, evidence: d.evidence as f64, abstained: d.abstained, cites }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ledger(rows: &[(&str, &str, &str, &str)]) -> String {
        let mut s = String::from("{\"kind\":\"cal\",\"ok\":true,\"code\":{\"tps\":200.0}}\n{\"kind\":\"cal\",\"ok\":true,\"code\":{\"tps\":202.0}}\n");
        for (i, (name, op, value, verdict)) in rows.iter().enumerate() {
            s += &format!("{{\"kind\":\"prediction\",\"ts\":{},\"pred\":\"{}.tsv\",\"name\":\"{}\",\"value\":\"{}\",\"op\":\"{}\",\"threshold\":\"arm B\",\"verdict\":\"{}\"}}\n", i + 1, i, name, value, op, verdict);
        }
        s
    }

    #[test]
    fn learns_that_thresholds_near_the_noise_band_are_fragile() {
        // Band here ~1.99%. Ambitious thresholds (~5x band) were decisive; thresholds at the band were fragile.
        let mut rows = Vec::new();
        for i in 0..12 {
            rows.push(("tps-big", ">%", if i % 2 == 0 { "130 vs 100 (30.00%, need >% 10%)" } else { "128 vs 100 (28.00%, need >% 10%)" }, "pass"));
            rows.push(("tps-near", ">%", "102 vs 100 (2.30%, need >% 2%)", "pass"));
        }
        let l = ledger(&rows);
        let lab = crate::lab::analyze(&l);
        let m = Mind::from_lab(&lab);
        assert_eq!(m.episodes, 24);
        let band = lab.pair_band_pct.max(lab.floor_pct);
        let ambitious = m.estimate(&situation_planned("code_tps", 5.0 * band, band));
        let timid = m.estimate(&situation_planned("code_tps", band, band));
        assert!(!ambitious.abstained && !timid.abstained);
        assert!(ambitious.expected > timid.expected + 0.3, "{:?} vs {:?}", ambitious, timid);
        assert!(!ambitious.cites.is_empty() && ambitious.cites[0].contains("tps-big"), "{:?}", ambitious.cites);
    }

    #[test]
    fn goal_registry_parses_and_aims_the_reward() {
        let reg = "# comment\nchunk64\t33d0f7d\tENV=1\t-\t1.tsv\t-\tnone\t-\ngqa2\tx\tE=2\t-\t2.tsv\tfast-at-64K,correct\tnone\t-\n";
        let g = parse_goal_features(reg);
        assert_eq!(g.len(), 2);
        assert!(g[0].required.is_empty() && g[1].required == vec!["fast-at-64K".to_string(), "correct".to_string()]);
        // Same evidence pattern for two metric families; only the speed ones prove a registered feature.
        let mut rows = Vec::new();
        for _ in 0..10 {
            rows.push(("tps-fast-at-64K", ">%", "130 vs 100 (30.00%, need >% 10%)", "pass"));
            rows.push(("quality-correct-x", ">%", "130 vs 100 (30.00%, need >% 10%)", "pass"));
        }
        let mut l = ledger(&rows);
        // ledger() names pred files by row index; make every speed row belong to 2.tsv with the required name
        l = l.lines().map(|line| if line.contains("tps-fast-at-64K") { line.replace(|c: char| false, "").replacen("\"pred\":\"", "\"pred\":\"2.tsv-", 1) } else { line.to_string() }).collect::<Vec<_>>().join("\n");
        let lab = crate::lab::analyze(&l);
        let goal = vec![GoalFeature { feature: "gqa2".into(), pred: "2.tsv".into(), required: vec![] }];
        let plain = Mind::from_lab_goal(&lab, &[], &[]);
        let aimed = Mind::from_lab_goal(&lab, &[], &goal);
        assert!(aimed.goal_aware && !plain.goal_aware);
        assert_eq!(plain.goal_predictions, 0);
    }

    #[test]
    fn abstains_on_an_empty_ledger() {
        let lab = crate::lab::analyze(&ledger(&[]));
        let m = Mind::from_lab(&lab);
        assert!(m.estimate(&situation_planned("code_tps", 1.6, 1.6)).abstained);
    }

    #[test]
    fn exact_answers_at_their_threshold_are_decisive_not_fragile() {
        // The single-run band is throughput noise: it makes a timing claim inside it fragile, but an
        // exact answer (a count of correct answers) at its threshold cannot flip within it.
        assert_eq!(reward("pass", Some(0.0), 2.0, 1.5, true), 0.0);
        assert_eq!(reward("pass", Some(0.0), 2.0, 1.5, false), 1.0);
        assert_eq!(reward("void", None, 2.0, 1.5, false), 0.0);
        assert_eq!(reward("pass", Some(11.0), 2.0, 1.5, false), 0.5, "an easy pass stays easy");
        let mut l = ledger(&[("tps-near", ">%", "102 vs 100 (1.00%, need >% 1%)", "pass")]);
        l += "{\"kind\":\"prediction\",\"ts\":9,\"pred\":\"9.tsv\",\"name\":\"cand-short-quality\",\"value\":\"10\",\"op\":\">=\",\"threshold\":\"10\",\"verdict\":\"pass\"}\n";
        let lab = crate::lab::analyze(&l);
        let m = Mind::from_lab(&lab);
        let r: Vec<(String, f64)> = learning_order(&lab).iter().map(|p| (p.name.clone(), m.reward_of(p, &[]))).collect();
        assert_eq!(r, vec![("tps-near".to_string(), 0.0), ("cand-short-quality".to_string(), 1.0)]);
        assert_eq!(lab.fragile, 1, "the lab counts the same one fragile");
    }

    /// A world where the claim shape says nothing about the reward (fair coin per prediction).
    fn coin_world(n: usize, seed: u64) -> String {
        let mut s = seed;
        let mut rows: Vec<(String, &str, &str, &str)> = Vec::new();
        for i in 0..n {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let verdict = if (s >> 33) & 1 == 1 { "pass" } else { "void" };
            let (name, op, value) = match i % 3 {
                0 => (format!("needle-found-{}", i), "~", "FOUND"),
                1 => (format!("tps-gain-{}", i), ">%", "103 vs 100 (3.00%, need >% 0%)"),
                _ => (format!("restore-{}", i), "<", "3"),
            };
            rows.push((name, op, value, verdict));
        }
        let r: Vec<(&str, &str, &str, &str)> = rows.iter().map(|(a, b, c, d)| (a.as_str(), *b, *c, *d)).collect();
        ledger(&r)
    }

    #[test]
    fn falls_back_to_the_base_rate_when_its_record_shows_no_skill() {
        // Raw, the brain chases the coin (its memory trusts 8 noisy neighbours) and loses to the base
        // rate; its own record earns it little trust, so its yields stay near the base rate.
        for seed in 1..=4u64 {
            let lab = crate::lab::analyze(&coin_world(240, seed));
            let order = learning_order(&lab);
            let mut m = Mind::blank(&lab, &[], &[]);
            let (mut e_raw, mut e_yield, mut e_base, mut n) = (0.0, 0.0, 0.0, 0usize);
            for p in &order {
                let (y, b) = (m.estimate(&m.situation_of_prediction(p, &[])), m.base_rate());
                let r = m.learn_prediction(p, &[], &[]);
                if let (Some(b), false) = (b, y.abstained) {
                    e_raw += (y.raw - r) * (y.raw - r); e_yield += (y.expected - r) * (y.expected - r); e_base += (b - r) * (b - r); n += 1;
                }
            }
            let (e_raw, e_yield, e_base) = (e_raw / n as f64, e_yield / n as f64, e_base / n as f64);
            std::println!("coin world {}: brain {:.4} yield {:.4} base rate {:.4}; trust {:.3}", seed, e_raw, e_yield, e_base, m.trust());
            assert!(e_raw > 1.05 * e_base, "the raw brain chases the coin: {} vs {}", e_raw, e_base);
            assert!(m.trust() < 0.25, "no demonstrated skill, little trust: {}", m.trust());
            assert!(e_yield < e_raw && e_yield < 1.03 * e_base, "shrunk yields are about the base rate: {} vs {}", e_yield, e_base);
            let (y, b) = (m.estimate(&m.situation_of_prediction(order[0], &[])), m.base_rate().unwrap());
            assert!((y.expected - b).abs() <= m.trust() * (y.raw - b).abs() + 1e-12, "{:?}", y);
        }
    }

    #[test]
    fn attributes_predictions_to_knobs_by_arm_label() {
        let knobs = parse_knobs(BUILTIN).unwrap();
        assert_eq!(knob_of("ub1024-faster-than-256-at-100K", &knobs).map(|k| k.name.as_str()), Some("ub"));
        assert_eq!(knob_of("draft3-beats-draft7-code_tps", &knobs).map(|k| k.name.as_str()), Some("draft"));
        assert_eq!(knob_of("gqa2-5pct-at-64K", &knobs), None, "no knob arm named");
        assert_eq!(knob_of("stock-short", &knobs), None, "a plain alias (`stock`) does not attribute");
        assert_eq!(knob_of("chunk248-piggyback-beats-64-sum", &knobs).map(|k| k.name.as_str()), Some("chunk"), "knob name + value does");
    }

    #[test]
    fn a_knob_whose_tests_were_decisive_outranks_one_whose_tests_were_fragile() {
        // Same claim shape (relative, same threshold), different knob histories:
        // ub's past tests were decisive (margin between the 1.95% band and 3 floors = 4.22%),
        // draft's were fragile (margin inside the band).
        let mut rows = Vec::new();
        for i in 0..10 {
            rows.push((if i % 2 == 0 { "ub512-faster" } else { "ub1024-faster" }, ">%", "105 vs 100 (5.00%, need >% 2%)", "pass"));
            rows.push((if i % 2 == 0 { "draft3-faster" } else { "draft5-faster" }, ">%", "101 vs 100 (1.00%, need >% 0.5%)", "pass"));
        }
        let lab = crate::lab::analyze(&ledger(&rows));
        let knobs = parse_knobs(BUILTIN).unwrap();
        let m = Mind::from_lab_knobs(&lab, &knobs);
        let (ub, draft) = (knobs.iter().find(|k| k.name == "ub").unwrap(), knobs.iter().find(|k| k.name == "draft").unwrap());
        assert_eq!((m.history("ub").n, m.history("ub").decisive), (10, 10));
        assert_eq!((m.history("draft").n, m.history("draft").decisive), (10, 0));
        let band = lab.pair_band_pct.max(lab.floor_pct);
        let y_ub = m.estimate_challenger(ub, band, None);
        let y_draft = m.estimate_challenger(draft, band, None);
        std::println!("synthetic: ub challenger yield {:.3}, draft challenger yield {:.3}", y_ub.expected, y_draft.expected);
        assert!(!y_ub.abstained && !y_draft.abstained);
        assert!(y_ub.expected > y_draft.expected + 0.3, "{:?} vs {:?}", y_ub, y_draft);
    }

    /// Falsifier on the real lab: challengers must get different yields. Reads the live ledger and job
    /// logs from STRUKTURA_LAB_DIR (ledger.jsonl, lab-*.out); skipped when absent (lab data is not
    /// committed).
    #[test]
    fn live_ledger_challengers_get_different_yields() {
        let dir = std::env::var("STRUKTURA_LAB_DIR").unwrap_or_else(|_| "/tmp/claude-1000/-home-yin/05eb9a0d-7dac-4311-a190-0501e9f40dbe/scratchpad/lab".into());
        let Ok(ledger) = std::fs::read_to_string(format!("{}/ledger.jsonl", dir)) else { std::println!("live ledger absent: skipped"); return };
        let logs: Vec<(String, String)> = ["115", "120", "130", "160"].iter()
            .filter_map(|n| std::fs::read_to_string(format!("{}/lab-{}.out", dir, n)).ok().map(|t| (format!("lab-{}.out", n), t))).collect();
        let obs = super::super::observe(&ledger, &logs);
        let lessons = super::super::learn::learn(&obs);
        let knobs = parse_knobs(BUILTIN).unwrap();
        let cons = super::super::knobs::constraints(&[], &obs.lab.constraints);
        let ag = super::super::agenda::agenda(&obs, &lessons, &knobs, &cons);
        let m = Mind::from_lab_knobs(&obs.lab, &knobs);
        let mut ys = Vec::new();
        for it in ag.iter().filter(|i| i.kind == super::super::agenda::Kind::Challenger) {
            let k = knobs.iter().find(|k| k.name == it.knob).unwrap();
            let thr = super::super::design::design(it, k, &lessons, obs.lab.cal_cv_pct, 0, None).map(|d| d.threshold_pct).unwrap_or(lessons.band_pct);
            let y = m.estimate_challenger(k, thr, it.observed_effect_pct);
            std::println!("live: {}={} yield {:.3} (evidence {:.1}, knob history n={} decisive={}) cites {:?}",
                it.knob, it.value, y.expected, y.evidence, m.history(&it.knob).n, m.history(&it.knob).decisive, y.cites);
            ys.push(y.expected);
        }
        assert!(ys.len() >= 2, "need at least two challengers on the live agenda");
        let spread = ys.iter().cloned().fold(f64::MIN, f64::max) - ys.iter().cloned().fold(f64::MAX, f64::min);
        std::println!("live: yield spread across {} challengers = {:.3}", ys.len(), spread);
        assert!(spread > 1e-6, "challengers still indistinguishable: {:?}", ys);
    }
}
