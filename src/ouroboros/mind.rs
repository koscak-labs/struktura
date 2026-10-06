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
//! single-run band) or void (instrument bug: the run taught nothing).
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
    /// Knob history after the whole ledger, by knob name.
    history: Vec<(String, History)>,
}

#[derive(Clone, Debug)]
pub struct Yield {
    /// Expected information yield in [0, 1] (1 = decisive).
    pub expected: f64,
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
        let mut brain: Box<Brain<256, DG, 1>> = Box::new(Brain::new(0.0, 1.0));
        let mut grower: crate::brain_grow::Grower<D, G> = crate::brain_grow::Grower::new(0x6D696E64);
        grower.constant[D - 1] = true;
        grower.every = 12;
        let mut grown = Vec::new();
        let mut goal_predictions = 0usize;
        let mut labels = vec![String::new(); 256];
        let band = lab.pair_band_pct.max(lab.floor_pct);
        let mut preds: Vec<&crate::lab::Prediction> = lab.predictions.iter()
            .filter(|p| matches!(p.verdict.as_str(), "pass" | "fail" | "void")).collect();
        preds.sort_by(|a, b| a.ts.partial_cmp(&b.ts).unwrap_or(std::cmp::Ordering::Equal).then(a.pred.cmp(&b.pred)).then(a.name.cmp(&b.name)));
        let mut history: Vec<(String, History)> = knobs.iter().map(|k| (k.name.clone(), History::default())).collect();
        let mut episodes = 0;
        for p in preds {
            let reward = match (p.verdict.as_str(), p.margin_pct) {
                ("void", _) => 0.0,
                (_, Some(m)) if m.abs() < band => 0.0,
                ("pass", Some(m)) if m > 3.0 * lab.floor_pct => 0.5,
                _ => 1.0,
            };
            let is_goal = goal.iter().any(|g| g.pred == p.pred && (g.required.is_empty() || g.required.iter().any(|r| r == &p.name)));
            if is_goal { goal_predictions += 1; }
            let reward = if goal.is_empty() || is_goal { reward } else { reward * 0.6 };
            let knob = knob_of(&p.name, knobs).map(|k| k.name.clone());
            let h = knob.as_ref().and_then(|k| history.iter().find(|(n, _)| n == k)).map(|(_, h)| *h).unwrap_or_default();
            let base = situation_of(&p.op, &p.value, &p.name, band, &h);
            let x: [f32; DG] = grower.situation(&base);
            let class = if reward == 0.0 { "fragile/void" } else if (p.verdict == "pass" && p.margin_pct.map(|m| m > 3.0 * lab.floor_pct).unwrap_or(false)) { "easy" } else { "decisive" };
            if let Some(s) = brain.learn(&x, 0, reward) { labels[s] = format!("{}::{} ({}{})", p.pred, p.name, class, if is_goal { ", proves a feature" } else { "" }); }
            episodes += 1;
            if let Some(g) = grower.after_learn(&mut brain) { if let Some(f) = g.adopted { grown.push((episodes, describe(&f), g.gain)); } }
            // Update the knob's history only after its situation was taken (no leakage).
            if let Some(k) = knob {
                if let Some((_, h)) = history.iter_mut().find(|(n, _)| *n == k) {
                    h.n += 1;
                    if reward >= 1.0 { h.decisive += 1; }
                    if let Some((d, _, _)) = super::learn::parse_relative(&p.value) {
                        if band > 0.0 { h.effect_bands_sum += d.abs() / band; h.effect_n += 1; }
                    }
                }
            }
        }
        Mind { brain, grower, grown, goal_aware: !goal.is_empty(), goal_predictions, labels, episodes, band_pct: band, history }
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

    pub fn estimate(&self, x: &Situation) -> Yield {
        let xx: [f32; DG] = self.grower.situation(x);
        let d = self.brain.decide(&xx, &[true], 0);
        let cites = (0..d.n_cited as usize).take(3).filter_map(|k| {
            let s = d.cited[k] as usize;
            self.brain.episode(s).map(|_| self.labels[s].clone())
        }).collect();
        Yield { expected: (d.expected as f64).clamp(0.0, 1.0), evidence: d.evidence as f64, abstained: d.abstained, cites }
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
            let thr = super::super::design::design(it, k, &lessons, obs.lab.cal_cv_pct, 0).map(|d| d.threshold_pct).unwrap_or(lessons.band_pct);
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
