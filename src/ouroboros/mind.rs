//! Mind: the loop's native brain. It replays every scored prediction in the
//! ledger into a [`crate::brain::Brain`] that learns which experiments teach
//! the lab something, then estimates the information yield of each candidate
//! experiment and cites the past predictions its estimate rests on.
//!
//! Situation (6 features), known before an experiment runs:
//! relative claim (A vs B), how ambitious the threshold is in noise bands,
//! equivalence test, speed metric, quality metric, constant.
//! Reward, known after it was scored: 1 = decisive verdict; 0.5 = an easy pass
//! (margin > 3 noise floors: it was a safe bet); 0 = fragile (margin inside the
//! single-run band) or void (instrument bug: the run taught nothing).
//!
//! The mind only re-orders challengers among themselves; fragile re-measures,
//! constraints and missing instruments are decided by the agenda's rules.

#![cfg(feature = "std")]

use crate::brain::Brain;
use crate::lab::LabReport;

pub const D: usize = 6;
pub type Situation = [f32; D];

fn has(name: &str, words: &[&str]) -> bool {
    let n = name.to_ascii_lowercase();
    words.iter().any(|w| n.contains(w))
}

const SPEED: &[&str] = &["tps", "ms", "fast", "slow", "pp", "prefill", "decode", "ttd", "speed", "_s", "latency", "throughput"];
const QUALITY: &[&str] = &["pass", "found", "needle", "correct", "acc", "identical", "quality", "solved", "ppl"];

/// Situation of a scored prediction (from lab-score's op and value text) as it looked before the run.
pub fn situation_of(op: &str, value: &str, name: &str, band_pct: f64) -> Situation {
    let relative = op.ends_with('%');
    let thr = if relative { crate::ouroboros::learn::parse_relative(value).map(|(_, _, p)| p).unwrap_or(0.0) } else { 0.0 };
    let ambition = if relative && band_pct > 0.0 { (thr / band_pct).min(10.0) / 10.0 } else { 0.0 };
    [relative as u8 as f32, ambition as f32, (op == "~%") as u8 as f32,
     has(name, SPEED) as u8 as f32, has(name, QUALITY) as u8 as f32, 1.0]
}

/// Situation of a planned challenger: a relative claim on `metric` with `threshold_pct`.
pub fn situation_planned(metric: &str, threshold_pct: f64, band_pct: f64) -> Situation {
    let ambition = if band_pct > 0.0 { (threshold_pct / band_pct).min(10.0) / 10.0 } else { 0.0 };
    let quality = has(metric, QUALITY);
    [1.0, ambition as f32, 0.0, (!quality) as u8 as f32, quality as u8 as f32, 1.0]
}

pub struct Mind {
    brain: Box<Brain<256, D, 1>>,
    /// "pred::name" per memory slot, for citations.
    labels: Vec<String>,
    pub episodes: usize,
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
    /// Learn from every scored prediction in the ledger (latest verdict each, in time order).
    pub fn from_lab(lab: &LabReport) -> Self {
        let mut brain: Box<Brain<256, D, 1>> = Box::new(Brain::new(0.0, 1.0));
        let mut labels = vec![String::new(); 256];
        let band = lab.pair_band_pct.max(lab.floor_pct);
        let mut preds: Vec<&crate::lab::Prediction> = lab.predictions.iter()
            .filter(|p| matches!(p.verdict.as_str(), "pass" | "fail" | "void")).collect();
        preds.sort_by(|a, b| a.ts.partial_cmp(&b.ts).unwrap_or(std::cmp::Ordering::Equal).then(a.pred.cmp(&b.pred)).then(a.name.cmp(&b.name)));
        let mut episodes = 0;
        for p in preds {
            let reward = match (p.verdict.as_str(), p.margin_pct) {
                ("void", _) => 0.0,
                (_, Some(m)) if m.abs() < band => 0.0,
                ("pass", Some(m)) if m > 3.0 * lab.floor_pct => 0.5,
                _ => 1.0,
            };
            let x = situation_of(&p.op, &p.value, &p.name, band);
            let before: Vec<bool> = (0..256).map(|s| brain.episode(s).is_some()).collect();
            brain.learn(&x, 0, reward);
            // Find the slot that changed (new or replaced) to label it.
            let slot = (0..256).find(|&s| brain.episode(s).map(|e| e.key == x && e.reward == reward).unwrap_or(false) && !before[s])
                .or_else(|| (0..256).rev().find(|&s| brain.episode(s).map(|e| e.key == x && e.reward == reward).unwrap_or(false)));
            if let Some(s) = slot { labels[s] = format!("{}::{}", p.pred, p.name); }
            episodes += 1;
        }
        Mind { brain, labels, episodes }
    }

    pub fn estimate(&self, x: &Situation) -> Yield {
        let d = self.brain.decide(x, &[true], 0);
        let cites = (0..d.n_cited as usize).take(3).filter_map(|k| {
            let s = d.cited[k] as usize;
            self.brain.episode(s).map(|e| format!("{} ({})", self.labels[s], match e.reward { r if r >= 1.0 => "decisive", r if r >= 0.5 => "easy", _ => "fragile/void" }))
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
    fn abstains_on_an_empty_ledger() {
        let lab = crate::lab::analyze(&ledger(&[]));
        let m = Mind::from_lab(&lab);
        assert!(m.estimate(&situation_planned("code_tps", 1.6, 1.6)).abstained);
    }
}
