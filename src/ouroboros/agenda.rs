//! The ranked agenda: what the lab should measure next, and why.
//!
//! Order of value (score):
//! 1. **Re-measure** a fragile verdict (score 1..2, thinner margin first): a
//!    load-bearing fact that could be noise is worth more than a new idea.
//! 2. **Challengers** of each instrumented knob (current value excluded):
//!    inside the noise band 0.8 (one more power-sized test decides it), never
//!    measured 0.5, measured better once 0.7 (confirm with repeats), measured
//!    worse 0.1, settled by repeated runs or a falsified prediction 0.05.
//! 3. Listed but never run (score 0): **infeasible** values (a constraint
//!    forbids them) and **needs instrument** knobs (no job template measures
//!    the metric they move).

use super::knobs::{violates, Constraint, Knob};
use super::learn::Lessons;
use super::{arm_value, Observation};

#[derive(Clone, Debug, PartialEq)]
pub enum Kind { Remeasure, Challenger, Settled, Infeasible, NeedsInstrument }

impl Kind {
    pub fn as_str(&self) -> &'static str {
        match self { Kind::Remeasure => "remeasure", Kind::Challenger => "challenger", Kind::Settled => "settled",
            Kind::Infeasible => "infeasible", Kind::NeedsInstrument => "needs-instrument" }
    }
}

#[derive(Clone, Debug)]
pub struct Item {
    pub kind: Kind,
    pub knob: String,
    pub value: String,
    /// (pred file, prediction name) for re-measure items.
    pub pred: Option<(String, String)>,
    pub score: f64,
    /// Challenger vs incumbent on the knob's metric, percent, positive = better.
    pub observed_effect_pct: Option<f64>,
    pub samples: usize,
    pub why: String,
}

/// Effect of `value` over the knob's current value on its metric (positive = better), and min samples.
pub fn observed_effect(obs: &Observation, k: &Knob, value: &str) -> Option<(f64, usize)> {
    let (g, a) = k.arm_of(value);
    let (gi, ai) = k.arm_of(&k.current);
    let (x, nx) = arm_value(obs, &g, &a, &k.metric)?;
    let (y, ny) = arm_value(obs, &gi, &ai, &k.metric)?;
    if y == 0.0 { return None; }
    let e = 100.0 * (x - y) / y;
    Some((if k.lower_is_better { -e } else { e }, nx.min(ny)))
}

pub fn agenda(obs: &Observation, lessons: &Lessons, knobs: &[Knob], cons: &[Constraint]) -> Vec<Item> {
    let band = lessons.band_pct;
    let mut items = Vec::new();
    for f in &lessons.fragile {
        items.push(Item { kind: Kind::Remeasure, knob: String::new(), value: String::new(),
            pred: Some((f.pred.clone(), f.name.clone())), score: 2.0 - (f.margin_pct.abs() / band).min(1.0),
            observed_effect_pct: None, samples: 1,
            why: format!("{} by {:+.2}% inside the {:.2}% single-run band: a re-run could flip it", f.verdict, f.margin_pct, band) });
    }
    for k in knobs {
        let con = cons.iter().find(|c| c.knob == k.name);
        for v in k.values.iter().filter(|v| **v != k.current) {
            let eff = observed_effect(obs, k, v);
            let (e, n) = match eff { Some((e, n)) => (Some(e), n), None => (None, 0) };
            let mut it = Item { kind: Kind::Challenger, knob: k.name.clone(), value: v.clone(), pred: None, score: 0.0,
                observed_effect_pct: e, samples: n, why: String::new() };
            if let Some(c) = con.filter(|c| violates(c, v)) {
                it.kind = Kind::Infeasible;
                it.why = format!("violates {} {} {} ({})", c.knob, c.op, c.value, c.source);
            } else if !k.has_instrument() {
                it.kind = Kind::NeedsInstrument;
                it.why = format!("no job template measures {} ({})", k.metric, k.what);
            } else if falsified(obs, k, v) {
                it.kind = Kind::Settled; it.score = 0.05;
                it.why = "a prediction naming it was FALSIFIED; not re-proposed".into();
            } else if let Some((m, n)) = proven(obs, k, v, band) {
                it.kind = Kind::Settled; it.score = 0.05;
                it.why = format!("PROVEN: its prediction against {} passed on {} agreeing measurements, {:+.2}% past threshold (band {:.2}%): a ship candidate, not re-measured", k.current, n, m, band);
            } else {
                match e {
                    None => { it.score = 0.5; it.why = format!("never measured on {}", k.metric); }
                    Some(e) if n >= crate::arms::MIN_BOOT && e.abs() >= band => {
                        it.kind = Kind::Settled; it.score = 0.05;
                        it.why = format!("{:+.1}% over {} runs: already decided", e, n);
                    }
                    Some(e) if e.abs() < band => { it.score = 0.8; it.why = format!("{:+.1}% is inside the {:.2}% band: a power-sized test decides it", e, band); }
                    Some(e) if e > 0.0 => { it.score = 0.7; it.why = format!("{:+.1}% better in {} run(s): confirm with repeats", e, n); }
                    Some(e) => { it.score = 0.1; it.why = format!("{:+.1}% worse in {} run(s): low value", e, n); }
                }
            }
            items.push(it);
        }
    }
    items.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        .then(a.knob.cmp(&b.knob)).then(a.value.cmp(&b.value)).then(a.pred.cmp(&b.pred)));
    items
}

/// R5 (a decided knob gets no challenger): a prediction that compares this arm with the CURRENT one
/// (its name carries both labels, e.g. "draft5-beats-draft7-code_tps") PASSED, never flipped, was
/// replicated (>= 2 independent agreeing measurements) and cleared its threshold by at least the
/// single-run band. Re-measuring a proven winner teaches nothing; promoting it is the gate's job.
/// Returns (margin %, agreeing measurements).
fn proven(obs: &Observation, k: &Knob, v: &str, band: f64) -> Option<(f64, usize)> {
    let (_, label) = k.arm_of(v);
    let (_, incumbent) = k.arm_of(&k.current);
    obs.lab.predictions.iter()
        .filter(|p| p.verdict == "pass" && p.flips == 0 && p.agreeing >= 2)
        .filter(|p| { let t: Vec<&str> = p.name.split(|c: char| !c.is_ascii_alphanumeric()).collect(); t.contains(&label.as_str()) && t.contains(&incumbent.as_str()) })
        .filter_map(|p| p.margin_pct.filter(|m| *m >= band).map(|m| (m, p.agreeing)))
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
}

/// A failed prediction whose name mentions the arm label (e.g. "ub512-...").
fn falsified(obs: &Observation, k: &Knob, v: &str) -> bool {
    let (_, label) = k.arm_of(v);
    obs.lab.predictions.iter().any(|p| p.verdict == "fail" && p.name.split(|c: char| !c.is_ascii_alphanumeric()).any(|t| t == label))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ouroboros::{knobs::{constraints, parse_knobs, BUILTIN}, learn::learn, observe};

    fn setup(extra_ledger: &str) -> (Observation, Vec<Knob>) {
        let mut l = String::from("{\"kind\":\"cal\",\"ok\":true,\"code\":{\"tps\":200.0}}\n{\"kind\":\"cal\",\"ok\":true,\"code\":{\"tps\":202.0}}\n");
        l += extra_ledger;
        let log = "ub128\td100000=1390\nub256\td100000=1397\nub512\td100000=1484\n".to_string();
        (observe(&l, &[("lab-115.out".into(), log)]), parse_knobs(BUILTIN).unwrap())
    }

    #[test]
    fn fragile_remeasure_outranks_every_challenger() {
        let (o, k) = setup("{\"kind\":\"prediction\",\"ts\":1,\"pred\":\"130.tsv\",\"name\":\"gqa2-5pct-at-64K\",\"value\":\"1658 vs 1568 (5.74%, need >% 5%)\",\"op\":\">%\",\"threshold\":\"arm B\",\"verdict\":\"pass\"}\n");
        let a = agenda(&o, &learn(&o), &k, &constraints(&[], &[]));
        assert_eq!(a[0].kind, Kind::Remeasure);
        assert_eq!(a[0].pred, Some(("130.tsv".into(), "gqa2-5pct-at-64K".into())));
        assert!(a[0].score > a.iter().filter(|i| i.kind == Kind::Challenger).map(|i| i.score).fold(0.0, f64::max));
    }

    #[test]
    fn constraints_and_missing_instruments_are_listed_not_run() {
        let (o, k) = setup("");
        let a = agenda(&o, &learn(&o), &k, &constraints(&[], &[]));
        let ub512 = a.iter().find(|i| i.knob == "ub" && i.value == "512").unwrap();
        assert_eq!(ub512.kind, Kind::Infeasible);
        assert!((ub512.observed_effect_pct.unwrap() - 6.23).abs() < 0.01, "evidence still reported");
        let ub128 = a.iter().find(|i| i.knob == "ub" && i.value == "128").unwrap();
        assert_eq!(ub128.kind, Kind::Challenger);
        assert!((ub128.score - 0.8).abs() < 1e-12, "-0.5% is inside the band: {}", ub128.why);
        assert!(a.iter().filter(|i| i.knob == "chunk").all(|i| i.kind == Kind::NeedsInstrument));
        assert!(a.iter().all(|i| !(i.knob == "ub" && i.value == "256")), "incumbent never a challenger");
    }

    #[test]
    fn falsified_value_is_settled() {
        let (o, k) = setup("{\"kind\":\"prediction\",\"ts\":1,\"pred\":\"9.tsv\",\"name\":\"draft9-faster\",\"value\":\"90 vs 100 (-10.00%, need >% 2%)\",\"op\":\">%\",\"threshold\":\"arm B\",\"verdict\":\"fail\"}\n");
        let a = agenda(&o, &learn(&o), &k, &constraints(&[], &[]));
        assert_eq!(a.iter().find(|i| i.knob == "draft" && i.value == "9").unwrap().kind, Kind::Settled);
    }

    #[test]
    fn r5_proven_winner_is_settled_but_one_pass_is_not() {
        let p = |ts: u32, v: &str| format!("{{\"kind\":\"prediction\",\"ts\":{ts},\"pred\":\"210.tsv\",\"name\":\"draft5-beats-draft7-code_tps\",\"value\":\"{v}\",\"op\":\">%\",\"threshold\":\"arm B\",\"verdict\":\"pass\"}}\n");
        let once = p(1, "212.0 vs 200.0 (6.00%, need >% 3.5%)");
        let (o, k) = setup(&once);
        let a = agenda(&o, &learn(&o), &k, &constraints(&[], &[]));
        assert_eq!(a.iter().find(|i| i.knob == "draft" && i.value == "5").unwrap().kind, Kind::Challenger, "one pass: confirm, not settled");
        let twice = once + &p(2, "211.4 vs 199.6 (5.91%, need >% 3.5%)");
        let (o, k) = setup(&twice);
        let a = agenda(&o, &learn(&o), &k, &constraints(&[], &[]));
        let it = a.iter().find(|i| i.knob == "draft" && i.value == "5").unwrap();
        assert_eq!(it.kind, Kind::Settled, "replicated decisive pass: {}", it.why);
        assert!(it.why.starts_with("PROVEN"), "{}", it.why);
        // the other draft values are untouched by draft5's proof
        assert_eq!(a.iter().find(|i| i.knob == "draft" && i.value == "3").unwrap().kind, Kind::Challenger);
    }
}
