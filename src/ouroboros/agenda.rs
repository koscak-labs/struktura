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
    // a fragile verdict scored while a different config was live measured something no longer running:
    // re-running its job now answers a different question, so it is stale, not a re-measure.
    // Compared by config, not by time: after a rollback the older verdicts are live again.
    let live = live_config(obs, f64::INFINITY);
    for f in &lessons.fragile {
        let mut it = Item { kind: Kind::Remeasure, knob: String::new(), value: String::new(),
            pred: Some((f.pred.clone(), f.name.clone())), score: 2.0 - (f.margin_pct.abs() / band).min(1.0),
            observed_effect_pct: None, samples: 1,
            why: format!("{} by {:+.2}% inside the {:.2}% single-run band: a re-run could flip it", f.verdict, f.margin_pct, band) };
        if let Some((bin, r)) = rolled_back_gate(obs, &f.pred, f.ts) {
            it.kind = Kind::Settled; it.score = 0.05;
            it.why = format!("QUARANTINED: gate for {}, which was rolled back at {:.0}; its verdicts are not shippable until a new gate passes", bin, r);
        } else if let (Some(now), Some(then)) = (live, live_config(obs, f.ts)) {
            if !same_config(now, then) {
                it.kind = Kind::Settled; it.score = 0.05;
                it.why = format!("STALE: {} by {:+.2}% was measured on {} (deployed {:.0}), not the live {}; a re-run answers an old question",
                    f.verdict, f.margin_pct, then.1, then.0, now.1);
            }
        }
        items.push(it);
    }
    let mut grown_from: Vec<(String, String, String)> = Vec::new();
    for k in knobs {
        let con = cons.iter().find(|c| c.knob == k.name);
        let grown = grow(obs, k, cons, band);
        let values: Vec<String> = k.values.iter().filter(|v| **v != k.current).cloned()
            .chain(grown.iter().map(|(v, _)| v.clone())).collect();
        for (v, why) in &grown { grown_from.push((k.name.clone(), v.clone(), why.clone())); }
        for v in values.iter() {
            let eff = observed_effect(obs, k, v);
            let (e, n) = match eff { Some((e, n)) => (Some(e), n), None => (None, 0) };
            let mut it = Item { kind: Kind::Challenger, knob: k.name.clone(), value: v.clone(), pred: None, score: 0.0,
                observed_effect_pct: e, samples: n, why: String::new() };
            if let Some(c) = con.filter(|c| violates(c, v)) {
                it.kind = Kind::Infeasible;
                it.why = format!("violates {} {} {} ({})", c.knob, c.op, c.value, c.source);
            } else if let Some(n) = unmeasurable(obs, k, v) {
                it.kind = Kind::Infeasible;
                it.why = format!("INVALID VALUE: {} designed job(s) against {} produced no measurement (missing/void); the server or instrument cannot run it", n, k.current);
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
    // R5 for a whole knob: once one value is PROVEN better than the current one, the knob is decided (ship that
    // value first). Other values of it wait: measured against the old current they would answer a stale question;
    // after the ship they re-open against the new current.
    let decided: Vec<(String, String)> = items.iter().filter(|i| i.kind == Kind::Settled && i.why.starts_with("PROVEN"))
        .map(|i| (i.knob.clone(), i.value.clone())).collect();
    for it in items.iter_mut().filter(|i| i.kind == Kind::Challenger) {
        if let Some((_, v)) = decided.iter().find(|(k, _)| *k == it.knob) {
            it.kind = Kind::Settled; it.score = 0.05;
            it.why = format!("knob decided: {}={} is a PROVEN ship candidate; this value re-opens against the new current after it ships", it.knob, v);
        }
    }
    for it in items.iter_mut() {
        if let Some((_, _, g)) = grown_from.iter().find(|(k, v, _)| *k == it.knob && *v == it.value) { it.why = format!("{}; {}", it.why, g); }
    }
    items.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        .then(a.knob.cmp(&b.knob)).then(a.value.cmp(&b.value)).then(a.pred.cmp(&b.pred)));
    items
}

type Config = (f64, String, String, Option<String>);

/// The deploy row live at `ts` (the latest one at or before it); None before the first deploy.
fn live_config(obs: &Observation, ts: f64) -> Option<&Config> {
    obs.lab.configs.iter().filter(|c| c.0 <= ts).max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
}

/// Same binary and env; flags compared only when both rows record them (older deploy rows do not).
fn same_config(a: &Config, b: &Config) -> bool {
    a.1 == b.1 && a.2 == b.2 && match (&a.3, &b.3) { (Some(x), Some(y)) => x == y, _ => true }
}

/// The gate `pred` tested a candidate binary that was later deployed and rolled back (after `ts`):
/// (binary, rollback ts). Production refuted the gate, so a re-run of it measures a known-bad candidate.
fn rolled_back_gate<'a>(obs: &'a Observation, pred: &str, ts: f64) -> Option<(&'a str, f64)> {
    for &r in &obs.lab.rollbacks {
        if r <= ts { continue; }
        let Some(bad) = live_config(obs, r - 1e-6) else { continue };
        if obs.lab.gates.iter().any(|(g, b)| g == pred && *b == bad.1) { return Some((bad.1.as_str(), r)); }
    }
    None
}

/// Tokens of a prediction name ("draft5-beats-draft7-code_tps" -> draft5 beats draft7 code tps).
fn tokens(name: &str) -> Vec<&str> { name.split(|c: char| !c.is_ascii_alphanumeric()).collect() }

/// A prediction about `label` measured against `incumbent`: both arm labels appear, challenger first
/// (design names them `<knob><challenger>-beats-<knob><incumbent>-<metric>`). Order matters once the
/// winner ships: "draft5-beats-draft7" must not read as a result about draft7 against current draft5.
fn names_pair(name: &str, label: &str, incumbent: &str) -> bool {
    let t = tokens(name);
    match (t.iter().position(|x| *x == label), t.iter().position(|x| *x == incumbent)) { (Some(a), Some(b)) => a < b, _ => false }
}

/// Integer value of an arm token of this knob ("draft5" -> 5); prefix-matched knobs only.
fn int_of(k: &Knob, tok: &str) -> Option<i64> {
    tok.strip_prefix(k.name.as_str())?.parse::<i64>().ok().filter(|_| k.matcher == format!("prefix:{}", k.name))
}

/// ADAPTIVE CATALOG (zero LLM, deterministic). The hand-written values of a knob are a seed; every PROVEN
/// win grows its untested integer neighbours in the winning direction:
///
/// - win = a prediction `<knob>W-beats-<knob>L` that passed, never flipped, agreed >= 2 times and cleared its
///   threshold by >= one band (the same bar as R5's `proven`). L is the value it beat (the then-current one).
/// - **midpoint** between W and L (rounded toward W), when they are >= 2 apart;
/// - **beyond**: W + (W - L), one gap further in the winning direction, clamped into `range=`. If that value is
///   already listed, infeasible or blocked, bisect back toward W (W+d*gap/2, gap/4, ...) and take the first
///   value that is none of those (draft 5 beat 7: beyond is 3; with 3 already listed it grows 4).
///
/// A grown value is never a duplicate (listed, current, or grown this turn), always an integer, inside
/// `range=` and every constraint on the knob, and never further from current than a value that was FALSIFIED
/// or INVALID on the same side, nor that value itself ("draft9 INVALID" closes 9+). At most `grow=` (<= 2) values per knob per turn,
/// strongest win first. No proven win: nothing grows and the agenda is exactly the catalog's.
/// Returns (value, "grown from W ...").
fn grow(obs: &Observation, k: &Knob, cons: &[Constraint], band: f64) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    let Ok(cur) = k.current.parse::<i64>() else { return out };
    if k.grow == 0 || !k.has_instrument() || int_of(k, &format!("{}{}", k.name, cur)).is_none() { return out; }
    let mut wins: Vec<(f64, i64, i64)> = Vec::new();
    let mut seen: Vec<i64> = k.values.iter().filter_map(|v| v.parse().ok()).collect();
    for p in &obs.lab.predictions {
        let t = tokens(&p.name);
        seen.extend(t.iter().filter_map(|x| int_of(k, x)));
        if !(p.verdict == "pass" && p.flips == 0 && p.agreeing >= 2) { continue; }
        let Some(m) = p.margin_pct.filter(|m| *m >= band) else { continue };
        let Some(i) = t.iter().position(|x| *x == "beats") else { continue };
        if i == 0 || i + 1 >= t.len() { continue; }
        if let (Some(w), Some(l)) = (int_of(k, t[i - 1]), int_of(k, t[i + 1])) {
            if w == l { continue; }
            match wins.iter_mut().find(|x| x.1 == w && x.2 == l) { Some(x) => x.0 = x.0.max(m), None => wins.push((m, w, l)) }
        }
    }
    wins.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    seen.sort(); seen.dedup();
    let feasible = |g: i64| !cons.iter().any(|c| c.knob == k.name && violates(c, &g.to_string()));
    // FALSIFIED / INVALID values close everything further out on their side of current
    let walls: Vec<i64> = seen.iter().copied().filter(|x| *x != cur)
        .filter(|x| { let s = x.to_string(); falsified(obs, k, &s) || unmeasurable(obs, k, &s).is_some() }).collect();
    let blocked = |g: i64| walls.iter().any(|x| (g - cur).signum() == (x - cur).signum() && (g - cur).abs() >= (x - cur).abs());
    let listed = |g: i64, out: &[(String, String)]| { let s = g.to_string(); g == cur || k.values.contains(&s) || out.iter().any(|(v, _)| *v == s) };
    for (_, w, l) in wins {
        if out.len() >= k.grow { break; }
        if !feasible(w) { continue; }
        let (d, gap) = ((w - l).signum(), (w - l).abs());
        let ok = |g: i64, out: &[(String, String)]| g >= k.range.0 && g <= k.range.1 && !listed(g, out) && feasible(g) && !blocked(g);
        let why = |g: i64, how: &str| (g.to_string(), format!("grown from {} ({} of the PROVEN win {}{} over {}{})", w, how, k.name, w, k.name, l));
        let mid = w - d * (gap / 2);
        if gap >= 2 && ok(mid, &out) && out.len() < k.grow { let x = why(mid, "midpoint"); out.push(x); }
        let beyond = (w.saturating_add(d.saturating_mul(gap))).clamp(k.range.0, k.range.1);
        let mut dist = (beyond - w).abs();
        while dist >= 1 && out.len() < k.grow {
            let g = w + d * dist;
            if ok(g, &out) { let x = why(g, "one step beyond"); out.push(x); break; }
            dist /= 2;
        }
    }
    out
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
        .filter(|p| names_pair(&p.name, &label, &incumbent))
        .filter_map(|p| p.margin_pct.filter(|m| *m >= band).map(|m| (m, p.agreeing)))
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
}

/// INVALID VALUE: the prediction comparing this arm with the CURRENT one came back with no measurement
/// (verdict missing/void) from at least 2 separate jobs and was never scored pass/fail. A value the
/// server cannot even load (draft9: "failed to create MTP context") must not be re-designed as
/// "never measured" forever. Returns how many jobs produced nothing.
fn unmeasurable(obs: &Observation, k: &Knob, v: &str) -> Option<usize> {
    let (_, label) = k.arm_of(v);
    let (_, incumbent) = k.arm_of(&k.current);
    let names_it = |name: &str| { let t: Vec<&str> = name.split(|c: char| !c.is_ascii_alphanumeric()).collect(); t.contains(&label.as_str()) && t.contains(&incumbent.as_str()) };
    let mine: Vec<_> = obs.lab.predictions.iter().filter(|p| names_it(&p.name)).collect();
    if mine.iter().any(|p| p.verdict == "pass" || p.verdict == "fail") { return None; }
    let mut jobs: Vec<&str> = mine.iter().filter(|p| p.verdict == "missing" || p.verdict == "void").map(|p| p.pred.as_str()).collect();
    jobs.sort(); jobs.dedup();
    (jobs.len() >= 2).then_some(jobs.len())
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
    fn fragile_verdicts_are_stale_only_when_measured_on_another_config() {
        let frag = |ts: u32| format!("{{\"kind\":\"prediction\",\"ts\":{},\"pred\":\"130.tsv\",\"name\":\"gqa2-5pct-at-64K\",\"value\":\"1658 vs 1568 (5.74%, need >% 5%)\",\"op\":\">%\",\"threshold\":\"arm B\",\"verdict\":\"pass\"}}\n", ts);
        let dep = |ts: u32, bin: &str, flags: Option<&str>| format!("{{\"kind\":\"deploy\",\"ts\":{},\"binary\":\"{}\",\"env\":\"X=1\"{}}}\n", ts, bin,
            flags.map(|f| format!(",\"flags\":\"{}\"", f)).unwrap_or_default());
        let top = |l: String| { let (o, k) = setup(&l); agenda(&o, &learn(&o), &k, &constraints(&[], &[])) };
        // measured on old, then new is deployed: stale
        let a = top(dep(0, "old", None) + &frag(1) + &dep(50, "new", Some("c=393216")));
        let it = a.iter().find(|i| i.pred.is_some()).unwrap();
        assert_eq!(it.kind, Kind::Settled, "{}", it.why);
        assert!(it.why.starts_with("STALE") && it.why.contains("measured on old") && it.score < 0.1, "{}", it.why);
        assert!(a[0].kind != Kind::Remeasure, "a stale re-measure no longer leads the agenda");
        // ... then rolled back to old: the verdict is about the live config again
        let a = top(dep(0, "old", None) + &frag(1) + &dep(50, "new", Some("c=393216")) + &dep(70, "old", Some("c=262144")));
        assert_eq!(a[0].kind, Kind::Remeasure, "rollback revives it: {}", a[0].why);
        // measured on new, then rolled back: stale
        let a = top(dep(0, "old", None) + &dep(50, "new", Some("c=393216")) + &frag(60) + &dep(70, "old", Some("c=262144")));
        assert!(a.iter().find(|i| i.pred.is_some()).unwrap().why.contains("measured on new"));
        // same binary+env, flags changed (both recorded): stale
        let a = top(dep(0, "old", Some("ub=256")) + &frag(1) + &dep(50, "old", Some("ub=512")));
        assert_eq!(a.iter().find(|i| i.pred.is_some()).unwrap().kind, Kind::Settled);
        // no deploy before the verdict: config unknown, never called stale
        let a = top(frag(1) + &dep(50, "new", None));
        assert_eq!(a[0].kind, Kind::Remeasure);
    }

    #[test]
    fn a_gate_for_a_rolled_back_candidate_is_quarantined() {
        let frag = format!("{{\"kind\":\"prediction\",\"ts\":10,\"pred\":\"452.tsv\",\"name\":\"cand-decode-pooled\",\"value\":\"1658 vs 1568 (5.74%, need >% 5%)\",\"op\":\">%\",\"threshold\":\"arm B\",\"verdict\":\"pass\"}}\n");
        let base = "{\"kind\":\"deploy\",\"ts\":0,\"binary\":\"llama.cpp-old\",\"env\":\"X=1\"}\n".to_string();
        let link = "{\"kind\":\"daemon\",\"ts\":11,\"phase\":\"ship\",\"outcome\":\"gate-link\",\"detail\":\"gate=452.tsv | manual deploy of llama.cpp-new\"}\n";
        let new = "{\"kind\":\"deploy\",\"ts\":50,\"binary\":\"llama.cpp-new\",\"env\":\"X=1\"}\n";
        let back = "{\"kind\":\"deploy\",\"ts\":70,\"binary\":\"llama.cpp-old\",\"env\":\"X=1\",\"rollback\":true}\n";
        let top = |l: String| { let (o, k) = setup(&l); agenda(&o, &learn(&o), &k, &constraints(&[], &[])) };
        let gate_item = |a: &[Item]| a.iter().find(|i| i.pred.is_some()).unwrap().clone();
        // gate linked to llama.cpp-new, deployed, rolled back: quarantined even though old is live again
        let it = gate_item(&top(base.clone() + &frag + link + new + back));
        assert_eq!(it.kind, Kind::Settled);
        assert!(it.why.starts_with("QUARANTINED: gate for llama.cpp-new"), "{}", it.why);
        // no rollback yet: an ordinary re-measure
        assert_eq!(gate_item(&top(base.clone() + &frag + link + new)).kind, Kind::Settled, "stale on the new config");
        assert_eq!(gate_item(&top(base.clone() + &frag + link)).kind, Kind::Remeasure);
        // an unlinked gate is not quarantined by a rollback (falls back to the config rule: old is live)
        assert_eq!(gate_item(&top(base + &frag + new + back)).kind, Kind::Remeasure);
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
        // the knob is decided: other draft values wait for draft5 to ship, then re-open against the new current
        let other = a.iter().find(|i| i.knob == "draft" && i.value == "3").unwrap();
        assert_eq!(other.kind, Kind::Settled, "{}", other.why);
        assert!(other.why.starts_with("knob decided: draft=5 is a PROVEN ship candidate"), "{}", other.why);
        // other knobs are untouched
        assert_eq!(a.iter().find(|i| i.knob == "ub" && i.value == "128").unwrap().kind, Kind::Challenger);
    }

    #[test]
    fn a_value_that_never_produces_a_measurement_is_invalid_not_retried() {
        let m = |ts: u32, pf: &str| format!("{{\"kind\":\"prediction\",\"ts\":{ts},\"pred\":\"{pf}\",\"name\":\"draft9-beats-draft7-code_tps\",\"value\":\"-\",\"verdict\":\"missing\"}}\n");
        let once = m(1, "lab-346-draft9.tsv");
        let (o, k) = setup(&once);
        let a = agenda(&o, &learn(&o), &k, &constraints(&[], &[]));
        assert_eq!(a.iter().find(|i| i.knob == "draft" && i.value == "9").unwrap().kind, Kind::Challenger, "one empty job may be bad luck");
        let twice = once + &m(2, "lab-347-draft9.tsv") + &m(3, "lab-348-draft9.tsv");
        let (o, k) = setup(&twice);
        let a = agenda(&o, &learn(&o), &k, &constraints(&[], &[]));
        let it = a.iter().find(|i| i.knob == "draft" && i.value == "9").unwrap();
        assert_eq!(it.kind, Kind::Infeasible, "{}", it.why);
        assert!(it.why.starts_with("INVALID VALUE: 3 designed job(s)"), "{}", it.why);
        // a value that was ever scored stays a normal candidate even if some jobs came back empty
        let scored = twice + "{\"kind\":\"prediction\",\"ts\":4,\"pred\":\"lab-349-draft9.tsv\",\"name\":\"draft9-beats-draft7-code_tps\",\"value\":\"190 vs 200 (-5.00%, need >% 2%)\",\"op\":\">%\",\"threshold\":\"arm B\",\"verdict\":\"fail\"}\n";
        let (o, k) = setup(&scored);
        let a = agenda(&o, &learn(&o), &k, &constraints(&[], &[]));
        assert_ne!(a.iter().find(|i| i.knob == "draft" && i.value == "9").unwrap().kind, Kind::Infeasible);
    }

    // ---- adaptive catalog ----
    fn win(name: &str) -> String {
        let p = |ts: u32, v: &str| format!("{{\"kind\":\"prediction\",\"ts\":{ts},\"pred\":\"g.tsv\",\"name\":\"{name}\",\"value\":\"{v}\",\"op\":\">%\",\"threshold\":\"arm B\",\"verdict\":\"pass\"}}\n");
        p(10, "212.0 vs 200.0 (6.00%, need >% 3.5%)") + &p(11, "211.4 vs 199.6 (5.91%, need >% 3.5%)")
    }
    fn invalid(name: &str) -> String {
        (20..23).map(|ts| format!("{{\"kind\":\"prediction\",\"ts\":{ts},\"pred\":\"m{ts}.tsv\",\"name\":\"{name}\",\"value\":\"-\",\"verdict\":\"missing\"}}\n")).collect()
    }
    fn run(knobs: &str, ledger: &str, cons: &[Constraint]) -> Vec<Item> {
        let (o, _) = setup(ledger);
        agenda(&o, &learn(&o), &parse_knobs(knobs).unwrap(), cons)
    }
    fn grown(a: &[Item], knob: &str) -> Vec<String> {
        let mut v: Vec<String> = a.iter().filter(|i| i.knob == knob && i.why.contains("grown from")).map(|i| i.value.clone()).collect();
        v.sort_by_key(|x| x.parse::<i64>().unwrap()); v
    }
    const DRAFT: &str = "template=draft metric=code_tps match=prefix:draft";

    #[test]
    fn proven_win_grows_midpoint_and_one_step_beyond() {
        // sparse catalog: draft5 PROVEN over current 7 -> midpoint 6, beyond 5-2=3
        let a = run(&format!("knob draft 5 7 current=7 {DRAFT}"), &win("draft5-beats-draft7-code_tps"), &[]);
        assert_eq!(grown(&a, "draft"), ["3", "6"]);
        let six = a.iter().find(|i| i.knob == "draft" && i.value == "6").unwrap();
        assert!(six.why.contains("grown from 5 (midpoint of the PROVEN win draft5 over draft7)"), "{}", six.why);
        // R5 still holds: before draft5 ships, grown values wait like every other draft value
        assert_eq!(six.kind, Kind::Settled);
        assert!(six.why.starts_with("knob decided: draft=5"), "{}", six.why);
        // 3 already listed (untested): beyond bisects back toward the winner -> 4
        let a = run(&format!("knob draft 3 5 7 current=7 {DRAFT}"), &win("draft5-beats-draft7-code_tps"), &[]);
        assert_eq!(grown(&a, "draft"), ["4", "6"]);
        // after draft5 ships (current=5) the same win keeps growing: grown values are live challengers vs 5
        let a = run(&format!("knob draft 5 7 current=5 {DRAFT}"), &win("draft5-beats-draft7-code_tps"), &[]);
        assert_eq!(grown(&a, "draft"), ["3", "6"]);
        assert!(a.iter().filter(|i| i.knob == "draft").all(|i| i.kind == Kind::Challenger), "draft7 is not 'proven' by draft5-beats-draft7");
        // grow=0 freezes the catalog; grow is capped at 2
        assert!(grown(&run(&format!("knob draft 5 7 current=7 grow=0 {DRAFT}"), &win("draft5-beats-draft7-code_tps"), &[]), "draft").is_empty());
        assert_eq!(parse_knobs(&format!("knob draft 5 7 current=7 grow=9 {DRAFT}")).unwrap()[0].grow, 2);
    }

    #[test]
    fn invalid_or_falsified_value_closes_its_side() {
        // draft8 PROVEN over current 6 would grow 7 and 10; draft9 INVALID closes 10+
        let k = format!("knob draft 6 8 9 current=6 {DRAFT}");
        let a = run(&k, &(win("draft8-beats-draft6-code_tps") + &invalid("draft9-beats-draft6-code_tps")), &[]);
        assert_eq!(grown(&a, "draft"), ["7"]);
        assert!(a.iter().all(|i| !(i.knob == "draft" && i.value.parse::<i64>().unwrap() >= 10)));
        // same with a FALSIFIED draft9
        let fail = "{\"kind\":\"prediction\",\"ts\":30,\"pred\":\"f.tsv\",\"name\":\"draft9-beats-draft6-code_tps\",\"value\":\"190 vs 200 (-5.00%, need >% 2%)\",\"op\":\">%\",\"threshold\":\"arm B\",\"verdict\":\"fail\"}\n";
        let a = run(&k, &(win("draft8-beats-draft6-code_tps") + fail), &[]);
        assert_eq!(grown(&a, "draft"), ["7"]);
        // a FALSIFIED value is itself a wall: draft5 over 7 with draft3 falsified grows 6 and 4, never 3
        let f3 = fail.replace("draft9-beats-draft6", "draft3-beats-draft7");
        let a = run(&format!("knob draft 5 7 current=7 {DRAFT}"), &(win("draft5-beats-draft7-code_tps") + &f3), &[]);
        assert_eq!(grown(&a, "draft"), ["4", "6"]);
        // without the wall, 10 grows
        assert_eq!(grown(&run(&k, &win("draft8-beats-draft6-code_tps"), &[]), "draft"), ["7", "10"]);
    }

    #[test]
    fn constraint_bounds_growth() {
        // ub256 PROVEN over current 128 under ub<=256: midpoint 192; beyond 384 and every bisection (320..257) violate
        let k = "knob ub 128 256 current=128 metric=d100000 match=prefix:ub template=ub";
        let a = run(k, &win("ub256-beats-ub128-d100000"), &constraints(&[], &[]));
        assert_eq!(grown(&a, "ub"), ["192"]);
        assert!(a.iter().all(|i| !(i.knob == "ub" && i.value.parse::<i64>().unwrap() > 256)));
        // a winner the constraint forbids grows nothing
        let k = "knob ub 256 512 current=256 metric=d100000 match=prefix:ub template=ub";
        assert!(grown(&run(k, &win("ub512-beats-ub256-d100000"), &constraints(&[], &[])), "ub").is_empty());
        // range= clamps: draftmin 1 beat 3 -> beyond -1 clamps to 0
        let a = run("knob draftmin 1 3 current=3 metric=code_tps match=prefix:draftmin template=draftmin range=0:8", &win("draftmin1-beats-draftmin3-code_tps"), &[]);
        assert_eq!(grown(&a, "draftmin"), ["0", "2"]);
    }

    #[test]
    fn nothing_proven_means_the_catalog_is_unchanged() {
        let frozen: String = BUILTIN.lines().map(|l| format!("{l} grow=0\n")).collect();
        let fail = "{\"kind\":\"prediction\",\"ts\":1,\"pred\":\"9.tsv\",\"name\":\"draft9-faster\",\"value\":\"90 vs 100 (-10.00%, need >% 2%)\",\"op\":\">%\",\"threshold\":\"arm B\",\"verdict\":\"fail\"}\n";
        let once = "{\"kind\":\"prediction\",\"ts\":1,\"pred\":\"210.tsv\",\"name\":\"draft5-beats-draft7-code_tps\",\"value\":\"212.0 vs 200.0 (6.00%, need >% 3.5%)\",\"op\":\">%\",\"threshold\":\"arm B\",\"verdict\":\"pass\"}\n";
        for ledger in ["", fail, once, &invalid("draft9-beats-draft7-code_tps")] {
            let a = run(BUILTIN, ledger, &constraints(&[], &[]));
            let b = run(&frozen, ledger, &constraints(&[], &[]));
            assert_eq!(format!("{a:?}"), format!("{b:?}"), "ledger {ledger:?}");
            assert!(a.iter().all(|i| !i.why.contains("grown from")));
        }
    }
}
