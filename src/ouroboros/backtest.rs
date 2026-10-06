//! Backtest: replay the ledger as it stood at each job's start and check the
//! brain's signals against what happened afterwards.
//!
//! - **Fragile flags**: a prediction flagged fragile at cut t is a claim
//!   "this could flip". It is checked on later scored rows of the same
//!   prediction: flipped / held. The same is counted for predictions not
//!   flagged, so the flag can be compared with the base rate.
//! - **Calibration stability**: the calibration factor at each cut versus the
//!   final one (does the brain's self-knowledge settle?).
//!
//! Job logs have no timestamps, so arm evidence is not replayed. With too few
//! re-scored predictions the verdict says so instead of quoting a hit rate.

use std::collections::{BTreeMap, BTreeSet};

use super::learn::learn;
use super::observe;
use crate::lab::parse_json;

#[derive(Clone, Debug, Default)]
pub struct Backtest {
    pub cuts: usize,
    pub flagged: usize,
    pub flagged_rescored: usize,
    pub flagged_flipped: usize,
    pub unflagged_rescored: usize,
    pub unflagged_flipped: usize,
    /// (cut ts, calibration, n) per cut.
    pub calibration_path: Vec<(f64, f64, usize)>,
    pub verdict: String,
}

/// Rows with a time: `ts`, or a job's `start`.
fn timed(ledger: &str) -> Vec<(f64, String)> {
    let mut v: Vec<(f64, String)> = ledger.lines().filter_map(|l| {
        let j = parse_json(l)?;
        Some((j.num("ts").or_else(|| j.num("start"))?, l.to_string()))
    }).collect();
    v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    v
}

pub fn backtest(ledger: &str) -> Backtest {
    let rows = timed(ledger);
    let mut cuts: Vec<f64> = rows.iter().filter_map(|(_, l)| {
        let j = parse_json(l)?;
        if j.str("kind") == Some("job") { j.num("start") } else { None }
    }).collect();
    cuts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    cuts.dedup();
    // Scored verdict history per (pred, name).
    let mut hist: BTreeMap<(String, String), Vec<(f64, String)>> = BTreeMap::new();
    for (ts, l) in &rows {
        let Some(j) = parse_json(l) else { continue };
        if j.str("kind") != Some("prediction") { continue; }
        let v = j.str("verdict").unwrap_or("");
        if v != "pass" && v != "fail" { continue; }
        hist.entry((j.str("pred").unwrap_or("").into(), j.str("name").unwrap_or("").into())).or_default().push((*ts, v.into()));
    }
    let mut b = Backtest { cuts: cuts.len(), ..Default::default() };
    let mut ever_flagged: BTreeSet<(String, String)> = BTreeSet::new();
    let mut judged: BTreeSet<(String, String)> = BTreeSet::new();
    for &cut in &cuts {
        let prefix: String = rows.iter().filter(|(ts, _)| *ts < cut).map(|(_, l)| format!("{}\n", l)).collect();
        let ls = learn(&observe(&prefix, &[]));
        b.calibration_path.push((cut, ls.calibration, ls.calibration_n));
        let flagged: BTreeSet<(String, String)> = ls.fragile.iter().map(|f| (f.pred.clone(), f.name.clone())).collect();
        for (key, h) in &hist {
            // Judge each prediction once, at the first cut where it already had a verdict.
            if judged.contains(key) { continue; }
            let Some(at) = h.iter().filter(|(t, _)| *t < cut).last() else { continue };
            let later: Vec<&(f64, String)> = h.iter().filter(|(t, _)| *t >= cut).collect();
            if later.is_empty() { continue; }
            judged.insert(key.clone());
            let flipped = later.iter().any(|(_, v)| *v != at.1);
            if flagged.contains(key) {
                ever_flagged.insert(key.clone());
                b.flagged_rescored += 1;
                if flipped { b.flagged_flipped += 1; }
            } else {
                b.unflagged_rescored += 1;
                if flipped { b.unflagged_flipped += 1; }
            }
        }
        for f in &flagged { ever_flagged.insert(f.clone()); }
    }
    b.flagged = ever_flagged.len();
    b.verdict = if b.cuts == 0 {
        "no job rows: nothing to replay".into()
    } else if b.flagged_rescored < 3 {
        format!("too little history: {} fragile-flagged prediction(s) were re-scored later (need >= 3 to judge the flag); {} unflagged re-scored, {} flipped",
            b.flagged_rescored, b.unflagged_rescored, b.unflagged_flipped)
    } else {
        let fr = b.flagged_flipped as f64 / b.flagged_rescored as f64;
        let ur = if b.unflagged_rescored > 0 { b.unflagged_flipped as f64 / b.unflagged_rescored as f64 } else { f64::NAN };
        format!("fragile flag: {}/{} flipped ({:.0}%) vs unflagged {}/{} ({:.0}%)", b.flagged_flipped, b.flagged_rescored, 100.0 * fr,
            b.unflagged_flipped, b.unflagged_rescored, 100.0 * ur)
    };
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(ts: u32, name: &str, value: &str, verdict: &str) -> String {
        format!("{{\"kind\":\"prediction\",\"ts\":{},\"pred\":\"1.tsv\",\"name\":\"{}\",\"value\":\"{}\",\"op\":\">%\",\"threshold\":\"arm B\",\"verdict\":\"{}\"}}\n", ts, name, value, verdict)
    }
    fn job(start: u32) -> String { format!("{{\"kind\":\"job\",\"job\":\"j{}\",\"start\":{},\"end\":{},\"rc\":0}}\n", start, start, start + 10) }

    #[test]
    fn fragile_flag_is_scored_against_later_reruns() {
        let mut l = String::from("{\"kind\":\"cal\",\"ts\":1,\"ok\":true,\"code\":{\"tps\":200.0}}\n{\"kind\":\"cal\",\"ts\":2,\"ok\":true,\"code\":{\"tps\":202.0}}\n");
        // Three thin passes (fragile) and three wide passes, all re-run after job 100.
        for (i, n) in ["t1", "t2", "t3"].iter().enumerate() { l += &p(10 + i as u32, n, "105.3 vs 100 (5.30%, need >% 5%)", "pass"); }
        for (i, n) in ["w1", "w2", "w3"].iter().enumerate() { l += &p(20 + i as u32, n, "130 vs 100 (30.00%, need >% 5%)", "pass"); }
        l += &job(100);
        l += &p(110, "t1", "104 vs 100 (4.00%, need >% 5%)", "fail");
        l += &p(111, "t2", "104.5 vs 100 (4.50%, need >% 5%)", "fail");
        l += &p(112, "t3", "106 vs 100 (6.00%, need >% 5%)", "pass");
        for n in ["w1", "w2", "w3"] { l += &p(120, n, "131 vs 100 (31.00%, need >% 5%)", "pass"); }
        let b = backtest(&l);
        assert_eq!(b.cuts, 1);
        assert_eq!((b.flagged_rescored, b.flagged_flipped), (3, 2));
        assert_eq!((b.unflagged_rescored, b.unflagged_flipped), (3, 0));
        assert!(b.verdict.contains("2/3 flipped"), "{}", b.verdict);
    }

    #[test]
    fn honest_when_history_is_short() {
        let l = format!("{}{}", p(10, "a", "105.3 vs 100 (5.30%, need >% 5%)", "pass"), job(100));
        let b = backtest(&l);
        assert!(b.verdict.starts_with("too little history"), "{}", b.verdict);
        assert_eq!(backtest("").verdict, "no job rows: nothing to replay");
    }
}
