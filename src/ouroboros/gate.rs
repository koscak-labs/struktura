//! Prod gate: before any promotion, production must not have regressed.
//!
//! Requests (prod-pulse `req.csv`) are grouped by the ledger's `deploy` rows
//! (a request up to [`DEPLOY_LEAD_S`] before a row belongs to it: the lab
//! writes the row after the deploy's verify request was served) and judged by
//! `pulse::analyze`. Three states:
//! - **open**: the latest judged deploy is not slower (solo and contention);
//! - **closed**: the latest judged deploy is slower;
//! - **no evidence**: no deploy rows, or too few requests to judge.
//! Server processes are reloads, not deploys, so without ledger deploy rows
//! the gate never closes.

use crate::pulse::{analyze, Row, Verdict};

pub const DEPLOY_LEAD_S: u64 = 15;

#[derive(Clone, Debug, PartialEq)]
pub enum Gate { Open(String), Closed(String), NoEvidence(String) }

impl Gate {
    pub fn as_str(&self) -> &'static str { match self { Gate::Open(_) => "open", Gate::Closed(_) => "closed", Gate::NoEvidence(_) => "no-evidence" } }
    pub fn why(&self) -> &str { match self { Gate::Open(w) | Gate::Closed(w) | Gate::NoEvidence(w) => w } }
}

/// `req_csv`: prod-pulse rows (pid,ctx,step_ms,busy,tps,start,mlen,model at cols 1,4,11,12,8,3,10,13).
/// `deploys`: (unix ts, label) from the ledger. `model`: keep one alias ("" = all).
pub fn gate(req_csv: &str, deploys: &[(f64, String)], model: &str, min_effect_pct: f64) -> Gate {
    if deploys.is_empty() { return Gate::NoEvidence("no kind:\"deploy\" rows in the ledger".into()); }
    let mut dep: Vec<u64> = deploys.iter().map(|(t, _)| *t as u64).collect();
    dep.sort();
    let mut rows: Vec<(u64, Row)> = Vec::new();
    for line in req_csv.lines() {
        let f: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if !model.is_empty() && f.get(13).copied().unwrap_or("") != model { continue; }
        let g = |k: usize| f.get(k).and_then(|v| v.parse::<f64>().ok());
        if let (Some(start), Some(ctx), Some(busy), Some(tps)) = (g(3), g(4), g(12), g(8)) {
            if tps <= 0.0 { continue; }
            let s = start as u64;
            let seg = dep.iter().rev().find(|t| **t <= s + DEPLOY_LEAD_S).copied().unwrap_or(0);
            rows.push((s, Row { seg, ctx_k: ctx / 1000.0, busy, ms: 1000.0 / tps, tps, mlen: g(10).unwrap_or(f64::NAN) }));
        }
    }
    rows.sort_by_key(|r| r.0);
    let rows: Vec<Row> = rows.into_iter().map(|r| r.1).collect();
    if rows.len() < 2 { return Gate::NoEvidence("fewer than 2 prod requests".into()); }
    let r = analyze(&rows, min_effect_pct, 2000, 42);
    let latest = r.comparisons.iter().rev().find(|c| c.verdict != Verdict::Insufficient && c.verdict != Verdict::Inconclusive);
    match latest {
        None => Gate::NoEvidence(format!("{} requests over {} deploy segment(s): no deploy has enough requests to judge", r.rows, r.segments.len())),
        Some(c) if c.verdict == Verdict::Slower => Gate::Closed(format!("{} {} -> {}: {:+.1} [{:+.1}, {:+.1}] SLOWER", c.kind, c.from, c.to, c.delta, c.ci_low, c.ci_high)),
        Some(c) => Gate::Open(format!("{} {} -> {}: {:+.1} [{:+.1}, {:+.1}] {}", c.kind, c.from, c.to, c.delta, c.ci_low, c.ci_high, c.verdict.as_str().to_uppercase())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn csv(level_before: f64, level_after: f64, dep_ts: u64) -> String {
        let mut s = String::new();
        for i in 0..80u64 {
            let start = dep_ts - 4000 + i * 100;
            let ms = if start + DEPLOY_LEAD_S >= dep_ts { level_after } else { level_before } + (i % 5) as f64 * 0.05;
            s += &format!("{},1,0,{},{},0,0,100,{},1,3,{},0,qwen38,5000\n", start + 50, start, 20000 + (i % 7) * 1000, 1000.0 / ms, ms * 3.0);
        }
        s
    }

    #[test]
    fn three_states() {
        let d = 1_000_000u64;
        let dep = vec![(d as f64 - 5000.0, "old".to_string()), (d as f64, "new".to_string())];
        assert_eq!(gate(&csv(10.0, 7.0, d), &dep, "qwen38", 1.6).as_str(), "open");
        assert_eq!(gate(&csv(7.0, 10.0, d), &dep, "qwen38", 1.6).as_str(), "closed");
        assert_eq!(gate(&csv(7.0, 10.0, d), &[], "qwen38", 1.6).as_str(), "no-evidence");
        assert_eq!(gate(&csv(7.0, 10.0, d), &dep, "qwen36", 1.6).as_str(), "no-evidence", "other model filtered out");
    }
}
