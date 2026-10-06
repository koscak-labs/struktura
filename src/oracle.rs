//! Oracle: will this registered lab prediction PASS? A calibrated `p_pass`, zero LLM,
//! from the lab ledger alone, so a judging strand can earn the right to lead by its
//! forecast record (Brier score against a base-rate forecaster, scored by a neutral arbiter).
//!
//! Sources, most specific first:
//! 1. **Re-measure.** The same `(pred, name)` was scored before. When its margin is
//!    measurable and the noise band applies, `p = Phi(margin / sigma)` with
//!    `sigma = band / 1.96` (one run's difference noise, from the calibration arm):
//!    a pass by +0.3% inside a 2% band is a coin flip, a pass by +6% is near certain.
//!    Exact-count claims (no band) use their own pass history.
//! 2. **Kin.** Earlier predictions sharing a word (`faster`, `identical`, `beats`, ...)
//!    or the comparison operator: Beta-shrunk pass rates, averaged.
//! 3. **Base.** The running pass rate (Laplace).
//!
//! VOID / missing verdicts are instrument failures: neither learned nor scored.
//! The backtest is prequential: every scored row is forecast from strictly earlier
//! rows; rows scored within `gap` seconds of each other (one job's results) are
//! forecast together, before any of them is learned.

use crate::lab::{band_applies, margin, parse_json, Json};
use std::collections::BTreeMap;
use std::string::{String, ToString};
use std::vec::Vec;

/// Beta prior strength pulling a kin rate toward the base rate.
pub const KIN_PRIOR: f64 = 3.0;
/// Forecasts are kept inside [P_MIN, 1 - P_MIN]: no claim is ever certain.
pub const P_MIN: f64 = 0.03;

/// One scored prediction row.
#[derive(Clone, Debug, PartialEq)]
pub struct Scored {
    pub ts: f64,
    pub pred: String,
    pub name: String,
    pub op: String,
    pub value: String,
    pub threshold: String,
    pub pass: bool,
}

impl Scored {
    pub fn key(&self) -> String { std::format!("{}::{}", self.pred, self.name) }
}

/// Scored prediction rows (pass/fail only) and calibration samples `(ts, tps)`, in ledger order.
pub fn read(ledger: &str) -> (Vec<Scored>, Vec<(f64, f64)>) {
    let (mut rows, mut cal) = (Vec::new(), Vec::new());
    for line in ledger.lines() {
        let Some(j) = parse_json(line) else { continue };
        match j.str("kind") {
            Some("prediction") => {
                let pass = match j.str("verdict") { Some("pass") => true, Some("fail") => false, _ => continue };
                let s = |k: &str| match j.get(k) { Some(Json::Str(s)) => s.clone(), Some(Json::Num(x)) => std::format!("{}", x), _ => String::new() };
                rows.push(Scored { ts: j.num("ts").unwrap_or(0.0), pred: s("pred"), name: s("name"), op: s("op"),
                    value: s("value"), threshold: s("threshold"), pass });
            }
            Some("cal") if j.boolean("ok") != Some(false) => {
                if let Some(Json::Obj(_)) = j.get("code") {
                    if let Some(t) = j.get("code").and_then(|c| c.num("tps")) { cal.push((j.num("ts").unwrap_or(0.0), t)); }
                }
            }
            _ => {}
        }
    }
    rows.sort_by(|a, b| a.ts.partial_cmp(&b.ts).unwrap_or(core::cmp::Ordering::Equal));
    (rows, cal)
}

/// Word tokens of a prediction name (lowercase letter runs of length >= 3, minus filler)
/// plus the operator: the features kin predictions share.
pub fn features(name: &str, op: &str) -> Vec<String> {
    const FILLER: &[&str] = &["the", "and", "than", "with", "pct", "at"];
    let mut f: Vec<String> = name.split(|c: char| !c.is_ascii_alphabetic()).map(|t| t.to_ascii_lowercase())
        .filter(|t| t.len() >= 3 && !FILLER.contains(&t.as_str())).collect();
    f.sort();
    f.dedup();
    if !op.is_empty() { f.push(std::format!("op:{}", op)); }
    f
}

/// Standard normal CDF (Abramowitz-Stegun 7.1.26 via erf; |error| < 1.5e-7).
pub fn phi(z: f64) -> f64 {
    let x = z.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-x * x).exp();
    if z >= 0.0 { 0.5 * (1.0 + y) } else { 0.5 * (1.0 - y) }
}

/// What the oracle has learned so far.
#[derive(Clone, Debug, Default)]
pub struct Oracle {
    n: usize,
    pass: usize,
    /// key -> (n, pass, last margin, band applies)
    keys: BTreeMap<String, (usize, usize, Option<f64>, bool)>,
    feats: BTreeMap<String, (usize, usize)>,
    cal: Vec<f64>,
}

impl Oracle {
    pub fn new() -> Self { Self::default() }

    pub fn add_cal(&mut self, tps: f64) { if tps.is_finite() && tps > 0.0 { self.cal.push(tps); } }

    /// Single-run difference band (percent) from the calibration samples so far (5% until two exist).
    pub fn band(&self) -> f64 {
        let n = self.cal.len();
        if n < 2 { return 5.0; }
        let m = self.cal.iter().sum::<f64>() / n as f64;
        let sd = (self.cal.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1) as f64).sqrt();
        1.959964 * std::f64::consts::SQRT_2 * 100.0 * sd / m
    }

    /// Running pass rate, Laplace-smoothed.
    pub fn base(&self) -> f64 { (self.pass as f64 + 1.0) / (self.n as f64 + 2.0) }

    /// `p_pass` for a prediction about to be scored, and the source that decided it.
    pub fn forecast(&self, key: &str, name: &str, op: &str) -> (f64, &'static str) {
        let base = self.base();
        let kin: Vec<f64> = features(name, op).iter().filter_map(|f| self.feats.get(f))
            .map(|(n, p)| (*p as f64 + KIN_PRIOR * base) / (*n as f64 + KIN_PRIOR)).collect();
        let kin_p = if kin.is_empty() { None } else { Some(kin.iter().sum::<f64>() / kin.len() as f64) };
        let (p, basis) = match self.keys.get(key) {
            Some((_, _, Some(m), true)) => (phi(*m / (self.band() / 1.959964).max(1e-9)), "remeasure-margin"),
            // own history shrunk with the same prior strength as kin: the lab re-runs a failed claim AFTER a fix,
            // so one earlier verdict is weak evidence about the next
            Some((n, p, _, _)) => ((*p as f64 + KIN_PRIOR * kin_p.unwrap_or(base)) / (*n as f64 + KIN_PRIOR), "remeasure-history"),
            None => match kin_p { Some(k) => (k, "kin"), None => (base, "base") },
        };
        (p.clamp(P_MIN, 1.0 - P_MIN), basis)
    }

    pub fn learn(&mut self, r: &Scored) {
        self.n += 1;
        self.pass += r.pass as usize;
        let m = margin(&r.op, &r.value, &r.threshold);
        let e = self.keys.entry(r.key()).or_insert((0, 0, None, false));
        e.0 += 1; e.1 += r.pass as usize; e.2 = m; e.3 = m.is_some() && band_applies(&r.name, &r.value);
        for f in features(&r.name, &r.op) {
            let e = self.feats.entry(f).or_insert((0, 0));
            e.0 += 1; e.1 += r.pass as usize;
        }
    }
}

/// Prequential backtest of the oracle against the base-rate forecaster.
#[derive(Clone, Debug, Default)]
pub struct Backtest {
    pub n: usize,
    pub brier: f64,
    pub base_brier: f64,
    /// 1 - brier / base_brier (positive = better than the base rate).
    pub skill: f64,
    /// Per window of `w` forecasts: (first, last, oracle brier, base brier).
    pub windows: Vec<(usize, usize, f64, f64)>,
    /// Calibration bins over p: (lo, hi, n, mean p, pass rate).
    pub bins: Vec<(f64, f64, usize, f64, f64)>,
    /// How many forecasts each source made: (basis, n, brier).
    pub by_basis: Vec<(&'static str, usize, f64, f64)>,
    /// Every forecast in order: (ts, key, p, base p, pass, basis).
    pub forecasts: Vec<(f64, String, f64, f64, bool, &'static str)>,
}

/// Replay `ledger` in time order; each group of rows scored within `gap` s is forecast before it is learned.
pub fn backtest(ledger: &str, gap: f64, window: usize, nbins: usize) -> Backtest {
    let (rows, cal) = read(ledger);
    let mut o = Oracle::new();
    let mut ci = 0;
    let mut bt = Backtest::default();
    let mut i = 0;
    while i < rows.len() {
        let mut j = i + 1;
        while j < rows.len() && rows[j].ts - rows[j - 1].ts <= gap { j += 1; }
        while ci < cal.len() && cal[ci].0 < rows[i].ts { o.add_cal(cal[ci].1); ci += 1; }
        let base = o.base();
        for r in &rows[i..j] {
            let (p, basis) = o.forecast(&r.key(), &r.name, &r.op);
            bt.forecasts.push((r.ts, r.key(), p, base, r.pass, basis));
        }
        for r in &rows[i..j] { o.learn(r); }
        i = j;
    }
    let sq = |p: f64, y: bool| { let t = if y { 1.0 } else { 0.0 }; (p - t) * (p - t) };
    bt.n = bt.forecasts.len();
    if bt.n == 0 { return bt; }
    bt.brier = bt.forecasts.iter().map(|f| sq(f.2, f.4)).sum::<f64>() / bt.n as f64;
    bt.base_brier = bt.forecasts.iter().map(|f| sq(f.3, f.4)).sum::<f64>() / bt.n as f64;
    bt.skill = if bt.base_brier > 0.0 { 1.0 - bt.brier / bt.base_brier } else { 0.0 };
    let w = window.max(1);
    for (k, ch) in bt.forecasts.chunks(w).enumerate() {
        let n = ch.len() as f64;
        bt.windows.push((k * w + 1, k * w + ch.len(), ch.iter().map(|f| sq(f.2, f.4)).sum::<f64>() / n, ch.iter().map(|f| sq(f.3, f.4)).sum::<f64>() / n));
    }
    let nb = nbins.max(1);
    for b in 0..nb {
        let (lo, hi) = (b as f64 / nb as f64, (b + 1) as f64 / nb as f64);
        let inb: Vec<_> = bt.forecasts.iter().filter(|f| f.2 >= lo && (f.2 < hi || (b + 1 == nb && f.2 <= hi))).collect();
        if inb.is_empty() { continue; }
        let n = inb.len() as f64;
        bt.bins.push((lo, hi, inb.len(), inb.iter().map(|f| f.2).sum::<f64>() / n, inb.iter().filter(|f| f.4).count() as f64 / n));
    }
    for basis in ["remeasure-margin", "remeasure-history", "kin", "base"] {
        let v: Vec<_> = bt.forecasts.iter().filter(|f| f.5 == basis).collect();
        if !v.is_empty() { let n = v.len() as f64; bt.by_basis.push((basis, v.len(), v.iter().map(|f| sq(f.2, f.4)).sum::<f64>() / n, v.iter().map(|f| sq(f.3, f.4)).sum::<f64>() / n)); }
    }
    bt
}

/// The oracle trained on the whole ledger (for live forecasts).
pub fn trained(ledger: &str) -> Oracle {
    let (rows, cal) = read(ledger);
    let mut o = Oracle::new();
    for (_, t) in cal { o.add_cal(t); }
    for r in &rows { o.learn(r); }
    o
}

/// A queued job's registered prediction that has not been scored since it was queued.
#[derive(Clone, Debug, PartialEq)]
pub struct Pending {
    pub queued_ts: f64,
    pub job: String,
    pub pred_file: String,
    pub name: String,
    pub designed_by: String,
}

/// `kind:"queued"` rows whose names have no pass/fail/void/missing verdict (same pred file) after the queue time.
pub fn pending(ledger: &str) -> Vec<Pending> {
    let mut q: Vec<Pending> = Vec::new();
    let mut verdicts: Vec<(f64, String, String)> = Vec::new();
    for line in ledger.lines() {
        let Some(j) = parse_json(line) else { continue };
        match j.str("kind") {
            Some("queued") => {
                let ts = j.num("ts").unwrap_or(0.0);
                let names: Vec<String> = match j.get("names") { Some(Json::Arr(v)) => v.iter().filter_map(|x| if let Json::Str(s) = x { Some(s.clone()) } else { None }).collect(), _ => Vec::new() };
                for n in names {
                    q.push(Pending { queued_ts: ts, job: j.str("job").unwrap_or("").to_string(), pred_file: j.str("pred_file").unwrap_or("").to_string(),
                        name: n, designed_by: j.str("designed_by").unwrap_or("").to_string() });
                }
            }
            Some("prediction") => verdicts.push((j.num("ts").unwrap_or(0.0), j.str("pred").unwrap_or("").to_string(), j.str("name").unwrap_or("").to_string())),
            _ => {}
        }
    }
    q.retain(|p| !verdicts.iter().any(|(ts, pf, n)| *ts >= p.queued_ts && *n == p.name && (p.pred_file.is_empty() || *pf == p.pred_file)));
    q
}

/// Stable forecast id for a queued prediction (FNV-1a over job, pred file, name and queue time).
pub fn forecast_id(p: &Pending) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in std::format!("{}\u{1f}{}\u{1f}{}\u{1f}{}", p.job, p.pred_file, p.name, p.queued_ts as u64).bytes() { h ^= b as u64; h = h.wrapping_mul(0x100000001b3); }
    std::format!("{:016x}", h)
}

/// Resolve `kind:"forecast", target:"pass"` rows (from a forecast ledger) against the lab ledger:
/// the first pass/fail/void/missing verdict for the same name (and pred file) at or after the
/// forecast's queue time. Returns (id, outcome, p, brier or None for void/missing).
pub fn resolve(forecasts: &str, ledger: &str) -> Vec<(String, String, f64, Option<f64>)> {
    let mut verdicts: Vec<(f64, String, String, String)> = Vec::new();
    for line in ledger.lines() {
        let Some(j) = parse_json(line) else { continue };
        if j.str("kind") != Some("prediction") { continue; }
        verdicts.push((j.num("ts").unwrap_or(0.0), j.str("pred").unwrap_or("").to_string(), j.str("name").unwrap_or("").to_string(), j.str("verdict").unwrap_or("").to_string()));
    }
    verdicts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
    // pass 1: every id already scored, wherever its row sits (score rows follow their forecasts)
    let scored: Vec<String> = forecasts.lines().filter_map(parse_json)
        .filter(|j| j.str("kind") == Some("score") && j.str("target") == Some("pass")).filter_map(|j| j.str("id").map(|s| s.to_string())).collect();
    let mut out = Vec::new();
    for line in forecasts.lines() {
        let Some(j) = parse_json(line) else { continue };

        if j.str("kind") != Some("forecast") || j.str("target") != Some("pass") { continue; }
        let (Some(id), Some(name), Some(p)) = (j.str("id"), j.str("name"), j.num("p_pass")) else { continue };
        if scored.iter().any(|s| s == id) || out.iter().any(|o: &(String, String, f64, Option<f64>)| o.0 == id) { continue; }
        let since = j.num("queued_ts").unwrap_or_else(|| j.num("ts").unwrap_or(0.0));
        let pf = j.str("pred_file").unwrap_or("");
        if let Some(v) = verdicts.iter().find(|v| v.0 >= since && v.2 == name && (pf.is_empty() || v.1 == pf)) {
            let brier = match v.3.as_str() { "pass" => Some((p - 1.0) * (p - 1.0)), "fail" => Some(p * p), _ => None };
            out.push((id.to_string(), v.3.clone(), p, brier));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pred(ts: u32, pf: &str, name: &str, value: &str, verdict: &str) -> String {
        std::format!("{{\"kind\":\"prediction\",\"ts\":{ts},\"pred\":\"{pf}\",\"name\":\"{name}\",\"value\":\"{value}\",\"op\":\">%\",\"threshold\":\"arm B\",\"verdict\":\"{verdict}\"}}\n")
    }
    fn cal(ts: u32, tps: f64) -> String { std::format!("{{\"kind\":\"cal\",\"ts\":{ts},\"ok\":true,\"code\":{{\"tps\":{tps}}}}}\n") }

    #[test]
    fn phi_is_the_normal_cdf() {
        assert!((phi(0.0) - 0.5).abs() < 1e-7);
        assert!((phi(1.959964) - 0.975).abs() < 1e-6);
        assert!((phi(-1.0) - 0.158655).abs() < 1e-5);
    }

    #[test]
    fn remeasure_uses_the_margin_against_the_noise_band() {
        // calibration CV ~0.7% -> band ~1.96%: +4.5% past threshold is near-certain, +0.3% is a coin flip
        let mut l = cal(1, 200.0) + &cal(2, 202.0) + &cal(3, 199.0);
        l += &pred(10, "a.tsv", "x-faster", "210 vs 200 (5.00%, need >% 0.5%)", "pass");
        l += &pred(11, "b.tsv", "y-faster", "201.6 vs 200 (0.80%, need >% 0.5%)", "pass");
        let o = trained(&l);
        let (pa, ba) = o.forecast("a.tsv::x-faster", "x-faster", ">%");
        let (pb, _) = o.forecast("b.tsv::y-faster", "y-faster", ">%");
        assert_eq!(ba, "remeasure-margin");
        assert!(pa > 0.95 && pb > 0.5 && pb < 0.7, "pa={pa} pb={pb} band={}", o.band());
    }

    #[test]
    fn backtest_never_sees_the_future() {
        let mut l = cal(1, 200.0) + &cal(2, 202.0);
        for k in 0..30u32 {
            let v = if k % 4 == 0 { "fail" } else { "pass" };
            l += &pred(100 + 100 * k, &std::format!("{k}.tsv"), if k % 2 == 0 { "cand-not-slower" } else { "big-faster" }, "1 vs 1 (0.00%, need >% 1%)", v);
        }
        let full = backtest(&l, 30.0, 10, 5);
        let cut: String = l.lines().take(2 + 12).map(|x| std::format!("{x}\n")).collect();
        let part = backtest(&cut, 30.0, 10, 5);
        assert_eq!(part.n, 12);
        for (a, b) in part.forecasts.iter().zip(&full.forecasts) { assert_eq!(a, b, "a forecast changed when later rows were added"); }
        // flipping a LATER outcome leaves every earlier forecast bit-identical
        let flipped = l.replacen("\"29.tsv\",\"name\":\"big-faster\",\"value\":\"1 vs 1 (0.00%, need >% 1%)\",\"op\":\">%\",\"threshold\":\"arm B\",\"verdict\":\"pass\"", "\"29.tsv\",\"name\":\"big-faster\",\"value\":\"1 vs 1 (0.00%, need >% 1%)\",\"op\":\">%\",\"threshold\":\"arm B\",\"verdict\":\"fail\"", 1);
        assert_ne!(flipped, l);
        let f2 = backtest(&flipped, 30.0, 10, 5);
        assert_eq!(&f2.forecasts[..29], &full.forecasts[..29]);
    }

    #[test]
    fn pending_and_resolve_follow_the_queue() {
        let mut l = String::from("{\"kind\":\"queued\",\"ts\":100,\"job\":\"301-ub128\",\"pred_file\":\"301.tsv\",\"names\":[\"ub128-beats\",\"ub128-not-slower\"],\"designed_by\":\"fuxi\"}\n");
        l += &pred(50, "301.tsv", "ub128-beats", "1 vs 1 (0.0%, need >% 1%)", "fail"); // BEFORE the queue: does not resolve it
        let p = pending(&l);
        assert_eq!(p.iter().map(|x| x.name.as_str()).collect::<Vec<_>>(), std::vec!["ub128-beats", "ub128-not-slower"]);
        let id0 = forecast_id(&p[0]);
        assert_eq!(id0, forecast_id(&p[0].clone()), "stable");
        let fx = std::format!("{{\"kind\":\"forecast\",\"id\":\"{id0}\",\"target\":\"pass\",\"name\":\"ub128-beats\",\"pred_file\":\"301.tsv\",\"p_pass\":0.8,\"queued_ts\":100}}\n\
                               {{\"kind\":\"forecast\",\"id\":\"ff\",\"target\":\"pass\",\"name\":\"ub128-not-slower\",\"pred_file\":\"301.tsv\",\"p_pass\":0.9,\"queued_ts\":100}}\n");
        assert!(resolve(&fx, &l).is_empty(), "nothing scored after the queue yet");
        l += &pred(200, "301.tsv", "ub128-beats", "1 vs 1 (2.0%, need >% 1%)", "pass");
        l += "{\"kind\":\"prediction\",\"ts\":201,\"pred\":\"301.tsv\",\"name\":\"ub128-not-slower\",\"value\":\"-\",\"verdict\":\"void\"}\n";
        assert!(pending(&l).is_empty());
        let r = resolve(&fx, &l);
        assert_eq!(r.len(), 2);
        assert_eq!((r[0].1.as_str(), r[0].3.map(|b| (b * 1000.0).round())), ("pass", Some(40.0)));
        assert_eq!((r[1].1.as_str(), r[1].3), ("void", None), "void is resolved but never scored");
        let already = fx + &std::format!("{{\"kind\":\"score\",\"target\":\"pass\",\"id\":\"{id0}\"}}\n");
        assert_eq!(resolve(&already, &l).len(), 1, "a scored id is never scored twice");
    }
}
