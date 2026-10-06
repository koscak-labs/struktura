//! Lab memory: what an experiment ledger has established, and how well the
//! lab predicts its own results. No model in the loop.
//!
//! Input is the append-only JSON-lines ledger the raven lab writes
//! (`kind` = job | cal | prediction | window | deploy | constraint ...).
//!
//! - **Noise floor**: coefficient of variation of the calibration arm's
//!   throughput; the smallest believable effect is `max(2 × CV, 1%)`.
//! - **Knowledge**: the latest verdict per (pred file, name) is authoritative;
//!   `void` (instrument bug) supersedes and is never a fail; `missing` never
//!   overrides a scored verdict. Each pred file is CONFIRMED (all pass),
//!   FALSIFIED (all fail), MIXED, or VOID.
//! - **Self-calibration**: for every scored prediction the margin by which it
//!   cleared (or missed) its threshold, in percent. Margins inside the noise
//!   floor are fragile verdicts; verdicts that changed when re-scored are
//!   flips. Both say how much to trust the lab's own scoreboard.
//! - **Efficiency**: job minutes inside lab windows versus window minutes.

#![cfg(feature = "std")]

use std::collections::BTreeMap;

// ---------------------------------------------------------------- JSON ----

/// Minimal JSON value (enough for ledger rows; strict about syntax).
#[derive(Clone, Debug, PartialEq)]
pub enum Json { Null, Bool(bool), Num(f64), Str(String), Arr(Vec<Json>), Obj(Vec<(String, Json)>) }

impl Json {
    pub fn get(&self, k: &str) -> Option<&Json> {
        if let Json::Obj(v) = self { v.iter().find(|(kk, _)| kk == k).map(|(_, x)| x) } else { None }
    }
    pub fn str(&self, k: &str) -> Option<&str> { if let Some(Json::Str(s)) = self.get(k) { Some(s) } else { None } }
    pub fn num(&self, k: &str) -> Option<f64> {
        match self.get(k) { Some(Json::Num(x)) => Some(*x), Some(Json::Str(s)) => s.trim().parse().ok(), _ => None }
    }
    pub fn boolean(&self, k: &str) -> Option<bool> { if let Some(Json::Bool(b)) = self.get(k) { Some(*b) } else { None } }
}

pub fn parse_json(s: &str) -> Option<Json> {
    let b = s.as_bytes();
    let mut i = 0;
    let v = value(b, &mut i)?;
    ws(b, &mut i);
    if i == b.len() { Some(v) } else { None }
}

fn ws(b: &[u8], i: &mut usize) { while *i < b.len() && (b[*i] as char).is_ascii_whitespace() { *i += 1; } }

fn value(b: &[u8], i: &mut usize) -> Option<Json> {
    ws(b, i);
    match *b.get(*i)? {
        b'{' => {
            *i += 1; let mut out = Vec::new(); ws(b, i);
            if b.get(*i) == Some(&b'}') { *i += 1; return Some(Json::Obj(out)); }
            loop {
                ws(b, i);
                let Json::Str(k) = string(b, i)? else { return None };
                ws(b, i); if b.get(*i) != Some(&b':') { return None; } *i += 1;
                out.push((k, value(b, i)?)); ws(b, i);
                match b.get(*i)? { b',' => *i += 1, b'}' => { *i += 1; return Some(Json::Obj(out)); } _ => return None }
            }
        }
        b'[' => {
            *i += 1; let mut out = Vec::new(); ws(b, i);
            if b.get(*i) == Some(&b']') { *i += 1; return Some(Json::Arr(out)); }
            loop {
                out.push(value(b, i)?); ws(b, i);
                match b.get(*i)? { b',' => *i += 1, b']' => { *i += 1; return Some(Json::Arr(out)); } _ => return None }
            }
        }
        b'"' => string(b, i),
        b't' if b[*i..].starts_with(b"true") => { *i += 4; Some(Json::Bool(true)) }
        b'f' if b[*i..].starts_with(b"false") => { *i += 5; Some(Json::Bool(false)) }
        b'n' if b[*i..].starts_with(b"null") => { *i += 4; Some(Json::Null) }
        _ => {
            let st = *i;
            while *i < b.len() && matches!(b[*i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') { *i += 1; }
            std::str::from_utf8(&b[st..*i]).ok()?.parse().ok().map(Json::Num)
        }
    }
}

fn string(b: &[u8], i: &mut usize) -> Option<Json> {
    if b.get(*i) != Some(&b'"') { return None; }
    *i += 1;
    let mut out: Vec<u8> = Vec::new();
    while *i < b.len() {
        match b[*i] {
            b'"' => { *i += 1; return String::from_utf8(out).ok().map(Json::Str); }
            b'\\' => {
                *i += 1;
                match *b.get(*i)? {
                    b'n' => out.push(b'\n'), b't' => out.push(b'\t'), b'r' => out.push(b'\r'),
                    b'b' => out.push(8), b'f' => out.push(12),
                    b'u' => {
                        let h = std::str::from_utf8(b.get(*i + 1..*i + 5)?).ok()?;
                        let c = char::from_u32(u32::from_str_radix(h, 16).ok()?).unwrap_or('\u{fffd}');
                        let mut buf = [0u8; 4]; out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes()); *i += 4;
                    }
                    c => out.push(c),
                }
                *i += 1;
            }
            c => { out.push(c); *i += 1; }
        }
    }
    None
}

// ---------------------------------------------------------- analysis ----

#[derive(Clone, Debug, PartialEq)]
pub struct Prediction {
    pub pred: String,
    pub name: String,
    pub ts: f64,
    pub verdict: String,
    pub value: String,
    pub op: String,
    pub threshold: String,
    /// Percent by which the result cleared (+) or missed (-) its threshold.
    pub margin_pct: Option<f64>,
    /// Earlier scored verdicts for the same (pred, name) that differ from this one.
    pub flips: usize,
}

#[derive(Clone, Debug, Default)]
pub struct PredFile {
    pub pred: String,
    pub pass: usize,
    pub fail: usize,
    pub void: usize,
    pub missing: usize,
    pub status: &'static str,
}

#[derive(Clone, Debug, Default)]
pub struct LabReport {
    pub rows: usize,
    pub bad_rows: usize,
    pub kinds: BTreeMap<String, usize>,
    pub cal_n: usize,
    pub cal_mean_tps: f64,
    pub cal_cv_pct: f64,
    pub floor_pct: f64,
    /// 95% band of the difference of two single runs: 1.96 x sqrt(2) x CV. A margin
    /// inside it is fragile (2 x CV alone is ~1.4 sd of a single-run difference,
    /// so ~16% of no-effect comparisons cross it).
    pub pair_band_pct: f64,
    pub predictions: Vec<Prediction>,
    pub files: Vec<PredFile>,
    pub fragile: usize,
    pub flipped: usize,
    /// Passes that cleared their threshold by more than 3 noise floors: the
    /// prediction was a safe bet, so its verdict carried little information.
    pub easy: usize,
    pub median_pass_margin: f64,
    pub jobs: usize,
    pub jobs_failed: usize,
    pub job_min: f64,
    pub window_min: f64,
    pub job_min_in_windows: f64,
    pub cal_min: f64,
    pub deploys: Vec<(f64, String)>,
    pub constraints: Vec<String>,
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() { return f64::NAN; }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

/// Margin in percent for one scored prediction, from lab-score's value text.
/// Relative ops: value = "A vs B (d%, need OP p%)". Absolute: numeric value vs threshold.
pub fn margin(op: &str, value: &str, threshold: &str) -> Option<f64> {
    if let Some(open) = value.find('(') {
        let inner = &value[open + 1..];
        let d: f64 = inner.split('%').next()?.trim().parse().ok()?;
        let need = inner.split("need").nth(1)?;
        let p: f64 = need.split_whitespace().nth(1)?.trim_end_matches(|c| c == '%' || c == ')').parse().ok()?;
        let rel = need.split_whitespace().next()?;
        return match rel { ">%" => Some(d - p), "<%" => Some(-d - p), "~%" => Some(p - d.abs()), _ => None };
    }
    let v: f64 = value.trim().parse().ok()?;
    let t: f64 = threshold.trim().parse().ok()?;
    let scale = if t.abs() > 1e-12 { t.abs() } else { return None };
    match op {
        ">=" | ">" => Some(100.0 * (v - t) / scale),
        "<=" | "<" => Some(100.0 * (t - v) / scale),
        "==" => Some(-100.0 * (v - t).abs() / scale),
        _ => None,
    }
}

pub fn analyze(ledger: &str) -> LabReport {
    let mut r = LabReport::default();
    let mut cal: Vec<f64> = Vec::new();
    let mut history: BTreeMap<(String, String), Vec<Prediction>> = BTreeMap::new();
    let mut windows: Vec<(f64, f64)> = Vec::new();
    let mut open: Option<f64> = None;
    let mut jobs: Vec<(f64, f64)> = Vec::new();
    for line in ledger.lines().filter(|l| !l.trim().is_empty()) {
        r.rows += 1;
        let Some(j) = parse_json(line) else { r.bad_rows += 1; continue };
        let kind = j.str("kind").unwrap_or("?").to_string();
        *r.kinds.entry(kind.clone()).or_default() += 1;
        match kind.as_str() {
            "cal" => {
                if j.boolean("ok") != Some(false) {
                    if let Some(t) = j.get("code").and_then(|c| c.num("tps")) { cal.push(t); }
                }
                r.cal_min += j.num("dur_s").unwrap_or(0.0) / 60.0;
            }
            "prediction" => {
                let p = Prediction {
                    pred: j.str("pred").unwrap_or("").to_string(), name: j.str("name").unwrap_or("").to_string(),
                    ts: j.num("ts").unwrap_or(0.0), verdict: j.str("verdict").unwrap_or("").to_string(),
                    value: j.str("value").unwrap_or("").to_string(), op: j.str("op").unwrap_or("").to_string(),
                    threshold: j.str("threshold").unwrap_or("").to_string(), margin_pct: None, flips: 0,
                };
                history.entry((p.pred.clone(), p.name.clone())).or_default().push(p);
            }
            "job" => {
                r.jobs += 1;
                if j.num("rc").map(|c| c != 0.0).unwrap_or(false) { r.jobs_failed += 1; }
                if let (Some(s), Some(e)) = (j.num("start"), j.num("end")) { if e >= s { jobs.push((s, e)); r.job_min += (e - s) / 60.0; } }
            }
            "window" => {
                let ts = j.num("ts").unwrap_or(0.0);
                match j.str("event") {
                    Some("up") => { if let Some(o) = open { windows.push((o, ts)); } open = Some(ts); }
                    Some(_) => { if let Some(o) = open.take() { windows.push((o, ts)); } }
                    None => {}
                }
            }
            "deploy" => {
                let mut label = j.str("binary").unwrap_or("deploy").to_string();
                if let Some(c) = j.num("chunk") { label.push_str(&format!(" chunk={}", c)); }
                r.deploys.push((j.num("ts").unwrap_or(0.0), label));
            }
            "constraint" => {
                r.constraints.push(format!("{} {} {}", j.str("knob").unwrap_or("?"), j.str("op").unwrap_or("?"),
                    j.get("value").map(|v| match v { Json::Num(x) => format!("{}", x), Json::Str(s) => s.clone(), _ => "?".into() }).unwrap_or_default()));
            }
            _ => {}
        }
    }

    // Noise floor from the calibration arm.
    r.cal_n = cal.len();
    if cal.len() >= 2 {
        let m = cal.iter().sum::<f64>() / cal.len() as f64;
        let sd = (cal.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (cal.len() - 1) as f64).sqrt();
        r.cal_mean_tps = m;
        r.cal_cv_pct = 100.0 * sd / m;
        r.floor_pct = (2.0 * r.cal_cv_pct).max(1.0);
        r.pair_band_pct = 1.959964 * std::f64::consts::SQRT_2 * r.cal_cv_pct;
    } else {
        r.cal_mean_tps = f64::NAN; r.cal_cv_pct = f64::NAN; r.floor_pct = 5.0; r.pair_band_pct = 5.0;
    }

    // Latest authoritative verdict per (pred, name).
    let mut files: BTreeMap<String, PredFile> = BTreeMap::new();
    let mut pass_margins = Vec::new();
    for ((pred, _), mut rows) in history {
        rows.sort_by(|a, b| a.ts.partial_cmp(&b.ts).unwrap_or(std::cmp::Ordering::Equal));
        let scored: Vec<&Prediction> = rows.iter().filter(|p| p.verdict != "missing" && !p.verdict.is_empty()).collect();
        let mut latest = match scored.last() { Some(p) => (*p).clone(), None => rows.last().unwrap().clone() };
        latest.flips = scored.iter().filter(|p| p.verdict != latest.verdict && p.verdict != "void" && latest.verdict != "void").count();
        latest.margin_pct = margin(&latest.op, &latest.value, &latest.threshold);
        let f = files.entry(pred.clone()).or_insert_with(|| PredFile { pred: pred.clone(), ..Default::default() });
        match latest.verdict.as_str() { "pass" => f.pass += 1, "fail" => f.fail += 1, "void" => f.void += 1, _ => f.missing += 1 }
        if latest.flips > 0 { r.flipped += 1; }
        if let Some(m) = latest.margin_pct {
            if latest.verdict == "pass" || latest.verdict == "fail" {
                if m.abs() < r.pair_band_pct.max(r.floor_pct) { r.fragile += 1; }
                if latest.verdict == "pass" { pass_margins.push(m); if m > 3.0 * r.floor_pct { r.easy += 1; } }
            }
        }
        r.predictions.push(latest);
    }
    r.median_pass_margin = median(&mut pass_margins);
    for f in files.values_mut() {
        f.status = if f.pass + f.fail == 0 { if f.void > 0 { "VOID" } else { "PENDING" } }
            else if f.fail == 0 { "CONFIRMED" } else if f.pass == 0 { "FALSIFIED" } else { "MIXED" };
    }
    r.files = files.into_values().collect();

    // Efficiency: job time that fell inside lab windows.
    if let Some(o) = open { if let Some(&(_, e)) = jobs.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()) { if e > o { windows.push((o, e)); } } }
    r.window_min = windows.iter().map(|(a, b)| (b - a) / 60.0).sum();
    r.job_min_in_windows = jobs.iter().map(|&(s, e)| windows.iter()
        .map(|&(a, b)| (e.min(b) - s.max(a)).max(0.0)).sum::<f64>()).sum::<f64>() / 60.0;
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_parses_nested_and_escapes() {
        let j = parse_json(r#"{"kind":"cal","ok":true,"code":{"tps":200.5,"acc":236},"s":"a\"b\tc","arr":[1,null,false]}"#).unwrap();
        assert_eq!(j.get("code").unwrap().num("tps"), Some(200.5));
        assert_eq!(j.str("s"), Some("a\"b\tc"));
        assert_eq!(j.boolean("ok"), Some(true));
        assert!(parse_json(r#"{"a":1,}"#).is_none());
        assert!(parse_json(r#"{"a":1} x"#).is_none());
    }

    #[test]
    fn margins_from_lab_score_value_text() {
        assert!((margin(">%", "1484 vs 1397 (6.23%, need >% 1.16%)", "arm B").unwrap() - 5.07).abs() < 1e-9);
        assert!((margin("<%", "90 vs 100 (-10.00%, need <% 2%)", "arm B").unwrap() - 8.0).abs() < 1e-9);
        assert!((margin("~%", "101 vs 100 (1.00%, need ~% 1.16%)", "arm B").unwrap() - 0.16).abs() < 1e-9);
        assert!((margin(">=", "10", "9").unwrap() - 11.111).abs() < 1e-3);
        assert!((margin("<=", "45", "50").unwrap() - 10.0).abs() < 1e-9);
        assert_eq!(margin("~", "FOUND", "FOUND"), None);
    }

    const LEDGER: &str = r#"{"kind":"window","ts":1000,"event":"up"}
{"kind":"cal","ts":1001,"ok":true,"dur_s":60,"code":{"tps":200.0}}
{"kind":"cal","ts":1002,"ok":true,"dur_s":60,"code":{"tps":202.0}}
{"kind":"cal","ts":1003,"ok":false,"code":{"tps":10.0}}
{"kind":"job","job":"1.sh","start":1100,"end":1700,"rc":0}
{"kind":"job","job":"2.sh","start":1700,"end":1800,"rc":1}
{"kind":"prediction","ts":1700,"pred":"1.tsv","name":"a","value":"","op":">=","threshold":"9","verdict":"missing"}
{"kind":"prediction","ts":1701,"pred":"1.tsv","name":"a","value":"10","op":">=","threshold":"9","verdict":"pass"}
{"kind":"prediction","ts":1701,"pred":"1.tsv","name":"b","value":"101 vs 100 (1.00%, need >% 0.5%)","op":">%","threshold":"arm B","verdict":"pass"}
{"kind":"prediction","ts":1702,"pred":"2.tsv","name":"x","value":"0","op":">=","threshold":"1","verdict":"fail"}
{"kind":"prediction","ts":1900,"pred":"2.tsv","name":"x","value":"","op":">=","threshold":"1","verdict":"void"}
{"kind":"prediction","ts":1702,"pred":"3.tsv","name":"y","value":"3","op":">=","threshold":"5","verdict":"pass"}
{"kind":"prediction","ts":1800,"pred":"3.tsv","name":"y","value":"3","op":">=","threshold":"5","verdict":"fail"}
{"kind":"window","ts":2000,"event":"stop"}
not json
{"kind":"deploy","ts":1950,"binary":"alien2","chunk":64}
"#;

    #[test]
    fn ledger_knowledge_noise_and_efficiency() {
        let r = analyze(LEDGER);
        assert_eq!(r.bad_rows, 1);
        assert_eq!(r.cal_n, 2, "failed calibration excluded");
        assert!((r.cal_cv_pct - 0.7036).abs() < 1e-3, "cv={}", r.cal_cv_pct);
        assert!((r.floor_pct - 1.4072).abs() < 1e-3);
        assert!((r.pair_band_pct - 1.95 ).abs() < 0.01, "band={}", r.pair_band_pct);
        let f: BTreeMap<&str, &PredFile> = r.files.iter().map(|f| (f.pred.as_str(), f)).collect();
        assert_eq!(f["1.tsv"].status, "CONFIRMED");
        assert_eq!(f["2.tsv"].status, "VOID", "void supersedes the earlier fail");
        assert_eq!(f["3.tsv"].status, "FALSIFIED", "latest verdict wins");
        assert_eq!(r.flipped, 1, "3.tsv::y flipped pass -> fail");
        assert_eq!(r.easy, 1, "1.tsv::a passed by 11% > 3 floors");
        // 1.tsv::b passed by 0.5 points: inside the 1.41% floor -> fragile; 3.tsv::y missed by 40% -> not fragile.
        assert_eq!(r.fragile, 1);
        assert_eq!((r.jobs, r.jobs_failed), (2, 1));
        assert!((r.window_min - 1000.0 / 60.0).abs() < 1e-9);
        assert!((r.job_min_in_windows - 700.0 / 60.0).abs() < 1e-9);
        assert_eq!(r.deploys.len(), 1);
    }
}

/// Two-sided standard-normal tail P(|Z| > z) (Abramowitz-Stegun 7.1.26 erfc, |error| < 1.5e-7).
pub fn two_sided_tail(z: f64) -> f64 {
    let x = z.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let erfc = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))) * (-x * x).exp();
    erfc
}
