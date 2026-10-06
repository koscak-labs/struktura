//! Track: the brain's track record. Is the mind getting smarter, measured
//! against the simplest honest rival?
//!
//! **Prequential replay** ([`replay`], [`summarize`]). The ledger is replayed
//! exactly as [`Mind::from_lab_goal`] learns it (same order, same situations,
//! same rewards: the same [`Mind::learn_prediction`] steps). Before a batch of
//! predictions is learned, the mind forecasts each one: its yield for that
//! row's situation, i.e. the reward it expects to learn. A batch is a scoring
//! burst: consecutive rows less than `gap_s` seconds apart (one job's
//! predictions are scored together, so none of them is known before the
//! others). The baseline forecast is the running mean of the rewards learned
//! before the batch. Both arms see only earlier batches. Squared error on the
//! reward scale (Brier for 0/1 rewards); skill = 1 - mind / baseline.
//!
//! What both arms share as fixed definitions rather than data: the reward
//! scale (the noise band and floor from the ledger's calibration rows, and
//! the goal registry when given), exactly as the Mind defines its reward.
//!
//! **Live forecasts** ([`resolve`]). `loop` tags each brain estimate with a
//! stable [`forecast_id`]; a forecast row is scored once, by the first later
//! scored prediction of that challenger's experiment.

#![cfg(feature = "std")]

use crate::lab::LabReport;
use super::knobs::Knob;
use super::mind::{learning_order, GoalFeature, Mind};

/// Rows scored within this many seconds of the previous one belong to one scoring burst.
pub const BATCH_GAP_S: f64 = 30.0;

/// One prediction of the replay, in the mind's learning order.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub pred: String,
    pub name: String,
    pub ts: f64,
    /// Scoring burst this row belongs to (0 = first).
    pub batch: usize,
    /// The mind's yield for this row, before its batch was learned.
    pub forecast: f64,
    pub abstained: bool,
    /// Running mean of the rewards learned before this row's batch (NaN in the first batch).
    pub baseline: f64,
    /// The reward the mind then learned.
    pub reward: f64,
    /// Sequential forecast and baseline: just before this row, with earlier rows of its own
    /// batch already learned (the Mind's raw order; optimistic, kept for comparison only).
    pub seq_forecast: f64,
    pub seq_baseline: f64,
}

/// Replay the ledger as the mind learns it, recording each row's forecast before it is learned.
pub fn replay(lab: &LabReport, knobs: &[Knob], goal: &[GoalFeature], gap_s: f64) -> Vec<Row> {
    let order = learning_order(lab);
    let mut m = Mind::blank(lab, knobs, goal);
    let (mut sum, mut n) = (0.0f64, 0usize);
    let mut rows = Vec::with_capacity(order.len());
    let (mut i, mut batch) = (0usize, 0usize);
    while i < order.len() {
        let mut j = i + 1;
        while j < order.len() && order[j].ts - order[j - 1].ts < gap_s { j += 1; }
        // Strict forecasts for the whole burst, before any of it is learned.
        let strict: Vec<(f64, bool)> = order[i..j].iter()
            .map(|p| { let y = m.estimate(&m.situation_of_prediction(p, knobs)); (y.expected, y.abstained) }).collect();
        let base = if n > 0 { sum / n as f64 } else { f64::NAN };
        for (k, p) in order[i..j].iter().enumerate() {
            let seq = m.estimate(&m.situation_of_prediction(p, knobs)).expected;
            let seq_base = if n > 0 { sum / n as f64 } else { f64::NAN };
            let r = m.learn_prediction(p, knobs, goal);
            rows.push(Row { pred: p.pred.clone(), name: p.name.clone(), ts: p.ts, batch, forecast: strict[k].0, abstained: strict[k].1,
                baseline: base, reward: r, seq_forecast: seq, seq_baseline: seq_base });
            sum += r; n += 1;
        }
        i = j; batch += 1;
    }
    rows
}

#[derive(Clone, Debug, PartialEq)]
pub struct Window { pub first: usize, pub last: usize, pub n: usize, pub mind: f64, pub base: f64, pub skill: f64 }

#[derive(Clone, Debug, PartialEq)]
pub struct Bin { pub lo: f64, pub hi: f64, pub n: usize, pub forecast: f64, pub realised: f64 }

#[derive(Clone, Debug)]
pub struct Track {
    pub rows: usize,
    pub batches: usize,
    /// Rows outside the first batch (both arms have data); the rest are warm-up.
    pub scored: usize,
    /// Mean squared error of the mind's and the baseline's forecasts over scored rows.
    pub mind: f64,
    pub base: f64,
    pub skill: f64,
    /// Batch-level paired z of (baseline error - mind error): rows of one job are correlated,
    /// so each scoring burst is one observation. Positive = mind better.
    pub z: f64,
    pub seq_mind: f64,
    pub seq_base: f64,
    pub seq_skill: f64,
    pub windows: Vec<Window>,
    pub bins: Vec<Bin>,
    pub verdict: String,
}

fn mean(v: impl Iterator<Item = f64>) -> f64 {
    let (mut s, mut n) = (0.0, 0usize);
    for x in v { s += x; n += 1; }
    if n == 0 { f64::NAN } else { s / n as f64 }
}

fn skill(mind: f64, base: f64) -> f64 { if base > 0.0 { 1.0 - mind / base } else { f64::NAN } }

/// Score a replay: overall errors and skill, a learning curve in windows of `window` scored rows,
/// and calibration of the mind's forecasts in `bins` equal-width bins over [0, 1].
pub fn summarize(rows: &[Row], window: usize, bins: usize) -> Track {
    let sc: Vec<&Row> = rows.iter().filter(|r| r.batch > 0).collect();
    let se = |f: f64, r: f64| (f - r) * (f - r);
    let mind = mean(sc.iter().map(|r| se(r.forecast, r.reward)));
    let base = mean(sc.iter().map(|r| se(r.baseline, r.reward)));
    let seq_mind = mean(sc.iter().map(|r| se(r.seq_forecast, r.reward)));
    let seq_base = mean(sc.iter().map(|r| se(r.seq_baseline, r.reward)));
    // Per-burst summed loss differences.
    let mut d: Vec<f64> = Vec::new();
    let mut last = usize::MAX;
    for r in &sc {
        if r.batch != last { d.push(0.0); last = r.batch; }
        *d.last_mut().unwrap() += se(r.baseline, r.reward) - se(r.forecast, r.reward);
    }
    let z = if d.len() >= 2 {
        let m = mean(d.iter().cloned());
        let sd = (d.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (d.len() - 1) as f64).sqrt();
        if sd > 0.0 { m / (sd / (d.len() as f64).sqrt()) } else if m == 0.0 { 0.0 } else { m.signum() * f64::INFINITY }
    } else { f64::NAN };
    let w = window.max(1);
    let windows = sc.chunks(w).enumerate().map(|(k, c)| {
        let (mi, ba) = (mean(c.iter().map(|r| se(r.forecast, r.reward))), mean(c.iter().map(|r| se(r.baseline, r.reward))));
        Window { first: k * w + 1, last: k * w + c.len(), n: c.len(), mind: mi, base: ba, skill: skill(mi, ba) }
    }).collect();
    let nb = bins.max(1);
    let bins = (0..nb).filter_map(|b| {
        let (lo, hi) = (b as f64 / nb as f64, (b + 1) as f64 / nb as f64);
        let inb: Vec<&&Row> = sc.iter().filter(|r| r.forecast >= lo && (r.forecast < hi || (b + 1 == nb && r.forecast <= hi))).collect();
        if inb.is_empty() { return None; }
        Some(Bin { lo, hi, n: inb.len(), forecast: mean(inb.iter().map(|r| r.forecast)), realised: mean(inb.iter().map(|r| r.reward)) })
    }).collect();
    let s = skill(mind, base);
    let verdict = if sc.is_empty() {
        "nothing scored yet: the ledger needs scored predictions in at least two scoring bursts".to_string()
    } else if !(s > 0.0) {
        format!("the mind does NOT beat the running-mean baseline (skill {:+.3}): its yield forecasts are no better than the average past reward", s)
    } else if z >= 1.96 {
        format!("the mind beats the running-mean baseline (skill {:+.3}, burst-level z {:.2})", s, z)
    } else {
        format!("the mind is ahead of the running-mean baseline (skill {:+.3}) but not significantly (burst-level z {:.2} < 1.96): not yet evidence that it learns", s, z)
    };
    Track { rows: rows.len(), batches: rows.last().map(|r| r.batch + 1).unwrap_or(0), scored: sc.len(), mind, base, skill: s, z,
        seq_mind, seq_base, seq_skill: skill(seq_mind, seq_base), windows, bins, verdict }
}

// ------------------------------------------------------------ live forecasts ----

/// Rows of a ledger as the lab counts them (non-empty lines; see [`crate::lab::analyze`]).
pub fn ledger_rows(ledger: &str) -> usize { ledger.lines().filter(|l| !l.trim().is_empty()).count() }

/// Stable id of a brain forecast: the challenger (`knob=value`), the metric its test measures, and
/// the ledger row count it was made from. Re-running `loop` on an unchanged ledger repeats the id.
pub fn forecast_id(challenger: &str, metric: &str, ledger_rows: usize) -> String {
    super::memory::decision_id(&["forecast", challenger, metric, &ledger_rows.to_string()])
}

/// A forecast row: `{"kind":"forecast","id","challenger","yield","ts"[,"ledger_rows"]}`.
#[derive(Clone, Debug, PartialEq)]
pub struct Forecast { pub id: String, pub challenger: String, pub yield_: f64, pub ts: f64, pub ledger_rows: Option<usize> }

/// A score row: the forecast, the reward the mind learned from the evidence that resolved it, and the error.
#[derive(Clone, Debug, PartialEq)]
pub struct Score {
    pub id: String,
    pub challenger: String,
    pub yield_: f64,
    pub outcome: f64,
    pub sq_err: f64,
    /// `pred::name` of the resolving prediction row.
    pub resolved_by: String,
    pub verdict: String,
    pub ts: f64,
    /// 0-based row of the resolving prediction in the ledger.
    pub ledger_row: usize,
    /// The job row of the challenger's experiment, if the ledger has one.
    pub job: Option<String>,
}

impl Score {
    pub fn to_json(&self) -> String {
        use super::memory::esc;
        format!("{{\"kind\":\"score\",\"id\":\"{}\",\"challenger\":\"{}\",\"yield\":{:.4},\"outcome\":{:.4},\"sq_err\":{:.6},\"resolved_by\":\"{}\",\"verdict\":\"{}\",\"ts\":{},\"ledger_row\":{},\"job\":{}}}",
            esc(&self.id), esc(&self.challenger), self.yield_, self.outcome, self.sq_err, esc(&self.resolved_by), esc(&self.verdict), self.ts, self.ledger_row,
            self.job.as_ref().map(|j| format!("\"{}\"", esc(j))).unwrap_or_else(|| "null".into()))
    }
}

#[derive(Clone, Debug, Default)]
pub struct Resolution {
    /// New scores, in forecast order.
    pub scores: Vec<Score>,
    /// Forecasts with no resolving evidence yet.
    pub pending: Vec<Forecast>,
    /// Forecasts skipped because the fuxi ledger already has a score row for their id.
    pub already_scored: usize,
    /// Repeated forecast ids (a loop re-run on an unchanged ledger); scored once.
    pub duplicates: usize,
    /// Forecast rows without a usable id, challenger or yield.
    pub invalid: usize,
    /// Squared errors of every score row: existing ones in the fuxi ledger plus the new ones.
    pub all_sq_err: Vec<f64>,
}

/// `(job number, tag)` of a job or pred file named `[lab-]NNN-<tag>[.sh|.tsv]`; `(NNN, "")` for `NNN.tsv`.
pub fn job_tag(name: &str) -> Option<(String, String)> {
    let s = name.rsplit('/').next().unwrap_or(name);
    let s = s.strip_suffix(".sh").or_else(|| s.strip_suffix(".tsv")).unwrap_or(s);
    let s = s.strip_prefix("lab-").unwrap_or(s);
    let (num, tag) = match s.split_once('-') { Some((n, t)) => (n, t), None => (s, "") };
    if num.is_empty() || !num.bytes().all(|b| b.is_ascii_digit()) { return None; }
    Some((num.to_string(), tag.to_string()))
}

/// Score each forecast in `fuxi` (not already scored there) by the first later scored prediction of its
/// challenger's experiment in `ledger`. Naming (see `design`): the loop names a challenger's job
/// `lab-NNN-<knob><value>`; it is queued as `NNN-<knob><value>.sh` (the ledger's job row), and lab-score
/// writes its prediction rows with `pred` = the pred file (`lab-NNN-<knob><value>.tsv`) and the name
/// `<knob><value>-beats-...`. A prediction resolves a challenger when its pred file carries the tag, when
/// it is `NNN.tsv` of a job or lab-q `queued` row carrying the tag, or when its name starts with
/// `<knob><value>-beats-`. It must be scored (pass, fail or void; `missing` is not evidence) after the
/// forecast: `ts` later than the forecast's, at or beyond the forecast's `ledger_rows` when given, and
/// from a job that started after the forecast (when the job row has a start). The outcome is the reward
/// the mind learns for that verdict, on the noise band known when it was written (calibration rows up
/// to it) and aimed at the goal registry when one is given.
pub fn resolve(fuxi: &str, ledger: &str, goal: &[GoalFeature]) -> Resolution {
    use crate::lab::parse_json;
    let mut res = Resolution::default();
    let mut scored: Vec<String> = Vec::new();
    let mut forecasts: Vec<Forecast> = Vec::new();
    for line in fuxi.lines().filter(|l| !l.trim().is_empty()) {
        let Some(j) = parse_json(line) else { continue };
        match j.str("kind") {
            Some("score") => {
                if let Some(id) = j.str("id") { scored.push(id.to_string()); }
                if let Some(e) = j.num("sq_err") { res.all_sq_err.push(e); }
            }
            Some("forecast") => {
                let (id, ch, y) = (j.str("id"), j.str("challenger"), j.num("yield"));
                match (id, ch, y) {
                    (Some(id), Some(ch), Some(y)) if !id.is_empty() && ch.contains('=') && y.is_finite() => forecasts.push(Forecast {
                        id: id.into(), challenger: ch.into(), yield_: y, ts: j.num("ts").unwrap_or(0.0),
                        ledger_rows: j.num("ledger_rows").filter(|n| *n >= 0.0).map(|n| n as usize) }),
                    _ => res.invalid += 1,
                }
            }
            _ => {}
        }
    }
    let lines: Vec<&str> = ledger.lines().filter(|l| !l.trim().is_empty()).collect();
    let rows: Vec<Option<crate::lab::Json>> = lines.iter().map(|l| parse_json(l)).collect();
    let mut seen: Vec<String> = Vec::new();
    for f in forecasts {
        if seen.contains(&f.id) { res.duplicates += 1; continue; }
        seen.push(f.id.clone());
        if scored.contains(&f.id) { res.already_scored += 1; continue; }
        let (knob, value) = f.challenger.split_once('=').unwrap();
        let tag = format!("{}{}", knob.trim(), value.trim());
        // This challenger's jobs: the ledger's job rows (`NNN-<tag>.sh`, with their start) and lab-q's queued rows (`NNN-<tag>`).
        let jobs: Vec<(String, String, Option<f64>)> = rows.iter().flatten().filter(|j| matches!(j.str("kind"), Some("job") | Some("queued")))
            .filter_map(|j| { let name = j.str("job")?; let (n, t) = job_tag(name)?; if t == tag { Some((n, name.to_string(), j.num("start"))) } else { None } }).collect();
        let hit = rows.iter().enumerate().find_map(|(i, j)| {
            let j = j.as_ref()?;
            if j.str("kind") != Some("prediction") { return None; }
            let (pred, name, verdict) = (j.str("pred").unwrap_or(""), j.str("name").unwrap_or(""), j.str("verdict").unwrap_or(""));
            if !matches!(verdict, "pass" | "fail" | "void") { return None; }
            let ts = j.num("ts").unwrap_or(0.0);
            if ts <= f.ts || f.ledger_rows.map(|n| i < n).unwrap_or(false) { return None; }
            let by_file = job_tag(pred).map(|(n, t)| t == tag || (t.is_empty() && jobs.iter().any(|(jn, _, _)| *jn == n))).unwrap_or(false);
            let by_name = name.starts_with(&format!("{}-beats-", tag));
            if !(by_file || by_name) { return None; }
            // Only a forecast made before its experiment started counts for that experiment.
            let num = job_tag(pred).map(|(n, _)| n);
            if jobs.iter().any(|(n, _, s)| Some(n) == num.as_ref() && s.map(|s| f.ts >= s).unwrap_or(false)) { return None; }
            Some((i, j))
        });
        let Some((i, j)) = hit else { res.pending.push(f); continue };
        let lab = crate::lab::analyze(&lines[..=i].join("\n"));
        let band = lab.pair_band_pct.max(lab.floor_pct);
        let (pred, name, verdict) = (j.str("pred").unwrap_or(""), j.str("name").unwrap_or(""), j.str("verdict").unwrap_or(""));
        let margin = crate::lab::margin(j.str("op").unwrap_or(""), j.str("value").unwrap_or(""), j.str("threshold").unwrap_or(""));
        let outcome = super::mind::goal_reward(goal, pred, name, super::mind::reward(verdict, margin, band, lab.floor_pct, crate::lab::band_applies(name, j.str("value").unwrap_or(""))));
        let num = job_tag(pred).map(|(n, _)| n);
        let job = jobs.iter().find(|(n, _, _)| Some(n) == num.as_ref()).map(|(_, name, _)| name.clone());
        let sq = (f.yield_ - outcome) * (f.yield_ - outcome);
        res.all_sq_err.push(sq);
        res.scores.push(Score { id: f.id, challenger: f.challenger, yield_: f.yield_, outcome, sq_err: sq,
            resolved_by: format!("{}::{}", pred, name), verdict: verdict.into(), ts: j.num("ts").unwrap_or(0.0), ledger_row: i, job });
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ouroboros::knobs::{parse_knobs, BUILTIN};

    const LEARNABLE: &str = include_str!("../../tests/fixtures/track/learnable.jsonl");
    const RESOLVE_LEDGER: &str = include_str!("../../tests/fixtures/track/resolve-ledger.jsonl");
    const FUXI: &str = include_str!("../../tests/fixtures/track/fuxi.jsonl");

    fn run(ledger: &str) -> Vec<Row> { replay(&crate::lab::analyze(ledger), &parse_knobs(BUILTIN).unwrap(), &[], BATCH_GAP_S) }

    /// Same forecast, bit for bit (NaN baselines of the warm-up batch compare equal).
    fn same(a: &Row, b: &Row) -> bool {
        let eq = |x: f64, y: f64| x.to_bits() == y.to_bits();
        a.pred == b.pred && a.name == b.name && a.batch == b.batch && a.abstained == b.abstained
            && eq(a.forecast, b.forecast) && eq(a.baseline, b.baseline)
    }

    #[test]
    fn replay_learns_exactly_what_the_mind_learns() {
        let lab = crate::lab::analyze(LEARNABLE);
        let knobs = parse_knobs(BUILTIN).unwrap();
        let rows = replay(&lab, &knobs, &[], BATCH_GAP_S);
        let mind = Mind::from_lab_goal(&lab, &knobs, &[]);
        assert_eq!(rows.len(), mind.episodes);
        assert_eq!(rows.len(), 40);
        let blank = Mind::blank(&lab, &knobs, &[]);
        for (r, p) in rows.iter().zip(learning_order(&lab)) {
            assert_eq!((r.pred.as_str(), r.name.as_str()), (p.pred.as_str(), p.name.as_str()));
            assert_eq!(r.reward, blank.reward_of(p, &[]));
        }
        assert_eq!(rows.iter().map(|r| r.batch).max(), Some(19), "two rows 1 s apart are one scoring burst");
        assert!(rows[0].abstained && rows[0].baseline.is_nan(), "first burst: neither arm has data");
        assert!(!rows[2].abstained && (rows[2].baseline - 0.5).abs() < 1e-12, "{:?}", rows[2]);
    }

    #[test]
    fn no_leakage_forecasts_use_only_earlier_bursts() {
        let full = run(LEARNABLE);
        // (a) The future does not exist yet: cut the ledger after burst 10 (2 cal rows + 10 x 3 rows).
        let cut: String = LEARNABLE.lines().take(32).map(|l| format!("{}\n", l)).collect();
        let part = run(&cut);
        assert_eq!(part.len(), 20);
        for (a, b) in part.iter().zip(&full) { assert!(same(a, b), "{:?} vs {:?}", a, b); }
        // (b) Outcomes of burst 10 change (pre-registered thresholds kept): its own forecasts and
        // baselines and everything before stay identical; its rewards and later forecasts move.
        let flipped = LEARNABLE.replace("\"name\":\"tps-ambitious-10\",\"value\":\"108 vs 100 (8.00%", "\"name\":\"tps-ambitious-10\",\"value\":\"105.5 vs 100 (5.50%")
            .replace("\"name\":\"tps-timid-10\",\"value\":\"101.5 vs 100 (1.50%", "\"name\":\"tps-timid-10\",\"value\":\"104 vs 100 (4.00%");
        assert_ne!(flipped, LEARNABLE);
        let f = run(&flipped);
        for i in 0..20 { assert!(same(&f[i], &full[i]), "row {}: {:?} vs {:?}", i, f[i], full[i]); }
        assert!((full[18].reward, full[19].reward) == (1.0, 0.0) && (f[18].reward, f[19].reward) == (0.0, 1.0));
        assert!((20..40).any(|i| f[i].forecast != full[i].forecast), "later forecasts must see the changed outcomes");
        // (c) Inside one burst: the outcome of its first row never reaches the strict forecast of its second
        // (the sequential, optimistic forecast does see it).
        let one = LEARNABLE.replace("\"name\":\"tps-ambitious-10\",\"value\":\"108 vs 100 (8.00%", "\"name\":\"tps-ambitious-10\",\"value\":\"105.5 vs 100 (5.50%");
        let o = run(&one);
        assert!(same(&o[19], &full[19]));
        assert_ne!(o[19].seq_forecast, full[19].seq_forecast);
    }

    #[test]
    fn the_mind_beats_the_baseline_in_a_learnable_world() {
        let t = summarize(&run(LEARNABLE), 10, 5);
        std::println!("synthetic learnable: mind {:.4} baseline {:.4} skill {:+.3} z {:.2}; {}", t.mind, t.base, t.skill, t.z, t.verdict);
        assert_eq!((t.rows, t.batches, t.scored), (40, 20, 38));
        assert!(t.skill > 0.5 && t.z > 1.96, "{:?}", t);
        assert!(t.verdict.starts_with("the mind beats"), "{}", t.verdict);
        assert_eq!(t.windows.iter().map(|w| w.n).sum::<usize>(), t.scored);
        assert_eq!(t.bins.iter().map(|b| b.n).sum::<usize>(), t.scored);
        assert!(t.windows.last().unwrap().mind < t.windows[0].mind, "it gets better: {:?}", t.windows);
        // Its record earned it trust: it keeps (nearly) all of its own deviation from the base rate.
        let m = Mind::from_lab_goal(&crate::lab::analyze(LEARNABLE), &parse_knobs(BUILTIN).unwrap(), &[]);
        std::println!("synthetic learnable: trust {:.3}, base rate {:.3}", m.trust(), m.base_rate().unwrap());
        assert!(m.trust() > 0.8, "trust {}", m.trust());
    }

    fn row(batch: usize, forecast: f64, baseline: f64, reward: f64) -> Row {
        Row { pred: "x.tsv".into(), name: format!("r{}", batch), ts: batch as f64 * 100.0, batch, forecast, abstained: false, baseline, reward,
            seq_forecast: forecast, seq_baseline: baseline }
    }

    #[test]
    fn says_plainly_when_the_mind_does_not_beat_the_baseline() {
        let mut rows = vec![row(0, 0.0, f64::NAN, 1.0)];
        for b in 1..8 { rows.push(row(b, if b % 2 == 0 { 0.9 } else { 0.1 }, 0.5, if b % 2 == 0 { 0.0 } else { 1.0 })); }
        let t = summarize(&rows, 3, 4);
        assert!(t.skill < 0.0 && t.verdict.contains("does NOT beat"), "{}", t.verdict);
        // Ahead, but on too little evidence.
        let rows = vec![row(0, 0.0, f64::NAN, 1.0), row(1, 0.7, 0.5, 1.0), row(2, 0.6, 0.5, 0.0), row(3, 0.7, 0.5, 1.0)];
        let t = summarize(&rows, 3, 4);
        assert!(t.verdict.contains("not significantly"), "{} (skill {})", t.verdict, t.skill);
        let t = summarize(&rows[..1], 3, 4);
        assert!(t.scored == 0 && t.verdict.starts_with("nothing scored"), "{}", t.verdict);
    }

    #[test]
    fn job_and_pred_file_names() {
        assert_eq!(job_tag("lab-300-ub128.tsv"), Some(("300".into(), "ub128".into())));
        assert_eq!(job_tag("300-ub128.sh"), Some(("300".into(), "ub128".into())));
        assert_eq!(job_tag("302.tsv"), Some(("302".into(), "".into())));
        assert_eq!(job_tag("101-retro.tsv"), Some(("101".into(), "retro".into())));
        assert_eq!(job_tag("lab-pred/lab-305-draft5.tsv"), Some(("305".into(), "draft5".into())));
        assert_eq!(job_tag("x-ub128.sh"), None);
    }

    #[test]
    fn forecast_ids_are_stable() {
        let a = forecast_id("ub=128", "d100000", 558);
        assert_eq!(a.len(), 16);
        assert_eq!(a, forecast_id("ub=128", "d100000", 558));
        assert_ne!(a, forecast_id("ub=128", "d100000", 559), "a grown ledger is a new forecast");
        assert_ne!(a, forecast_id("ub=128", "d0", 558));
        assert_ne!(a, forecast_id("ub=1280", "d100000", 558));
        assert_eq!(ledger_rows("{}\n\n{}\n"), 2);
    }

    #[test]
    fn resolves_forecasts_by_the_first_later_evidence_once() {
        let r = resolve(FUXI, RESOLVE_LEDGER, &[]);
        let ids: Vec<&str> = r.scores.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ids, vec!["a1", "b2", "j10"]);
        let a = &r.scores[0];
        // ts-500 evidence predates the forecast; the `missing` row is not evidence; margin 3.8% (band 1.95%) is decisive.
        assert_eq!((a.resolved_by.as_str(), a.verdict.as_str(), a.ledger_row, a.job.as_deref()),
            ("lab-301-ub128.tsv::ub128-beats-ub256-d100000", "pass", 5, Some("301-ub128.sh")));
        assert!((a.outcome - 1.0).abs() < 1e-12 && (a.sq_err - 0.16).abs() < 1e-12, "{:?}", a);
        let b = &r.scores[1];
        // Old-style `NNN.tsv` pred file, tied to the challenger by its job row; margin 0.5% inside the band: fragile.
        assert_eq!((b.resolved_by.as_str(), b.job.as_deref()), ("302.tsv::code-tps-check", Some("302-draft5.sh")));
        assert!(b.outcome == 0.0 && (b.sq_err - 0.0625).abs() < 1e-12);
        let pending: Vec<&str> = r.pending.iter().map(|f| f.id.as_str()).collect();
        assert_eq!(pending, vec!["d4", "f6", "g7", "i9"], "no evidence / ub1280 is not ub128 / evidence inside ledger_rows / made after its job started");
        let j = &r.scores[2];
        // Pred file `306.tsv` tied to ub=512 by lab-q's queued row; margin 15% > 3 floors: an easy pass.
        assert_eq!((j.resolved_by.as_str(), j.job.as_deref(), j.outcome, j.sq_err), ("306.tsv::ub-check", Some("306-ub512"), 0.5, 0.0));
        assert_eq!((r.already_scored, r.duplicates, r.invalid), (1, 1, 1));
        assert_eq!(r.all_sq_err.len(), 4);
        // Appending the scores makes the next run score nothing new.
        let mut fuxi = FUXI.to_string();
        for s in &r.scores { fuxi += &s.to_json(); fuxi.push('\n'); }
        assert!(crate::lab::parse_json(&r.scores[0].to_json()).is_some());
        let again = resolve(&fuxi, RESOLVE_LEDGER, &[]);
        assert!(again.scores.is_empty() && again.already_scored == 4 && again.pending.len() == 4);
        // Aimed at the goal: a prediction that proves no registered feature counts 0.6.
        let goal = vec![GoalFeature { feature: "other".into(), pred: "999.tsv".into(), required: vec![] }];
        let g = resolve(FUXI, RESOLVE_LEDGER, &goal);
        assert!((g.scores[0].outcome - 0.6).abs() < 1e-12);
    }
}
