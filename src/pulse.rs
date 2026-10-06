//! Production pulse: judge serving deploys from per-request timing rows,
//! with no model in the loop.
//!
//! Each row is one finished request: which server process (segment) served
//! it, the context depth, the share of its lifetime the other slot was also
//! busy, and the measured per-step milliseconds. Raw step time is dominated
//! by context depth and slot contention, so a deploy that happened to see
//! busier or deeper traffic would look slower than it is. The pulse removes
//! that confound before comparing deploys:
//!
//! 1. **Within-segment fit.** `step = level[seg] + b·ctx_k + c·busy`, with the
//!    slopes estimated only from variation *inside* each segment (fixed
//!    effects), so deploy-to-deploy level differences cannot leak into the
//!    slopes. Two passes: the second refits without rows whose robust z
//!    exceeds [`OUTLIER_Z`] (stalls must not bend the fit that detects them).
//! 2. **Deploy verdicts.** Consecutive segments are compared on their median
//!    adjusted step, as a percent of the earlier one, with a deterministic
//!    bootstrap 95% interval. The verdict needs the whole interval beyond the
//!    caller's minimum believable effect (the lab's calibration noise floor).
//! 3. **Incidents.** Rows whose adjusted step sits more than [`INCIDENT_Z`]
//!    robust standard deviations above their segment's level.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(not(feature = "std"))]
use alloc::vec;
use crate::{ln, powf, sqrt};


/// Robust z above which a row is excluded from the second fitting pass.
pub const OUTLIER_Z: f64 = 4.0;
/// Robust z above which a row is reported as an incident.
pub const INCIDENT_Z: f64 = 6.0;
/// An incident must also be this many times slower than expected, so a very
/// steady deploy (tiny robust sd) does not report harmless millisecond bumps.
pub const INCIDENT_RATIO: f64 = 1.5;
/// Segments with fewer solo rows are reported but never judged on speed.
pub const MIN_SEG_ROWS: usize = 10;
/// Contention is judged only when both deploys have this many busy rows.
pub const MIN_BUSY_ROWS: usize = 5;
/// A deploy's solo speed is judged only with at least this many solo sessions.
pub const MIN_SOLO_SESSIONS: usize = 8;
/// Contention is judged only with at least this many sessions that saw a busy neighbour.
pub const MIN_BUSY_SESSIONS: usize = 5;
/// A gap longer than this (seconds) on one server slot starts a new session.
pub const SESSION_GAP_S: u64 = 1800;
/// A request is "busy" when the other slot worked for at least this share of its lifetime.
pub const BUSY_CUT: f64 = 0.5;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Row {
    /// Server process / deploy identifier. Rows must be grouped by time;
    /// a segment is a maximal run of consecutive rows with the same id.
    pub seg: u64,
    /// Context depth in thousands of tokens.
    pub ctx_k: f64,
    /// Fraction of the request's lifetime the other slot was busy, 0..=1.
    pub busy: f64,
    /// The timing to judge, in milliseconds: per generated token (user-facing,
    /// comparable across drafter changes) or per decode step.
    pub ms: f64,
    /// Generated tokens per second (reported, not modelled).
    pub tps: f64,
    /// Mean accepted draft length (tokens per step), NaN when unknown.
    /// ms per token = ms per step / mlen, so this separates drafter or content
    /// effects from per-step hardware cost. Reported, not modelled.
    pub mlen: f64,
    /// Conversation / session id (see [`assign_sessions`]). Requests of one session are
    /// not independent: verdicts weigh sessions equally and resample whole sessions.
    pub session: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Verdict { Faster, Slower, Same, Inconclusive, Insufficient }

impl Verdict {
    pub fn as_str(&self) -> &'static str {
        match self {
            Verdict::Faster => "faster",
            Verdict::Slower => "slower",
            Verdict::Same => "same",
            Verdict::Inconclusive => "inconclusive",
            Verdict::Insufficient => "insufficient",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Segment {
    pub seg: u64,
    pub first_row: usize,
    pub n: usize,
    pub solo_n: usize,
    pub busy_n: usize,
    /// Distinct sessions among the solo / busy requests.
    pub solo_sessions: usize,
    pub busy_sessions: usize,
    /// Median over solo sessions of each session's median `ms - b·ctx_k`: the
    /// deploy's timing at zero context with an idle neighbour, one vote per session.
    pub level_ms: f64,
    /// How much slower a request runs while the other slot is busy, percent
    /// of `level_ms` (NaN when there are too few busy rows).
    pub contention_pct: f64,
    /// Median accepted draft length over solo requests (NaN when unknown).
    pub solo_mlen: f64,
    pub median_ms: f64,
    pub median_tps: f64,
    pub median_ctx_k: f64,
    pub busy_share: f64,
}

#[derive(Clone, Debug)]
pub struct Comparison {
    /// "solo": change of solo level, percent of the earlier deploy.
    /// "contention": change of contention_pct, percentage points.
    /// Negative = faster / cheaper in both.
    pub kind: &'static str,
    pub from: u64,
    pub to: u64,
    pub delta: f64,
    pub ci_low: f64,
    pub ci_high: f64,
    pub verdict: Verdict,
}

#[derive(Clone, Debug)]
pub struct Incident {
    pub row: usize,
    pub seg: u64,
    pub ms: f64,
    pub expected_ms: f64,
    pub z: f64,
}

#[derive(Clone, Debug)]
pub struct PulseReport {
    pub rows: usize,
    pub slope_ms_per_kctx: f64,
    /// The within-deploy fit came out negative (cost cannot fall with context:
    /// the traffic mix is leaking into it), so the slope was set to 0.
    pub slope_clamped: bool,
    pub segments: Vec<Segment>,
    pub comparisons: Vec<Comparison>,
    pub incidents: Vec<Incident>,
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() { return f64::NAN; }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

/// Median and MAD-based robust sd (×1.4826), floored to avoid a zero scale.
fn robust_scale(v: &[f64]) -> (f64, f64) {
    let mut a: Vec<f64> = v.to_vec();
    let m = median(&mut a);
    let mut d: Vec<f64> = v.iter().map(|x| (x - m).abs()).collect();
    let s = 1.4826 * median(&mut d);
    (m, if s > 1e-9 { s } else { 1e-9 })
}

/// Segment boundaries as half-open index ranges over `rows`.
fn segments_of(rows: &[Row]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start = 0;
    for i in 1..=rows.len() {
        if i == rows.len() || rows[i].seg != rows[start].seg {
            out.push((start, i));
            start = i;
        }
    }
    out
}

/// Least squares for the context slope over `keep` rows, fitted only from
/// variation inside each segment (fixed effects).
fn fit_ctx_slope(rows: &[Row], segs: &[(usize, usize)], keep: &[bool]) -> f64 {
    let (mut sxx, mut sxy) = (0.0, 0.0);
    for &(s, e) in segs {
        let idx: Vec<usize> = (s..e).filter(|&i| keep[i]).collect();
        if idx.len() < 2 { continue; }
        let n = idx.len() as f64;
        let mx = idx.iter().map(|&i| rows[i].ctx_k).sum::<f64>() / n;
        let my = idx.iter().map(|&i| rows[i].ms).sum::<f64>() / n;
        for &i in &idx { let x = rows[i].ctx_k - mx; sxx += x * x; sxy += x * (rows[i].ms - my); }
    }
    if sxx > 1e-12 { sxy / sxx } else { 0.0 }
}

/// Deterministic xorshift64* in [0, n).
struct Rng(u64);
impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 >> 12; self.0 ^= self.0 << 25; self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) % n as u64) as usize
    }
}

fn boot_median(v: &[f64], rng: &mut Rng, buf: &mut Vec<f64>) -> f64 {
    buf.clear();
    for _ in 0..v.len() { buf.push(v[rng.below(v.len())]); }
    median(buf)
}

fn ci(mut ds: Vec<f64>) -> (f64, f64) {
    ds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    (ds[(ds.len() as f64 * 0.025) as usize], ds[((ds.len() as f64 * 0.975) as usize).min(ds.len() - 1)])
}

fn verdict_of(lo: f64, hi: f64, min_effect: f64) -> Verdict {
    if hi < -min_effect { Verdict::Faster }
    else if lo > min_effect { Verdict::Slower }
    else if lo >= -min_effect && hi <= min_effect { Verdict::Same }
    else { Verdict::Inconclusive }
}

/// Assign session ids to requests. A session is a run of requests on one
/// server child and slot whose context keeps growing (a conversation); it ends
/// when the context drops below 90% of the previous request's (a new
/// conversation, or a compaction) or after [`SESSION_GAP_S`] idle seconds.
/// Input per request: (child id, slot, start unix s, context in K tokens);
/// output: one session id per request, in input order.
pub fn assign_sessions(keys: &[(u64, u64, u64, f64)]) -> Vec<u64> {
    let mut idx: Vec<usize> = (0..keys.len()).collect();
    idx.sort_by(|&a, &b| (keys[a].0, keys[a].1, keys[a].2).cmp(&(keys[b].0, keys[b].1, keys[b].2)));
    let mut out = vec![0u64; keys.len()];
    let mut sid = 0u64;
    let mut prev: Option<usize> = None;
    for &i in &idx {
        let new = match prev {
            None => true,
            Some(p) => keys[p].0 != keys[i].0 || keys[p].1 != keys[i].1
                || keys[i].2.saturating_sub(keys[p].2) > SESSION_GAP_S
                || keys[i].3 < 0.9 * keys[p].3,
        };
        if new { sid += 1; }
        out[i] = sid;
        prev = Some(i);
    }
    out
}

/// Per-session medians of `vals` (paired with session ids), in first-seen order.
fn session_medians(vals: &[(u64, f64)]) -> Vec<f64> {
    let mut order: Vec<u64> = Vec::new();
    let mut groups: Vec<Vec<f64>> = Vec::new();
    for &(s, v) in vals {
        match order.iter().position(|&o| o == s) {
            Some(k) => groups[k].push(v),
            None => { order.push(s); groups.push(vec![v]); }
        }
    }
    groups.iter_mut().map(|g| median(g)).collect()
}

/// Analyze timing rows. `min_effect_pct` is the smallest deploy effect the
/// caller believes (e.g. 2 × calibration CV); `n_boot` bootstrap resamples;
/// `seed` makes the intervals reproducible.
pub fn analyze(rows: &[Row], min_effect_pct: f64, n_boot: usize, seed: u64) -> PulseReport {
    let segs = segments_of(rows);
    let solo: Vec<bool> = rows.iter().map(|r| r.busy < BUSY_CUT).collect();
    let mut keep = solo.clone();
    let mut b = fit_ctx_slope(rows, &segs, &keep);
    // Pass 2: drop solo outliers by within-segment robust z, refit.
    for &(s, e) in &segs {
        let idx: Vec<usize> = (s..e).filter(|&i| solo[i]).collect();
        let adj: Vec<f64> = idx.iter().map(|&i| rows[i].ms - b * rows[i].ctx_k).collect();
        let (m, sd) = robust_scale(&adj);
        for (k, &i) in idx.iter().enumerate() { keep[i] = (adj[k] - m) / sd <= OUTLIER_Z; }
    }
    b = fit_ctx_slope(rows, &segs, &keep);
    let slope_clamped = b < 0.0;
    if slope_clamped { b = 0.0; }

    let mut segments = Vec::new();
    // Per segment: one value per SESSION (its median), so a chatty session gets one vote
    // and the bootstrap resamples sessions, not their correlated requests.
    let mut solo_adj: Vec<Vec<f64>> = Vec::new();
    let mut busy_adj: Vec<Vec<f64>> = Vec::new();
    let mut incidents = Vec::new();
    for &(s, e) in &segs {
        let adj = |i: usize| (rows[i].session, rows[i].ms - b * rows[i].ctx_k);
        let sa_rows: Vec<(u64, f64)> = (s..e).filter(|&i| keep[i]).map(adj).collect();
        let sa_rows = if sa_rows.len() >= 2 { sa_rows } else { (s..e).filter(|&i| solo[i]).map(adj).collect() };
        let ba_rows: Vec<(u64, f64)> = (s..e).filter(|&i| !solo[i]).map(adj).collect();
        let sa_flat: Vec<f64> = sa_rows.iter().map(|x| x.1).collect();
        let ba_flat: Vec<f64> = ba_rows.iter().map(|x| x.1).collect();
        let sa = session_medians(&sa_rows);
        let ba = session_medians(&ba_rows);
        // Row-level spread drives incident detection; levels are medians of session medians.
        let (_, sd_solo) = if sa_flat.is_empty() { (f64::NAN, 1e-9) } else { robust_scale(&sa_flat) };
        let (_, sd_busy) = if ba_flat.is_empty() { (f64::NAN, 1e-9) } else { robust_scale(&ba_flat) };
        let level = if sa.is_empty() { f64::NAN } else { median(&mut sa.clone()) };
        let busy_level = if ba.is_empty() { f64::NAN } else { median(&mut ba.clone()) };
        let contention_pct = if ba.len() >= MIN_BUSY_SESSIONS && sa.len() >= 2 { 100.0 * (busy_level - level) / level } else { f64::NAN };
        if e - s >= MIN_SEG_ROWS && level.is_finite() {
            for i in s..e {
                let (centre, sd) = if solo[i] || !busy_level.is_finite() { (level, sd_solo) } else { (busy_level, sd_busy) };
                let expected = centre + b * rows[i].ctx_k;
                let z = (rows[i].ms - expected) / sd;
                if z > INCIDENT_Z && rows[i].ms > INCIDENT_RATIO * expected {
                    incidents.push(Incident { row: i, seg: rows[i].seg, ms: rows[i].ms, expected_ms: expected, z });
                }
            }
        }
        let mut st: Vec<f64> = (s..e).map(|i| rows[i].ms).collect();
        let mut tp: Vec<f64> = (s..e).map(|i| rows[i].tps).collect();
        let mut cx: Vec<f64> = (s..e).map(|i| rows[i].ctx_k).collect();
        let mut ml: Vec<f64> = (s..e).filter(|&i| solo[i] && rows[i].mlen.is_finite()).map(|i| rows[i].mlen).collect();
        segments.push(Segment { seg: rows[s].seg, first_row: s, n: e - s, solo_n: sa_flat.len(), busy_n: ba_flat.len(),
            solo_sessions: sa.len(), busy_sessions: ba.len(),
            level_ms: level, contention_pct, solo_mlen: median(&mut ml), median_ms: median(&mut st), median_tps: median(&mut tp),
            median_ctx_k: median(&mut cx), busy_share: ba_flat.len() as f64 / (e - s) as f64 });
        solo_adj.push(sa);
        busy_adj.push(ba);
    }

    let mut comparisons = Vec::new();
    let mut rng = Rng(seed | 1);
    let mut buf = Vec::new();
    let boots = n_boot.max(20);
    // Solo speed: each segment against the nearest earlier segment with enough solo rows.
    let mut prev: Option<usize> = None;
    for j in 0..segments.len() {
        let enough = solo_adj[j].len() >= MIN_SOLO_SESSIONS;
        if let Some(p) = prev {
            let (from, to) = (segments[p].seg, segments[j].seg);
            if !enough {
                comparisons.push(Comparison { kind: "solo", from, to, delta: f64::NAN, ci_low: f64::NAN, ci_high: f64::NAN, verdict: Verdict::Insufficient });
                continue;
            }
            let delta = 100.0 * (segments[j].level_ms - segments[p].level_ms) / segments[p].level_ms;
            let (lo, hi) = ci((0..boots).map(|_| {
                let a = boot_median(&solo_adj[p], &mut rng, &mut buf);
                let z = boot_median(&solo_adj[j], &mut rng, &mut buf);
                100.0 * (z - a) / a
            }).collect());
            comparisons.push(Comparison { kind: "solo", from, to, delta, ci_low: lo, ci_high: hi, verdict: verdict_of(lo, hi, min_effect_pct) });
        }
        if enough { prev = Some(j); }
    }
    // Contention cost (percentage points of slowdown when the neighbour is busy).
    let has = |j: usize| solo_adj[j].len() >= MIN_SOLO_SESSIONS && busy_adj[j].len() >= MIN_BUSY_SESSIONS;
    let mut prev: Option<usize> = None;
    for j in 0..segments.len() {
        if !has(j) { continue; }
        if let Some(p) = prev {
            let delta = segments[j].contention_pct - segments[p].contention_pct;
            let (lo, hi) = ci((0..boots).map(|_| {
                let pen = |k: usize, rng: &mut Rng, buf: &mut Vec<f64>| {
                    let s = boot_median(&solo_adj[k], rng, buf);
                    100.0 * (boot_median(&busy_adj[k], rng, buf) - s) / s
                };
                let a = pen(p, &mut rng, &mut buf);
                pen(j, &mut rng, &mut buf) - a
            }).collect());
            comparisons.push(Comparison { kind: "contention", from: segments[p].seg, to: segments[j].seg, delta, ci_low: lo, ci_high: hi, verdict: verdict_of(lo, hi, min_effect_pct) });
        }
        prev = Some(j);
    }

    PulseReport { rows: rows.len(), slope_ms_per_kctx: b, slope_clamped, segments, comparisons, incidents }
}

/// One request for the model-churn view: which server child served it, when
/// it started (unix seconds), and how long its prompt took to read.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChurnRow {
    pub child: u64,
    pub start: u64,
    pub prompt_s: f64,
}

/// Cold-cache tax of model reloads: requests served within `window_s` of a
/// child's first observed request re-read their prompts from scratch.
#[derive(Clone, Debug, PartialEq)]
pub struct Churn {
    /// Server children (model instances) that served at least one request.
    pub children: usize,
    pub span_h: f64,
    pub cold_n: usize,
    pub warm_n: usize,
    pub cold_median_s: f64,
    pub warm_median_s: f64,
    pub cold_mean_s: f64,
    pub warm_mean_s: f64,
    pub cold_total_s: f64,
    /// Prompt seconds above what the same number of warm requests would cost
    /// on average (cold total - cold_n x warm mean).
    pub excess_s: f64,
}

pub fn churn(rows: &[ChurnRow], window_s: u64) -> Churn {
    let mut first: Vec<(u64, u64)> = Vec::new();
    for r in rows {
        match first.iter_mut().find(|(c, _)| *c == r.child) {
            Some(e) => { if r.start < e.1 { e.1 = r.start; } }
            None => first.push((r.child, r.start)),
        }
    }
    let (mut cold, mut warm) = (Vec::new(), Vec::new());
    for r in rows {
        let t0 = first.iter().find(|(c, _)| *c == r.child).map(|e| e.1).unwrap_or(r.start);
        if r.start.saturating_sub(t0) <= window_s { cold.push(r.prompt_s) } else { warm.push(r.prompt_s) }
    }
    let lo = rows.iter().map(|r| r.start).min().unwrap_or(0);
    let hi = rows.iter().map(|r| r.start).max().unwrap_or(0);
    let cold_total: f64 = cold.iter().sum();
    let warm_total: f64 = warm.iter().sum();
    let (cn, wn) = (cold.len(), warm.len());
    let cm = if cold.is_empty() { f64::NAN } else { median(&mut cold) };
    let wm = if warm.is_empty() { f64::NAN } else { median(&mut warm) };
    let warm_mean = if wn > 0 { warm_total / wn as f64 } else { f64::NAN };
    let cold_mean = if cn > 0 { cold_total / cn as f64 } else { f64::NAN };
    let excess = if warm_mean.is_finite() { (cold_total - cn as f64 * warm_mean).max(0.0) } else { f64::NAN };
    Churn { children: first.len(), span_h: (hi - lo) as f64 / 3600.0, cold_n: cn, warm_n: wn,
        cold_median_s: cm, warm_median_s: wm, cold_mean_s: cold_mean, warm_mean_s: warm_mean, cold_total_s: cold_total, excess_s: excess }
}

// ---------------------------------------------------------------------------
// Reload churn from the router's spawn stream.
// ---------------------------------------------------------------------------

/// One model reload: the router spawned a child for `alias` on `port`.
/// `alias` is an index into a caller-held name table.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spawn {
    pub ts: u64,
    pub pid: u64,
    pub port: u64,
    pub alias: usize,
}

/// One served request, as seen by the spawn-stream churn view.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpawnReq {
    pub pid: u64,
    pub port: u64,
    pub start: u64,
    pub prompt_s: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SpawnChurn {
    /// Every reload in the stream (also the ones that never served a request).
    pub reloads: usize,
    pub span_h: f64,
    /// (alias index, reloads), in order of first appearance.
    pub per_alias: Vec<(usize, usize)>,
    /// (hour start, reloads) for every hour that had a reload.
    pub per_hour: Vec<(u64, usize)>,
    pub worst_hour: Option<(u64, usize)>,
    /// Requests whose child's spawn is in the stream.
    pub matched: usize,
    pub cold_n: usize,
    pub warm_n: usize,
    pub cold_mean_s: f64,
    pub warm_mean_s: f64,
    pub cold_total_s: f64,
    pub total_prompt_s: f64,
    /// cold total - cold_n x warm mean, floored at 0.
    pub excess_s: f64,
    /// excess_s as a percent of all prompt-read seconds.
    pub excess_pct: f64,
}

/// Seconds since the spawn of the child that served each request: the latest
/// spawn of the same (pid, port) at or before the request's start. `None`
/// when that spawn is older than the stream (the child was already warm).
pub fn spawn_ages(spawns: &[Spawn], reqs: &[SpawnReq]) -> Vec<Option<u64>> {
    reqs.iter().map(|r| {
        spawns.iter()
            .filter(|s| s.pid == r.pid && s.port == r.port && s.ts <= r.start)
            .map(|s| s.ts).max()
            .map(|ts| r.start - ts)
    }).collect()
}

fn is_cold(age: Option<u64>, window_s: u64) -> bool { matches!(age, Some(a) if a <= window_s) }

pub fn spawn_churn(spawns: &[Spawn], reqs: &[SpawnReq], window_s: u64) -> SpawnChurn {
    let ages = spawn_ages(spawns, reqs);
    let mut per_alias: Vec<(usize, usize)> = Vec::new();
    let mut per_hour: Vec<(u64, usize)> = Vec::new();
    for s in spawns {
        match per_alias.iter_mut().find(|(a, _)| *a == s.alias) { Some(e) => e.1 += 1, None => per_alias.push((s.alias, 1)) }
        let h = s.ts / 3600 * 3600;
        match per_hour.iter_mut().find(|(t, _)| *t == h) { Some(e) => e.1 += 1, None => per_hour.push((h, 1)) }
    }
    per_hour.sort();
    let worst_hour = per_hour.iter().copied().max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0)));
    let (mut cold_n, mut warm_n, mut cold_total, mut warm_total) = (0usize, 0usize, 0.0, 0.0);
    for (r, age) in reqs.iter().zip(ages.iter()) {
        if is_cold(*age, window_s) { cold_n += 1; cold_total += r.prompt_s } else { warm_n += 1; warm_total += r.prompt_s }
    }
    let times = spawns.iter().map(|s| s.ts).chain(reqs.iter().map(|r| r.start));
    let (lo, hi) = times.fold((u64::MAX, 0u64), |(lo, hi), t| (lo.min(t), hi.max(t)));
    let warm_mean = if warm_n > 0 { warm_total / warm_n as f64 } else { f64::NAN };
    let cold_mean = if cold_n > 0 { cold_total / cold_n as f64 } else { f64::NAN };
    let excess = if warm_mean.is_finite() { (cold_total - cold_n as f64 * warm_mean).max(0.0) } else { f64::NAN };
    let total = cold_total + warm_total;
    SpawnChurn {
        reloads: spawns.len(), span_h: if hi > lo { (hi - lo) as f64 / 3600.0 } else { 0.0 },
        per_alias, per_hour, worst_hour, matched: ages.iter().filter(|a| a.is_some()).count(),
        cold_n, warm_n, cold_mean_s: cold_mean, warm_mean_s: warm_mean, cold_total_s: cold_total,
        total_prompt_s: total, excess_s: excess, excess_pct: if total > 0.0 { 100.0 * excess / total } else { f64::NAN },
    }
}

/// Reload tax before vs after a fix (e.g. slot save/restore around reloads).
#[derive(Clone, Debug, PartialEq)]
pub struct ChurnSplit {
    pub split_ts: u64,
    pub before_cold_n: usize,
    pub after_cold_n: usize,
    /// Mean extra prompt-read seconds per cold request (cold mean - warm mean).
    pub before_excess_s: f64,
    pub after_excess_s: f64,
    /// (after - before) / before, percent. Negative = the tax shrank.
    pub delta_pct: f64,
    pub ci_low: f64,
    pub ci_high: f64,
    pub verdict: Verdict,
}

/// Rows per set (cold/warm, before/after) needed to judge a split.
pub const MIN_SPLIT_ROWS: usize = 5;

fn mean(v: &[f64]) -> f64 { v.iter().sum::<f64>() / v.len() as f64 }

fn boot_mean(v: &[f64], rng: &mut Rng) -> f64 {
    let mut s = 0.0;
    for _ in 0..v.len() { s += v[rng.below(v.len())]; }
    s / v.len() as f64
}

pub fn churn_split(spawns: &[Spawn], reqs: &[SpawnReq], window_s: u64, split_ts: u64,
                   min_effect_pct: f64, n_boot: usize, seed: u64) -> ChurnSplit {
    let ages = spawn_ages(spawns, reqs);
    let (mut bc, mut bw, mut ac, mut aw) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (r, age) in reqs.iter().zip(ages.iter()) {
        let cold = is_cold(*age, window_s);
        match (r.start < split_ts, cold) {
            (true, true) => bc.push(r.prompt_s), (true, false) => bw.push(r.prompt_s),
            (false, true) => ac.push(r.prompt_s), (false, false) => aw.push(r.prompt_s),
        }
    }
    let nan = f64::NAN;
    let enough = [&bc, &bw, &ac, &aw].iter().all(|v| v.len() >= MIN_SPLIT_ROWS);
    let (be, ae) = if enough { (mean(&bc) - mean(&bw), mean(&ac) - mean(&aw)) } else { (nan, nan) };
    if !enough || be <= 0.0 {
        return ChurnSplit { split_ts, before_cold_n: bc.len(), after_cold_n: ac.len(), before_excess_s: be,
            after_excess_s: ae, delta_pct: nan, ci_low: nan, ci_high: nan, verdict: Verdict::Insufficient };
    }
    let mut rng = Rng(seed | 1);
    let (lo, hi) = ci((0..n_boot.max(100)).map(|_| {
        let b = boot_mean(&bc, &mut rng) - boot_mean(&bw, &mut rng);
        let a = boot_mean(&ac, &mut rng) - boot_mean(&aw, &mut rng);
        if b > 0.0 { 100.0 * (a - b) / b } else { f64::INFINITY }
    }).collect());
    ChurnSplit { split_ts, before_cold_n: bc.len(), after_cold_n: ac.len(), before_excess_s: be, after_excess_s: ae,
        delta_pct: 100.0 * (ae - be) / be, ci_low: lo, ci_high: hi, verdict: verdict_of(lo, hi, min_effect_pct) }
}

// ---------------------------------------------------------------------------
// Live deploy gate: an anytime-valid sequential test.
// ---------------------------------------------------------------------------

/// Betting fractions mixed into each e-process. An average of test
/// martingales is a test martingale, so the mixture needs no tuning and stays
/// valid; small fractions detect small shifts, large ones detect big shifts fast.
const LAMBDAS: [f64; 6] = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5];
/// z for the 99% order-statistic interval of the reference median.
const REF_Z: f64 = 2.576;

#[derive(Clone, Debug, PartialEq)]
pub struct WatchState {
    /// Solo requests seen on the new deploy.
    pub n: usize,
    /// Faster / Slower / Same once decided, Inconclusive while pending,
    /// Insufficient when the reference deploy is too short.
    pub verdict: Verdict,
    /// Index into the new stream where the verdict was reached.
    pub decided_at: Option<usize>,
    /// Rough requests still needed by the leading test (None: no trend yet).
    pub need_more: Option<usize>,
    pub ref_median: f64,
    pub ref_low: f64,
    pub ref_high: f64,
    /// ln e-values: [slower, faster, not slower, not faster].
    pub log_e: [f64; 4],
    /// The reference interval is wider than ±min effect, so "same" cannot be
    /// shown however many requests arrive; only faster/slower can be.
    pub same_reachable: bool,
}

fn lse(v: &[f64]) -> f64 {
    let m = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() { return m; }
    m + ln(v.iter().map(|x| powf(core::f64::consts::E, x - m)).sum::<f64>())
}

/// Decide whether a new deploy's solo timing moved by more than `min_effect_pct`
/// versus the previous deploy, re-checkable after every request.
///
/// `reference` and `stream` are (ctx_k, ms) of solo requests; `stream` in
/// arrival order. The context cost is fitted on the reference only (so it is
/// fixed before the stream arrives). Each request becomes four sign bets
/// against thresholds built from a 99% interval of the reference median,
/// so reference sampling error cannot manufacture a verdict; by Ville's
/// inequality each e-process crosses 1/α with probability <= α under its
/// null, at any stopping time. Faster and slower each use α/2. "Same" needs
/// both "not slower" and "not faster" to reach 1/α (intersection-union).
pub fn watch(reference: &[(f64, f64)], stream: &[(f64, f64)], min_effect_pct: f64, alpha: f64) -> WatchState {
    let nan = f64::NAN;
    let mut st = WatchState { n: stream.len(), verdict: Verdict::Insufficient, decided_at: None, need_more: None,
        ref_median: nan, ref_low: nan, ref_high: nan, log_e: [0.0; 4], same_reachable: false };
    if reference.len() < MIN_SEG_ROWS { return st; }
    let n = reference.len() as f64;
    let (mx, my) = (reference.iter().map(|r| r.0).sum::<f64>() / n, reference.iter().map(|r| r.1).sum::<f64>() / n);
    let (mut sxx, mut sxy) = (0.0, 0.0);
    for r in reference { sxx += (r.0 - mx) * (r.0 - mx); sxy += (r.0 - mx) * (r.1 - my); }
    let b = if sxx > 1e-12 { (sxy / sxx).max(0.0) } else { 0.0 };
    let mut adj: Vec<f64> = reference.iter().map(|r| r.1 - b * r.0).collect();
    let m0 = median(&mut adj); // sorts adj
    let k = ((n / 2.0) - REF_Z * sqrt(n) / 2.0).max(0.0) as usize; // truncation = floor for x >= 0
    let (lo, hi) = (adj[k], adj[adj.len() - 1 - k]);
    st.ref_median = m0; st.ref_low = lo; st.ref_high = hi;
    let d = min_effect_pct / 100.0;
    let (up, dn) = (hi * (1.0 + d), lo * (1.0 - d));
    let (ns, nf) = (lo * (1.0 + d), hi * (1.0 - d));
    st.same_reachable = nf < ns;
    // per-λ log wealth for each of the four bets
    let mut w = [[0.0f64; LAMBDAS.len()]; 4];
    let thr_dir = ln(2.0 / alpha);
    let thr_same = ln(1.0 / alpha);
    st.verdict = Verdict::Inconclusive;
    for (i, &(ctx, ms)) in stream.iter().enumerate() {
        let x = ms - b * ctx;
        // +1 when the bet's alternative is favoured by this request, -1 otherwise
        let sign = [
            if x > up { 1.0 } else { -1.0 },   // slower
            if x < dn { 1.0 } else { -1.0 },   // faster
            if x < ns { 1.0 } else { -1.0 },   // not slower: median below lo(1+d)
            if x > nf { 1.0 } else { -1.0 },   // not faster: median above hi(1-d)
        ];
        for t in 0..4 { for (j, l) in LAMBDAS.iter().enumerate() { w[t][j] += ln(1.0 + l * sign[t]); } }
        for t in 0..4 { st.log_e[t] = lse(&w[t]) - ln(LAMBDAS.len() as f64); }
        let v = if st.log_e[0] >= thr_dir { Verdict::Slower }
            else if st.log_e[1] >= thr_dir { Verdict::Faster }
            else if st.log_e[2] >= thr_same && st.log_e[3] >= thr_same { Verdict::Same }
            else { Verdict::Inconclusive };
        if v != Verdict::Inconclusive { st.verdict = v; st.decided_at = Some(i); break; }
    }
    if st.verdict == Verdict::Inconclusive && st.n > 0 {
        let targets = [thr_dir, thr_dir, thr_same, thr_same];
        st.need_more = (0..4).filter_map(|t| {
            let g = st.log_e[t] / st.n as f64;
            if g > 0.0 && (t < 2 || st.same_reachable) { { let q = (targets[t] - st.log_e[t]) / g; let c = q.max(1.0) as usize; Some(if (c as f64) < q { c + 1 } else { c }) } } else { None }
        }).min();
    }
    st
}

/// Which requests started near one of our OWN traffic events (a lab hot lane, an eval harness):
/// `mask[i]` is true when `starts[i]` lies in `[e - before, e + after]` for some epoch `e`.
///
/// A deploy verdict must be judged on real traffic only. Self-traffic is shaped by the maker
/// (its prompts, its timing, its concurrency), so letting it count would let the strand that
/// ships a change also certify it. The excluded rows are still useful, kept apart, as an
/// advisory "eval lane" verdict. `epochs` need not be sorted; O((n + m) log m).
pub fn near_mask(starts: &[u64], epochs: &[u64], before: u64, after: u64) -> Vec<bool> {
    let mut e: Vec<u64> = epochs.to_vec();
    e.sort_unstable();
    starts.iter().map(|&s| {
        // the first epoch >= s - after is the only candidate that can still cover s from below
        let lo = s.saturating_sub(after);
        let i = e.partition_point(|&x| x < lo);
        e.get(i).is_some_and(|&x| x <= s.saturating_add(before))
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn near_mask_window_edges() {
        // epoch 1000, window [998, 1600]
        let starts = [997, 998, 1000, 1600, 1601, 5000];
        let m = near_mask(&starts, &[1000], 2, 600);
        assert_eq!(m, vec![false, true, true, true, false, false]);
        // unsorted epochs, several windows, no epochs at all
        let m = near_mask(&[10, 500, 4990], &[5000, 0], 10, 20);
        assert_eq!(m, vec![true, false, true]);
        assert_eq!(near_mask(&[1, 2], &[], 2, 600), vec![false, false]);
        // saturating at 0 and at u64::MAX
        assert_eq!(near_mask(&[0, u64::MAX], &[1, u64::MAX], 2, 600), vec![true, true]);
    }

    /// Two deploys. Solo level 60 -> 40 ms (-33%); contention multiplies the
    /// whole timing by (1 + pen) while the neighbour is busy. The faster
    /// deploy deliberately sees deeper and busier traffic, so raw medians
    /// hide its gain. One stall is injected at row 150.
    fn synth(pen1: f64, pen2: f64) -> Vec<Row> {
        let mut rng = Rng(7);
        let mut rows = Vec::new();
        for i in 0..200 {
            let (seg, level, ctx_base, busy_p, pen) = if i < 100 { (1, 60.0, 10.0, 0.3, pen1) } else { (2, 40.0, 60.0, 0.6, pen2) };
            let ctx = ctx_base + rng.below(40) as f64;
            let busy = if (rng.below(100) as f64) < busy_p * 100.0 { 1.0 } else { 0.0 };
            let noise = rng.below(100) as f64 / 100.0 - 0.5;
            let mut ms = (level + 0.25 * ctx) * (1.0 + pen * busy) + noise;
            if i == 150 { ms += 300.0; }
            rows.push(Row { seg, ctx_k: ctx, busy, ms, tps: 1000.0 / ms, mlen: 3.0, session: i as u64 });
        }
        rows
    }

    fn kind<'a>(r: &'a PulseReport, k: &str) -> Vec<&'a Comparison> {
        r.comparisons.iter().filter(|c| c.kind == k).collect()
    }

    #[test]
    fn recovers_ctx_slope_and_solo_effect_despite_confounds() {
        let r = analyze(&synth(0.5, 0.5), 1.16, 500, 42);
        assert!((r.slope_ms_per_kctx - 0.25).abs() < 0.02, "b={}", r.slope_ms_per_kctx);
        let solo = kind(&r, "solo");
        assert_eq!(solo.len(), 1);
        assert_eq!(solo[0].verdict, Verdict::Faster, "{:?}", solo[0]);
        assert!((solo[0].delta + 33.3).abs() < 2.0, "delta={}", solo[0].delta);
        // Raw medians would hide most of the gain.
        assert!(r.segments[1].median_ms > 0.85 * r.segments[0].median_ms,
            "raw {} vs {}", r.segments[1].median_ms, r.segments[0].median_ms);
    }

    #[test]
    fn contention_is_judged_separately_from_solo_speed() {
        // Second deploy cuts the busy penalty from +100% to +20%.
        let r = analyze(&synth(1.0, 0.2), 1.16, 500, 11);
        let c = kind(&r, "contention");
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].verdict, Verdict::Faster, "{:?}", c[0]);
        assert!(r.segments[0].contention_pct > 80.0 && r.segments[1].contention_pct < 40.0,
            "{} {}", r.segments[0].contention_pct, r.segments[1].contention_pct);
        // The solo verdict is unaffected by the contention change.
        assert_eq!(kind(&r, "solo")[0].verdict, Verdict::Faster);
    }

    #[test]
    fn flags_the_stall_and_nothing_else() {
        let r = analyze(&synth(0.5, 0.5), 1.16, 200, 1);
        assert_eq!(r.incidents.len(), 1, "{:?}", r.incidents);
        assert_eq!(r.incidents[0].row, 150);
    }

    #[test]
    fn identical_deploys_are_never_judged_different() {
        let mut rng = Rng(99);
        let rows: Vec<Row> = (0..200).map(|i| {
            let ctx = 20.0 + rng.below(20) as f64;
            let ms = 50.0 + 0.25 * ctx + (rng.below(100) as f64 / 100.0 - 0.5);
            Row { seg: if i < 100 { 1 } else { 2 }, ctx_k: ctx, busy: 0.0, ms, tps: 1000.0 / ms, mlen: f64::NAN, session: i as u64 }
        }).collect();
        let r = analyze(&rows, 1.16, 500, 3);
        let v = kind(&r, "solo")[0].verdict;
        assert!(v == Verdict::Same || v == Verdict::Inconclusive, "{:?}", r.comparisons);
        assert!(kind(&r, "contention").is_empty(), "no busy rows -> no contention verdict");
    }

    #[test]
    fn deterministic_for_a_seed() {
        let a = analyze(&synth(0.5, 0.5), 1.16, 300, 9);
        let b = analyze(&synth(0.5, 0.5), 1.16, 300, 9);
        assert_eq!(a.comparisons[0].ci_low, b.comparisons[0].ci_low);
        assert_eq!(a.comparisons[0].ci_high, b.comparisons[0].ci_high);
    }

    #[test]
    fn negative_context_slope_is_clamped_and_flagged() {
        let mut rng = Rng(5);
        let rows: Vec<Row> = (0..200).map(|i| {
            let ctx = rng.below(100) as f64;
            let ms = 50.0 - 0.2 * ctx + (rng.below(100) as f64 / 100.0);
            Row { seg: if i < 100 { 1 } else { 2 }, ctx_k: ctx, busy: 0.0, ms, tps: 1000.0 / ms, mlen: f64::NAN, session: i as u64 }
        }).collect();
        let r = analyze(&rows, 1.16, 200, 4);
        assert!(r.slope_clamped);
        assert_eq!(r.slope_ms_per_kctx, 0.0);
    }

    /// Sessions with their own level (between-session sd ~15%) and tiny within-session noise.
    fn clustered(seg_sessions: &[(u64, usize, usize, f64)], rng: &mut Rng) -> Vec<Row> {
        // (segment, sessions, rows per session, level multiplier); one extra chatty session can be added by the caller.
        let mut rows = Vec::new();
        let mut sid = 0u64;
        for &(seg, n_sess, per, mult) in seg_sessions {
            for _ in 0..n_sess {
                sid += 1;
                let off = 1.0 + (rng.below(300) as f64 / 1000.0 - 0.15);
                for _ in 0..per {
                    let ms = 10.0 * mult * off * (1.0 + (rng.below(100) as f64 / 10000.0));
                    rows.push(Row { seg, ctx_k: 20.0, busy: 0.0, ms, tps: 1000.0 / ms, mlen: f64::NAN, session: sid });
                }
            }
        }
        rows
    }

    #[test]
    fn one_chatty_session_does_not_make_a_verdict() {
        // Same config twice. Deploy 2 is dominated by ONE session that happens to run 40% slower
        // (the audit's failure: request-level resampling called this SLOWER +75%).
        let mut rng = Rng(21);
        let mut rows = clustered(&[(1, 12, 6, 1.0), (2, 12, 6, 1.0)], &mut rng);
        for _ in 0..200 { rows.push(Row { seg: 2, ctx_k: 20.0, busy: 0.0, ms: 14.0, tps: 1000.0 / 14.0, mlen: f64::NAN, session: 999 }); }
        let r = analyze(&rows, 1.16, 1000, 7);
        let v = kind(&r, "solo")[0].verdict;
        assert!(v != Verdict::Slower && v != Verdict::Faster, "{:?}", r.comparisons);
        assert_eq!(r.segments[1].solo_sessions, 13);
        // The same rows treated as independent requests reproduce the false verdict.
        let flat: Vec<Row> = rows.iter().enumerate().map(|(i, r)| Row { session: 100_000 + i as u64, ..*r }).collect();
        let f = analyze(&flat, 1.16, 1000, 7);
        assert_eq!(kind(&f, "solo")[0].verdict, Verdict::Slower, "request-level resampling should be fooled: {:?}", f.comparisons);
    }

    #[test]
    fn real_effect_across_many_sessions_is_still_found() {
        let mut rng = Rng(5);
        let rows = clustered(&[(1, 20, 6, 1.0), (2, 20, 6, 0.7)], &mut rng);
        let r = analyze(&rows, 1.16, 1000, 3);
        let c = kind(&r, "solo")[0];
        assert_eq!(c.verdict, Verdict::Faster, "{:?}", c);
        assert!((c.delta + 30.0).abs() < 8.0, "{:?}", c);
    }

    #[test]
    fn too_few_sessions_is_insufficient_however_many_rows() {
        let mut rng = Rng(8);
        let rows = clustered(&[(1, 20, 6, 1.0), (2, 4, 100, 0.5)], &mut rng);
        let r = analyze(&rows, 1.16, 300, 3);
        assert_eq!(kind(&r, "solo")[0].verdict, Verdict::Insufficient, "400 rows from 4 sessions is not enough");
    }

    #[test]
    fn sessions_follow_growing_context_and_split_on_drop_gap_or_slot() {
        let keys = [(1, 0, 100, 10.0), (1, 0, 110, 12.0), (1, 0, 120, 14.0), // one conversation
                    (1, 0, 130, 2.0),                                     // context dropped: new
                    (1, 0, 130 + SESSION_GAP_S + 5, 3.0),                 // long gap: new
                    (1, 1, 111, 12.5),                                    // other slot: new
                    (2, 0, 112, 12.6)];                                   // other child: new
        let s = assign_sessions(&keys);
        assert_eq!(s[0], s[1]); assert_eq!(s[1], s[2]);
        let mut u = s.to_vec(); u.sort(); u.dedup();
        assert_eq!(u.len(), 5, "{:?}", s);
    }

    #[test]
    fn churn_counts_cold_window_and_excess() {
        let mut rows = Vec::new();
        for child in 0..3u64 {
            let t0 = 1000 + child * 3600;
            rows.push(ChurnRow { child, start: t0, prompt_s: 20.0 });        // cold: full re-read
            rows.push(ChurnRow { child, start: t0 + 60, prompt_s: 18.0 });   // still in window
            for k in 1..6 { rows.push(ChurnRow { child, start: t0 + 600 + k * 60, prompt_s: 1.0 }); }
        }
        let c = churn(&rows, 300);
        assert_eq!(c.children, 3);
        assert_eq!((c.cold_n, c.warm_n), (6, 15));
        assert_eq!(c.warm_median_s, 1.0);
        assert!((c.excess_s - (3.0 * 38.0 - 6.0)).abs() < 1e-9, "{:?}", c);
    }

    #[test]
    fn small_segment_is_insufficient_not_judged() {
        let mut rows = synth(0.5, 0.5);
        rows.truncate(105);
        let r = analyze(&rows, 1.16, 100, 5);
        assert_eq!(kind(&r, "solo")[0].verdict, Verdict::Insufficient);
        assert!(r.incidents.iter().all(|x| x.seg == 1), "no incidents judged in a 5-row deploy");
    }

    fn gauss(rng: &mut Rng) -> f64 {
        let u1 = (rng.below(1 << 30) as f64 + 0.5) / (1u64 << 30) as f64;
        let u2 = (rng.below(1 << 30) as f64 + 0.5) / (1u64 << 30) as f64;
        (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
    }

    #[test]
    fn spawn_ages_match_the_latest_spawn_of_the_same_child() {
        let spawns = [
            Spawn { ts: 100, pid: 1, port: 10, alias: 0 },
            Spawn { ts: 500, pid: 1, port: 10, alias: 0 },
            Spawn { ts: 450, pid: 1, port: 11, alias: 1 },
        ];
        let reqs = [
            SpawnReq { pid: 1, port: 10, start: 400, prompt_s: 1.0 }, // after spawn@100
            SpawnReq { pid: 1, port: 10, start: 520, prompt_s: 1.0 }, // after spawn@500
            SpawnReq { pid: 1, port: 12, start: 600, prompt_s: 1.0 }, // child never spawned in stream
            SpawnReq { pid: 2, port: 10, start: 600, prompt_s: 1.0 }, // other router
            SpawnReq { pid: 1, port: 10, start: 50, prompt_s: 1.0 },  // before any spawn of the child
        ];
        assert_eq!(spawn_ages(&spawns, &reqs), vec![Some(300), Some(20), None, None, None]);
    }

    #[test]
    fn spawn_churn_counts_every_reload_per_alias_and_hour() {
        let mut spawns = Vec::new();
        for k in 0..5u64 { spawns.push(Spawn { ts: 3600 + k * 60, pid: 1, port: 10 + k, alias: 0 }); }
        for k in 0..2u64 { spawns.push(Spawn { ts: 7200 + k * 60, pid: 1, port: 20 + k, alias: 1 }); }
        let mut reqs = Vec::new();
        reqs.push(SpawnReq { pid: 1, port: 10, start: 3600 + 10, prompt_s: 30.0 }); // cold
        reqs.push(SpawnReq { pid: 1, port: 11, start: 3660 + 100, prompt_s: 20.0 }); // cold
        for k in 0..8u64 { reqs.push(SpawnReq { pid: 1, port: 10, start: 3600 + 1000 + k, prompt_s: 2.0 }); }
        let c = spawn_churn(&spawns, &reqs, 300);
        assert_eq!(c.reloads, 7);
        assert_eq!(c.per_alias, vec![(0, 5), (1, 2)]);
        assert_eq!(c.per_hour, vec![(3600, 5), (7200, 2)]);
        assert_eq!(c.worst_hour, Some((3600, 5)));
        assert_eq!((c.cold_n, c.warm_n, c.matched), (2, 8, 10));
        assert!((c.excess_s - (50.0 - 2.0 * 2.0)).abs() < 1e-9, "{:?}", c);
        assert!((c.excess_pct - 100.0 * 46.0 / 66.0).abs() < 1e-9);
    }

    fn split_data(after_cold_scale: f64, seed: u64) -> (Vec<Spawn>, Vec<SpawnReq>) {
        let mut rng = Rng(seed);
        let (mut spawns, mut reqs) = (Vec::new(), Vec::new());
        for k in 0..40u64 {
            let ts = k * 3600;
            let port = 1000 + k;
            spawns.push(Spawn { ts, pid: 1, port, alias: 0 });
            let scale = if ts >= 20 * 3600 { after_cold_scale } else { 1.0 };
            for j in 0..3u64 { reqs.push(SpawnReq { pid: 1, port, start: ts + 10 + j * 60, prompt_s: scale * (20.0 + 4.0 * gauss(&mut rng)).max(1.0) }); }
            for j in 0..10u64 { reqs.push(SpawnReq { pid: 1, port, start: ts + 900 + j * 120, prompt_s: (2.0 + 0.5 * gauss(&mut rng)).max(0.1) }); }
        }
        (spawns, reqs)
    }

    #[test]
    fn split_judges_a_fix_that_halves_cold_reads_as_faster() {
        let (s, r) = split_data(0.5, 3);
        let v = churn_split(&s, &r, 300, 20 * 3600, 1.16, 1000, 7);
        assert_eq!(v.verdict, Verdict::Faster, "{:?}", v);
        assert!(v.delta_pct < -40.0 && v.delta_pct > -65.0, "{:?}", v);
    }

    #[test]
    fn split_with_identical_halves_is_never_a_directional_verdict() {
        // Count directional verdicts over many seeds: a 95% interval must keep them rare.
        let wrong = (1..60).filter(|&seed| {
            let (s, r) = split_data(1.0, seed);
            matches!(churn_split(&s, &r, 300, 20 * 3600, 1.16, 400, seed).verdict, Verdict::Faster | Verdict::Slower)
        }).count();
        assert!(wrong <= 6, "{} of 59 identical splits got a directional verdict", wrong);
    }

    #[test]
    fn split_without_enough_rows_is_insufficient() {
        let (s, r) = split_data(0.5, 4);
        let v = churn_split(&s, &r, 300, 10, 1.16, 200, 1); // nothing before the split
        assert_eq!(v.verdict, Verdict::Insufficient);
    }

    fn stream(rng: &mut Rng, n: usize, level: f64, sd: f64) -> Vec<(f64, f64)> {
        (0..n).map(|_| { let ctx = rng.below(100) as f64; (ctx, level * (1.0 + sd * gauss(rng)) + 0.05 * ctx) }).collect()
    }

    #[test]
    fn watch_aa_false_verdict_rate_stays_under_five_percent() {
        // No change: same distribution before and after. Re-checked after every request.
        let runs = 400;
        let mut wrong = 0;
        for seed in 1..=runs {
            let mut rng = Rng(seed * 7919);
            let reference = stream(&mut rng, 200, 10.0, 0.2);
            let new = stream(&mut rng, 600, 10.0, 0.2);
            let st = watch(&reference, &new, 1.16, 0.05);
            if matches!(st.verdict, Verdict::Faster | Verdict::Slower) { wrong += 1; }
        }
        let rate = wrong as f64 / runs as f64;
        eprintln!("A/A false-verdict rate {:.4} ({} of {} runs, 600 requests each, checked after every request)", rate, wrong, runs);
        assert!(rate <= 0.05, "A/A false-verdict rate {:.3}", rate);
    }

    #[test]
    fn watch_detects_a_real_regression_and_a_real_speedup_early() {
        let (mut slow_hits, mut fast_hits, mut slow_n) = (0, 0, 0usize);
        for seed in 1..=50u64 {
            let mut rng = Rng(seed * 104729);
            let reference = stream(&mut rng, 200, 10.0, 0.2);
            let slower = stream(&mut rng, 600, 11.5, 0.2);
            let faster = stream(&mut rng, 600, 8.5, 0.2);
            let s = watch(&reference, &slower, 1.16, 0.05);
            if s.verdict == Verdict::Slower { slow_hits += 1; slow_n += s.decided_at.unwrap() + 1; }
            if watch(&reference, &faster, 1.16, 0.05).verdict == Verdict::Faster { fast_hits += 1; }
        }
        eprintln!("power at a 15% shift: slower {}/50, faster {}/50, mean requests to a slower verdict {}", slow_hits, fast_hits, slow_n / slow_hits.max(1));
        assert!(slow_hits >= 45 && fast_hits >= 45, "slower {} faster {}", slow_hits, fast_hits);
        assert!(slow_n / slow_hits < 300, "mean requests to a verdict {}", slow_n / slow_hits);
    }

    #[test]
    fn watch_can_show_same_only_when_the_reference_is_tight() {
        let mut rng = Rng(11);
        let tight_ref = stream(&mut rng, 4000, 10.0, 0.02);
        let tight_new = stream(&mut rng, 4000, 10.0, 0.02);
        let st = watch(&tight_ref, &tight_new, 1.16, 0.05);
        assert!(st.same_reachable);
        assert_eq!(st.verdict, Verdict::Same, "{:?}", st);
        let noisy_ref = stream(&mut rng, 200, 10.0, 0.2);
        let st2 = watch(&noisy_ref, &stream(&mut rng, 300, 10.0, 0.2), 1.16, 0.05);
        assert!(!st2.same_reachable);
        assert_ne!(st2.verdict, Verdict::Same);
    }

    #[test]
    fn watch_needs_a_reference() {
        let mut rng = Rng(2);
        let st = watch(&stream(&mut rng, 5, 10.0, 0.2), &stream(&mut rng, 50, 10.0, 0.2), 1.16, 0.05);
        assert_eq!(st.verdict, Verdict::Insufficient);
    }
}
