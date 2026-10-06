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
    /// Median of `ms - b·ctx_k` over solo requests: the deploy's timing at
    /// zero context with an idle neighbour.
    pub level_ms: f64,
    /// How much slower a request runs while the other slot is busy, percent
    /// of `level_ms` (NaN when there are too few busy rows).
    pub contention_pct: f64,
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

    let mut segments = Vec::new();
    let mut solo_adj: Vec<Vec<f64>> = Vec::new();
    let mut busy_adj: Vec<Vec<f64>> = Vec::new();
    let mut incidents = Vec::new();
    for &(s, e) in &segs {
        let sa: Vec<f64> = (s..e).filter(|&i| keep[i]).map(|i| rows[i].ms - b * rows[i].ctx_k).collect();
        let sa = if sa.len() >= 2 { sa } else { (s..e).filter(|&i| solo[i]).map(|i| rows[i].ms - b * rows[i].ctx_k).collect() };
        let ba: Vec<f64> = (s..e).filter(|&i| !solo[i]).map(|i| rows[i].ms - b * rows[i].ctx_k).collect();
        let (level, sd_solo) = if sa.is_empty() { (f64::NAN, 1e-9) } else { robust_scale(&sa) };
        let (busy_level, sd_busy) = if ba.is_empty() { (f64::NAN, 1e-9) } else { robust_scale(&ba) };
        let contention_pct = if ba.len() >= MIN_BUSY_ROWS && sa.len() >= 2 { 100.0 * (busy_level - level) / level } else { f64::NAN };
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
        segments.push(Segment { seg: rows[s].seg, first_row: s, n: e - s, solo_n: sa.len(), busy_n: ba.len(),
            level_ms: level, contention_pct, median_ms: median(&mut st), median_tps: median(&mut tp),
            median_ctx_k: median(&mut cx), busy_share: ba.len() as f64 / (e - s) as f64 });
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
        let enough = solo_adj[j].len() >= MIN_SEG_ROWS;
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
    let has = |j: usize| solo_adj[j].len() >= MIN_SEG_ROWS && busy_adj[j].len() >= MIN_BUSY_ROWS;
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

    PulseReport { rows: rows.len(), slope_ms_per_kctx: b, segments, comparisons, incidents }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            rows.push(Row { seg, ctx_k: ctx, busy, ms, tps: 1000.0 / ms });
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
            Row { seg: if i < 100 { 1 } else { 2 }, ctx_k: ctx, busy: 0.0, ms, tps: 1000.0 / ms }
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
    fn small_segment_is_insufficient_not_judged() {
        let mut rows = synth(0.5, 0.5);
        rows.truncate(105);
        let r = analyze(&rows, 1.16, 100, 5);
        assert_eq!(kind(&r, "solo")[0].verdict, Verdict::Insufficient);
        assert!(r.incidents.iter().all(|x| x.seg == 1), "no incidents judged in a 5-row deploy");
    }
}
