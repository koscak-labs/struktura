//! Detector adapters. Common contract: each detector is calibrated ONLY on
//! `calib` (never on labels), then scored on `rest` (the post-calibration
//! segment). Streaming detectors process `rest` one sample at a time with
//! bounded state; batch detectors see all of `rest` at once. Both return
//! alarm indices into `rest` (0-based) plus a mean per-sample processing
//! time in nanoseconds and a `streaming` flag.
//!
//! Alarms are counted in episodes (see `dedupe`), the same rule as
//! examples/nab_eval.rs, applied uniformly so every detector is scored under
//! the same alarm-counting rule.

use std::time::Instant;

use struktura::autopilot::{AutoPilot, Event};
use struktura::monitor::{HybridMonitor, MonitorConfig};

pub const ALARM_COOLDOWN: usize = 50;

pub struct DetResult {
    pub alarms: Vec<usize>,
    pub per_sample_ns: f64,
    pub streaming: bool,
}

pub fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n == 0 {
        0.0
    } else if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

pub fn percentile(v: &[f64], p: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n == 0 {
        return 0.0;
    }
    let idx = ((p * n as f64).ceil() as usize).saturating_sub(1).min(n - 1);
    s[idx]
}

pub fn mean_std(v: &[f64]) -> (f64, f64) {
    let n = v.len().max(1) as f64;
    let mean = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    (mean, var.sqrt())
}

/// Merge raw alarms into episodes: an alarm less than ALARM_COOLDOWN ticks
/// after the previous raw alarm belongs to the same episode, and each episode
/// counts once. Applied identically to every detector.
fn dedupe(mut alarms: Vec<usize>) -> Vec<usize> {
    alarms.sort_unstable();
    alarms.dedup();
    let mut out = Vec::new();
    let mut last: Option<usize> = None;
    for a in alarms {
        if last.is_none_or(|l| a - l >= ALARM_COOLDOWN) {
            out.push(a);
        }
        last = Some(a);
    }
    out
}

// ---------------------------------------------------------------------
// 1. struktura guard: AutoPilot + HybridMonitor, exactly as
//    examples/nab_eval.rs / cmd_guard use it.
// ---------------------------------------------------------------------
pub fn struktura_guard(calib: &[f64], rest: &[f64], quiet_drift: bool) -> Option<DetResult> {
    let config = MonitorConfig { quiet_drift, ..MonitorConfig::default() };
    let mon = HybridMonitor::calibrate_with(&[calib.to_vec()], config)?;
    let mut ap = AutoPilot::new(mon);
    let mut raw_alarms = Vec::new();
    let t0 = Instant::now();
    for (t, &v) in rest.iter().enumerate() {
        for ev in ap.push(&[v], &[true]) {
            if let Event::Alarm { .. } = ev {
                raw_alarms.push(t);
            }
        }
    }
    let elapsed = t0.elapsed();
    let per_sample_ns = elapsed.as_nanos() as f64 / rest.len().max(1) as f64;
    Some(DetResult { alarms: dedupe(raw_alarms), per_sample_ns, streaming: true })
}

// ---------------------------------------------------------------------
// 2/3. augurs-changepoint (BOCPD via NormalGammaDetector, which wraps
//    changepoint::BocpdTruncated). This satisfies both the "grafana augurs"
//    and "changepoint crate (BOCPD)" contenders honestly: they are the same
//    underlying algorithm (augurs-changepoint is a thin wrapper crate around
//    `changepoint`). Batch API (detect_changepoints on a whole slice), but
//    internally calls .step() once per sample so it is streaming-capable;
//    we run it on calib+rest concatenated and keep only alarms landing at or
//    after calib.len(), so calib plays the same "warm-up, no label use"
//    role as for the other detectors.
// ---------------------------------------------------------------------
pub fn augurs_bocpd(calib: &[f64], rest: &[f64]) -> DetResult {
    use augurs_changepoint::{Detector, NormalGammaDetector};
    let mut full = calib.to_vec();
    full.extend_from_slice(rest);
    let mut det = NormalGammaDetector::default();
    let t0 = Instant::now();
    let cps = det.detect_changepoints(&full);
    let elapsed = t0.elapsed();
    let per_sample_ns = elapsed.as_nanos() as f64 / rest.len().max(1) as f64;
    let alarms: Vec<usize> =
        cps.into_iter().filter(|&i| i >= calib.len()).map(|i| i - calib.len()).collect();
    DetResult { alarms: dedupe(alarms), per_sample_ns, streaming: false }
}

// ---------------------------------------------------------------------
// 4. anomaly_detection (ankane, STL-based). Batch, needs a seasonality
//    period — NAB series have no declared period, so we estimate
//    samples-per-day from the median timestamp spacing (same estimate
//    examples/nab_eval.rs's diagnose() uses) and fall back to the crate's
//    own doc-example period (a documented default) when that is not
//    sensible. This is a batch detector: scored on `rest` alone, no use of
//    `calib` beyond period estimation.
// ---------------------------------------------------------------------
pub fn ankane_stl(rest: &[f64], period: usize) -> DetResult {
    let data_f32: Vec<f32> = rest.iter().map(|&v| v as f32).collect();
    let period = period.min(data_f32.len() / 4).max(2);
    let t0 = Instant::now();
    let result = anomaly_detection::AnomalyDetector::fit(&data_f32, period);
    let elapsed = t0.elapsed();
    let per_sample_ns = elapsed.as_nanos() as f64 / rest.len().max(1) as f64;
    let alarms = match result {
        Ok(r) => r.anomalies().to_vec(),
        Err(_) => Vec::new(),
    };
    DetResult { alarms: dedupe(alarms), per_sample_ns, streaming: false }
}

// ---------------------------------------------------------------------
// 5. extended-isolation-forest: fit on calib (2-D features: [value,
//    first-difference]), score rest, threshold = calib scores' 99.9th
//    percentile (per task spec). Batch (fit once on calib, no online
//    update).
// ---------------------------------------------------------------------
pub fn isolation_forest(calib: &[f64], rest: &[f64]) -> DetResult {
    // Heavily quantized calibration data (few distinct values, e.g. NAB's
    // realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv with 29 distinct
    // values over 4000+ rows) was observed to make tree-building in
    // extended-isolation-forest 0.2.3 pathologically slow (minutes on a
    // few hundred points) even though recursion is nominally depth-bounded.
    // Run on a watchdog thread and skip (report as "timed out, no alarms")
    // rather than let one series stall the whole comparison.
    let calib_v = calib.to_vec();
    let rest_v = rest.to_vec();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(isolation_forest_inner(&calib_v, &rest_v));
    });
    match rx.recv_timeout(std::time::Duration::from_secs(8)) {
        Ok(res) => res,
        Err(_) => {
            eprintln!("    [isolation-forest] TIMED OUT (>8s) on a series with degenerate/quantized values — skipped, no alarms recorded");
            DetResult { alarms: Vec::new(), per_sample_ns: 0.0, streaming: false }
        }
    }
}

fn isolation_forest_inner(calib: &[f64], rest: &[f64]) -> DetResult {
    use extended_isolation_forest::{Forest, ForestOptions};

    let feats = |v: &[f64]| -> Vec<[f64; 2]> {
        let mut out = Vec::with_capacity(v.len());
        for i in 0..v.len() {
            let d = if i == 0 { 0.0 } else { v[i] - v[i - 1] };
            out.push([v[i], d]);
        }
        out
    };
    let calib_feats = feats(calib);
    let rest_feats = feats(rest);

    let options = ForestOptions {
        n_trees: 128,
        sample_size: calib_feats.len().min(256).max(8),
        max_tree_depth: None,
        extension_level: 1,
    };
    let forest = match Forest::from_slice(&calib_feats, &options) {
        Ok(f) => f,
        Err(_) => {
            return DetResult { alarms: Vec::new(), per_sample_ns: 0.0, streaming: false };
        }
    };
    let calib_scores: Vec<f64> = calib_feats.iter().map(|f| forest.score(f)).collect();
    let thr = percentile(&calib_scores, 0.999);

    let t0 = Instant::now();
    let scores: Vec<f64> = rest_feats.iter().map(|f| forest.score(f)).collect();
    let elapsed = t0.elapsed();
    let per_sample_ns = elapsed.as_nanos() as f64 / rest.len().max(1) as f64;

    let alarms: Vec<usize> =
        scores.iter().enumerate().filter(|&(_, &s)| s > thr).map(|(i, _)| i).collect();
    DetResult { alarms: dedupe(alarms), per_sample_ns, streaming: false }
}

// ---------------------------------------------------------------------
// 6. baseline: limit check. |x - calib_median| > 1.5 * calib_p95(|dev|),
//    3 consecutive exceedances to confirm; alarms merge into episodes.
//    (identical to examples/nab_eval.rs's "limit" baseline)
// ---------------------------------------------------------------------
pub fn limit_check(calib: &[f64], rest: &[f64]) -> DetResult {
    let med = median(calib);
    let dev: Vec<f64> = calib.iter().map(|v| (v - med).abs()).collect();
    let p95 = percentile(&dev, 0.95);
    let thr = 1.5 * p95.max(1e-12);

    let mut alarms = Vec::new();
    let mut streak = 0usize;
    let t0 = Instant::now();
    for (t, &v) in rest.iter().enumerate() {
        let d = (v - med).abs();
        if d > thr {
            streak += 1;
        } else {
            streak = 0;
        }
        if streak >= 3 {
            alarms.push(t);
        }
    }
    let elapsed = t0.elapsed();
    let per_sample_ns = elapsed.as_nanos() as f64 / rest.len().max(1) as f64;
    DetResult { alarms: dedupe(alarms), per_sample_ns, streaming: true }
}

// ---------------------------------------------------------------------
// 7. baseline: EWMA control chart. lambda = 0.2, L = 3 (control limits
//    from calib mean/std, steady-state sigma_z = sigma * sqrt(lambda /
//    (2 - lambda))). Alarms merge into episodes, same as the other
//    detectors here.
// ---------------------------------------------------------------------
pub fn ewma(calib: &[f64], rest: &[f64]) -> DetResult {
    const LAMBDA: f64 = 0.2;
    const L: f64 = 3.0;
    let (mean, std) = mean_std(calib);
    let sigma_z = std.max(1e-12) * (LAMBDA / (2.0 - LAMBDA)).sqrt();
    let ucl = mean + L * sigma_z;
    let lcl = mean - L * sigma_z;

    let mut z = mean;
    let mut alarms = Vec::new();
    let t0 = Instant::now();
    for (t, &v) in rest.iter().enumerate() {
        z = LAMBDA * v + (1.0 - LAMBDA) * z;
        if z > ucl || z < lcl {
            alarms.push(t);
        }
    }
    let elapsed = t0.elapsed();
    let per_sample_ns = elapsed.as_nanos() as f64 / rest.len().max(1) as f64;
    DetResult { alarms: dedupe(alarms), per_sample_ns, streaming: true }
}

// ---------------------------------------------------------------------
// 8. baseline: raw-value two-sided CUSUM. k = 0.5 * sigma_calib, h = 5 *
//    sigma_calib. Reset accumulators on alarm; alarms merge into episodes.
// ---------------------------------------------------------------------
pub fn cusum(calib: &[f64], rest: &[f64]) -> DetResult {
    let (mean, std) = mean_std(calib);
    let std = std.max(1e-12);
    let k = 0.5 * std;
    let h = 5.0 * std;

    let mut sh = 0.0f64;
    let mut sl = 0.0f64;
    let mut alarms = Vec::new();
    let t0 = Instant::now();
    for (t, &v) in rest.iter().enumerate() {
        let d = v - mean;
        sh = (sh + d - k).max(0.0);
        sl = (sl - d - k).max(0.0);
        if sh > h || sl > h {
            alarms.push(t);
            sh = 0.0;
            sl = 0.0;
        }
    }
    let elapsed = t0.elapsed();
    let per_sample_ns = elapsed.as_nanos() as f64 / rest.len().max(1) as f64;
    DetResult { alarms: dedupe(alarms), per_sample_ns, streaming: true }
}
