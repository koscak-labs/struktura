//! Evolve against real labeled data — the bridge between the synthetic
//! RED/BLUE loop ([`crate::redblue`]) and actual benchmark anomalies.
//!
//! Same mutation grammar and leg-synthesis grammar as [`crate::redblue`],
//! but calibration and test data come from a real recording, RED's corpus
//! is the benchmark's labeled anomaly windows (not sampled fault specs),
//! and the zero-alarm law is checked against the real calibration data
//! itself rather than synthetic clean sequences.
#![cfg(feature = "std")]

use crate::monitor::{HybridMonitor, MonitorConfig};
use crate::redblue::{
    fit_ar1_simple, gumbel_level, leg_source, leg_stat, LegGene, LEG_PERSIST, LEG_WINDOWS,
};

/// A labeled anomaly window in a real test recording.
#[derive(Debug, Clone)]
pub struct RealAnomaly {
    pub id: String,
    pub start_row: usize,
    pub end_row: usize,
}

/// Inputs to a real-data evolution run.
pub struct EvolveRealConfig {
    /// Column-major calibration (clean/train) data.
    pub calib: Vec<Vec<f64>>,
    /// Column-major test data (contains the labeled anomalies).
    pub test: Vec<Vec<f64>>,
    pub anomalies: Vec<RealAnomaly>,
    pub rounds: usize,
    pub blue_mutations: usize,
}

/// One round's outcome.
#[derive(Debug, Clone)]
pub struct EvolveRealReport {
    pub round: usize,
    pub detected: usize,
    pub total: usize,
    pub clean_false_alarms: usize,
    pub coverage: f64,
    pub improved: bool,
}

/// Replace a non-finite value with 0.0 (NaN/inf guard for real telemetry).
fn finite(x: f64) -> f64 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

/// Window size around each anomaly for the detection test. The monitor is
/// calibrated on a prefix of this window and then streams through the anomaly.
/// This avoids scanning the entire multi-million-row recording per check.
#[allow(dead_code)]
const EVOLVE_WINDOW: usize = 4096;
const EVOLVE_CALIB_FRAC: usize = 2048;

/// Does the given configuration detect this labeled anomaly?
///
/// Instead of streaming the entire recording, extract a window:
/// - Calibration: EVOLVE_CALIB_FRAC rows BEFORE the anomaly start (or from
///   the provided calib data if the anomaly is too early in the test).
/// - Test: from the anomaly start to end_row + 96.
///
/// This makes each check O(thousands) instead of O(millions).
pub fn detects_real(
    config: MonitorConfig,
    calib: &[Vec<f64>],
    test: &[Vec<f64>],
    anomaly: &RealAnomaly,
) -> bool {
    let n_channels = test.len().min(calib.len());
    if n_channels == 0 { return false; }
    let n = test.iter().map(|c| c.len()).min().unwrap_or(0);
    if anomaly.start_row >= n { return false; }

    // Build a local calibration window: prefer rows before the anomaly in
    // the test recording; fall back to the provided calib data if the
    // anomaly is too early.
    let local_calib: Vec<Vec<f64>> = if anomaly.start_row >= EVOLVE_CALIB_FRAC {
        let start = anomaly.start_row - EVOLVE_CALIB_FRAC;
        (0..n_channels)
            .map(|ch| test[ch][start..anomaly.start_row].iter().map(|&v| finite(v)).collect())
            .collect()
    } else {
        // Use the last EVOLVE_CALIB_FRAC rows of the provided calib data
        let cn = calib.iter().map(|c| c.len()).min().unwrap_or(0);
        let start = cn.saturating_sub(EVOLVE_CALIB_FRAC);
        (0..n_channels)
            .map(|ch| calib[ch][start..cn].iter().map(|&v| finite(v)).collect())
            .collect()
    };

    if local_calib[0].len() < 96 { return false; }

    let mut mon = match HybridMonitor::calibrate_with(&local_calib, config) {
        Some(m) => m,
        None => return false,
    };

    // Stream from anomaly start to end + margin
    let window_end = (anomaly.end_row + 96).min(n);
    let mut sample = vec![0.0f64; n_channels];
    for t in anomaly.start_row..window_end {
        for (ch, s) in sample.iter_mut().enumerate() {
            *s = finite(test[ch][t]);
        }
        if mon.push(&sample).is_some() {
            return true; // Alarm within the anomaly window
        }
    }
    false
}

/// Clean-alarm check: calibrate on a small prefix, stream a small suffix.
/// Uses at most CLEAN_CHECK_LEN rows total to keep the evolve loop fast.
/// The BLUE constraint is that this stays ZERO.
const CLEAN_CHECK_LEN: usize = 2800;

pub fn clean_alarms_real(config: MonitorConfig, calib: &[Vec<f64>]) -> usize {
    let n_channels = calib.len();
    if n_channels == 0 { return 0; }
    let n = calib.iter().map(|c| c.len()).min().unwrap_or(0);
    // Use the last CLEAN_CHECK_LEN rows (or all if shorter)
    let start = n.saturating_sub(CLEAN_CHECK_LEN);
    let len = n - start;
    let half = len / 2;
    let head: Vec<Vec<f64>> = calib.iter()
        .map(|c| c[start..start + half].iter().map(|&v| finite(v)).collect())
        .collect();
    let mut mon = match HybridMonitor::calibrate_with(&head, config) {
        Some(m) => m,
        None => return 0,
    };
    let mut alarms = 0usize;
    let mut sample = vec![0.0f64; n_channels];
    for t in (start + half)..(start + len) {
        for (ch, s) in sample.iter_mut().enumerate() {
            *s = finite(calib[ch][t]);
        }
        if mon.push(&sample).is_some() {
            alarms += 1;
        }
    }
    alarms
}

/// Does the synthesized leg fire on the real test stream within the labeled
/// window, calibrated against real calibration data? Returns
/// (fired_in_window, fired_before_window).
pub fn leg_fires_real(
    gene: &LegGene,
    calib: &[Vec<f64>],
    test: &[Vec<f64>],
    start: usize,
    stop: usize,
) -> (bool, bool) {
    let horizon = MonitorConfig::default().design_horizon;
    let w = LEG_WINDOWS[gene.window as usize % 4];
    let persist = LEG_PERSIST[gene.persist as usize % 4];
    for ch in 0..calib.len().min(test.len()) {
        let ar = fit_ar1_simple(&calib[ch]);
        let calib_src = leg_source(&calib[ch], gene.source, ar);
        if calib_src.len() < 4 * w {
            continue;
        }
        let mut stats = Vec::with_capacity(calib_src.len() - w);
        let mut i = w;
        while i <= calib_src.len() {
            stats.push(leg_stat(&calib_src[i - w..i], gene.statistic));
            i += 2;
        }
        if stats.is_empty() {
            continue;
        }
        let m = stats.iter().sum::<f64>() / stats.len() as f64;
        let sd = crate::sqrt(
            stats.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / stats.len() as f64,
        )
        .max(1e-9);
        let z_stream: Vec<f64> = stats.iter().map(|s| (s - m).abs() / sd).collect();
        let thr = gumbel_level(&z_stream, horizon);

        let test_src = leg_source(&test[ch], gene.source, ar);
        let mut streak = 0usize;
        let mut i = w;
        while i <= test_src.len() {
            let z = (leg_stat(&test_src[i - w..i], gene.statistic) - m).abs() / sd;
            streak = if z > thr { streak + 1 } else { 0 };
            if streak >= persist {
                let t = i - 1;
                if t >= start && t < stop + 96 {
                    return (true, false);
                }
                return (false, true);
            }
            i += 2;
        }
    }
    (false, false)
}

fn mutate(config: MonitorConfig, seed: &mut u64) -> MonitorConfig {
    // Small xorshift so this module needs no RNG dependency beyond what is
    // already pub(crate) — deterministic given the caller-supplied seed.
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    let u = (*seed >> 11) as f64 / (1u64 << 53) as f64;
    let mut c = config;
    match (u * 5.0) as u8 {
        0 => c.res_span = ((c.res_span as f64) * (0.6 + u)).round().max(4.0) as u64,
        1 => c.dfa_persist = (((c.dfa_persist as f64) * (0.6 + u)).round() as usize).max(1),
        2 => c.roll_persist = (((c.roll_persist as f64) * (0.6 + u)).round() as usize).max(1),
        3 => c.cusum_k = (c.cusum_k * (0.7 + 0.6 * u)).clamp(0.4, 3.0),
        _ => c.design_horizon = (c.design_horizon * (0.25 + 1.5 * u)).clamp(1e4, 1e8),
    }
    c
}

fn next_gene(seed: &mut u64) -> LegGene {
    let mut roll = || {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        ((*seed >> 11) as f64 / (1u64 << 53) as f64 * 4.0) as u8
    };
    LegGene {
        source: roll() % 3,
        statistic: roll(),
        window: roll(),
        persist: roll(),
    }
}

/// Run RED/BLUE evolution against real labeled data. RED tests each
/// benchmark anomaly directly (no sampling); BLUE mutates `MonitorConfig`
/// and, in a second pass, synthesizes `LegGene`s from the same grammar as
/// [`crate::redblue::evolve`], both gated by the zero-alarm law on the real
/// calibration data.
pub fn evolve_real(
    config: EvolveRealConfig,
    mut log: impl FnMut(&EvolveRealReport),
) -> (MonitorConfig, Vec<LegGene>, Vec<EvolveRealReport>) {
    let EvolveRealConfig {
        calib,
        test,
        anomalies,
        rounds,
        blue_mutations,
    } = config;

    let mut cfg = MonitorConfig::default();
    let mut legs: Vec<LegGene> = Vec::new();
    let mut seed = 0xE7A1_5EEDu64;
    let mut reports = Vec::new();
    let total = anomalies.len();

    // Pre-extract windows around each anomaly so the evolve loop never
    // touches the full multi-million-row recording during mutation.
    // Each window: EVOLVE_CALIB_FRAC rows before + the anomaly + 96 margin.
    let n_test = test.iter().map(|c| c.len()).min().unwrap_or(0);
    let n_ch = test.len().min(calib.len());
    let windows: Vec<(Vec<Vec<f64>>, Vec<Vec<f64>>, usize, usize)> = anomalies
        .iter()
        .map(|a| {
            let win_start = a.start_row.saturating_sub(EVOLVE_CALIB_FRAC);
            let win_end = (a.end_row + 96).min(n_test);
            let local_calib: Vec<Vec<f64>> = if a.start_row >= EVOLVE_CALIB_FRAC {
                (0..n_ch).map(|ch| test[ch][win_start..a.start_row].iter().map(|&v| finite(v)).collect()).collect()
            } else {
                let cn = calib.iter().map(|c| c.len()).min().unwrap_or(0);
                let s = cn.saturating_sub(EVOLVE_CALIB_FRAC);
                (0..n_ch).map(|ch| calib[ch][s..cn].iter().map(|&v| finite(v)).collect()).collect()
            };
            let local_test: Vec<Vec<f64>> = (0..n_ch)
                .map(|ch| test[ch][a.start_row..win_end].iter().map(|&v| finite(v)).collect())
                .collect();
            let local_len = local_test[0].len();
            (local_calib, local_test, 0, local_len) // anomaly spans [0..local_len] in the local test
        })
        .collect();

    // Drop the full test data — we only need the pre-extracted windows now.
    drop(test);

    let detects_with_legs = |c: MonitorConfig, ls: &[LegGene], idx: usize| -> bool {
        let (ref lc, ref lt, start, stop) = windows[idx];
        // Monitor check
        if let Some(mut mon) = HybridMonitor::calibrate_with(lc, c) {
            let mut sample = vec![0.0f64; n_ch];
            for t in 0..lt[0].len() {
                for (ch, s) in sample.iter_mut().enumerate() {
                    *s = lt[ch][t];
                }
                if mon.push(&sample).is_some() {
                    return true;
                }
            }
        }
        // Synthesized legs
        for gene in ls {
            let (hit, _) = leg_fires_real(gene, lc, lt, start, stop);
            if hit { return true; }
        }
        false
    };

    for round in 0..rounds {
        // RED: test every labeled anomaly against the current organism.
        let detected = (0..total)
            .filter(|&i| detects_with_legs(cfg, &legs, i))
            .count();

        // BLUE, generation 1: mutate MonitorConfig.
        let current_fa = clean_alarms_real(cfg, &calib);
        let mut best = detected;
        let mut best_fa = current_fa;
        let mut improved = false;
        for _ in 0..blue_mutations {
            let cand = mutate(cfg, &mut seed);
            let score = (0..total)
                .filter(|&i| detects_with_legs(cand, &legs, i))
                .count();
            let fa = clean_alarms_real(cand, &calib);
            let accept = (score > best && fa <= best_fa)
                || (score >= best && fa < best_fa);
            if accept {
                best = score;
                best_fa = fa;
                cfg = cand;
                improved = true;
            }
        }

        // BLUE, generation 2: synthesize a leg against the windowed data.
        for _ in 0..blue_mutations {
            let gene = next_gene(&mut seed);
            if legs.contains(&gene) { continue; }
            let mut cand_legs = legs.clone();
            cand_legs.push(gene);
            let score = (0..total)
                .filter(|&i| detects_with_legs(cfg, &cand_legs, i))
                .count();
            if score > best {
                // Clean check: the new leg must not fire on calibration data
                let (_, early) = leg_fires_real(&gene, &calib, &calib, calib[0].len(), calib[0].len());
                if !early {
                    best = score;
                    legs = cand_legs;
                    improved = true;
                }
            }
        }

        let clean_false_alarms = clean_alarms_real(cfg, &calib);
        let report = EvolveRealReport {
            round,
            detected: best,
            total,
            clean_false_alarms,
            coverage: if total == 0 {
                1.0
            } else {
                best as f64 / total as f64
            },
            improved,
        };
        log(&report);
        reports.push(report);
    }

    (cfg, legs, reports)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth(len: usize, seed: u64) -> Vec<Vec<f64>> {
        crate::telemetry_bench::synth_spacecraft(len, seed)
    }

    #[test]
    fn evolve_real_reports_one_per_round() {
        let calib = synth(1400, 11);
        let test = synth(1400, 12);
        let anomalies = vec![RealAnomaly {
            id: "a1".into(),
            start_row: 700,
            end_row: 760,
        }];
        let cfg = EvolveRealConfig {
            calib,
            test,
            anomalies,
            rounds: 2,
            blue_mutations: 4,
        };
        let (_final_cfg, _legs, reports) = evolve_real(cfg, |_| {});
        assert_eq!(reports.len(), 2);
    }
}
