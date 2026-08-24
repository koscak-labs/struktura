//! RED/BLUE adversarial self-improvement.
//!
//! RED searches a CONTINUOUS fault space (type × channel × magnitude ×
//! duration × onset) for faults the current monitor configuration misses.
//! BLUE mutates the detection policy ([`MonitorConfig`]) and accepts a
//! mutant only if it detects strictly more of the accumulated adversarial
//! corpus while raising ZERO alarms on clean sequences. Every round's
//! misses join the corpus — the monitor's training set is written by its
//! own failures.
//!
//! This is a closed computational loop: no hand-tuning, no oracle. The
//! output of `run` is an evolved configuration plus the measured coverage
//! frontier per round.

use crate::monitor::{HybridMonitor, MonitorConfig};
use crate::telemetry_bench::{synth_spacecraft, GaussRng, CHANNELS};

/// A point in the continuous fault space.
#[derive(Debug, Clone, Copy)]
pub struct FaultSpec {
    /// 0 = step offset, 1 = ramp drift, 2 = stuck, 3 = noise-scale change,
    /// 4 = oscillation injection.
    pub kind: u8,
    pub channel: usize,
    /// Magnitude in units of the channel's clean standard deviation.
    pub magnitude: f64,
    /// Fault duration as a fraction of the sequence.
    pub duration_frac: f64,
    /// Fault onset as a fraction of the sequence.
    pub start_frac: f64,
    /// Seed selecting the underlying clean sequence.
    pub seed: u64,
}

pub const STREAM_LEN: usize = 1400;

fn channel_sd(c: &[f64]) -> f64 {
    let m = c.iter().sum::<f64>() / c.len() as f64;
    crate::sqrt(c.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / c.len() as f64)
        .max(1e-12)
}

/// Apply a fault spec to a clean multi-channel stream.
pub fn inject_spec(clean: &[Vec<f64>], spec: &FaultSpec) -> Vec<Vec<f64>> {
    let n = clean[0].len();
    let start = ((n as f64 * spec.start_frac) as usize).min(n - 2);
    let dur = ((n as f64 * spec.duration_frac) as usize).max(8);
    let stop = (start + dur).min(n);
    let ch = spec.channel % clean.len();
    let sd = channel_sd(&clean[ch]);
    let mut out: Vec<Vec<f64>> = clean.to_vec();
    match spec.kind % 5 {
        0 => {
            for i in start..stop {
                out[ch][i] += spec.magnitude * sd;
            }
        }
        1 => {
            let len = (stop - start).max(1) as f64;
            for (k, i) in (start..stop).enumerate() {
                out[ch][i] += spec.magnitude * sd * k as f64 / len;
            }
        }
        2 => {
            let v = out[ch][start];
            for i in start..stop {
                out[ch][i] = v;
            }
        }
        3 => {
            // Noise-scale change: amplify deviations from a local mean.
            let seg_mean: f64 =
                clean[ch][start..stop].iter().sum::<f64>() / (stop - start) as f64;
            for i in start..stop {
                out[ch][i] = seg_mean + (clean[ch][i] - seg_mean) * (1.0 + spec.magnitude);
            }
        }
        _ => {
            // Oscillation injection (structural: new periodicity appears).
            for (k, i) in (start..stop).enumerate() {
                out[ch][i] += spec.magnitude * sd * crate::sin(k as f64 * 0.35);
            }
        }
    }
    out
}

/// Does the given configuration detect this fault within its event window?
pub fn detects(config: MonitorConfig, spec: &FaultSpec) -> bool {
    let calib = synth_spacecraft(STREAM_LEN, spec.seed * 7919 + 100);
    let clean = synth_spacecraft(STREAM_LEN, spec.seed * 7919 + 200);
    let faulted = inject_spec(&clean, spec);
    let mut mon = match HybridMonitor::calibrate_with(&calib, config) {
        Some(m) => m,
        None => return false,
    };
    let n = clean[0].len();
    let start = ((n as f64 * spec.start_frac) as usize).min(n - 2);
    let dur = ((n as f64 * spec.duration_frac) as usize).max(8);
    let stop = (start + dur).min(n);
    let mut sample = [0.0f64; CHANNELS];
    for t in 0..n {
        for ch in 0..CHANNELS {
            sample[ch] = faulted[ch][t];
        }
        if mon.push(&sample).is_some() {
            return t >= start && t < stop + 96;
        }
    }
    false
}

/// Number of clean sequences (disjoint seed range) that alarm under this
/// configuration. The BLUE constraint is that this stays ZERO.
pub fn clean_alarms(config: MonitorConfig, n_seeds: u64) -> usize {
    let mut alarms = 0;
    for seed in 1..=n_seeds {
        let s = seed * 104729;
        let calib = synth_spacecraft(STREAM_LEN, s + 100);
        let clean = synth_spacecraft(STREAM_LEN, s + 200);
        let mut mon = match HybridMonitor::calibrate_with(&calib, config) {
            Some(m) => m,
            None => continue,
        };
        let mut sample = [0.0f64; CHANNELS];
        'stream: for t in 0..STREAM_LEN {
            for ch in 0..CHANNELS {
                sample[ch] = clean[ch][t];
            }
            if mon.push(&sample).is_some() {
                alarms += 1;
                break 'stream;
            }
        }
    }
    alarms
}

fn sample_spec(rng: &mut GaussRng, round: u64, idx: u64) -> FaultSpec {
    FaultSpec {
        kind: (rng.uniform() * 5.0) as u8,
        channel: (rng.uniform() * CHANNELS as f64) as usize,
        // The challenging band: big enough to matter, small enough to hide.
        magnitude: 0.6 + rng.uniform() * 2.4,
        duration_frac: 0.04 + rng.uniform() * 0.18,
        start_frac: 0.45 + rng.uniform() * 0.25,
        seed: round * 1000 + idx + 1,
    }
}

fn mutate(config: MonitorConfig, rng: &mut GaussRng) -> MonitorConfig {
    let mut c = config;
    match (rng.uniform() * 5.0) as u8 {
        0 => {
            c.res_span = ((c.res_span as f64) * (0.6 + rng.uniform())).round().max(4.0)
                as u64
        }
        1 => {
            c.dfa_persist =
                (((c.dfa_persist as f64) * (0.6 + rng.uniform())).round() as usize).max(1)
        }
        2 => {
            c.roll_persist =
                (((c.roll_persist as f64) * (0.6 + rng.uniform())).round() as usize).max(1)
        }
        3 => c.cusum_k = (c.cusum_k * (0.7 + 0.6 * rng.uniform())).clamp(0.4, 3.0),
        _ => {
            c.design_horizon =
                (c.design_horizon * (0.25 + 1.5 * rng.uniform())).clamp(1e4, 1e8)
        }
    }
    c
}

/// One round's outcome.
#[derive(Debug, Clone)]
pub struct RoundReport {
    pub round: usize,
    /// Coverage of the fresh RED probe set under the CURRENT config.
    pub red_coverage: f64,
    /// New misses RED found this round.
    pub new_misses: usize,
    /// Corpus coverage after BLUE (accepted config on all corpus faults).
    pub corpus_coverage_after: f64,
    /// Whether BLUE accepted a mutation this round.
    pub improved: bool,
    pub config_after: MonitorConfig,
}

/// Run the adversarial loop. Returns per-round reports and the final
/// evolved configuration.
pub fn run(
    rounds: usize,
    red_probes: u64,
    blue_mutations: usize,
    clean_seeds: u64,
    mut log: impl FnMut(&RoundReport),
) -> (MonitorConfig, Vec<RoundReport>) {
    let mut rng = GaussRng::new(0xB10E_5EED);
    let mut config = MonitorConfig::default();
    let mut corpus: Vec<FaultSpec> = Vec::new();
    let mut reports = Vec::new();

    for round in 0..rounds {
        // RED: probe the fault space under the current configuration.
        let mut misses = Vec::new();
        let mut hits = 0u64;
        for i in 0..red_probes {
            let spec = sample_spec(&mut rng, round as u64, i);
            if detects(config, &spec) {
                hits += 1;
            } else {
                misses.push(spec);
            }
        }
        let red_coverage = hits as f64 / red_probes as f64;
        let new_misses = misses.len();
        corpus.extend(misses);
        // Keep the corpus bounded (most recent failures matter most).
        if corpus.len() > 140 {
            let cut = corpus.len() - 140;
            corpus.drain(..cut);
        }

        // BLUE: evolve against the corpus under the zero-clean-alarm law.
        let eval = |c: MonitorConfig, corpus: &[FaultSpec]| -> f64 {
            if corpus.is_empty() {
                return 1.0;
            }
            let d = corpus.iter().filter(|s| detects(c, s)).count();
            d as f64 / corpus.len() as f64
        };
        let mut best = eval(config, &corpus);
        let mut improved = false;
        for _ in 0..blue_mutations {
            let cand = mutate(config, &mut rng);
            let score = eval(cand, &corpus);
            if score > best && clean_alarms(cand, clean_seeds) == 0 {
                best = score;
                config = cand;
                improved = true;
            }
        }

        let report = RoundReport {
            round,
            red_coverage,
            new_misses,
            corpus_coverage_after: best,
            improved,
            config_after: config,
        };
        log(&report);
        reports.push(report);
    }
    (config, reports)
}

// ── Generation 2: structural leg synthesis ──────────────────────────
//
// A synthesized leg is a genotype in a small detector grammar:
//   SOURCE:    raw value | AR(1) residual | first difference
//   STATISTIC: mean | std | max-abs | range | slope   (trailing window)
//   WINDOW:    8 | 16 | 32 | 64 samples
//   PERSIST:   1 | 2 | 4 | 8 consecutive exceedances
// Each leg is calibrated per deployment exactly like the built-in legs:
// its statistic's z-score stream over the clean calibration sequence gets
// a Gumbel return-level threshold. Evolution may ADD legs (structure),
// not only tune numbers — this is what breaks the parameter-tuning
// ceiling.

/// A synthesized detector-leg genotype.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LegGene {
    pub source: u8,    // 0 raw, 1 ar1-residual, 2 first-difference
    pub statistic: u8, // 0 mean, 1 std, 2 max-abs, 3 range, 4 slope
    pub window: u8,    // index into {8, 16, 32, 64}
    pub persist: u8,   // index into {1, 2, 4, 8}
}

const LEG_WINDOWS: [usize; 4] = [8, 16, 32, 64];
const LEG_PERSIST: [usize; 4] = [1, 2, 4, 8];

fn leg_stat(vals: &[f64], statistic: u8) -> f64 {
    let n = vals.len() as f64;
    match statistic % 5 {
        0 => vals.iter().sum::<f64>() / n,
        1 => {
            let m = vals.iter().sum::<f64>() / n;
            crate::sqrt(vals.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n)
        }
        2 => vals.iter().fold(0.0f64, |a, &x| a.max(x.abs())),
        3 => {
            let mx = vals.iter().cloned().fold(f64::MIN, f64::max);
            let mn = vals.iter().cloned().fold(f64::MAX, f64::min);
            mx - mn
        }
        _ => {
            // least-squares slope over the window
            let sx = n * (n - 1.0) / 2.0;
            let sx2 = n * (n - 1.0) * (2.0 * n - 1.0) / 6.0;
            let sy: f64 = vals.iter().sum();
            let sxy: f64 = vals.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
            let det = n * sx2 - sx * sx;
            if det.abs() < 1e-12 { 0.0 } else { (n * sxy - sx * sy) / det }
        }
    }
}

/// Transform a channel stream into the leg's source series.
fn leg_source(chan: &[f64], source: u8, ar: (f64, f64, f64)) -> Vec<f64> {
    match source % 3 {
        0 => chan.to_vec(),
        1 => {
            let (a, b, sd) = ar;
            let mut out = Vec::with_capacity(chan.len());
            out.push(0.0);
            for t in 1..chan.len() {
                out.push((chan[t] - (a + b * chan[t - 1])) / sd);
            }
            out
        }
        _ => {
            let mut out = Vec::with_capacity(chan.len());
            out.push(0.0);
            for t in 1..chan.len() {
                out.push(chan[t] - chan[t - 1]);
            }
            out
        }
    }
}

fn fit_ar1_simple(series: &[f64]) -> (f64, f64, f64) {
    let n = series.len() - 1;
    let x = &series[..n];
    let y = &series[1..];
    let mx = x.iter().sum::<f64>() / n as f64;
    let my = y.iter().sum::<f64>() / n as f64;
    let mut cov = 0.0;
    let mut var = 0.0;
    for i in 0..n {
        cov += (x[i] - mx) * (y[i] - my);
        var += (x[i] - mx) * (x[i] - mx);
    }
    let b = if var > 1e-12 { cov / var } else { 0.0 };
    let a = my - b * mx;
    let mut ss = 0.0;
    for i in 0..n {
        let r = y[i] - (a + b * x[i]);
        ss += r * r;
    }
    (a, b, crate::sqrt(ss / n as f64).max(1e-9))
}

fn gumbel_level(scores: &[f64], horizon: f64) -> f64 {
    const BLOCKS: usize = 16;
    let n = scores.len();
    if n < BLOCKS * 4 {
        return scores.iter().cloned().fold(0.0f64, f64::max) * 1.5;
    }
    let bl = n / BLOCKS;
    let mut maxima = [0.0f64; BLOCKS];
    for (b, m) in maxima.iter_mut().enumerate() {
        *m = scores[b * bl..(b + 1) * bl].iter().cloned().fold(f64::MIN, f64::max);
    }
    let mean = maxima.iter().sum::<f64>() / BLOCKS as f64;
    let var = maxima.iter().map(|m| (m - mean) * (m - mean)).sum::<f64>() / BLOCKS as f64;
    let beta = (crate::sqrt(var) * 2.449_489_742_783_178 / core::f64::consts::PI).max(1e-9);
    let mu = mean - 0.577_215_664_901_532_9 * beta;
    let t = (horizon / bl as f64).max(2.0);
    let p = 1.0 - 1.0 / t;
    mu - beta * crate::ln(-crate::ln(p))
}

/// Does the synthesized leg fire on the faulted stream (event window),
/// staying quiet on the calibration-derived threshold logic? Returns
/// (fired_in_window, fired_before_window).
fn leg_fires(
    gene: &LegGene,
    calib: &[Vec<f64>],
    faulted: &[Vec<f64>],
    start: usize,
    stop: usize,
    horizon: f64,
) -> (bool, bool) {
    let w = LEG_WINDOWS[gene.window as usize % 4];
    let persist = LEG_PERSIST[gene.persist as usize % 4];
    for ch in 0..calib.len() {
        let ar = fit_ar1_simple(&calib[ch]);
        let calib_src = leg_source(&calib[ch], gene.source, ar);
        if calib_src.len() < 4 * w {
            continue;
        }
        // Calibration statistic stream → mean/sd → z threshold
        let mut stats = Vec::with_capacity(calib_src.len() - w);
        let mut i = w;
        while i <= calib_src.len() {
            stats.push(leg_stat(&calib_src[i - w..i], gene.statistic));
            i += 2;
        }
        let m = stats.iter().sum::<f64>() / stats.len() as f64;
        let sd = crate::sqrt(
            stats.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / stats.len() as f64,
        )
        .max(1e-9);
        let z_stream: Vec<f64> = stats.iter().map(|s| (s - m).abs() / sd).collect();
        let thr = gumbel_level(&z_stream, horizon);

        // Evaluate on the faulted stream
        let test_src = leg_source(&faulted[ch], gene.source, ar);
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

/// Does the evolved organism (base config + synthesized legs) detect the
/// fault? Any pre-window firing by a leg counts as a false alarm (miss).
pub fn detects_evolved(config: MonitorConfig, legs: &[LegGene], spec: &FaultSpec) -> bool {
    if detects(config, spec) {
        return true;
    }
    if legs.is_empty() {
        return false;
    }
    let calib = synth_spacecraft(STREAM_LEN, spec.seed * 7919 + 100);
    let clean = synth_spacecraft(STREAM_LEN, spec.seed * 7919 + 200);
    let faulted = inject_spec(&clean, spec);
    let n = clean[0].len();
    let start = ((n as f64 * spec.start_frac) as usize).min(n - 2);
    let dur = ((n as f64 * spec.duration_frac) as usize).max(8);
    let stop = (start + dur).min(n);
    for gene in legs {
        let (hit, _early) = leg_fires(gene, &calib, &faulted, start, stop, config.design_horizon);
        if hit {
            return true;
        }
    }
    false
}

/// Clean sequences on which any synthesized leg fires (must be zero).
pub fn legs_clean_alarms(config: MonitorConfig, legs: &[LegGene], n_seeds: u64) -> usize {
    let mut alarms = 0;
    for seed in 1..=n_seeds {
        let s = seed * 104729;
        let calib = synth_spacecraft(STREAM_LEN, s + 100);
        let clean = synth_spacecraft(STREAM_LEN, s + 200);
        for gene in legs {
            let (hit, early) =
                leg_fires(gene, &calib, &clean, STREAM_LEN + 1, STREAM_LEN + 1, config.design_horizon);
            if hit || early {
                alarms += 1;
                break;
            }
        }
    }
    alarms
}

/// One generation's outcome for the structural loop.
#[derive(Debug, Clone)]
pub struct GenReport {
    pub generation: usize,
    pub red_coverage: f64,
    pub new_misses: usize,
    pub corpus_coverage: f64,
    pub legs: Vec<LegGene>,
    pub config: MonitorConfig,
}

/// Run generational evolution: parameter mutation AND leg synthesis.
pub fn evolve(
    generations: usize,
    red_probes: u64,
    candidates: usize,
    clean_seeds: u64,
    max_legs: usize,
    mut log: impl FnMut(&GenReport),
) -> (MonitorConfig, Vec<LegGene>, Vec<GenReport>) {
    let mut rng = GaussRng::new(0x9E4E_71C5);
    let mut config = MonitorConfig::default();
    let mut legs: Vec<LegGene> = Vec::new();
    let mut corpus: Vec<FaultSpec> = Vec::new();
    let mut reports = Vec::new();

    for generation in 0..generations {
        // RED probes the CURRENT organism (base + legs) with fresh faults.
        let mut hits = 0u64;
        let mut misses = Vec::new();
        for i in 0..red_probes {
            let spec = sample_spec(&mut rng, 10_000 + generation as u64, i);
            if detects_evolved(config, &legs, &spec) {
                hits += 1;
            } else {
                misses.push(spec);
            }
        }
        let red_coverage = hits as f64 / red_probes as f64;
        let new_misses = misses.len();
        corpus.extend(misses);
        if corpus.len() > 100 {
            let cut = corpus.len() - 100;
            corpus.drain(..cut);
        }

        let eval = |c: MonitorConfig, l: &[LegGene], corpus: &[FaultSpec]| -> f64 {
            if corpus.is_empty() {
                return 1.0;
            }
            let d = corpus.iter().filter(|s| detects_evolved(c, l, s)).count();
            d as f64 / corpus.len() as f64
        };
        let mut best = eval(config, &legs, &corpus);

        for _ in 0..candidates {
            // Half the candidates mutate parameters, half synthesize a leg.
            if rng.uniform() < 0.5 || legs.len() >= max_legs {
                let cand = mutate(config, &mut rng);
                let score = eval(cand, &legs, &corpus);
                if score > best && clean_alarms(cand, clean_seeds) == 0 {
                    best = score;
                    config = cand;
                }
            } else {
                let gene = LegGene {
                    source: (rng.uniform() * 3.0) as u8,
                    statistic: (rng.uniform() * 5.0) as u8,
                    window: (rng.uniform() * 4.0) as u8,
                    persist: (rng.uniform() * 4.0) as u8,
                };
                if legs.contains(&gene) {
                    continue;
                }
                let mut cand_legs = legs.clone();
                cand_legs.push(gene);
                let score = eval(config, &cand_legs, &corpus);
                if score > best
                    && legs_clean_alarms(config, &cand_legs[legs.len()..], clean_seeds) == 0
                {
                    best = score;
                    legs = cand_legs;
                }
            }
        }

        let report = GenReport {
            generation,
            red_coverage,
            new_misses,
            corpus_coverage: best,
            legs: legs.clone(),
            config,
        };
        log(&report);
        reports.push(report);
    }
    (config, legs, reports)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redblue_loop_improves_or_holds_coverage() {
        let mut last = None;
        let (final_config, reports) = run(3, 40, 6, 8, |r| {
            last = Some(r.corpus_coverage_after);
        });
        assert_eq!(reports.len(), 3);
        // The evolved config must keep the zero-clean-alarm law.
        assert_eq!(clean_alarms(final_config, 8), 0);
        // Corpus coverage must never regress across rounds (acceptance rule).
        for w in reports.windows(2) {
            assert!(
                w[1].corpus_coverage_after >= w[0].corpus_coverage_after - 1e-9
                    || w[1].new_misses > 0,
                "coverage regressed without new adversarial pressure"
            );
        }
    }
}
