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
