//! Coupled-spacecraft telemetry benchmark — DFA vs the standard telemetry
//! fault taxonomy (packet loss, spike, stuck, drift, regime shift, mixed).
//!
//! The simulator reproduces the coupled power/thermal/wheel/pointing/payload
//! dynamics used in telemetry-assurance benchmarks: 6 channels driven by a
//! shared orbit cycle, eclipse flag, payload duty cycle, and slew schedule.
//! Faults are injected with the same parameters those benchmarks use
//! (fault start at 58% of the sequence, duration 12%, per-channel targets,
//! sigma-scaled magnitudes).
//!
//! Detection methodology: for each seed, DFA α is measured per channel on a
//! clean calibration sequence and on the faulted test sequence. The null
//! distribution comes from clean-vs-clean α shifts across seeds; a fault is
//! detected when any channel's shift exceeds that channel's 95th-percentile
//! null shift (false positive rate ≤ 5% per channel).

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::string::String;

use crate::dfa;

pub const CHANNELS: usize = 6;
pub const CHANNEL_NAMES: [&str; CHANNELS] = [
    "soc", "bus_voltage", "temp", "wheel", "pointing", "payload_current",
];

/// Simple deterministic Gaussian RNG (xorshift64* + Box-Muller).
/// Not numpy-bit-exact, but statistically equivalent dynamics.
pub struct GaussRng {
    state: u64,
    spare: Option<f64>,
}

impl GaussRng {
    pub fn new(seed: u64) -> Self {
        GaussRng { state: seed.max(1).wrapping_mul(0x9E3779B97F4A7C15), spare: None }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Uniform in (0, 1).
    pub fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// Standard normal via Box-Muller.
    pub fn normal(&mut self, mean: f64, std: f64) -> f64 {
        if let Some(z) = self.spare.take() {
            return mean + std * z;
        }
        let u1 = self.uniform();
        let u2 = self.uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * core::f64::consts::PI * u2;
        self.spare = Some(r * theta.sin());
        mean + std * r * theta.cos()
    }
}

/// Generate coupled 6-channel spacecraft telemetry.
/// Channels: [soc, bus_voltage, temp, wheel, pointing, payload_current].
pub fn synth_spacecraft(length: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = GaussRng::new(seed);

    let mut sun = vec![0.0f64; length];
    let mut eclipse = vec![0.0f64; length];
    let mut payload = vec![0.0f64; length];
    let mut slew = vec![0.0f64; length];
    let mut orbit = vec![0.0f64; length];
    for i in 0..length {
        let t = i as f64;
        orbit[i] = 2.0 * core::f64::consts::PI * t / 96.0;
        let s = orbit[i].sin();
        sun[i] = if s > 0.0 { s } else { 0.0 };
        eclipse[i] = if sun[i] < 0.08 { 1.0 } else { 0.0 };
        payload[i] = if (i / 64) % 4 == 1 { 1.0 } else { 0.0 };
        let ph = i % 120;
        slew[i] = if ph > 88 && ph < 101 { 1.0 } else { 0.0 };
    }

    let mut soc = vec![0.0f64; length];
    let mut temp = vec![0.0f64; length];
    let mut wheel = vec![0.0f64; length];
    soc[0] = 0.72;
    temp[0] = 18.0;
    wheel[0] = 2200.0;
    for i in 1..length {
        let charge = 0.006 * sun[i] - 0.0028 - 0.002 * payload[i] - 0.0012 * slew[i];
        soc[i] = (soc[i - 1] + charge + rng.normal(0.0, 0.0007)).clamp(0.2, 0.98);
        let target_temp = 13.0 + 10.0 * sun[i] + 5.0 * payload[i] + 2.0 * slew[i];
        temp[i] = temp[i - 1] + 0.075 * (target_temp - temp[i - 1]) + rng.normal(0.0, 0.12);
        let wheel_target = 2100.0 + 950.0 * slew[i] + 130.0 * (orbit[i] * 0.5).sin();
        wheel[i] = wheel[i - 1] + 0.16 * (wheel_target - wheel[i - 1]) + rng.normal(0.0, 20.0);
    }

    let mut out = vec![vec![0.0f64; length]; CHANNELS];
    for i in 0..length {
        out[0][i] = soc[i];
        out[1][i] = 26.5 + 3.4 * soc[i] - 0.25 * eclipse[i] - 0.38 * payload[i]
            + rng.normal(0.0, 0.045);
        out[2][i] = temp[i];
        out[3][i] = wheel[i];
        out[4][i] = 0.015 + 0.000025 * (wheel[i] - 2200.0).abs() + 0.11 * slew[i]
            + rng.normal(0.0, 0.004);
        out[5][i] = 0.65 + 1.9 * payload[i] + 0.22 * sun[i] + 0.35 * slew[i]
            + rng.normal(0.0, 0.045);
    }
    out
}

fn channel_std(v: &[f64]) -> f64 {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt()
}

/// The standard telemetry fault taxonomy, plus `correlation_change` — the
/// structural fault class the taxonomy is missing.
pub const FAULT_TYPES: [&str; 7] = [
    "packet_loss", "spike", "stuck", "drift", "regime_shift", "mixed",
    "correlation_change",
];

/// Inject a fault into 6-channel telemetry.
/// Fault window: start = 58% of length, duration = 12% (min 8 samples).
/// Packet-loss NaNs are forward-filled (as a value-only detector would see them).
pub fn inject_fault(clean: &[Vec<f64>], fault: &str, seed: u64) -> Vec<Vec<f64>> {
    let length = clean[0].len();
    let channels = clean.len();
    let start = (length as f64 * 0.58) as usize;
    let duration = ((length as f64 * 0.12) as usize).max(8);
    let stop = (start + duration).min(length);
    let mut observed: Vec<Vec<f64>> = clean.iter().map(|c| c.clone()).collect();
    let mut rng = GaussRng::new(seed ^ 0xFA17);

    match fault {
        "packet_loss" => {
            // every 2nd sample missing on ch1 → forward-filled
            let ch = 1;
            let mut i = start;
            while i < stop {
                observed[ch][i] = observed[ch][i.saturating_sub(1)];
                i += 2;
            }
        }
        "spike" => {
            let ch = 4.min(channels - 1);
            let spike_stop = (start + (duration / 5).max(5)).min(length);
            let scale = channel_std(&clean[ch]) + 1e-6;
            for i in start..spike_stop {
                observed[ch][i] += 6.0 * scale;
            }
        }
        "stuck" => {
            let ch = 2;
            let v = observed[ch][start];
            for i in start..stop {
                observed[ch][i] = v;
            }
        }
        "drift" => {
            let ch = 0;
            let scale = channel_std(&clean[ch]) + 1e-6;
            let n = (stop - start) as f64;
            for (k, i) in (start..stop).enumerate() {
                observed[ch][i] += 3.5 * scale * (k as f64) / (n - 1.0).max(1.0);
            }
        }
        "regime_shift" => {
            for ch in 0..channels {
                let frac = if channels > 1 { ch as f64 / (channels - 1) as f64 } else { 0.0 };
                let shift = (0.6 + 0.8 * frac) * channel_std(&clean[ch]);
                for i in start..stop {
                    observed[ch][i] += shift;
                }
            }
        }
        "mixed" => {
            let ch = 1;
            let second = start + duration / 3;
            let third = start + (2 * duration) / 3;
            let mut i = start;
            while i < second {
                observed[ch][i] = observed[ch][i.saturating_sub(1)];
                i += 2;
            }
            let spike_ch = 4.min(channels - 1);
            let sscale = channel_std(&clean[spike_ch]) + 1e-6;
            for i in second..third {
                observed[spike_ch][i] += 4.5 * sscale;
            }
            let dscale = channel_std(&clean[0]) + 1e-6;
            let n = (stop - third) as f64;
            for (k, i) in (third..stop).enumerate() {
                observed[0][i] += 3.0 * dscale * (k as f64) / (n - 1.0).max(1.0);
            }
        }
        "correlation_change" => {
            // Structural fault: same mean, same amplitude, destroyed temporal
            // correlation — the fault class the additive taxonomy can't express.
            let ch = 0;
            let seg = &clean[ch][start..stop];
            let mean = seg.iter().sum::<f64>() / seg.len() as f64;
            let std = channel_std(seg);
            for i in start..stop {
                observed[ch][i] = rng.normal(mean, std);
            }
        }
        _ => {}
    }
    observed
}

/// Per-channel DFA α for a multi-channel signal.
pub fn channel_alphas(signal: &[Vec<f64>]) -> Vec<f64> {
    signal.iter().map(|c| dfa(c).alpha).collect()
}

/// Result of the statistical telemetry benchmark for one fault type.
#[derive(Debug, Clone)]
pub struct FaultDetectResult {
    pub fault: String,
    pub detect_rate: f64,
    pub mean_max_shift: f64,
    pub best_channel: usize,
}

/// Run the full statistical benchmark.
///
/// For each seed: clean calibration (seed+100) vs faulted test (seed+200),
/// per-channel α shift. Null = clean-vs-clean shifts. Detection = any channel
/// shift > that channel's null 95th percentile.
pub fn run_benchmark(length: usize, n_seeds: u64) -> (Vec<f64>, Vec<FaultDetectResult>) {
    // Null distribution per channel
    let mut null_shifts: Vec<Vec<f64>> = vec![Vec::new(); CHANNELS];
    for seed in 1..=n_seeds {
        let s = seed * 7919;
        let calib = synth_spacecraft(length, s + 100);
        let test = synth_spacecraft(length, s + 200);
        let a_calib = channel_alphas(&calib);
        let a_test = channel_alphas(&test);
        for ch in 0..CHANNELS {
            null_shifts[ch].push((a_test[ch] - a_calib[ch]).abs());
        }
    }
    let p95: Vec<f64> = null_shifts
        .iter()
        .map(|shifts| {
            let mut s = shifts.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let idx = ((s.len() as f64 * 0.95) as usize).min(s.len() - 1);
            s[idx]
        })
        .collect();

    let mut results = Vec::new();
    for fault in FAULT_TYPES.iter() {
        let mut detections = 0usize;
        let mut max_shifts = Vec::new();
        let mut channel_hits = vec![0usize; CHANNELS];
        for seed in 1..=n_seeds {
            let s = seed * 7919;
            let calib = synth_spacecraft(length, s + 100);
            let test_clean = synth_spacecraft(length, s + 200);
            let test_faulted = inject_fault(&test_clean, fault, s);
            let a_calib = channel_alphas(&calib);
            let a_test = channel_alphas(&test_faulted);
            let mut detected = false;
            let mut max_shift = 0.0f64;
            for ch in 0..CHANNELS {
                let shift = (a_test[ch] - a_calib[ch]).abs();
                if shift > max_shift {
                    max_shift = shift;
                }
                if shift > p95[ch] {
                    detected = true;
                    channel_hits[ch] += 1;
                }
            }
            if detected {
                detections += 1;
            }
            max_shifts.push(max_shift);
        }
        let best_channel = channel_hits
            .iter()
            .enumerate()
            .max_by_key(|(_, &c)| c)
            .map(|(i, _)| i)
            .unwrap_or(0);
        results.push(FaultDetectResult {
            fault: String::from(*fault),
            detect_rate: detections as f64 / n_seeds as f64,
            mean_max_shift: max_shifts.iter().sum::<f64>() / max_shifts.len() as f64,
            best_channel,
        });
    }
    (p95, results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spacecraft_sim_produces_six_channels() {
        let sig = synth_spacecraft(700, 42);
        assert_eq!(sig.len(), CHANNELS);
        assert_eq!(sig[0].len(), 700);
        // SOC stays in physical bounds
        assert!(sig[0].iter().all(|&v| (0.2..=0.98).contains(&v)));
        // Bus voltage near 26.5-30V range
        assert!(sig[1].iter().all(|&v| (20.0..35.0).contains(&v)));
    }

    #[test]
    fn spacecraft_sim_is_deterministic() {
        let a = synth_spacecraft(300, 7);
        let b = synth_spacecraft(300, 7);
        assert_eq!(a[3], b[3]);
    }

    #[test]
    fn inject_fault_changes_only_fault_window() {
        let clean = synth_spacecraft(700, 42);
        let faulted = inject_fault(&clean, "stuck", 42);
        let start = (700.0 * 0.58) as usize;
        // before fault window: unchanged
        assert_eq!(clean[2][..start], faulted[2][..start]);
        // inside fault window: stuck at one value
        let stop = start + (700.0 * 0.12) as usize;
        assert!(faulted[2][start..stop].iter().all(|&v| v == faulted[2][start]));
    }

    #[test]
    fn correlation_change_preserves_mean() {
        let clean = synth_spacecraft(700, 42);
        let faulted = inject_fault(&clean, "correlation_change", 42);
        let start = (700.0 * 0.58) as usize;
        let stop = start + (700.0 * 0.12) as usize;
        let clean_mean: f64 =
            clean[0][start..stop].iter().sum::<f64>() / (stop - start) as f64;
        let fault_mean: f64 =
            faulted[0][start..stop].iter().sum::<f64>() / (stop - start) as f64;
        // mean preserved within half a sigma of the segment
        let std = {
            let seg = &clean[0][start..stop];
            let m = clean_mean;
            (seg.iter().map(|x| (x - m).powi(2)).sum::<f64>() / seg.len() as f64).sqrt()
        };
        assert!((clean_mean - fault_mean).abs() < 0.5 * std + 1e-9);
    }

    #[test]
    fn gauss_rng_mean_and_std() {
        let mut rng = GaussRng::new(1234);
        let samples: Vec<f64> = (0..20000).map(|_| rng.normal(5.0, 2.0)).collect();
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / samples.len() as f64;
        assert!((mean - 5.0).abs() < 0.1, "mean {mean}");
        assert!((var.sqrt() - 2.0).abs() < 0.1, "std {}", var.sqrt());
    }
}
