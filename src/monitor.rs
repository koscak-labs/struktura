//! Flight-grade streaming hybrid health monitor.
//!
//! Design constraints follow flight-software practice (cFS/F Prime style):
//! - **No heap allocation after initialization.** All runtime state lives in
//!   fixed-size ring buffers and scalars; the DFA scratch buffer is allocated
//!   once at init and reused.
//! - **Bounded, deterministic per-sample cost.** The worst-case path is one
//!   DFA evaluation over a fixed `WINDOW`-sample ring (every `DFA_STRIDE`
//!   samples); every loop is bounded by compile-time constants.
//! - **Self-calibrating.** All thresholds are learned from a clean
//!   calibration stream; no magic numbers tuned per deployment.
//!
//! Four orthogonal detector legs, OR-fused (measured on the coupled
//! spacecraft benchmark: 6/7 fault-taxonomy coverage at 100% event
//! detection each, 6.7% event false-alarm rate — see `telemetry_bench`):
//! 1. AR(1) residual — spikes, steps, correlation loss (instant)
//! 2. repeated-value run — stuck sensors (calibrated per-channel limit)
//! 3. windowed DFA — slow structural drift (integrative)
//! 4. rolling-mean level shift — regime changes (cancels the dominant
//!    periodic driver when `ROLL` matches its period)

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::dfa_into;

/// DFA ring window length. Must exceed 64 (below that `dfa` cannot form a
/// valid box-size range) — 96 gives box sizes 16..24.
pub const WINDOW: usize = 96;
/// Recompute the DFA leg every this many samples.
pub const DFA_STRIDE: usize = 2;
/// Rolling-mean window; set to the dominant periodic driver's period.
pub const ROLL: usize = 96;
/// Extra repeated samples tolerated beyond the calibration maximum run.
const REPEAT_MARGIN: usize = 4;
/// Residual exceedances within `RES_SPAN` samples required to alarm.
const RES_HITS: usize = 2;
const RES_SPAN: usize = 20;
/// Consecutive DFA exceedances required to alarm.
const DFA_PERSIST: usize = 5;
/// Consecutive rolling-mean exceedances required to alarm.
const ROLL_PERSIST: usize = 10;
/// Design horizon: each leg's threshold is set for one expected false alarm
/// per this many samples (before persistence requirements, which reduce the
/// realized rate further).
pub const DESIGN_HORIZON: f64 = 1_000_000.0;
/// CUSUM slack per step (in residual standard deviations). Must exceed the
/// residual-mean transfer bias between calibration and operation (AR(1)
/// coefficients fitted on one noise realization leave a small nonzero
/// residual mean on another); a slow drift contributes ~2σ per step, so
/// k = 1.0 absorbs transfer bias without hiding drift.
const CUSUM_K: f64 = 1.0;

/// Which detector leg raised the alarm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Leg {
    Residual,
    RepeatedValue,
    Dfa,
    LevelShift,
    /// Two-sided CUSUM over standardized AR(1) residuals — catches slow
    /// drift: a ramp adds a constant offset to every one-step residual,
    /// invisible per step but unbounded cumulatively. CUSUM on residuals
    /// works where CUSUM on raw values fails, because residuals carry no
    /// periodic (orbital) structure.
    ResidualCusum,
}

/// Per-channel calibration statistics (immutable after calibration).
#[derive(Debug, Clone)]
struct ChannelCalib {
    ar_a: f64,
    ar_b: f64,
    ar_sd: f64,
    alpha_mean: f64,
    alpha_sd: f64,
    mean: f64,
    roll_thr: f64,
    max_run: usize,
    /// A channel whose clean calibration already shows repeated runs longer
    /// than `REPEAT_MARGIN` saturates legitimately (clamp, quantization);
    /// run-length is then not evidence of a stuck sensor, and no run-length
    /// threshold learned from a finite calibration extrapolates. The leg is
    /// disabled for such channels.
    repeat_enabled: bool,
}

/// Per-channel runtime state (fixed size, mutated every sample).
#[derive(Debug, Clone)]
struct ChannelState {
    ring: [f64; WINDOW],
    roll_ring: [f64; ROLL],
    prev: f64,
    run: usize,
    cusum_pos: f64,
    cusum_neg: f64,
}

/// Streaming hybrid monitor over `n_channels` telemetry channels.
///
/// Feed one sample per channel per tick via [`HybridMonitor::push`].
/// Returns `Some(Leg)` on the tick an alarm is raised.
pub struct HybridMonitor {
    calib: Vec<ChannelCalib>,
    state: Vec<ChannelState>,
    res_thr: f64,
    dfa_thr: f64,
    cusum_thr: f64,
    res_hit_times: [u64; RES_HITS],
    dfa_streak: usize,
    roll_streak: usize,
    t: u64,
    scratch: Vec<f64>,
    alarmed: bool,
}

/// Extreme-value (Gumbel) return-level threshold.
///
/// Max-over-calibration thresholds do not extrapolate: a stream 100x longer
/// than calibration will exceed the calibration maximum by chance. Instead,
/// split the calibration score stream into blocks, take block maxima, fit a
/// Gumbel distribution by moments (location μ, scale β = σ√6/π,
/// μ = m − γβ), and return the level expected to be exceeded once per
/// `horizon` samples. This is how flight monitors express "false alarms per
/// mission hour" as a design parameter.
fn gumbel_return_level(scores: &[f64], horizon: f64) -> f64 {
    const BLOCKS: usize = 16;
    let n = scores.len();
    if n < BLOCKS * 4 {
        // Not enough data for block maxima — fall back to max with margin.
        return scores.iter().cloned().fold(0.0f64, f64::max) * 1.5;
    }
    let block_len = n / BLOCKS;
    let mut maxima = [0.0f64; BLOCKS];
    for (b, m) in maxima.iter_mut().enumerate() {
        let start = b * block_len;
        *m = scores[start..start + block_len]
            .iter()
            .cloned()
            .fold(f64::MIN, f64::max);
    }
    let mean = maxima.iter().sum::<f64>() / BLOCKS as f64;
    let var = maxima.iter().map(|m| (m - mean).powi(2)).sum::<f64>() / BLOCKS as f64;
    let beta = (var.sqrt() * 2.449_489_742_783_178 / core::f64::consts::PI).max(1e-9); // σ√6/π
    let mu = mean - 0.577_215_664_901_532_9 * beta;
    // Return period in blocks for one expected exceedance per `horizon` samples
    let t = (horizon / block_len as f64).max(2.0);
    // Gumbel quantile at exceedance probability 1/T: x = μ − β ln(−ln(1 − 1/T))
    let p = 1.0 - 1.0 / t;
    mu - beta * (-(p.ln())).ln()
}

fn alloc_zeroed(n: usize) -> Vec<f64> {
    let mut v = Vec::with_capacity(n);
    v.resize(n, 0.0);
    v
}

fn fit_ar1(series: &[f64]) -> (f64, f64, f64) {
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
    (a, b, (ss / n as f64).sqrt().max(1e-9))
}

impl HybridMonitor {
    /// Calibrate from a clean multi-channel sequence (one inner slice per
    /// channel, all the same length, at least `2 * WINDOW` samples).
    ///
    /// This is the only phase that allocates.
    pub fn calibrate(clean: &[Vec<f64>]) -> Option<HybridMonitor> {
        let channels = clean.len();
        if channels == 0 {
            return None;
        }
        let length = clean[0].len();
        if length < 2 * WINDOW || length <= ROLL {
            return None;
        }

        let mut scratch: Vec<f64> = Vec::with_capacity(WINDOW);
        let mut calib = Vec::with_capacity(channels);
        let mut per_channel_alphas: Vec<Vec<f64>> = Vec::with_capacity(channels);

        for c in clean.iter() {
            let (ar_a, ar_b, ar_sd) = fit_ar1(c);

            // Windowed alpha statistics over the calibration stream
            let mut alphas = Vec::new();
            let mut end = WINDOW;
            while end <= length {
                alphas.push(dfa_into(&c[end - WINDOW..end], &mut scratch).alpha);
                end += DFA_STRIDE;
            }
            let na = alphas.len() as f64;
            let alpha_mean = alphas.iter().sum::<f64>() / na;
            let alpha_var =
                alphas.iter().map(|a| (a - alpha_mean).powi(2)).sum::<f64>() / na;

            let mean = c.iter().sum::<f64>() / length as f64;
            let mut roll_devs = Vec::with_capacity(length - ROLL);
            let mut sum = 0.0f64;
            for (t, &v) in c.iter().enumerate() {
                sum += v;
                if t >= ROLL {
                    sum -= c[t - ROLL];
                    roll_devs.push((sum / ROLL as f64 - mean).abs());
                }
            }
            let roll_thr = gumbel_return_level(&roll_devs, DESIGN_HORIZON);

            let mut max_run = 1usize;
            let mut run = 1usize;
            for t in 1..length {
                if c[t] == c[t - 1] {
                    run += 1;
                    if run > max_run {
                        max_run = run;
                    }
                } else {
                    run = 1;
                }
            }

            per_channel_alphas.push(alphas);
            calib.push(ChannelCalib {
                ar_a,
                ar_b,
                ar_sd,
                alpha_mean,
                alpha_sd: alpha_var.sqrt().max(1e-6),
                mean,
                roll_thr: roll_thr.max(1e-9),
                max_run,
                repeat_enabled: max_run <= REPEAT_MARGIN,
            });
        }

        // Residual threshold: Gumbel return level of the max-over-channels
        // standardized-residual stream at the design horizon.
        let mut res_scores = Vec::with_capacity(length - 1);
        for t in 1..length {
            let mut mz = 0.0f64;
            for (ch, c) in clean.iter().enumerate() {
                let cc = &calib[ch];
                let z = (c[t] - (cc.ar_a + cc.ar_b * c[t - 1])).abs() / cc.ar_sd;
                if z > mz {
                    mz = z;
                }
            }
            res_scores.push(mz);
        }
        let res_thr = gumbel_return_level(&res_scores, DESIGN_HORIZON);

        // Residual-CUSUM threshold: run the two-sided CUSUM (slack k = 0.5)
        // over each channel's SIGNED standardized calibration residuals,
        // collect the max-over-channels CUSUM path, Gumbel return level.
        let mut cusum_path = Vec::with_capacity(length - 1);
        {
            let mut pos = alloc_zeroed(channels);
            let mut neg = alloc_zeroed(channels);
            for t in 1..length {
                let mut mc = 0.0f64;
                for (ch, c) in clean.iter().enumerate() {
                    let cc = &calib[ch];
                    let z = (c[t] - (cc.ar_a + cc.ar_b * c[t - 1])) / cc.ar_sd;
                    pos[ch] = (pos[ch] + z - CUSUM_K).max(0.0);
                    neg[ch] = (neg[ch] - z - CUSUM_K).max(0.0);
                    let m = pos[ch].max(neg[ch]);
                    if m > mc {
                        mc = m;
                    }
                }
                cusum_path.push(mc);
            }
        }
        let cusum_thr = gumbel_return_level(&cusum_path, DESIGN_HORIZON);

        // DFA threshold: Gumbel return level of the max-over-channels
        // windowed-alpha z stream (horizon scaled by the stride).
        let n_alpha = per_channel_alphas[0].len();
        let mut dfa_scores = Vec::with_capacity(n_alpha);
        for w in 0..n_alpha {
            let mut mz = 0.0f64;
            for (ch, alphas) in per_channel_alphas.iter().enumerate() {
                let cc = &calib[ch];
                let z = (alphas[w] - cc.alpha_mean).abs() / cc.alpha_sd;
                if z > mz {
                    mz = z;
                }
            }
            dfa_scores.push(mz);
        }
        let dfa_thr = gumbel_return_level(&dfa_scores, DESIGN_HORIZON / DFA_STRIDE as f64);

        let state = clean
            .iter()
            .map(|c| ChannelState {
                ring: [0.0; WINDOW],
                roll_ring: [0.0; ROLL],
                prev: c[length - 1],
                run: 1,
                cusum_pos: 0.0,
                cusum_neg: 0.0,
            })
            .collect();

        Some(HybridMonitor {
            calib,
            state,
            res_thr,
            dfa_thr,
            cusum_thr,
            res_hit_times: [u64::MAX; RES_HITS],
            dfa_streak: 0,
            roll_streak: 0,
            t: 0,
            scratch,
            alarmed: false,
        })
    }

    /// Number of monitored channels.
    #[must_use]
    pub fn channels(&self) -> usize {
        self.calib.len()
    }

    /// Feed one sample per channel; returns `Some(leg)` on the alarm tick.
    /// After an alarm, the monitor latches (returns `None`) until `reset`.
    ///
    /// Allocation-free. Worst-case cost: `channels` DFA evaluations over
    /// `WINDOW` samples (on ticks where `t % DFA_STRIDE == 0` and the ring
    /// is full), plus O(channels) scalar work.
    pub fn push(&mut self, sample: &[f64]) -> Option<Leg> {
        if self.alarmed || sample.len() != self.calib.len() {
            return None;
        }
        let t = self.t;
        self.t += 1;
        let ring_full = t >= WINDOW as u64;
        let roll_full = t >= ROLL as u64;

        // First sample of a stream primes per-channel state without scoring:
        // `prev` still holds the calibration sequence's last value, and a
        // fresh stream starts at an unrelated level — scoring across that
        // boundary poisons the residual and CUSUM legs with one giant
        // artificial residual.
        if t == 0 {
            for (ch, &v) in sample.iter().enumerate() {
                let st = &mut self.state[ch];
                st.prev = v;
                st.ring[0] = v;
                st.roll_ring[0] = v;
            }
            return None;
        }

        // Legs 1 + 2 + ring maintenance
        let mut res_hit = false;
        let mut repeat_alarm = false;
        let mut cusum_alarm = false;
        for (ch, &v) in sample.iter().enumerate() {
            let cc = &self.calib[ch];
            let st = &mut self.state[ch];

            let zs = (v - (cc.ar_a + cc.ar_b * st.prev)) / cc.ar_sd;
            let z = zs.abs();
            if z > self.res_thr {
                res_hit = true;
            }

            st.cusum_pos = (st.cusum_pos + zs - CUSUM_K).max(0.0);
            st.cusum_neg = (st.cusum_neg - zs - CUSUM_K).max(0.0);
            if st.cusum_pos.max(st.cusum_neg) > self.cusum_thr {
                cusum_alarm = true;
            }

            if v == st.prev {
                st.run += 1;
                if cc.repeat_enabled && st.run >= cc.max_run + REPEAT_MARGIN {
                    repeat_alarm = true;
                }
            } else {
                st.run = 1;
            }
            st.prev = v;

            st.ring[(t % WINDOW as u64) as usize] = v;
            st.roll_ring[(t % ROLL as u64) as usize] = v;
        }
        if res_hit {
            // shift the fixed-size hit history (RES_HITS entries)
            for i in 1..RES_HITS {
                self.res_hit_times[i - 1] = self.res_hit_times[i];
            }
            self.res_hit_times[RES_HITS - 1] = t;
            let oldest = self.res_hit_times[0];
            if oldest != u64::MAX && t - oldest < RES_SPAN as u64 {
                self.alarmed = true;
                return Some(Leg::Residual);
            }
        }
        if repeat_alarm {
            self.alarmed = true;
            return Some(Leg::RepeatedValue);
        }
        if cusum_alarm {
            self.alarmed = true;
            return Some(Leg::ResidualCusum);
        }

        // Leg 3: DFA every DFA_STRIDE ticks once the ring is full
        if ring_full && t % DFA_STRIDE as u64 == 0 {
            let mut hit = false;
            for ch in 0..self.calib.len() {
                let cc = &self.calib[ch];
                let st = &self.state[ch];
                // Copy ring into scratch order (oldest..newest). DFA is
                // invariant to a cyclic rotation ONLY if we linearize; do the
                // bounded copy (WINDOW elements) into a stack array.
                let mut lin = [0.0f64; WINDOW];
                let start = (t + 1) % WINDOW as u64;
                for (i, slot) in lin.iter_mut().enumerate() {
                    *slot = st.ring[((start + i as u64) % WINDOW as u64) as usize];
                }
                let a = dfa_into(&lin, &mut self.scratch).alpha;
                if (a - cc.alpha_mean).abs() / cc.alpha_sd > self.dfa_thr {
                    hit = true;
                    break;
                }
            }
            self.dfa_streak = if hit { self.dfa_streak + DFA_STRIDE } else { 0 };
            if self.dfa_streak >= DFA_PERSIST {
                self.alarmed = true;
                return Some(Leg::Dfa);
            }
        }

        // Leg 4: rolling-mean level shift
        if roll_full {
            let mut hit = false;
            for ch in 0..self.calib.len() {
                let cc = &self.calib[ch];
                let st = &self.state[ch];
                let sum: f64 = st.roll_ring.iter().sum();
                if (sum / ROLL as f64 - cc.mean).abs() > cc.roll_thr {
                    hit = true;
                    break;
                }
            }
            self.roll_streak = if hit { self.roll_streak + 1 } else { 0 };
            if self.roll_streak >= ROLL_PERSIST {
                self.alarmed = true;
                return Some(Leg::LevelShift);
            }
        }

        None
    }

    /// Clear the alarm latch and detector streaks (ring contents persist).
    pub fn reset(&mut self) {
        self.alarmed = false;
        self.dfa_streak = 0;
        self.roll_streak = 0;
        self.res_hit_times = [u64::MAX; RES_HITS];
        for st in self.state.iter_mut() {
            st.cusum_pos = 0.0;
            st.cusum_neg = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry_bench::{inject_fault, synth_spacecraft};

    fn run_stream(mon: &mut HybridMonitor, signal: &[Vec<f64>]) -> Option<(usize, Leg)> {
        let length = signal[0].len();
        let channels = signal.len();
        let mut sample = vec![0.0f64; channels];
        for t in 0..length {
            for ch in 0..channels {
                sample[ch] = signal[ch][t];
            }
            if let Some(leg) = mon.push(&sample) {
                return Some((t, leg));
            }
        }
        None
    }

    #[test]
    fn streaming_monitor_catches_stuck_and_stays_quiet_on_clean() {
        let calib = synth_spacecraft(700, 7919 + 100);
        let clean = synth_spacecraft(700, 7919 + 200);
        let faulted = inject_fault(&clean, "stuck", 7919);

        let mut mon = HybridMonitor::calibrate(&calib).expect("calibration");
        assert!(run_stream(&mut mon, &clean).is_none(), "clean must not alarm");

        let mut mon2 = HybridMonitor::calibrate(&calib).expect("calibration");
        let hit = run_stream(&mut mon2, &faulted).expect("stuck must alarm");
        assert_eq!(hit.1, Leg::RepeatedValue);
        assert!(hit.0 >= 406, "alarm at {} before fault start", hit.0);
    }

    #[test]
    fn streaming_monitor_catches_correlation_change_via_residual() {
        let calib = synth_spacecraft(700, 15838 + 100);
        let clean = synth_spacecraft(700, 15838 + 200);
        let faulted = inject_fault(&clean, "correlation_change", 15838);
        let mut mon = HybridMonitor::calibrate(&calib).expect("calibration");
        let hit = run_stream(&mut mon, &faulted).expect("correlation_change must alarm");
        assert_eq!(hit.1, Leg::Residual);
    }

    #[test]
    fn alarm_latches_until_reset() {
        let calib = synth_spacecraft(700, 100);
        let clean = synth_spacecraft(700, 200);
        let faulted = inject_fault(&clean, "stuck", 1);
        let mut mon = HybridMonitor::calibrate(&calib).expect("calibration");
        assert!(run_stream(&mut mon, &faulted).is_some());
        // latched: pushing more never alarms again
        assert!(run_stream(&mut mon, &faulted).is_none());
        mon.reset();
        assert!(run_stream(&mut mon, &faulted).is_some());
    }

    #[test]
    fn calibrate_rejects_too_short() {
        let short = synth_spacecraft(100, 42);
        assert!(HybridMonitor::calibrate(&short).is_none());
    }
}

#[cfg(test)]
mod debug_monitor {
    use super::*;
    use crate::telemetry_bench::synth_spacecraft;

    #[test]
    fn debug_clean_alarm() {
        let calib = synth_spacecraft(700, 7919 + 100);
        let clean = synth_spacecraft(700, 7919 + 200);
        let mut mon = HybridMonitor::calibrate(&calib).expect("calibration");
        println!("res_thr {:.2} dfa_thr {:.2} cusum_thr {:.2}", mon.res_thr, mon.dfa_thr, mon.cusum_thr);
        let mut sample = [0.0f64; 6];
        for t in 0..700 {
            for ch in 0..6 { sample[ch] = clean[ch][t]; }
            if let Some(leg) = mon.push(&sample) {
                println!("CLEAN ALARM at t={} leg={:?}", t, leg);
                for ch in 0..6 {
                    println!("  ch{} cusum {:.2}/{:.2}", ch, mon.state[ch].cusum_pos, mon.state[ch].cusum_neg);
                }
                return;
            }
        }
        println!("no alarm");
    }
}
