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
//! Five orthogonal detector legs, OR-fused (measured on the coupled
//! spacecraft benchmark: 6/6 detectable fault types at 100% event
//! detection, 0 false alarms across 200K clean samples):
//! 1. AR(1) residual — spikes, steps, correlation loss (instant)
//! 2. repeated-value run — stuck sensors (calibrated per-channel limit,
//!    auto-disabled for channels that legitimately saturate)
//! 3. windowed DFA — slow structural drift (integrative)
//! 4. rolling-mean level shift — regime changes (cancels the dominant
//!    periodic driver when `ROLL` matches its period)
//! 5. residual CUSUM — slow drift (cumulative mean shift in residuals)
//!
//! # Bounded work per tick (`push`)
//!
//! With `C` channels, every tick executes exactly:
//! - Legs 1+2+5 + ring writes: `C ×` (1 mul + 3 add + 1 div + 1 abs for the
//!   residual; 2 CUSUM updates; 1 compare for the run counter; 2 ring stores).
//! - Leg 4 (once `t ≥ ROLL`): `C × ROLL` additions (rolling sum) + `C`
//!   compares. (A running-sum variant would make this O(C); kept as a
//!   bounded loop for simplicity — still constant work.)
//! - Leg 3 (only when `t % DFA_STRIDE == 0` and `t ≥ WINDOW`): `C ×` one
//!   DFA evaluation over `WINDOW` samples = `C × (WINDOW linearize copies +
//!   WINDOW cumsum + Σ_boxes segments×boxsize single-pass sums + ≤12 ln
//!   calls + one 12-point linear regression)`.
//!
//! No branch depends on data values in a way that changes the bound; the
//! worst-case tick is `t % DFA_STRIDE == 0` with all legs enabled. There is
//! no allocation, no recursion, and no unbounded loop in `push`.
//!
//! # Memory bound
//!
//! Per channel: `WINDOW + ROLL` f64 ring slots + 6 calibration scalars +
//! ~10 state scalars ≈ `(WINDOW + ROLL + 16) × 8` bytes (= 1,664 bytes at
//! the default `WINDOW = ROLL = 96`). Monitor-level: one scratch buffer of
//! `3 × (WINDOW + 1)` f64 + a handful of scalars. All fixed after
//! `calibrate`.
//!
//! # Statistical guarantees (assumptions stated)
//!
//! **FAR design bound.** Each leg's threshold is the Gumbel return level of
//! its calibration score stream at [`DESIGN_HORIZON`]: under the
//! assumptions that (1) the operational stream is distributed as the
//! calibration stream (no regime change — that IS the fault case), (2)
//! block maxima of the score stream are approximately Gumbel (holds for
//! light-tailed score distributions by Fisher–Tippett), and (3) 16 blocks
//! estimate the Gumbel moments adequately, each leg's expected false-alarm
//! rate is ≤ 1 per `DESIGN_HORIZON` samples BEFORE persistence rules, which
//! only lower it. Measured evidence: 0 alarms over 200,000 clean samples
//! (97.7× the calibration length) with all six legs armed.
//!
//! **What is NOT guaranteed.** Channels violating assumption (1) at
//! calibration scale — e.g. random-walk channels whose level wanders
//! beyond the calibrated horizon, or naturally trending instruments —
//! must have the affected legs disabled via
//! [`HybridMonitor::set_leg_enabled`] (the repeated-value leg auto-disables
//! for saturating channels). The Voyager magnetometer case in
//! `struktura monitor-real` demonstrates both the failure and the remedy.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::dfa_fast_into as dfa_into;

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

/// Evolvable detection policy. The defaults are the hand-tuned values;
/// the RED/BLUE adversarial loop (`redblue` module) searches this space
/// for configurations with strictly better fault coverage at equal or
/// better false-alarm behavior.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MonitorConfig {
    /// Residual/parity exceedance pairing window (samples).
    pub res_span: u64,
    /// Consecutive flagged DFA evaluations required to alarm.
    pub dfa_persist: usize,
    /// Consecutive rolling-mean exceedances required to alarm.
    pub roll_persist: usize,
    /// CUSUM slack per step (residual sigmas).
    pub cusum_k: f64,
    /// Threshold design horizon (expected false alarms: 1 per this many).
    pub design_horizon: f64,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        MonitorConfig {
            res_span: RES_SPAN as u64,
            dfa_persist: DFA_PERSIST,
            roll_persist: ROLL_PERSIST,
            cusum_k: CUSUM_K,
            design_horizon: DESIGN_HORIZON,
        }
    }
}

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
    /// Explicit missingness: invalid samples reported via
    /// [`HybridMonitor::push_with_validity`]. Calibration data is assumed
    /// fully valid, so sustained missingness is itself a fault (packet
    /// loss) — value-and-structure detectors are provably blind to
    /// forward-filled gaps, only an explicit validity signal sees them.
    Missingness,
    /// Cross-channel parity: this channel's value is inconsistent with
    /// what the OTHER channels' coupled physics predicts for it
    /// (analytical redundancy). Catches faults where every channel looks
    /// individually plausible but the relations between them break.
    Parity,
}

/// Invalid samples within the trailing `MISS_SPAN` ticks required to alarm.
const MISS_HITS: usize = 4;
const MISS_SPAN: usize = 32;

/// Structured provenance for an alarm: WHY the monitor fired — which leg,
/// which channel, at what tick, how far past which threshold. This is what
/// a flight review asks of every alarm.
#[derive(Debug, Clone, Copy)]
pub struct AlarmReport {
    pub leg: Leg,
    pub channel: usize,
    /// The alarming channel's own tick at alarm time.
    pub tick: u64,
    /// The score that crossed (leg-specific units: z for residual/DFA/CUSUM,
    /// absolute deviation for level shift, run length for repeated,
    /// invalid count for missingness).
    pub observed: f64,
    /// The calibrated threshold it crossed.
    pub threshold: f64,
    /// Residual leg only: ticks between the two exceedances that formed the
    /// alarm — a step fault shows entry/exit hits far apart, a correlation
    /// change shows dense hits. 0 for other legs.
    pub hit_gap: u64,
}

/// Rule-based fault-class identification from alarm provenance.
/// Returns the most likely fault class name for a report.
#[must_use]
pub fn classify_alarm(r: &AlarmReport) -> &'static str {
    match r.leg {
        Leg::Missingness => "packet_loss",
        Leg::Parity => {
            // A transient spike violates the cross-channel relation by many
            // multiples of its threshold; a subtle inconsistency (sensor
            // miscalibration, gain error) sits just above it.
            if r.observed > 2.0 * r.threshold {
                "spike"
            } else {
                "cross_channel_inconsistency"
            }
        }
        Leg::RepeatedValue => "stuck",
        Leg::Dfa => "drift",
        Leg::ResidualCusum => "drift",
        Leg::LevelShift => {
            // A spike-class fault (large transient, ~6 sigma) drives the
            // rolling deviation far past a threshold calibrated for
            // ~1-sigma regime shifts; magnitude separates the two.
            if r.observed > 2.0 * r.threshold {
                "spike"
            } else {
                "regime_shift"
            }
        }
        Leg::Residual => {
            if r.hit_gap >= 8 {
                "spike"
            } else {
                "correlation_change"
            }
        }
    }
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
/// Every counter is clocked in THAT channel's own ticks, so channels may
/// arrive at different rates ([`HybridMonitor::push_channel`]).
#[derive(Debug, Clone)]
struct ChannelState {
    ring: [f64; WINDOW],
    roll_ring: [f64; ROLL],
    prev: f64,
    run: usize,
    cusum_pos: f64,
    cusum_neg: f64,
    /// This channel's own sample counter.
    t: u64,
    res_hit_times: [u64; RES_HITS],
    miss_times: [u64; MISS_HITS],
    parity_hit_times: [u64; RES_HITS],
    dfa_streak: usize,
    roll_streak: usize,
}

/// Streaming hybrid monitor over `n_channels` telemetry channels.
///
/// Feed one sample per channel per tick via [`HybridMonitor::push`], or
/// per-channel at independent rates via [`HybridMonitor::push_channel`].
/// Returns `Some(Leg)` on the tick an alarm is raised.
pub struct HybridMonitor {
    calib: Vec<ChannelCalib>,
    state: Vec<ChannelState>,
    res_thr: f64,
    dfa_thr: f64,
    cusum_thr: f64,
    scratch: Vec<f64>,
    alarmed: bool,
    last_alarm: Option<AlarmReport>,
    /// Cross-channel reconstruction models (analytical redundancy).
    recon: Vec<Reconstructor>,
    /// Parity-leg threshold (Gumbel return level of calibration parity z).
    parity_thr: f64,
    /// Channels switched to virtual mode (dead sensor; reconstruction
    /// substitutes, own legs disabled).
    quarantined: Vec<bool>,
    /// Per-leg enable mask: [residual, repeated, dfa, level, cusum, miss, parity].
    /// Legs whose stationarity assumptions a deployment cannot meet
    /// (e.g. level-shift on a naturally trending channel) are disabled
    /// at configuration time — standard flight-monitor practice.
    leg_enabled: [bool; 7],
    config: MonitorConfig,
}

/// Linear reconstruction model: one channel estimated from all others.
/// Learned at calibration by least squares (analytical redundancy — the
/// coupled physics that relates channels is captured empirically).
#[derive(Debug, Clone)]
struct Reconstructor {
    /// Weight per source channel (own channel's weight is 0).
    weights: Vec<f64>,
    bias: f64,
    /// Residual sd of the reconstruction on calibration data.
    sd: f64,
    /// Fraction of the channel's variance explained (R²) on calibration.
    r2: f64,
}

/// Calibrated constants exported for code generation.
#[derive(Debug, Clone)]
pub struct MonitorExport {
    pub res_thr: f64,
    pub dfa_thr: f64,
    pub cusum_thr: f64,
    pub channels: Vec<ChannelExport>,
}

/// One channel's calibrated constants.
#[derive(Debug, Clone, Copy)]
pub struct ChannelExport {
    pub ar_a: f64,
    pub ar_b: f64,
    pub ar_sd: f64,
    pub alpha_mean: f64,
    pub alpha_sd: f64,
    pub mean: f64,
    pub roll_thr: f64,
    pub max_run: usize,
    pub repeat_enabled: bool,
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
    let var = maxima.iter().map(|m| crate::powi(m - mean, 2)).sum::<f64>() / BLOCKS as f64;
    let beta = (crate::sqrt(var) * 2.449_489_742_783_178 / core::f64::consts::PI).max(1e-9); // σ√6/π
    let mu = mean - 0.577_215_664_901_532_9 * beta;
    // Return period in blocks for one expected exceedance per `horizon` samples
    let t = (horizon / block_len as f64).max(2.0);
    // Gumbel quantile at exceedance probability 1/T: x = μ − β ln(−ln(1 − 1/T))
    let p = 1.0 - 1.0 / t;
    mu - beta * crate::ln(-crate::ln(p))
}

/// Solve the linear system A x = b in place (Gauss–Jordan with partial
/// pivoting). A is n×n row-major. Returns false when singular.
fn solve_linear(a: &mut [f64], b: &mut [f64], n: usize) -> bool {
    for col in 0..n {
        let mut pivot = col;
        for row in col + 1..n {
            if a[row * n + col].abs() > a[pivot * n + col].abs() {
                pivot = row;
            }
        }
        if a[pivot * n + col].abs() < 1e-12 {
            return false;
        }
        if pivot != col {
            for k in 0..n {
                a.swap(col * n + k, pivot * n + k);
            }
            b.swap(col, pivot);
        }
        let d = a[col * n + col];
        for k in 0..n {
            a[col * n + k] /= d;
        }
        b[col] /= d;
        for row in 0..n {
            if row != col {
                let f = a[row * n + col];
                if f != 0.0 {
                    for k in 0..n {
                        a[row * n + k] -= f * a[col * n + k];
                    }
                    b[row] -= f * b[col];
                }
            }
        }
    }
    true
}

/// Fit `target = bias + Σ w_j · source_j` (j ≠ target) by least squares
/// over the calibration streams. Ridge-regularized (λ = 1e-6 · trace/n)
/// for numerical safety on collinear channels.
fn fit_reconstructor(clean: &[Vec<f64>], target: usize) -> Reconstructor {
    let channels = clean.len();
    let length = clean[0].len();
    let sources: Vec<usize> = (0..channels).filter(|&c| c != target).collect();
    let p = sources.len();

    // Standardize every channel first — raw scales differ by orders of
    // magnitude (wheel speed ~2200 vs SOC ~0.7) and make the raw normal
    // equations ill-conditioned. Fit in z-space, back-transform after.
    let stats: Vec<(f64, f64)> = clean
        .iter()
        .map(|c| {
            let m = c.iter().sum::<f64>() / length as f64;
            let v = c.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / length as f64;
            (m, crate::sqrt(v).max(1e-12))
        })
        .collect();

    // Normal equations in standardized space (centered → no bias column)
    let mut xtx = vec![0.0f64; p * p];
    let mut xty = vec![0.0f64; p];
    let (ym, ys) = stats[target];
    for t in 0..length {
        let yz = (clean[target][t] - ym) / ys;
        let mut row = Vec::with_capacity(p);
        for &s in &sources {
            let (m, sd) = stats[s];
            row.push((clean[s][t] - m) / sd);
        }
        for i in 0..p {
            xty[i] += row[i] * yz;
            for j in 0..p {
                xtx[i * p + j] += row[i] * row[j];
            }
        }
    }
    // Ridge in standardized space (diagonal ≈ length per entry)
    let lambda = 1e-4 * length as f64;
    for i in 0..p {
        xtx[i * p + i] += lambda;
    }
    let mut coef = xty.clone();
    let ok = solve_linear(&mut xtx, &mut coef, p);

    // Back-transform: y = ym + ys·Σ wz_s·(x_s − m_s)/sd_s
    let mut weights = vec![0.0f64; channels];
    let mut bias = ym;
    if ok {
        for (k, &s) in sources.iter().enumerate() {
            let (m, sd) = stats[s];
            let w_raw = ys * coef[k] / sd;
            weights[s] = w_raw;
            bias -= w_raw * m;
        }
    }

    // Residual sd + R² on calibration
    let ymean: f64 = clean[target].iter().sum::<f64>() / length as f64;
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    for t in 0..length {
        let mut pred = bias;
        for &s in &sources {
            pred += weights[s] * clean[s][t];
        }
        let y = clean[target][t];
        ss_res += (y - pred) * (y - pred);
        ss_tot += (y - ymean) * (y - ymean);
    }
    let sd = crate::sqrt(ss_res / length as f64).max(1e-9);
    let r2 = if ss_tot > 1e-12 { 1.0 - ss_res / ss_tot } else { 0.0 };
    Reconstructor { weights, bias, sd, r2 }
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
    (a, b, crate::sqrt(ss / n as f64).max(1e-9))
}

impl HybridMonitor {
    /// Calibrate from a clean multi-channel sequence (one inner slice per
    /// channel, all the same length, at least `2 * WINDOW` samples).
    ///
    /// This is the only phase that allocates.
    pub fn calibrate(clean: &[Vec<f64>]) -> Option<HybridMonitor> {
        Self::calibrate_with(clean, MonitorConfig::default())
    }

    /// Calibrate with an explicit detection policy (see [`MonitorConfig`]).
    pub fn calibrate_with(clean: &[Vec<f64>], config: MonitorConfig) -> Option<HybridMonitor> {
        let channels = clean.len();
        if channels == 0 {
            return None;
        }
        let length = clean[0].len();
        if length < 2 * WINDOW || length <= ROLL {
            return None;
        }
        if clean.iter().any(|c| c.len() != length) {
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
                alphas.iter().map(|a| crate::powi(a - alpha_mean, 2)).sum::<f64>() / na;

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
            let roll_thr = gumbel_return_level(&roll_devs, config.design_horizon);

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
                alpha_sd: crate::sqrt(alpha_var).max(1e-6),
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
        let res_thr = gumbel_return_level(&res_scores, config.design_horizon);

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
                    pos[ch] = (pos[ch] + z - config.cusum_k).max(0.0);
                    neg[ch] = (neg[ch] - z - config.cusum_k).max(0.0);
                    let m = pos[ch].max(neg[ch]);
                    if m > mc {
                        mc = m;
                    }
                }
                cusum_path.push(mc);
            }
        }
        let cusum_thr = gumbel_return_level(&cusum_path, config.design_horizon);

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
        let dfa_thr = gumbel_return_level(&dfa_scores, config.design_horizon / DFA_STRIDE as f64);

        // Analytical redundancy: reconstruct each channel from the others,
        // and calibrate the parity leg on the max-over-channels
        // standardized reconstruction error.
        let recon: Vec<Reconstructor> =
            (0..channels).map(|t| fit_reconstructor(clean, t)).collect();
        let mut parity_scores = Vec::with_capacity(length);
        for t in 0..length {
            let mut mz = 0.0f64;
            for ch in 0..channels {
                let r = &recon[ch];
                let mut pred = r.bias;
                for (s, c) in clean.iter().enumerate() {
                    pred += r.weights[s] * c[t];
                }
                let z = (clean[ch][t] - pred).abs() / r.sd;
                if z > mz {
                    mz = z;
                }
            }
            parity_scores.push(mz);
        }
        let parity_thr = gumbel_return_level(&parity_scores, config.design_horizon);

        let state = clean
            .iter()
            .map(|c| ChannelState {
                ring: [0.0; WINDOW],
                roll_ring: [0.0; ROLL],
                prev: c[length - 1],
                run: 1,
                cusum_pos: 0.0,
                cusum_neg: 0.0,
                t: 0,
                res_hit_times: [u64::MAX; RES_HITS],
                miss_times: [u64::MAX; MISS_HITS],
                parity_hit_times: [u64::MAX; RES_HITS],
                dfa_streak: 0,
                roll_streak: 0,
            })
            .collect();

        Some(HybridMonitor {
            calib,
            state,
            res_thr,
            dfa_thr,
            cusum_thr,
            scratch,
            alarmed: false,
            last_alarm: None,
            recon,
            parity_thr,
            quarantined: {
                let mut q = Vec::with_capacity(channels);
                q.resize(channels, false);
                q
            },
            leg_enabled: [true; 7],
            config,
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
        self.push_with_validity(sample, &[])
    }

    /// Like [`HybridMonitor::push`], with an explicit per-channel validity
    /// flag. An empty `valid` slice means all channels valid. An invalid
    /// channel sample this tick is excluded from every value/structure leg
    /// (its rings are forward-filled with the last valid value) and counted
    /// by the missingness leg: `MISS_HITS` invalid samples within
    /// `MISS_SPAN` ticks raise [`Leg::Missingness`]. Calibration data is
    /// assumed fully valid.
    pub fn push_with_validity(&mut self, sample: &[f64], valid: &[bool]) -> Option<Leg> {
        if sample.len() != self.calib.len() {
            return None;
        }
        let mut alarm = None;
        for (ch, &v) in sample.iter().enumerate() {
            let is_valid = valid.get(ch).copied().unwrap_or(true);
            let value = if is_valid { Some(v) } else { None };
            if let Some(leg) = self.push_channel(ch, value) {
                alarm = Some(leg);
            }
        }
        alarm
    }

    /// Feed one sample for ONE channel — channels may arrive at independent
    /// rates (multi-rate telemetry). `None` marks a missing/invalid sample
    /// slot. Every detector counter is clocked in this channel's own ticks.
    /// Returns `Some(leg)` on the alarm tick; the monitor then latches.
    pub fn push_channel(&mut self, ch: usize, value: Option<f64>) -> Option<Leg> {
        if self.alarmed || ch >= self.calib.len() {
            return None;
        }
        // Quarantined sensor: substitute the reconstructed reading so the
        // ring state stays coherent for a later unquarantine; its own
        // detector legs stay silent (the sensor is declared dead).
        if self.quarantined[ch] {
            let virt = self.virtual_value(ch).map(|(v, _)| v).unwrap_or(0.0);
            let st = &mut self.state[ch];
            let t = st.t;
            st.t += 1;
            st.prev = virt;
            st.ring[(t % WINDOW as u64) as usize] = virt;
            st.roll_ring[(t % ROLL as u64) as usize] = virt;
            return None;
        }
        // Parity prediction must be computed before borrowing state mutably.
        let parity_pred = {
            let r = &self.recon[ch];
            let mut pred = r.bias;
            for (s, stx) in self.state.iter().enumerate() {
                pred += r.weights[s] * stx.prev;
            }
            (pred, r.sd)
        };
        let cc = &self.calib[ch];
        let st = &mut self.state[ch];
        let t = st.t;
        st.t += 1;

        // First sample of a stream primes state without scoring: `prev`
        // still holds the calibration sequence's last value, and a fresh
        // stream starts at an unrelated level — scoring across that boundary
        // poisons the residual and CUSUM legs with one giant residual.
        if t == 0 {
            if let Some(v) = value {
                st.prev = v;
                st.ring[0] = v;
                st.roll_ring[0] = v;
            }
            return None;
        }

        let v = match value {
            Some(v) => v,
            None => {
                // Missing sample: forward-fill rings, count for leg 6.
                st.ring[(t % WINDOW as u64) as usize] = st.prev;
                st.roll_ring[(t % ROLL as u64) as usize] = st.prev;
                if self.leg_enabled[5] {
                    for i in 1..MISS_HITS {
                        st.miss_times[i - 1] = st.miss_times[i];
                    }
                    st.miss_times[MISS_HITS - 1] = t;
                    let oldest = st.miss_times[0];
                    if oldest != u64::MAX && t - oldest < MISS_SPAN as u64 {
                        self.alarmed = true;
                        self.last_alarm = Some(AlarmReport {
                            leg: Leg::Missingness,
                            channel: ch,
                            tick: t,
                            observed: MISS_HITS as f64,
                            threshold: MISS_HITS as f64,
                            hit_gap: 0,
                        });
                        return Some(Leg::Missingness);
                    }
                }
                return None;
            }
        };

        // Leg 1: residual + leg 5: CUSUM (both from the same z-score)
        let zs = (v - (cc.ar_a + cc.ar_b * st.prev)) / cc.ar_sd;
        st.cusum_pos = (st.cusum_pos + zs - self.config.cusum_k).max(0.0);
        st.cusum_neg = (st.cusum_neg - zs - self.config.cusum_k).max(0.0);
        let cusum_alarm = st.cusum_pos.max(st.cusum_neg) > self.cusum_thr;

        let res_hit = zs.abs() > self.res_thr;

        // Leg 2: repeated value
        let mut repeat_alarm = false;
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

        if res_hit && self.leg_enabled[0] {
            for i in 1..RES_HITS {
                st.res_hit_times[i - 1] = st.res_hit_times[i];
            }
            st.res_hit_times[RES_HITS - 1] = t;
            let oldest = st.res_hit_times[0];
            if oldest != u64::MAX && t - oldest < self.config.res_span {
                self.alarmed = true;
                self.last_alarm = Some(AlarmReport {
                    leg: Leg::Residual,
                    channel: ch,
                    tick: t,
                    observed: zs.abs(),
                    threshold: self.res_thr,
                    hit_gap: t - oldest,
                });
                return Some(Leg::Residual);
            }
        }
        // Leg 7: cross-channel parity (2-in-20 persistence). Skipped while
        // any channel is quarantined — a substituted reading would feed the
        // predictor its own reconstruction.
        if self.leg_enabled[6] && !self.quarantined.iter().any(|&q| q) {
            let (pred, sd) = parity_pred;
            let pz = (v - pred).abs() / sd;
            if pz > self.parity_thr {
                for i in 1..RES_HITS {
                    st.parity_hit_times[i - 1] = st.parity_hit_times[i];
                }
                st.parity_hit_times[RES_HITS - 1] = t;
                let oldest = st.parity_hit_times[0];
                if oldest != u64::MAX && t - oldest < self.config.res_span {
                    self.alarmed = true;
                    self.last_alarm = Some(AlarmReport {
                        leg: Leg::Parity,
                        channel: ch,
                        tick: t,
                        observed: pz,
                        threshold: self.parity_thr,
                        hit_gap: t - oldest,
                    });
                    return Some(Leg::Parity);
                }
            }
        }

        if repeat_alarm && self.leg_enabled[1] {
            self.alarmed = true;
            self.last_alarm = Some(AlarmReport {
                leg: Leg::RepeatedValue,
                channel: ch,
                tick: t,
                observed: self.state[ch].run as f64,
                threshold: (cc.max_run + REPEAT_MARGIN) as f64,
                hit_gap: 0,
            });
            return Some(Leg::RepeatedValue);
        }
        if cusum_alarm && self.leg_enabled[4] {
            let st = &self.state[ch];
            self.alarmed = true;
            self.last_alarm = Some(AlarmReport {
                leg: Leg::ResidualCusum,
                channel: ch,
                tick: t,
                observed: st.cusum_pos.max(st.cusum_neg),
                threshold: self.cusum_thr,
                hit_gap: 0,
            });
            return Some(Leg::ResidualCusum);
        }

        // Leg 3: DFA every DFA_STRIDE of this channel's ticks, ring full
        if self.leg_enabled[2] && t >= WINDOW as u64 && t % DFA_STRIDE as u64 == 0 {
            // Linearize the ring (oldest..newest) — bounded WINDOW copy.
            let mut lin = [0.0f64; WINDOW];
            let start = (t + 1) % WINDOW as u64;
            for (i, slot) in lin.iter_mut().enumerate() {
                *slot = st.ring[((start + i as u64) % WINDOW as u64) as usize];
            }
            let a = dfa_into(&lin, &mut self.scratch).alpha;
            let st = &mut self.state[ch];
            let hit = (a - cc.alpha_mean).abs() / cc.alpha_sd > self.dfa_thr;
            st.dfa_streak = if hit { st.dfa_streak + DFA_STRIDE } else { 0 };
            if st.dfa_streak >= self.config.dfa_persist {
                self.alarmed = true;
                self.last_alarm = Some(AlarmReport {
                    leg: Leg::Dfa,
                    channel: ch,
                    tick: t,
                    observed: (a - cc.alpha_mean).abs() / cc.alpha_sd,
                    threshold: self.dfa_thr,
                    hit_gap: 0,
                });
                return Some(Leg::Dfa);
            }
        }

        // Leg 4: rolling-mean level shift
        if self.leg_enabled[3] && t >= ROLL as u64 {
            let st = &mut self.state[ch];
            let sum: f64 = st.roll_ring.iter().sum();
            let hit = (sum / ROLL as f64 - cc.mean).abs() > cc.roll_thr;
            st.roll_streak = if hit { st.roll_streak + 1 } else { 0 };
            if st.roll_streak >= self.config.roll_persist {
                self.alarmed = true;
                self.last_alarm = Some(AlarmReport {
                    leg: Leg::LevelShift,
                    channel: ch,
                    tick: t,
                    observed: (sum / ROLL as f64 - cc.mean).abs(),
                    threshold: cc.roll_thr,
                    hit_gap: 0,
                });
                return Some(Leg::LevelShift);
            }
        }

        None
    }

    /// Clear the alarm latch and detector streaks (ring contents persist).
    /// Provenance of the most recent alarm, if any.
    #[must_use]
    pub fn last_alarm(&self) -> Option<AlarmReport> {
        self.last_alarm
    }

    /// Switch a channel to virtual mode: its own detector legs stop (the
    /// sensor is declared dead), and [`HybridMonitor::virtual_value`]
    /// serves a reconstructed reading from the surviving channels.
    /// The rest of the monitor keeps operating — degraded, not blind.
    pub fn quarantine(&mut self, ch: usize) {
        if ch < self.quarantined.len() {
            self.quarantined[ch] = true;
        }
    }

    /// Return a quarantined channel to normal operation.
    pub fn unquarantine(&mut self, ch: usize) {
        if ch < self.quarantined.len() {
            self.quarantined[ch] = false;
        }
    }

    /// Reconstructed estimate of a channel from the most recent values of
    /// the other channels, with its calibrated 1-sigma reconstruction
    /// uncertainty. Works for any channel; this is the virtual reading a
    /// downstream consumer uses when the physical sensor is quarantined.
    #[must_use]
    pub fn virtual_value(&self, ch: usize) -> Option<(f64, f64)> {
        if ch >= self.recon.len() {
            return None;
        }
        let r = &self.recon[ch];
        let mut pred = r.bias;
        for (s, st) in self.state.iter().enumerate() {
            pred += r.weights[s] * st.prev;
        }
        Some((pred, r.sd))
    }

    /// Reconstruction quality (R², residual sd) for a channel, measured on
    /// the calibration data. R² near 1 = the channel is physically coupled
    /// to the others and a virtual reading is trustworthy.
    #[must_use]
    pub fn reconstruction_quality(&self, ch: usize) -> Option<(f64, f64)> {
        self.recon.get(ch).map(|r| (r.r2, r.sd))
    }

    /// Export every calibrated constant — for baking a calibration into
    /// generated flight code (see `codegen::generate_hybrid_c`).
    #[must_use]
    pub fn export(&self) -> MonitorExport {
        MonitorExport {
            res_thr: self.res_thr,
            dfa_thr: self.dfa_thr,
            cusum_thr: self.cusum_thr,
            channels: self
                .calib
                .iter()
                .map(|c| ChannelExport {
                    ar_a: c.ar_a,
                    ar_b: c.ar_b,
                    ar_sd: c.ar_sd,
                    alpha_mean: c.alpha_mean,
                    alpha_sd: c.alpha_sd,
                    mean: c.mean,
                    roll_thr: c.roll_thr,
                    max_run: c.max_run,
                    repeat_enabled: c.repeat_enabled,
                })
                .collect(),
        }
    }

    /// Enable or disable one detector leg (all enabled by default).
    pub fn set_leg_enabled(&mut self, leg: Leg, on: bool) {
        let idx = match leg {
            Leg::Residual => 0,
            Leg::RepeatedValue => 1,
            Leg::Dfa => 2,
            Leg::LevelShift => 3,
            Leg::ResidualCusum => 4,
            Leg::Missingness => 5,
            Leg::Parity => 6,
        };
        self.leg_enabled[idx] = on;
    }

    pub fn reset(&mut self) {
        self.alarmed = false;
        for st in self.state.iter_mut() {
            st.cusum_pos = 0.0;
            st.cusum_neg = 0.0;
            st.dfa_streak = 0;
            st.roll_streak = 0;
            st.res_hit_times = [u64::MAX; RES_HITS];
            st.miss_times = [u64::MAX; MISS_HITS];
            st.parity_hit_times = [u64::MAX; RES_HITS];
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
    fn multi_rate_channels_detect_stuck() {
        // Channels pushed at different rates: ch0 every tick, ch1 every 2nd,
        // ch2 every 4th, others every tick. Stuck fault on ch2 (temp) must
        // still be caught even though ch2 sees 1/4 the samples.
        let calib = synth_spacecraft(1400, 7919 + 100);
        let clean = synth_spacecraft(1400, 7919 + 200);
        let faulted = inject_fault(&clean, "stuck", 7919);
        let mut mon = HybridMonitor::calibrate(&calib).expect("calibration");
        let mut alarm = None;
        for t in 0..1400usize {
            for ch in 0..6 {
                let rate = match ch { 1 => 2, 2 => 4, _ => 1 };
                if t % rate == 0 {
                    if let Some(leg) = mon.push_channel(ch, Some(faulted[ch][t])) {
                        alarm = Some((t, leg));
                    }
                }
            }
            if alarm.is_some() {
                break;
            }
        }
        let (t, leg) = alarm.expect("stuck must alarm under multi-rate");
        assert_eq!(leg, Leg::RepeatedValue);
        // fault starts at 58% of 1400 = 812
        assert!(t >= 812, "alarm at {} before fault", t);
    }

    #[test]
    fn multi_rate_clean_stays_quiet() {
        let calib = synth_spacecraft(1400, 31 + 100);
        let clean = synth_spacecraft(1400, 31 + 200);
        let mut mon = HybridMonitor::calibrate(&calib).expect("calibration");
        for t in 0..1400usize {
            for ch in 0..6 {
                let rate = match ch { 1 => 2, 2 => 4, _ => 1 };
                if t % rate == 0 {
                    assert!(
                        mon.push_channel(ch, Some(clean[ch][t])).is_none(),
                        "clean multi-rate alarmed at t={} ch={}", t, ch
                    );
                }
            }
        }
    }

    #[test]
    fn virtual_sensor_reconstruction_is_accurate() {
        // Quarantine bus_voltage (physically coupled to SOC: v = 26.5 +
        // 3.4*soc + ...). The virtual reading must track the real value.
        let calib = synth_spacecraft(1400, 7 + 100);
        let clean = synth_spacecraft(1400, 7 + 200);
        let mut mon = HybridMonitor::calibrate(&calib).expect("calibration");
        let (r2, _sd) = mon.reconstruction_quality(1).expect("recon");
        assert!(r2 > 0.9, "bus_voltage should be well-coupled, R2 = {}", r2);

        mon.quarantine(1);
        let mut err_sum = 0.0f64;
        let mut n = 0usize;
        for t in 0..1400usize {
            for ch in 0..6 {
                if ch == 1 {
                    // dead sensor: no data pushed
                    mon.push_channel(1, None);
                } else {
                    mon.push_channel(ch, Some(clean[ch][t]));
                }
            }
            if t > 10 {
                let (virt, _) = mon.virtual_value(1).expect("virtual");
                err_sum += (virt - clean[1][t]).abs();
                n += 1;
            }
        }
        let mae = err_sum / n as f64;
        let true_sd = {
            let m = clean[1].iter().sum::<f64>() / 1400.0;
            (clean[1].iter().map(|x| (x - m).powi(2)).sum::<f64>() / 1400.0).sqrt()
        };
        // Virtual reading must be far better than just guessing the mean.
        assert!(
            mae < 0.3 * true_sd,
            "virtual MAE {} vs channel sd {}",
            mae, true_sd
        );
    }

    #[test]
    fn survives_double_fault_with_dead_sensor() {
        // Iron Man scenario: temp sensor (ch2) declared dead and
        // quarantined. A drift fault later hits SOC. The monitor must still
        // catch it — degraded, not blind.
        let calib = synth_spacecraft(1400, 99 + 100);
        let clean = synth_spacecraft(1400, 99 + 200);
        let faulted = inject_fault(&clean, "drift", 99);
        let mut mon = HybridMonitor::calibrate(&calib).expect("calibration");
        mon.quarantine(2);
        let mut alarm = None;
        for t in 0..1400usize {
            for ch in 0..6 {
                let v = if ch == 2 { None } else { Some(faulted[ch][t]) };
                if let Some(leg) = mon.push_channel(ch, v) {
                    alarm = Some((t, leg));
                }
            }
            if alarm.is_some() {
                break;
            }
        }
        let (t, _leg) = alarm.expect("drift must still be caught with ch2 dead");
        // fault starts at 58% of 1400 = 812
        assert!(t >= 812, "alarm at {} before fault", t);
    }

    #[test]
    fn quarantined_channel_never_alarms_clean() {
        let calib = synth_spacecraft(1400, 55 + 100);
        let clean = synth_spacecraft(1400, 55 + 200);
        let mut mon = HybridMonitor::calibrate(&calib).expect("calibration");
        mon.quarantine(0);
        for t in 0..1400usize {
            for ch in 0..6 {
                let v = if ch == 0 { None } else { Some(clean[ch][t]) };
                assert!(
                    mon.push_channel(ch, v).is_none(),
                    "clean stream alarmed at t={} ch={}",
                    t, ch
                );
            }
        }
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
