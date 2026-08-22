//! Universal anomaly detection via Detrended Fluctuation Analysis.
//!
//! ```
//! use struktura::{analyze, health_check, HealthVerdict};
//!
//! let signal = vec![0.1, 0.3, 0.2, 0.5, 0.4, 0.6, 0.3, 0.7,
//!                   0.2, 0.4, 0.5, 0.3, 0.6, 0.4, 0.5, 0.3,
//!                   0.1, 0.3, 0.2, 0.5, 0.4, 0.6, 0.3, 0.7,
//!                   0.2, 0.4, 0.5, 0.3, 0.6, 0.4, 0.5, 0.3,
//!                   0.1, 0.3, 0.2, 0.5, 0.4, 0.6, 0.3, 0.7,
//!                   0.2, 0.4, 0.5, 0.3, 0.6, 0.4, 0.5, 0.3,
//!                   0.1, 0.3, 0.2, 0.5, 0.4, 0.6, 0.3, 0.7,
//!                   0.2, 0.4, 0.5, 0.3, 0.6, 0.4, 0.5, 0.3];
//! let law = analyze(&signal);
//! assert!(law.n == 64);
//! ```

use std::fmt;

/// Result of a DFA or ACR computation.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DfaResult {
    /// Scaling exponent (slope in log-log space).
    pub alpha: f64,
    /// Coefficient of determination of the log-log fit.
    pub r_squared: f64,
}

impl fmt::Display for DfaResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "alpha={:.3} R2={:.4}", self.alpha, self.r_squared)
    }
}

/// How confident the analysis is in the derived exponent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LawQuality {
    Exact,
    Strong,
    Good,
    Approx,
    Abstain,
    Insufficient,
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StructuralLaw {
    pub hurst: f64,
    pub dfa: DfaResult,
    pub acr: DfaResult,
    pub mean: f64,
    pub std_dev: f64,
    pub kurtosis: f64,
    pub p99: f64,
    pub max: f64,
    pub n: usize,
    pub quality: LawQuality,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum HealthVerdict {
    Healthy,
    Watch,
    Warning,
    Critical,
}

impl HealthVerdict {
    pub fn from_shift(shift: f64) -> Self {
        let s = if shift < 0.0 { -shift } else { shift };
        if s < 0.03 {
            HealthVerdict::Healthy
        } else if s < 0.08 {
            HealthVerdict::Watch
        } else if s < 0.15 {
            HealthVerdict::Warning
        } else {
            HealthVerdict::Critical
        }
    }
}

pub fn dfa(values: &[f64]) -> DfaResult {
    let n = values.len();
    if n < 64 {
        return DfaResult { alpha: 0.5, r_squared: 0.0 };
    }

    let mean = values.iter().sum::<f64>() / n as f64;

    let mut y = Vec::with_capacity(n);
    let mut cum = 0.0;
    for &v in values {
        cum += v - mean;
        y.push(cum);
    }

    const BOXES: [usize; 8] = [16, 24, 36, 54, 81, 121, 181, 271];
    let mut log_s = [0.0f64; 8];
    let mut log_f = [0.0f64; 8];
    let mut pts = 0usize;

    for &s in &BOXES {
        if s > n / 4 { break; }
        let num_segs = n / s;
        if num_segs == 0 { continue; }
        let mut f2_sum = 0.0;
        for seg in 0..num_segs {
            let start = seg * s;
            let (mut sx, mut sy, mut sxy, mut sx2) = (0.0, 0.0, 0.0, 0.0);
            for i in 0..s {
                let xi = i as f64;
                sx += xi;
                sy += y[start + i];
                sxy += xi * y[start + i];
                sx2 += xi * xi;
            }
            let k = s as f64;
            let det = k * sx2 - sx * sx;
            if det.abs() < 1e-15 { continue; }
            let a0 = (sx2 * sy - sx * sxy) / det;
            let a1 = (k * sxy - sx * sy) / det;
            let mut resid = 0.0;
            for i in 0..s {
                let d = y[start + i] - (a0 + a1 * i as f64);
                resid += d * d;
            }
            f2_sum += resid / k;
        }
        let f = (f2_sum / num_segs as f64).sqrt();
        if f > 0.0 {
            log_s[pts] = (s as f64).ln();
            log_f[pts] = f.ln();
            pts += 1;
        }
    }

    if pts < 3 {
        return DfaResult { alpha: 0.5, r_squared: 0.0 };
    }

    linreg(&log_s[..pts], &log_f[..pts])
}

pub fn acr(values: &[f64]) -> DfaResult {
    let n = values.len();
    if n < 20 {
        return DfaResult { alpha: 0.0, r_squared: 0.0 };
    }

    let mean = values.iter().sum::<f64>() / n as f64;
    let var: f64 = values.iter().map(|&x| (x - mean) * (x - mean)).sum();
    if var < 1e-15 {
        return DfaResult { alpha: 0.0, r_squared: 0.0 };
    }

    const LAGS: [usize; 10] = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89];
    let mut log_lag = [0.0f64; 10];
    let mut log_r = [0.0f64; 10];
    let mut pts = 0usize;

    for &lag in &LAGS {
        if lag >= n / 2 { break; }
        let mut num = 0.0;
        for i in 0..n - lag {
            num += (values[i] - mean) * (values[i + lag] - mean);
        }
        let r = num / var;
        if r > 0.001 {
            log_lag[pts] = (lag as f64).ln();
            log_r[pts] = r.ln();
            pts += 1;
        }
    }

    if pts < 3 {
        return DfaResult { alpha: 0.0, r_squared: 0.0 };
    }

    linreg(&log_lag[..pts], &log_r[..pts])
}

pub fn analyze(values: &[f64]) -> StructuralLaw {
    let n = values.len();
    if n < 20 {
        return StructuralLaw {
            hurst: 0.5, dfa: DfaResult { alpha: 0.5, r_squared: 0.0 },
            acr: DfaResult { alpha: 0.0, r_squared: 0.0 },
            mean: 0.0, std_dev: 0.0, kurtosis: 0.0, p99: 0.0, max: 0.0,
            n, quality: LawQuality::Insufficient,
        };
    }

    let mean = values.iter().sum::<f64>() / n as f64;
    let var: f64 = values.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
    let std_dev = var.sqrt();
    let sd = if std_dev > 1e-15 { std_dev } else { 1e-15 };
    let kurtosis = values.iter().map(|&v| {
        let z = (v - mean) / sd;
        z * z * z * z
    }).sum::<f64>() / n as f64;

    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let p99 = sorted[((n as f64 * 0.99) as usize).min(n - 1)];

    let dfa_result = dfa(values);
    let acr_result = acr(values);
    let hurst = clamp(1.0 + acr_result.alpha / 2.0, 0.0, 1.0);

    let best_r2 = if dfa_result.r_squared > acr_result.r_squared { dfa_result.r_squared } else { acr_result.r_squared };
    let quality = if best_r2 > 0.95 { LawQuality::Exact }
        else if best_r2 > 0.85 { LawQuality::Strong }
        else if best_r2 > 0.7 { LawQuality::Good }
        else if best_r2 > 0.3 { LawQuality::Approx }
        else { LawQuality::Abstain };

    StructuralLaw { hurst, dfa: dfa_result, acr: acr_result, mean, std_dev, kurtosis, p99, max, n, quality }
}

impl StructuralLaw {
    pub fn is_healthy(&self) -> bool {
        self.quality != LawQuality::Abstain && self.quality != LawQuality::Insufficient
    }
}

impl DfaResult {
    pub fn is_reliable(&self) -> bool {
        self.r_squared > 0.7
    }
}

pub fn health_check(current: &StructuralLaw, baseline_alpha: f64) -> HealthVerdict {
    HealthVerdict::from_shift(current.dfa.alpha - baseline_alpha)
}

fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    if v < lo { lo } else if v > hi { hi } else { v }
}

fn linreg(x: &[f64], y: &[f64]) -> DfaResult {
    let k = x.len() as f64;
    let (mut sx, mut sy, mut sxy, mut sx2) = (0.0, 0.0, 0.0, 0.0);
    for i in 0..x.len() {
        sx += x[i]; sy += y[i]; sxy += x[i] * y[i]; sx2 += x[i] * x[i];
    }
    let slope = (k * sxy - sx * sy) / (k * sx2 - sx * sx);
    let ic = (sy - slope * sx) / k;
    let ym = sy / k;
    let mut sst = 0.0;
    let mut ssr = 0.0;
    for i in 0..x.len() {
        sst += (y[i] - ym) * (y[i] - ym);
        ssr += (y[i] - slope * x[i] - ic) * (y[i] - slope * x[i] - ic);
    }
    let r2 = 1.0 - ssr / if sst > 1e-15 { sst } else { 1e-15 };
    DfaResult { alpha: slope, r_squared: r2 }
}

pub struct SlidingWindow {
    buffer: Vec<f64>,
    capacity: usize,
    pos: usize,
    filled: bool,
}

impl SlidingWindow {
    pub fn new(capacity: usize) -> Self {
        SlidingWindow {
            buffer: vec![0.0; capacity],
            capacity,
            pos: 0,
            filled: false,
        }
    }

    pub fn push(&mut self, value: f64) {
        self.buffer[self.pos] = value;
        self.pos += 1;
        if self.pos >= self.capacity {
            self.pos = 0;
            self.filled = true;
        }
    }

    pub fn is_ready(&self) -> bool {
        self.filled
    }

    pub fn analyze(&self) -> StructuralLaw {
        if !self.filled {
            return analyze(&self.buffer[..self.pos]);
        }
        let mut ordered = Vec::with_capacity(self.capacity);
        ordered.extend_from_slice(&self.buffer[self.pos..]);
        ordered.extend_from_slice(&self.buffer[..self.pos]);
        analyze(&ordered)
    }
}

pub struct BaselineTracker {
    window: SlidingWindow,
    baseline: Option<f64>,
    learning_samples: usize,
    samples_seen: usize,
}

impl BaselineTracker {
    pub fn new(window_size: usize, learning_samples: usize) -> Self {
        BaselineTracker {
            window: SlidingWindow::new(window_size),
            baseline: None,
            learning_samples,
            samples_seen: 0,
        }
    }

    pub fn push(&mut self, value: f64) -> Option<HealthVerdict> {
        self.window.push(value);
        self.samples_seen += 1;

        if !self.window.is_ready() {
            return None;
        }

        if self.samples_seen <= self.learning_samples {
            let law = self.window.analyze();
            if law.dfa.r_squared > 0.7 {
                self.baseline = Some(law.dfa.alpha);
            }
            return None;
        }

        let baseline = self.baseline?;
        let law = self.window.analyze();
        Some(health_check(&law, baseline))
    }

    pub fn baseline(&self) -> Option<f64> {
        self.baseline
    }

    pub fn is_learning(&self) -> bool {
        self.samples_seen <= self.learning_samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn white_noise(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        (0..n).map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as f64 / (1u64 << 31) as f64 - 0.5
        }).collect()
    }

    fn brownian(n: usize, seed: u64) -> Vec<f64> {
        let noise = white_noise(n, seed);
        let mut walk = Vec::with_capacity(n);
        let mut sum = 0.0;
        for v in noise {
            sum += v;
            walk.push(sum);
        }
        walk
    }

    #[test]
    fn white_noise_alpha_near_half() {
        let data = white_noise(4096, 42);
        let result = dfa(&data);
        assert!(result.alpha > 0.35 && result.alpha < 0.65,
            "white noise DFA alpha should be near 0.5, got {}", result.alpha);
        assert!(result.r_squared > 0.8, "R2 should be high, got {}", result.r_squared);
    }

    #[test]
    fn brownian_alpha_above_one() {
        let data = brownian(4096, 42);
        let result = dfa(&data);
        assert!(result.alpha > 1.2 && result.alpha < 1.8,
            "brownian DFA alpha should be near 1.5, got {}", result.alpha);
    }

    #[test]
    fn deterministic() {
        let data = white_noise(1024, 7);
        let r1 = dfa(&data);
        let r2 = dfa(&data);
        assert!((r1.alpha - r2.alpha).abs() < 1e-10);
    }

    #[test]
    fn too_short_returns_half() {
        let data = [1.0; 10];
        let result = dfa(&data);
        assert_eq!(result.alpha, 0.5);
        assert_eq!(result.r_squared, 0.0);
    }

    #[test]
    fn analyze_produces_quality() {
        let data = white_noise(2048, 7);
        let law = analyze(&data);
        assert_eq!(law.n, 2048);
        assert!(law.quality != LawQuality::Insufficient);
    }

    #[test]
    fn health_verdict_thresholds() {
        assert_eq!(HealthVerdict::from_shift(0.01), HealthVerdict::Healthy);
        assert_eq!(HealthVerdict::from_shift(0.05), HealthVerdict::Watch);
        assert_eq!(HealthVerdict::from_shift(0.10), HealthVerdict::Warning);
        assert_eq!(HealthVerdict::from_shift(0.20), HealthVerdict::Critical);
        assert_eq!(HealthVerdict::from_shift(-0.20), HealthVerdict::Critical);
    }

    #[test]
    fn acr_detects_correlation() {
        let data = brownian(2048, 99);
        let result = acr(&data);
        assert!(result.alpha < -0.05, "brownian ACR exponent should be negative, got {}", result.alpha);
    }

    #[test]
    fn sliding_window_detects_after_fill() {
        let mut sw = SlidingWindow::new(256);
        assert!(!sw.is_ready());
        let noise = white_noise(256, 77);
        for v in &noise { sw.push(*v); }
        assert!(sw.is_ready());
        let law = sw.analyze();
        assert!(law.n == 256);
        assert!(law.dfa.alpha > 0.3);
    }

    #[test]
    fn baseline_tracker_learns_then_verdicts() {
        let mut bt = BaselineTracker::new(256, 500);
        let normal = brownian(600, 88);
        for (i, v) in normal.iter().enumerate() {
            let result = bt.push(*v);
            if i < 500 {
                assert!(result.is_none(), "should be learning at sample {}", i);
            }
        }
        assert!(!bt.is_learning());
    }

    #[test]
    fn sliding_window_before_fill_still_works() {
        let mut sw = SlidingWindow::new(512);
        for i in 0..100 {
            sw.push(i as f64 * 0.1);
        }
        assert!(!sw.is_ready());
        let law = sw.analyze();
        assert!(law.n == 100);
    }

    #[test]
    fn builtin_demo_data_detects_fault() {
        let normal: Vec<f64> = include_str!("../data/normal_sample.csv")
            .lines().filter_map(|l| l.trim().parse().ok()).collect();
        let fault: Vec<f64> = include_str!("../data/fault_sample.csv")
            .lines().filter_map(|l| l.trim().parse().ok()).collect();
        let law_n = analyze(&normal);
        let law_f = analyze(&fault);
        let verdict = health_check(&law_f, law_n.dfa.alpha);
        assert_eq!(verdict, HealthVerdict::Critical);
        assert!(law_n.dfa.r_squared > 0.9);
        assert!(law_f.dfa.r_squared > 0.9);
    }
}

impl fmt::Display for LawQuality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LawQuality::Exact => write!(f, "EXACT"),
            LawQuality::Strong => write!(f, "STRONG"),
            LawQuality::Good => write!(f, "GOOD"),
            LawQuality::Approx => write!(f, "APPROX"),
            LawQuality::Abstain => write!(f, "ABSTAIN"),
            LawQuality::Insufficient => write!(f, "INSUFFICIENT"),
        }
    }
}

impl fmt::Display for HealthVerdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HealthVerdict::Healthy => write!(f, "HEALTHY"),
            HealthVerdict::Watch => write!(f, "WATCH"),
            HealthVerdict::Warning => write!(f, "WARNING"),
            HealthVerdict::Critical => write!(f, "CRITICAL"),
        }
    }
}

impl fmt::Display for StructuralLaw {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "alpha={:.3} R2={:.4} H={:.3} quality={}", self.dfa.alpha, self.dfa.r_squared, self.hurst, self.quality)
    }
}

impl From<&[f64]> for SlidingWindow {
    fn from(data: &[f64]) -> Self {
        let mut sw = SlidingWindow::new(data.len().max(64));
        for &v in data { sw.push(v); }
        sw
    }
}

impl From<Vec<f64>> for SlidingWindow {
    fn from(data: Vec<f64>) -> Self {
        SlidingWindow::from(data.as_slice())
    }
}

impl Default for SlidingWindow {
    fn default() -> Self {
        SlidingWindow::new(256)
    }
}

impl Default for BaselineTracker {
    fn default() -> Self {
        BaselineTracker::new(256, 1000)
    }
}
