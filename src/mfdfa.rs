//! Multifractal DFA (MFDFA) — reveals signals with multiple scaling regimes.
//!
//! Regular DFA gives one number (α). MFDFA gives a SPECTRUM — how the
//! scaling exponent varies with the fluctuation order q. A monofractal
//! signal has a flat spectrum (same α everywhere). A multifractal signal
//! has a curved spectrum (different scales behave differently).
//!
//! This is what separates heartbeats from white noise, financial crashes
//! from normal trading, and diseased tissue from healthy.
//!
//! ```
//! use struktura::mfdfa::{mfdfa, MultifractalSpectrum};
//! let data: Vec<f64> = (0..1024).map(|i| (i as f64 * 0.1).sin()).collect();
//! let spectrum = mfdfa(&data, &[-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]);
//! println!("width: {:.3}", spectrum.width);
//! ```

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::{ln, powf};
use core::fmt;

/// Result for a single q-order in the MFDFA spectrum.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MfdfaPoint {
    pub q: f64,
    pub h_q: f64,
    pub r_squared: f64,
}

/// Full multifractal spectrum from MFDFA.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MultifractalSpectrum {
    pub points: Vec<MfdfaPoint>,
    /// Width of the spectrum: max(h_q) - min(h_q). Wider = more multifractal.
    pub width: f64,
    /// h(2) — equivalent to standard DFA α.
    pub h2: f64,
    /// Is the signal multifractal? (width > 0.05 with reliable R²)
    pub is_multifractal: bool,
}

impl fmt::Display for MultifractalSpectrum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MFDFA: h(2)={:.3} width={:.3} multifractal={}",
            self.h2, self.width, if self.is_multifractal { "YES" } else { "no" })
    }
}

/// Compute the generalized Hurst exponent h(q) for a single q value.
fn hurst_q(cumulative: &[f64], n: usize, q: f64) -> (f64, f64) {
    let s_min = 16usize.max(n / 50);
    let s_max = n / 4;
    if s_min >= s_max {
        return (0.5, 0.0);
    }
    let ratio = powf(s_max as f64 / s_min as f64, 1.0 / 7.0);

    let mut log_s = [0.0f64; 8];
    let mut log_fq = [0.0f64; 8];
    let mut pts = 0usize;
    let mut prev_s = 0usize;

    for step in 0..8 {
        let s = (s_min as f64 * powf(ratio, step as f64)) as usize;
        if s == prev_s || s > s_max { continue; }
        prev_s = s;

        let num_segs = n / s;
        if num_segs == 0 { continue; }

        let k = s as f64;
        let sx = k * (k - 1.0) / 2.0;
        let sx2 = k * (k - 1.0) * (2.0 * k - 1.0) / 6.0;
        let det = k * sx2 - sx * sx;
        if det.abs() < 1e-15 { continue; }

        let mut fq_sum = 0.0;
        for seg in 0..num_segs {
            let start = seg * s;
            let mut sy = 0.0;
            let mut sxy = 0.0;
            for i in 0..s {
                let yi = cumulative[start + i];
                sy += yi;
                sxy += i as f64 * yi;
            }
            let a0 = (sx2 * sy - sx * sxy) / det;
            let a1 = (k * sxy - sx * sy) / det;
            let mut f2 = 0.0;
            for i in 0..s {
                let d = cumulative[start + i] - (a0 + a1 * i as f64);
                f2 += d * d;
            }
            f2 /= k;

            if q == 0.0 {
                fq_sum += ln(f2.max(1e-30)) / 2.0;
            } else {
                fq_sum += powf(f2.max(1e-30), q / 2.0);
            }
        }

        let fq = if q == 0.0 {
            exp(fq_sum / num_segs as f64)
        } else {
            powf(fq_sum / num_segs as f64, 1.0 / q)
        };

        if fq > 0.0 {
            log_s[pts] = ln(s as f64);
            log_fq[pts] = ln(fq);
            pts += 1;
        }
    }

    if pts < 3 {
        return (0.5, 0.0);
    }

    let result = linreg_simple(&log_s[..pts], &log_fq[..pts]);
    (result.0, result.1)
}

fn linreg_simple(x: &[f64], y: &[f64]) -> (f64, f64) {
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
    (slope, r2)
}

#[cfg(not(feature = "std"))]
fn exp(x: f64) -> f64 { libm::exp(x) }
#[cfg(feature = "std")]
fn exp(x: f64) -> f64 { x.exp() }

/// Compute the multifractal DFA spectrum.
///
/// `q_values` controls which fluctuation orders to compute. Typical:
/// `&[-5.0, -3.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0]`
///
/// Negative q emphasizes small fluctuations, positive q emphasizes large ones.
/// If h(q) varies with q, the signal is multifractal.
pub fn mfdfa(values: &[f64], q_values: &[f64]) -> MultifractalSpectrum {
    let n = values.len();
    if n < 64 || q_values.is_empty() {
        return MultifractalSpectrum {
            points: Vec::new(), width: 0.0, h2: 0.5, is_multifractal: false,
        };
    }

    let mean = values.iter().sum::<f64>() / n as f64;
    let mut cumulative = Vec::with_capacity(n);
    let mut cum = 0.0;
    for &v in values {
        cum += v - mean;
        cumulative.push(cum);
    }

    let mut points = Vec::with_capacity(q_values.len());
    let mut h2 = 0.5;

    for &q in q_values {
        let (hq, r2) = hurst_q(&cumulative, n, q);
        if (q - 2.0).abs() < 0.01 { h2 = hq; }
        points.push(MfdfaPoint { q, h_q: hq, r_squared: r2 });
    }

    let reliable_count = points.iter().filter(|p| p.r_squared > 0.5).count();
    let (h_min, h_max) = if reliable_count >= 2 {
        let mn = points.iter().filter(|p| p.r_squared > 0.5).map(|p| p.h_q).fold(f64::INFINITY, f64::min);
        let mx = points.iter().filter(|p| p.r_squared > 0.5).map(|p| p.h_q).fold(f64::NEG_INFINITY, f64::max);
        (mn, mx)
    } else {
        (h2, h2)
    };
    let width = h_max - h_min;
    let is_multifractal = width > 0.05 && reliable_count >= 4;

    MultifractalSpectrum { points, width, h2, is_multifractal }
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

    #[test]
    fn white_noise_is_monofractal() {
        let data = white_noise(4096, 42);
        let spectrum = mfdfa(&data, &[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]);
        assert!(spectrum.width < 0.3, "white noise should be near-monofractal, width={:.3}", spectrum.width);
        assert!((spectrum.h2 - 0.5).abs() < 0.2, "h(2) should be near 0.5, got {:.3}", spectrum.h2);
    }

    #[test]
    fn mfdfa_produces_spectrum() {
        let data = white_noise(2048, 99);
        let qs = [-5.0, -3.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0];
        let spectrum = mfdfa(&data, &qs);
        assert_eq!(spectrum.points.len(), qs.len());
    }
}
