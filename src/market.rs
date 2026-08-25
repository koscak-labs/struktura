//! Financial time series regime detection via DFA.
//!
//! DFA alpha on returns classifies the market regime:
//! - α > 0.6: trending (momentum strategies work)
//! - α ≈ 0.5: random walk (nothing works reliably)
//! - α < 0.4: mean-reverting (reversal strategies work)
//!
//! ```
//! use struktura::market::{returns_from_prices, regime_detect};
//! let prices = vec![100.0, 101.0, 99.5, 102.0, 103.5, 101.0, 104.0, 105.0];
//! let returns = returns_from_prices(&prices);
//! // Need 200+ price points for reliable regime detection
//! ```

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::string::String;

use crate::{dfa, analyze, StructuralLaw};
use core::fmt;

/// Market regime classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketRegime {
    /// α > 0.6 — persistent trends, momentum works
    Trending,
    /// 0.45 < α < 0.6 — efficient/random, no edge
    RandomWalk,
    /// α < 0.45 — mean-reverting, reversal works
    MeanReverting,
}

impl fmt::Display for MarketRegime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MarketRegime::Trending => write!(f, "TRENDING"),
            MarketRegime::RandomWalk => write!(f, "RANDOM WALK"),
            MarketRegime::MeanReverting => write!(f, "MEAN-REVERTING"),
        }
    }
}

/// Result of a market regime analysis.
#[derive(Debug, Clone)]
pub struct RegimeResult {
    pub alpha: f64,
    pub r_squared: f64,
    pub regime: MarketRegime,
    pub n_returns: usize,
    pub law: StructuralLaw,
}

impl fmt::Display for RegimeResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "α={:.3} R²={:.4} regime={} (n={})",
            self.alpha, self.r_squared, self.regime, self.n_returns)
    }
}

/// Convert a price series to log-returns.
pub fn returns_from_prices(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 { return Vec::new(); }
    prices.windows(2)
        .map(|w| if w[0] > 0.0 { crate::ln(w[1] / w[0]) } else { 0.0 })
        .collect()
}

/// Detect the current market regime from a return series.
pub fn regime_detect(returns: &[f64]) -> RegimeResult {
    let law = analyze(returns);
    let dfa_result = dfa(returns);
    let regime = if dfa_result.alpha > 0.6 {
        MarketRegime::Trending
    } else if dfa_result.alpha < 0.45 {
        MarketRegime::MeanReverting
    } else {
        MarketRegime::RandomWalk
    };
    RegimeResult {
        alpha: dfa_result.alpha,
        r_squared: dfa_result.r_squared,
        regime,
        n_returns: returns.len(),
        law,
    }
}

/// Sliding-window regime detector for real-time use.
pub struct RegimeMonitor {
    window: Vec<f64>,
    capacity: usize,
    pos: usize,
    filled: bool,
}

impl RegimeMonitor {
    pub fn new(window_size: usize) -> Self {
        RegimeMonitor {
            window: vec![0.0; window_size],
            capacity: window_size,
            pos: 0,
            filled: false,
        }
    }

    /// Push a new return value and get the current regime.
    pub fn push(&mut self, ret: f64) -> Option<RegimeResult> {
        self.window[self.pos] = ret;
        self.pos += 1;
        if self.pos >= self.capacity {
            self.pos = 0;
            self.filled = true;
        }
        if !self.filled { return None; }

        let mut ordered = Vec::with_capacity(self.capacity);
        ordered.extend_from_slice(&self.window[self.pos..]);
        ordered.extend_from_slice(&self.window[..self.pos]);
        Some(regime_detect(&ordered))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_from_prices_basic() {
        let prices = vec![100.0, 110.0, 121.0];
        let rets = returns_from_prices(&prices);
        assert_eq!(rets.len(), 2);
        assert!((rets[0] - 0.0953).abs() < 0.001); // ln(110/100)
    }

    #[test]
    fn regime_monitor_learns() {
        let mut mon = RegimeMonitor::new(256);
        let mut state = 42u64;
        for _ in 0..256 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let r = (state >> 33) as f64 / (1u64 << 31) as f64 - 0.5;
            let result = mon.push(r * 0.02);
            if let Some(r) = result {
                assert!(r.alpha > 0.0);
                return;
            }
        }
        panic!("monitor should have produced a result after 256 samples");
    }
}
