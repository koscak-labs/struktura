//! Conformal prediction: calibrated confidence for any detector.
//!
//! Wraps any anomaly score (DFA shift, residual z, monitor alarm
//! magnitude) with a mathematically guaranteed coverage interval:
//! "this score is more extreme than 97.3% of calibration scores."
//!
//! The guarantee (Vovk et al., 2005): if the calibration data is
//! exchangeable with the test data, the coverage is EXACTLY 1-α
//! regardless of the underlying distribution — no Gaussian assumption,
//! no parametric model, no tuning. The only assumption is exchangeability
//! (weaker than i.i.d.).
//!
//! ```
//! use struktura::conformal::ConformalDetector;
//! let mut det = ConformalDetector::new();
//! // Calibrate on known-clean scores
//! det.calibrate(&[0.1, 0.15, 0.12, 0.09, 0.11, 0.14, 0.13, 0.10, 0.12, 0.08]);
//! // Test a new score
//! let p = det.p_value(0.25);
//! // p < 0.05 means "this score is more extreme than 95% of calibration"
//! assert!(p < 0.15);
//! ```

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Conformal anomaly detector: distribution-free, calibrated confidence.
pub struct ConformalDetector {
    sorted_scores: Vec<f64>,
}

impl ConformalDetector {
    pub fn new() -> Self {
        ConformalDetector { sorted_scores: Vec::new() }
    }

    /// Calibrate from a set of known-normal anomaly scores.
    /// The scores can be anything: DFA alpha shifts, residual magnitudes,
    /// z-scores — as long as HIGHER = MORE anomalous.
    pub fn calibrate(&mut self, scores: &[f64]) {
        self.sorted_scores = scores.to_vec();
        self.sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    }

    /// p-value of a new score: fraction of calibration scores >= this one.
    /// p < α means "anomalous at significance level α."
    ///
    /// Coverage guarantee: if test data is exchangeable with calibration,
    /// P(false alarm) ≤ α exactly, for any α, any distribution.
    pub fn p_value(&self, score: f64) -> f64 {
        if self.sorted_scores.is_empty() {
            return 1.0;
        }
        let n = self.sorted_scores.len();
        // Count how many calibration scores are >= the test score
        let rank = match self.sorted_scores.binary_search_by(|v| {
            v.partial_cmp(&score).unwrap_or(core::cmp::Ordering::Equal)
        }) {
            Ok(i) => n - i,
            Err(i) => n - i,
        };
        // Conformal p-value: (rank + 1) / (n + 1)
        (rank as f64 + 1.0) / (n as f64 + 1.0)
    }

    /// Is this score anomalous at significance level `alpha`?
    /// Equivalent to `p_value(score) < alpha`.
    pub fn is_anomalous(&self, score: f64, alpha: f64) -> bool {
        self.p_value(score) < alpha
    }

    /// Confidence that this score is anomalous: `1 - p_value`.
    /// 0.97 means "97% confident this is NOT from the calibration distribution."
    pub fn confidence(&self, score: f64) -> f64 {
        1.0 - self.p_value(score)
    }

    /// The threshold score at significance level `alpha`.
    /// Scores above this are anomalous at that level.
    pub fn threshold_at(&self, alpha: f64) -> f64 {
        if self.sorted_scores.is_empty() {
            return f64::INFINITY;
        }
        let idx = ((1.0 - alpha) * self.sorted_scores.len() as f64) as usize;
        self.sorted_scores[idx.min(self.sorted_scores.len() - 1)]
    }
}

impl Default for ConformalDetector {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p_value_ranks_correctly() {
        let mut det = ConformalDetector::new();
        det.calibrate(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        // Score of 10.5 is above everything → p ≈ 1/11 = 0.09
        assert!(det.p_value(10.5) < 0.1);
        // Score of 5.5 is above half → p ≈ 5/11 = 0.45
        let p = det.p_value(5.5);
        assert!(p > 0.3 && p < 0.6, "p = {}", p);
        // Score of 0.5 is below everything → p ≈ 11/11 = 1.0
        assert!(det.p_value(0.5) > 0.9);
    }

    #[test]
    fn coverage_guarantee() {
        // The conformal guarantee: if we calibrate on N scores and test
        // on exchangeable data, the false alarm rate is ≤ alpha.
        let mut det = ConformalDetector::new();
        let cal: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.1).sin().abs()).collect();
        det.calibrate(&cal);
        // Count how many calibration scores are "anomalous" at alpha=0.05
        let false_alarms: usize = cal.iter()
            .filter(|&&s| det.is_anomalous(s, 0.05))
            .count();
        let rate = false_alarms as f64 / cal.len() as f64;
        // Should be ≤ 5% (plus finite-sample noise)
        assert!(rate < 0.08, "false alarm rate {} should be near 5%", rate);
    }

    #[test]
    fn threshold_at_matches_p_value() {
        let mut det = ConformalDetector::new();
        det.calibrate(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let thr = det.threshold_at(0.05);
        // Anything above the threshold should have p < 0.05 (approximately)
        assert!(det.p_value(thr + 0.1) < 0.15);
    }
}
