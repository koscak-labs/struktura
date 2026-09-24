//! JavaScript/WebAssembly bindings for struktura. Every function converts its
//! arguments and calls the Rust crate; no computation lives here.

use stk::monitor::{HybridMonitor, Leg};
use wasm_bindgen::prelude::*;

/// Result of detrended fluctuation analysis.
#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct DfaResult {
    /// Scaling exponent: ~0.5 uncorrelated, ~1.0 1/f, ~1.5 Brownian.
    pub alpha: f64,
    /// Fit quality of log F(s) against log s.
    #[wasm_bindgen(js_name = rSquared)]
    pub r_squared: f64,
}

impl From<stk::DfaResult> for DfaResult {
    fn from(r: stk::DfaResult) -> Self {
        DfaResult { alpha: r.alpha, r_squared: r.r_squared }
    }
}

/// DFA of a series. Below 64 samples this returns the placeholder alpha 0.5
/// with rSquared 0, which is not a measurement: use dfaShort.
#[wasm_bindgen]
pub fn dfa(values: &[f64]) -> DfaResult {
    stk::dfa(values).into()
}

/// DFA for short series (from about 24 samples). undefined when the series
/// cannot be measured (too short, constant, or too few usable box sizes).
#[wasm_bindgen(js_name = dfaShort)]
pub fn dfa_short(values: &[f64]) -> Option<DfaResult> {
    stk::dfa_short(values).map(Into::into)
}

/// Structural summary of a series.
#[wasm_bindgen]
pub struct Analysis {
    pub alpha: f64,
    #[wasm_bindgen(js_name = rSquared)]
    pub r_squared: f64,
    pub hurst: f64,
    pub mean: f64,
    #[wasm_bindgen(js_name = stdDev)]
    pub std_dev: f64,
    pub kurtosis: f64,
    pub n: usize,
    quality: String,
}

#[wasm_bindgen]
impl Analysis {
    #[wasm_bindgen(getter)]
    pub fn quality(&self) -> String {
        self.quality.clone()
    }
}

#[wasm_bindgen]
pub fn analyze(values: &[f64]) -> Analysis {
    let law = stk::analyze(values);
    Analysis {
        alpha: law.dfa.alpha,
        r_squared: law.dfa.r_squared,
        hurst: law.hurst,
        mean: law.mean,
        std_dev: law.std_dev,
        kurtosis: law.kurtosis,
        n: law.n,
        quality: law.quality.to_string(),
    }
}

/// Human-readable comparison of two series' structure.
#[wasm_bindgen]
pub fn compare(baseline: &[f64], current: &[f64]) -> String {
    stk::compare(baseline, current).to_string()
}

#[wasm_bindgen(js_name = isDegraded)]
pub fn is_degraded(baseline: &[f64], current: &[f64]) -> bool {
    stk::is_degraded(baseline, current)
}

#[wasm_bindgen(js_name = anomalyScores)]
pub fn anomaly_scores(values: &[f64], window: usize, step: usize, threshold: f64) -> Vec<f64> {
    stk::anomaly_scores(values, window, step, threshold)
}

fn leg_name(leg: Leg) -> &'static str {
    match leg {
        Leg::Residual => "residual",
        Leg::RepeatedValue => "repeated_value",
        Leg::Dfa => "dfa",
        Leg::LevelShift => "level_shift",
        Leg::ResidualCusum => "residual_cusum",
        Leg::Missingness => "missingness",
        Leg::Parity => "parity",
    }
}

/// The most recent alarm raised by a Monitor.
#[wasm_bindgen]
pub struct Alarm {
    leg: &'static str,
    pub channel: usize,
    pub tick: f64,
    pub observed: f64,
    pub threshold: f64,
    explanation: &'static str,
}

#[wasm_bindgen]
impl Alarm {
    #[wasm_bindgen(getter)]
    pub fn leg(&self) -> String {
        self.leg.to_string()
    }
    #[wasm_bindgen(getter)]
    pub fn explanation(&self) -> String {
        self.explanation.to_string()
    }
}

/// Self-calibrating streaming monitor (struktura's HybridMonitor).
///
///   const m = new Monitor(cleanFlat, channels); // cleanFlat: channel-major, channels * n values
///   const leg = m.push(sample);                 // sample: one value per channel; undefined or leg name
#[wasm_bindgen]
pub struct Monitor {
    inner: HybridMonitor,
}

#[wasm_bindgen]
impl Monitor {
    /// Calibrate on clean data. `clean` holds `channels` equal-length series
    /// back to back (channel 0's samples, then channel 1's, ...).
    #[wasm_bindgen(constructor)]
    pub fn new(clean: &[f64], channels: usize) -> Result<Monitor, JsError> {
        if channels == 0 || clean.len() % channels != 0 {
            return Err(JsError::new("clean.length must be a positive multiple of channels"));
        }
        let n = clean.len() / channels;
        let rows: Vec<Vec<f64>> = clean.chunks(n).map(<[f64]>::to_vec).collect();
        HybridMonitor::calibrate(&rows)
            .map(|inner| Monitor { inner })
            .ok_or_else(|| JsError::new("calibration failed: need enough clean samples per channel"))
    }

    #[wasm_bindgen(getter)]
    pub fn channels(&self) -> usize {
        self.inner.channels()
    }

    /// Feed one sample (one value per channel). Returns undefined, or the name
    /// of the detector leg that raised an alarm.
    pub fn push(&mut self, sample: &[f64]) -> Result<Option<String>, JsError> {
        if sample.len() != self.inner.channels() {
            return Err(JsError::new(&format!(
                "sample has {} values, monitor has {} channels",
                sample.len(),
                self.inner.channels()
            )));
        }
        Ok(self.inner.push(sample).map(|l| leg_name(l).to_string()))
    }

    /// The most recent alarm, or undefined.
    #[wasm_bindgen(js_name = lastAlarm)]
    pub fn last_alarm(&self) -> Option<Alarm> {
        self.inner.last_alarm().map(|r| Alarm {
            leg: leg_name(r.leg),
            channel: r.channel,
            tick: r.tick as f64,
            observed: r.observed,
            threshold: r.threshold,
            explanation: stk::monitor::explain_alarm(&r),
        })
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }
}
