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

/// One decision or observation from a Guard. `kind` is "alarm", "quarantined",
/// "adaptation_started", "recalibrated" or "rolled_back"; the alarm fields are
/// set for "alarm" and "rolled_back", `channel` also for "quarantined".
#[wasm_bindgen]
pub struct GuardEvent {
    kind: &'static str,
    pub tick: f64,
    channel: Option<usize>,
    leg: Option<&'static str>,
    class: Option<&'static str>,
    explanation: Option<&'static str>,
    observed: Option<f64>,
    threshold: Option<f64>,
}

#[wasm_bindgen]
impl GuardEvent {
    #[wasm_bindgen(getter)]
    pub fn kind(&self) -> String {
        self.kind.to_string()
    }
    #[wasm_bindgen(getter)]
    pub fn channel(&self) -> Option<usize> {
        self.channel
    }
    #[wasm_bindgen(getter)]
    pub fn leg(&self) -> Option<String> {
        self.leg.map(str::to_string)
    }
    #[wasm_bindgen(getter)]
    pub fn class(&self) -> Option<String> {
        self.class.map(str::to_string)
    }
    #[wasm_bindgen(getter)]
    pub fn explanation(&self) -> Option<String> {
        self.explanation.map(str::to_string)
    }
    #[wasm_bindgen(getter)]
    pub fn observed(&self) -> Option<f64> {
        self.observed
    }
    #[wasm_bindgen(getter)]
    pub fn threshold(&self) -> Option<f64> {
        self.threshold
    }
}

impl GuardEvent {
    fn bare(kind: &'static str, tick: u64) -> GuardEvent {
        GuardEvent { kind, tick: tick as f64, channel: None, leg: None, class: None, explanation: None, observed: None, threshold: None }
    }

    fn with_report(kind: &'static str, tick: u64, r: &stk::monitor::AlarmReport) -> GuardEvent {
        GuardEvent {
            channel: Some(r.channel),
            leg: Some(leg_name(r.leg)),
            explanation: Some(stk::monitor::explain_alarm(r)),
            observed: Some(r.observed),
            threshold: Some(r.threshold),
            ..GuardEvent::bare(kind, tick)
        }
    }
}

/// What `struktura guard` runs: the monitor inside struktura's AutoPilot. It keeps
/// watching after an alarm, quarantines a dead channel, and recalibrates after a
/// level shift that settles into a new normal (rolling back if that fails).
///
///   const g = new Guard(cleanFlat, channels);
///   for (const e of g.push(sample)) console.log(e.kind, e.explanation); // [] in the steady state
#[wasm_bindgen]
pub struct Guard {
    inner: stk::autopilot::AutoPilot,
    channels: usize,
    cooldown: u64,
    tick: u64,
    recent: Vec<(u64, u8)>,
}

#[wasm_bindgen]
impl Guard {
    /// Calibrate on clean data, laid out as for Monitor (channel-major).
    /// `cooldown` (default 50, as in the CLI): an alarm from the same detector within
    /// this many samples of its previous one is not reported again; 0 reports all.
    #[wasm_bindgen(constructor)]
    pub fn new(clean: &[f64], channels: usize, cooldown: Option<u32>) -> Result<Guard, JsError> {
        let m = Monitor::new(clean, channels)?.inner;
        Ok(Guard {
            inner: stk::autopilot::AutoPilot::new(m),
            channels,
            cooldown: u64::from(cooldown.unwrap_or(50)),
            tick: 0,
            recent: Vec::new(),
        })
    }

    #[wasm_bindgen(getter)]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Feed one sample (one value per channel; NaN counts as a missing reading).
    /// Returns the events this sample produced, usually none.
    pub fn push(&mut self, sample: &[f64]) -> Result<Vec<GuardEvent>, JsError> {
        use stk::autopilot::Event;
        if sample.len() != self.channels {
            return Err(JsError::new(&format!(
                "sample has {} values, guard has {} channels",
                sample.len(),
                self.channels
            )));
        }
        let valid: Vec<bool> = sample.iter().map(|v| v.is_finite()).collect();
        let clean: Vec<f64> = sample.iter().map(|v| if v.is_finite() { *v } else { 0.0 }).collect();
        let (t, c) = (self.tick, self.cooldown);
        self.tick += 1;
        let recent = &mut self.recent;
        Ok(self
            .inner
            .push(&clean, &valid)
            .into_iter()
            .filter(|ev| {
                // Same rule as the CLI's guard output: drop a repeat of the same detector.
                let Event::Alarm { report, .. } = ev else { return true };
                let leg = report.leg as u8;
                let dup = recent.iter().any(|&(lt, l)| l == leg && t - lt < c);
                recent.retain(|&(lt, _)| t - lt < c);
                recent.push((t, leg));
                !dup
            })
            .map(|ev| match ev {
                Event::Alarm { tick, report, class } => GuardEvent { class: Some(class), ..GuardEvent::with_report("alarm", tick, &report) },
                Event::Quarantined { tick, channel } => GuardEvent { channel: Some(channel), ..GuardEvent::bare("quarantined", tick) },
                Event::Unquarantined { tick, channel } => GuardEvent { channel: Some(channel), ..GuardEvent::bare("unquarantined", tick) },
                Event::AdaptationStarted { tick } => GuardEvent::bare("adaptation_started", tick),
                Event::Recalibrated { tick } => GuardEvent::bare("recalibrated", tick),
                Event::RolledBack { tick, guard_report } => GuardEvent::with_report("rolled_back", tick, &guard_report),
            })
            .collect())
    }
}
