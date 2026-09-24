//! Python bindings for struktura. Every function converts its arguments and
//! calls the Rust crate; no computation lives here.

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use stk::monitor::{HybridMonitor, Leg};

/// A 1-D float series from Python. A C-contiguous float64 numpy array is read
/// in place (no copy); anything else goes through the generic sequence path.
/// numpy is only touched when the object already is an ndarray, so the
/// package works without numpy installed.
enum Series<'py> {
    Numpy(PyReadonlyArray1<'py, f64>),
    Owned(Vec<f64>),
}

impl<'py> FromPyObject<'py> for Series<'py> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if ob.get_type().name()?.to_string() == "ndarray" {
            if let Ok(arr) = ob.downcast::<PyArray1<f64>>() {
                if let Ok(ro) = arr.try_readonly() {
                    if ro.as_slice().is_ok() {
                        return Ok(Series::Numpy(ro));
                    }
                }
            }
        }
        Ok(Series::Owned(ob.extract::<Vec<f64>>()?))
    }
}

impl Series<'_> {
    fn as_slice(&self) -> &[f64] {
        match self {
            Series::Numpy(a) => a.as_slice().expect("contiguity checked at extraction"),
            Series::Owned(v) => v,
        }
    }
}

/// Result of detrended fluctuation analysis.
#[pyclass(frozen, get_all)]
#[derive(Clone)]
struct DfaResult {
    /// Scaling exponent: ~0.5 uncorrelated, ~1.0 1/f, ~1.5 Brownian.
    alpha: f64,
    /// Fit quality of log F(s) against log s.
    r_squared: f64,
}

#[pymethods]
impl DfaResult {
    fn __repr__(&self) -> String {
        format!("DfaResult(alpha={:.4}, r_squared={:.4})", self.alpha, self.r_squared)
    }
}

impl From<stk::DfaResult> for DfaResult {
    fn from(r: stk::DfaResult) -> Self {
        DfaResult { alpha: r.alpha, r_squared: r.r_squared }
    }
}

/// DFA of a series. Below 64 samples this returns the placeholder
/// alpha 0.5 with r_squared 0.0, which is not a measurement: use dfa_short.
#[pyfunction]
fn dfa(values: Series) -> DfaResult {
    stk::dfa(values.as_slice()).into()
}

/// DFA for short series (from about 24 samples). None when the series cannot
/// be measured (too short, constant, or too few usable box sizes).
#[pyfunction]
fn dfa_short(values: Series) -> Option<DfaResult> {
    stk::dfa_short(values.as_slice()).map(Into::into)
}

/// Structural summary of a series.
#[pyclass(frozen, get_all)]
#[derive(Clone)]
struct Analysis {
    alpha: f64,
    r_squared: f64,
    hurst: f64,
    mean: f64,
    std_dev: f64,
    kurtosis: f64,
    n: usize,
    quality: String,
}

#[pymethods]
impl Analysis {
    fn __repr__(&self) -> String {
        format!("Analysis(alpha={:.4}, r_squared={:.4}, quality={})", self.alpha, self.r_squared, self.quality)
    }
}

#[pyfunction]
fn analyze(values: Series) -> Analysis {
    let law = stk::analyze(values.as_slice());
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
#[pyfunction]
fn compare(baseline: Series, current: Series) -> String {
    stk::compare(baseline.as_slice(), current.as_slice()).to_string()
}

#[pyfunction]
fn is_degraded(baseline: Series, current: Series) -> bool {
    stk::is_degraded(baseline.as_slice(), current.as_slice())
}

#[pyfunction]
fn anomaly_scores(values: Series, window: usize, step: usize, threshold: f64) -> Vec<f64> {
    stk::anomaly_scores(values.as_slice(), window, step, threshold)
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

/// Self-calibrating streaming monitor (struktura's HybridMonitor).
///
///     m = Monitor([ch0_clean, ch1_clean])   # one list per channel, equal length
///     for sample in stream:                 # one value per channel
///         leg = m.push(sample)              # None, or the detector that fired
#[pyclass(unsendable)]
struct Monitor {
    inner: HybridMonitor,
}

#[pymethods]
impl Monitor {
    #[new]
    fn new(clean: Vec<Vec<f64>>) -> PyResult<Self> {
        HybridMonitor::calibrate(&clean)
            .map(|inner| Monitor { inner })
            .ok_or_else(|| {
                PyValueError::new_err(
                    "calibration failed: need at least one channel, equal lengths, and enough clean samples",
                )
            })
    }

    #[getter]
    fn channels(&self) -> usize {
        self.inner.channels()
    }

    /// Feed one sample (one value per channel). Returns None, or the name of
    /// the detector leg that raised an alarm.
    fn push(&mut self, sample: Vec<f64>) -> PyResult<Option<&'static str>> {
        if sample.len() != self.inner.channels() {
            return Err(PyValueError::new_err(format!(
                "sample has {} values, monitor has {} channels",
                sample.len(),
                self.inner.channels()
            )));
        }
        Ok(self.inner.push(&sample).map(leg_name))
    }

    /// The most recent alarm as a dict, or None.
    fn last_alarm<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, pyo3::types::PyDict>>> {
        let Some(r) = self.inner.last_alarm() else { return Ok(None) };
        let d = pyo3::types::PyDict::new(py);
        d.set_item("leg", leg_name(r.leg))?;
        d.set_item("channel", r.channel)?;
        d.set_item("tick", r.tick)?;
        d.set_item("observed", r.observed)?;
        d.set_item("threshold", r.threshold)?;
        d.set_item("explanation", stk::monitor::explain_alarm(&r))?;
        Ok(Some(d))
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

#[pymodule]
fn struktura(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(dfa, m)?)?;
    m.add_function(wrap_pyfunction!(dfa_short, m)?)?;
    m.add_function(wrap_pyfunction!(analyze, m)?)?;
    m.add_function(wrap_pyfunction!(compare, m)?)?;
    m.add_function(wrap_pyfunction!(is_degraded, m)?)?;
    m.add_function(wrap_pyfunction!(anomaly_scores, m)?)?;
    m.add_class::<DfaResult>()?;
    m.add_class::<Analysis>()?;
    m.add_class::<Monitor>()?;
    Ok(())
}
