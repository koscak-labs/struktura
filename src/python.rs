use pyo3::prelude::*;
use pyo3::types::PyList;
use crate::{dfa, analyze, compare, anomaly_scores, health_check, HealthVerdict};

#[pyclass]
#[derive(Clone)]
pub struct PyDfaResult {
    #[pyo3(get)]
    pub alpha: f64,
    #[pyo3(get)]
    pub r_squared: f64,
}

#[pymethods]
impl PyDfaResult {
    fn __repr__(&self) -> String {
        format!("DfaResult(alpha={:.4}, r_squared={:.4})", self.alpha, self.r_squared)
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyAnalysis {
    #[pyo3(get)]
    pub alpha: f64,
    #[pyo3(get)]
    pub r_squared: f64,
    #[pyo3(get)]
    pub hurst: f64,
    #[pyo3(get)]
    pub mean: f64,
    #[pyo3(get)]
    pub std_dev: f64,
    #[pyo3(get)]
    pub kurtosis: f64,
    #[pyo3(get)]
    pub n: usize,
    #[pyo3(get)]
    pub quality: String,
}

#[pymethods]
impl PyAnalysis {
    fn __repr__(&self) -> String {
        format!("Analysis(alpha={:.4}, R²={:.4}, quality={})", self.alpha, self.r_squared, self.quality)
    }
}

#[pyfunction]
fn py_dfa(values: Vec<f64>) -> PyDfaResult {
    let r = dfa(&values);
    PyDfaResult { alpha: r.alpha, r_squared: r.r_squared }
}

#[pyfunction]
fn py_analyze(values: Vec<f64>) -> PyAnalysis {
    let law = analyze(&values);
    PyAnalysis {
        alpha: law.dfa.alpha,
        r_squared: law.dfa.r_squared,
        hurst: law.hurst,
        mean: law.mean,
        std_dev: law.std_dev,
        kurtosis: law.kurtosis,
        n: law.n,
        quality: format!("{}", law.quality),
    }
}

#[pyfunction]
fn py_compare(baseline: Vec<f64>, current: Vec<f64>) -> String {
    let r = compare(&baseline, &current);
    format!("{}", r)
}

#[pyfunction]
fn py_anomaly_scores(values: Vec<f64>, window: usize, step: usize, threshold: f64) -> Vec<f64> {
    anomaly_scores(&values, window, step, threshold)
}

#[pyfunction]
fn py_is_degraded(baseline: Vec<f64>, current: Vec<f64>) -> bool {
    crate::is_degraded(&baseline, &current)
}

#[pymodule]
fn struktura(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_dfa, m)?)?;
    m.add_function(wrap_pyfunction!(py_analyze, m)?)?;
    m.add_function(wrap_pyfunction!(py_compare, m)?)?;
    m.add_function(wrap_pyfunction!(py_anomaly_scores, m)?)?;
    m.add_function(wrap_pyfunction!(py_is_degraded, m)?)?;
    m.add_class::<PyDfaResult>()?;
    m.add_class::<PyAnalysis>()?;
    Ok(())
}
