use wasm_bindgen::prelude::*;
use crate::{dfa, analyze, compare, anomaly_scores};

#[wasm_bindgen]
pub struct WasmDfaResult {
    pub alpha: f64,
    pub r_squared: f64,
}

#[wasm_bindgen]
pub struct WasmAnalysis {
    pub alpha: f64,
    pub r_squared: f64,
    pub hurst: f64,
    pub mean: f64,
    pub std_dev: f64,
    pub kurtosis: f64,
    pub n: usize,
    quality: String,
}

#[wasm_bindgen]
impl WasmAnalysis {
    #[wasm_bindgen(getter)]
    pub fn quality(&self) -> String { self.quality.clone() }
}

#[wasm_bindgen]
pub fn struktura_dfa(values: &[f64]) -> WasmDfaResult {
    let r = dfa(values);
    WasmDfaResult { alpha: r.alpha, r_squared: r.r_squared }
}

#[wasm_bindgen]
pub fn struktura_analyze(values: &[f64]) -> WasmAnalysis {
    let law = analyze(values);
    WasmAnalysis {
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

#[wasm_bindgen]
pub fn struktura_compare(baseline: &[f64], current: &[f64]) -> String {
    let r = compare(baseline, current);
    format!("{}", r)
}

#[wasm_bindgen]
pub fn struktura_scores(values: &[f64], window: usize, step: usize, threshold: f64) -> Vec<f64> {
    anomaly_scores(values, window, step, threshold)
}
