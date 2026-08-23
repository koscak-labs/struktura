use crate::{dfa, analyze, health_check, DfaResult, StructuralLaw, HealthVerdict, LawQuality};
use core::slice;

/// C-compatible DFA result.
#[repr(C)]
pub struct CStructuralLaw {
    pub dfa_alpha: f64,
    pub dfa_r2: f64,
    pub acr_alpha: f64,
    pub acr_r2: f64,
    pub hurst: f64,
    pub mean: f64,
    pub std_dev: f64,
    pub kurtosis: f64,
    pub n: u32,
    pub quality: u8,
}

/// C-compatible DFA scaling result.
#[repr(C)]
pub struct CDfaResult {
    pub alpha: f64,
    pub r_squared: f64,
}

/// Compute DFA scaling exponent from a C array.
///
/// # Safety
///
/// `data` must point to a valid array of at least `len` f64 values.
#[no_mangle]
pub unsafe extern "C" fn struktura_dfa(data: *const f64, len: u32) -> CDfaResult {
    if data.is_null() || len < 64 {
        return CDfaResult { alpha: 0.5, r_squared: 0.0 };
    }
    let values = slice::from_raw_parts(data, len as usize);
    let result = dfa(values);
    CDfaResult { alpha: result.alpha, r_squared: result.r_squared }
}

/// Full structural analysis from a C array.
///
/// # Safety
///
/// `data` must point to a valid array of at least `len` f64 values.
#[no_mangle]
pub unsafe extern "C" fn struktura_analyze(data: *const f64, len: u32) -> CStructuralLaw {
    if data.is_null() || len < 20 {
        return CStructuralLaw {
            dfa_alpha: 0.5, dfa_r2: 0.0, acr_alpha: 0.0, acr_r2: 0.0,
            hurst: 0.5, mean: 0.0, std_dev: 0.0, kurtosis: 0.0,
            n: 0, quality: 5,
        };
    }
    let values = slice::from_raw_parts(data, len as usize);
    let law = analyze(values);
    let q = match law.quality {
        LawQuality::Exact => 0,
        LawQuality::Strong => 1,
        LawQuality::Good => 2,
        LawQuality::Approx => 3,
        LawQuality::Abstain => 4,
        LawQuality::Insufficient => 5,
    };
    CStructuralLaw {
        dfa_alpha: law.dfa.alpha, dfa_r2: law.dfa.r_squared,
        acr_alpha: law.acr.alpha, acr_r2: law.acr.r_squared,
        hurst: law.hurst, mean: law.mean, std_dev: law.std_dev,
        kurtosis: law.kurtosis, n: law.n as u32, quality: q,
    }
}

/// Compare current DFA alpha against baseline. Returns verdict (0-3).
///
/// # Safety
///
/// No pointer arguments. Safe to call from any context.
#[no_mangle]
pub unsafe extern "C" fn struktura_health_check(current_alpha: f64, baseline_alpha: f64) -> u8 {
    let law = StructuralLaw {
        hurst: 0.0, dfa: DfaResult { alpha: current_alpha, r_squared: 1.0 },
        acr: DfaResult { alpha: 0.0, r_squared: 0.0 },
        mean: 0.0, std_dev: 0.0, kurtosis: 0.0, p99: 0.0, max: 0.0, n: 0,
        quality: LawQuality::Exact,
    };
    match health_check(&law, baseline_alpha) {
        HealthVerdict::Healthy => 0,
        HealthVerdict::Watch => 1,
        HealthVerdict::Warning => 2,
        HealthVerdict::Critical => 3,
    }
}
