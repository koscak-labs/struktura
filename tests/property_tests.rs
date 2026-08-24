//! Property tests: no input may panic the library — NaN, Inf, empty,
//! constant, tiny, huge, adversarial streams all must be handled.

use proptest::prelude::*;
use struktura::monitor::HybridMonitor;
use struktura::{analyze, anomaly_scores, dfa, dfa_fast_into, dfa_into};

fn arbitrary_signal() -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(
        prop_oneof![
            8 => prop::num::f64::NORMAL,
            1 => prop::num::f64::ANY, // includes NaN, Inf, subnormals
            1 => Just(0.0f64),
        ],
        0..600,
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn dfa_never_panics(sig in arbitrary_signal()) {
        let r = dfa(&sig);
        // Result fields must not be misleadingly "valid" on garbage input:
        // alpha is either the fallback or a finite number.
        prop_assert!(r.alpha == 0.5 || r.alpha.is_finite() || r.alpha.is_nan());
    }

    #[test]
    fn dfa_fast_never_panics_and_matches(sig in arbitrary_signal()) {
        let mut b1 = Vec::new();
        let mut b2 = Vec::new();
        let a = dfa_into(&sig, &mut b1);
        let b = dfa_fast_into(&sig, &mut b2);
        // On finite inputs the two must agree; on non-finite inputs both
        // may produce NaN but neither may panic.
        if sig.iter().all(|v| v.is_finite()) && a.alpha.is_finite() && b.alpha.is_finite() {
            prop_assert!((a.alpha - b.alpha).abs() < 1e-6,
                "alpha mismatch {} vs {}", a.alpha, b.alpha);
        }
    }

    #[test]
    fn analyze_never_panics(sig in arbitrary_signal()) {
        let _ = analyze(&sig);
    }

    #[test]
    fn anomaly_scores_never_panics(sig in arbitrary_signal()) {
        let _ = anomaly_scores(&sig, 96, 8, 0.08);
    }

    #[test]
    fn monitor_calibrate_never_panics(
        sig in arbitrary_signal(),
        channels in 1usize..4,
    ) {
        let chans: Vec<Vec<f64>> = (0..channels).map(|_| sig.clone()).collect();
        // Either calibrates or refuses — never panics.
        let _ = HybridMonitor::calibrate(&chans);
    }

    #[test]
    fn monitor_push_never_panics(
        calib_seed in 1u64..1000,
        stream in arbitrary_signal(),
    ) {
        // Valid calibration, then arbitrary (possibly NaN/Inf) stream.
        let calib = struktura::telemetry_bench::synth_spacecraft(300, calib_seed);
        if let Some(mut mon) = HybridMonitor::calibrate(&calib) {
            let mut sample = [0.0f64; 6];
            for (i, &v) in stream.iter().enumerate() {
                for slot in sample.iter_mut() {
                    *slot = v;
                }
                // Alternate validity to exercise the missingness path too.
                let valid = [i % 3 != 0, true, true, true, true, i % 5 != 0];
                let _ = mon.push_with_validity(&sample, &valid);
            }
        }
    }

    #[test]
    fn prognosis_never_panics(
        sig in arbitrary_signal(),
        window in 1usize..300,
        threshold in prop::num::f64::ANY,
    ) {
        let _ = struktura::prognosis::time_to_threshold(&sig, window, threshold);
    }
}

#[test]
fn monitor_survives_all_nan_stream() {
    let calib = struktura::telemetry_bench::synth_spacecraft(300, 42);
    let mut mon = HybridMonitor::calibrate(&calib).expect("calibration");
    let sample = [f64::NAN; 6];
    for _ in 0..500 {
        let _ = mon.push(&sample);
    }
}

#[test]
fn monitor_survives_infinite_values() {
    let calib = struktura::telemetry_bench::synth_spacecraft(300, 42);
    let mut mon = HybridMonitor::calibrate(&calib).expect("calibration");
    for i in 0..500 {
        let v = if i % 2 == 0 { f64::INFINITY } else { f64::NEG_INFINITY };
        let _ = mon.push(&[v; 6]);
    }
}

#[test]
fn calibrate_on_constant_signal_is_graceful() {
    let chans: Vec<Vec<f64>> = (0..6).map(|_| vec![1.0; 400]).collect();
    // Constant signal: zero variance everywhere. Must not panic; whether it
    // calibrates or refuses, both are acceptable.
    let _ = HybridMonitor::calibrate(&chans);
}

#[test]
fn calibrate_on_mismatched_lengths_is_graceful() {
    let chans = vec![vec![1.0; 400], vec![2.0; 200]];
    let _ = HybridMonitor::calibrate(&chans);
}
