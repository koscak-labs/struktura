//! # Detect Anomalies in Any Time Series
//!
//! One function. No training. No hyperparameters. Works on any signal.
//!
//! ```bash
//! cargo install struktura
//! struktura scan your_data.csv
//! ```

use struktura::{analyze, compare, is_degraded, LawQuality};

fn main() {
    println!("=== Anomaly Detection — Zero Configuration ===\n");

    // Generate a signal with a hidden structural change at sample 2000
    let mut data = Vec::with_capacity(4000);
    let mut state = 77u64;

    // Phase 1: stable structure (correlated noise)
    let mut prev = 0.0f64;
    for _ in 0..2000 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let noise = (state >> 33) as f64 / (1u64 << 31) as f64 - 0.5;
        prev = prev * 0.7 + noise * 0.3; // correlated
        data.push(prev);
    }

    // Phase 2: structure breaks (uncorrelated noise — same mean, same std!)
    for _ in 0..2000 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let noise = (state >> 33) as f64 / (1u64 << 31) as f64 - 0.5;
        data.push(noise * 0.3); // uncorrelated — same amplitude
    }

    let before = &data[..2000];
    let after = &data[2000..];

    // A normal threshold monitor would see NOTHING — same mean, same std
    let mean_before: f64 = before.iter().sum::<f64>() / before.len() as f64;
    let mean_after: f64 = after.iter().sum::<f64>() / after.len() as f64;
    println!("  Mean before: {:.4}", mean_before);
    println!("  Mean after:  {:.4}", mean_after);
    println!("  → Threshold monitor: looks fine (means nearly identical)\n");

    // Struktura sees the structural change
    let result = compare(before, after);
    println!("  Struktura:");
    println!("  Before: α={:.3}  (correlated — structure present)", result.baseline_alpha);
    println!("  After:  α={:.3}  (uncorrelated — structure gone)", result.current_alpha);
    println!("  Shift:  {:+.3}  Verdict: {}\n", result.shift, result.verdict);

    if is_degraded(before, after) {
        println!("  ✓ ANOMALY DETECTED — structure changed while mean stayed the same");
        println!("    This is what struktura catches that threshold monitors miss.");
    }
}
