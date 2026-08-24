//! # Predict Server Crashes Before They Happen
//!
//! Monitor CPU load, memory pressure, or disk I/O — struktura detects
//! when the STRUCTURE of your metrics changes, before averages or
//! thresholds fire.
//!
//! ```bash
//! # Linux: pipe vmstat into struktura
//! vmstat 1 300 | awk '{print $13}' | struktura scan -
//!
//! # Or analyze a CSV export from Prometheus/Grafana:
//! struktura scan cpu_load.csv --baseline healthy_baseline.csv
//! ```

use struktura::{compare, is_degraded, BaselineTracker, HealthVerdict};

fn main() {
    println!("=== Predict Server Crashes ===\n");

    // Simulate: 2000 samples of healthy CPU, then 500 of degrading
    let mut healthy: Vec<f64> = Vec::new();
    let mut state = 42u64;
    for _ in 0..2000 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let noise = (state >> 33) as f64 / (1u64 << 31) as f64;
        healthy.push(35.0 + noise * 15.0 + (state as f64 / 1e18).sin() * 5.0);
    }

    let mut degraded = healthy[1500..].to_vec();
    for i in 0..500 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let noise = (state >> 33) as f64 / (1u64 << 31) as f64;
        let load = 35.0 + noise * 15.0 + (i as f64 * 0.05).powi(2) + i as f64 * 0.1;
        degraded.push(load);
    }

    let result = compare(&healthy[..1000], &degraded);
    println!("  Baseline (healthy):  α={:.3}", result.baseline_alpha);
    println!("  Current (degrading): α={:.3}", result.current_alpha);
    println!("  Shift: {:+.3}  Verdict: {}", result.shift, result.verdict);
    println!();

    if is_degraded(&healthy[..1000], &degraded) {
        println!("  ⚠ SERVER STRUCTURE CHANGED — investigate before it crashes");
    } else {
        println!("  ✓ Server structure stable");
    }

    // Real-time streaming version
    println!("\n=== Streaming Monitor ===\n");
    let mut tracker = BaselineTracker::new(256, 1000);
    let mut alerts = 0;
    for (i, &v) in healthy.iter().chain(degraded.iter()).enumerate() {
        if let Some(verdict) = tracker.push(v) {
            if verdict != HealthVerdict::Healthy {
                if alerts < 3 {
                    println!("  Alert at sample {}: {:?}", i, verdict);
                }
                alerts += 1;
            }
        }
    }
    if alerts > 3 {
        println!("  ... and {} more alerts", alerts - 3);
    }
    println!("\n  Total alerts: {} (out of {} samples)", alerts, healthy.len() + degraded.len());
}
