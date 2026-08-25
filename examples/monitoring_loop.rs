//! # Production Monitoring Loop
//!
//! Shows how to use struktura as a library in a continuous monitoring
//! system — like a Prometheus exporter or a health-check daemon.
//!
//! This pattern works for:
//! - Server CPU/memory monitoring
//! - IoT sensor networks
//! - Spacecraft telemetry
//! - Financial market surveillance
//! - Industrial equipment health

use struktura::{BaselineTracker, HealthVerdict, compare};

fn main() {
    println!("=== Production Monitoring Pattern ===\n");

    // Simulate a sensor that degrades over time
    let mut state = 42u64;
    let mut tracker = BaselineTracker::new(256, 500);
    let mut alerts = Vec::new();

    println!("  Phase 1: Learning baseline (500 samples)...");
    let mut prev = 0.0f64;
    for i in 0..2000 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let noise = (state >> 33) as f64 / (1u64 << 31) as f64 - 0.5;

        // Correlated signal (brownian-like) with degradation at sample 1000
        let correlation = if i > 1000 {
            // Structure breaks down — becomes uncorrelated
            let progress = (i - 1000) as f64 / 500.0;
            (0.8 - progress * 0.6).max(0.0)
        } else {
            0.8 // healthy: strong correlation
        };
        prev = prev * correlation + noise * (1.0 - correlation);
        let value = 50.0 + prev * 10.0;

        if let Some(verdict) = tracker.push(value) {
            match verdict {
                HealthVerdict::Healthy => {}
                v => {
                    if alerts.len() < 5 {
                        println!("  [sample {}] ALERT: {:?}", i, v);
                    }
                    alerts.push((i, v));
                }
            }
        }
    }

    if alerts.len() > 5 {
        println!("  ... {} more alerts", alerts.len() - 5);
    }

    let first_alert: usize = alerts.first().map(|(i, _)| *i).unwrap_or(0);
    let degradation_start: usize = 1000;

    println!("\n  Results:");
    println!("    Degradation started at:  sample {}", degradation_start);
    println!("    First alert at:          sample {}", first_alert);
    println!("    Detection lag:           {} samples", first_alert.saturating_sub(degradation_start));
    println!("    Total alerts:            {}", alerts.len());

    // Batch comparison pattern — useful for periodic health checks
    println!("\n=== Batch Comparison Pattern ===\n");

    let baseline: Vec<f64> = (0..1024).map(|i| {
        let mut s = (i as u64).wrapping_mul(6364136223846793005).wrapping_add(42);
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 33) as f64 / (1u64 << 31) as f64
    }).collect();

    let current: Vec<f64> = (0..1024).map(|i| {
        let mut s = (i as u64).wrapping_mul(6364136223846793005).wrapping_add(99);
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let v = (s >> 33) as f64 / (1u64 << 31) as f64;
        v + (i as f64 * 0.001) // slight drift
    }).collect();

    let result = compare(&baseline, &current);
    println!("  {}", result);
    println!("  Action: {}", match result.verdict {
        HealthVerdict::Healthy => "None — system nominal",
        HealthVerdict::Watch => "Log and monitor more frequently",
        HealthVerdict::Warning => "Alert on-call engineer",
        HealthVerdict::Critical => "Page immediately, prepare for failover",
        _ => "Unknown",
    });
}
