//! # Analyze Your Git Commit Rhythm
//!
//! Are you a bursty creative or a metronomic machine?
//! DFA on your commit timestamps reveals your work pattern.
//!
//! ```bash
//! # Analyze any repo:
//! git log --format=%at | struktura rhythm -
//!
//! # Compare repos:
//! cd linux && git log --format=%at > /tmp/linux.csv
//! cd my-project && git log --format=%at > /tmp/mine.csv
//! struktura rhythm /tmp/linux.csv
//! struktura rhythm /tmp/mine.csv
//! ```

use struktura::rhythm::{intervals_from_timestamps, rhythm_analyze};

fn main() {
    println!("=== Git Commit Rhythm Analysis ===\n");
    println!("  Run on YOUR repo:");
    println!("    git log --format=%at | struktura rhythm -\n");

    // Simulate three developer patterns
    let mut state = 42u64;

    // Developer A: bursty (intense sessions with long breaks)
    let mut timestamps_a = vec![0.0f64];
    for _ in 0..500 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let r = (state >> 33) as f64 / (1u64 << 31) as f64;
        // 80% chance of short gap (in-session), 20% long gap (between sessions)
        let gap = if r < 0.8 { 60.0 + r * 300.0 } else { 3600.0 + r * 28800.0 };
        timestamps_a.push(timestamps_a.last().unwrap() + gap);
    }

    // Developer B: metronomic (CI bot / scheduled commits)
    let mut timestamps_b = vec![0.0f64];
    for i in 0..500 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let jitter = (state >> 33) as f64 / (1u64 << 31) as f64 * 60.0;
        timestamps_b.push(3600.0 * (i + 1) as f64 + jitter); // hourly ± 1min
    }

    // Developer C: random (sporadic contributor)
    let mut timestamps_c = vec![0.0f64];
    for _ in 0..500 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let gap = (state >> 33) as f64 / (1u64 << 31) as f64 * 86400.0;
        timestamps_c.push(timestamps_c.last().unwrap() + gap);
    }

    for (name, ts) in [
        ("Bursty developer (deep work sessions)", &timestamps_a),
        ("Metronomic (CI bot / cron)", &timestamps_b),
        ("Random contributor", &timestamps_c),
    ] {
        let intervals = intervals_from_timestamps(ts);
        let result = rhythm_analyze(&intervals);
        println!("  {name}");
        println!("    α={:.3}  R²={:.4}  rhythm={}  mean_gap={:.0}s",
            result.alpha, result.r_squared, result.rhythm, result.mean_interval);
        println!();
    }
}
