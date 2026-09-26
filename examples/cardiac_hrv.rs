//! cardiac HRV analysis: DFA exponent of RR (beat-to-beat) intervals
//!
//! In heart-rate studies the short-term exponent (alpha1, about 4-16
//! beats) is lower in heart failure than in healthy hearts, and the
//! long-term exponent changes much less (Peng et al. 1995, Chaos 5:82).
//! It does not drop to 0.5 (no correlation). This example is a DFA demo,
//! not a medical device, and it prints no diagnosis.
//!
//! pipe your Apple Watch / Garmin / Polar HRV export through this:
//!   cargo run --example cardiac_hrv -- hrv_export.csv
//!
//! or use the built-in synthetic demo:
//!   cargo run --example cardiac_hrv

use struktura::{analyze, health_check};

fn synth_rr_intervals(n: usize, seed: u64, healthy: bool) -> Vec<f64> {
    let mut state = seed;
    let mut rr = Vec::with_capacity(n);
    let mut prev = 800.0; // ms, typical RR interval
    for _ in 0..n {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let noise = (state >> 33) as f64 / (1u64 << 31) as f64 - 0.5;
        if healthy {
            // correlated series (short-range memory)
            prev = prev * 0.85 + 800.0 * 0.15 + noise * 40.0;
        } else {
            // uncorrelated series, smaller spread
            prev = 800.0 + noise * 15.0;
        }
        rr.push(prev);
    }
    rr
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let (healthy_rr, unhealthy_rr) = if args.len() > 1 {
        let data: Vec<f64> = std::fs::read_to_string(&args[1])
            .expect("cannot read file")
            .lines()
            .filter_map(|l| l.trim().split(',').next_back()?.trim().parse().ok())
            .collect();
        println!("  loaded {} RR intervals from {}", data.len(), args[1]);
        let law = analyze(&data);
        println!("  α = {:.3}  R² = {:.4}  quality = {}", law.dfa.alpha, law.dfa.r_squared, law.quality);
        println!("  this is the whole-record exponent, not the short-term alpha1 used in");
        println!("  heart-rate studies, and it is not a diagnosis.");
        return;
    } else {
        (synth_rr_intervals(2048, 42, true), synth_rr_intervals(2048, 42, false))
    };

    println!();
    println!("  RR INTERVALS: DFA ON TWO SYNTHETIC SERIES");
    println!("  ================================================================");

    let law_h = analyze(&healthy_rr);
    let law_u = analyze(&unhealthy_rr);

    println!();
    println!("  correlated series    α = {:.3}  R² = {:.4}  H = {:.3}", law_h.dfa.alpha, law_h.dfa.r_squared, law_h.hurst);
    println!("  uncorrelated series  α = {:.3}  R² = {:.4}  H = {:.3}", law_u.dfa.alpha, law_u.dfa.r_squared, law_u.hurst);

    let shift = law_u.dfa.alpha - law_h.dfa.alpha;
    let verdict = health_check(&law_u, law_h.dfa.alpha);
    println!("                       shift = {:.3}  {:?}", shift, verdict);

    println!();
    println!("  synthetic series only: they show that DFA separates correlated from");
    println!("  uncorrelated beat sequences, not what a failing heart looks like.");
    println!("  In heart-rate studies heart failure lowers the short-term exponent");
    println!("  (alpha1); it does not drop to 0.5. This is not a medical device.");
    println!();
}
