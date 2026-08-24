//! cardiac HRV analysis — detect heart rate variability changes
//!
//! healthy hearts have α ≈ 1.0 (complex long-range correlations)
//! heart failure drops α toward 0.5 (loss of fractal structure)
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
            // healthy: correlated noise with long-range structure
            prev = prev * 0.85 + 800.0 * 0.15 + noise * 40.0;
        } else {
            // heart failure: uncorrelated, reduced variability
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
            .filter_map(|l| l.trim().split(',').last()?.trim().parse().ok())
            .collect();
        println!("  loaded {} RR intervals from {}", data.len(), args[1]);
        let law = analyze(&data);
        println!("  α = {:.3}  R² = {:.4}  quality = {}", law.dfa.alpha, law.dfa.r_squared, law.quality);
        if law.dfa.alpha > 0.8 {
            println!("  interpretation: healthy fractal structure 💚");
        } else if law.dfa.alpha > 0.6 {
            println!("  interpretation: reduced complexity ⚠️ (consider medical consultation)");
        } else {
            println!("  interpretation: significant loss of fractal structure 🔴");
        }
        return;
    } else {
        (synth_rr_intervals(2048, 42, true), synth_rr_intervals(2048, 42, false))
    };

    println!();
    println!("  CARDIAC HRV — STRUCTURAL HEALTH ANALYSIS");
    println!("  ================================================================");

    let law_h = analyze(&healthy_rr);
    let law_u = analyze(&unhealthy_rr);

    println!();
    println!("  healthy heart     α = {:.3}  R² = {:.4}  H = {:.3}", law_h.dfa.alpha, law_h.dfa.r_squared, law_h.hurst);
    println!("  heart failure     α = {:.3}  R² = {:.4}  H = {:.3}", law_u.dfa.alpha, law_u.dfa.r_squared, law_u.hurst);

    let shift = law_u.dfa.alpha - law_h.dfa.alpha;
    let verdict = health_check(&law_u, law_h.dfa.alpha);
    println!("                    shift = {:.3}  {:?}", shift, verdict);

    println!();
    println!("  α ≈ 1.0 = healthy complex dynamics (the heart adapts)");
    println!("  α ≈ 0.5 = loss of fractal structure (the heart lost flexibility)");
    println!("  this is a screening tool, not a diagnosis.");
    println!();
}
