//! GNSS satellite signal quality monitoring
//!
//! DFA detects ionospheric scintillation — when the atmosphere disrupts
//! GPS/Galileo signal structure. α shifts when scintillation degrades
//! positioning accuracy, often before the receiver reports a problem.
//!
//! real use: pipe RINEX observation data or receiver SNR logs through this.
//!   cargo run --example gnss_signal -- snr_log.csv
//!
//! synthetic demo:
//!   cargo run --example gnss_signal

use struktura::{analyze, compare};

fn synth_gnss_signal(n: usize, seed: u64, scintillation: bool) -> Vec<f64> {
    let mut state = seed;
    let mut signal = Vec::with_capacity(n);
    let mut phase = 0.0f64;
    for i in 0..n {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let noise = (state >> 33) as f64 / (1u64 << 31) as f64 - 0.5;
        phase += 0.1;
        let carrier = 45.0 + 2.0 * phase.sin(); // nominal SNR ~45 dB-Hz
        if scintillation && i > n / 3 {
            // ionospheric scintillation: rapid fading + phase jumps
            let fade = (i as f64 * 0.3).sin() * 12.0;
            signal.push(carrier + fade + noise * 5.0);
        } else {
            signal.push(carrier + noise * 1.5);
        }
    }
    signal
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() > 1 {
        let data: Vec<f64> = std::fs::read_to_string(&args[1])
            .expect("cannot read file")
            .lines()
            .filter_map(|l| l.trim().split(',').next_back()?.trim().parse().ok())
            .collect();
        println!("  loaded {} samples from {}", data.len(), args[1]);
        let law = analyze(&data);
        println!("  α = {:.3}  R² = {:.4}", law.dfa.alpha, law.dfa.r_squared);
        if law.dfa.alpha > 0.7 {
            println!("  signal: structured (clean reception) ✅");
        } else {
            println!("  signal: scintillation detected ⚠️ (positioning may be degraded)");
        }
        return;
    }

    println!();
    println!("  GNSS SIGNAL QUALITY — SCINTILLATION DETECTION");
    println!("  ================================================================");

    let clean = synth_gnss_signal(4096, 42, false);
    let scint = synth_gnss_signal(4096, 42, true);

    let law_clean = analyze(&clean);
    let law_scint = analyze(&scint);

    println!();
    println!("  clean signal      α = {:.3}  R² = {:.4}", law_clean.dfa.alpha, law_clean.dfa.r_squared);
    println!("  scintillation     α = {:.3}  R² = {:.4}", law_scint.dfa.alpha, law_scint.dfa.r_squared);

    let result = compare(&clean, &scint);
    println!("                    {}", result);

    println!();
    println!("  use case: detect when ionospheric conditions degrade GPS accuracy");
    println!("  before the receiver reports position errors.");
    println!();
}
