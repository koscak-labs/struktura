use struktura::{analyze, health_check, HealthVerdict, LawQuality};

fn lcg_noise(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..n).map(|_| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as f64 / (1u64 << 31) as f64 - 0.5
    }).collect()
}

fn fractal_noise(n: usize, h: f64, seed: u64) -> Vec<f64> {
    let white = lcg_noise(n, seed);
    let mut data = vec![0.0; n];
    data[0] = white[0];
    let mut step = n / 2;
    let mut scale = 1.0f64;
    while step >= 1 {
        let mut idx = 0;
        for i in (step..n).step_by(step * 2) {
            let left = if i >= step { data[i - step] } else { 0.0 };
            let right = if i + step < n { data[i + step] } else { data[i - step] };
            data[i] = (left + right) / 2.0 + scale * white[idx % n];
            idx += 1;
        }
        step /= 2;
        scale *= (0.5f64).powf(h);
    }
    data
}

fn shuffle(data: &[f64], seed: u64) -> Vec<f64> {
    let mut out = data.to_vec();
    let n = out.len();
    let mut state = seed;
    for i in (1..n).rev() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (state >> 33) as usize % (i + 1);
        out.swap(i, j);
    }
    out
}

fn verdict_str(v: HealthVerdict) -> &'static str {
    match v {
        HealthVerdict::Healthy => "HEALTHY",
        HealthVerdict::Watch => "WATCH",
        HealthVerdict::Warning => "WARNING",
        HealthVerdict::Critical => "CRITICAL",
        _ => "UNKNOWN",
    }
}

fn quality_str(q: LawQuality) -> &'static str {
    match q {
        LawQuality::Exact => "EXACT",
        LawQuality::Strong => "STRONG",
        LawQuality::Good => "GOOD",
        LawQuality::Approx => "APPROX",
        LawQuality::Abstain => "ABSTAIN",
        LawQuality::Insufficient => "INSUFFICIENT",
        _ => "UNKNOWN",
    }
}

fn main() {
    println!("================================================================");
    println!("  STRUKTURA BENCHMARK");
    println!("  Predict failure before it happens.");
    println!("================================================================\n");

    // Generate synthetic signals that match our verified cross-domain results
    let bearing_normal = fractal_noise(4096, 0.4, 42);
    let bearing_fault = fractal_noise(4096, 0.15, 43);
    let cardiac_normal = fractal_noise(2048, 0.7, 44);
    let cardiac_arrhythmic = lcg_noise(2048, 45); // white noise = broken structure
    let genome_like = fractal_noise(8000, 0.9, 46);
    let spacecraft = fractal_noise(500, 0.6, 47);

    let signals: Vec<(&str, &[f64], Option<f64>)> = vec![
        ("Bearing (normal)", &bearing_normal, None),
        ("Bearing (fault)", &bearing_fault, None),
        ("Cardiac (normal)", &cardiac_normal, None),
        ("Cardiac (arrhythmic)", &cardiac_arrhythmic, None),
        ("Genome-like", &genome_like, None),
        ("Spacecraft queue", &spacecraft, None),
    ];

    println!("  {:<25} {:>8} {:>8} {:>8} {:>10}", "Signal", "DFA a", "R2", "H", "Quality");
    println!("  {}", "-".repeat(65));

    let mut baselines = Vec::new();
    for (name, data, _) in &signals {
        let law = analyze(data);
        baselines.push(law.dfa.alpha);
        println!("  {:<25} {:>8.3} {:>8.4} {:>8.3} {:>10}",
            name, law.dfa.alpha, law.dfa.r_squared, law.hurst, quality_str(law.quality));
    }

    // Fault detection
    println!("\n  === FAULT DETECTION ===\n");
    let baseline = baselines[0]; // bearing normal
    for (i, (name, data, _)) in signals.iter().enumerate() {
        if i < 2 { // only bearing signals
            let law = analyze(data);
            let verdict = health_check(&law, baseline);
            let shift = law.dfa.alpha - baseline;
            println!("  {:<25} shift={:>+7.3}  {}", name, shift, verdict_str(verdict));
        }
    }

    // Shuffle control — THE empirical proof
    println!("\n  === SHUFFLE CONTROL (proves structure is real) ===\n");
    println!("  {:<25} {:>10} {:>10} {:>10}", "Signal", "Real a", "Shuffled a", "Proof");
    println!("  {}", "-".repeat(60));

    for (name, data, _) in &signals {
        let real = analyze(data);
        let shuffled_data = shuffle(data, 999);
        let shuffled = analyze(&shuffled_data);
        let destroyed = (shuffled.dfa.alpha - 0.5).abs() < (real.dfa.alpha - 0.5).abs();
        println!("  {:<25} {:>10.3} {:>10.3} {:>10}",
            name, real.dfa.alpha, shuffled.dfa.alpha,
            if destroyed { "REAL" } else { "INCONCLUSIVE" });
    }

    println!("\n  Shuffle control: if permuting the signal moves alpha toward 0.5,");
    println!("  the original structure was real (not an artifact of the data distribution).\n");

    println!("================================================================");
    println!("  All results from struktura v{}", env!("CARGO_PKG_VERSION"));
    println!("  https://github.com/koscak-labs/struktura");
    println!("================================================================");
}
