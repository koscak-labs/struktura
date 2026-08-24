use struktura::mfdfa::mfdfa;

fn main() {
    let qs = [-5.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0];

    // Voyager healthy
    let healthy: Vec<f64> = include_str!("../data/voyager1_healthy_4k.csv")
        .lines().filter_map(|l| l.trim().parse().ok()).collect();
    let spectrum_h = mfdfa(&healthy, &qs);

    // Voyager anomaly
    let anomaly: Vec<f64> = include_str!("../data/voyager1_anomaly_4k.csv")
        .lines().filter_map(|l| l.trim().parse().ok()).collect();
    let spectrum_a = mfdfa(&anomaly, &qs);

    // White noise control
    let mut state = 42u64;
    let noise: Vec<f64> = (0..4096).map(|_| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as f64 / (1u64 << 31) as f64 - 0.5
    }).collect();
    let spectrum_n = mfdfa(&noise, &qs);

    println!("MULTIFRACTAL DFA SPECTRUM");
    println!("========================\n");

    for (name, spec) in [
        ("Voyager 1 (healthy)", &spectrum_h),
        ("Voyager 1 (anomaly)", &spectrum_a),
        ("White noise", &spectrum_n),
    ] {
        println!("  {name}:");
        println!("    {spec}");
        print!("    h(q): ");
        for p in &spec.points {
            print!("{:.2} ", p.h_q);
        }
        println!("\n");
    }
}
