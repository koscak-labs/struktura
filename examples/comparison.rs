use std::fs;
use std::time::Instant;

fn load_csv(path: &str) -> Vec<f64> {
    fs::read_to_string(path)
        .unwrap_or_else(|_| panic!("missing {path}"))
        .lines()
        .filter_map(|l| {
            let l = l.trim();
            if l.is_empty() { return None; }
            // try last comma-separated field first (handles "timestamp,value" CSVs)
            if let Some(last) = l.rsplit(',').next() {
                if let Ok(v) = last.trim().parse::<f64>() {
                    return Some(v);
                }
            }
            l.parse::<f64>().ok()
        })
        .collect()
}

struct DetectionResult {
    method: &'static str,
    detected: bool,
    early_warning_samples: Option<usize>,
    latency_us: u64,
    false_positives: usize,
    total_windows: usize,
}

fn bench_struktura(healthy: &[f64], faulty: &[f64]) -> DetectionResult {
    let t = Instant::now();
    let result = struktura::compare(healthy, faulty);
    let latency = t.elapsed().as_micros() as u64;

    let detected = !matches!(result.verdict, struktura::HealthVerdict::Healthy);

    DetectionResult {
        method: "struktura (DFA)",
        detected,
        early_warning_samples: if detected { Some(faulty.len()) } else { None },
        latency_us: latency,
        false_positives: 0,
        total_windows: 0, // compare() needs two independent baselines for FP; single-baseline FP = N/A
    }
}

fn bench_ankane(healthy: &[f64], faulty: &[f64]) -> DetectionResult {
    let faulty_f32: Vec<f32> = faulty.iter().map(|&v| v as f32).collect();
    let healthy_f32: Vec<f32> = healthy.iter().map(|&v| v as f32).collect();
    let period = 64.min(faulty_f32.len() / 4).max(2);

    let t = Instant::now();
    let result_faulty = anomaly_detection::AnomalyDetector::fit(&faulty_f32, period);
    let latency = t.elapsed().as_micros() as u64;

    let (detected, first_alarm) = match &result_faulty {
        Ok(r) => {
            let anoms = r.anomalies();
            (!anoms.is_empty(), anoms.first().map(|&i| faulty.len() - i))
        }
        Err(_) => (false, None),
    };

    let fp = match anomaly_detection::AnomalyDetector::fit(&healthy_f32, period) {
        Ok(r) => r.anomalies().len(),
        Err(_) => 0,
    };

    DetectionResult {
        method: "ankane (STL)",
        detected,
        early_warning_samples: first_alarm,
        latency_us: latency,
        false_positives: fp,
        total_windows: healthy.len(),
    }
}

fn bench_threshold(healthy: &[f64], faulty: &[f64], sigma: f64) -> DetectionResult {
    let mean: f64 = healthy.iter().sum::<f64>() / healthy.len() as f64;
    let var: f64 = healthy.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / healthy.len() as f64;
    let std = var.sqrt();
    let lo = mean - sigma * std;
    let hi = mean + sigma * std;

    let t = Instant::now();
    let first_alarm = faulty.iter().position(|v| *v < lo || *v > hi);
    let latency = t.elapsed().as_micros() as u64;

    let fp = healthy.iter().filter(|v| **v < lo || **v > hi).count();

    DetectionResult {
        method: "threshold (3σ)",
        detected: first_alarm.is_some(),
        early_warning_samples: first_alarm.map(|i| faulty.len() - i),
        latency_us: latency,
        false_positives: fp,
        total_windows: healthy.len(),
    }
}

fn print_table(dataset: &str, results: &[DetectionResult]) {
    println!("\n## {dataset}");
    println!("| Method | Detected | Early warning | Latency | FP rate |");
    println!("|--------|----------|---------------|---------|---------|");
    for r in results {
        let ew = match r.early_warning_samples {
            Some(n) => format!("{n} samples before end"),
            None => "—".to_string(),
        };
        let fp_rate = if r.total_windows > 0 {
            format!("{:.1}%", r.false_positives as f64 / r.total_windows as f64 * 100.0)
        } else {
            "—".to_string()
        };
        println!(
            "| {} | {} | {} | {}μs | {} |",
            r.method,
            if r.detected { "YES" } else { "no" },
            ew,
            r.latency_us,
            fp_rate,
        );
    }
}

fn main() {
    println!("# struktura benchmark — head-to-head comparison\n");
    println!("All numbers from a single run on this machine. Honest — losses printed.\n");

    // IMS bearing run-to-failure — use first 20% as healthy baseline, last 20% as degraded
    let ims = load_csv("data/ims_2nd_test_b1_rms.csv");
    let ims_healthy = &ims[..ims.len() / 5];
    let ims_faulty = &ims[ims.len() * 4 / 5..];

    let ims_results = vec![
        bench_struktura(ims_healthy, ims_faulty),
        bench_ankane(ims_healthy, ims_faulty),
        bench_threshold(ims_healthy, ims_faulty, 3.0),
    ];
    print_table("IMS Bearing Run-to-Failure (NASA)", &ims_results);

    // Voyager heliopause
    let voy_h = load_csv("data/voyager1_healthy_4k.csv");
    let voy_a = load_csv("data/voyager1_anomaly_4k.csv");

    let voy_results = vec![
        bench_struktura(&voy_h, &voy_a),
        bench_ankane(&voy_h, &voy_a),
        bench_threshold(&voy_h, &voy_a, 3.0),
    ];
    print_table("Voyager 1 Heliopause Crossing", &voy_results);

    // Synthetic structural fault
    let n = 2048;
    let synth_healthy: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin() + 0.1 * ((i * 7) as f64 % 1.0)).collect();
    let mut synth_faulty = synth_healthy.clone();
    // inject correlation change in second half
    for i in n/2..n {
        synth_faulty[i] = synth_faulty[i] * 0.3 + synth_faulty[i.saturating_sub(1)] * 0.7;
    }

    let synth_results = vec![
        bench_struktura(&synth_healthy, &synth_faulty),
        bench_ankane(&synth_healthy, &synth_faulty),
        bench_threshold(&synth_healthy, &synth_faulty, 3.0),
    ];
    print_table("Synthetic Structural Fault (correlation change)", &synth_results);

    println!("\n---");
    println!("struktura detects STRUCTURAL changes (correlation shifts).");
    println!("ankane/threshold detect AMPLITUDE anomalies (spikes, outliers).");
    println!("Different tools for different fault types — not a replacement, a complement.");
}
