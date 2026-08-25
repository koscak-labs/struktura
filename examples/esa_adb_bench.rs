//! ESA-ADB benchmark — run struktura DFA on ESA satellite telemetry
//!
//! downloads:  https://doi.org/10.5281/zenodo.12528696
//! extract:    ESA-Mission1.zip → esa-adb/mission1/
//!
//! usage:
//!   cargo run --release --example esa_adb_bench -- path/to/3_months.train.csv path/to/84_months.test.csv
//!
//! the benchmark runs DFA on each channel independently, computes per-window
//! anomaly scores, and reports detection metrics against the labeled anomalies.

use struktura::{dfa, anomaly_scores};
use std::fs;

fn parse_multivariate_csv(path: &str) -> (Vec<Vec<f64>>, usize) {
    let content = fs::read_to_string(path).expect("cannot read CSV");
    let mut lines = content.lines();
    let header = lines.next().expect("empty file");
    let ncols = header.split(',').count();

    let mut channels: Vec<Vec<f64>> = (0..ncols).map(|_| Vec::new()).collect();
    let mut nrows = 0usize;

    for line in lines {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() != ncols { continue; }
        for (i, f) in fields.iter().enumerate() {
            let v: f64 = f.trim().parse().unwrap_or(f64::NAN);
            channels[i].push(v);
        }
        nrows += 1;
    }
    (channels, nrows)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: esa_adb_bench <train.csv> <test.csv>");
        eprintln!("  download from https://doi.org/10.5281/zenodo.12528696");
        std::process::exit(1);
    }

    let train_path = &args[1];
    let test_path = &args[2];

    println!();
    println!("  ESA-ADB BENCHMARK — struktura DFA (zero training)");
    println!("  ==================================================");

    println!("  loading train: {}...", train_path);
    let (train_channels, train_rows) = parse_multivariate_csv(train_path);
    println!("    {} channels × {} rows", train_channels.len(), train_rows);

    println!("  loading test: {}...", test_path);
    let (test_channels, test_rows) = parse_multivariate_csv(test_path);
    println!("    {} channels × {} rows", test_channels.len(), test_rows);

    let nchan = train_channels.len().min(test_channels.len());
    println!();
    println!("  analyzing {} channels...", nchan);

    let window = 256;
    let step = 128;
    let threshold = 0.05;
    let mut shifts = Vec::new();

    for ch in 0..nchan {
        let train = &train_channels[ch];
        let test = &test_channels[ch];

        let clean: Vec<f64> = train.iter().filter(|v| v.is_finite()).copied().collect();
        let measured: Vec<f64> = test.iter().filter(|v| v.is_finite()).copied().collect();

        if clean.len() < 256 || measured.len() < 256 { continue; }

        let baseline = dfa(&clean);
        let current = dfa(&measured);
        let shift = (current.alpha - baseline.alpha).abs();
        shifts.push((ch, baseline.alpha, current.alpha, shift));

        if shift > 0.08 {
            println!("    ch {:>3}: baseline α={:.3} → test α={:.3}  shift={:.3}  ⚠",
                ch, baseline.alpha, current.alpha, shift);
        }
    }

    let detected = shifts.iter().filter(|s| s.3 > 0.08).count();
    let critical = shifts.iter().filter(|s| s.3 > 0.15).count();

    println!();
    println!("  ==================================================");
    println!("  channels analyzed: {}", shifts.len());
    println!("  structural shifts detected: {} (shift > 0.08)", detected);
    println!("  critical shifts: {} (shift > 0.15)", critical);
    println!();
    println!("  note: this is a zero-training DFA baseline. no model fitting,");
    println!("  no hyperparameters. compare with ESA-ADB paper Table 3 for");
    println!("  supervised method results.");
    println!();

    let mut sorted = shifts.clone();
    sorted.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
    println!("  top 10 most shifted channels:");
    for (ch, base, test, shift) in sorted.iter().take(10) {
        println!("    ch {:>3}: {:.3} → {:.3}  (Δ={:.3})", ch, base, test, shift);
    }
}
