//! ESA-ADB Mission 1 benchmark — run struktura on real ESA satellite telemetry
//!
//! data: https://zenodo.org/records/12528696 (ESA-Mission1.zip)
//! convert pickles to CSV first (see esa-adb/convert.py)
//!
//! usage:
//!   cargo run --release --example esa_adb_mission1 -- path/to/csv/

use struktura::dfa;
use std::{fs, time::Instant};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let csv_dir = if args.len() > 1 { &args[1] } else {
        eprintln!("usage: esa_adb_mission1 <path/to/csv/>");
        eprintln!("  download ESA-Mission1.zip from https://zenodo.org/records/12528696");
        eprintln!("  convert pickle → CSV, then point here");
        std::process::exit(1);
    };

    println!();
    println!("  ESA-ADB MISSION 1 — struktura DFA benchmark");
    println!("  zero training · zero tuning · 76 channels");
    println!("  ==============================================");

    let t0 = Instant::now();
    let mut results: Vec<(String, f64, f64, usize)> = Vec::new();

    let mut entries: Vec<_> = fs::read_dir(csv_dir)
        .expect("cannot read dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|x| x == "csv"))
        .collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in &entries {
        let path = entry.path();
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let content = fs::read_to_string(&path).unwrap();
        let values: Vec<f64> = content.lines().skip(1)
            .filter_map(|l| l.split(',').next_back()?.trim().parse().ok())
            .filter(|v: &f64| v.is_finite())
            .collect();

        if values.len() < 256 { continue; }

        let n = values.len();
        let mid = n / 2;
        let baseline = dfa(&values[..mid]);
        let current = dfa(&values[mid..]);
        let shift = (current.alpha - baseline.alpha).abs();

        let marker = if shift > 0.15 { "CRITICAL" }
            else if shift > 0.08 { "WARNING" }
            else if shift > 0.03 { "WATCH" }
            else { "HEALTHY" };

        if shift > 0.03 {
            println!("  {:>12}: α {:.3} → {:.3}  shift {:.3}  {}  ({} pts)",
                name, baseline.alpha, current.alpha, shift, marker, n);
        }
        results.push((name, baseline.alpha, shift, n));
    }

    let elapsed = t0.elapsed();
    let total_pts: usize = results.iter().map(|r| r.3).sum();
    let detected = results.iter().filter(|r| r.1 > 0.0 && r.2 > 0.03).count();
    let critical = results.iter().filter(|r| r.2 > 0.15).count();

    println!();
    println!("  ==============================================");
    println!("  channels:   {}", results.len());
    println!("  total pts:  {} ({:.1}M)", total_pts, total_pts as f64 / 1e6);
    println!("  shifts:     {} watch/warning/critical (> 0.03)", detected);
    println!("  critical:   {} (> 0.15)", critical);
    println!("  time:       {:.1}s", elapsed.as_secs_f64());
    println!("  throughput: {:.0} pts/sec", total_pts as f64 / elapsed.as_secs_f64());
    println!();
    println!("  method: split each channel 50/50, DFA on each half, compare α");
    println!("  zero training, zero tuning, zero hyperparameters");
    println!();
}
