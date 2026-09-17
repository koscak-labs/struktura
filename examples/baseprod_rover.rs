//! BASEPROD rover benchmark — struktura on ESA's Bardenas planetary rover dataset.
//!
//! data: https://doi.org/10.57780/esa-xxd1ysw (MIT; Gerdes et al., Sci Data 11, 1054, 2024)
//! Each traverse archive already contains exported CSVs: six force/torque legs
//! (FTS_{FL,FR,CL,CR,BL,BR}_CORRECTED.csv, ~99 Hz) and IMU.csv (~50 Hz).
//!
//! usage:
//!   cargo run --release --example baseprod_rover -- path/to/2023-07-20_20-01-38/
//!
//! Prints, per channel: full-length α, the 1024-sample block-α series, and the
//! changepoints found by `changepoint::find_changepoints`. Then reports which
//! changepoints are shared by several legs (same ±1 block), which is the
//! signature of a whole-rover event rather than a single-sensor artifact.

use std::{fs, path::Path};
use struktura::changepoint::{block_alphas, find_changepoints, BLOCK};
use struktura::dfa;

const LEGS: [&str; 6] = ["FL", "FR", "CL", "CR", "BL", "BR"];

fn column(path: &Path, col: usize) -> Vec<f64> {
    let content = fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {}", path.display(), e));
    content.lines().skip(1)
        .filter_map(|l| l.split(',').nth(col)?.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite())
        .collect()
}

fn spark(alphas: &[f64]) -> String {
    const BARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    alphas.iter().map(|&a| {
        let i = (((a - 0.3) / 1.2) * 8.0).clamp(0.0, 7.0) as usize;
        BARS[i]
    }).collect()
}

fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: baseprod_rover <traverse dir containing FTS_*_CORRECTED.csv and IMU.csv>");
        std::process::exit(1);
    });
    let dir = Path::new(&dir);

    // (name, file, column index, sample rate)
    let mut channels: Vec<(String, std::path::PathBuf, usize, f64)> = LEGS.iter()
        .map(|l| (format!("{}_Fz", l), dir.join(format!("FTS_{}_CORRECTED.csv", l)), 3, 99.0))
        .collect();
    channels.push(("IMU_accZ".into(), dir.join("IMU.csv"), 10, 50.0));

    println!();
    println!("  BASEPROD rover — struktura block-α changepoints");
    println!("  traverse: {}", dir.file_name().map(|s| s.to_string_lossy()).unwrap_or_default());
    println!("  block = {} samples; sparkline spans α 0.3 (▁) … 1.5 (█)", BLOCK);
    println!("  ==============================================================");

    let mut events: Vec<(String, usize, f64, f64)> = Vec::new(); // (channel, block index, shift, z)
    for (name, file, col, hz) in &channels {
        if !file.exists() { println!("  {:9} missing", name); continue; }
        let v = column(file, *col);
        let full = dfa(&v);
        let blocks = block_alphas(&v);
        let series: Vec<f64> = blocks.iter().map(|b| b.1).collect();
        let cps = find_changepoints(&v, BLOCK, 20);
        println!();
        println!("  {:9} n={:6} ({:5.0} s)  α={:.3} R²={:.3}  blocks={}",
            name, v.len(), v.len() as f64 / hz, full.alpha, full.r_squared, blocks.len());
        println!("            {}", spark(&series));
        for cp in &cps {
            println!("            change @ {:5.0} s (sample {:6}): α {:.3} → {:.3}  shift {:+.3}  z={:.1}",
                cp.location as f64 / hz, cp.location, cp.alpha_before, cp.alpha_after, cp.shift, cp.confidence);
            events.push((name.clone(), cp.location / BLOCK, cp.shift, cp.confidence));
        }
        if cps.is_empty() { println!("            no structural change"); }
    }

    // Cross-channel agreement: events within ±1 block of each other.
    println!();
    println!("  ==============================================================");
    println!("  shared events (≥ 2 channels within ±1 block):");
    let mut used = vec![false; events.len()];
    let mut any = false;
    for i in 0..events.len() {
        if used[i] { continue; }
        let group: Vec<usize> = (0..events.len())
            .filter(|&j| !used[j] && (events[j].1 as i64 - events[i].1 as i64).abs() <= 1)
            .collect();
        if group.len() >= 2 {
            any = true;
            for &j in &group { used[j] = true; }
            let t = events[i].1 * BLOCK;
            let names: Vec<String> = group.iter().map(|&j| format!("{}({:+.2},z{:.0})", events[j].0, events[j].2, events[j].3)).collect();
            println!("    block {:4} (~{:.0} s @99 Hz): {}", events[i].1, t as f64 / 99.0, names.join("  "));
        }
    }
    if !any { println!("    none"); }
    println!();
}
