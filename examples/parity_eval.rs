//! Multi-channel fault isolation on the simulated rover: which sensor a
//! monitor blames, how fast, and how often it blames a healthy one.
//!
//! For each seed: one run with a single scripted fault at step 1500 (type,
//! wheel/side/channel and size drawn from the seed) and one clean run with
//! the same seed. Calibrate on steps 0..1000, stream 1000..3000 through
//! AutoPilot. A fault is detected at the first alarm or quarantine on a
//! channel it changes (ground truth from src/rover.rs); every other alarm or
//! quarantine is false. Alarms are merged per (channel, leg) within 50 steps,
//! as `struktura guard` does.
//!
//! `cargo run --release --example parity_eval` (SEEDS=n, default 60).

use struktura::autopilot::{AutoPilot, Event};
use struktura::monitor::HybridMonitor;
use struktura::rover::{RoverFault, RoverSim, ROVER_CHANNELS};

const ONSET: usize = 1500;
const MOTOR_TEMP: usize = 8;

fn fault_for(seed: u64) -> (RoverFault, &'static str, Vec<usize>) {
    // splitmix64 of (seed, k): independent draws per seed.
    let r = |k: u64| {
        let mut z = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(k.wrapping_mul(0xBF58_476D_1CE4_E5B9));
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as usize
    };
    let w = r(1) % 4;
    match r(2) % 6 {
        0 => (RoverFault::WheelBearing { wheel: w, severity: 0.5 + (r(3) % 5) as f64 * 0.1 }, "bearing", vec![w, MOTOR_TEMP]),
        1 => (RoverFault::WheelStall { wheel: w }, "stall", vec![w, MOTOR_TEMP]),
        2 => (RoverFault::BatteryCell { severity: 0.5 + (r(3) % 5) as f64 * 0.1 }, "battery", vec![5, 6]),
        3 => {
            let ch = 7 + r(3) % 2;
            (RoverFault::ThermalRunaway { channel: ch, rate: 0.5 + (r(4) % 5) as f64 * 0.25 }, "thermal", vec![ch])
        }
        4 => (RoverFault::CommFade { rate: 1.0 + (r(3) % 5) as f64 * 0.5 }, "comm", vec![9]),
        _ => {
            let side = r(3) % 2;
            (RoverFault::SuspensionAsymmetry { side, offset: 0.2 + (r(4) % 4) as f64 * 0.1 }, "suspension",
             if side == 0 { vec![0, 1, MOTOR_TEMP] } else { vec![2, 3, MOTOR_TEMP] })
        }
    }
}

/// (first detection step on a ground-truth channel, false alarms, false quarantines)
fn run(seed: u64, fault: Option<(RoverFault, Vec<usize>)>) -> (Option<usize>, usize, usize) {
    let mut sim = RoverSim::new(seed);
    let truth = match fault {
        Some((f, t)) => {
            sim.inject(ONSET, f);
            t
        }
        None => Vec::new(),
    };
    let data = sim.run(3000);
    let calib: Vec<Vec<f64>> = data.iter().map(|c| c[..1000].to_vec()).collect();
    let Some(mon) = HybridMonitor::calibrate(&calib) else { return (None, 0, 0) };
    let mut ap = AutoPilot::new(mon);
    let valid = [true; ROVER_CHANNELS];
    let mut sample = [0.0f64; ROVER_CHANNELS];
    let (mut detect, mut false_alarms, mut false_q) = (None, 0usize, 0usize);
    let mut last: Vec<(usize, (usize, u8))> = Vec::new();
    for t in 1000..3000 {
        for ch in 0..ROVER_CHANNELS {
            sample[ch] = data[ch][t];
        }
        for ev in ap.push(&sample, &valid) {
            let (ch, is_q, what) = match &ev {
                Event::Alarm { report, class, .. } => {
                    let key = (report.channel, report.leg as u8);
                    let dup = last.iter().any(|&(lt, k)| k == key && t - lt < 50);
                    last.retain(|&(lt, _)| t - lt < 50);
                    last.push((t, key));
                    if dup {
                        continue;
                    }
                    (report.channel, false, format!("{:?}/{class}", report.leg))
                }
                Event::Quarantined { channel, .. } => (*channel, true, "quarantine".to_string()),
                _ => continue,
            };
            if t >= ONSET && truth.contains(&ch) {
                detect.get_or_insert(t);
            } else {
                if is_q {
                    false_q += 1;
                } else {
                    false_alarms += 1;
                }
                if std::env::var("BREAKDOWN").is_ok() {
                    let run = if truth.is_empty() { "clean" } else { "fault" };
                    eprintln!("FALSE {run} {} {what}", struktura::rover::ROVER_CHANNEL_NAMES[ch]);
                }
            }
        }
    }
    (detect, false_alarms, false_q)
}

fn main() {
    let seeds: u64 = std::env::var("SEEDS").ok().and_then(|s| s.parse().ok()).unwrap_or(60);
    let kinds = ["bearing", "stall", "battery", "thermal", "comm", "suspension"];
    let mut by_kind: Vec<(usize, usize, Vec<usize>)> = vec![(0, 0, Vec::new()); kinds.len()];
    let (mut fa_fault, mut fq_fault, mut fa_clean, mut fq_clean, mut clean_runs_alarming) = (0, 0, 0, 0, 0);
    for seed in 1..=seeds {
        let (f, kind, truth) = fault_for(seed);
        let k = kinds.iter().position(|&x| x == kind).unwrap();
        let (det, fa, fq) = run(seed, Some((f, truth)));
        by_kind[k].0 += 1;
        if let Some(t) = det {
            by_kind[k].1 += 1;
            by_kind[k].2.push(t - ONSET);
        }
        fa_fault += fa;
        fq_fault += fq;
        let (_, ca, cq) = run(seed, None);
        fa_clean += ca;
        fq_clean += cq;
        if ca + cq > 0 {
            clean_runs_alarming += 1;
        }
    }
    println!("parity_eval: {seeds} seeds, fault at step {ONSET}, streamed 1000..3000");
    println!("| fault | runs | detected | median delay |");
    println!("|---|---|---|---|");
    for (k, name) in kinds.iter().enumerate() {
        let (n, d, delays) = &mut by_kind[k];
        delays.sort_unstable();
        let med = if delays.is_empty() { "-".to_string() } else { delays[delays.len() / 2].to_string() };
        println!("| {name} | {n} | {d} | {med} |");
    }
    let total: usize = by_kind.iter().map(|b| b.0).sum();
    let det: usize = by_kind.iter().map(|b| b.1).sum();
    println!("detected {det}/{total}");
    println!("fault runs: false alarms {fa_fault}, false quarantines {fq_fault}");
    println!("clean runs: false alarms {fa_clean}, false quarantines {fq_clean}, runs with any {clean_runs_alarming}/{seeds}");
}
