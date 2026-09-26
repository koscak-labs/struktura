//! UCR Time Series Anomaly Archive 2021 (250 series, one anomaly each): can struktura
//! find the anomaly, and how does it compare with simple baselines?
//!
//! Each file name gives the training end and the anomaly's first and last index. One
//! predicted location per series; it is correct if it lies within [begin - 100, end + 100]
//! (the archive's KDD Cup 2021 convention: a tolerance window, not exact localisation).
//!
//! Methods, fixed before running:
//! - guard: the streaming monitor as `struktura guard` runs it (AutoPilot), calibrated on
//!   the last min(train_end, 20000) training rows; prediction = row of the first alarm or
//!   rolled-back adaptation in the test part; no event = no prediction (a miss).
//! - offline dfa: rolling dfa_short alpha, window 128, stride 16; baseline = median and
//!   MAD * 1.4826 of windows inside the training part; prediction = centre of the test
//!   window with the largest robust z.
//!
//! Baselines: largest first difference in the test part; largest |z| of the raw value
//! against the training mean and std; random location (expected accuracy, exact).
//!
//! Run: UCR_DIR=path/to/UCR_Anomaly_FullData cargo run --release --example ucr_eval

use struktura::autopilot::{AutoPilot, Event};
use struktura::monitor::HybridMonitor;

struct Series {
    x: Vec<f64>,
    train_end: usize,
    begin: usize,
    end: usize,
}

fn load(path: &std::path::Path) -> Series {
    let text = std::fs::read_to_string(path).expect("read series");
    let x: Vec<f64> = text
        .split_whitespace()
        .map(|t| t.parse().expect("number"))
        .collect();
    let stem = path.file_stem().unwrap().to_string_lossy().to_string();
    let parts: Vec<&str> = stem.split('_').collect();
    let n = parts.len();
    let num = |s: &str| s.parse::<usize>().expect("index in file name");
    Series {
        x,
        train_end: num(parts[n - 3]),
        begin: num(parts[n - 2]),
        end: num(parts[n - 1]),
    }
}

/// Predicted rows are 1-based: row r is x[r - 1].
fn hit(p: Option<usize>, s: &Series) -> bool {
    p.is_some_and(|r| r + 100 >= s.begin && r <= s.end + 100)
}

fn guard(s: &Series) -> Option<usize> {
    let calib = s.x[s.train_end.saturating_sub(20000)..s.train_end].to_vec();
    let mon = HybridMonitor::calibrate(&[calib])?;
    let mut ap = AutoPilot::new(mon);
    for (i, &v) in s.x[s.train_end..].iter().enumerate() {
        for e in ap.push(&[v], &[true]) {
            if matches!(e, Event::Alarm { .. } | Event::RolledBack { .. }) {
                return Some(s.train_end + i + 1);
            }
        }
    }
    None
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

fn offline_dfa(s: &Series) -> Option<usize> {
    let (w, stride) = (128, 16);
    let (mut base, mut cand) = (Vec::new(), Vec::new());
    let mut a = 0;
    while a + w <= s.x.len() {
        if let Some(r) = struktura::dfa_short(&s.x[a..a + w]) {
            if r.alpha.is_finite() {
                if a + w <= s.train_end {
                    base.push(r.alpha);
                } else if a >= s.train_end {
                    cand.push((a, r.alpha));
                }
            }
        }
        a += stride;
    }
    if base.len() < 2 || cand.is_empty() {
        return None;
    }
    let med = median(&mut base.clone());
    let mut dev: Vec<f64> = base.iter().map(|b| (b - med).abs()).collect();
    let scale = (median(&mut dev) * 1.4826).max(1e-12);
    let mut best = cand[0];
    for &c in &cand {
        if (c.1 - med).abs() / scale > (best.1 - med).abs() / scale {
            best = c;
        }
    }
    Some(best.0 + w / 2 + 1)
}

fn first_difference(s: &Series) -> Option<usize> {
    let t = &s.x[s.train_end - 1..];
    let mut best = (0, f64::MIN);
    for i in 1..t.len() {
        let d = (t[i] - t[i - 1]).abs();
        if d > best.1 {
            best = (i - 1, d);
        }
    }
    Some(s.train_end + best.0 + 1)
}

fn raw_z(s: &Series) -> Option<usize> {
    let tr = &s.x[..s.train_end];
    let m = tr.iter().sum::<f64>() / tr.len() as f64;
    let sd = (tr.iter().map(|v| (v - m).powi(2)).sum::<f64>() / tr.len() as f64).sqrt() + 1e-12;
    let mut best = (0, f64::MIN);
    for (i, v) in s.x[s.train_end..].iter().enumerate() {
        let z = (v - m).abs() / sd;
        if z > best.1 {
            best = (i, z);
        }
    }
    Some(s.train_end + best.0 + 1)
}

/// Exact two-sided McNemar p-value from the discordant counts.
fn mcnemar(b: u64, c: u64) -> f64 {
    let n = b + c;
    if n == 0 {
        return 1.0;
    }
    let k = b.min(c);
    let mut cdf = 0.0;
    let mut term = 0.5f64.powi(n as i32); // C(n,0) / 2^n
    for i in 0..=k {
        cdf += term;
        term *= (n - i) as f64 / (i + 1) as f64;
    }
    (2.0 * cdf).min(1.0)
}

fn main() {
    let dir = std::env::var("UCR_DIR").expect("set UCR_DIR to the UCR_Anomaly_FullData folder");
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .expect("read UCR_DIR")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "txt"))
        .collect();
    files.sort();
    let series: Vec<Series> = files.iter().map(|p| load(p)).collect();
    let n = series.len();

    let names = ["guard", "offline dfa", "first difference", "raw z"];
    let preds: Vec<[Option<usize>; 4]> = series
        .iter()
        .map(|s| [guard(s), offline_dfa(s), first_difference(s), raw_z(s)])
        .collect();
    let hits: Vec<[bool; 4]> = series
        .iter()
        .zip(&preds)
        .map(|(s, p)| std::array::from_fn(|k| hit(p[k], s)))
        .collect();
    let random: f64 = series
        .iter()
        .map(|s| ((s.end - s.begin + 1 + 200) as f64 / (s.x.len() - s.train_end) as f64).min(1.0))
        .sum();

    println!("series={n}, hit = predicted row within [begin - 100, end + 100]");
    println!();
    println!("| method | accuracy | correct / {n} | no prediction |");
    println!("|---|---|---|---|");
    for k in [2, 0, 1, 3] {
        let c = hits.iter().filter(|h| h[k]).count();
        let none = preds.iter().filter(|p| p[k].is_none()).count();
        println!(
            "| {} | {:.3} | {} | {} |",
            names[k],
            c as f64 / n as f64,
            c,
            none
        );
    }
    println!(
        "| random location (expected) | {:.3} | {:.1} | 0 |",
        random / n as f64,
        random
    );

    for (k, name) in [(0, "guard"), (1, "offline dfa")] {
        let b = hits.iter().filter(|h| h[k] && !h[2]).count() as u64;
        let c = hits.iter().filter(|h| !h[k] && h[2]).count() as u64;
        println!("{name} vs first difference: only {name} right {b}, only first difference right {c}, exact McNemar p = {:.3}", mcnemar(b, c));
    }
    let alarmed: Vec<usize> = (0..n).filter(|&i| preds[i][0].is_some()).collect();
    let silent: Vec<usize> = (0..n).filter(|&i| preds[i][0].is_none()).collect();
    let rate = |idx: &[usize], k: usize| idx.iter().filter(|&&i| hits[i][k]).count();
    println!(
        "guard alarmed on {}/{} series; its first alarm was within tolerance on {} of them",
        alarmed.len(),
        n,
        rate(&alarmed, 0)
    );
    println!(
        "those series are easier for every method: first difference {}/{} there vs {}/{} on the series where guard stayed silent",
        rate(&alarmed, 2),
        alarmed.len(),
        rate(&silent, 2),
        silent.len()
    );
    let only = (0..n)
        .filter(|&i| (hits[i][0] || hits[i][1]) && !hits[i][2] && !hits[i][3])
        .count();
    println!("series right by guard or offline dfa and by neither baseline: {only}");
}
