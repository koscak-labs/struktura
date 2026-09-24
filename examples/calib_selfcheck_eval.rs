//! How reliable is guard's calibration self-check? guard assumes its
//! calibration rows (by default the first third of the file) are healthy and
//! warns when calibrating on their first half makes the second half alarm.
//!
//! Real data: every NAB series with guard's default calibration (n / 3), split
//! by whether that stretch contains a labelled anomaly window. Clean control:
//! synthetic white / AR(0.7) / AR(0.95), 30 seeds, 6000 samples, so the
//! calibration (2000) is long enough for the check to run.
//!
//! Run: NAB_DIR=path/to/NAB cargo run --release --example calib_selfcheck_eval

use struktura::monitor::{HybridMonitor, MonitorConfig};

const MIN_HALF: usize = 768;

/// Same logic as `calibration_self_check` in src/bin/struktura.rs.
fn self_check(calib: &[f64]) -> Option<Option<usize>> {
    let half = calib.len() / 2;
    if half < MIN_HALF {
        return None; // check skipped
    }
    let mut mon = HybridMonitor::calibrate_with(&[calib[..half].to_vec()], MonitorConfig::default())?;
    Some((half..calib.len()).find(|&t| mon.push(&[calib[t]]).is_some()))
}

fn parse_ts(s: &str) -> Option<i64> {
    // "YYYY-MM-DD HH:MM:SS[.ffffff]" -> minutes since an arbitrary epoch
    let s = s.trim().trim_matches('"');
    let (d, t) = s.split_once(' ')?;
    let mut dp = d.split('-').map(|x| x.parse::<i64>().ok());
    let (y, m, day) = (dp.next()??, dp.next()??, dp.next()??);
    let mut tp = t.split(':');
    let (h, mi) = (tp.next()?.parse::<i64>().ok()?, tp.next()?.parse::<i64>().ok()?);
    let (y2, m2) = if m <= 2 { (y - 1, m + 9) } else { (y, m - 3) };
    let days = 365 * y2 + y2 / 4 - y2 / 100 + y2 / 400 + (153 * m2 + 2) / 5 + day;
    Some((days * 24 + h) * 60 + mi)
}

/// Windows per file from labels/combined_windows.json, parsed by scanning for
/// "file": [ ["start", "end"], ... ].
fn load_windows(text: &str) -> Vec<(String, Vec<(i64, i64)>)> {
    // Every key ends in `.csv"`; its windows are all timestamps up to the next key.
    let keys: Vec<usize> = text.match_indices(".csv\"").map(|(i, _)| i).collect();
    let mut out = Vec::new();
    for (k, &q) in keys.iter().enumerate() {
        let name_start = text[..q].rfind('"').map(|i| i + 1).unwrap_or(0);
        let name = format!("{}.csv", &text[name_start..q]);
        let end = keys.get(k + 1).map_or(text.len(), |&n| text[..n].rfind('"').unwrap_or(n));
        let block = &text[q + 5..end];
        let stamps: Vec<i64> = block.split('"').filter_map(parse_ts).collect();
        let wins = stamps.chunks(2).filter(|c| c.len() == 2).map(|c| (c[0], c[1])).collect();
        out.push((name, wins));
    }
    out
}

fn main() {
    // CI sets NAB_DIR on every run but only fetches NAB on the weekly one.
    match std::env::var("NAB_DIR") {
        Ok(nab) if std::path::Path::new(&format!("{nab}/labels/combined_windows.json")).exists() => nab_part(&nab),
        _ => println!("NAB not found (NAB_DIR unset or empty): skipping the NAB part"),
    }
    synthetic_part();
}

fn nab_part(nab: &str) {
    let labels = std::fs::read_to_string(format!("{nab}/labels/combined_windows.json")).expect("labels");
    let (mut warn_dirty, mut quiet_dirty, mut warn_clean, mut quiet_clean, mut skipped) = (0, 0, 0, 0, 0);
    let mut warned_clean = Vec::new();
    let all = load_windows(&labels);
    let total_windows: usize = all.iter().map(|(_, w)| w.len()).sum();
    println!("parsed {} files, {} labelled windows (NAB has 58 files, 116 windows)", all.len(), total_windows);
    for (name, windows) in all {
        let Ok(csv) = std::fs::read_to_string(format!("{nab}/data/{name}")) else { continue };
        let rows: Vec<(i64, f64)> = csv
            .lines()
            .skip(1)
            .filter_map(|l| {
                let (ts, v) = l.rsplit_once(',')?;
                Some((parse_ts(ts)?, v.trim().parse().ok()?))
            })
            .collect();
        let calib_n = (rows.len() / 3).clamp(192, 100_000);
        let calib: Vec<f64> = rows[..calib_n].iter().map(|r| r.1).collect();
        let (t0, t1) = (rows[0].0, rows[calib_n - 1].0);
        let dirty = windows.iter().any(|&(a, b)| a <= t1 && b >= t0);
        match self_check(&calib) {
            None => skipped += 1,
            Some(w) => match (w.is_some(), dirty) {
                (true, true) => warn_dirty += 1,
                (false, true) => quiet_dirty += 1,
                (true, false) => {
                    warn_clean += 1;
                    warned_clean.push(name.clone());
                }
                (false, false) => quiet_clean += 1,
            },
        }
    }
    println!("NAB, guard default calibration (first third), self-check:");
    println!("| calibration stretch | warned | not warned |");
    println!("|---|---|---|");
    println!("| contains a labelled anomaly | {warn_dirty} | {quiet_dirty} |");
    println!("| no labelled anomaly | {warn_clean} | {quiet_clean} |");
    println!("check skipped (calibration half < {MIN_HALF}): {skipped}");
    for n in &warned_clean {
        println!("  warned on clean calibration: {n}");
    }
}

fn synthetic_part() {
    let mut fa = [0usize; 3];
    for s in 0..30u64 {
        let mut st = s.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut normal = || {
            let mut u = || {
                st ^= st >> 12;
                st ^= st << 25;
                st ^= st >> 27;
                ((st.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
            };
            let (a, b) = (u(), u());
            (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
        };
        for (k, phi) in [0.0, 0.7, 0.95].iter().enumerate() {
            let mut x = 0.0;
            let v: Vec<f64> = (0..6000).map(|_| {
                x = phi * x + normal();
                x
            }).collect();
            if let Some(Some(_)) = self_check(&v[..2000]) {
                fa[k] += 1;
            }
        }
    }
    println!();
    println!("clean synthetic, 30 seeds, calibration 2000: warnings white {}/30, AR(0.7) {}/30, AR(0.95) {}/30", fa[0], fa[1], fa[2]);
}
