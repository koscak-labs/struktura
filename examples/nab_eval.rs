//! NAB (Numenta Anomaly Benchmark) first-look diagnostic for `struktura guard`.
//!
//! This is NOT a tuning exercise: `guard`'s parameters are used exactly as
//! `struktura guard` uses them (same `AutoPilot` + `HybridMonitor::calibrate`
//! path, same 50-tick same-leg alarm dedupe as `cmd_guard` in
//! `src/bin/struktura.rs`). A simple median/p95 "limit check" detector is run
//! alongside it as a naive baseline for comparison.
//!
//! ## Usage
//! ```bash
//! git clone --depth 1 https://github.com/numenta/NAB C:\Projects\_nab
//! NAB_DIR=C:\Projects\_nab cargo run --release --example nab_eval
//! ```
//! `NAB_DIR` is REQUIRED (no default): it must point at a checkout of
//! https://github.com/numenta/NAB containing `data/` and `labels/`.

use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::Path;
use std::time::Instant;

use struktura::autopilot::{AutoPilot, Event};
use struktura::monitor::{HybridMonitor, Leg, MonitorConfig};

const CATEGORIES: &[&str] = &[
    "artificialNoAnomaly",
    "artificialWithAnomaly",
    "realAdExchange",
    "realAWSCloudwatch",
    "realKnownCause",
    "realTraffic",
    "realTweets",
];

/// Mirrors `cmd_guard`'s own dedupe in src/bin/struktura.rs: same leg firing
/// again within 50 ticks of its last alarm on that leg is suppressed.
const ALARM_COOLDOWN: usize = 50;

/// `guard`'s stated reliable calibration minimum (MIN_RELIABLE_CALIB in
/// src/bin/struktura.rs), used as the calibration target here too.
const RELIABLE_CALIB: usize = 768;

// ---------------------------------------------------------------------
// Minimal hand-rolled JSON reader, just enough for combined_windows.json:
// a flat object of string -> array of [string, string] pairs. No numbers,
// bools or nulls appear in that file, so this parser does not handle them
// beyond skipping unknown tokens.
// ---------------------------------------------------------------------
enum Json {
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

struct JsonParser<'a> {
    s: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn new(s: &'a str) -> Self {
        JsonParser { s: s.as_bytes(), pos: 0 }
    }
    fn peek(&self) -> Option<u8> {
        self.s.get(self.pos).copied()
    }
    fn skip_ws(&mut self) {
        while let Some(c) = self.peek() {
            if (c as char).is_whitespace() {
                self.pos += 1;
            } else {
                break;
            }
        }
    }
    fn parse_string(&mut self) -> String {
        let mut out = String::new();
        if self.peek() != Some(b'"') {
            return out;
        }
        self.pos += 1;
        while let Some(c) = self.peek() {
            self.pos += 1;
            if c == b'"' {
                break;
            }
            if c == b'\\' {
                if let Some(esc) = self.peek() {
                    self.pos += 1;
                    match esc {
                        b'n' => out.push('\n'),
                        b't' => out.push('\t'),
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        other => out.push(other as char),
                    }
                }
            } else {
                out.push(c as char);
            }
        }
        out
    }
    fn parse_array(&mut self) -> Vec<Json> {
        let mut items = Vec::new();
        if self.peek() != Some(b'[') {
            return items;
        }
        self.pos += 1;
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return items;
        }
        loop {
            self.skip_ws();
            items.push(self.parse_value());
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b']') => {
                    self.pos += 1;
                    break;
                }
                _ => break,
            }
        }
        items
    }
    fn parse_object(&mut self) -> Vec<(String, Json)> {
        let mut items = Vec::new();
        if self.peek() != Some(b'{') {
            return items;
        }
        self.pos += 1;
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return items;
        }
        loop {
            self.skip_ws();
            let key = self.parse_string();
            self.skip_ws();
            if self.peek() == Some(b':') {
                self.pos += 1;
            }
            self.skip_ws();
            let val = self.parse_value();
            items.push((key, val));
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b'}') => {
                    self.pos += 1;
                    break;
                }
                _ => break,
            }
        }
        items
    }
    fn parse_value(&mut self) -> Json {
        self.skip_ws();
        match self.peek() {
            Some(b'"') => Json::Str(self.parse_string()),
            Some(b'[') => Json::Arr(self.parse_array()),
            Some(b'{') => Json::Obj(self.parse_object()),
            _ => {
                while let Some(c) = self.peek() {
                    if c == b',' || c == b']' || c == b'}' {
                        break;
                    }
                    self.pos += 1;
                }
                Json::Str(String::new())
            }
        }
    }
}

/// Days since 1970-01-01 (Howard Hinnant's civil_from_days algorithm).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// Parses "YYYY-MM-DD HH:MM:SS[.ffffff]" into epoch seconds. Returns None on
/// malformed input.
fn parse_ts(s: &str) -> Option<i64> {
    let s = s.trim();
    let mut parts = s.splitn(2, ' ');
    let date = parts.next()?;
    let time = parts.next().unwrap_or("00:00:00");
    let mut dp = date.split('-');
    let y: i64 = dp.next()?.parse().ok()?;
    let mo: i64 = dp.next()?.parse().ok()?;
    let d: i64 = dp.next()?.parse().ok()?;
    let time = time.split('.').next().unwrap_or(time);
    let mut tp = time.split(':');
    let hh: i64 = tp.next().unwrap_or("0").parse().unwrap_or(0);
    let mm: i64 = tp.next().unwrap_or("0").parse().unwrap_or(0);
    let ss: i64 = tp.next().unwrap_or("0").parse().unwrap_or(0);
    Some(days_from_civil(y, mo, d) * 86400 + hh * 3600 + mm * 60 + ss)
}

fn load_labels(nab_dir: &Path) -> HashMap<String, Vec<(i64, i64)>> {
    let path = nab_dir.join("labels").join("combined_windows.json");
    let content = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", path.display(), e));
    let mut parser = JsonParser::new(&content);
    let root = parser.parse_value();
    let mut map = HashMap::new();
    if let Json::Obj(entries) = root {
        for (key, val) in entries {
            let mut windows = Vec::new();
            if let Json::Arr(wins) = val {
                for w in wins {
                    if let Json::Arr(pair) = w {
                        if pair.len() == 2 {
                            if let (Json::Str(a), Json::Str(b)) = (&pair[0], &pair[1]) {
                                if let (Some(sa), Some(sb)) = (parse_ts(a), parse_ts(b)) {
                                    windows.push((sa, sb));
                                }
                            }
                        }
                    }
                }
            }
            map.insert(key, windows);
        }
    }
    map
}

fn read_series(path: &Path) -> Vec<(i64, f64)> {
    let content = fs::read_to_string(path).unwrap_or_default();
    let mut rows = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 {
            continue; // header: timestamp,value
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.splitn(2, ',');
        let ts = match parts.next().and_then(parse_ts) {
            Some(t) => t,
            None => continue,
        };
        let v: f64 = match parts.next().and_then(|s| s.trim().parse().ok()) {
            Some(v) => v,
            None => continue,
        };
        rows.push((ts, v));
    }
    rows
}

fn leg_idx(l: Leg) -> usize {
    match l {
        Leg::Residual => 0,
        Leg::RepeatedValue => 1,
        Leg::Dfa => 2,
        Leg::LevelShift => 3,
        Leg::ResidualCusum => 4,
        Leg::Missingness => 5,
        Leg::Parity => 6,
    }
}
const LEG_NAMES: [&str; 7] = [
    "Residual", "RepeatedValue", "Dfa", "LevelShift", "ResidualCusum", "Missingness", "Parity",
];

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n == 0 {
        0.0
    } else if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

fn percentile(v: &[f64], p: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n == 0 {
        return 0.0;
    }
    let idx = ((p * n as f64).ceil() as usize).saturating_sub(1).min(n - 1);
    s[idx]
}

fn in_any_window(ts: i64, windows: &[(i64, i64)]) -> bool {
    windows.iter().any(|&(a, b)| ts >= a && ts <= b)
}

struct SeriesResult {
    category: String,
    name: String,
    total_rows: usize,
    calib_n: usize,
    insufficient_calib: bool,
    #[allow(dead_code)]
    guard_alarms: Vec<(usize, i64, Leg)>, // (row idx, ts, leg) — kept for inspection/debugging
    guard_false: usize,
    guard_leg_counts: [usize; 7],
    guard_false_leg_counts: [usize; 7],
    #[allow(dead_code)]
    limit_alarms: Vec<(usize, i64)>,
    limit_false: usize,
    windows: Vec<(i64, i64)>,
    windows_detected_guard: usize,
    windows_detected_limit: usize,
    post_calib_samples: usize,
    rows: Vec<(i64, f64)>,
    calib_median: f64,
    calib_p95: f64,
}

fn eval_series(category: &str, name: &str, rows: Vec<(i64, f64)>, windows: Vec<(i64, i64)>) -> Option<SeriesResult> {
    let total_rows = rows.len();
    if total_rows < 192 {
        return None; // cannot calibrate at all (HybridMonitor min)
    }
    let pct15 = ((total_rows as f64) * 0.15).round() as usize;
    let target = pct15.max(RELIABLE_CALIB);
    let calib_n = target.min(total_rows.saturating_sub(1)).max(192);
    let insufficient_calib = calib_n < RELIABLE_CALIB;

    let values: Vec<f64> = rows.iter().map(|(_, v)| *v).collect();
    let calib_slice = &values[..calib_n];
    let calib_median = median(calib_slice);
    let dev: Vec<f64> = calib_slice.iter().map(|v| (v - calib_median).abs()).collect();
    let calib_p95 = percentile(&dev, 0.95);

    // ---- guard path: AutoPilot + HybridMonitor, exactly as cmd_guard ----
    // QUIET=1 runs the opt-in quiet-drift mode (MonitorConfig::quiet_drift).
    // HORIZON=<samples> sets the threshold design horizon (1 expected false
    // alarm per this many clean samples; default 1e6). Lower = more sensitive.
    let mut config = MonitorConfig { quiet_drift: env::var("QUIET").is_ok_and(|v| v == "1"), ..MonitorConfig::default() };
    if let Some(h) = env::var("HORIZON").ok().and_then(|v| v.parse::<f64>().ok()) {
        config.design_horizon = h;
    }
    let mon = HybridMonitor::calibrate_with(&[calib_slice.to_vec()], config)?;
    let mut ap = AutoPilot::new(mon);
    // Every detector is scored in EPISODES: alarms less than ALARM_COOLDOWN
    // ticks after the previous raw alarm belong to the same episode, which
    // counts once (a pager sees one incident). The same rule is applied to
    // the limit check below, so the two are counted the same way.
    let mut guard_alarms: Vec<(usize, i64, Leg)> = Vec::new();
    let mut last_raw: Option<usize> = None;
    for t in calib_n..total_rows {
        let sample = [values[t]];
        for ev in ap.push(&sample, &[true]) {
            if let Event::Alarm { report, .. } = &ev {
                let new_episode = last_raw.map_or(true, |l| t - l >= ALARM_COOLDOWN);
                last_raw = Some(t);
                if new_episode {
                    guard_alarms.push((t, rows[t].0, report.leg));
                }
            }
        }
    }

    let mut guard_leg_counts = [0usize; 7];
    let mut guard_false_leg_counts = [0usize; 7];
    let mut guard_false = 0usize;
    for &(_, ts, leg) in &guard_alarms {
        guard_leg_counts[leg_idx(leg)] += 1;
        if !in_any_window(ts, &windows) {
            guard_false += 1;
            guard_false_leg_counts[leg_idx(leg)] += 1;
        }
    }
    let windows_detected_guard = windows
        .iter()
        .filter(|&&(a, b)| guard_alarms.iter().any(|&(_, ts, _)| ts >= a && ts <= b))
        .count();

    // ---- baseline limit-check detector on the same post-calib data ----
    // Raw alarm on every tick with 3+ consecutive exceedances; episodes as above.
    let mut limit_alarms: Vec<(usize, i64)> = Vec::new();
    let mut streak = 0usize;
    let mut last_raw: Option<usize> = None;
    for t in calib_n..total_rows {
        let dev = (values[t] - calib_median).abs();
        if dev > 1.5 * calib_p95.max(1e-12) {
            streak += 1;
        } else {
            streak = 0;
        }
        if streak >= 3 {
            if last_raw.map_or(true, |l| t - l >= ALARM_COOLDOWN) {
                limit_alarms.push((t, rows[t].0));
            }
            last_raw = Some(t);
        }
    }
    let limit_false = limit_alarms.iter().filter(|&&(_, ts)| !in_any_window(ts, &windows)).count();
    let windows_detected_limit = windows
        .iter()
        .filter(|&&(a, b)| limit_alarms.iter().any(|&(_, ts)| ts >= a && ts <= b))
        .count();

    Some(SeriesResult {
        category: category.to_string(),
        name: name.to_string(),
        total_rows,
        calib_n,
        insufficient_calib,
        guard_alarms,
        guard_false,
        guard_leg_counts,
        guard_false_leg_counts,
        limit_alarms,
        limit_false,
        windows: windows.clone(),
        windows_detected_guard,
        windows_detected_limit,
        post_calib_samples: total_rows - calib_n,
        rows,
        calib_median,
        calib_p95,
    })
}

fn autocorr_at_lag(values: &[f64], lag: usize) -> Option<f64> {
    let n = values.len();
    if lag == 0 || lag >= n {
        return None;
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    let mut num = 0.0;
    let mut den = 0.0;
    for &v in values.iter() {
        den += (v - mean).powi(2);
    }
    for i in 0..(n - lag) {
        num += (values[i] - mean) * (values[i + lag] - mean);
    }
    if den <= 1e-12 {
        None
    } else {
        Some(num / den)
    }
}

fn diagnose(sr: &SeriesResult) {
    let n = sr.rows.len();
    let ts: Vec<i64> = sr.rows.iter().map(|(t, _)| *t).collect();
    let vals: Vec<f64> = sr.rows.iter().map(|(_, v)| v).copied().collect();

    let mut diffs: Vec<i64> = Vec::new();
    for i in 1..n {
        diffs.push(ts[i] - ts[i - 1]);
    }
    let diffs_f: Vec<f64> = diffs.iter().map(|&d| d as f64).collect();
    let interval = median(&diffs_f).max(1.0);
    let samples_per_day = 86400.0 / interval;

    let lag_day = (86400.0 / interval).round() as usize;
    let lag_week = (7.0 * 86400.0 / interval).round() as usize;
    let ac_day = autocorr_at_lag(&vals, lag_day);
    let ac_week = autocorr_at_lag(&vals, lag_week);

    let mut repeated = 0usize;
    for i in 1..n {
        if (vals[i] - vals[i - 1]).abs() < 1e-12 {
            repeated += 1;
        }
    }
    let mut gaps = 0usize;
    for &d in &diffs {
        if (d as f64) > 1.5 * interval {
            gaps += 1;
        }
    }
    let mut spikes = 0usize;
    for t in sr.calib_n..(n.saturating_sub(1)) {
        let dev0 = (vals[t] - sr.calib_median).abs();
        let dev1 = (vals[t + 1] - sr.calib_median).abs();
        if dev0 > 4.0 * sr.calib_p95.max(1e-12) && dev1 <= 1.5 * sr.calib_p95.max(1e-12) {
            spikes += 1;
        }
    }

    let mut leg_max = (0usize, 0usize);
    for (i, &c) in sr.guard_leg_counts.iter().enumerate() {
        if c > leg_max.1 {
            leg_max = (i, c);
        }
    }

    println!(
        "  {} / {}  (rows={}, calib={}, guard_false={}, samples/day={:.1})",
        sr.category, sr.name, sr.total_rows, sr.calib_n, sr.guard_false, samples_per_day
    );
    println!(
        "    autocorr@1day({})={}  autocorr@1week({})={}",
        lag_day,
        ac_day.map(|v| format!("{:.3}", v)).unwrap_or_else(|| "n/a (series too short)".into()),
        lag_week,
        ac_week.map(|v| format!("{:.3}", v)).unwrap_or_else(|| "n/a (series too short)".into()),
    );
    println!(
        "    repeated-consecutive={}  gaps={}  spike-count={}  top leg={} ({} alarms)",
        repeated, gaps, spikes, LEG_NAMES[leg_max.0], leg_max.1
    );
}

fn main() {
    let start = Instant::now();
    let nab_dir = env::var("NAB_DIR").unwrap_or_else(|_| {
        eprintln!("NAB_DIR is required: set it to a checkout of https://github.com/numenta/NAB");
        eprintln!("  e.g. git clone --depth 1 https://github.com/numenta/NAB C:\\Projects\\_nab");
        eprintln!("       NAB_DIR=C:\\Projects\\_nab cargo run --release --example nab_eval");
        std::process::exit(2);
    });
    let nab_dir = Path::new(&nab_dir);
    let labels = load_labels(nab_dir);

    let mut results: Vec<SeriesResult> = Vec::new();
    let mut skipped_too_short: Vec<String> = Vec::new();

    for cat in CATEGORIES {
        let dir = nab_dir.join("data").join(cat);
        let entries = match fs::read_dir(&dir) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("cannot read {}: {}", dir.display(), e);
                continue;
            }
        };
        let mut files: Vec<_> = entries.filter_map(|e| e.ok()).map(|e| e.path()).collect();
        files.sort();
        for path in files {
            if path.extension().and_then(|e| e.to_str()) != Some("csv") {
                continue;
            }
            let fname = path.file_name().unwrap().to_string_lossy().to_string();
            let key = format!("{}/{}", cat, fname);
            let windows = labels.get(&key).cloned().unwrap_or_default();
            let rows = read_series(&path);
            match eval_series(cat, &fname, rows, windows) {
                Some(sr) => results.push(sr),
                None => skipped_too_short.push(key),
            }
        }
    }

    let elapsed = start.elapsed();

    // ---- per-category aggregate table ----
    println!("=== NAB first-look: struktura guard vs. limit-check baseline ===\n");
    println!(
        "{:<24} {:>4} {:>7} {:>9} {:>9} {:>10}   |   {:>9} {:>9} {:>10}",
        "category", "n", "windows", "det(grd)", "fa(grd)", "fa/1k(grd)", "det(lim)", "fa(lim)", "fa/1k(lim)"
    );
    let mut grand_guard_leg = [0usize; 7];
    let mut grand_guard_false_leg = [0usize; 7];
    for cat in CATEGORIES {
        let subset: Vec<&SeriesResult> = results.iter().filter(|r| r.category == *cat).collect();
        if subset.is_empty() {
            continue;
        }
        let n = subset.len();
        let windows: usize = subset.iter().map(|r| r.windows.len()).sum();
        let det_g: usize = subset.iter().map(|r| r.windows_detected_guard).sum();
        let fa_g: usize = subset.iter().map(|r| r.guard_false).sum();
        let post_g: usize = subset.iter().map(|r| r.post_calib_samples).sum();
        let det_l: usize = subset.iter().map(|r| r.windows_detected_limit).sum();
        let fa_l: usize = subset.iter().map(|r| r.limit_false).sum();
        let fa_g_1k = if post_g > 0 { fa_g as f64 * 1000.0 / post_g as f64 } else { 0.0 };
        let fa_l_1k = if post_g > 0 { fa_l as f64 * 1000.0 / post_g as f64 } else { 0.0 };
        for r in &subset {
            for (i, c) in r.guard_leg_counts.iter().enumerate() {
                grand_guard_leg[i] += c;
            }
            for (i, c) in r.guard_false_leg_counts.iter().enumerate() {
                grand_guard_false_leg[i] += c;
            }
        }
        println!(
            "{:<24} {:>4} {:>7} {:>9} {:>9} {:>10.2}   |   {:>9} {:>9} {:>10.2}",
            cat, n, windows, det_g, fa_g, fa_g_1k, det_l, fa_l, fa_l_1k
        );
    }
    let total_windows: usize = results.iter().map(|r| r.windows.len()).sum();
    let total_det_g: usize = results.iter().map(|r| r.windows_detected_guard).sum();
    let total_fa_g: usize = results.iter().map(|r| r.guard_false).sum();
    let total_post: usize = results.iter().map(|r| r.post_calib_samples).sum();
    let total_det_l: usize = results.iter().map(|r| r.windows_detected_limit).sum();
    let total_fa_l: usize = results.iter().map(|r| r.limit_false).sum();
    println!(
        "{:<24} {:>4} {:>7} {:>9} {:>9} {:>10.2}   |   {:>9} {:>9} {:>10.2}",
        "TOTAL",
        results.len(),
        total_windows,
        total_det_g,
        total_fa_g,
        if total_post > 0 { total_fa_g as f64 * 1000.0 / total_post as f64 } else { 0.0 },
        total_det_l,
        total_fa_l,
        if total_post > 0 { total_fa_l as f64 * 1000.0 / total_post as f64 } else { 0.0 },
    );

    println!("\n=== guard alarm breakdown by leg (all categories): total fired / false-alarm ===");
    for (i, &c) in grand_guard_leg.iter().enumerate() {
        println!("  {:<16} {:>4} total   {:>4} false", LEG_NAMES[i], c, grand_guard_false_leg[i]);
    }

    println!("\n=== insufficient calibration (<{} rows) ===", RELIABLE_CALIB);
    let insuff: Vec<&SeriesResult> = results.iter().filter(|r| r.insufficient_calib).collect();
    println!("  {} of {} series had calib_n < {}", insuff.len(), results.len(), RELIABLE_CALIB);
    if !insuff.is_empty() {
        let avg_fa_insuff: f64 = insuff.iter().map(|r| r.guard_false as f64 / r.post_calib_samples.max(1) as f64 * 1000.0).sum::<f64>() / insuff.len() as f64;
        let suff: Vec<&SeriesResult> = results.iter().filter(|r| !r.insufficient_calib).collect();
        let avg_fa_suff: f64 = if suff.is_empty() { 0.0 } else {
            suff.iter().map(|r| r.guard_false as f64 / r.post_calib_samples.max(1) as f64 * 1000.0).sum::<f64>() / suff.len() as f64
        };
        println!("  avg guard fa/1000 (insufficient calib): {:.2}", avg_fa_insuff);
        println!("  avg guard fa/1000 (sufficient calib):   {:.2}", avg_fa_suff);
    }
    if !skipped_too_short.is_empty() {
        println!("  (additionally {} series had <192 rows total and could not calibrate at all: {:?})",
            skipped_too_short.len(), skipped_too_short);
    }

    println!("\n=== top 5 series by guard false-alarm count ===");
    let mut by_fa: Vec<&SeriesResult> = results.iter().collect();
    by_fa.sort_by_key(|r| std::cmp::Reverse(r.guard_false));
    for sr in by_fa.into_iter().take(5) {
        diagnose(sr);
    }

    println!("\nwall-clock runtime: {:.2}s ({} series evaluated)", elapsed.as_secs_f64(), results.len());
}
