//! NAB evaluation harness. Loading, timestamp parsing and window-scoring
//! logic are copied from ../../examples/nab_eval.rs (same file, same rules)
//! so every detector here is scored identically to struktura's own NAB
//! example.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::detectors::{self, DetResult};

const CATEGORIES: &[&str] = &[
    "artificialNoAnomaly",
    "artificialWithAnomaly",
    "realAdExchange",
    "realAWSCloudwatch",
    "realKnownCause",
    "realTraffic",
    "realTweets",
];

const RELIABLE_CALIB: usize = 768;

// ---- minimal JSON reader (copied from nab_eval.rs) ----
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

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

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
            continue;
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

fn in_any_window(ts: i64, windows: &[(i64, i64)]) -> bool {
    windows.iter().any(|&(a, b)| ts >= a && ts <= b)
}

const DET_NAMES: [&str; 8] = [
    "struktura guard (default)",
    "struktura guard (quiet_drift)",
    "augurs-changepoint BOCPD",
    "ankane STL (anomaly_detection)",
    "extended-isolation-forest",
    "limit check (baseline)",
    "EWMA chart (baseline)",
    "CUSUM (baseline)",
];

struct Agg {
    windows_caught: usize,
    total_windows: usize,
    false_alarms: usize,
    post_calib_samples: usize,
    time_ns_sum: f64,
    time_n: usize,
    streaming: bool,
}

impl Agg {
    fn new(streaming: bool) -> Self {
        Agg {
            windows_caught: 0,
            total_windows: 0,
            false_alarms: 0,
            post_calib_samples: 0,
            time_ns_sum: 0.0,
            time_n: 0,
            streaming,
        }
    }
    fn add(&mut self, res: &DetResult, rest_ts: &[i64], windows: &[(i64, i64)]) {
        self.total_windows += windows.len();
        self.post_calib_samples += rest_ts.len();
        let alarm_ts: Vec<i64> = res.alarms.iter().map(|&i| rest_ts[i]).collect();
        for &ts in &alarm_ts {
            if !in_any_window(ts, windows) {
                self.false_alarms += 1;
            }
        }
        self.windows_caught +=
            windows.iter().filter(|&&(a, b)| alarm_ts.iter().any(|&ts| ts >= a && ts <= b)).count();
        self.time_ns_sum += res.per_sample_ns * rest_ts.len() as f64;
        self.time_n += rest_ts.len();
    }
}

pub fn run(nab_dir: &str) {
    let nab_dir = Path::new(nab_dir);
    let labels = load_labels(nab_dir);

    let mut aggs: [Agg; 8] = [
        Agg::new(true),
        Agg::new(true),
        Agg::new(false),
        Agg::new(false),
        Agg::new(false),
        Agg::new(true),
        Agg::new(true),
        Agg::new(true),
    ];
    let mut series_count = 0usize;
    let mut skipped = 0usize;

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
            let total_rows = rows.len();
            if total_rows < 192 {
                skipped += 1;
                continue;
            }
            let pct15 = ((total_rows as f64) * 0.15).round() as usize;
            let target = pct15.max(RELIABLE_CALIB);
            let calib_n = target.min(total_rows.saturating_sub(1)).max(192);

            let values: Vec<f64> = rows.iter().map(|(_, v)| *v).collect();
            let ts: Vec<i64> = rows.iter().map(|(t, _)| *t).collect();
            let calib = &values[..calib_n];
            let rest = &values[calib_n..];
            let rest_ts = &ts[calib_n..];
            if rest.is_empty() {
                skipped += 1;
                continue;
            }

            // period estimate for STL, from median timestamp spacing
            let mut diffs: Vec<i64> = (1..calib.len()).map(|i| ts[i] - ts[i - 1]).collect();
            diffs.sort_unstable();
            let interval = if diffs.is_empty() { 1 } else { diffs[diffs.len() / 2] }.max(1);
            let period = ((86400 / interval).max(2)) as usize;

            let t_series = std::time::Instant::now();
            let names = ["guard", "guard-quiet", "bocpd", "stl", "eif", "limit", "ewma", "cusum"];
            let mut results: Vec<DetResult> = Vec::with_capacity(8);
            for (di, name) in names.iter().enumerate() {
                let t0 = std::time::Instant::now();
                let res = match di {
                    0 => detectors::struktura_guard(calib, rest, false)
                        .unwrap_or(DetResult { alarms: vec![], per_sample_ns: 0.0, streaming: true }),
                    1 => detectors::struktura_guard(calib, rest, true)
                        .unwrap_or(DetResult { alarms: vec![], per_sample_ns: 0.0, streaming: true }),
                    2 => detectors::augurs_bocpd(calib, rest),
                    3 => detectors::ankane_stl(rest, period),
                    4 => detectors::isolation_forest(calib, rest),
                    5 => detectors::limit_check(calib, rest),
                    6 => detectors::ewma(calib, rest),
                    _ => detectors::cusum(calib, rest),
                };
                eprintln!("    {} {} took {:.2}s", key, name, t0.elapsed().as_secs_f64());
                results.push(res);
            }
            eprintln!(
                "  [{}/58ish] {} rows={} calib={} rest={} total {:.2}s",
                series_count + 1,
                key,
                total_rows,
                calib_n,
                rest.len(),
                t_series.elapsed().as_secs_f64()
            );
            for (i, res) in results.iter().enumerate() {
                aggs[i].add(res, rest_ts, &windows);
            }
            series_count += 1;
        }
    }

    println!("## NAB (commit ea702d7, {} series evaluated, {} skipped <192 rows)\n", series_count, skipped);
    println!("| detector | windows caught | / total | false alarms | FA / 1000 samples |");
    println!("|---|---:|---:|---:|---:|");
    for (i, agg) in aggs.iter().enumerate() {
        let fa_1k = if agg.post_calib_samples > 0 {
            agg.false_alarms as f64 * 1000.0 / agg.post_calib_samples as f64
        } else {
            0.0
        };
        println!(
            "| {} | {} | {} | {} | {:.2} |",
            DET_NAMES[i], agg.windows_caught, agg.total_windows, agg.false_alarms, fa_1k
        );
    }

    println!("\n### Per-sample time & capability\n");
    println!("| detector | streaming | mean ns/sample (post-calib) |");
    println!("|---|---|---:|");
    for (i, agg) in aggs.iter().enumerate() {
        let ns = if agg.time_n > 0 { agg.time_ns_sum / agg.time_n as f64 } else { 0.0 };
        println!("| {} | {} | {:.0} |", DET_NAMES[i], agg.streaming, ns);
    }
}
