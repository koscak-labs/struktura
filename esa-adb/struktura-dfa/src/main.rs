//! TimeEval adapter for struktura DFA anomaly detection.
//!
//! Contract: receives exactly one CLI argument — a JSON string describing
//! the execution (see ESA-ADB / TimeEval-algorithms spec).
//!
//! Input CSV: `timestamp,value[,value_1,...],is_anomaly`
//!   - Skip column 0 (timestamp) and last column (is_anomaly).
//!   - For univariate: one feature column.
//!   - For multivariate: multiple feature columns, scored independently,
//!     output is max score across channels per row.
//!
//! Output: headerless single-column file, one float per line, length = input rows.

use std::fs;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: struktura-dfa-timeeval '<json config>'");
        std::process::exit(1);
    }

    let config = &args[1];

    // Minimal JSON parsing (no serde dependency for a 6-field config)
    let execution_type = extract_str(config, "executionType").unwrap_or_default();
    let data_input = extract_str(config, "dataInput").unwrap_or_default();
    let data_output = extract_str(config, "dataOutput").unwrap_or_default();
    let custom = extract_obj(config, "customParameters").unwrap_or_default();

    let window: usize = extract_num(&custom, "window_size")
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    let step: usize = extract_num(&custom, "step_size")
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let threshold: f64 = extract_num(&custom, "threshold")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.05);

    // Unsupervised: nothing to train
    if execution_type == "train" {
        eprintln!("struktura-dfa is unsupervised — nothing to train, finished.");
        // Write an empty model file if modelOutput is specified
        if let Some(model_out) = extract_str(config, "modelOutput") {
            let _ = fs::write(&model_out, "");
        }
        return;
    }

    if data_input.is_empty() || data_output.is_empty() {
        eprintln!("error: dataInput and dataOutput are required for executionType=execute");
        std::process::exit(1);
    }

    eprintln!("struktura-dfa: reading {}", data_input);
    let content = fs::read_to_string(&data_input).unwrap_or_else(|e| {
        eprintln!("cannot read {}: {}", data_input, e);
        std::process::exit(2);
    });

    // Parse CSV: skip column 0 (timestamp) and last column (is_anomaly)
    let mut lines = content.lines();
    let header = match lines.next() {
        Some(h) => h,
        None => {
            eprintln!("empty input file");
            std::process::exit(2);
        }
    };
    let ncols = header.split(',').count();
    if ncols < 3 {
        eprintln!("need at least 3 columns (timestamp, value, is_anomaly), got {}", ncols);
        std::process::exit(2);
    }
    let feature_cols = ncols - 2; // skip first and last

    let mut channels: Vec<Vec<f64>> = (0..feature_cols).map(|_| Vec::new()).collect();
    let mut nrows = 0usize;

    for line in lines {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < ncols {
            // Malformed row: push NaN to maintain alignment
            for ch in &mut channels {
                ch.push(f64::NAN);
            }
            nrows += 1;
            continue;
        }
        // Skip column 0 (timestamp) and last column (is_anomaly)
        for i in 0..feature_cols {
            let v: f64 = fields[i + 1].trim().parse().unwrap_or(f64::NAN);
            channels[i].push(v);
        }
        nrows += 1;
    }

    eprintln!(
        "struktura-dfa: {} rows × {} channels, window={}, step={}, threshold={}",
        nrows, feature_cols, window, step, threshold
    );

    if nrows == 0 {
        eprintln!("no data rows");
        std::process::exit(2);
    }

    // Score each channel independently using struktura::anomaly_scores
    // Then take max across channels per timestep
    let mut max_scores = vec![0.0f64; nrows];

    for (ch_idx, values) in channels.iter().enumerate() {
        // Filter to finite values for DFA, keeping index mapping
        let scores = struktura::anomaly_scores(values, window, step, threshold);

        if scores.is_empty() {
            eprintln!("  ch {}: too short for window={} ({} finite samples), skipping", ch_idx, window, values.iter().filter(|v| v.is_finite()).count());
            continue;
        }

        // anomaly_scores returns one score per window position.
        // We need to expand back to nrows (one score per input row).
        // Each score at index j covers rows [j*step .. j*step + window).
        // Assign each row the max score of all windows that cover it.
        let mut per_row = vec![0.0f64; nrows];
        for (j, &s) in scores.iter().enumerate() {
            let start = j * step;
            let end = (start + window).min(nrows);
            for slot in &mut per_row[start..end] {
                if s > *slot {
                    *slot = s;
                }
            }
        }

        // Max across channels
        for r in 0..nrows {
            if per_row[r] > max_scores[r] {
                max_scores[r] = per_row[r];
            }
        }

        eprintln!("  ch {}: {} window scores, max={:.4}", ch_idx, scores.len(), scores.iter().cloned().fold(0.0f64, f64::max));
    }

    // Write output: one score per line, no header
    let output: String = max_scores.iter().map(|s| format!("{:.6}", s)).collect::<Vec<_>>().join("\n");
    fs::write(&data_output, format!("{}\n", output)).unwrap_or_else(|e| {
        eprintln!("cannot write {}: {}", data_output, e);
        std::process::exit(2);
    });

    eprintln!("struktura-dfa: wrote {} scores to {}", nrows, data_output);
}

// Minimal JSON helpers (no serde dependency)

fn extract_str(json: &str, key: &str) -> Option<String> {
    let needle = format!("\"{}\"", key);
    let pos = json.find(&needle)? + needle.len();
    let rest = &json[pos..];
    let colon = rest.find(':')?;
    let after = rest[colon + 1..].trim_start();
    if let Some(inner) = after.strip_prefix('"') {
        let end = inner.find('"')?;
        Some(inner[..end].to_string())
    } else {
        None
    }
}

fn extract_num(json: &str, key: &str) -> Option<String> {
    let needle = format!("\"{}\"", key);
    let pos = json.find(&needle)? + needle.len();
    let rest = &json[pos..];
    let colon = rest.find(':')?;
    let after = rest[colon + 1..].trim_start();
    let end = after.find([',', '}', ']']).unwrap_or(after.len());
    Some(after[..end].trim().to_string())
}

fn extract_obj(json: &str, key: &str) -> Option<String> {
    let needle = format!("\"{}\"", key);
    let pos = json.find(&needle)? + needle.len();
    let rest = &json[pos..];
    let open = rest.find('{')?;
    let start = pos + open;
    // Find matching close brace
    let mut depth = 0;
    for (i, c) in json[start..].chars().enumerate() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(json[start..start + i + 1].to_string());
                }
            }
            _ => {}
        }
    }
    None
}
