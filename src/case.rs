//! A case directory: saves a recording, its incidents, detector config, and
//! a manifest so `struktura replay` can reproduce and diff the analysis.
#![cfg(feature = "std")]

use std::path::{Path, PathBuf};

use crate::context::ContextEvent;
use crate::incident::{Evidence, Incident};
use crate::monitor::Leg;
/// A case directory's manifest (hand-rolled JSON; no serde_json dependency).
#[derive(Debug, Clone, PartialEq)]
pub struct CaseManifest {
    pub name: String,
    /// ISO 8601 UTC, e.g. `2026-09-17T12:00:00Z`.
    pub created: String,
    pub detector_version: String,
    pub baseline_samples: usize,
    pub recording_path: String,
    pub incidents_count: usize,
    pub channels: usize,
    pub samples: usize,
}

impl CaseManifest {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"name\":\"{}\",\"created\":\"{}\",\"detector_version\":\"{}\",\
             \"baseline_samples\":{},\"recording_path\":\"{}\",\"incidents_count\":{},\
             \"channels\":{},\"samples\":{}}}",
            json_escape(&self.name), json_escape(&self.created), json_escape(&self.detector_version),
            self.baseline_samples, json_escape(&self.recording_path), self.incidents_count,
            self.channels, self.samples
        )
    }

    /// Minimal hand-rolled parser: finds each key, extracts its value.
    pub fn from_json(s: &str) -> Result<Self, String> {
        let req_s = |k: &str| extract_str(s, k).ok_or_else(|| format!("manifest: missing {}", k));
        let req_n = |k: &str| extract_u64(s, k).ok_or_else(|| format!("manifest: missing {}", k));
        Ok(Self {
            name: req_s("name")?,
            created: req_s("created")?,
            detector_version: req_s("detector_version")?,
            baseline_samples: req_n("baseline_samples")? as usize,
            recording_path: req_s("recording_path")?,
            incidents_count: req_n("incidents_count")? as usize,
            channels: req_n("channels")? as usize,
            samples: req_n("samples")? as usize,
        })
    }
}

/// A saved investigation case: recording + incidents + config + manifest.
#[derive(Debug, Clone)]
pub struct Case {
    dir: PathBuf,
    pub manifest: CaseManifest,
}

fn write_file(path: PathBuf, content: String) -> Result<(), String> {
    std::fs::write(&path, content).map_err(|e| format!("write {}: {}", path.display(), e))
}

impl Case {
    /// Path to the case directory.
    pub fn dir(&self) -> &std::path::Path { &self.dir }

    /// `recording` is row-major (one `Vec<f64>` per tick), matching what
    /// `AutoPilot::push` consumes during live monitoring. [`Case::recording`]
    /// hands it back channel-major, the shape `HybridMonitor::calibrate` wants.
    pub fn save(
        dir: &Path,
        recording: &[Vec<f64>],
        incidents: &[Incident],
        baseline_samples: usize,
        name: &str,
    ) -> Result<Self, String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("create_dir_all: {}", e))?;
        let channels = recording.first().map(|r| r.len()).unwrap_or(0);
        let manifest = CaseManifest {
            name: name.to_string(),
            created: iso8601_now(),
            detector_version: env!("CARGO_PKG_VERSION").to_string(),
            baseline_samples,
            recording_path: dir.join("recording.csv").to_string_lossy().into_owned(),
            incidents_count: incidents.len(),
            channels,
            samples: recording.len(),
        };
        let mut csv = (0..channels).map(|c| format!("ch{}", c)).collect::<Vec<_>>().join(",");
        csv.push('\n');
        for row in recording {
            csv.push_str(&row.iter().map(f64::to_string).collect::<Vec<_>>().join(","));
            csv.push('\n');
        }
        let incidents_json =
            format!("[{}]", incidents.iter().map(Incident::to_json).collect::<Vec<_>>().join(","));
        let config = format!(
            "{{\"baseline_samples\":{},\"detector_version\":\"{}\"}}", baseline_samples, env!("CARGO_PKG_VERSION")
        );
        write_file(dir.join("manifest.json"), manifest.to_json())?;
        write_file(dir.join("recording.csv"), csv)?;
        write_file(dir.join("incidents.json"), incidents_json)?;
        write_file(dir.join("config.json"), config)?;
        Ok(Case { dir: dir.to_path_buf(), manifest })
    }

    pub fn load(dir: &Path) -> Result<Self, String> {
        let path = dir.join("manifest.json");
        if !path.exists() {
            return Err(format!("no manifest.json in {}", dir.display()));
        }
        let text = std::fs::read_to_string(&path).map_err(|e| format!("read manifest.json: {}", e))?;
        Ok(Case { dir: dir.to_path_buf(), manifest: CaseManifest::from_json(&text)? })
    }

    /// Reads `recording.csv` back as column-major vectors (one `Vec<f64>`
    /// per channel) — a transpose of the row-per-tick CSV layout.
    pub fn recording(&self) -> Result<Vec<Vec<f64>>, String> {
        let path = self.dir.join("recording.csv");
        let text = std::fs::read_to_string(&path).map_err(|e| format!("read recording.csv: {}", e))?;
        let mut rows: Vec<Vec<f64>> = Vec::new();
        for (n, line) in text.lines().skip(1).enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let row: Result<Vec<f64>, String> = line.split(',')
                .map(|f| f.trim().parse::<f64>().map_err(|e| format!("recording.csv line {}: {}", n + 2, e)))
                .collect();
            rows.push(row?);
        }
        let channels = rows.first().map(|r| r.len()).unwrap_or(0);
        let mut cols: Vec<Vec<f64>> = vec![Vec::with_capacity(rows.len()); channels];
        for row in &rows {
            for (c, v) in row.iter().enumerate().take(channels) {
                cols[c].push(*v);
            }
        }
        Ok(cols)
    }

    /// Reads `incidents.json` back into [`Incident`]s (see `parse_incident`).
    pub fn incidents(&self) -> Result<Vec<Incident>, String> {
        let path = self.dir.join("incidents.json");
        let text = std::fs::read_to_string(&path).map_err(|e| format!("read incidents.json: {}", e))?;
        split_by_anchor(&text, "{\"id\":").iter().map(|o| parse_incident(o)).collect()
    }
}

// incidents.json parsing: anchor-split on each object's unique key.
fn parse_incident(sub: &str) -> Result<Incident, String> {
    Ok(Incident {
        id: extract_u64(sub, "id").ok_or("incident: missing id")?,
        start_tick: extract_u64(sub, "start_tick").ok_or("incident: missing start_tick")?,
        end_tick: extract_u64(sub, "end_tick").ok_or("incident: missing end_tick")?,
        evidence: split_by_anchor(array_content(sub, "evidence").unwrap_or(""), "{\"channel\":")
            .iter().map(|o| parse_evidence(o)).collect::<Result<_, _>>()?,
        context: split_by_anchor(array_content(sub, "context").unwrap_or(""), "{\"tick\":")
            .iter().map(|o| parse_context_event(o)).collect::<Result<_, _>>()?,
        channels_involved: array_content(sub, "channels_involved").map(parse_num_list).unwrap_or_default(),
        resolution: if sub.contains("\"resolution\":null") { None } else { extract_str(sub, "resolution") },
        unresolved: split_string_array(array_content(sub, "unresolved").unwrap_or("")),
    })
}

fn parse_num_list(s: &str) -> Vec<usize> {
    s.split(',').map(str::trim).filter(|s| !s.is_empty()).filter_map(|s| s.parse().ok()).collect()
}
fn parse_evidence(sub: &str) -> Result<Evidence, String> {
    let leg_str = extract_str(sub, "leg").ok_or("evidence: missing leg")?;
    Ok(Evidence {
        channel: extract_u64(sub, "channel").ok_or("evidence: missing channel")? as usize,
        leg: parse_leg(&leg_str).ok_or_else(|| format!("evidence: unknown leg {}", leg_str))?,
        tick: extract_u64(sub, "tick").ok_or("evidence: missing tick")?,
        observed: extract_f64(sub, "observed").ok_or("evidence: missing observed")?,
        threshold: extract_f64(sub, "threshold").ok_or("evidence: missing threshold")?,
        explanation: extract_str(sub, "explanation").ok_or("evidence: missing explanation")?,
    })
}

fn parse_context_event(sub: &str) -> Result<ContextEvent, String> {
    Ok(ContextEvent::new(
        extract_u64(sub, "tick").ok_or("context: missing tick")?,
        extract_str(sub, "kind").ok_or("context: missing kind")?,
        extract_str(sub, "value").ok_or("context: missing value")?,
    ))
}

fn parse_leg(s: &str) -> Option<Leg> {
    const NAMES: [(&str, Leg); 7] = [
        ("Residual", Leg::Residual), ("RepeatedValue", Leg::RepeatedValue), ("Dfa", Leg::Dfa),
        ("LevelShift", Leg::LevelShift), ("ResidualCusum", Leg::ResidualCusum),
        ("Missingness", Leg::Missingness), ("Parity", Leg::Parity),
    ];
    NAMES.iter().find(|(n, _)| *n == s).map(|(_, l)| *l)
}

// generic hand-rolled JSON helpers.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

fn extract_str(s: &str, key: &str) -> Option<String> {
    let pat = format!("\"{}\":\"", key);
    Some(read_quoted(&mut s[s.find(&pat)? + pat.len()..].chars()))
}

/// Reads a JSON string body (decoding `\n`/`\r`/`\t`/`\"`/`\\`) up to and
/// including its closing `"`. Assumes well-formed input (own writer only).
fn read_quoted<I: Iterator<Item = char>>(chars: &mut I) -> String {
    let mut out = String::new();
    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                if let Some(n) = chars.next() {
                    out.push(match n { 'n' => '\n', 'r' => '\r', 't' => '\t', other => other });
                }
            }
            '"' => break,
            c => out.push(c),
        }
    }
    out
}

fn extract_u64(s: &str, key: &str) -> Option<u64> { extract_num(s, key)?.parse().ok() }
fn extract_f64(s: &str, key: &str) -> Option<f64> { extract_num(s, key)?.parse().ok() }

fn extract_num<'a>(s: &'a str, key: &str) -> Option<&'a str> {
    let rest = &s[s.find(&format!("\"{}\":", key))? + key.len() + 3..];
    Some(rest[..rest.find([',', '}', ']']).unwrap_or(rest.len())].trim())
}

/// Raw text inside `"key":[ ... ]` (no nested arrays occur here, so the
/// close is the first unquoted `]` after the open).
fn array_content<'a>(s: &'a str, key: &str) -> Option<&'a str> {
    let open = s.find(&format!("\"{}\":[", key))? + key.len() + 4;
    Some(&s[open..find_unquoted(s, open, b']')?])
}
// First occurrence of `target` not inside a quoted string.
fn find_unquoted(s: &str, from: usize, target: u8) -> Option<usize> {
    let (mut in_string, mut escape) = (false, false);
    for (i, &b) in s.as_bytes().iter().enumerate().skip(from) {
        if escape {
            escape = false;
        } else if in_string {
            match b { b'\\' => escape = true, b'"' => in_string = false, _ => {} }
        } else if b == b'"' {
            in_string = true;
        } else if b == target {
            return Some(i);
        }
    }
    None
}
// Splits `s` at each occurrence of `anchor` (a key unique to the object
// type being split, e.g. `{"id":` for incidents) into unbroken chunks.
fn split_by_anchor<'a>(s: &'a str, anchor: &str) -> Vec<&'a str> {
    let starts: Vec<usize> = s.match_indices(anchor).map(|(i, _)| i).collect();
    starts.iter().enumerate()
        .map(|(k, &start)| &s[start..starts.get(k + 1).copied().unwrap_or(s.len())])
        .collect()
}
fn split_string_array(content: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut chars = content.chars();
    while let Some(c) = chars.next() {
        if c == '"' {
            out.push(read_quoted(&mut chars));
        }
    }
    out
}

// Seconds-since-epoch -> `YYYY-MM-DDTHH:MM:SSZ` (no chrono dependency).
fn iso8601_now() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
    let (y, m, d) = civil_from_days((secs / 86400) as i64);
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, m, d, secs % 86400 / 3600, secs % 3600 / 60, secs % 60)
}
// Howard Hinnant's civil_from_days: days-since-epoch -> (year, month, day).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::AlarmReport;

    fn tmp_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("struktura_case_test_{}", name))
    }

    #[test]
    fn manifest_json_round_trips() {
        let m = CaseManifest {
            name: "say \"hi\"\\bye".to_string(), created: "2026-09-17T12:00:00Z".to_string(),
            detector_version: "1.7.3".to_string(), baseline_samples: 200,
            recording_path: "C:\\Oura\\case\\recording.csv".to_string(), incidents_count: 2,
            channels: 3, samples: 500,
        };
        assert_eq!(CaseManifest::from_json(&m.to_json()).expect("parses"), m);
    }

    #[test]
    fn save_load_round_trips_recording_and_incidents() {
        let dir = tmp_dir("roundtrip");
        let _ = std::fs::remove_dir_all(&dir);
        let recording = vec![vec![1.0, 2.0], vec![1.5, 2.5], vec![2.0, 3.0]];
        let mut builder = crate::incident::IncidentBuilder::new(10);
        builder.push_alarm(&AlarmReport {
            leg: Leg::LevelShift, channel: 0, tick: 1, observed: 4.2, threshold: 3.0, hit_gap: 0,
        });
        let incidents = builder.finalize();
        let case = Case::save(&dir, &recording, &incidents, 1, "roundtrip").expect("saves");
        assert_eq!((case.manifest.channels, case.manifest.samples), (2, 3));

        let loaded = Case::load(&dir).expect("loads");
        assert_eq!(loaded.manifest, case.manifest);
        assert_eq!(
            loaded.recording().expect("reads recording"),
            vec![vec![1.0, 1.5, 2.0], vec![2.0, 2.5, 3.0]]
        );
        let back = loaded.incidents().expect("reads incidents");
        assert_eq!(back[0].evidence[0].channel, 0);
        assert_eq!(back[0].evidence[0].leg, Leg::LevelShift);
        assert!(Case::load(&tmp_dir("missing")).unwrap_err().contains("manifest.json"));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
