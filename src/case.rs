//! A case directory: saves a recording, its incidents, detector config, and
//! a manifest so `struktura replay` can reproduce and diff the analysis.
#![cfg(feature = "std")]

use std::path::{Path, PathBuf};

use crate::context::{ColumnSchema, ContextEvent, ContextTimeline};
use crate::incident::{Evidence, Incident};
use crate::monitor::{ChannelExport, Leg, MonitorExport};
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
    /// Content fingerprint of the *original input file* (see
    /// [`fingerprint_content`]). `None` for cases saved before this field
    /// existed. NOT what `replay()` should diff `recording.csv` against;
    /// see [`Self::recording_hash`].
    pub input_hash: Option<String>,
    /// Content fingerprint of the exact bytes written to `recording.csv`
    /// (the transformed `ch0,ch1,...`-header, normalized-value file, not
    /// the original input). This is what `replay()` compares a fresh read
    /// of `recording.csv` against: `input_hash` fingerprints a different
    /// byte representation and comparing against it always mismatched,
    /// even for an untouched case. `None` for cases saved before this
    /// field existed.
    pub recording_hash: Option<String>,
    /// Case-file layout version, independent of `detector_version`.
    pub schema_version: String,
}

impl CaseManifest {
    pub fn to_json(&self) -> String {
        let input_hash = match &self.input_hash {
            Some(h) => format!("\"{}\"", json_escape(h)),
            None => "null".to_string(),
        };
        let recording_hash = match &self.recording_hash {
            Some(h) => format!("\"{}\"", json_escape(h)),
            None => "null".to_string(),
        };
        format!(
            "{{\"name\":\"{}\",\"created\":\"{}\",\"detector_version\":\"{}\",\
             \"baseline_samples\":{},\"recording_path\":\"{}\",\"incidents_count\":{},\
             \"channels\":{},\"samples\":{},\"input_hash\":{},\"recording_hash\":{},\"schema_version\":\"{}\"}}",
            json_escape(&self.name), json_escape(&self.created), json_escape(&self.detector_version),
            self.baseline_samples, json_escape(&self.recording_path), self.incidents_count,
            self.channels, self.samples, input_hash, recording_hash, json_escape(&self.schema_version)
        )
    }

    /// Minimal hand-rolled parser: finds each key, extracts its value.
    /// `input_hash`, `recording_hash` and `schema_version` are optional for
    /// backward compatibility with manifests saved before they existed.
    pub fn from_json(s: &str) -> Result<Self, String> {
        let req_s = |k: &str| extract_str(s, k).ok_or_else(|| format!("manifest: missing {}", k));
        let req_n = |k: &str| extract_u64(s, k).ok_or_else(|| format!("manifest: missing {}", k));
        let input_hash = if s.contains("\"input_hash\":null") { None } else { extract_str(s, "input_hash") };
        let recording_hash =
            if s.contains("\"recording_hash\":null") { None } else { extract_str(s, "recording_hash") };
        let schema_version = extract_str(s, "schema_version").unwrap_or_else(|| "0.1".to_string());
        Ok(Self {
            name: req_s("name")?,
            created: req_s("created")?,
            detector_version: req_s("detector_version")?,
            baseline_samples: req_n("baseline_samples")? as usize,
            recording_path: req_s("recording_path")?,
            incidents_count: req_n("incidents_count")? as usize,
            channels: req_n("channels")? as usize,
            samples: req_n("samples")? as usize,
            input_hash,
            recording_hash,
            schema_version,
        })
    }
}

/// Full detector configuration for a saved case: what `Case::save` needs
/// beyond the recording/incidents/baseline to make the case reproducible
/// and to detect drift on `struktura replay`.
#[derive(Debug, Clone)]
pub struct CaseConfig {
    /// Content fingerprint of the input recording, computed with
    /// [`fingerprint_content`] on the raw file bytes before parsing.
    pub input_hash: String,
    /// The calibrated monitor's exported thresholds and AR coefficients.
    pub monitor_export: MonitorExport,
    /// The input CSV's column classification (measurement/mode/command/…).
    pub column_schema: ColumnSchema,
    /// Per-channel calibration-window imputation counts from
    /// `replay::run_investigation`, as `(channel_index, imputed_count)`
    /// pairs, saved so a later `struktura replay` (or a human reading
    /// `config.json`) can see how much of the original investigation's
    /// calibration was fabricated from the channel mean rather than
    /// observed.
    pub imputation: Vec<(usize, usize)>,
}

/// A content fingerprint (FNV-1a, 64-bit) over the full byte content of an
/// input file, rendered as 16 lowercase hex digits.
///
/// This is NOT a cryptographic hash (no crypto-hash dependency is pulled
/// in for it) and must not be used for integrity/security purposes. Its
/// only job is reproducibility detection: "was this case built from
/// exactly this recording file?"
#[must_use]
pub fn fingerprint_content(bytes: &[u8]) -> String {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV_OFFSET;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    format!("{:016x}", hash)
}

fn monitor_export_json(m: &MonitorExport) -> String {
    let channels = m
        .channels
        .iter()
        .map(channel_export_json)
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"res_thr\":{},\"dfa_thr\":{},\"cusum_thr\":{},\"channels\":[{}]}}",
        m.res_thr, m.dfa_thr, m.cusum_thr, channels
    )
}

fn channel_export_json(c: &ChannelExport) -> String {
    format!(
        "{{\"ar_a\":{},\"ar_b\":{},\"ar_sd\":{},\"alpha_mean\":{},\"alpha_sd\":{},\
         \"mean\":{},\"roll_thr\":{},\"max_run\":{},\"repeat_enabled\":{}}}",
        c.ar_a, c.ar_b, c.ar_sd, c.alpha_mean, c.alpha_sd, c.mean, c.roll_thr, c.max_run, c.repeat_enabled
    )
}

/// Serialize per-channel imputation counts (see [`CaseConfig::imputation`])
/// to the `config.json` `imputation` array: `[{"channel":N,"count":M},...]`.
fn imputation_json(counts: &[(usize, usize)]) -> String {
    let entries = counts
        .iter()
        .map(|(ch, n)| format!("{{\"channel\":{},\"count\":{}}}", ch, n))
        .collect::<Vec<_>>()
        .join(",");
    format!("[{}]", entries)
}

fn column_schema_json(schema: &ColumnSchema) -> String {
    let cols = schema
        .columns
        .iter()
        .map(|c| format!("{{\"name\":\"{}\",\"role\":\"{:?}\"}}", json_escape(&c.name), c.role))
        .collect::<Vec<_>>()
        .join(",");
    format!("[{}]", cols)
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
    ///
    /// `timeline` is the investigation's context sidecar (if any), saved to
    /// `context.json` so `replay()` can reattach the same context at the
    /// same ticks instead of always replaying against an empty timeline.
    /// `channel_names` are the measurement column names in recording-column
    /// order, saved to `schema.json` so replay can produce named reports;
    /// pass an empty slice (or a slice whose length doesn't match the
    /// recording's channel count) to fall back to `ch0, ch1, ...`.
    #[allow(clippy::too_many_arguments)]
    pub fn save(
        dir: &Path,
        recording: &[Vec<f64>],
        incidents: &[Incident],
        baseline_samples: usize,
        name: &str,
        config: &CaseConfig,
        timeline: &ContextTimeline,
        channel_names: &[String],
    ) -> Result<Self, String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("create_dir_all: {}", e))?;
        let channels = recording.first().map(|r| r.len()).unwrap_or(0);
        let mut csv = (0..channels).map(|c| format!("ch{}", c)).collect::<Vec<_>>().join(",");
        csv.push('\n');
        for row in recording {
            csv.push_str(&row.iter().map(f64::to_string).collect::<Vec<_>>().join(","));
            csv.push('\n');
        }
        // Fingerprint the exact bytes about to be written to
        // recording.csv -- a different byte representation than the
        // original input file (`config.input_hash`) -- so `replay()` has
        // the right baseline to diff a fresh read of recording.csv
        // against.
        let recording_hash = fingerprint_content(csv.as_bytes());

        let names: Vec<String> = if channel_names.len() == channels {
            channel_names.to_vec()
        } else {
            (0..channels).map(|c| format!("ch{}", c)).collect()
        };
        let schema_json =
            format!("[{}]", names.iter().map(|n| format!("\"{}\"", json_escape(n))).collect::<Vec<_>>().join(","));

        let manifest = CaseManifest {
            name: name.to_string(),
            created: iso8601_now(),
            detector_version: env!("CARGO_PKG_VERSION").to_string(),
            baseline_samples,
            recording_path: dir.join("recording.csv").to_string_lossy().into_owned(),
            incidents_count: incidents.len(),
            channels,
            samples: recording.len(),
            input_hash: Some(config.input_hash.clone()),
            recording_hash: Some(recording_hash),
            schema_version: "0.1".to_string(),
        };
        let incidents_json =
            format!("[{}]", incidents.iter().map(Incident::to_json).collect::<Vec<_>>().join(","));
        let config_json = format!(
            "{{\"baseline_samples\":{},\"detector_version\":\"{}\",\"input_hash\":\"{}\",\
             \"monitor_export\":{},\"column_schema\":{},\"imputation\":{}}}",
            baseline_samples, env!("CARGO_PKG_VERSION"), json_escape(&config.input_hash),
            monitor_export_json(&config.monitor_export), column_schema_json(&config.column_schema),
            imputation_json(&config.imputation)
        );
        write_file(dir.join("manifest.json"), manifest.to_json())?;
        write_file(dir.join("recording.csv"), csv)?;
        write_file(dir.join("incidents.json"), incidents_json)?;
        write_file(dir.join("config.json"), config_json)?;
        write_file(dir.join("context.json"), context_timeline_json(timeline))?;
        write_file(dir.join("schema.json"), schema_json)?;
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
    /// per channel), a transpose of the row-per-tick CSV layout.
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

    /// Raw text of this case's `config.json` (the detector configuration
    /// saved alongside the recording, see [`CaseConfig`]). Used by
    /// `struktura replay` (Finding 5) to compare the saved configuration
    /// and recording fingerprint against a fresh recalibration.
    pub fn config_json(&self) -> Result<String, String> {
        let path = self.dir.join("config.json");
        std::fs::read_to_string(&path).map_err(|e| format!("read config.json: {}", e))
    }

    /// Reads `context.json` back into a [`ContextTimeline`] (the sidecar
    /// [`Case::save`] wrote alongside the recording). Legacy cases saved
    /// before `context.json` existed have no such file — that's not an
    /// error, it just means an empty timeline, same as an investigation
    /// with no `--context` given.
    pub fn context(&self) -> Result<ContextTimeline, String> {
        let path = self.dir.join("context.json");
        if !path.exists() {
            return Ok(ContextTimeline::new());
        }
        let text = std::fs::read_to_string(&path).map_err(|e| format!("read context.json: {}", e))?;
        parse_context_timeline(&text)
    }

    /// Per-channel display names in recording-column order, read back from
    /// `schema.json`. Falls back to `ch0, ch1, ...` for legacy cases saved
    /// before `schema.json` existed (or if it's unreadable/empty).
    pub fn channel_names(&self) -> Vec<String> {
        let path = self.dir.join("schema.json");
        if let Ok(text) = std::fs::read_to_string(&path) {
            let names = split_string_array(&text);
            if !names.is_empty() {
                return names;
            }
        }
        (0..self.manifest.channels).map(|c| format!("ch{}", c)).collect()
    }
}

/// Serialize a [`ContextTimeline`] to the `context.json` sidecar format: a
/// JSON array of `{"tick":N,"kind":"...","value":"..."}` objects.
fn context_timeline_json(timeline: &ContextTimeline) -> String {
    let events = timeline
        .iter()
        .map(|e| {
            format!(
                "{{\"tick\":{},\"kind\":\"{}\",\"value\":\"{}\"}}",
                e.tick,
                json_escape(&e.kind),
                json_escape(&e.value)
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!("[{}]", events)
}

/// Parse a `context.json` sidecar (see [`context_timeline_json`]) back into
/// a [`ContextTimeline`]. Anchor-split on each object's unique key, same
/// pattern as `parse_incident`.
fn parse_context_timeline(json: &str) -> Result<ContextTimeline, String> {
    let mut timeline = ContextTimeline::new();
    for obj in split_by_anchor(json, "{\"tick\":") {
        let tick = extract_u64(obj, "tick").ok_or("context event: missing tick")?;
        let kind = extract_str(obj, "kind").ok_or("context event: missing kind")?;
        let value = extract_str(obj, "value").ok_or("context event: missing value")?;
        timeline.push(ContextEvent::new(tick, kind, value));
    }
    Ok(timeline)
}

/// Extract just the `input_hash` field from a case's `config.json` text
/// (see [`Case::config_json`]): cheap drift detection without parsing the
/// full monitor export.
#[must_use]
pub fn parse_config_input_hash(json: &str) -> Option<String> {
    extract_str(json, "input_hash")
}

/// Extract the saved `(res_thr, dfa_thr, cusum_thr)` from a case's
/// `config.json` `monitor_export` object, for side-by-side comparison
/// against a fresh recalibration on `struktura replay`.
#[must_use]
pub fn parse_config_monitor_thresholds(json: &str) -> Option<(f64, f64, f64)> {
    Some((extract_f64(json, "res_thr")?, extract_f64(json, "dfa_thr")?, extract_f64(json, "cusum_thr")?))
}

/// Extract the *full* saved [`MonitorExport`] (every threshold and every
/// per-channel calibration field, not just the three global thresholds)
/// from a case's `config.json` `monitor_export` object, for a complete
/// drift comparison on `struktura replay`.
#[must_use]
pub fn parse_config_monitor_export(json: &str) -> Option<MonitorExport> {
    let res_thr = extract_f64(json, "res_thr")?;
    let dfa_thr = extract_f64(json, "dfa_thr")?;
    let cusum_thr = extract_f64(json, "cusum_thr")?;
    let channels_content = array_content(json, "channels")?;
    let channels: Vec<ChannelExport> =
        split_by_anchor(channels_content, "{\"ar_a\":").iter().map(|s| parse_channel_export(s)).collect::<Option<_>>()?;
    Some(MonitorExport { res_thr, dfa_thr, cusum_thr, channels })
}

fn parse_channel_export(s: &str) -> Option<ChannelExport> {
    Some(ChannelExport {
        ar_a: extract_f64(s, "ar_a")?,
        ar_b: extract_f64(s, "ar_b")?,
        ar_sd: extract_f64(s, "ar_sd")?,
        alpha_mean: extract_f64(s, "alpha_mean")?,
        alpha_sd: extract_f64(s, "alpha_sd")?,
        mean: extract_f64(s, "mean")?,
        roll_thr: extract_f64(s, "roll_thr")?,
        max_run: extract_u64(s, "max_run")? as usize,
        repeat_enabled: s.contains("\"repeat_enabled\":true"),
    })
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
            input_hash: Some("deadbeef01234567".to_string()),
            recording_hash: Some("beadfeed76543210".to_string()),
            schema_version: "0.1".to_string(),
        };
        assert_eq!(CaseManifest::from_json(&m.to_json()).expect("parses"), m);
    }

    #[test]
    fn manifest_json_parses_pre_v0_1_manifests_without_input_hash() {
        // Older manifests lack `input_hash`/`recording_hash`/`schema_version`
        // entirely; from_json must default rather than fail.
        let old = "{\"name\":\"old\",\"created\":\"2026-01-01T00:00:00Z\",\
                    \"detector_version\":\"1.0.0\",\"baseline_samples\":100,\
                    \"recording_path\":\"r.csv\",\"incidents_count\":0,\
                    \"channels\":2,\"samples\":300}";
        let m = CaseManifest::from_json(old).expect("parses");
        assert_eq!(m.input_hash, None);
        assert_eq!(m.recording_hash, None);
        assert_eq!(m.schema_version, "0.1");
    }

    fn test_case_config() -> CaseConfig {
        CaseConfig {
            input_hash: fingerprint_content(b"test fixture content"),
            monitor_export: MonitorExport {
                res_thr: 3.0, dfa_thr: 3.0, cusum_thr: 6.0,
                channels: vec![ChannelExport {
                    ar_a: 0.5, ar_b: 0.0, ar_sd: 1.0, alpha_mean: 0.5, alpha_sd: 0.05,
                    mean: 0.0, roll_thr: 2.0, max_run: 10, repeat_enabled: true,
                }],
            },
            column_schema: ColumnSchema::from_header(&["ch0", "ch1"]),
            imputation: vec![],
        }
    }

    #[test]
    fn fingerprint_content_is_deterministic_and_content_sensitive() {
        let a = fingerprint_content(b"hello world");
        let b = fingerprint_content(b"hello world");
        let c = fingerprint_content(b"hello world!");
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(a.len(), 16); // 64-bit hash as hex
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
        let config = test_case_config();
        let timeline = ContextTimeline::new();
        let case =
            Case::save(&dir, &recording, &incidents, 1, "roundtrip", &config, &timeline, &[]).expect("saves");
        assert_eq!((case.manifest.channels, case.manifest.samples), (2, 3));
        assert_eq!(case.manifest.input_hash, Some(config.input_hash.clone()));
        assert!(case.manifest.recording_hash.is_some());
        assert_eq!(case.manifest.schema_version, "0.1");

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
        // No channel_names given -> falls back to ch0..chN.
        assert_eq!(loaded.channel_names(), vec!["ch0", "ch1"]);
        // No context given -> empty timeline, not an error.
        assert!(loaded.context().expect("reads context.json").is_empty());

        let config_json = std::fs::read_to_string(dir.join("config.json")).expect("reads config.json");
        assert!(config_json.contains(&format!("\"input_hash\":\"{}\"", config.input_hash)));
        assert!(config_json.contains("\"monitor_export\":{"));
        assert!(config_json.contains("\"ar_a\":0.5"));
        assert!(config_json.contains("\"column_schema\":["));
        assert!(config_json.contains("\"name\":\"ch0\""));
        assert_eq!(config_json.matches('{').count(), config_json.matches('}').count());
        assert_eq!(config_json.matches('[').count(), config_json.matches(']').count());

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Finding 5: `Case::config_json` + `parse_config_input_hash` +
    /// `parse_config_monitor_thresholds` are what `replay()` uses to
    /// compare a fresh recalibration against what was saved.
    #[test]
    fn config_json_round_trips_input_hash_and_thresholds() {
        let dir = tmp_dir("config_json");
        let _ = std::fs::remove_dir_all(&dir);
        let recording = vec![vec![1.0, 2.0], vec![1.5, 2.5]];
        let config = test_case_config();
        let timeline = ContextTimeline::new();
        let case = Case::save(&dir, &recording, &[], 1, "config_json", &config, &timeline, &[]).expect("saves");

        let text = case.config_json().expect("reads config.json");
        assert_eq!(parse_config_input_hash(&text), Some(config.input_hash.clone()));
        assert_eq!(
            parse_config_monitor_thresholds(&text),
            Some((config.monitor_export.res_thr, config.monitor_export.dfa_thr, config.monitor_export.cusum_thr))
        );
        let full = parse_config_monitor_export(&text).expect("parses full monitor export");
        assert_eq!(full.channels.len(), config.monitor_export.channels.len());
        assert_eq!(full.channels[0].ar_a, config.monitor_export.channels[0].ar_a);
        assert_eq!(full.channels[0].max_run, config.monitor_export.channels[0].max_run);
        assert_eq!(full.channels[0].repeat_enabled, config.monitor_export.channels[0].repeat_enabled);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn config_json_errors_when_missing() {
        let dir = tmp_dir("config_json_missing");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create dir");
        let manifest = CaseManifest {
            name: "missing".to_string(), created: "2026-09-17T12:00:00Z".to_string(),
            detector_version: "1.0.0".to_string(), baseline_samples: 1,
            recording_path: "r.csv".to_string(), incidents_count: 0,
            channels: 1, samples: 1, input_hash: None, recording_hash: None,
            schema_version: "0.1".to_string(),
        };
        write_file(dir.join("manifest.json"), manifest.to_json()).expect("writes manifest");
        let case = Case::load(&dir).expect("loads");
        assert!(case.config_json().unwrap_err().contains("config.json"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Issue 3: a saved case's context sidecar (`context.json`) must round
    /// trip back into an equivalent `ContextTimeline`, and per-channel
    /// names given at save time must round trip via `schema.json`.
    #[test]
    fn context_and_schema_round_trip() {
        let dir = tmp_dir("context_schema");
        let _ = std::fs::remove_dir_all(&dir);
        let recording = vec![vec![1.0, 2.0], vec![1.5, 2.5], vec![2.0, 3.0]];
        let config = test_case_config();
        let mut timeline = ContextTimeline::new();
        timeline.push(ContextEvent::new(0, "mode", "boot"));
        timeline.push(ContextEvent::new(2, "cmd", "arm"));
        let names = vec!["motor_current_A".to_string(), "wheel_rpm".to_string()];

        let case = Case::save(&dir, &recording, &[], 1, "ctx", &config, &timeline, &names).expect("saves");
        assert_eq!(case.channel_names(), names);

        let loaded = Case::load(&dir).expect("loads");
        assert_eq!(loaded.channel_names(), names);
        let loaded_timeline = loaded.context().expect("reads context.json");
        assert_eq!(loaded_timeline.len(), 2);
        assert_eq!(loaded_timeline.active_at(0).unwrap().value, "boot");
        assert_eq!(loaded_timeline.active_at(2).unwrap().value, "arm");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
