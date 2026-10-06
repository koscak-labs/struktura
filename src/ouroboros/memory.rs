//! Memory: an append-only `knowledge.jsonl` next to the outbox.
//!
//! Each decision is one row keyed by a deterministic id (a hash of the
//! ledger, the logs, the knob space and the decision itself). Running the
//! loop again on unchanged inputs finds the id and appends nothing, and the
//! job it emitted keeps its name: the loop is idempotent, so a scheduler can
//! call it as often as it likes.

use std::io::Write;
use std::path::Path;

/// FNV-1a 64-bit: stable across runs and platforms (unlike std's hasher).
pub fn fnv64(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() { h ^= b as u64; h = h.wrapping_mul(0x0000_0100_0000_01b3); }
    h
}

pub fn decision_id(parts: &[&str]) -> String {
    format!("{:016x}", fnv64(&parts.join("\u{1f}")))
}

/// Rows of an existing knowledge file (missing file = empty memory).
pub fn load(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path).map(|t| t.lines().filter(|l| !l.trim().is_empty()).map(String::from).collect()).unwrap_or_default()
}

/// The job a recorded decision emitted, if this decision id is already in memory.
pub fn recalled(rows: &[String], id: &str) -> Option<Option<String>> {
    let key = format!("\"id\":\"{}\"", id);
    let row = rows.iter().find(|r| r.contains(&key))?;
    let job = crate::lab::parse_json(row).and_then(|j| j.str("emitted").map(String::from));
    Some(job)
}

/// Append one row unless a row with the same id exists. Returns true if written.
pub fn remember(path: &Path, id: &str, row: &str) -> std::io::Result<bool> {
    if recalled(&load(path), id).is_some() { return Ok(false); }
    if let Some(dir) = path.parent() { std::fs::create_dir_all(dir)?; }
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(f, "{}", row)?;
    Ok(true)
}

/// Next lab job number: one above any `lab-<N>-...` in the outbox or memory, at least `floor`.
pub fn next_job_id(outbox: &Path, rows: &[String], floor: u32) -> u32 {
    let mut max = floor.saturating_sub(1);
    let mut scan = |name: &str| {
        if let Some(rest) = name.strip_prefix("lab-") {
            if let Ok(n) = rest.split('-').next().unwrap_or("").parse::<u32>() { max = max.max(n); }
        }
    };
    if let Ok(rd) = std::fs::read_dir(outbox) {
        for e in rd.flatten() { scan(&e.file_name().to_string_lossy()); }
    }
    for r in rows {
        if let Some(j) = crate::lab::parse_json(r).and_then(|j| j.str("emitted").map(String::from)) { scan(&j); }
    }
    max + 1
}

/// JSON string escaping for row fields.
pub fn esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for c in s.chars() {
        match c { '"' => o.push_str("\\\""), '\\' => o.push_str("\\\\"), '\n' => o.push_str("\\n"), '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)), c => o.push(c) }
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("struktura-mem-{}-{}", tag, std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn fnv_is_stable() {
        assert_eq!(fnv64(""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv64("a"), 0xaf63_dc4c_8601_ec8c);
        assert_ne!(decision_id(&["a", "b"]), decision_id(&["ab", ""]));
    }

    #[test]
    fn remember_is_idempotent_and_recalls_the_job() {
        let d = tmp("idem");
        let p = d.join("knowledge.jsonl");
        let row = "{\"kind\":\"decision\",\"id\":\"00ff\",\"emitted\":\"lab-300-ub128\"}";
        assert!(remember(&p, "00ff", row).unwrap());
        assert!(!remember(&p, "00ff", row).unwrap());
        assert_eq!(load(&p).len(), 1);
        assert_eq!(recalled(&load(&p), "00ff"), Some(Some("lab-300-ub128".to_string())));
        assert_eq!(recalled(&load(&p), "beef"), None);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn job_ids_continue_after_outbox_and_memory() {
        let d = tmp("ids");
        std::fs::create_dir_all(&d).unwrap();
        assert_eq!(next_job_id(&d, &[], 300), 300);
        std::fs::write(d.join("lab-304-ub128.sh"), "").unwrap();
        assert_eq!(next_job_id(&d, &[], 300), 305);
        let rows = vec!["{\"id\":\"x\",\"emitted\":\"lab-311-draft5\"}".to_string()];
        assert_eq!(next_job_id(&d, &rows, 300), 312);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn escaping() {
        assert_eq!(esc("a\"b\\c\nd"), "a\\\"b\\\\c\\nd");
    }
}
