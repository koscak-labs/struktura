//! `struktura ledger-check` on the synthetic ledger fixtures: exit codes and
//! the key lines of the text and JSON reports.

use std::process::{Command, Output};

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/ledger");

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_struktura")).arg("ledger-check").args(args).output().expect("run struktura")
}

fn fx(name: &str) -> String { format!("{}/{}", DIR, name) }

fn check(args: &[&str], code: i32) -> String {
    let out = run(args);
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    assert_eq!(out.status.code(), Some(code), "args {:?}\nstdout:\n{}\nstderr:\n{}", args, text, String::from_utf8_lossy(&out.stderr));
    text
}

#[test]
fn valid_ledger_passes_strict() {
    let out = check(&[&fx("valid.jsonl"), "--strict"], 0);
    assert!(out.contains("issues: none"), "{}", out);
    assert!(out.contains("summary: 1 file(s), 18 lines, 18 rows, 11 kinds, 0 unparseable, 0 errors, 0 warnings (strict) -> OK"), "{}", out);
    assert!(out.lines().any(|l| l.split_whitespace().collect::<Vec<_>>() == ["job", "nuwa", "2", "0", "0"]), "{}", out);
}

#[test]
fn fuxi_ledger_passes_strict() {
    let out = check(&[&fx("fuxi.jsonl"), "--strict"], 0);
    for k in ["brain", "forecast", "score", "judge", "contract", "heartbeat", "gym"] {
        assert!(out.lines().any(|l| l.split_whitespace().collect::<Vec<_>>() == [k, "fuxi", "1", "0", "0"]), "{}: {}", k, out);
    }
}

#[test]
fn torn_tail_is_a_warning() {
    let out = check(&[&fx("torn-tail.jsonl")], 0);
    assert!(out.contains("torn-tail.jsonl:4: WARN torn tail"), "{}", out);
    assert!(out.contains("-> OK (warnings)"), "{}", out);
    // An in-flight append is not an unknown kind: still fine when strict.
    check(&[&fx("torn-tail.jsonl"), "--strict"], 0);
}

#[test]
fn mid_file_corruption_is_an_error() {
    let out = check(&[&fx("mid-corrupt.jsonl")], 1);
    assert!(out.contains("mid-corrupt.jsonl:4: ERROR unparseable JSON"), "{}", out);
    assert!(out.contains("-> FAIL"), "{}", out);
}

#[test]
fn unknown_kind_warns_and_strict_fails() {
    let out = check(&[&fx("unknown-kind.jsonl")], 0);
    assert!(out.contains("unknown-kind.jsonl:3: WARN [telemetry] unknown kind \"telemetry\""), "{}", out);
    assert!(out.contains("unknown-kind.jsonl:4: WARN [dream] unknown kind \"dream\""), "{}", out);
    let strict = check(&[&fx("unknown-kind.jsonl"), "--strict"], 1);
    assert!(strict.contains("unknown-kind.jsonl:3: ERROR [telemetry]"), "{}", strict);
}

#[test]
fn missing_or_ill_typed_fields_cite_lines() {
    let out = check(&[&fx("missing-field.jsonl")], 1);
    for want in [
        "missing-field.jsonl:2: ERROR [job] missing required field rc (int)",
        "missing-field.jsonl:3: ERROR [prediction] field verdict: expected str, got num 1",
        "missing-field.jsonl:4: ERROR [deploy] missing one of env|chunk",
        "missing-field.jsonl:5: ERROR [forecast] missing required field ts (num)",
    ] {
        assert!(out.contains(want), "missing {:?} in\n{}", want, out);
    }
    assert!(out.contains("0 unparseable, 4 errors, 0 warnings -> FAIL"), "{}", out);
}

#[test]
fn json_report_is_one_object() {
    let out = check(&[&fx("valid.jsonl"), &fx("torn-tail.jsonl"), &fx("missing-field.jsonl"), "--json"], 1);
    assert_eq!(out.trim_end().lines().count(), 1, "{}", out);
    let j = out.trim();
    assert!(j.starts_with("{\"ok\":false,\"status\":\"error\",\"strict\":false,"), "{}", j);
    assert!(j.contains("\"errors\":4,\"warnings\":1,\"files\":["), "{}", j);
    assert!(j.contains("\"torn_tail\":true"), "{}", j);
    assert!(j.contains("\"job\":{\"strand\":\"nuwa\",\"rows\":4,\"errors\":1,\"warnings\":0}"), "{}", j);
    assert!(j.contains("\"issues_total\":5,"), "{}", j);
    assert!(j.contains("\"line\":2,\"severity\":\"error\",\"kind\":\"job\",\"field\":\"rc\""), "{}", j);
    let ok = check(&[&fx("fuxi.jsonl"), "--json"], 0);
    assert!(ok.starts_with("{\"ok\":true,\"status\":\"ok\","), "{}", ok);
}

#[test]
fn issues_are_capped_at_twenty() {
    let dir = std::env::temp_dir().join(format!("struktura-ledger-check-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("many.jsonl");
    std::fs::write(&p, "{\"kind\":\"job\",\"job\":\"x\",\"start\":1,\"end\":2}\n".repeat(25)).unwrap();
    let out = check(&[p.to_str().unwrap()], 1);
    std::fs::remove_dir_all(&dir).ok();
    assert!(out.contains("issues (first 20 of 25):"), "{}", out);
    assert_eq!(out.lines().filter(|l| l.contains(": ERROR [job]")).count(), 20, "{}", out);
}

#[test]
fn schema_tsv_and_usage_errors() {
    let tsv = check(&["--schema"], 0);
    assert!(tsv.starts_with("kind\tfield\ttype\trequired\tknown\n"), "{}", tsv);
    assert!(tsv.contains("\njob\trc\tint\tyes\t-\n"), "{}", tsv);
    assert!(tsv.contains("\nheartbeat\tv\tconst:1\tyes\t-\n"), "{}", tsv);
    check(&[], 2);
    check(&["--bogus", &fx("valid.jsonl")], 2);
    check(&[&fx("does-not-exist.jsonl")], 2);
}
