//! `struktura brain gym --seed 1 --json` is deterministic: two runs print identical output and
//! exit with the same status (0, or 3 when an expected-pass trial flipped).
//!
//! The full gym is a release-mode workload (about 25 s wall on 14 cores); an unoptimised build
//! is far slower, so a debug `cargo test` checks the same property on the cheapest trials.

use std::process::{Command, Output};

fn gym(extra: &[&str]) -> std::process::Child {
    Command::new(env!("CARGO_BIN_EXE_struktura"))
        .args(["brain", "gym", "--seed", "1", "--json"])
        .args(extra)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn struktura")
}

#[test]
fn brain_gym_seed_1_json_is_deterministic() {
    let extra: &[&str] = if cfg!(debug_assertions) { &["--only", "recorder,regime,sleep"] } else { &[] };
    // Both runs at once: each trial is a pure function of the seed, so contention cannot change output.
    let (a, b) = (gym(extra), gym(extra));
    let (a, b): (Output, Output) = (a.wait_with_output().unwrap(), b.wait_with_output().unwrap());
    let code = a.status.code();
    assert!(matches!(code, Some(0) | Some(3)), "exit status {:?}", a.status);
    assert_eq!(code, b.status.code(), "same exit status");
    let out = String::from_utf8(a.stdout).unwrap();
    assert_eq!(out, String::from_utf8(b.stdout).unwrap(), "two runs of the same seed differ");
    let lines: Vec<&str> = out.lines().collect();
    let trials = lines.iter().filter(|l| l.starts_with("{\"event\":\"trial\"")).count();
    let expect = if cfg!(debug_assertions) { 3 } else { struktura::brain_gym::TRIALS.len() };
    assert_eq!(trials, expect, "{}", out);
    let summary = lines.last().unwrap();
    assert!(summary.starts_with("{\"event\":\"summary\",\"seed\":1,"), "{}", summary);
    // Exit 3 exactly when an expected-pass trial flipped.
    assert_eq!(code == Some(3), summary.contains("\"regression\":true"), "{}", summary);
}

#[test]
fn brain_gym_rejects_an_unknown_trial() {
    let out = Command::new(env!("CARGO_BIN_EXE_struktura")).args(["brain", "gym", "--only", "no-such-trial"]).output().unwrap();
    assert_eq!(out.status.code(), Some(2));
}
