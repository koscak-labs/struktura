//! `arms --tasks` must not mix workbench lanes: L01-L11 (live) and t01-t11 (bench) share numbers.
use std::process::Command;

fn verdicts() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("struktura-arms-lanes-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut s = String::new();
    // bench tasks: plain1 always passes, ladder3 always fails; live tasks: the reverse
    for t in ["t01-a", "t02-b", "t03-c", "t31-d"] { for seed in 1..=2 { s += &format!("plain1\t{t}\t{seed}\tpass\tttd_s=1\nladder3\t{t}\t{seed}\tfail\tttd_s=1\n"); } }
    for t in ["L01-x", "L02-y", "L03-z"] { for seed in 1..=2 { s += &format!("plain1\t{t}\t{seed}\tfail\tttd_s=1\nladder3\t{t}\t{seed}\tpass\tttd_s=1\n"); } }
    let p = dir.join("verdicts.tsv");
    std::fs::write(&p, s).unwrap();
    p
}

fn pass_rate_tasks(spec: &str) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_struktura"))
        .args(["arms", verdicts().to_str().unwrap(), "--tasks", spec, "--json"]).output().unwrap();
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    text.lines().find(|l| l.contains("\"metric\":\"pass_rate(")).map(|l| {
        let m = l.split("\"metric\":\"").nth(1).unwrap(); m[..m.find('"').unwrap()].to_string()
    }).unwrap_or_default()
}

#[test]
fn live_lane_range_never_pulls_in_bench_tasks() {
    assert_eq!(pass_rate_tasks("L01-L11"), "pass_rate(3 tasks)", "live lane only");
    assert_eq!(pass_rate_tasks("t01-t11"), "pass_rate(3 tasks)", "bench lane only");
    assert_eq!(pass_rate_tasks("t31-"), "pass_rate(1 tasks)");
    assert_eq!(pass_rate_tasks("1-11"), "pass_rate(6 tasks)", "a bare numeric range keeps its old meaning: every lane");
}

#[test]
fn split_mode_judges_each_lane_with_its_own_holm_family() {
    let out = Command::new(env!("CARGO_BIN_EXE_struktura"))
        .args(["arms", verdicts().to_str().unwrap(), "--split", "bench=t01-t31", "--split", "live=L01-L11", "--pairs", "plain1:ladder3", "--json"])
        .output().unwrap();
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    let splits: Vec<&str> = text.lines().filter(|l| l.contains("\"event\":\"split\"")).collect();
    assert_eq!(splits.len(), 2, "{text}");
    assert!(splits[0].contains("\"split\":\"bench\"") && splits[0].contains("\"tasks\":4") && splits[0].contains("\"family\":1"));
    assert!(splits[1].contains("\"split\":\"live\"") && splits[1].contains("\"tasks\":3"));
    // bench: plain1 passes every task, ladder3 none -> 4 discordant, p = 0.125: not decisive; live: the reverse, 3 tasks
    let pass: Vec<&str> = text.lines().filter(|l| l.contains("\"event\":\"split_pair\"") && l.contains("pass_rate")).collect();
    assert_eq!(pass.len(), 2);
    assert!(pass[0].contains("\"n_discordant\":4") && pass[0].contains("\"decisive\":false"));
    assert!(pass[1].contains("\"n_discordant\":3") && pass[1].contains("\"p_holm\":"));
    let bad = Command::new(env!("CARGO_BIN_EXE_struktura")).args(["arms", verdicts().to_str().unwrap(), "--split", "noequals"]).output().unwrap();
    assert_eq!(bad.status.code(), Some(2));
}
