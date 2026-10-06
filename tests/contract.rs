//! Frozen decision contract: `contract/run.sh` replays synthetic lab fixtures through the
//! freshly built `struktura` binary (pulse, arms, lab, loop, power) and diffs the normalized
//! decisions against `contract/expected/`. A change to any decision the lab daemon acts on
//! (rollback verdicts, fragility, the next designed job) fails here until it is re-blessed
//! on purpose with `contract/run.sh <binary> --bless`.
//!
//! Needs bash and jq; skipped (with a note) when either is missing.

use std::process::Command;

fn have(tool: &str) -> bool {
    Command::new(tool).arg("--version").output().map(|o| o.status.success()).unwrap_or(false)
}

#[test]
fn decision_contract_holds() {
    if !have("bash") || !have("jq") {
        eprintln!("contract: bash or jq not found: skipped");
        return;
    }
    let script = concat!(env!("CARGO_MANIFEST_DIR"), "/contract/run.sh");
    let out = Command::new("bash")
        .arg(script)
        .arg(env!("CARGO_BIN_EXE_struktura"))
        .output()
        .expect("run contract/run.sh");
    let stdout = String::from_utf8_lossy(&out.stdout);
    println!("{}", stdout);
    eprintln!("{}", String::from_utf8_lossy(&out.stderr));
    assert!(out.status.success(), "decision contract broken (see FAIL lines above):\n{}", stdout);
    assert!(stdout.contains("SUMMARY cases="), "contract runner printed no summary");
}
