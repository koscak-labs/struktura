//! Helpers for the tests that compile generated C and compare it with Rust.
// Each test binary uses only some of these helpers.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::Command;

/// The seed for a differential test: `STRUKTURA_DIFF_SEED` when set (the
/// scheduled job passes the date), otherwise `default`. Printed so a failing
/// run can be replayed with the same value.
pub fn diff_seed(test: &str, default: u64) -> u64 {
    let seed = std::env::var("STRUKTURA_DIFF_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default);
    println!("{test}: STRUKTURA_DIFF_SEED={seed}");
    seed
}

/// xorshift64*, so the inputs are the same on every platform.
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// A C compiler: $CC, then cc, gcc, clang.
///
/// A missing compiler fails the test on CI (`CI` set) and on Linux, so a
/// comparison can never pass there without running. Elsewhere the test prints
/// `SKIPPED` and returns; the tests print a `compared ...` line only when the
/// C was compiled and compared, and the claims rows require that line.
pub fn c_compiler() -> Option<String> {
    let mut candidates: Vec<String> = std::env::var("CC").into_iter().collect();
    candidates.extend(["cc", "gcc", "clang"].iter().map(|s| s.to_string()));
    let found = candidates.into_iter().find(|c| {
        Command::new(c)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    });
    if found.is_none() {
        if cfg!(target_os = "linux") || std::env::var_os("CI").is_some() {
            panic!("no C compiler found (tried $CC, cc, gcc, clang)");
        }
        println!("SKIPPED: no C compiler found (tried $CC, cc, gcc, clang)");
    }
    found
}

/// A fresh scratch directory for one test.
pub fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("struktura-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Compile `source` (in `dir`) with `cc`, run it with `stdin_text`, return stdout.
pub fn compile_and_run(cc: &str, dir: &Path, source: &str, stdin_text: &str) -> String {
    let exe = dir.join(if cfg!(windows) { "prog.exe" } else { "prog" });
    let out = Command::new(cc)
        .current_dir(dir)
        .args(["-std=c99", "-O2", "-o"])
        .arg(&exe)
        .arg(source)
        .arg("-lm")
        .output()
        .expect("run C compiler");
    assert!(
        out.status.success(),
        "{cc} failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let input = dir.join("stdin.txt");
    std::fs::write(&input, stdin_text).unwrap();
    let run = Command::new(&exe)
        .stdin(std::fs::File::open(&input).unwrap())
        .output()
        .unwrap();
    assert!(run.status.success(), "generated program failed");
    String::from_utf8_lossy(&run.stdout).into_owned()
}
