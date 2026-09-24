//! The C99 hybrid monitor must compute the same DFA alpha as the Rust monitor
//! that calibrates it. `generate_hybrid_c` bakes `alpha_mean` and `alpha_sd`,
//! computed with `dfa_fast_into`, into C; if the C measured alpha on other box
//! sizes, the DFA leg would score flight data on a different scale.
//!
//! This compiles the generated file with the system C compiler and compares
//! `hyb_dfa_alpha` against `dfa_fast_into` on seeded random windows of
//! `monitor::WINDOW` samples (white noise, random walk, AR(1), sine plus noise,
//! ramp plus noise).

mod common;

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

use struktura::codegen::generate_hybrid_c;
use struktura::monitor::{HybridMonitor, WINDOW};

/// xorshift64*, so the windows are the same on every platform.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn window(rng: &mut Rng, family: usize) -> Vec<f64> {
    let mut v = Vec::with_capacity(WINDOW);
    match family {
        0 => (0..WINDOW).for_each(|_| v.push(rng.normal())),
        1 => {
            let mut x = 0.0;
            for _ in 0..WINDOW {
                x += rng.normal();
                v.push(x);
            }
        }
        2 => {
            let phi = 0.5 + 0.49 * rng.uniform();
            let mut x = 0.0;
            for _ in 0..WINDOW {
                x = phi * x + rng.normal();
                v.push(x);
            }
        }
        3 => {
            let f = 0.02 + 0.2 * rng.uniform();
            let a = 0.5 + 3.0 * rng.uniform();
            for i in 0..WINDOW {
                v.push(a * (f * i as f64).sin() + rng.normal());
            }
        }
        _ => {
            let slope = 0.1 * rng.normal();
            for i in 0..WINDOW {
                v.push(slope * i as f64 + rng.normal());
            }
        }
    }
    v
}

#[test]
fn hybrid_c_dfa_alpha_matches_rust() {
    // Fails on CI and Linux without a C compiler; prints SKIPPED elsewhere.
    let Some(cc) = common::c_compiler() else {
        return;
    };

    // Fixed seed by default; a scheduled job can set STRUKTURA_DIFF_SEED to
    // try new windows. The seed is printed so a failure can be replayed.
    let seed: u64 = std::env::var("STRUKTURA_DIFF_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0x5EED_0F5A_B1E5);
    println!("hybrid_c_matches_rust: STRUKTURA_DIFF_SEED={seed}");
    let mut rng = Rng(seed | 1);
    let clean: Vec<Vec<f64>> = (0..2)
        .map(|_| {
            let mut x = 0.0;
            (0..4 * WINDOW)
                .map(|_| {
                    x = 0.7 * x + rng.normal();
                    x
                })
                .collect()
        })
        .collect();
    let export = HybridMonitor::calibrate(&clean)
        .expect("calibration")
        .export();

    let windows: Vec<Vec<f64>> = (0..5)
        .flat_map(|f| (0..40).map(move |_| f))
        .map(|f| window(&mut rng, f))
        .collect();
    let mut buf = Vec::new();
    let rust: Vec<f64> = windows
        .iter()
        .map(|w| struktura::dfa_fast_into(w, &mut buf).alpha)
        .collect();

    let dir: PathBuf =
        std::env::temp_dir().join(format!("struktura-hybrid-c-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("hybrid_monitor.c"), generate_hybrid_c(&export)).unwrap();
    fs::write(
        dir.join("harness.c"),
        "#include <stdio.h>\n#include \"hybrid_monitor.c\"\n\
         int main(void) {\n    double v[HYB_WINDOW];\n    int i;\n    for (;;) {\n\
         for (i = 0; i < HYB_WINDOW; i++) if (scanf(\"%lf\", &v[i]) != 1) return 0;\n\
         printf(\"%.17e\\n\", hyb_dfa_alpha(v, HYB_WINDOW));\n    }\n}\n",
    )
    .unwrap();
    let mut input = fs::File::create(dir.join("windows.txt")).unwrap();
    for w in &windows {
        let line: Vec<String> = w.iter().map(|x| format!("{x:.17e}")).collect();
        writeln!(input, "{}", line.join(" ")).unwrap();
    }
    drop(input);

    let exe = dir.join(if cfg!(windows) {
        "harness.exe"
    } else {
        "harness"
    });
    let out = Command::new(&cc)
        .current_dir(&dir)
        .args(["-std=c99", "-O2", "-o"])
        .arg(&exe)
        .arg("harness.c")
        .arg("-lm")
        .output()
        .expect("run C compiler");
    assert!(
        out.status.success(),
        "{cc} failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let run = Command::new(&exe)
        .stdin(fs::File::open(dir.join("windows.txt")).unwrap())
        .output()
        .unwrap();
    let c: Vec<f64> = String::from_utf8_lossy(&run.stdout)
        .lines()
        .map(|l| l.trim().parse().unwrap())
        .collect();
    assert_eq!(c.len(), rust.len());

    let (worst, i) = rust
        .iter()
        .zip(&c)
        .map(|(r, c)| (r - c).abs())
        .enumerate()
        .fold(
            (0.0f64, 0usize),
            |acc, (i, d)| if d > acc.0 { (d, i) } else { acc },
        );
    let _ = fs::remove_dir_all(&dir);
    assert!(
        worst <= 1e-10,
        "window {i}: Rust alpha {} vs C alpha {} (|d| = {worst:e})",
        rust[i],
        c[i]
    );
    // Printed only on this path, after the C was compiled and compared.
    println!(
        "hybrid_c_matches_rust: compared {} windows, max |dalpha| {worst:e}",
        c.len()
    );
}
