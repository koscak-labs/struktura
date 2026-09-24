//! The standalone C monitor (`generate_c_monitor`) and the cFS app
//! (`generate_cfs_app`) keep the last `window` samples in a ring buffer. DFA
//! needs them in time order; computed on the ring as stored, the cumulative
//! profile has a jump where the newest samples meet the oldest.
//!
//! - `generate_c_monitor`: after every sample, the alpha the generated C
//!   computes must equal Rust `dfa()` on the same samples in time order, at
//!   the window used in practice (512), at the smallest window the generator
//!   accepts (64, where both return the 0.5 placeholder), and at the smallest
//!   window with a real alpha (72, three box sizes).
//! - `generate_cfs_app` calls the Rust FFI, so only the order matters: with
//!   `cfe.h` and `struktura.h` stubbed, the array it passes to
//!   `struktura_dfa` must be the last `window` samples in time order.

mod common;

use common::{c_compiler, compile_and_run, diff_seed, scratch, Rng};
use std::fs;
use struktura::codegen::{generate_c_monitor, generate_cfs_app};

fn series(seed: u64, n: usize) -> Vec<f64> {
    let mut rng = Rng(seed | 1);
    let mut x = 0.0;
    (0..n)
        .map(|i| match i / (n / 3).max(1) {
            0 => {
                x += rng.normal();
                x
            }
            1 => {
                x = 0.9 * x + rng.normal();
                x
            }
            _ => 0.01 * i as f64 + rng.normal(),
        })
        .collect()
}

fn check_c_monitor(cc: &str, window: usize) {
    let dir = scratch(&format!("c-monitor-{window}"));
    fs::write(dir.join("dfa_monitor.c"), generate_c_monitor(window, 0.08)).unwrap();
    fs::write(
        dir.join("harness.c"),
        "#include <stdio.h>\n#include \"dfa_monitor.c\"\n\
         int main(void) {\n    dfa_monitor_t m;\n    double v;\n    dfa_monitor_init(&m);\n\
         while (scanf(\"%lf\", &v) == 1) {\n        dfa_monitor_push(&m, v);\n\
         if (m.filled) printf(\"%.17e\\n\", dfa_compute(m.ordered, DFA_WINDOW_SIZE).alpha);\n\
         }\n    return 0;\n}\n",
    )
    .unwrap();

    let seed = diff_seed("c_monitor_matches_rust", 0x5EED_0F5A_B1E5);
    let x = series(seed + window as u64, 3 * window + window / 2);
    let input: Vec<String> = x.iter().map(|v| format!("{v:.17e}")).collect();
    let out = compile_and_run(cc, &dir, "harness.c", &(input.join("\n") + "\n"));
    let c: Vec<f64> = out.lines().map(|l| l.trim().parse().unwrap()).collect();

    // The monitor is filled after `window` samples; each later push scores the
    // last `window` samples.
    let rust: Vec<f64> = (window..=x.len())
        .map(|end| struktura::dfa(&x[end - window..end]).alpha)
        .collect();
    assert_eq!(
        c.len(),
        rust.len(),
        "window {window}: C scored {} windows, Rust {}",
        c.len(),
        rust.len()
    );

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
        "window {window}, step {i}: Rust alpha {} vs C alpha {} (|d| = {worst:e})",
        rust[i],
        c[i]
    );
}

#[test]
fn c_monitor_alpha_matches_rust_dfa() {
    let Some(cc) = c_compiler() else {
        eprintln!("skipping: no C compiler found");
        return;
    };
    for window in [512, 64, 72] {
        check_c_monitor(&cc, window);
    }
}

#[test]
fn cfs_app_passes_the_window_in_time_order() {
    let Some(cc) = c_compiler() else {
        eprintln!("skipping: no C compiler found");
        return;
    };
    let window = 96;
    let dir = scratch("cfs-app");
    fs::write(dir.join("health_app.c"), generate_cfs_app("Health", window)).unwrap();
    // Minimal stand-ins for the cFS and struktura headers the app includes.
    fs::write(
        dir.join("cfe.h"),
        "#include <stdint.h>\n#include <string.h>\n\
         typedef uint32_t uint32;\ntypedef uint8_t uint8;\n\
         #define CFE_EVS_EventType_INFORMATION 0\n#define CFE_EVS_EventType_ERROR 1\n\
         #define CFE_EVS_SendEvent(...) ((void)0)\n",
    )
    .unwrap();
    fs::write(
        dir.join("struktura.h"),
        "typedef struct { double alpha; double r_squared; } struktura_dfa_result_t;\n\
         #define STRUKTURA_WARNING 3\n\
         struktura_dfa_result_t struktura_dfa(const double *data, unsigned int len);\n\
         unsigned char struktura_health_check(double alpha, double baseline);\n",
    )
    .unwrap();
    fs::write(
        dir.join("harness.c"),
        "#include <stdio.h>\n#include \"health_app.c\"\n\
         static double seen[4096];\nstatic unsigned int seen_len;\nstatic int calls;\n\
         struktura_dfa_result_t struktura_dfa(const double *data, unsigned int len) {\n\
         struktura_dfa_result_t r = { 0.5, 0.0 };\n    unsigned int i;\n\
         for (i = 0; i < len; i++) seen[i] = data[i];\n    seen_len = len;\n    calls++;\n    return r;\n}\n\
         unsigned char struktura_health_check(double a, double b) { (void)a; (void)b; return 0; }\n\
         int main(void) {\n    double v;\n    unsigned int i;\n    Health_Init();\n\
         while (scanf(\"%lf\", &v) == 1) {\n        int before = calls;\n        Health_ProcessSample(v);\n\
         if (calls != before) {\n            for (i = 0; i < seen_len; i++) printf(\"%.17e \", seen[i]);\n\
         printf(\"\\n\");\n        }\n    }\n    return 0;\n}\n",
    )
    .unwrap();

    let x = series(diff_seed("cfs_app_time_order", 0xCF5), 3 * window + 17);
    let input: Vec<String> = x.iter().map(|v| format!("{v:.17e}")).collect();
    let out = compile_and_run(&cc, &dir, "harness.c", &(input.join("\n") + "\n"));
    let calls: Vec<Vec<f64>> = out
        .lines()
        .map(|l| l.split_whitespace().map(|t| t.parse().unwrap()).collect())
        .collect();
    let _ = fs::remove_dir_all(&dir);

    // One call per sample once the ring is full, each on the last `window` samples in order.
    assert_eq!(calls.len(), x.len() - window + 1);
    for (k, seen) in calls.iter().enumerate() {
        let end = window + k;
        assert_eq!(
            seen.as_slice(),
            &x[end - window..end],
            "call {k}: window not in time order"
        );
    }
}
