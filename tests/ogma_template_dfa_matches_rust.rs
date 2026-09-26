//! `ogma-template/` ships hand-written cFS/F Prime templates for the
//! nasa/ogma discussion-#315 custom-template workflow (see
//! `ogma-template/TEMPLATE_GUIDE.md`), not `struktura::codegen` output --
//! most of them carry `{{mustache}}` placeholders a real `struktura
//! generate` invocation never emits, so they are hand-maintained, not
//! regenerated, and this crate's Cargo.toml excludes the directory from the
//! published crate. They used to carry the same bug this crate's other
//! differential tests found and fixed in the generated C: a fixed box list
//! that never matched Rust's `dfa_box_sizes`, a window-sized profile array
//! on the stack, and no ring reorder into time order.
//!
//! `dfa_core.h` itself has no mustache tags (self-contained, shared
//! verbatim into every generated app), so it is compiled directly here. The
//! per-channel ring state and push logic (`dfa_channel_t`/`dfa_channel_push`
//! in `dfa_monitor_cfs.c`) also has no mustache tags -- only the
//! surrounding message-handler boilerplate does -- so it is extracted from
//! the actual file on disk by exact text markers (this test fails loudly if
//! those markers stop matching, rather than silently drifting from what is
//! shipped) and compiled against minimal stand-in cFS macros.

mod common;

use common::{c_compiler, compile_and_run, diff_seed, scratch, Rng};
use std::fs;
use std::path::PathBuf;

/// ogma-template's `DFA_WINDOW_SIZE` default (both `dfa_core.h` files and
/// `DfaMonitor.hpp` all `#ifndef`-default to 256).
const WINDOW: usize = 256;

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

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Pull `dfa_channel_t` and `dfa_channel_push` out of the real
/// `dfa_monitor_cfs.c` by exact text markers. Panics (failing the test) if
/// the file no longer contains them verbatim, rather than silently testing
/// stale, copy-pasted logic.
fn extract_between(src: &str, start_marker: &str, end_marker: &str, what: &str) -> String {
    let start = src
        .find(start_marker)
        .unwrap_or_else(|| panic!("{what}: start marker not found in dfa_monitor_cfs.c (file layout changed?)"));
    let end_rel = src[start..]
        .find(end_marker)
        .unwrap_or_else(|| panic!("{what}: end marker not found in dfa_monitor_cfs.c (file layout changed?)"));
    let end = start + end_rel + end_marker.len();
    src[start..end].to_string()
}

/// Pull just `dfa_channel_t` and `dfa_channel_push` out of the real
/// `dfa_monitor_cfs.c` by exact text markers, skipping the mustache/CFE
/// boilerplate declared between them (global telemetry state, event
/// filters, `dfa_channel_init`) that this test does not need and has not
/// stubbed. Panics (failing the test) if the file no longer contains them
/// verbatim, rather than silently testing stale, copy-pasted logic.
fn extract_channel_push(src: &str) -> String {
    // ogma-template's files carry CRLF line endings; normalize before
    // searching so the markers below (written with plain `\n`) match.
    let src = src.replace("\r\n", "\n");
    let dfa_channel_t = extract_between(
        &src,
        "typedef struct {\n    double buffer[DFA_WINDOW_SIZE];",
        "\n} dfa_channel_t;\n",
        "dfa_channel_t struct",
    );
    let dfa_channel_push = extract_between(
        &src,
        "static int dfa_channel_push(dfa_channel_t *ch, double value,\n                            const char *name) {",
        "\n    return 1;\n}\n",
        "dfa_channel_push function",
    );
    format!("{dfa_channel_t}\n{dfa_channel_push}\n")
}

#[test]
fn ogma_template_dfa_core_matches_rust() {
    let Some(cc) = c_compiler() else {
        return;
    };
    let hdr_path = repo_root().join("ogma-template/cfs/dfa_monitor/fsw/src/dfa_core.h");
    let hdr = fs::read_to_string(&hdr_path).expect("read ogma-template cfs dfa_core.h");
    assert!(
        hdr.contains("{16, 18, 20, 23, 26, 30, 34, 38, 43, 49, 56, 64}"),
        "expected dfa_core.h's box table to be struktura::dfa_box_sizes(256)"
    );

    let dir = scratch("ogma-template-dfa-core");
    fs::write(dir.join("dfa_core.h"), &hdr).unwrap();
    fs::write(
        dir.join("harness.c"),
        "#include <stdio.h>\n#include \"dfa_core.h\"\n\
         int main(void) {\n    static double v[4096];\n    int n = 0;\n    double x;\n\
         while (n < 4096 && scanf(\"%lf\", &x) == 1) v[n++] = x;\n\
         printf(\"%.17e\\n\", dfa_compute(v, n).alpha);\n    return 0;\n}\n",
    )
    .unwrap();

    let seed = diff_seed("ogma_template_dfa_core_matches_rust", 424_242);
    let x = series(seed, WINDOW);
    let input: String = x.iter().map(|v| format!("{v:.17e}\n")).collect();
    let out = compile_and_run(&cc, &dir, "harness.c", &input);
    let c_alpha: f64 = out.trim().parse().unwrap();
    let rust_alpha = struktura::dfa(&x).alpha;
    let _ = fs::remove_dir_all(&dir);

    let d = (c_alpha - rust_alpha).abs();
    assert!(
        d <= 1e-9,
        "ogma-template cfs dfa_core.h: alpha {c_alpha} vs Rust {rust_alpha} (|d| = {d:e})"
    );
    println!("ogma_template_dfa_matches_rust: cfs dfa_core.h window {WINDOW}, |dalpha| {d:e}");

    // The F Prime template's dfa_core.h is the same fix applied by hand a
    // second time; check it did not drift from the cFS one.
    let fp_hdr = fs::read_to_string(
        repo_root().join("ogma-template/fprime/dfa_monitor/dfa_core.h"),
    )
    .expect("read ogma-template fprime dfa_core.h");
    assert!(
        fp_hdr.contains("{16, 18, 20, 23, 26, 30, 34, 38, 43, 49, 56, 64}"),
        "expected F Prime dfa_core.h's box table to be struktura::dfa_box_sizes(256) too"
    );
}

#[test]
fn ogma_template_ring_reorder_matches_rust() {
    let Some(cc) = c_compiler() else {
        return;
    };
    let src_path = repo_root().join("ogma-template/cfs/dfa_monitor/fsw/src/dfa_monitor_cfs.c");
    let src = fs::read_to_string(&src_path).expect("read ogma-template dfa_monitor_cfs.c");
    let snippet = extract_channel_push(&src);
    assert!(
        snippet.contains("ch->ordered[i] = ch->buffer[(ch->pos + i) % DFA_WINDOW_SIZE]"),
        "extracted dfa_channel_push no longer reorders the ring into time order"
    );
    assert!(
        snippet.contains("dfa_compute(ch->ordered, DFA_WINDOW_SIZE)"),
        "extracted dfa_channel_push no longer calls dfa_compute on the ordered window"
    );

    let hdr = fs::read_to_string(
        repo_root().join("ogma-template/cfs/dfa_monitor/fsw/src/dfa_core.h"),
    )
    .unwrap();

    let dir = scratch("ogma-template-ring");
    fs::write(dir.join("dfa_core.h"), &hdr).unwrap();
    fs::write(
        dir.join("harness.c"),
        format!(
            "#include <stdio.h>\n#include <string.h>\n#include <math.h>\n\
             #include \"dfa_core.h\"\n\
             typedef unsigned int uint32;\ntypedef unsigned char uint8;\n\
             #define DFA_LEARN_WINDOWS 10\n#define DFA_R2_MIN 0.7\n#define DFA_THRESHOLD 0.08\n\
             #define CFE_EVS_SendEvent(...) ((void)0)\n\
             #define CFE_EVS_EventType_INFORMATION 0\n#define CFE_EVS_EventType_ERROR 1\n\
             #define DFA_MONITOR_BASELINE_INF_EID 1\n#define DFA_MONITOR_SHIFT_ERR_EID 2\n\
             #define DFA_MONITOR_INSUFFICIENT_INF_EID 3\n\
             {snippet}\n\
             int main(void) {{\n    dfa_channel_t ch;\n    double v;\n    memset(&ch, 0, sizeof(ch));\n\
             while (scanf(\"%lf\", &v) == 1) {{\n        dfa_channel_push(&ch, v, \"harness\");\n\
             if (ch.filled) printf(\"%.17e\\n\", ch.last_alpha);\n    }}\n    return 0;\n}}\n"
        ),
    )
    .unwrap();

    let seed = diff_seed("ogma_template_ring_reorder_matches_rust", 0xC0DE);
    let x = series(seed, 3 * WINDOW + WINDOW / 2);
    let input: String = x.iter().map(|v| format!("{v:.17e}\n")).collect();
    let out = compile_and_run(&cc, &dir, "harness.c", &input);
    let c: Vec<f64> = out.lines().map(|l| l.trim().parse().unwrap()).collect();
    let rust: Vec<f64> = (WINDOW..=x.len())
        .map(|end| struktura::dfa(&x[end - WINDOW..end]).alpha)
        .collect();
    let _ = fs::remove_dir_all(&dir);

    assert_eq!(c.len(), rust.len(), "ogma-template scored {} windows, Rust {}", c.len(), rust.len());
    let (worst, i) = rust
        .iter()
        .zip(&c)
        .map(|(r, c)| (r - c).abs())
        .enumerate()
        .fold((0.0f64, 0usize), |acc, (i, d)| if d > acc.0 { (d, i) } else { acc });
    assert!(
        worst <= 1e-9,
        "ogma-template dfa_channel_push step {i}: Rust alpha {} vs C alpha {} (|d| = {worst:e})",
        rust[i],
        c[i]
    );
    println!(
        "ogma_template_dfa_matches_rust: compared {} windows, max |dalpha| {worst:e}",
        c.len()
    );
}
