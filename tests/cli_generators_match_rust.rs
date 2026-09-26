//! `struktura generate --cfs|--fprime|--ros` (the directory generators in
//! `src/bin/struktura.rs`, backed by `struktura::codegen::{generate_cfs_main,
//! generate_fprime_cpp, generate_ros_monitor, generate_dfa_core_h}`) used to
//! carry a *second*, untested DFA implementation baked into `dfa_core.h`: a
//! fixed box list `{16,24,36,54,81,121}` that never matched Rust's
//! `dfa_box_sizes` at any window -- including 512 (up to about 0.39 alpha
//! off there) and, below window 144, always the `{0.5, 0.0}` placeholder, so
//! those monitors could never set a baseline or alarm -- a `y[512]`-capped
//! scratch array on top of that, and no reordering of the ring buffer into
//! time order before scoring it. This compares their generated C/C++
//! output's alpha against Rust `struktura::dfa` on the same window, in time
//! order, at windows 64/72/96/97/128/250/512/1024 (64 compares the
//! `{0.5, 0.0}` placeholder on both sides; real alpha coverage starts at 72)
//! plus explicit white-noise/AR(1)/random-walk regimes at window 96, and a
//! negative control that reverts the time-order fix and asserts the
//! comparison actually notices.
//!
//! Each generator now includes the shared, tested `dfa_core.h`
//! (`generate_dfa_core_h`, built from `struktura::dfa_box_sizes(window)`,
//! same body as `generate_c_monitor`'s `dfa_compute`) and reorders its ring
//! into time order (`ordered`) before calling it -- the same fix already
//! proven for `generate_c_monitor`/`generate_cfs_app` in
//! `c_monitor_matches_rust.rs`. Each push function also stores the
//! `dfa_result_t` it computed into a `last` field on the channel, and this
//! test reads alpha from there -- the value the generated monitor's own
//! baseline/shift logic acted on -- rather than recomputing it, since
//! `dfa_compute` now overwrites its input with the cumulative profile (see
//! `src/codegen.rs`'s `dfa_compute_block` doc) and a second call on the same
//! buffer would not be the raw window any more.
//!
//! The cFS target is exercised by calling the generated `dfa_push` directly
//! (it becomes visible when the harness `#include`s the generated `.c`, as
//! the other differential tests do), against a minimal stand-in `cfe.h`. F
//! Prime and ROS 2 need their real framework types (an autocoder-generated
//! component base, or a real ROS 2 install) to build as shipped, neither of
//! which struktura provides or this machine has; those are exercised the
//! same way against minimal stand-in headers, reaching the generated
//! `pushSample`/`push_sample` (both private) via `#define private public`.
//! Every standard header either stub or `dfa_core.h` needs (`<cstdio>`,
//! `<cstdint>`, `<cstring>`, `<cmath>`, `<math.h>`, `<functional>`,
//! `<memory>`, `<string>`) is included, and so already guarded, in both
//! harnesses before that macro is defined -- reusing a keyword as a macro
//! name while a standard header is still being parsed is undefined
//! behavior ([macro.names]); pre-including everything first means the
//! guards skip those headers entirely once the macro is live.

mod common;

use common::{diff_seed, scratch, Rng};
use std::fs;
use std::path::Path;
use std::process::Command;
use struktura::codegen::{
    generate_cfs_header, generate_cfs_main, generate_cfs_msgids, generate_dfa_core_h,
    generate_fprime_cpp, generate_fprime_hpp, generate_ros_monitor, Channel,
};

/// 64 only exercises the `{0.5, 0.0}` placeholder (both sides agree
/// trivially); 72 is the smallest window with a real alpha; 97 and 250 are
/// odd/non-power windows; 512 and 1024 cover the large end (1024 wraps the
/// ring three and a half times over, per `blended_series` below).
const WINDOWS: [usize; 8] = [64, 72, 96, 97, 128, 250, 512, 1024];

fn default_channel() -> Channel {
    Channel::new("input_value", "SAMPLE_MID", "payload", "sample_msg_t")
}

// ---- series generators: white noise, AR(1), random walk -------------------

fn white_noise(seed: u64, n: usize) -> Vec<f64> {
    let mut rng = Rng(seed | 1);
    (0..n).map(|_| rng.normal()).collect()
}

fn ar1_series(seed: u64, n: usize) -> Vec<f64> {
    let mut rng = Rng(seed | 1);
    let mut x = 0.0;
    (0..n)
        .map(|_| {
            x = 0.9 * x + rng.normal();
            x
        })
        .collect()
}

fn random_walk(seed: u64, n: usize) -> Vec<f64> {
    let mut rng = Rng(seed | 1);
    let mut x = 0.0;
    (0..n)
        .map(|_| {
            x += rng.normal();
            x
        })
        .collect()
}

/// Forward-filled telemetry: a quantized random walk that holds each value
/// for 1 to 90 samples, so many windows contain box sizes that see only a
/// constant run (F exactly 0). Rounding noise there once gave alpha off by up
/// to 285 between Rust and the generated C on ESA-ADB data.
fn forward_filled(seed: u64, n: usize) -> Vec<f64> {
    let mut rng = Rng(seed | 1);
    let (mut x, mut hold) = (0.0, 0usize);
    (0..n)
        .map(|_| {
            if hold == 0 {
                x += (rng.normal() * 2.0).round() * 0.25;
                hold = 1 + (rng.uniform() * 90.0) as usize;
            }
            hold -= 1;
            x
        })
        .collect()
}

/// White noise, then AR(1), then a slow drift -- a blend of regimes in one
/// run, as the other differential tests in this crate use.
fn blended_series(seed: u64, n: usize) -> Vec<f64> {
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

fn rust_alphas(x: &[f64], window: usize) -> Vec<f64> {
    (window..=x.len())
        .map(|end| struktura::dfa(&x[end - window..end]).alpha)
        .collect()
}

fn worst_diff(rust: &[f64], other: &[f64], label: &str, window: usize) -> f64 {
    assert_eq!(
        rust.len(),
        other.len(),
        "{label} window {window}: Rust scored {} windows, generated C/C++ scored {}",
        rust.len(),
        other.len()
    );
    let (worst, i) = rust
        .iter()
        .zip(other)
        .map(|(r, c)| (r - c).abs())
        .enumerate()
        .fold((0.0f64, 0usize), |acc, (i, d)| if d > acc.0 { (d, i) } else { acc });
    assert!(
        worst <= 1e-9,
        "{label} window {window}, step {i}: Rust alpha {} vs generated {} (|d| = {worst:e})",
        rust[i],
        other[i]
    );
    worst
}

/// Count of `|rust - other| > 1e-9`, for the negative control (which expects
/// mismatches rather than asserting their absence).
fn count_mismatches(rust: &[f64], other: &[f64]) -> usize {
    assert_eq!(rust.len(), other.len());
    rust.iter().zip(other).filter(|(r, c)| (**r - **c).abs() > 1e-9).count()
}

// ---- C++ compiler helper (mirrors common::c_compiler, for a C compiler) ---

fn cxx_compiler() -> Option<String> {
    let mut candidates: Vec<String> = std::env::var("CXX").into_iter().collect();
    candidates.extend(["c++", "g++", "clang++"].iter().map(|s| s.to_string()));
    let found = candidates.into_iter().find(|c| {
        Command::new(c)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    });
    if found.is_none() {
        if cfg!(target_os = "linux") || std::env::var_os("CI").is_some() {
            panic!("no C++ compiler found (tried $CXX, c++, g++, clang++)");
        }
        println!("SKIPPED: no C++ compiler found (tried $CXX, c++, g++, clang++)");
    }
    found
}

fn write_all(dir: &Path, files: &[(&str, String)]) {
    for (name, content) in files {
        let path = dir.join(name);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, content).unwrap();
    }
}

fn compile_and_run_cxx(cxx: &str, dir: &Path, source: &str, stdin_text: &str) -> String {
    let exe = dir.join(if cfg!(windows) { "prog.exe" } else { "prog" });
    let out = Command::new(cxx)
        .current_dir(dir)
        .args(["-std=c++17", "-O2", "-o"])
        .arg(&exe)
        .arg(source)
        .arg("-lm")
        .output()
        .expect("run C++ compiler");
    assert!(
        out.status.success(),
        "{cxx} failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let input = dir.join("stdin.txt");
    fs::write(&input, stdin_text).unwrap();
    let run = Command::new(&exe)
        .stdin(fs::File::open(&input).unwrap())
        .output()
        .unwrap();
    assert!(run.status.success(), "generated program failed");
    String::from_utf8_lossy(&run.stdout).into_owned()
}

fn feed(cc_out: &str) -> Vec<f64> {
    cc_out.lines().map(|l| l.trim().parse().unwrap()).collect()
}

/// Standard headers every stub or `dfa_core.h` needs, pre-included (and so
/// already guarded) before either C++ harness defines `private` as `public`.
const CXX_HARNESS_PRELUDE: &str = "\
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <math.h>
#include <functional>
#include <memory>
#include <string>
";

const CFE_H_STUB: &str = "\
/* Minimal stand-in for NASA cFS's cfe.h: just enough for the generated
 * dfa_monitor_cfs.c to compile. Nothing here is exercised at runtime --
 * the differential test calls the generated dfa_push directly, bypassing
 * the cFS message bus entirely. */
#ifndef CFE_H
#define CFE_H

#include <stdint.h>
#include <stdbool.h>
#include <string.h>

typedef uint32_t uint32;
typedef uint8_t uint8;
typedef int32_t CFE_Status_t;
#define CFE_SUCCESS 0

typedef uint32_t CFE_SB_PipeId_t;
typedef struct { double payload; } sample_msg_t;
typedef struct { sample_msg_t Msg; } CFE_SB_Buffer_t;

typedef uint32_t CFE_SB_MsgId_t;
#define CFE_SB_INVALID_MSG_ID ((CFE_SB_MsgId_t)0xFFFFFFFFu)

#define CFE_ES_RunStatus_APP_RUN   1
#define CFE_ES_RunStatus_APP_ERROR 0
#define CFE_EVS_EventFilter_BINARY 0
#define CFE_EVS_EventType_INFORMATION 0
#define CFE_EVS_EventType_ERROR       1

static inline void CFE_EVS_Register(void *filters, uint32 n, int scheme) {
    (void)filters; (void)n; (void)scheme;
}
static inline CFE_Status_t CFE_SB_CreatePipe(CFE_SB_PipeId_t *pipe, uint32 depth, const char *name) {
    (void)depth; (void)name; *pipe = 1; return CFE_SUCCESS;
}
static inline void CFE_SB_Subscribe(CFE_SB_MsgId_t id, CFE_SB_PipeId_t pipe) { (void)id; (void)pipe; }
static inline CFE_SB_MsgId_t CFE_SB_ValueToMsgId(uint32 v) { return (CFE_SB_MsgId_t)v; }
static inline CFE_Status_t CFE_SB_ReceiveBuffer(CFE_SB_Buffer_t **buf, CFE_SB_PipeId_t pipe, int timeout) {
    (void)buf; (void)pipe; (void)timeout; return CFE_SUCCESS;
}
static inline bool CFE_ES_RunLoop(uint32 *status) { (void)status; return false; }
static inline void CFE_ES_ExitApp(uint32 status) { (void)status; }
static inline void CFE_MSG_GetMsgId(sample_msg_t *msg, CFE_SB_MsgId_t *id) { (void)msg; *id = CFE_SB_INVALID_MSG_ID; }
static inline int CFE_SB_MsgId_Equal(CFE_SB_MsgId_t a, CFE_SB_MsgId_t b) { return a == b; }

#define CFE_EVS_SendEvent(...) ((void)0)

#endif
";

const DFA_EVENTS_H_STUB: &str = "\
#ifndef DFA_MONITOR_CFS_EVENTS_H
#define DFA_MONITOR_CFS_EVENTS_H
#define DFA_MON_INIT_EID     1
#define DFA_MON_BASELINE_EID 2
#define DFA_MON_SHIFT_EID    3
#endif
";

/// Feed `x` through the generated cFS app's `dfa_push` (visible once the
/// harness `#include`s the generated `.c`, as the other differential tests
/// in this crate do) and return one alpha per push once the ring is full,
/// read from `ch.last.alpha` -- the value `dfa_push` itself fed to the
/// baseline/shift logic -- not recomputed by the harness. `mutate` is
/// applied to the generated `dfa_monitor_cfs.c` text before it is compiled,
/// so a caller can plant a regression (the negative control below reverts
/// the time-order fix this way).
fn alphas_from_cfs(cc: &str, window: usize, x: &[f64], mutate: impl Fn(String) -> String) -> Vec<f64> {
    let channels = [default_channel()];
    let dir = scratch(&format!("cli-cfs-{window}-{}", x.len()));
    write_all(
        &dir,
        &[
            ("cfe.h", CFE_H_STUB.to_string()),
            ("dfa_core.h", generate_dfa_core_h(window)),
            ("dfa_monitor_cfs.h", generate_cfs_header(&channels)),
            ("dfa_monitor_cfs_events.h", DFA_EVENTS_H_STUB.to_string()),
            ("dfa_monitor_cfs_msgids.h", generate_cfs_msgids(&channels)),
            ("dfa_monitor_cfs.c", mutate(generate_cfs_main(&channels, window, 0.08))),
            (
                "harness.c",
                "#include <stdio.h>\n#include <string.h>\n#include \"dfa_monitor_cfs.c\"\n\
                 int main(void) {\n    dfa_channel_t ch;\n    double v;\n\
                 memset(&ch, 0, sizeof(ch));\n\
                 while (scanf(\"%lf\", &v) == 1) {\n        dfa_push(&ch, v, \"harness\");\n\
                 if (ch.filled) printf(\"%.17e\\n\", ch.last.alpha);\n\
                 }\n    return 0;\n}\n"
                    .to_string(),
            ),
        ],
    );
    let input: String = x.iter().map(|v| format!("{v:.17e}\n")).collect();
    let out = common::compile_and_run(cc, &dir, "harness.c", &input);
    let alphas = feed(&out);
    let _ = fs::remove_dir_all(&dir);
    alphas
}

const FPRIME_AC_STUB: &str = "\
#ifndef DFA_MONITOR_COMPONENT_AC_STUB_HPP
#define DFA_MONITOR_COMPONENT_AC_STUB_HPP
/* Minimal stand-in for F Prime's autocoder-generated component base: just
 * enough for the generated DfaMonitor to compile. The differential test
 * reaches its private pushSample directly (via `#define private public`),
 * so these methods are never exercised at runtime -- only their
 * signatures need to match. */
#include <cstdint>

typedef double F64;
typedef std::uint32_t U32;
typedef int NATIVE_INT_TYPE;

struct FwTlmBuffer {
    double v = 0.0;
    void deserialize(F64 &out) const { out = v; }
};

class DfaMonitorComponentBase {
public:
    explicit DfaMonitorComponentBase(const char *name) { (void)name; }
    void log_ACTIVITY_HI_BaselineEstablished(const char *name, F64 alpha, F64 r2) {
        (void)name; (void)alpha; (void)r2;
    }
    void log_WARNING_HI_StructuralShift(const char *name, F64 baseline, F64 current, F64 delta) {
        (void)name; (void)baseline; (void)current; (void)delta;
    }
    void tlmWrite_DfaAlpha(F64 a) { (void)a; }
    void tlmWrite_DfaR2(F64 r2) { (void)r2; }
};

#endif
";

/// Feed `x` through the generated F Prime component's `pushSample`, reached
/// via `#define private public` after every standard header the stub and
/// `dfa_core.h` need is already pre-included (see `CXX_HARNESS_PRELUDE`).
/// Returns one alpha per push once the ring is full, read from
/// `ch.last.alpha`. `mutate` is applied to the generated `DfaMonitor.cpp`
/// text before it is compiled.
fn alphas_from_fprime(cxx: &str, window: usize, x: &[f64], mutate: impl Fn(String) -> String) -> Vec<f64> {
    let channels = [default_channel()];
    let dir = scratch(&format!("cli-fprime-{window}-{}", x.len()));
    write_all(
        &dir,
        &[
            ("DfaMonitorComponentAc.hpp", FPRIME_AC_STUB.to_string()),
            ("dfa_core.h", generate_dfa_core_h(window)),
            ("DfaMonitor.hpp", generate_fprime_hpp(&channels, window)),
            ("DfaMonitor.cpp", mutate(generate_fprime_cpp(&channels, window, 0.08))),
            (
                "harness.cpp",
                format!(
                    "{CXX_HARNESS_PRELUDE}\
                     #define private public\n#include \"DfaMonitor.cpp\"\n\
                     int main(void) {{\n    Svc::DfaMonitor mon(\"harness\");\n    double v;\n\
                     while (scanf(\"%lf\", &v) == 1) {{\n\
                     mon.pushSample(mon.m_ch_input_value, v, \"harness\");\n\
                     if (mon.m_ch_input_value.filled)\n\
                     printf(\"%.17e\\n\", mon.m_ch_input_value.last.alpha);\n\
                     }}\n    return 0;\n}}\n"
                ),
            ),
        ],
    );
    let input: String = x.iter().map(|v| format!("{v:.17e}\n")).collect();
    let out = compile_and_run_cxx(cxx, &dir, "harness.cpp", &input);
    let alphas = feed(&out);
    let _ = fs::remove_dir_all(&dir);
    alphas
}

const RCLCPP_STUB: &str = "\
#ifndef RCLCPP_STUB_HPP
#define RCLCPP_STUB_HPP
/* Minimal stand-in for ROS 2's rclcpp: just enough for the generated node
 * to compile. The differential test reaches its private push_sample
 * directly (via `#define private public`), so create_subscription/
 * create_publisher are never exercised at runtime -- only their types
 * need to match. */
#include <functional>
#include <memory>
#include <string>

namespace rclcpp {

template <typename MsgT>
struct Subscription {
    using SharedPtr = std::shared_ptr<Subscription<MsgT>>;
};

template <typename MsgT>
struct Publisher {
    using SharedPtr = std::shared_ptr<Publisher<MsgT>>;
    void publish(const MsgT &) {}
};

struct Logger {};

class Node {
public:
    explicit Node(const std::string &name) { (void)name; }

    template <typename MsgT>
    typename Subscription<MsgT>::SharedPtr create_subscription(
        const std::string &topic, int qos,
        std::function<void(const typename MsgT::SharedPtr)> cb) {
        (void)topic; (void)qos; (void)cb;
        return std::make_shared<Subscription<MsgT>>();
    }

    template <typename MsgT>
    typename Publisher<MsgT>::SharedPtr create_publisher(const std::string &topic, int qos) {
        (void)topic; (void)qos;
        return std::make_shared<Publisher<MsgT>>();
    }

    Logger get_logger() { return Logger{}; }
};

inline void init(int, char **) {}
inline void shutdown() {}
template <typename NodeT>
inline void spin(std::shared_ptr<NodeT>) {}

} // namespace rclcpp

#define RCLCPP_INFO(logger, ...) ((void)0)
#define RCLCPP_WARN(logger, ...) ((void)0)

#endif
";

const STD_MSGS_FLOAT64_STUB: &str = "\
#ifndef STD_MSGS_MSG_FLOAT64_STUB_HPP
#define STD_MSGS_MSG_FLOAT64_STUB_HPP
#include <memory>
namespace std_msgs { namespace msg {
struct Float64 {
    using SharedPtr = std::shared_ptr<Float64>;
    double data;
};
}}
#endif
";

const STD_MSGS_STRING_STUB: &str = "\
#ifndef STD_MSGS_MSG_STRING_STUB_HPP
#define STD_MSGS_MSG_STRING_STUB_HPP
#include <memory>
#include <string>
namespace std_msgs { namespace msg {
struct String {
    using SharedPtr = std::shared_ptr<String>;
    std::string data;
};
}}
#endif
";

/// Feed `x` through the generated ROS 2 node's `push_sample`, reached via
/// `#define private public` plus renaming its generated `main` out of the
/// way (`#define main dfa_generated_main`) so the harness can supply its
/// own, after every standard header the stubs and `dfa_core.h` need is
/// already pre-included (see `CXX_HARNESS_PRELUDE`). Returns one alpha per
/// push once the ring is full, read from `ch.last.alpha`. `mutate` is
/// applied to the generated `dfa_monitor_node.cpp` text before it is
/// compiled.
fn alphas_from_ros(cxx: &str, window: usize, x: &[f64], mutate: impl Fn(String) -> String) -> Vec<f64> {
    let channels = [default_channel()];
    let dir = scratch(&format!("cli-ros-{window}-{}", x.len()));
    write_all(
        &dir,
        &[
            ("rclcpp/rclcpp.hpp", RCLCPP_STUB.to_string()),
            ("std_msgs/msg/float64.hpp", STD_MSGS_FLOAT64_STUB.to_string()),
            ("std_msgs/msg/string.hpp", STD_MSGS_STRING_STUB.to_string()),
            ("dfa_core.h", generate_dfa_core_h(window)),
            (
                "dfa_monitor_node.cpp",
                mutate(generate_ros_monitor(&channels, window, 0.08)),
            ),
            (
                "harness.cpp",
                format!(
                    "{CXX_HARNESS_PRELUDE}\
                     #define private public\n#define main dfa_generated_main\n\
                     #include \"dfa_monitor_node.cpp\"\n#undef main\n\
                     int main(void) {{\n    DfaMonitorNode node;\n    double v;\n\
                     while (scanf(\"%lf\", &v) == 1) {{\n\
                     node.push_sample(node.ch_input_value, v, \"harness\");\n\
                     if (node.ch_input_value.filled)\n\
                     printf(\"%.17e\\n\", node.ch_input_value.last.alpha);\n\
                     }}\n    return 0;\n}}\n"
                ),
            ),
        ],
    );
    let input: String = x.iter().map(|v| format!("{v:.17e}\n")).collect();
    let out = compile_and_run_cxx(cxx, &dir, "harness.cpp", &input);
    let alphas = feed(&out);
    let _ = fs::remove_dir_all(&dir);
    alphas
}

/// Identity source transform: no planted regression.
fn no_mutation(s: String) -> String {
    s
}

#[test]
fn cli_generators_match_rust_dfa() {
    let Some(cc) = common::c_compiler() else {
        return;
    };
    let Some(cxx) = cxx_compiler() else {
        return;
    };

    let mut max_dalpha = 0.0f64;
    let mut compared = 0usize;

    for &window in &WINDOWS {
        let seed = diff_seed("cli_generators_match_rust", 0xC11_6E4E_5A70_0000 ^ window as u64);
        let x = blended_series(seed + window as u64, 3 * window + window / 2);
        let rust = rust_alphas(&x, window);

        let cfs = alphas_from_cfs(&cc, window, &x, no_mutation);
        max_dalpha = max_dalpha.max(worst_diff(&rust, &cfs, "cfs/blended", window));
        compared += rust.len();

        let fprime = alphas_from_fprime(&cxx, window, &x, no_mutation);
        max_dalpha = max_dalpha.max(worst_diff(&rust, &fprime, "fprime/blended", window));
        compared += rust.len();

        let ros = alphas_from_ros(&cxx, window, &x, no_mutation);
        max_dalpha = max_dalpha.max(worst_diff(&rust, &ros, "ros/blended", window));
        compared += rust.len();
    }

    // Explicit white-noise / AR(1) / random-walk coverage at one
    // representative window (the blended series above already mixes
    // regimes, but not one of these in isolation).
    let window = 96usize;
    let series: [(&str, Vec<f64>); 4] = [
        ("white", white_noise(diff_seed("cli_generators_match_rust_white", 0x5EED_0001), 3 * window)),
        ("ar1", ar1_series(diff_seed("cli_generators_match_rust_ar1", 0x5EED_0002), 3 * window)),
        ("random_walk", random_walk(diff_seed("cli_generators_match_rust_rw", 0x5EED_0003), 3 * window)),
        ("forward_filled", forward_filled(diff_seed("cli_generators_match_rust_ff", 0x5EED_0004), 12 * window)),
    ];
    for (name, x) in &series {
        let rust = rust_alphas(x, window);

        let cfs = alphas_from_cfs(&cc, window, x, no_mutation);
        max_dalpha = max_dalpha.max(worst_diff(&rust, &cfs, &format!("cfs/{name}"), window));
        compared += rust.len();

        let fprime = alphas_from_fprime(&cxx, window, x, no_mutation);
        max_dalpha = max_dalpha.max(worst_diff(&rust, &fprime, &format!("fprime/{name}"), window));
        compared += rust.len();

        let ros = alphas_from_ros(&cxx, window, x, no_mutation);
        max_dalpha = max_dalpha.max(worst_diff(&rust, &ros, &format!("ros/{name}"), window));
        compared += rust.len();
    }

    // Printed only on this path, after the C/C++ was compiled and compared.
    println!("cli_generators_match_rust: compared {compared} windows, max |dalpha| {max_dalpha:e}");
}

/// Proves the comparison above can actually fail: reverts the time-order fix
/// (the exact pre-fix bug -- `dfa_compute` scoring the ring in storage order
/// instead of `ordered`) in the generated source text for all three targets,
/// the way `hybrid_c_alarms_match_rust.rs`'s negative control scales its C
/// threshold, and asserts the comparison notices in every one.
#[test]
fn cli_generators_negative_control_detects_storage_order_regression() {
    let Some(cc) = common::c_compiler() else {
        return;
    };
    let Some(cxx) = cxx_compiler() else {
        return;
    };

    let window = 128usize;
    let seed = diff_seed("cli_generators_negative_control", 0xBAD_C0DE);
    let x = blended_series(seed, 3 * window + window / 2);
    let rust = rust_alphas(&x, window);

    let revert_cfs = move |s: String| -> String {
        let needle = format!("dfa_compute(ch->ordered, {window})");
        let replacement = format!("dfa_compute(ch->buffer, {window})");
        let out = s.replacen(&needle, &replacement, 1);
        assert_ne!(out, s, "negative control did not change the generated cFS C");
        out
    };
    let revert_fprime = move |s: String| -> String {
        let needle = format!("dfa_compute(ch.ordered, {window})");
        let replacement = format!("dfa_compute(ch.buffer, {window})");
        let out = s.replacen(&needle, &replacement, 1);
        assert_ne!(out, s, "negative control did not change the generated F Prime C++");
        out
    };
    let revert_ros = revert_fprime; // Copy: only captures `window` (usize)

    let cfs_bad = alphas_from_cfs(&cc, window, &x, revert_cfs);
    let fprime_bad = alphas_from_fprime(&cxx, window, &x, revert_fprime);
    let ros_bad = alphas_from_ros(&cxx, window, &x, revert_ros);

    let cfs_mismatches = count_mismatches(&rust, &cfs_bad);
    let fprime_mismatches = count_mismatches(&rust, &fprime_bad);
    let ros_mismatches = count_mismatches(&rust, &ros_bad);

    println!(
        "cli_generators_match_rust: negative control (storage-order regression, window {window}) \
         mismatches cfs {cfs_mismatches}/{n} fprime {fprime_mismatches}/{n} ros {ros_mismatches}/{n}",
        n = rust.len()
    );
    assert!(
        cfs_mismatches > 0,
        "negative control produced no cFS mismatches: the comparison is not sensitive"
    );
    assert!(
        fprime_mismatches > 0,
        "negative control produced no F Prime mismatches: the comparison is not sensitive"
    );
    assert!(
        ros_mismatches > 0,
        "negative control produced no ROS mismatches: the comparison is not sensitive"
    );
}

/// `struktura generate` rejects a `--window` too small for `dfa_compute` to
/// ever return a real alpha (below 72, `dfa_box_sizes` gives fewer than 3
/// box sizes, so the monitor could never set a baseline or alarm), instead
/// of silently generating one. Needs no C/C++ compiler: this only checks the
/// CLI's own argument validation.
#[test]
fn generate_rejects_too_small_window() {
    let exe = env!("CARGO_BIN_EXE_struktura");
    let dir = scratch("window-validation");

    let rejected = Command::new(exe)
        .args(["generate", "--cfs", "--window", "64", "-o"])
        .arg(dir.join("too_small"))
        .output()
        .expect("run struktura generate");
    assert!(
        !rejected.status.success(),
        "expected --window 64 to be rejected, got exit {:?}",
        rejected.status.code()
    );
    let stderr = String::from_utf8_lossy(&rejected.stderr);
    assert!(
        stderr.contains("too small") && stderr.contains("72"),
        "unexpected stderr for --window 64: {stderr}"
    );
    assert!(
        !dir.join("too_small").exists(),
        "rejected --window 64 should not have created an output directory"
    );

    let zero = Command::new(exe)
        .args(["generate", "--cfs", "--window", "0", "-o"])
        .arg(dir.join("zero"))
        .output()
        .expect("run struktura generate");
    assert!(!zero.status.success(), "expected --window 0 to be rejected");

    let accepted = Command::new(exe)
        .args(["generate", "--cfs", "--window", "72", "-o"])
        .arg(dir.join("minimum_ok"))
        .output()
        .expect("run struktura generate");
    assert!(
        accepted.status.success(),
        "expected --window 72 (the minimum) to be accepted: {}",
        String::from_utf8_lossy(&accepted.stderr)
    );

    let _ = fs::remove_dir_all(&dir);
}
