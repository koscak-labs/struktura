//! `struktura generate --cfs|--fprime|--ros` (the directory generators in
//! `src/bin/struktura.rs`, backed by `struktura::codegen::{generate_cfs_main,
//! generate_fprime_cpp, generate_ros_monitor, generate_dfa_core_h}`) used to
//! carry a *second*, untested DFA implementation baked into `dfa_core.h`:
//! a fixed box list good only for window 512, a `y[512]` cap that silently
//! truncated any `--window` above 512, and no reordering of the ring buffer
//! into time order before scoring it. This compares their generated C/C++
//! output's alpha against Rust `struktura::dfa` on the same window, in time
//! order, for every `--window` the CLI accepts (not just 512).
//!
//! Each generator now includes the shared, tested `dfa_core.h`
//! (`generate_dfa_core_h`, built from `struktura::dfa_box_sizes(window)`,
//! same body as `generate_c_monitor`'s `dfa_compute`) and reorders its ring
//! into time order (`ordered`) before calling it -- the same fix already
//! proven for `generate_c_monitor`/`generate_cfs_app` in
//! `c_monitor_matches_rust.rs`.
//!
//! The cFS target is exercised by calling the generated `dfa_push` directly
//! (it becomes visible when the harness `#include`s the generated `.c`, as
//! the other differential tests do), against a minimal stand-in `cfe.h`. F
//! Prime and ROS 2 need their real framework types (an autocoder-generated
//! component base, or a real ROS 2 install) to build as shipped, neither of
//! which struktura provides or this machine has; those are exercised the
//! same way against minimal stand-in headers, reaching the generated
//! `pushSample`/`push_sample` (both private) via `#define private public`,
//! a standard, narrowly-scoped test technique -- every standard header the
//! generated file needs is included (and so already guarded) before that
//! macro is defined.

mod common;

use common::{diff_seed, scratch, Rng};
use std::fs;
use std::path::Path;
use std::process::Command;
use struktura::codegen::{
    generate_cfs_header, generate_cfs_main, generate_cfs_msgids, generate_dfa_core_h,
    generate_fprime_cpp, generate_fprime_hpp, generate_ros_monitor, Channel,
};

const WINDOWS: [usize; 5] = [64, 96, 128, 512, 1024];

fn default_channel() -> Channel {
    Channel {
        name: "input_value".into(),
        c_type: "double".into(),
        topic: "SAMPLE_MID".into(),
        field: "payload".into(),
        msg_type: "sample_msg_t".into(),
    }
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
/// in this crate do) and return one alpha per push once the ring is full.
fn alphas_from_cfs(cc: &str, window: usize, x: &[f64]) -> Vec<f64> {
    let channels = [default_channel()];
    let dir = scratch(&format!("cli-cfs-{window}"));
    write_all(
        &dir,
        &[
            ("cfe.h", CFE_H_STUB.to_string()),
            ("dfa_core.h", generate_dfa_core_h(window)),
            ("dfa_monitor_cfs.h", generate_cfs_header(&channels)),
            ("dfa_monitor_cfs_events.h", DFA_EVENTS_H_STUB.to_string()),
            ("dfa_monitor_cfs_msgids.h", generate_cfs_msgids(&channels)),
            ("dfa_monitor_cfs.c", generate_cfs_main(&channels, window, 0.08)),
            (
                "harness.c",
                "#include <stdio.h>\n#include <string.h>\n#include \"dfa_monitor_cfs.c\"\n\
                 int main(void) {\n    dfa_channel_t ch;\n    double v;\n\
                 memset(&ch, 0, sizeof(ch));\n\
                 while (scanf(\"%lf\", &v) == 1) {\n        dfa_push(&ch, v, \"harness\");\n\
                 if (ch.filled) printf(\"%.17e\\n\", dfa_compute(ch.ordered, DFA_WINDOW_SIZE).alpha);\n\
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
/// via `#define private public` (a standard, narrowly-scoped test
/// technique; every standard header the generated file needs is included,
/// and so already guarded, before that macro is defined). Returns one alpha
/// per push once the ring is full.
fn alphas_from_fprime(cxx: &str, window: usize, x: &[f64]) -> Vec<f64> {
    let channels = [default_channel()];
    let dir = scratch(&format!("cli-fprime-{window}"));
    write_all(
        &dir,
        &[
            ("DfaMonitorComponentAc.hpp", FPRIME_AC_STUB.to_string()),
            ("dfa_core.h", generate_dfa_core_h(window)),
            ("DfaMonitor.hpp", generate_fprime_hpp(&channels, window)),
            ("DfaMonitor.cpp", generate_fprime_cpp(&channels, window, 0.08)),
            (
                "harness.cpp",
                "#include <cstdio>\n#include <cstring>\n#include <cstdint>\n\
                 #define private public\n#include \"DfaMonitor.cpp\"\n\
                 int main(void) {\n    Svc::DfaMonitor mon(\"harness\");\n    double v;\n\
                 while (scanf(\"%lf\", &v) == 1) {\n\
                 mon.pushSample(mon.m_ch_input_value, v, \"harness\");\n\
                 if (mon.m_ch_input_value.filled)\n\
                 printf(\"%.17e\\n\", dfa_compute(mon.m_ch_input_value.ordered, DFA_WINDOW_SIZE).alpha);\n\
                 }\n    return 0;\n}\n"
                    .to_string(),
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
/// own. Returns one alpha per push once the ring is full.
fn alphas_from_ros(cxx: &str, window: usize, x: &[f64]) -> Vec<f64> {
    let channels = [default_channel()];
    let dir = scratch(&format!("cli-ros-{window}"));
    write_all(
        &dir,
        &[
            ("rclcpp/rclcpp.hpp", RCLCPP_STUB.to_string()),
            ("std_msgs/msg/float64.hpp", STD_MSGS_FLOAT64_STUB.to_string()),
            ("std_msgs/msg/string.hpp", STD_MSGS_STRING_STUB.to_string()),
            ("dfa_core.h", generate_dfa_core_h(window)),
            ("dfa_monitor_node.cpp", generate_ros_monitor(&channels, window, 0.08)),
            (
                "harness.cpp",
                "#include <cstdio>\n#include <cstring>\n#include <cstdint>\n\
                 #include <functional>\n#include <memory>\n#include <cmath>\n\
                 #define private public\n#define main dfa_generated_main\n\
                 #include \"dfa_monitor_node.cpp\"\n#undef main\n\
                 int main(void) {\n    DfaMonitorNode node;\n    double v;\n\
                 while (scanf(\"%lf\", &v) == 1) {\n\
                 node.push_sample(node.ch_input_value, v, \"harness\");\n\
                 if (node.ch_input_value.filled)\n\
                 printf(\"%.17e\\n\", dfa_compute(node.ch_input_value.ordered, DFA_WINDOW_SIZE).alpha);\n\
                 }\n    return 0;\n}\n"
                    .to_string(),
            ),
        ],
    );
    let input: String = x.iter().map(|v| format!("{v:.17e}\n")).collect();
    let out = compile_and_run_cxx(cxx, &dir, "harness.cpp", &input);
    let alphas = feed(&out);
    let _ = fs::remove_dir_all(&dir);
    alphas
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

        let cfs = alphas_from_cfs(&cc, window, &x);
        max_dalpha = max_dalpha.max(worst_diff(&rust, &cfs, "cfs/blended", window));
        compared += rust.len();

        let fprime = alphas_from_fprime(&cxx, window, &x);
        max_dalpha = max_dalpha.max(worst_diff(&rust, &fprime, "fprime/blended", window));
        compared += rust.len();

        let ros = alphas_from_ros(&cxx, window, &x);
        max_dalpha = max_dalpha.max(worst_diff(&rust, &ros, "ros/blended", window));
        compared += rust.len();
    }

    // Explicit white-noise / AR(1) / random-walk coverage at one
    // representative window (the blended series above already mixes
    // regimes, but not one of these in isolation).
    let window = 96usize;
    let series: [(&str, Vec<f64>); 3] = [
        ("white", white_noise(diff_seed("cli_generators_match_rust_white", 0x5EED_0001), 3 * window)),
        ("ar1", ar1_series(diff_seed("cli_generators_match_rust_ar1", 0x5EED_0002), 3 * window)),
        ("random_walk", random_walk(diff_seed("cli_generators_match_rust_rw", 0x5EED_0003), 3 * window)),
    ];
    for (name, x) in &series {
        let rust = rust_alphas(x, window);

        let cfs = alphas_from_cfs(&cc, window, x);
        max_dalpha = max_dalpha.max(worst_diff(&rust, &cfs, &format!("cfs/{name}"), window));
        compared += rust.len();

        let fprime = alphas_from_fprime(&cxx, window, x);
        max_dalpha = max_dalpha.max(worst_diff(&rust, &fprime, &format!("fprime/{name}"), window));
        compared += rust.len();

        let ros = alphas_from_ros(&cxx, window, x);
        max_dalpha = max_dalpha.max(worst_diff(&rust, &ros, &format!("ros/{name}"), window));
        compared += rust.len();
    }

    // Printed only on this path, after the C/C++ was compiled and compared.
    println!("cli_generators_match_rust: compared {compared} windows, max |dalpha| {max_dalpha:e}");
}
