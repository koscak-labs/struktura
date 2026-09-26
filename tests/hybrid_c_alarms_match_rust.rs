//! Full-behaviour differential oracle: the C from `generate_hybrid_c` must raise
//! the same alarms as the Rust `HybridMonitor` it was generated from, on the same
//! streams: same tick, same channel, same leg.
//!
//! The C has five legs (residual, repeated value, DFA, level shift, residual
//! CUSUM) and no parity or missingness leg, so those two are disabled on the Rust
//! side. Streams come from seeded generators (white, AR(1) with phi 0.3..0.97,
//! random walk, quantized) with injected faults (step, stuck, variance change,
//! drift, spike, one NaN then a step) on 1..3 channel calibrations. One stream
//! per run is 40,000 samples long, past the C ring phase counter's wrap.
//!
//! Two comparisons:
//! - first alarm per stream through the per-channel `hyb_push`, no reset;
//! - the full alarm sequence through the generated `hyb_push_sample` and
//!   `hyb_reset`, against `HybridMonitor::push` and `reset`, resetting both
//!   sides after every alarm. (Before `hyb_push_sample` existed, driving the C
//!   one channel at a time made the channels after an alarming one skip that
//!   sample, and 2 of 240 sequences diverged after their first alarm.)
//!
//! A planted negative control (residual threshold scaled by 0.9 in the C) must
//! produce mismatches, so the comparison cannot pass vacuously.

mod common;

use std::fmt::Write as _;
use std::fs;
use std::process::Command;

use common::Rng;
use struktura::codegen::generate_hybrid_c;
use struktura::monitor::{HybridMonitor, Leg, WINDOW};

type Alarm = (usize, usize, u8); // (tick, channel, leg as in hyb_verdict_t)

fn leg_code(l: Leg) -> u8 {
    match l {
        Leg::Residual => 1,
        Leg::RepeatedValue => 2,
        Leg::Dfa => 3,
        Leg::LevelShift => 4,
        Leg::ResidualCusum => 5,
        Leg::Missingness | Leg::Parity => 9,
    }
}

const LEGS: [&str; 6] = ["-", "residual", "repeated", "dfa", "level", "cusum"];

#[derive(Clone, Copy)]
struct Proc {
    kind: usize, // 0 white, 1 AR(1), 2 random walk, 3 quantized AR(1), 4 TS ramp, 5 random walk with drift
    phi: f64,
    level: f64,
    scale: f64,
    quantum: f64,
    /// Kind 4 only: per-sample ramp rate (applied to `offset + i`, so the
    /// calibration and its continuation streams see one unbroken line).
    slope: f64,
    /// Kind 5 only: per-step random-walk drift.
    drift: f64,
}

fn draw_proc(rng: &mut Rng) -> Proc {
    let phi = 0.3 + 0.67 * rng.uniform();
    let scale = 0.2 + 3.0 * rng.uniform();
    // AR(phi) noise's own stationary standard deviation (unit innovation
    // variance): 1/sqrt(1-phi^2).
    let sigma_x = 1.0 / (1.0 - phi * phi).sqrt();
    Proc {
        kind: (rng.next_u64() % 6) as usize,
        phi,
        level: 10.0 * rng.normal(),
        scale,
        quantum: 0.25 + 0.75 * rng.uniform(),
        // Tuned so the certification gate's R2 lands around 0.6-0.9 over a
        // 768-sample calibration (8*WINDOW): the ramp's total excursion
        // (slope*768) is a few multiples of the AR(phi) noise's own
        // stationary spread (scale*sigma_x), the same ratio T1's fixture
        // (monitor.rs's drift_tests) uses.
        slope: (5.0 + 5.0 * rng.uniform()) * scale * sigma_x / 768.0,
        drift: 0.05 + 0.05 * rng.uniform(),
    }
}

/// `offset` is the absolute sample index of `x`'s first element — 0 for a
/// calibration, the calibration's own length for a stream that continues it
/// (kind 4's ramp is evaluated at `offset + i`, so a calibration and its
/// continuation streams see one unbroken line, not a discontinuity at the
/// boundary).
fn series(rng: &mut Rng, p: Proc, n: usize, offset: usize) -> Vec<f64> {
    let mut x = 0.0;
    (0..n)
        .map(|i| {
            x = match p.kind {
                0 => rng.normal(),
                2 => x + 0.1 * rng.normal(),
                5 => x + p.drift + 0.1 * rng.normal(),
                _ => p.phi * x + rng.normal(),
            };
            let v = p.level + p.scale * x
                + if p.kind == 4 { p.slope * (offset + i) as f64 } else { 0.0 };
            if p.kind == 3 {
                (v / (p.quantum * p.scale)).round() * p.quantum * p.scale
            } else {
                v
            }
        })
        .collect()
}

/// Inject one fault into channel `ch` from tick `at`. Returns a label.
fn inject(rng: &mut Rng, s: &mut [Vec<f64>], ch: usize, at: usize, scale: f64) -> &'static str {
    let n = s[ch].len();
    match rng.next_u64() % 8 {
        0 => {
            let k = (1.0 + 4.0 * rng.uniform()) * scale * if rng.uniform() < 0.5 { -1.0 } else { 1.0 };
            (at..n).for_each(|t| s[ch][t] += k);
            "step"
        }
        1 => {
            let v = s[ch][at];
            (at..n).for_each(|t| s[ch][t] = v);
            "stuck"
        }
        2 => {
            let m = s[ch][..at].iter().sum::<f64>() / at as f64;
            let g = 2.0 + 3.0 * rng.uniform();
            (at..n).for_each(|t| s[ch][t] = m + g * (s[ch][t] - m));
            "variance"
        }
        3 => {
            let r = (0.002 + 0.02 * rng.uniform()) * scale;
            (at..n).for_each(|t| s[ch][t] += r * (t - at) as f64);
            "drift"
        }
        4 => {
            for k in 0..(1 + rng.next_u64() % 4) as usize {
                if at + 3 * k < n {
                    s[ch][at + 3 * k] += (6.0 + 10.0 * rng.uniform()) * scale;
                }
            }
            "spike"
        }
        5 => {
            // One NaN reading, then a step: a NaN must not disable any leg
            // for the step that follows.
            s[ch][at] = f64::NAN;
            let k = (2.0 + 3.0 * rng.uniform()) * scale;
            (at + 200..n).for_each(|t| s[ch][t] += k);
            "nan+step"
        }
        6 => {
            // Rate change: extra slope from `at` onward, on top of
            // whatever the channel already does in calibration (including
            // a certified drift, on a kind-4/5 channel).
            let r = (0.002 + 0.02 * rng.uniform()) * scale;
            (at..n).for_each(|t| s[ch][t] += r * (t - at) as f64);
            "rate_change"
        }
        _ => "none",
    }
}

#[allow(clippy::needless_range_loop)] // channel-major data, indexed by tick
fn rust_alarms(mon: &mut HybridMonitor, stream: &[Vec<f64>], reset: bool) -> Vec<Alarm> {
    let (nch, n) = (stream.len(), stream[0].len());
    let mut out = Vec::new();
    let mut sample = vec![0.0; nch];
    for t in 0..n {
        for c in 0..nch {
            sample[c] = stream[c][t];
        }
        if let Some(leg) = mon.push(&sample) {
            let ch = mon.last_alarm().map_or(usize::MAX, |r| r.channel);
            out.push((t, ch, leg_code(leg)));
            if !reset {
                break;
            }
            mon.reset();
        }
    }
    out
}

/// C driver: reads "mode nch n" then n lines of nch values; prints "t c leg" per alarm.
const DRIVER: &str = r#"
#include <stdio.h>
#include "hybrid_monitor.c"
int main(void) {
    int mode, nch, n, t, c, stop = 0;
    static double x[HYB_CHANNELS];
    while (scanf("%d %d %d", &mode, &nch, &n) == 3) {
        hyb_monitor_t m;
        hyb_init(&m);
        stop = 0;
        for (t = 0; t < n; t++) {
            for (c = 0; c < nch; c++) if (scanf("%lf", &x[c]) != 1) return 2;
            if (stop) continue;
            if (mode == 1) {
                /* Whole samples, reset after every alarm. */
                int ch = -1;
                hyb_verdict_t v = hyb_push_sample(&m, x, &ch);
                if (v != HYB_OK) {
                    printf("%d %d %d\n", t, ch, (int)v);
                    hyb_reset(&m);
                }
                continue;
            }
            /* Mode 0: one channel at a time, first alarm only. */
            for (c = 0; c < nch; c++) {
                hyb_verdict_t v = hyb_push(&m, c, x[c]);
                if (v != HYB_OK) {
                    printf("%d %d %d\n", t, c, (int)v);
                    stop = 1;
                    break;
                }
            }
        }
        printf("END\n");
    }
    return 0;
}
"#;

struct Case {
    label: String,
    stream: Vec<Vec<f64>>,
}

fn parse(out: &str) -> Vec<Vec<Alarm>> {
    let mut all = vec![Vec::new()];
    for line in out.lines() {
        if line == "END" {
            all.push(Vec::new());
            continue;
        }
        let f: Vec<usize> = line.split_whitespace().map(|x| x.parse().unwrap()).collect();
        all.last_mut().unwrap().push((f[0], f[1], f[2] as u8));
    }
    all.pop();
    all
}

fn run_c(cc: &str, dir: &std::path::Path, c_src: &str, cases: &[Case], mode: u8) -> Vec<Vec<Alarm>> {
    fs::write(dir.join("hybrid_monitor.c"), c_src).unwrap();
    fs::write(dir.join("driver.c"), DRIVER).unwrap();
    let mut input = String::new();
    for k in cases {
        let (nch, n) = (k.stream.len(), k.stream[0].len());
        writeln!(input, "{mode} {nch} {n}").unwrap();
        for t in 0..n {
            let row: Vec<String> = (0..nch).map(|c| format!("{:.17e}", k.stream[c][t])).collect();
            writeln!(input, "{}", row.join(" ")).unwrap();
        }
    }
    let exe = dir.join(if cfg!(windows) { "driver.exe" } else { "driver" });
    let out = Command::new(cc)
        .current_dir(dir)
        .args(["-std=c99", "-O2", "-o"])
        .arg(&exe)
        .arg("driver.c")
        .arg("-lm")
        .output()
        .expect("run C compiler");
    assert!(out.status.success(), "{cc} failed:\n{}", String::from_utf8_lossy(&out.stderr));
    fs::write(dir.join("input.txt"), input).unwrap();
    let run = Command::new(&exe)
        .stdin(fs::File::open(dir.join("input.txt")).unwrap())
        .output()
        .unwrap();
    assert!(run.status.success(), "driver failed");
    parse(&String::from_utf8_lossy(&run.stdout))
}

fn fmt(a: &[Alarm]) -> String {
    if a.is_empty() {
        return "none".into();
    }
    a.iter().map(|&(t, c, l)| format!("t{t}/ch{c}/{}", LEGS.get(l as usize).unwrap_or(&"?"))).collect::<Vec<_>>().join(" ")
}

#[test]
fn hybrid_c_alarms_match_rust() {
    let Some(cc) = common::c_compiler() else {
        return;
    };
    let seed = common::diff_seed("hybrid_c_alarms_match_rust", 0xA1A2_2026_0926);
    let calibrations: usize = std::env::var("STRUKTURA_ORACLE_CALIBS").ok().and_then(|v| v.parse().ok()).unwrap_or(40);
    let per_calib = 6;
    let mut rng = Rng(seed | 1);
    let dir = common::scratch("hybrid-alarms");

    let (mut n_first, mut n_seq, mut bad_first, mut bad_seq, mut neg_bad) = (0, 0, 0, 0, 0);
    let mut trend_neg_bad = 0usize;
    let mut certified_total = 0usize;
    let mut long_case_certified = false;
    let mut examples: Vec<String> = Vec::new();
    let mut kinds = std::collections::BTreeMap::<String, usize>::new();
    let mut first_kinds = std::collections::BTreeMap::<&str, usize>::new();
    let mut long_done = false;

    for k in 0..calibrations {
        let nch = 1 + (rng.next_u64() % 3) as usize;
        let procs: Vec<Proc> = (0..nch).map(|_| draw_proc(&mut rng)).collect();
        let calib_len = 8 * WINDOW;
        let clean: Vec<Vec<f64>> = procs.iter().map(|&p| series(&mut rng, p, calib_len, 0)).collect();
        let Some(mut mon) = HybridMonitor::calibrate(&clean) else {
            continue;
        };
        mon.set_leg_enabled(Leg::Parity, false);
        mon.set_leg_enabled(Leg::Missingness, false);
        let this_run_certified = (0..nch).filter(|&ch| mon.trend(ch).is_some()).count();
        certified_total += this_run_certified;
        let c_src = generate_hybrid_c(&mon.export());

        let mut cases = Vec::new();
        for case_idx in 0..per_calib {
            // One long stream over the whole run takes the C ring phase
            // counter past its wrap (HYB_PHASE = WINDOW * ROLL * DFA_STRIDE
            // = 18432 samples). Deferred to the first calibration that
            // actually certified a channel (rather than unconditionally the
            // very first calibration), so the long stream is guaranteed —
            // not left to chance — to exercise a certified channel.
            let use_long = !long_done && this_run_certified > 0;
            let n = if use_long { 40_000 } else { 1500 };
            if use_long {
                long_done = true;
                long_case_certified = true;
            }
            let _ = case_idx;
            // Continuations: a kind-4 (TS ramp) channel's stream picks up
            // its ramp exactly where the calibration left off.
            let mut stream: Vec<Vec<f64>> =
                procs.iter().map(|&p| series(&mut rng, p, n, calib_len)).collect();
            let ch = (rng.next_u64() % nch as u64) as usize;
            let at = n - 1200 + (rng.next_u64() % 900) as usize;
            let fault = inject(&mut rng, &mut stream, ch, at, procs[ch].scale);
            *kinds.entry(fault.to_string()).or_default() += 1;
            cases.push(Case { label: format!("calib {k} nch {nch} ch {ch} {fault}@{at}"), stream });
        }

        // Rust needs a fresh monitor per stream: clone by recalibrating from the same data.
        let fresh = || {
            let mut m = HybridMonitor::calibrate(&clean).unwrap();
            m.set_leg_enabled(Leg::Parity, false);
            m.set_leg_enabled(Leg::Missingness, false);
            m
        };
        let c_first = run_c(&cc, &dir, &c_src, &cases, 0);
        let c_seq = run_c(&cc, &dir, &c_src, &cases, 1);
        let neg_src = c_src.replacen("#define HYB_RES_THR    ", "#define HYB_RES_THR    0.9 * ", 1);
        assert_ne!(neg_src, c_src, "negative control did not change the C");
        let c_neg = run_c(&cc, &dir, &neg_src, &cases, 1);
        // Second negative control, on the NEW trend path: weaken the level
        // leg's expectation line so a certified channel's level leg must
        // mismatch. Only bites on a run with a certified channel (a run
        // with none leaves c_src unchanged from a trendless run, so the
        // replacen is a no-op there — accounted for below).
        let trend_neg_src = c_src.replacen("e = cc->tr_mu + p;", "e = cc->tr_mu + 0.9 * p;", 1);
        let trend_neg_changed = trend_neg_src != c_src;
        let c_trend_neg = if trend_neg_changed {
            Some(run_c(&cc, &dir, &trend_neg_src, &cases, 1))
        } else {
            None
        };
        assert_eq!(c_first.len(), cases.len());

        for (i, case) in cases.iter().enumerate() {
            let r_first = rust_alarms(&mut fresh(), &case.stream, false);
            let r_seq = rust_alarms(&mut fresh(), &case.stream, true);
            n_first += 1;
            n_seq += 1;
            if r_first != c_first[i] {
                bad_first += 1;
                let what = match (r_first.first(), c_first[i].first()) {
                    (Some(r), Some(c)) if r.0 != c.0 && r.2 != c.2 => "tick+leg",
                    (Some(r), Some(c)) if r.0 != c.0 => "tick",
                    (Some(r), Some(c)) if r.2 != c.2 => "leg",
                    (Some(_), Some(_)) => "channel",
                    (Some(_), None) => "rust-only",
                    _ => "c-only",
                };
                *first_kinds.entry(what).or_default() += 1;
                if examples.len() < 12 {
                    examples.push(format!("FIRST {}: rust {} | c {}", case.label, fmt(&r_first), fmt(&c_first[i])));
                }
            }
            if r_seq != c_seq[i] {
                bad_seq += 1;
                if examples.len() < 24 {
                    examples.push(format!("SEQ   {}: rust {} | c {}", case.label, fmt(&r_seq), fmt(&c_seq[i])));
                }
            }
            if r_seq != c_neg[i] {
                neg_bad += 1;
            }
            if let Some(c_trend_neg) = &c_trend_neg {
                if r_seq != c_trend_neg[i] {
                    trend_neg_bad += 1;
                }
            }
        }
    }
    let _ = fs::remove_dir_all(&dir);
    for e in &examples {
        println!("{e}");
    }
    println!("faults: {kinds:?}");
    println!("first-alarm mismatch kinds: {first_kinds:?}");
    println!(
        "hybrid_c_alarms_match_rust: compared {n_first} streams; first-alarm mismatches {bad_first}; \
         full-sequence mismatches {bad_seq}/{n_seq}; negative control (C residual threshold x0.9) mismatches {neg_bad}/{n_seq}; \
         trend-path negative control (C level 0.9x) mismatches {trend_neg_bad}; certified channels {certified_total} \
         (>=1 in the 40,000-sample run: {long_case_certified})"
    );
    assert!(neg_bad > 0, "negative control produced no mismatches: the comparison is not sensitive");
    assert!(
        trend_neg_bad > 0,
        "trend-path negative control produced no mismatches: the certified-trend C path is not exercised/sensitive"
    );
    assert!(
        certified_total >= 20,
        "certified-channel count {certified_total} < 20: the oracle would pass vacuously on the trend path"
    );
    assert!(
        long_case_certified,
        "the 40,000-sample stream's calibration must carry at least one certified channel"
    );
    assert_eq!(bad_first, 0, "first alarm differs on {bad_first} streams");
    assert_eq!(bad_seq, 0, "alarm sequence differs on {bad_seq} streams");
}
