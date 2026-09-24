//! Structure vs amplitude: when does DFA see a change that a limit check
//! misses, and when is the limit check simply better?
//!
//! 30 seeds, 2000 samples, change at sample 1000, calibration on the first
//! 768 samples (set `CALIB` to change it). With 512 calibration samples the
//! full monitor's LevelShift leg raises 3-6/30 false alarms on the clean
//! streams; from 768 on it raises none. Three detectors see the same streams:
//!
//! - `dfa`: struktura's hybrid monitor with only the DFA leg enabled
//! - `hybrid`: the full monitor (all legs), what `struktura guard` runs
//! - `amp`: |x - median| > 1.5 x p95 of the calibration deviations,
//!   confirmed after 3 consecutive exceedances
//!
//! Clean streams count any alarm as a false alarm. Shift streams count an
//! alarm before sample 1000 as a false alarm and report the median delay
//! of alarms after it.
//!
//! Run: cargo run --release --example structure_vs_amplitude

use struktura::monitor::{HybridMonitor, Leg, MonitorConfig};

const N: usize = 2000;
const CHANGE: usize = 1000;
fn calib_len() -> usize {
    std::env::var("CALIB").ok().and_then(|v| v.parse().ok()).unwrap_or(768)
}
/// HORIZON=<samples> overrides the threshold design horizon (default 1e6).
fn calibrate(clean: &[f64]) -> Option<HybridMonitor> {
    let mut config = MonitorConfig::default();
    if let Some(h) = std::env::var("HORIZON").ok().and_then(|v| v.parse::<f64>().ok()) {
        config.design_horizon = h;
    }
    HybridMonitor::calibrate_with(&[clean.to_vec()], config)
}
const SEEDS: u64 = 30;

/// xorshift64* with Box-Muller: small, deterministic, no dependencies.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let x = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((x >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * core::f64::consts::PI * v).cos()
    }
}

#[derive(Clone, Copy)]
enum Kind {
    White,
    Ar(f64),
    ArMatched(f64),
    Brown,
}

/// Generate `len` samples of `kind`, continuing from the previous value
/// `x0` so the stream has no jump at the change point.
fn gen(rng: &mut Rng, kind: Kind, len: usize, x0: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(len);
    let mut x = x0;
    for _ in 0..len {
        let e = rng.normal();
        x = match kind {
            Kind::White => e,
            Kind::Ar(phi) => phi * x + e,
            Kind::ArMatched(phi) => phi * x + e * (1.0 - phi * phi).sqrt(),
            Kind::Brown => x + e,
        };
        out.push(x);
    }
    out
}

fn stream(seed: u64, before: Kind, after: Option<Kind>) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    match after {
        None => gen(&mut rng, before, N, 0.0),
        Some(k) => {
            let mut s = gen(&mut rng, before, CHANGE, 0.0);
            // White noise carries no state; start the new process from 0.
            let tail = gen(&mut rng, k, N - CHANGE, 0.0);
            s.extend(tail);
            s
        }
    }
}

/// First alarm tick (index into the full stream), or None.
fn first_monitor_alarm(x: &[f64], dfa_only: bool) -> Option<usize> {
    let mut m = calibrate(&x[..calib_len()])?;
    if dfa_only {
        for leg in [
            Leg::Residual,
            Leg::RepeatedValue,
            Leg::LevelShift,
            Leg::ResidualCusum,
            Leg::Missingness,
            Leg::Parity,
        ] {
            m.set_leg_enabled(leg, false);
        }
    }
    (calib_len()..x.len()).find(|&i| m.push(&[x[i]]).is_some())
}

/// Which leg the full monitor alarms on first, if any.
fn first_leg(x: &[f64]) -> Option<Leg> {
    let mut m = calibrate(&x[..calib_len()])?;
    (calib_len()..x.len()).find_map(|i| m.push(&[x[i]]))
}

fn leg_counts(streams: impl Iterator<Item = Vec<f64>>) -> String {
    let mut counts: Vec<(String, usize)> = Vec::new();
    for s in streams {
        if let Some(leg) = first_leg(&s) {
            let k = format!("{leg:?}");
            match counts.iter_mut().find(|(n, _)| *n == k) {
                Some((_, c)) => *c += 1,
                None => counts.push((k, 1)),
            }
        }
    }
    if counts.is_empty() {
        return "none".into();
    }
    counts.iter().map(|(n, c)| format!("{n} {c}")).collect::<Vec<_>>().join(", ")
}

fn first_amp_alarm(x: &[f64]) -> Option<usize> {
    let mut calib: Vec<f64> = x[..calib_len()].to_vec();
    calib.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = calib[calib_len() / 2];
    let mut dev: Vec<f64> = x[..calib_len()].iter().map(|v| (v - med).abs()).collect();
    dev.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let thr = 1.5 * dev[(0.95 * calib_len() as f64) as usize];
    let mut run = 0;
    for (i, v) in x.iter().enumerate().skip(calib_len()) {
        run = if (v - med).abs() > thr { run + 1 } else { 0 };
        if run >= 3 {
            return Some(i);
        }
    }
    None
}

fn median(v: &mut [usize]) -> String {
    if v.is_empty() {
        return "-".into();
    }
    v.sort_unstable();
    v[v.len() / 2].to_string()
}

fn main() {
    let clean = [
        ("white", Kind::White),
        ("AR(0.7)", Kind::Ar(0.7)),
        ("AR(0.95)", Kind::Ar(0.95)),
    ];
    let shifts = [
        ("white -> brown", Kind::Brown),
        ("white -> AR(0.9), std x2.3", Kind::Ar(0.9)),
        ("white -> AR(0.9), variance-matched", Kind::ArMatched(0.9)),
    ];
    type Det = (&'static str, fn(&[f64]) -> Option<usize>);
    let dets: [Det; 3] = [
        ("dfa", |x| first_monitor_alarm(x, true)),
        ("hybrid", |x| first_monitor_alarm(x, false)),
        ("amp", first_amp_alarm),
    ];

    println!("clean streams: false alarms out of {SEEDS} (after calibration on {} samples)", calib_len());
    println!("| stream | dfa | hybrid | amp |");
    println!("|---|---|---|---|");
    for (name, kind) in clean {
        let counts: Vec<String> = dets
            .iter()
            .map(|(_, d)| {
                let fa = (0..SEEDS).filter(|&s| d(&stream(s, kind, None)).is_some()).count();
                format!("{fa}/{SEEDS}")
            })
            .collect();
        println!("| {name} | {} | {} | {} |", counts[0], counts[1], counts[2]);
    }

    println!();
    println!("shifts at sample {CHANGE}: detected / early false alarm / median delay");
    println!("| shift | dfa | hybrid | amp |");
    println!("|---|---|---|---|");
    for (name, kind) in shifts {
        let cells: Vec<String> = dets
            .iter()
            .map(|(_, d)| {
                let (mut hit, mut early, mut delays) = (0, 0, Vec::new());
                for s in 0..SEEDS {
                    match d(&stream(1000 + s, Kind::White, Some(kind))) {
                        Some(t) if t < CHANGE => early += 1,
                        Some(t) => {
                            hit += 1;
                            delays.push(t - CHANGE);
                        }
                        None => {}
                    }
                }
                format!("{hit}/{SEEDS}, {early} early, median {}", median(&mut delays))
            })
            .collect();
        println!("| {name} | {} | {} | {} |", cells[0], cells[1], cells[2]);
    }

    println!();
    println!("hybrid: which leg fires first");
    for (name, kind) in clean {
        println!("  clean {name}: {}", leg_counts((0..SEEDS).map(|s| stream(s, kind, None))));
    }
    for (name, kind) in shifts {
        println!(
            "  {name}: {}",
            leg_counts((0..SEEDS).map(|s| stream(1000 + s, Kind::White, Some(kind))))
        );
    }
}
