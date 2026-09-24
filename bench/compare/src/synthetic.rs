//! Synthetic suite: reuses the stream generators from
//! ../../examples/structure_vs_amplitude.rs (same seeds, same change point,
//! same calibration length) across all 8 detectors here.

use crate::detectors::{self, DetResult};

const N: usize = 2000;
const CHANGE: usize = 1000;
const CALIB: usize = 768;
const SEEDS: u64 = 30;

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
            let tail = gen(&mut rng, k, N - CHANGE, 0.0);
            s.extend(tail);
            s
        }
    }
}

const DET_NAMES: [&str; 8] = [
    "struktura guard (default)",
    "struktura guard (quiet_drift)",
    "augurs-changepoint BOCPD",
    "ankane STL (anomaly_detection)",
    "extended-isolation-forest",
    "limit check (baseline)",
    "EWMA chart (baseline)",
    "CUSUM (baseline)",
];

fn run_detector(idx: usize, calib: &[f64], rest: &[f64]) -> DetResult {
    match idx {
        0 => detectors::struktura_guard(calib, rest, false)
            .unwrap_or(DetResult { alarms: vec![], per_sample_ns: 0.0, streaming: true }),
        1 => detectors::struktura_guard(calib, rest, true)
            .unwrap_or(DetResult { alarms: vec![], per_sample_ns: 0.0, streaming: true }),
        2 => detectors::augurs_bocpd(calib, rest),
        3 => detectors::ankane_stl(rest, 64),
        4 => detectors::isolation_forest(calib, rest),
        5 => detectors::limit_check(calib, rest),
        6 => detectors::ewma(calib, rest),
        _ => detectors::cusum(calib, rest),
    }
}

fn median_usize(v: &mut [usize]) -> String {
    if v.is_empty() {
        return "-".into();
    }
    v.sort_unstable();
    v[v.len() / 2].to_string()
}

pub fn run() {
    let clean = [("white", Kind::White), ("AR(0.7)", Kind::Ar(0.7)), ("AR(0.95)", Kind::Ar(0.95))];
    let shifts = [
        ("white -> brown (random walk)", Kind::Brown),
        ("white -> AR(0.9), std x2.3 (amplitude growth)", Kind::Ar(0.9)),
        ("white -> AR(0.9), variance-matched (correlation change)", Kind::ArMatched(0.9)),
    ];

    println!("## Synthetic suite ({SEEDS} seeds, calibration on first {CALIB} samples)\n");
    println!("### Clean streams: false alarms out of {SEEDS}\n");
    println!(
        "| stream | {} |",
        DET_NAMES.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(" | ")
    );
    println!("|---|{}|", "---|".repeat(DET_NAMES.len()));
    for (name, kind) in clean {
        let mut cells = Vec::new();
        for idx in 0..8 {
            let fa = (0..SEEDS)
                .filter(|&s| {
                    let x = stream(s, kind, None);
                    !run_detector(idx, &x[..CALIB], &x[CALIB..]).alarms.is_empty()
                })
                .count();
            cells.push(format!("{fa}/{SEEDS}"));
        }
        println!("| {name} | {} |", cells.join(" | "));
    }

    println!("\n### Shifts at sample {CHANGE}: detected / early-false-alarm / median delay\n");
    println!(
        "| shift | {} |",
        DET_NAMES.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(" | ")
    );
    println!("|---|{}|", "---|".repeat(DET_NAMES.len()));
    for (name, kind) in shifts {
        let mut cells = Vec::new();
        for idx in 0..8 {
            let (mut hit, mut early, mut delays) = (0usize, 0usize, Vec::new());
            for s in 0..SEEDS {
                let x = stream(1000 + s, Kind::White, Some(kind));
                let res = run_detector(idx, &x[..CALIB], &x[CALIB..]);
                match res.alarms.first() {
                    Some(&t0) => {
                        let t = t0 + CALIB;
                        if t < CHANGE {
                            early += 1;
                        } else {
                            hit += 1;
                            delays.push(t - CHANGE);
                        }
                    }
                    None => {}
                }
            }
            cells.push(format!("{hit}/{SEEDS}, {early} early, median {}", median_usize(&mut delays)));
        }
        println!("| {name} | {} |", cells.join(" | "));
    }
}
