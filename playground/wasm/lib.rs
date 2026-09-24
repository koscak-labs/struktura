//! What the playground page calls. `guard` runs the same AutoPilot as
//! `struktura guard`; `limit_check` and `demo_stream` match
//! examples/structure_vs_amplitude.rs, so the page shows what CI checks.

use struktura::autopilot::{AutoPilot, Event};
use struktura::monitor::{explain_alarm, HybridMonitor};
use wasm_bindgen::prelude::*;

fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Run guard on one channel. Returns JSON:
/// `{"ok":true,"calib":N,"events":[{"t":..,"kind":"alarm","leg":"..","text":".."}]}`
/// or `{"ok":false,"error":".."}`.
#[wasm_bindgen]
pub fn guard(values: &[f64], calib: usize) -> String {
    if calib < 192 || calib >= values.len() {
        return format!(
            "{{\"ok\":false,\"error\":\"need at least 192 calibration samples and some samples after them (got {} of {})\"}}",
            calib,
            values.len()
        );
    }
    let mon = match HybridMonitor::calibrate(&[values[..calib].to_vec()]) {
        Some(m) => m,
        None => return "{\"ok\":false,\"error\":\"calibration failed\"}".into(),
    };
    let mut ap = AutoPilot::new(mon);
    let mut events = Vec::new();
    for (t, &v) in values.iter().enumerate().skip(calib) {
        for ev in ap.push(&[v], &[true]) {
            let item = match ev {
                Event::Alarm { report, .. } => format!(
                    "{{\"t\":{},\"kind\":\"alarm\",\"leg\":\"{:?}\",\"ratio\":{:.2},\"text\":\"{}\"}}",
                    t,
                    report.leg,
                    report.observed / report.threshold,
                    esc(explain_alarm(&report))
                ),
                Event::AdaptationStarted { .. } => format!("{{\"t\":{},\"kind\":\"adapting\"}}", t),
                Event::Recalibrated { .. } => format!("{{\"t\":{},\"kind\":\"recalibrated\"}}", t),
                Event::RolledBack { .. } => format!("{{\"t\":{},\"kind\":\"confirmed\"}}", t),
                Event::Quarantined { .. } => format!("{{\"t\":{},\"kind\":\"quarantined\"}}", t),
            };
            events.push(item);
        }
    }
    format!("{{\"ok\":true,\"calib\":{},\"events\":[{}]}}", calib, events.join(","))
}

/// |x - median| > 1.5 x p95 of calibration deviations, 3 in a row.
/// Returns the first alarm index, or -1.
#[wasm_bindgen]
pub fn limit_check(values: &[f64], calib: usize) -> i32 {
    if calib == 0 || calib > values.len() {
        return -1;
    }
    let mut c: Vec<f64> = values[..calib].to_vec();
    c.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let med = c[calib / 2];
    let mut dev: Vec<f64> = values[..calib].iter().map(|v| (v - med).abs()).collect();
    dev.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let thr = 1.5 * dev[((0.95 * calib as f64) as usize).min(calib - 1)];
    let mut run = 0;
    for (i, v) in values.iter().enumerate().skip(calib) {
        run = if (v - med).abs() > thr { run + 1 } else { 0 };
        if run >= 3 {
            return i as i32;
        }
    }
    -1
}

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

/// Demo streams, 2000 samples, change at 1000 (same generator as the example):
/// "matched" white -> AR(0.9) variance-matched, "amp" white -> AR(0.9),
/// "walk" white -> random walk, "wander" healthy AR(0.95), "white" white noise.
#[wasm_bindgen]
pub fn demo_stream(kind: &str, seed: u32) -> Vec<f64> {
    let mut rng = Rng::new(seed as u64);
    let mut out = Vec::with_capacity(2000);
    let mut x = 0.0;
    for i in 0..2000 {
        let e = rng.normal();
        let after = i >= 1000;
        if after && i == 1000 {
            x = 0.0;
        }
        x = match (kind, after) {
            ("wander", _) => 0.95 * x + e,
            ("matched", true) => 0.9 * x + e * (1.0 - 0.81f64).sqrt(),
            ("amp", true) => 0.9 * x + e,
            ("walk", true) => x + e,
            _ => e,
        };
        out.push(x);
    }
    out
}
