//! A sensor that comes back while parity is suspended (README, Limitations).
//!
//! Three channels on a shared AR(0.9) component, each with its own AR(0.5)
//! noise. ch2 sticks for good at step 3000; ch1 sticks on [3000, 6000) and
//! comes back with a constant +5 offset (about 2 standard deviations, near
//! its level-leg threshold). With ch2 quarantined, parity cannot check ch1,
//! and the recovery residual compares each reading with the sensor's own
//! previous one, so the offset passes it: ch1 is released, its level leg
//! fires, and an accepted adaptation makes the offset the baseline. The
//! control is ch1 coming back without the offset. This is scenario s8 of the
//! oura-26 session's recovery harness.
//!
//! The obvious rule ("release only once the level is back in the calibrated
//! band while parity is suspended", branch fix/recovery-level) stopped every
//! adaptation here, and was rejected on real data: on ESA-ADB Mission 1,
//! outages end at a genuinely new level, and the rule blocked the
//! recalibration that absorbs it.
//!
//! `cargo run --release --example recovery_offset_eval`

use struktura::autopilot::{AutoPilot, Event};
use struktura::monitor::HybridMonitor;
use struktura::telemetry_bench::GaussRng;

fn shared_ar(n: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = GaussRng::new(seed);
    let (mut s, mut e) = (0.0f64, [0.0f64; 3]);
    let mut out: Vec<Vec<f64>> = (0..3).map(|_| Vec::with_capacity(n)).collect();
    for _ in 0..n {
        s = 0.9 * s + rng.normal(0.0, 1.0);
        for (c, ec) in e.iter_mut().enumerate() {
            *ec = 0.5 * *ec + rng.normal(0.0, 0.3);
            out[c].push(10.0 + c as f64 + s + *ec);
        }
    }
    out
}

/// (ch1 released, adaptation accepted after ch1 came back)
fn run(seed: u64, offset: f64) -> (bool, bool) {
    let calib = shared_ar(3000, seed * 100);
    let stream = shared_ar(9000, seed * 100 + 1);
    let mut ap = AutoPilot::new(HybridMonitor::calibrate(&calib).expect("calibration"));
    let (mut released, mut adapted) = (false, false);
    for t in 0..9000 {
        let mut sample = [stream[0][t], stream[1][t], stream[2][t]];
        if t >= 3000 {
            sample[2] = stream[2][2999];
        }
        if (3000..6000).contains(&t) {
            sample[1] = stream[1][2999];
        } else if t >= 6000 {
            sample[1] += offset;
        }
        for ev in ap.push(&sample, &[true; 3]) {
            match ev {
                Event::Unquarantined { channel: 1, .. } => released = true,
                Event::Recalibrated { tick } if tick >= 6000 => adapted = true,
                _ => {}
            }
        }
    }
    (released, adapted)
}

fn main() {
    let seeds = 10u64;
    for (label, offset) in [("back +5", 5.0), ("back healthy (control)", 0.0)] {
        let runs: Vec<(bool, bool)> = (1..=seeds).map(|s| run(s, offset)).collect();
        let released = runs.iter().filter(|r| r.0).count();
        let adapted = runs.iter().filter(|r| r.1).count();
        println!("ch1 {label}: released {released}/{seeds}, adaptation accepted afterwards {adapted}/{seeds}");
    }
}
