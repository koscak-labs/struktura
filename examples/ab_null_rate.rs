//! How often does `ab_compare` call a direction when A and B come from the
//! same distribution (false calls), and how often does it detect a real +10%
//! shift (power), by sample size? Heavy-tailed synthetic data:
//! `70 * (1 + 0.15 * (Pareto(2.2) - 1))`, a dense floor and a long right tail.
//!
//!     cargo run --release --example ab_null_rate
use struktura::ab::{ab_compare, AbConfig, AbVerdict};

struct SplitMix64(u64);
impl SplitMix64 {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (((z ^ (z >> 31)) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
}

fn heavy(n: usize, base: f64, seed: u64) -> Vec<f64> {
    let mut rng = SplitMix64(seed);
    (0..n).map(|_| base * (1.0 + 0.15 * (rng.next_f64().powf(-1.0 / 2.2) - 1.0))).collect()
}

fn main() {
    let pairs = 1000u64;
    let cfg = AbConfig { resamples: 2000, ..AbConfig::default() };
    println!("{} pairs per row, {} resamples, median, min effect 1%", pairs, cfg.resamples);
    // "interval alone": the verdict rule without the MIN_N_VERDICT floor.
    println!("   n  same dist: called  interval alone  MW p<0.05 | B +10%: B HIGHER  B LOWER");
    for &n in &[3usize, 4, 5, 6, 8, 10, 15, 20, 30, 50, 100, 200] {
        let (mut null_called, mut null_raw, mut null_mw, mut up, mut down) = (0, 0, 0, 0, 0);
        for s in 0..pairs {
            let a = heavy(n, 70.0, 5000 + 2 * s);
            let b = heavy(n, 70.0, 5001 + 2 * s);
            let r = ab_compare(&a, &b, &cfg).unwrap();
            if matches!(r.verdict, AbVerdict::Higher | AbVerdict::Lower) {
                null_called += 1;
            }
            if (r.ci_low_pct > 0.0 || r.ci_high_pct < 0.0) && r.diff_pct.abs() >= cfg.min_effect_pct {
                null_raw += 1;
            }
            if r.mann_whitney.p_value < 0.05 {
                null_mw += 1;
            }
            let b = heavy(n, 77.0, 9001 + 2 * s);
            match ab_compare(&a, &b, &cfg).unwrap().verdict {
                AbVerdict::Higher => up += 1,
                AbVerdict::Lower => down += 1,
                _ => {}
            }
        }
        let pct = |k: u32| 100.0 * k as f64 / pairs as f64;
        println!(
            "{:4}  {:16.1}%  {:13.1}%  {:8.1}% | {:15.1}%  {:6.1}%",
            n, pct(null_called), pct(null_raw), pct(null_mw), pct(up), pct(down)
        );
    }
}
