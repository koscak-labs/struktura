//! Synthetic heavy-tailed "step time" series for trying `struktura ab`:
//! `base * (1 + 0.15 * (Pareto(2.2) - 1))`, and in 3% of steps a stall of
//! 2-6x. Deterministic for a given seed.
//!
//!     cargo run --example ab_demo_data -- <n> <base> <seed> > out.csv
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: ab_demo_data <n> <base> <seed>");
        std::process::exit(1);
    }
    let n: usize = args[1].parse().expect("n");
    let base: f64 = args[2].parse().expect("base");
    let mut s: u64 = args[3].parse().expect("seed");
    let mut next = || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (((z ^ (z >> 31)) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    println!("step,step_ms");
    for i in 0..n {
        let u = next();
        let mut x = base * (1.0 + 0.15 * (u.powf(-1.0 / 2.2) - 1.0));
        if next() < 0.03 {
            x *= 2.0 + 4.0 * next();
        }
        println!("{},{:.3}", i, x);
    }
}
