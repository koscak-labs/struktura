//! Head-to-head comparison of struktura against other Rust time-series
//! anomaly/changepoint detectors, on identical NAB data + a synthetic suite,
//! with identical scoring (reused from ../../examples/nab_eval.rs and
//! ../../examples/structure_vs_amplitude.rs).
//!
//! Run: NAB_DIR=C:\Projects\_nab cargo run --release
//!
//! Standalone crate: this is bench/compare/Cargo.toml with its own
//! [workspace] table, depending on struktura via path = "../..". It is not
//! part of the struktura package (cargo excludes subdirs with their own
//! Cargo.toml automatically).

mod detectors;
mod nab;
mod synthetic;

use std::env;
use std::time::Instant;

fn main() {
    let start = Instant::now();
    println!("# struktura head-to-head comparison\n");
    println!("Date: 2026-09-24");
    println!("Machine: Intel(R) Core(TM) Ultra 9 185H (Windows 11), rustc 1.95.0");
    println!();

    let nab_dir = env::var("NAB_DIR").unwrap_or_else(|_| "C:\\Projects\\_nab".to_string());
    nab::run(&nab_dir);
    println!();
    synthetic::run();

    println!("\nwall-clock runtime: {:.2}s", start.elapsed().as_secs_f64());
}
