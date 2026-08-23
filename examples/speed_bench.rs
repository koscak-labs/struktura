use std::time::Instant;
use struktura::dfa;

fn lcg_noise(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..n).map(|_| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as f64 / (1u64 << 31) as f64 - 0.5
    }).collect()
}

fn main() {
    println!("STRUKTURA SPEED BENCHMARK");
    println!("========================\n");

    for &n in &[256, 1024, 4096, 16384, 65536] {
        let data = lcg_noise(n, 42);
        let warmup = Instant::now();
        let _ = dfa(&data);
        let _ = warmup.elapsed();

        let iters = if n <= 4096 { 100 } else { 10 };
        let start = Instant::now();
        for _ in 0..iters {
            let _ = dfa(&data);
        }
        let elapsed = start.elapsed();
        let per_call = elapsed.as_secs_f64() / iters as f64 * 1000.0;

        println!("  N={:>6}  {:.3} ms/call  ({} iters)", n, per_call, iters);
    }

    println!("\nCompare with Python nolds.dfa():");
    println!("  N=4096   ~15-25 ms/call (typical)");
    println!("  N=16384  ~60-100 ms/call (typical)");
    println!("  N=65536  ~250-400 ms/call (typical)");
}
