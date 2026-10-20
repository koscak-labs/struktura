"""Apples-to-apples benchmark: same DFA algorithm, Rust core vs CPython `nolds`.

The README quotes a speed ratio against a *published* CPython reference, not a
head-to-head run — so this script exists to let you do the head-to-head
yourself. It is the *same* Detrended Fluctuation Analysis either way, just
running in the Rust core instead of the CPython interpreter.

Run:
    pip install nolds
    python examples/nolds_compare.py

Then compare the per-call ms below against `cargo run --release --example speed_bench`
(reference numbers from this machine: N=4096 -> 0.059 ms, N=65536 -> 0.78 ms).

Report the ratio you actually measure, and label it as measured on your
hardware. The struktura Python wheel (maturin, `python` feature) binds that
SAME Rust core, so `struktura.py_dfa` is the fair Rust-vs-Rust check with zero
interpreter overhead in the inner loop.
"""

import time

import numpy as np

try:
    import nolds
except ImportError:
    raise SystemExit("install nolds first: pip install nolds")


def make_signal(n, seed=7):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal(n))  # brownian -> high alpha


def bench(n, iters):
    x = make_signal(n)
    # warmup
    nolds.dfa(x)
    t0 = time.perf_counter()
    for _ in range(iters):
        nolds.dfa(x)
    dt = (time.perf_counter() - t0) / iters * 1000.0
    return dt


if __name__ == "__main__":
    print("nolds.dfa() — same algorithm, CPython interpreter")
    print("=" * 48)
    for n, iters in [(4096, 30), (16384, 10), (65536, 5)]:
        ms = bench(n, iters)
        print(f"  N={n:>6}  {ms:7.2f} ms/call")
    print()
    print("Compare with: cargo run --release --example speed_bench")
