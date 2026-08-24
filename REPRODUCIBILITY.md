# Reproducibility index

Every measured claim in this repository maps to one command. All commands
run from the repository root; the binary builds with
`cargo build --release --bin struktura` (binary at `target/release/struktura`,
or `C:\Oura\target\release\struktura.exe` with a shared target dir).
All benchmarks use deterministic seeds — outputs reproduce exactly.

| Claim | Command | Verified output (2026-08-24) |
|---|---|---|
| DFA detects correlation-structure faults at 100% (6.9x null), blind to additive faults, on AR(1) signals (20 seeds, FPR ≤ 5%) | `struktura benchmark-faults` | structural 100%, additive at chance |
| Whole-sequence DFA on the coupled spacecraft sim: drift/regime_shift/mixed 100% (200 disjoint seeds), measured family-wise FPR 2.5%, Bonferroni thresholds | `struktura benchmark-telemetry` | FPR 2.5%; three 100% rows |
| Structural-fault resolution law: 84→35%, 140→42%, 210→74%, 294→88% | `struktura benchmark-telemetry` | resolution curve table |
| Per-timestep F1 low (DFA integrates, doesn't localize) + NAB event scoring with latencies | `struktura benchmark-telemetry` | F1/event table |
| 4-detector hybrid (bench variant): 6/7 fault types ≥80% event detection, each fault caught by a different leg | `struktura benchmark-telemetry` | taxonomy 6/7 line |
| Streaming HybridMonitor: 7/7 fault taxonomy at 100% each (missingness leg included), 0 false alarms on 200K clean samples | `struktura monitor-perf` | fault table + alarms line |
| Fault-class identification 97.1% from alarm provenance | `struktura monitor-perf` | class-ID table |
| Per-sample cost: P50 ~0.7µs / P99 ~7µs / P99.9 ~11µs (host-dependent) | `struktura monitor-perf` | timing lines |
| IMS bearing run-to-failure: structural warning ~2h before failure (DFA: α spikes rec 970→984). Full HybridMonitor on raw IMS data may alarm earlier — requires dataset in data/ims/ | `struktura ims` (embedded) or `struktura monitor-real` (full data) | IMS section |
| Prognosis: no-trend at 50% of life (correct), −66 recs at 75%, +91 at 90% (healing plateau) | `struktura monitor-real` | prognosis table |
| Voyager heliopause: honest no-alarm at streaming resolution (with trend-safe legs); whole-segment analysis detects it | `struktura monitor-real` + `struktura heliopause` | Voyager sections |
| Generated C99 flight monitor compiles `-Wall -Werror` clean; embedded self-test detects stuck at t=405 | `struktura generate-hybrid -o hybrid_monitor.c` then `gcc -std=c99 -Wall -Werror -O2 -DHYBRID_STANDALONE_TEST -o hybrid hybrid_monitor.c -lm && ./hybrid` | SELFTEST PASS |
| Prefix-sum DFA ≡ reference DFA to 1e-9 over 1000 windows | `cargo test --release dfa_fast_matches` | test passes |
| Analytic α-scatter lower bound holds; measured inflation 1.3–4x | `cargo test --release analytic_alpha` | test passes |
| Nothing panics on NaN/Inf/empty/adversarial input (proptest) | `cargo test --release --test property_tests` | 11 tests pass |
| no_std library build | `cargo rustc --release --lib --no-default-features --crate-type rlib` | EXIT 0 |
| Full test suite | `cargo test --release` | 84 tests pass |

Honesty conventions used throughout: measured false-alarm rates are printed
next to every detection rate; "no resolvable trend" and "no alarm" are
first-class outputs; every statistical threshold's assumptions are stated in
`src/monitor.rs` module docs.
