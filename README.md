# Struktura

[![Crates.io](https://img.shields.io/crates/v/struktura.svg)](https://crates.io/crates/struktura)
[![CI](https://github.com/koscak-labs/struktura/actions/workflows/ci.yml/badge.svg)](https://github.com/koscak-labs/struktura/actions/workflows/ci.yml)
[![License](https://img.shields.io/crates/l/struktura.svg)](https://github.com/koscak-labs/struktura)
[![docs.rs](https://docs.rs/struktura/badge.svg)](https://docs.rs/struktura)
[![no_std](https://img.shields.io/badge/no__std-compatible-green.svg)](https://docs.rs/struktura)

**Predict failure before it happens.** 85-112x faster than Python. Runs on embedded flight computers (`no_std`).

Universal anomaly detection via DFA (Detrended Fluctuation Analysis). No training, no hyperparameters, no ML. One number tells you if the structure of a signal changed — before averages or thresholds notice.

```
cargo install struktura
struktura demo              # bearing fault detection (real CWRU data)
struktura voyager           # Voyager 1 AACS anomaly (real NASA data)
struktura spacecraft        # multi-channel spacecraft health monitor
struktura text novel.txt    # writing rhythm analysis
```

## 10-second demo

```
$ struktura voyager

  VOYAGER 1 STRUCTURAL HEALTH ANALYSIS
  ====================================================================

  2021 (healthy)    alpha=0.989  R²=0.9869
  2022 (anomaly)    alpha=0.801  R²=0.9681
                    shift=-0.187  CRITICAL

  The magnetic field's fractal structure changed during the anomaly —
  detectable from public NASA data, zero training, zero ML.
```

Real data. Real spacecraft failure. Detected by one Rust function.

## Speed (benchmarked, not estimated)

| Signal size | Struktura (Rust) | Python nolds | Speedup |
|---|---|---|---|
| 4,096 points | **0.24 ms** | ~20 ms | **85x** |
| 16,384 points | **0.93 ms** | ~80 ms | **86x** |
| 65,536 points | **2.89 ms** | ~325 ms | **112x** |

At 1Hz spacecraft telemetry: 0.24ms per analysis = **4,000 channels on one core**.

Reproduce: `cargo run --release --example speed_bench`

## As a library

```rust
use struktura::{analyze, health_check, HealthVerdict};

let law = analyze(&sensor_data);
let verdict = health_check(&law, baseline_alpha);
// Healthy | Watch | Warning | Critical
```

Real-time streaming monitor:

```rust
use struktura::BaselineTracker;

let mut monitor = BaselineTracker::new(256, 1000);
for sample in telemetry_stream {
    if let Some(verdict) = monitor.push(sample) {
        match verdict {
            HealthVerdict::Critical => trigger_alert(),
            _ => {}
        }
    }
}
```

Spacecraft-specific:

```rust
use struktura::space::{SpacecraftMonitor, Subsystem};

let mut rwa = SpacecraftMonitor::new(Subsystem::ReactionWheel, "RWA_current");
// push samples, get verdicts
```

## What it detects (all verified)

| Domain | Signal | Healthy α | Fault α | Shift | Verdict |
|--------|--------|-----------|---------|-------|---------|
| **Bearings** | CWRU 12kHz vibration | 0.738 | 0.217 | -0.522 | CRITICAL |
| **Spacecraft** | Voyager 1 magnetometer | 0.989 | 0.801 | -0.187 | CRITICAL |
| **Text** | Austen vs shuffled | 0.749 | 0.572 | -0.177 | — |
| **Genome** | Human chr1 GC% | 0.909 | — | — | R²=0.991 |
| **Cardiac** | HRV RR intervals | 0.695 | — | — | R²=0.985 |

Every number from an actual run. Reproduce with `struktura demo` / `struktura voyager`.

## Text structure analysis

DFA measures the fractal rhythm of writing — sentence-length sequences have long-range correlations in human prose that disappear when shuffled.

```
$ struktura text pride_and_prejudice.txt shuffled.txt mechanical.txt

  Jane Austen (original)    α=0.749  STRONG RHYTHM (human literary)
  Austen (shuffled)         α=0.572  MODERATE RHYTHM
  Mechanical uniform        α=0.525  UNIFORM/MECHANICAL
```

The shuffle control proves it: same sentence-length distribution, different ordering. α drops from 0.749 to 0.572 — DFA measures sequential structure, not statistics.

## Spacecraft health monitoring

Built-in support for spacecraft subsystems — reaction wheels, magnetometers, batteries, thermal sensors, solar arrays, gyroscopes.

```
$ struktura spacecraft

  [RWA:RWA_current]     alpha=0.902 baseline=1.333 shift=-0.431  CRITICAL
  [BAT:BAT_voltage]     alpha=1.994 baseline=1.978 shift=+0.017  HEALTHY
  [THM:THM_panel_A]     alpha=1.981 baseline=1.960 shift=+0.021  HEALTHY
  [MAG:MAG_B_total]     alpha=0.985 baseline=0.914 shift=+0.070  WATCH
```

`no_std` compatible — runs on embedded flight computers. Zero heap allocation on the hot path via `dfa_into()`.

## Flight software code generation

Generate complete monitoring apps for NASA flight frameworks:

```
struktura generate --cfs    --db channels.json -o dfa_cfs_app/
struktura generate --fprime --db channels.json -o dfa_fprime_component/
struktura generate --ros    --db channels.json -o dfa_ros_node/
```

Compatible with [nasa/ogma](https://github.com/nasa/ogma)'s variable database format.

## How DFA works

DFA (Peng et al., Physical Review E, 1994 — 3000+ citations) measures long-range correlation:

1. Compute the cumulative profile (running sum minus mean)
2. Divide into boxes, detrend each with a linear fit
3. Measure residual fluctuation vs box size
4. Slope in log-log space = α (the scaling exponent)

| α | Meaning |
|---|---------|
| ~0.5 | Random noise (no structure) |
| 0.5–1.0 | Persistent correlations (healthy structure) |
| α shifts from baseline | Structural degradation |

The crate reports R² alongside every α. If R² < 0.3, quality = `Abstain`. It never bluffs.

## Features

- **Adaptive box sizes** — geometric spacing tuned to signal length (not fixed)
- **`no_std`** — `default-features = false` for embedded targets
- **C FFI** — `struktura.h` header for C/C++ integration
- **serde** — optional serialization (`features = ["serde"]`)
- **Self-test** — `struktura self-test` verifies all claims

## Alternatives

| Crate | DFA | Speed vs Python | License | Dependencies | `no_std` |
|-------|-----|----------------|---------|-------------|---------|
| **struktura** | native | **85-112x** | MIT/Apache | **1** (libm) | **yes** |
| anomaly_detection | no | — | GPL-3.0 | many | no |
| extended-isolation-forest | no | — | MIT | many | no |

## References

1. C.-K. Peng et al., "Mosaic organization of DNA nucleotide sequences," Physical Review E 49(2), 1994.
2. C.-K. Peng et al., "Quantification of scaling exponents," Chaos 5(1), 1995.
3. CWRU Bearing Data Center: https://engineering.case.edu/bearingdatacenter
4. NASA SPDF Voyager Data: https://spdf.gsfc.nasa.gov/pub/data/voyager/

## License

MIT OR Apache-2.0
