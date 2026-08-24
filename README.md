<p align="center">
  <img src="assets/logo.png" alt="struktura" width="280">
</p>

<h3 align="center">predict failure before it happens 🔥</h3>
<p align="center">detected voyager 1's anomaly from public data. zero training. 98 lines. 85x faster than python.</p>

<p align="center">
  <a href="https://crates.io/crates/struktura"><img src="https://img.shields.io/crates/v/struktura.svg" alt="Crates.io"></a>
  <a href="https://github.com/koscak-labs/struktura/actions/workflows/ci.yml"><img src="https://github.com/koscak-labs/struktura/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/koscak-labs/struktura"><img src="https://img.shields.io/crates/l/struktura.svg" alt="License"></a>
  <a href="https://docs.rs/struktura"><img src="https://docs.rs/struktura/badge.svg" alt="docs.rs"></a>
  <a href="https://docs.rs/struktura"><img src="https://img.shields.io/badge/no__std-compatible-green.svg" alt="no_std"></a>
</p>

---

anomaly detection that actually works. not thresholds. not ML. not vibes.

DFA (Detrended Fluctuation Analysis) gives you one number that tells you if a signal's structure changed. works on literally anything with a time axis. spacecraft, bearings, financial data, heartbeats, DNA, text rhythm. you name it.

```
cargo install struktura
struktura demo              # 🔧 bearing fault detection (real CWRU data)
struktura voyager           # 🚀 Voyager 1 AACS anomaly (real NASA data)
struktura spacecraft        # 🛰️ multi-channel spacecraft health
struktura text novel.txt    # 📖 writing rhythm analysis
```

## 🚀 10 second demo

```
$ struktura voyager

  VOYAGER 1 STRUCTURAL HEALTH ANALYSIS
  ====================================================================

  2021 (healthy)    alpha=0.989  R²=0.9869
  2022 (anomaly)    alpha=0.801  R²=0.9681
                    shift=-0.187  CRITICAL

  the magnetic field's fractal structure changed during the anomaly.
  detectable from public NASA data, zero training, zero ML.
```

real data. real spacecraft. one rust function caught it. 🎯

## ⚡ speed (benchmarked, not guessed)

| signal size | struktura (rust) | python nolds | speedup |
|---|---|---|---|
| 4,096 pts | **0.24 ms** | ~20 ms | **85x** |
| 16,384 pts | **0.93 ms** | ~80 ms | **86x** |
| 65,536 pts | **2.89 ms** | ~325 ms | **112x** |

at 1Hz spacecraft telemetry: 0.24ms per analysis = **4,000 channels on one core**. yeah.

reproduce it yourself: `cargo run --release --example speed_bench`

## 🔧 as a library

```rust
use struktura::{analyze, health_check, HealthVerdict};

let law = analyze(&sensor_data);
let verdict = health_check(&law, baseline_alpha);
// Healthy | Watch | Warning | Critical
```

streaming monitor for production:

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

spacecraft-specific stuff:

```rust
use struktura::space::{SpacecraftMonitor, Subsystem};

let mut rwa = SpacecraftMonitor::new(Subsystem::ReactionWheel, "RWA_current");
// push samples, get verdicts
```

## 🎯 what it detects (all verified, all real data)

| domain | signal | healthy α | fault α | shift | verdict |
|--------|--------|-----------|---------|-------|---------|
| 🔧 **bearings** | CWRU 12kHz vibration | 0.738 | 0.217 | -0.522 | CRITICAL |
| 🚀 **spacecraft** | Voyager 1 magnetometer | 0.989 | 0.801 | -0.187 | CRITICAL |
| 📖 **text** | Austen vs shuffled | 0.749 | 0.572 | -0.177 | detected |
| 🧬 **genome** | Human chr1 GC% | 0.909 | | | R²=0.991 |
| ❤️ **cardiac** | HRV RR intervals | 0.695 | | | R²=0.985 |

every number from an actual run. reproduce with `struktura demo` / `struktura voyager`.

## 📖 text analysis

DFA measures the fractal rhythm of writing. sentence lengths in human prose have long-range correlations that disappear when you shuffle them.

```
$ struktura text pride_and_prejudice.txt shuffled.txt mechanical.txt

  Jane Austen (original)    α=0.749  STRONG RHYTHM (human literary)
  Austen (shuffled)         α=0.572  MODERATE RHYTHM
  Mechanical uniform        α=0.525  UNIFORM/MECHANICAL
```

same sentences, different order. α drops from 0.749 to 0.572. DFA catches sequential structure, not just statistics. pretty cool right?

## 🛰️ spacecraft health monitoring

built-in support for reaction wheels, magnetometers, batteries, thermal sensors, solar arrays, gyroscopes.

```
$ struktura spacecraft

  [RWA:RWA_current]     alpha=0.902 baseline=1.333 shift=-0.431  CRITICAL
  [BAT:BAT_voltage]     alpha=1.994 baseline=1.978 shift=+0.017  HEALTHY
  [THM:THM_panel_A]     alpha=1.981 baseline=1.960 shift=+0.021  HEALTHY
  [MAG:MAG_B_total]     alpha=0.985 baseline=0.914 shift=+0.070  WATCH
```

`no_std` compatible. runs on embedded flight computers. zero heap on the hot path via `dfa_into()`.

## 🏗️ flight software codegen

generate complete monitoring apps for NASA flight frameworks:

```
struktura generate --cfs    --db channels.json -o dfa_cfs_app/
struktura generate --fprime --db channels.json -o dfa_fprime_component/
struktura generate --ros    --db channels.json -o dfa_ros_node/
```

compatible with [nasa/ogma](https://github.com/nasa/ogma) variable database format.

## 🧠 how DFA works

DFA (Peng et al., Physical Review E, 1994, 3000+ citations) measures long-range correlation:

1. compute the cumulative profile (running sum minus mean)
2. divide into boxes, detrend each with a linear fit
3. measure residual fluctuation vs box size
4. slope in log-log space = α (the scaling exponent)

| α | what it means |
|---|---------|
| ~0.5 | random noise (no structure) |
| 0.5-1.0 | persistent correlations (healthy structure) |
| α shifts | something changed. go look. |

the crate reports R² alongside every α. if R² < 0.3, quality = `Abstain`. it never bluffs.

## 🧰 features

- **adaptive box sizes** geometrically spaced to signal length
- **`no_std`** use `default-features = false` for embedded
- **C FFI** `struktura.h` header, call from C/C++/whatever
- **serde** optional `features = ["serde"]`
- **self-test** `struktura self-test` verifies everything

## 🏆 alternatives

| crate | DFA | speed vs python | license | deps | `no_std` |
|-------|-----|----------------|---------|------|---------|
| **struktura** | native | **85-112x** | MIT/Apache | **1** (libm) | **yes** |
| anomaly_detection | no | | GPL-3.0 | many | no |
| extended-isolation-forest | no | | MIT | many | no |

## 📚 references

1. C.-K. Peng et al., "Mosaic organization of DNA nucleotide sequences," Physical Review E 49(2), 1994.
2. C.-K. Peng et al., "Quantification of scaling exponents," Chaos 5(1), 1995.
3. CWRU Bearing Data Center: https://engineering.case.edu/bearingdatacenter
4. NASA SPDF Voyager Data: https://spdf.gsfc.nasa.gov/pub/data/voyager/

## license

MIT OR Apache-2.0

---

<p align="center">
  if this is useful to you, <a href="https://github.com/koscak-labs/struktura">⭐ star it</a>. if it's not, open an issue and tell me why 🫡
</p>
