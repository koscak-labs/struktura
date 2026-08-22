# Struktura

[![Crates.io](https://img.shields.io/crates/v/struktura.svg)](https://crates.io/crates/struktura)
[![CI](https://github.com/koscak-labs/struktura/actions/workflows/ci.yml/badge.svg)](https://github.com/koscak-labs/struktura/actions/workflows/ci.yml)
[![License](https://img.shields.io/crates/l/struktura.svg)](https://github.com/koscak-labs/struktura)
[![docs.rs](https://docs.rs/struktura/badge.svg)](https://docs.rs/struktura)

**Predict failure before it happens.**

The only Rust-native DFA (Detrended Fluctuation Analysis) implementation. No hyperparameters, no training, MIT/Apache licensed. Currently proposed into NASA fprime, ArduPilot, PX4, and cFS.

```
cargo install struktura
struktura self-test    # verifies all claims
struktura demo
```

```
  STRUKTURA DEMO
  Bearing fault detection from CWRU vibration data
  ================================================

  ##############..............  Normal bearing       alpha=0.738  [EXACT]

  ####..........................  Inner race FAULT   alpha=0.217  [STRONG]
                                  shift=-0.522  CRITICAL

  The bearing's vibration structure changed BEFORE
  any amplitude threshold would have fired.
```

## Try it now

```
cargo install struktura
struktura self-test    # verifies all claims
struktura demo                              # builtin bearing fault demo
struktura check your_data.csv               # analyze any signal
struktura check data.csv --baseline 0.39    # compare against baseline
struktura compare normal.csv faulted.csv    # side-by-side comparison
```

## As a library

```rust
use struktura::{analyze, health_check, HealthVerdict};

let law = analyze(&vibration_data);
let verdict = health_check(&law, 0.389);
// verdict == HealthVerdict::Critical
```

Real-time monitoring:

```rust
use struktura::{BaselineTracker, HealthVerdict};

let mut tracker = BaselineTracker::new(256, 1000);

for sample in sensor_stream {
    if let Some(verdict) = tracker.push(sample) {
        if verdict == HealthVerdict::Critical {
            trigger_alert();
        }
    }
}
```

## Bearing fault detection (CWRU data, builtin)

| Condition | DFA alpha | Shift | Verdict |
|-----------|-----------|-------|---------|
| Normal | 0.738 | -- | Healthy |
| Inner race fault | 0.217 | -0.522 | **CRITICAL** |

Reproduced by `struktura demo` using real CWRU Bearing Data Center samples bundled in the crate.

## Cross-domain proof

Same algorithm. Zero configuration. Every number from an actual run.

| Domain | Signal | DFA alpha | R2 |
|--------|--------|-----------|-----|
| Bearings | CWRU 12kHz vibration | 0.389 | 0.872 |
| Spacecraft | Queue depth telemetry | 0.593 | 0.789 |
| Genome | Human chr1 GC% (8K windows) | 0.909 | 0.991 |
| Cardiac | RR intervals (HRV) | 0.695 | 0.985 |

Genome: 8/8 chromosomes at R2 > 0.99.

## How it works

DFA (Peng et al., Physical Review E, 1994 -- 3000+ citations) measures long-range correlation:

1. Compute the cumulative profile
2. Divide into boxes, detrend each
3. Measure residual fluctuation vs box size
4. Slope in log-log space = alpha

- alpha ~ 0.5: random noise
- 0.5 < alpha < 1.0: healthy structure
- alpha shifts from baseline: degradation

The crate reports R2 alongside every alpha. If R2 < 0.3, quality = `Abstain`. It never bluffs.

## Alternatives

| Crate | DFA | License | Dependencies | Flight-software proposals |
|-------|-----|---------|-------------|--------------------------|
| **struktura** | native | MIT/Apache | **0** | 4 (fprime, ArduPilot, PX4, cFS) |
| anomaly_detection | no | GPL-3.0 | many | 0 |
| extended-isolation-forest | no | MIT | many | 0 |

## Proposed into

- [nasa/fprime #5772](https://github.com/nasa/fprime/issues/5772) -- onboard TelemetryOracle component
- [ArduPilot #34144](https://github.com/ArduPilot/ardupilot/issues/34144) -- AP_StructuralHealth library
- [PX4 #28341](https://github.com/PX4/PX4-Autopilot/issues/28341) -- structural_health module
- [nasa/cFS #1096](https://github.com/nasa/cFS/issues/1096) -- SH (Structural Health) app

## References

1. C.-K. Peng et al., "Mosaic organization of DNA nucleotide sequences," Physical Review E 49(2), 1994.
2. C.-K. Peng et al., "Quantification of scaling exponents," Chaos 5(1), 1995.
3. CWRU Bearing Data Center: https://engineering.case.edu/bearingdatacenter

## Links

- [Instagram](https://instagram.com/philphauler) | [X / Twitter](https://x.com/philphauler)
- [Docs](https://koscak-labs.github.io/struktura) | [API](https://docs.rs/struktura)

## License

MIT OR Apache-2.0
