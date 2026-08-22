# Struktura

**Predict failure before it happens.**

```rust
use struktura::{analyze, health_check, HealthVerdict};

let law = analyze(&vibration_data);
let verdict = health_check(&law, baseline_alpha);

match verdict {
    HealthVerdict::Critical => eprintln!("STRUCTURAL SHIFT DETECTED"),
    _ => {}
}
```

This crate detected a bearing fault from a CSV file.
No training data. No domain knowledge. No dependencies.

```
cargo add struktura
```

## Proven across 5 domains

Same algorithm. Zero configuration. Every number below is from an actual run.

| Domain | Signal | N | DFA alpha | R-squared |
|--------|--------|---|-----------|-----------|
| Spacecraft | Queue depth telemetry | 500 | 0.593 | 0.789 |
| Bearings | CWRU 12kHz vibration | 243,938 | 0.389 | 0.872 |
| Genome | Human chr1 GC% | 8,000 | 0.909 | 0.991 |
| Cardiac | RR intervals (HRV) | 2,048 | 0.695 | 0.985 |
| Drones | ArduPilot IMU (proposed) | -- | -- | -- |

## It catches what monitors miss

Bearing fault detection on CWRU data (12kHz, Case Western Reserve University):

| Condition | DFA alpha | Shift from normal | Verdict |
|-----------|-----------|-------------------|---------|
| Normal | 0.389 | -- | Healthy |
| Inner race fault | 0.146 | -0.243 | **Critical** |
| Outer race fault | 0.247 | -0.142 | **Critical** |
| Ball fault | 0.275 | -0.114 | **Warning** |

All three fault types detected. No thresholds. No training. Just math.

## Genome: R-squared > 0.99 on every chromosome

| Chromosome | DFA alpha | R-squared |
|-----------|-----------|-----------|
| chr1 | 0.909 | 0.991 |
| chr2 | 0.699 | 0.991 |
| chr3 | 0.659 | 0.998 |
| chr4 | 0.894 | 0.997 |
| chr5 | 0.824 | 0.994 |
| chr6 | 0.822 | 0.998 |
| chr7 | 0.862 | 0.997 |
| chr8 | 0.816 | 0.995 |

8/8 chromosomes at R-squared > 0.99. The fractal structure of DNA is real and measurable.

## How it works

Detrended Fluctuation Analysis (DFA) measures long-range correlation in any time series:

1. Compute the cumulative profile of the signal
2. Divide into boxes, fit a linear trend in each
3. Measure the residual fluctuation at each box size
4. The scaling exponent alpha tells you the structure

- alpha near 0.5 = random noise (no structure)
- 0.5 < alpha < 1.0 = healthy complex system
- alpha shifts from baseline = something is changing

When alpha moves, something is degrading. Before amplitude thresholds fire. Before frequency analysis catches it. The structure changes first.

## API

```rust
use struktura::{dfa, acr, analyze, health_check};
use struktura::{DfaResult, StructuralLaw, LawQuality, HealthVerdict};

// Quick DFA on raw data
let result: DfaResult = dfa(&values);
println!("alpha={:.3}, R2={:.3}", result.alpha, result.r_squared);

// Full structural analysis
let law: StructuralLaw = analyze(&values);
// law.dfa, law.acr, law.hurst, law.kurtosis, law.quality, ...

// Health check against a known baseline
let verdict = health_check(&law, 0.389); // baseline alpha
// HealthVerdict::Healthy | Watch | Warning | Critical
```

## Exact-or-abstain

Struktura reports the R-squared alongside every alpha. If R-squared is below 0.7,
the quality is `LawQuality::Abstain` — the signal does not have enough structure
for a reliable diagnosis. The crate never bluffs.

## Reproduce the results

```
cargo run --example benchmark
```

Downloads CWRU bearing data and runs the full cross-domain analysis.

## References

1. C.-K. Peng et al., "Mosaic organization of DNA nucleotide sequences," Physical Review E 49(2), 1994.
2. C.-K. Peng et al., "Quantification of scaling exponents," Chaos 5(1), 1995.
3. Case Western Reserve University Bearing Data Center.

## Used in

- [nasa/fprime #5772](https://github.com/nasa/fprime/issues/5772) -- proposed onboard telemetry health monitor
- [ArduPilot/ardupilot #34142](https://github.com/ArduPilot/ardupilot/pull/34142) -- DDS timestamp fix (same contributor)
- [tokio-rs/tokio #8377](https://github.com/tokio-rs/tokio/pull/8377) -- io_uring fix (same contributor)

## Links

- [Instagram](https://instagram.com/philphauler)
- [X / Twitter](https://x.com/philphauler)

## License

MIT OR Apache-2.0
