# Quick Start

## Analyze any signal

```rust
use struktura::{analyze, health_check, HealthVerdict};

// Load your time series data
let data: Vec<f64> = load_csv("vibration.csv");

// Analyze structural health
let law = analyze(&data);
println!("DFA alpha: {:.3}", law.dfa.alpha);
println!("R-squared: {:.4}", law.dfa.r_squared);
println!("Quality: {:?}", law.quality);

// Compare against a known healthy baseline
let verdict = health_check(&law, 0.389); // baseline from calibration
match verdict {
    HealthVerdict::Healthy => println!("System is healthy"),
    HealthVerdict::Watch => println!("Minor structural shift detected"),
    HealthVerdict::Warning => println!("Significant structural change"),
    HealthVerdict::Critical => println!("CRITICAL: Major structural departure"),
}
```

## What the numbers mean

- **DFA alpha near 0.5**: uncorrelated noise — no exploitable structure
- **DFA alpha 0.5-1.0**: long-range correlated — healthy complex system
- **Alpha shift > 0.08 from baseline**: something is changing
- **R-squared > 0.7**: the measurement is reliable
- **R-squared < 0.7**: ABSTAIN — not enough structure to diagnose
