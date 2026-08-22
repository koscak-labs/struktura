# Bearing Fault Detection

Struktura detects bearing faults from raw vibration data with zero domain knowledge.

## CWRU Bearing Data Center results

Using 12kHz vibration data from Case Western Reserve University:

| Condition | DFA alpha | Shift | Verdict |
|-----------|-----------|-------|---------|
| Normal (97.mat) | 0.389 | — | Healthy |
| Inner race fault (105.mat) | 0.146 | -0.243 | **Critical** |
| Outer race fault (130.mat) | 0.247 | -0.142 | **Critical** |
| Ball fault (118.mat) | 0.275 | -0.114 | **Warning** |

All three fault types detected. The shift magnitude correlates with fault severity.

## How to use it

```rust
use struktura::{analyze, health_check};

let normal = analyze(&normal_vibration);
let baseline = normal.dfa.alpha; // establish during healthy operation

// Later, during monitoring:
let current = analyze(&current_vibration);
let verdict = health_check(&current, baseline);
// verdict == HealthVerdict::Critical if bearing is degrading
```

## Why DFA catches what FFT misses

FFT detects frequency changes. But early bearing degradation changes the *correlation structure* of the vibration — the way peaks relate to each other over time — before it introduces new frequency components. DFA measures this correlation structure directly.
