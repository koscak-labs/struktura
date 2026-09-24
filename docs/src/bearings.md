# Bearing Fault Detection

On 12 kHz vibration data from the Case Western Reserve University Bearing Data Center, DFA α separates a normal bearing from one with an inner-race fault:

| Condition | DFA α |
|-----------|-------|
| Normal | 0.689 |
| Inner-race fault | 0.183 |

Reproduce with `struktura demo` (the data is embedded). CI re-checks these two numbers ([docs/claims.tsv](https://github.com/koscak-labs/struktura/blob/master/docs/claims.tsv), row `bearing-alpha`). Other fault types (outer race, ball) are not part of the checked set.

## How to use it

```rust
use struktura::{analyze, health_check};

let normal = analyze(&normal_vibration);
let baseline = normal.dfa.alpha; // establish during healthy operation

// Later, during monitoring:
let current = analyze(&current_vibration);
let verdict = health_check(&current, baseline);
```

`health_check` compares α against the baseline with fixed thresholds (0.03 / 0.08 / 0.15). They are defaults, not significance tests; decide what shift matters for your machine.

## What DFA measures

DFA measures how the fluctuations of a signal scale with the window size, which reflects its correlation structure. A fault can change that structure without adding a new frequency peak, so DFA complements spectral (FFT) and amplitude checks. On the NASA IMS run-to-failure bearing, a plain RMS threshold alarmed earlier than struktura, so do not treat DFA as an early-warning method.
