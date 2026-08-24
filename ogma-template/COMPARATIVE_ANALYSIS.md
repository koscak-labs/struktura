# Threshold monitoring vs. structural monitoring — empirical comparison

## The question

Can DFA-based structural monitoring detect failures that Copilot-style threshold monitors miss?

## Dataset

CWRU Bearing Vibration (12kHz accelerometer, Case Western Reserve University).
Normal bearing vs. inner race fault (0.007" diameter). 999 samples each.

## Threshold analysis (what Copilot-style monitors would check)

| Monitor | Normal | Faulted | Fired? |
|---------|--------|---------|--------|
| `mean > 0.1` | 0.027 | 0.006 | NO — moved TOWARD zero |
| `std > 0.15` | 0.076 | 0.090 | NO — 18% increase, within noise |
| `max > 0.5` | 0.281 | 0.326 | NO |
| `min < -0.5` | -0.197 | -0.290 | NO |
| `abs(value) > 0.4` | occasional | occasional | Would fire on BOTH — no discrimination |

Every reasonable amplitude threshold either doesn't fire on the fault OR fires on both normal and faulted signals. The fault changes statistical SHAPE, not magnitude.

## DFA analysis (what structural monitoring sees)

| Channel | DFA α | R² | Quality | Verdict |
|---------|-------|----|---------|---------|
| Normal bearing | 0.689 | 0.953 | EXACT | baseline |
| Faulted bearing | 0.183 | 0.918 | STRONG | **-0.506 shift = CRITICAL** |

73% collapse in the scaling exponent. The signal's temporal correlation structure — how each vibration depends on previous vibrations — is destroyed by the fault. This is invisible to any single-sample threshold.

## Voyager 1 comparison

| Period | DFA α | R² | Threshold breached? |
|--------|-------|----|---------------------|
| 2021 healthy | 0.989 | 0.987 | NO |
| May-Jul 2022 (AACS anomaly) | 0.801 | 0.968 | NO — magnetometer values stayed in range |

The Voyager 1 AACS anomaly did not produce out-of-range magnetometer readings. It changed the field's fractal structure. DFA detected it; threshold monitors would not have.

## The integration path

DFA and Copilot are not competitors — they monitor different things:

- **Copilot**: "Is value X out of bounds RIGHT NOW?" (temporal logic, formal guarantees)
- **DFA**: "Has the STRUCTURE of the signal changed?" (scaling analysis, convergence guarantees)

Together through ogma: one generated app subscribes to a telemetry channel, runs Copilot threshold checks AND DFA structural analysis. The DFA computation lives in C (`dfa_core.h`), called as a Copilot extern. The monitoring logic (shift detection, R² gating) is expressed in Copilot's DSL.

## Reproducibility

```
cargo install struktura
struktura demo        # CWRU bearing comparison
struktura voyager     # Voyager 1 anomaly detection
```

All data embedded in the crate. Zero dependencies. Zero training.
