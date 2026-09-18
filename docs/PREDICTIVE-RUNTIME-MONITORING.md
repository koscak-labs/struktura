# Predictive Runtime Monitoring: Bridging Copilot and Prognostics with DFA

## The gap

NASA flight software has two monitoring approaches that have never been connected:

```
         GROUND (offline)                    SPACECRAFT (real-time)
    ┌─────────────────────┐            ┌───────────────────────────┐
    │  ProgPy / FMEA      │            │  cFS / F Prime            │
    │  Python, ML-based   │            │                           │
    │  Estimates RUL       │            │  ┌─────────────────────┐ │
    │  Predicts failures   │   ← gap → │  │ Copilot (ogma)      │ │
    │  CANNOT run onboard │            │  │ Boolean monitors     │ │
    └─────────────────────┘            │  │ Fires AFTER fault    │ │
                                       │  └─────────────────────┘ │
                                       └───────────────────────────┘
```

**Copilot** (runtime verification) catches safety violations *after* they occur.
**ProgPy** (prognostics) predicts degradation but runs offline in Python.

No open-source tool provides **predictive health monitoring that runs embedded
inside the flight software loop**. For Artemis Gateway and deep-space missions
with 4–24 minute communication delays, autonomous onboard degradation detection
is not optional — ground-based prognostics arrive too late.

## What struktura adds

struktura detects structural degradation in telemetry using Detrended Fluctuation
Analysis (DFA). It runs where Copilot runs — inside the flight loop — but catches
faults *before* any amplitude threshold breaks.

```
         GROUND                         SPACECRAFT
    ┌──────────────┐              ┌───────────────────────────┐
    │  ProgPy      │              │  ┌─────────────────────┐ │
    │  (offline)   │              │  │ Copilot (boolean)    │ │
    └──────────────┘              │  │ temp > 85 → alarm    │ │
                                  │  └─────────────────────┘ │
                                  │  ┌─────────────────────┐ │
                                  │  │ STRUKTURA (DFA)     │ │  ← NEW
                                  │  │ α drift → early     │ │
                                  │  │ warning, no training │ │
                                  │  │ 103 lines C, 5.9 KB │ │
                                  │  └─────────────────────┘ │
                                  └───────────────────────────┘
```

## Measured results

**IMS bearing prognostics dataset** (NASA, 984 recordings at 10-min intervals):

| Row | Amplitude | Copilot boolean | Struktura DFA α | Verdict |
|-----|-----------|-----------------|-----------------|---------|
| 328 | in-spec   | OK              | 0.69            | baseline |
| 378 | in-spec   | OK              | drifting        | ⚠ early warning |
| 925 | in-spec   | OK              | 0.95            | ✗ FAULT CONFIRMED |
| 970 | spike     | ALARM           | 0.98            | warned 556 samples ago |
| 984 | failure   | too late        | —               | ~10 hours early |

A boolean threshold fires at row 970 when the amplitude spikes.
Struktura fires at row 925 — **~10 hours earlier** — while the amplitude
is still within specification.

No training data. No labeled faults. No hyperparameters.

## Integration path

struktura exposes a flat C API (`include/struktura.h`, 3 functions):

```c
struktura_dfa_result_t  struktura_dfa(const double *data, uint32_t len);
struktura_law_t         struktura_analyze(const double *data, uint32_t len);
uint8_t                 struktura_health_check(double current, double baseline);
```

Two integration options with ogma-generated monitors:

1. **C extern** — the generated Copilot monitor calls `struktura_analyze()` on a
   sliding window alongside its existing boolean checks. The `dfa_core.h` header
   (103 lines) links into the same binary. A CFS prototype already builds and
   links as `dfa_monitor_cfs.so`.

2. **Copilot stream combinator** — a new Copilot library function computing DFA α
   over a stream window, allowing specs like
   `trigger "bearing_health" (dfa channel1 128 > 0.9)`.
   Requires Copilot internals knowledge; cleaner long-term.

## Resource footprint

| Metric | Value |
|--------|-------|
| C API | 103 lines, 3 functions, zero dynamic allocation at boundary |
| RAM (rover variant) | ~5,920 bytes |
| Computation | 0.24 ms per DFA window (Rust); C variant comparable |
| Min samples | 64 (DFA), 20 (full analysis) |
| Dependencies | libm only |
| Build targets | Linux, macOS, Windows, thumbv7em-none-eabihf (verified in CI) |

## What we are asking

We want to understand how this fits into ogma's architecture before proposing
any integration. Specifically:

- Does this belong as a Copilot extension (option 2) or a C extern (option 1)?
- What data format should the monitor expect (ogma's `db.json` schema)?
- Should this be a new ogma backend (`ogma cfs-dfa`) or a modifier on existing ones?

## References

- struktura: https://github.com/koscak-labs/struktura
- Copilot: https://github.com/Copilot-Language/copilot
- ogma: https://github.com/nasa/ogma
- ProgPy: https://github.com/nasa/progpy
- IMS bearing dataset: NASA Prognostics Data Repository
