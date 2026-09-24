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

We have not found an open-source tool that provides **degradation monitoring
that runs embedded inside the flight software loop** next to Copilot monitors. For Artemis Gateway and deep-space missions
with 4–24 minute communication delays, autonomous onboard degradation detection
is not optional — ground-based prognostics arrive too late.

## What struktura adds

struktura watches the structure of telemetry (DFA scaling plus residual,
level, CUSUM, stuck-value and cross-channel legs) without training. It is
meant to run where Copilot runs, inside the flight loop. The aim is to
flag changes a fixed amplitude bound misses; the measured results below
say how far that aim is met today.

```
         GROUND                         SPACECRAFT
    ┌──────────────┐              ┌───────────────────────────┐
    │  ProgPy      │              │  ┌─────────────────────┐ │
    │  (offline)   │              │  │ Copilot (boolean)    │ │
    └──────────────┘              │  │ temp > 85 → alarm    │ │
                                  │  └─────────────────────┘ │
                                  │  ┌─────────────────────┐ │
                                  │  │ STRUKTURA (DFA)     │ │  ← NEW
                                  │  │ α drift → flag      │ │
                                  │  │ warning, no training │ │
                                  │  │ 103 lines C, 5.9 KB │ │
                                  │  └─────────────────────┘ │
                                  └───────────────────────────┘
```

## Measured results

**IMS bearing test 2** (984 recordings at 10-min intervals, 4 channels,
`struktura copilot-compare data/ims_monitor_stream.csv`, calibration on rows 0-327):

| Row | Event |
|-----|-------|
| 378 | struktura monitor: 1.0x-threshold drift flag on ch1 |
| 506 | struktura monitor: regime change suspected, re-learning baseline |
| 701 | amplitude threshold (1.5 x p95 of calibration \|x\|) first trips |
| 925 | struktura monitor: re-learned baseline rejected, fault confirmed |

**This run does not show earlier detection than a threshold.** The row-378
flag also appears when only rows 0-500 are fed (healthy period by common
IMS usage), so it is not evidence of early warning. The confirmed fault at
row 925 comes after the threshold at row 701. An earlier version of this
document claimed "~10 hours" and "323 samples" of early warning; both
figures are withdrawn (2026-09-24).

A C-MAPSS turbofan run (FD001-FD004) was also withdrawn: `guard` alarmed on
the healthy prefix of 35 of 36 engines, so its alarms on full runs carry
no information about wear.

What stands: the monitor needs no training, runs in constant memory, and
builds as a C component next to generated monitors. Whether its alarms lead
a threshold on real degradation is an open question this repo has not yet
answered.

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
