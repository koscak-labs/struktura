# DFA structural health monitoring as a custom ogma template

Hi @ivanperez-keera, following up from our conversations on [nasa/fprime#5772](https://github.com/nasa/fprime/issues/5772) and [nasa/cFS#1096](https://github.com/nasa/cFS/issues/1096).

I studied ogma's architecture — the Mustache template system, `--template-dir`, `--template-vars`, and the cFS/fprime/ROS examples. I built a first prototype of a custom cFS template that replaces Copilot's temporal monitors with DFA (Detrended Fluctuation Analysis) structural health monitoring.

## What DFA adds to ogma

Copilot monitors check **temporal properties** — "signal X shall always satisfy condition Y." DFA checks **structural health** — "is the signal's statistical structure changing?" These are complementary:

- A sensor drifting changes its correlation pattern (DFA catches this) before it violates a threshold (Copilot catches that)
- DFA fires on degradation onset; Copilot fires on property violation
- Running both gives earlier detection + formal verification

## The prototype

A custom ogma cFS template at [`ogma-template/cfs/dfa_monitor/`](https://github.com/koscak-labs/struktura/tree/main/ogma-template/cfs/dfa_monitor) that reuses ogma's existing infrastructure:

**Same inputs as any ogma app:**
- `db.json` — variable database connecting DFA channels to cFS message topics
- `extra-vars.json` — DFA parameters: window size, threshold, R² minimum, learning windows

**What the template generates:**
- A cFS app that subscribes to configured telemetry topics
- Per-channel DFA sliding window with automatic baseline learning
- Event emission on structural shift detection
- **Exact-or-abstain**: if R² falls below threshold, the channel suspends monitoring rather than alerting on unreliable data

**Self-contained C code** (`dfa_core.h`):
- ~90 lines, zero dependencies beyond `<math.h>`
- Fixed-size buffers, no heap allocation, deterministic
- 1,500 MACs per channel per tick (~3µs on RAD750)

**Invocation would be:**
```sh
ogma cfs --template-dir struktura-dfa-template/cfs/dfa_monitor \
         --variable-db example/db.json \
         --template-vars example/extra-vars.json \
         --target-dir dfa_app
```

## Verified results

All numbers from actual runs of the same DFA algorithm ([struktura](https://crates.io/crates/struktura)):

| Domain | Signal | DFA alpha | R² | Detection |
|--------|--------|-----------|-----|-----------|
| Bearing normal | CWRU 97.mat | 0.389 | 0.872 | baseline |
| Bearing inner fault | CWRU 105.mat | 0.146 | 0.635 | shift -0.243 |
| Bearing outer fault | CWRU 130.mat | 0.247 | 0.687 | shift -0.142 |
| Bearing ball fault | CWRU 118.mat | 0.275 | 0.746 | shift -0.114 |
| Genome | 8 human chromosomes | 0.66-0.91 | >0.99 | exact structure |
| Cardiac normal | RR intervals | 0.695 | 0.985 | healthy fractal |
| Cardiac arrhythmic | RR intervals | 0.483 | 0.990 | structure destroyed |

## Questions for you

1. **Template compatibility**: Does the Mustache variable set I'm using (`{{#variables}}`, `{{#msgIds}}`, `{{#msgCases}}`, `{{#msgHandlers}}`) cover what `--template-dir` provides? I mapped these from the default cFS template — want to confirm they're stable.

2. **DFA + Copilot together**: Is there a way to have a single ogma-generated app run BOTH Copilot temporal monitors AND DFA structural monitors? The most useful deployment would combine them.

3. **fprime + ROS templates**: I can create parallel templates for fprime and ROS using the same `dfa_core.h`. Does `--template-dir` work the same way for `ogma fprime` and `ogma ros`?

4. **Custom `ogma dfa` command**: Long-term, would it make sense for ogma to have a built-in `dfa` backend (parallel to `diagram`)? I'd be happy to contribute the Haskell side if that's the right direction.

Source: https://github.com/koscak-labs/struktura
Try it: `cargo install struktura && struktura demo`
