# bench/flight — embedded-monitor evidence for struktura's "embeddable" claim

Turns "embeddable" into a measured claim: takes the C99 monitor struktura
actually generates, runs it on an emulated ARM Cortex-M3 (QEMU
`mps2-an385`), and checks it against the Rust library monitor it was
generated from.

## What was measured

**Generated artifact under test: `hybrid_monitor.c`**, produced by
`struktura generate-hybrid` (`cmd_generate_hybrid` in
`src/bin/struktura.rs:2788`, which calls
`struktura::codegen::generate_hybrid_c` in `src/codegen.rs:297`). This is
the standalone C99 path — **not** `struktura generate --rover`, which emits
`RoverHealth.fpp` + a thin `RoverHealthImpl.c` wrapper that calls back into
a Rust staticlib (`rover_flight`) rather than being self-contained C, and
which the README documents as having no DFA leg.

`hybrid_monitor.c` contains **five legs**: residual (AR1), repeated-value,
windowed DFA (Hurst/alpha via box-counting), rolling-mean level shift, and
residual CUSUM — all in static memory, C99 + `<math.h>`, no dynamic
allocation. The Rust `HybridMonitor` (`src/monitor.rs`) that the C is
calibrated from has **two additional legs** the generator does not emit:
`Missingness` (explicit invalid-sample tracking via
`push_with_validity`) and `Parity` (cross-channel analytical redundancy).
This comparison only exercises the five legs common to both, fed with
fully-valid single-channel-fault data — it does not exercise
missingness/parity equivalence, because the C side has no such legs to
compare against.

## How the fixtures were built

`bench/flight/genrs/` is a small separate Cargo crate (path-depends on the
`struktura` lib crate, does not touch `src/` or the root `Cargo.toml`) that:

1. Calibrates a `HybridMonitor` exactly as `generate-hybrid`'s CLI does by
   default (`synth_spacecraft(2048, 424242)`), and writes the resulting
   `hybrid_monitor.c` — this is the *actual* generator output, not a
   hand-written stand-in.
2. Builds a second, longer synthetic stream (`synth_spacecraft(3000,
   909090)`), freezes the repeat-enabled channel from its midpoint (t=1500)
   onward (a stuck-sensor fault, same shape as the C generator's own
   `#ifdef HYBRID_STANDALONE_TEST` self-test), and replays it through a
   **fresh** Rust `HybridMonitor` built from the same calibration, recording
   the first-alarm tick + leg to `alarms_rust.csv`.
3. Emits the same stream as a static C array in `stream_data.h`, so the
   bare-metal ARM build can replay byte-identical input with no filesystem
   or semihosting file I/O required.

`harness.c` replays `stream_data.h` through the generated C monitor twice
(fresh state each run) and reports the first-alarm `(tick, leg)`, per-tick
timing, and whether the two runs agree (determinism). It compiles two ways:
native x86 (full libc, `clock_gettime` for wall time) and freestanding ARM
(`__arm__`, DWT `CYCCNT` for cycles, raw ARM semihosting `SYS_WRITE0`/
`SYS_EXIT` for output — no newlib, no libc).

`startup.c` + `mps2_an385.ld` are a from-scratch minimal Cortex-M3 startup
(vector table, `Reset_Handler` that copies `.data`/zeroes `.bss` then calls
`main`, and fault handlers that print `CFSR`/`HFSR`/`MMFAR`/`BFAR` via
semihosting instead of hanging silently) — the toolchain's own
`rdimon(-v2m).specs` startup objects target a different memory map
(vector table expected at `0x8000`, not `0x0`) and faulted immediately on
`mps2-an385`, so this was written instead of debugging that mismatch.

## Results

**Toolchain** (installed via `apt` in WSL Ubuntu; kali-linux WSL had
neither package and Ubuntu had passwordless sudo, so that's what was used):
`arm-none-eabi-gcc (15:13.2.rel1-2) 13.2.1 20231009`, `QEMU emulator
version 8.2.2 (Debian 1:8.2.2+ds-0ubuntu1.18)`.

**Equivalence — IDENTICAL across all three.** Alarm tick 1504, leg
`REPEATED` (repeated-value/stuck-sensor leg), on:
- the Rust `HybridMonitor` (`alarms_rust.csv`: `1504,REPEATED`)
- the generated C compiled natively for x86 (`gcc -O2`)
- the generated C compiled for Cortex-M3 and run under QEMU (`arm-none-eabi-gcc -O1`, `mps2-an385`)

Decisions (which tick, which leg) match exactly, not just "close" — this is
integer/enum output, not a floating-point comparison, so there is no
last-bit ambiguity to report here.

**Determinism.** Each platform ran the fixed stream twice through a fresh
monitor instance; RUN1 and RUN2 agree (`DETERMINISTIC=1`) on native x86 and
on ARM/QEMU.

**Static memory (Cortex-M3, `-O1`, isolated from the test harness via
`size_probe.c` — a minimal caller that references `hyb_init`/`hyb_push`
without embedding the 144000-byte test stream)**:
```
   text    data     bss     dec     hex filename
   7120       0    9520   16640    4100 size_probe.elf
```
7120 B flash (monitor code + soft-float `libm`/`libgcc`, since Cortex-M3 has
no FPU), 9520 B RAM (`.bss`, dominated by the 6-channel `hyb_monitor_t`
ring buffers: `2 x 96 x 8 B x 6 channels ~= 9216 B` plus scalars), 0 B
`.data` (all state zero-initialized).

**No dynamic allocation.** `grep -n 'malloc\|calloc\|realloc\|free('
hybrid_monitor.c` finds nothing. Independently confirmed structurally: the
bare-metal build is `-nostdlib` with no malloc/free implementation
provided anywhere in the link — if the generated code called one, the link
would fail. It links clean.

**Cycle/instruction cost — BLOCKED, not measured.** `DWT->CYCCNT` on
`qemu-system-arm -M mps2-an385 -cpu cortex-m3` reads a constant `0`
throughout the run: this QEMU machine/CPU model does not implement the DWT
cycle counter (confirmed by enabling `DEMCR.TRCENA` + `DWT_CTRL.CYCCNTENA`
per the architecture and reading `CYCCNT` immediately before/after each
tick — always `0`). No `-icount`-based instruction count was obtained
either (would need a TCG plugin, e.g. `contrib/plugins/execlog`, not
available in the `apt` qemu package used here). The only timing evidence is
host wall-clock for the whole 3000-tick QEMU run (~1 s including QEMU
process startup) — this is not per-sample cost and is not claimed as such.
**Getting a real per-sample cycle count needs either real Cortex-M3
hardware, or a cycle-accurate simulator (e.g. Renode), or a QEMU build with
working DWT/PMU + an instruction-count TCG plugin — none of which were
available in this timebox.**

**`-O2` blocker.** struktura's own printed compile line for
`generate-hybrid` is `gcc -std=c99 -Wall -Werror -O2 ...`. Under this
project's freestanding/`-nostdlib` ARM setup, `-O2` produces a binary that
**BusFaults** on `mps2-an385` (`CFSR=0x00008200` = imprecise data bus
error, `BFAR=0xfeedbebf` — a QEMU unmapped-memory poison value, not a real
address) before printing anything past `MARK-A`. `-O0` and `-O1` both build
and run correctly with identical results to native x86. Root cause not
isolated in this timebox (candidates: a miscompile interacting with
`-ffreestanding -nostdlib -mfloat-abi=soft` at `-O2` specifically, or a
QEMU TCG bug exposed by the code `-O2` generates — not narrowed further).
**This means the ARM numbers above are from an `-O1` build, not the `-O2`
build struktura's own docs tell a user to run** — that gap is the main
open item, not memory/equivalence/determinism, which are solid.

## Reproducing

```
bash bench/flight/run.sh
```

## Files

- `hybrid_monitor.c` — the actual `generate-hybrid` output (regenerated by `run.sh`/`genrs`)
- `stream_data.h`, `stream.csv` — the 3000-tick fixture stream (Rust-generated)
- `alarms_rust.csv` — ground truth: Rust `HybridMonitor`'s alarm tick/leg on that stream
- `genrs/` — separate Cargo crate (path-dep on the struktura lib) that generates the C + fixtures and runs the Rust side
- `harness.c` — shared equivalence/cost harness (native x86 and freestanding ARM)
- `startup.c`, `mps2_an385.ld` — minimal from-scratch Cortex-M3 startup + linker script for `mps2-an385`
- `size_probe.c` — isolates the generated monitor's own flash/RAM footprint from the test harness
- `run.sh` — reproduces all of the above end to end
