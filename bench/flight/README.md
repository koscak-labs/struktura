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

**Equivalence — IDENTICAL across every level.** Alarm tick 1504, leg
`REPEATED` (repeated-value/stuck-sensor leg), on:
- the Rust `HybridMonitor` (`alarms_rust.csv`: `1504,REPEATED`)
- the generated C compiled natively for x86 (`gcc -O2`)
- the generated C compiled for Cortex-M3 and run under QEMU
  (`arm-none-eabi-gcc`, `mps2-an385`), at `-O0`, `-O1`, `-O2`, `-O3`, and
  `-Os` — see "`-O2` needs `-fno-early-inlining`" below for what makes `-O2`
  (struktura's own suggested compile line) pass

Decisions (which tick, which leg) match exactly, not just "close" — this is
integer/enum output, not a floating-point comparison, so there is no
last-bit ambiguity to report here.

**Determinism.** Each platform ran the fixed stream twice through a fresh
monitor instance; RUN1 and RUN2 agree (`DETERMINISTIC=1`) on native x86 and
on ARM/QEMU, at every optimization level.

**Static memory (Cortex-M3, isolated from the test harness via
`size_probe.c` — a minimal caller that references `hyb_init`/`hyb_push`
without embedding the 144000-byte test stream), by optimization level (all
built with `run.sh`'s size_probe flags, `-fno-early-inlining` and
`-Wl,--gc-sections`, arm-none-eabi-gcc 13.2.1)**:

| level | flash (`.text`) | RAM (`.bss`) |
|-------|-----------------:|-------------:|
| `-O0` | 8644 B | 9520 B |
| `-O1` | 7128 B | 9520 B |
| `-O2` | 6600 B | 9520 B |
| `-O3` | 6872 B | 9520 B |
| `-Os` | 6284 B | 9520 B |

RAM is constant across levels: `.bss` is dominated by the 6-channel
`hyb_monitor_t` ring buffers (`2 x 96 x 8 B x 6 channels ~= 9216 B` plus
scalars), 0 B `.data` (all state zero-initialized). `run.sh` and the root
`README.md` quote the `-O2` row, matching struktura's own printed
`generate-hybrid` compile line.

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

**`-O2` needs `-fno-early-inlining` (fixed).** struktura's own printed
compile line for `generate-hybrid` is `gcc -std=c99 -Wall -Werror -O2 ...`.
Under this project's freestanding/`-nostdlib` ARM setup, plain `-O2`
produces a binary that **BusFaults** on `mps2-an385` (`CFSR=0x00008200` =
imprecise data bus error, `BFAR=0xfeedbec7` in one build, `0xfeedbebf` in
another) right after printing `MARK-A`, i.e. inside the very first call to
`run_once()` in `harness.c`.

What was measured (arm-none-eabi-gcc 13.2.1, plain `-O2`, HardFault handler
in a scratch copy of `startup.c` printing the stacked PC/LR):

- The faulting instruction is `str r2, [r6, #0]`, the store
  `*out_leg = (int)v;` in `run_once()`. Just before it, `ldrd r5, r6,
  [sp, #120]` reloads `out_t` and `out_leg` from the stack slot they were
  spilled to at function entry (`strd r5, r6, [sp, #120]`), and the reloaded
  value is not the pointer that was stored. With `-fstack-protector-all` the
  same happens to `out_t` instead (`BFAR=0xfeedbebf`), and the canary does not
  fire. So something overwrites that stack slot during the replay loop.
- The only instruction in `run_once()` that stores to `[sp, #120]` directly is
  that entry spill; the overwrite comes through some other pointer.
- Flag bisection, one flag at a time on top of `-O2`: `-fno-early-inlining`
  alone fixes it, and so does `-fno-inline`. `--param max-inline-insns-*`,
  `-fno-inline-functions[-called-once]`, `-fno-tree-tail-merge`,
  `-fno-ira-share-spill-slots`, `-fno-strict-aliasing`,
  `-fno-schedule-insns[2]`, `-fno-shrink-wrap`, `-fno-reorder-blocks`,
  `-fno-omit-frame-pointer` and `-fno-tree-loop-distribute-patterns` do not.
  Early inlining folds `hyb_push()` and `hyb_dfa_alpha()` into `run_once()`,
  giving one function with an 11 KB frame (the monitor struct is a local).

What was checked and found clean:

- The same `harness.c` + `hybrid_monitor.c` built natively (x86-64 gcc) with
  AddressSanitizer and UndefinedBehaviorSanitizer at `-O0`, `-O2` and `-O3`:
  no report, alarm at 1504 on the repeated-value leg, deterministic.
- The ARM `-O2` build with `-fsanitize=undefined -fsanitize=bounds-strict`
  (trap mode) on `harness.c`: `run_once()` completes with no trap. The first
  trap is later, in the harness's insertion sort, which runs after the
  monitor replay. The sanitizer also changes the generated code, so this does
  not reproduce the original layout.
- The C `memset` in `startup.c` does not call itself at `-O2` (`-fno-builtin`),
  and the link uses the `thumb/v7-m/nofp` libgcc and libm.

What is not established: whether this is a GCC code-generation bug or
something specific to this bare-metal setup. Only one ARM compiler (GCC
13.2.1) was available, and there is no minimized reproducer, so no compiler
bug has been reported. What is established: the generated monitor showed no
undefined behaviour under the sanitizers, and with `-fno-early-inlining` the
ARM build matches Rust at every optimization level below. `run.sh` passes
`-fno-early-inlining` on both the `size_probe` and the `harness.c` ARM
builds; `-Wno-error` is kept for the ARM build (pre-existing warnings from
the freestanding headers), and `-Wall -Werror` still applies to the native
x86 build and to `size_probe.c`.

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
